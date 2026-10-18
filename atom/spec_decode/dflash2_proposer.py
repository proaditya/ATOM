# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""DFlash 2 drafting (``--method dflash`` with a ``DFlash2DraftModel`` draft).

A block drafter on the verify side's plain chain: each round it drafts
``num_speculative_tokens`` (at most block - 1 = 7) tokens in one block pass,
and the target verifies them as a chain. The draft's context is absorbed after
EVERY target forward (``compute_draft_kv``), prefill and verify alike, from the
residual stream after the draft's target layers (forward hooks on the target's
decoder layers, so any target with a ``.layers`` stack works, and a mono target
writes the same buffers from its kernels).
"""

import logging

import torch
import triton
import triton.language as tl
from torch import nn
from torch.profiler import record_function

from atom.spec_decode.drafter import AuxCaptureSpec, Drafter
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

DFLASH2_ARCHITECTURES = ("DFlash2DraftModel",)


@triton.jit
def _anchor_kernel(NEXT, POS, LAST, IDS, BASE):
    """The single-sequence draft's anchor: the bonus token at one past the
    last accepted row."""
    tl.store(IDS, tl.load(NEXT).to(tl.int64))
    tl.store(BASE, tl.load(POS + tl.load(LAST)).to(tl.int64) + 1)


@triton.jit
def _mono_import_kernel(
    MK,
    MV,
    PK,
    PV,
    TABLE,
    REQ,
    MONO_REQ,
    MONO_LEN,
    P0,
    P: tl.constexpr,
    NKV: tl.constexpr,
    HD: tl.constexpr,
    BS: tl.constexpr,
    X: tl.constexpr,
    WIN: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    """The drafter's window before position P0 that its own cache lacks, from
    the paged pool (the record): positions [start, P0), start = how far it
    holds this request (``MONO_LEN``) if it does, else P0 - WIN."""
    layer = tl.program_id(0)
    h = tl.program_id(1)
    p = tl.program_id(2)
    p0 = tl.load(P0).to(tl.int32)
    held = tl.load(MONO_REQ) == tl.load(REQ).to(tl.int32)
    start = tl.maximum(tl.where(held, tl.minimum(tl.load(MONO_LEN), p0), 0), p0 - WIN)
    mk = tl.load(MK + layer).to(tl.pointer_type(tl.bfloat16))
    mv = tl.load(MV + layer).to(tl.pointer_type(tl.bfloat16))
    pk = tl.load(PK + layer).to(tl.pointer_type(tl.bfloat16))
    pv = tl.load(PV + layer).to(tl.pointer_type(tl.bfloat16))
    t = tl.arange(0, BS)
    dd = tl.arange(0, HD)
    for blk_i in range(start // BS + p, (p0 + BS - 1) // BS, PROGRAMS):
        pos = blk_i * BS + t
        ok = ((pos >= start) & (pos < p0))[:, None]
        hb = tl.load(TABLE + blk_i).to(tl.int64) * NKV + h
        k = tl.load(
            pk
            + ((hb * (HD // X) + dd[None, :] // X) * BS + t[:, None]) * X
            + dd[None, :] % X,
            mask=ok,
        )
        v = tl.load(
            pv
            + ((hb * (BS // X) + t[:, None] // X) * HD + dd[None, :]) * X
            + t[:, None] % X,
            mask=ok,
        )
        row = (h * P + pos).to(tl.int64)[:, None] * HD + dd[None, :]
        tl.store(mk + row, k, mask=ok)
        tl.store(mv + row, v, mask=ok)


@triton.jit
def _mono_export_kernel(
    MK,
    MV,
    PK,
    PV,
    POS,
    SLOTS,
    REQ,
    MONO_REQ,
    MONO_LEN,
    P: tl.constexpr,
    NKV: tl.constexpr,
    HD: tl.constexpr,
    BS: tl.constexpr,
    X: tl.constexpr,
):
    """Rows the drafter just took into its cache -> their paged slots; then
    it holds this request through the last of them."""
    layer = tl.program_id(0)
    r = tl.program_id(1)
    h = tl.program_id(2)
    pos = tl.load(POS + r).to(tl.int64)
    slot = tl.load(SLOTS + r).to(tl.int64)
    if slot >= 0:
        dd = tl.arange(0, HD)
        src = (h * P + pos) * HD + dd
        hb = (slot // BS) * NKV + h
        t = slot % BS
        pk = tl.load(PK + layer).to(tl.pointer_type(tl.bfloat16))
        pv = tl.load(PV + layer).to(tl.pointer_type(tl.bfloat16))
        mk = tl.load(MK + layer).to(tl.pointer_type(tl.bfloat16))
        mv = tl.load(MV + layer).to(tl.pointer_type(tl.bfloat16))
        tl.store(
            pk + ((hb * (HD // X) + dd // X) * BS + t) * X + dd % X, tl.load(mk + src)
        )
        tl.store(
            pv + ((hb * (BS // X) + t // X) * HD + dd) * X + t % X, tl.load(mv + src)
        )
    if (layer == 0) & (r == tl.num_programs(1) - 1) & (h == 0):
        tl.store(MONO_REQ, tl.load(REQ).to(tl.int32))
        tl.store(MONO_LEN, (pos + 1).to(tl.int32))


def _unsharded(weight: torch.Tensor, vocab: int) -> torch.Tensor:
    """A [vocab / tp (padded), H] shard -> the whole [vocab, H] (the weight
    itself at TP 1)."""
    from aiter.dist.parallel_state import get_tp_group

    tp = get_tp_group()
    if tp.world_size == 1 or weight.shape[0] >= vocab:
        return weight
    return tp.all_gather(weight.contiguous(), dim=0)[:vocab].contiguous()


def _residual_after(output, _module) -> torch.Tensor:
    """A decoder layer's ``(hidden, residual)`` -> the residual stream after it."""
    return output[0] + output[1]


def _addresses(tensors) -> tuple:
    """The (address, shape) of every tensor in a nest of tuples / lists: what
    a graph that reads them in place was recorded against."""
    out = []
    for t in tensors:
        if isinstance(t, (tuple, list)):
            out.extend(_addresses(t))
        else:
            out.append((t.data_ptr(), tuple(t.shape)))
    return tuple(out)


class DFlash2Proposer(Drafter):
    # a megakernel target hands its residual streams over in place
    # (``MonoDecodeModel.mono_aux``) instead of copying them into ours
    mono_aux_reader = True
    mono_aux_source = None

    def _resolve_mtp_k(self) -> int:
        draft_cfg = self.speculative_config.draft_model_hf_config
        block = int(draft_cfg.dflash_config["block_size"])
        k = self.speculative_config.num_speculative_tokens or block - 1
        if not 0 < k < block:
            raise ValueError(
                f"DFlash 2 drafts at most {block - 1} tokens a block, not {k}"
            )
        return k

    def _build_draft_model(self, model_class) -> nn.Module:
        import copy

        from atom.config import CompilationLevel
        from atom.spec_decode.draft_kv import draft_kv_builder

        draft_hf = self.speculative_config.draft_model_hf_config
        draft_atom_config = copy.copy(self.config)
        draft_atom_config.hf_config = draft_hf
        draft_atom_config.compilation_config = copy.copy(self.config.compilation_config)
        draft_atom_config.compilation_config.level = CompilationLevel.NO_COMPILATION
        model = model_class(
            draft_atom_config, layer_offset=self.config.hf_config.num_hidden_layers
        )
        # the draft's KV rides the target's block ids in a pool of its own
        builder = draft_kv_builder(self.runner, draft_hf)
        if builder is None:
            raise RuntimeError(
                "DFlash 2 needs a draft KV pool; the draft's attention backend owns none"
            )
        self.runner.draft_kv_builder = builder
        return model

    def load_model(self, target_model: nn.Module) -> None:
        target = getattr(target_model, "language_model", target_model)
        vocab = self.speculative_config.draft_model_hf_config.vocab_size
        # the draft runs whole on every rank: a vocab-sharded embedding and LM
        # head are gathered once
        embed = _unsharded(target.model.embed_tokens.weight, vocab)
        lm_head = _unsharded(target.lm_head.weight, vocab)
        if embed.shape[0] != vocab or lm_head.shape[0] != vocab:
            raise ValueError(
                f"DFlash 2 needs the target's whole embedding and LM head: vocab {vocab} vs "
                f"{tuple(embed.shape)} / {tuple(lm_head.shape)}"
            )
        self.model.load_checkpoint(
            self.speculative_config.model, embed, lm_head, self.device
        )
        logger.info(
            "DFlash 2 draft loaded from %s (target embedding and LM head)",
            self.speculative_config.model,
        )
        self._mono = self._load_mono_drafter(embed, lm_head)

    def _load_mono_drafter(self, embed, lm_head):
        """DFlash2Mono (persistent FlyDSL layers, MXFP8, an MXFP4-screened LM
        head) from exported code objects, for single-sequence steps:
        ``ATOM_DFLASH2_AOT=DIR`` (``mono/drafter/aot_runtime.py export``:
        *.hsaco + manifest.json). It keeps its own KV (positions < max_pos), so
        it serves one sequence at a time."""
        import json
        import os
        from pathlib import Path

        aot = os.environ.get("ATOM_DFLASH2_AOT")
        if not aot:
            return None
        ps = json.loads((Path(aot) / "manifest.json").read_text())["abi"]["Ps"]
        need = int(self.config.max_model_len) + 8
        fit = [p for p in ps if p >= need]
        if not fit:
            raise ValueError(
                f"ATOM_DFLASH2_AOT exports max_pos {ps}; max_model_len {need - 8} needs >= {need}"
            )
        from atom.models.qwen38_27b.mono.drafter.dflash2_mono import DFlash2Mono

        dm = DFlash2Mono(
            self.speculative_config.model, embed, lm_head, min(fit),
            prec={"layers": "aot8", "lm": "mx4r128", "ctx": "aot8"}, aot=aot,
        )  # fmt: skip
        dev = self.device
        self._ids8 = torch.zeros(8, dtype=torch.int64, device=dev)
        self._base = torch.zeros(1, dtype=torch.int64, device=dev)
        self._mono_out = None  # in the sampled tokens' dtype, at the first draft

        # its cache mirrors the paged pool, the record, for one request: which
        # (its first block) and through which position. Every single-sequence
        # step's rows go to both; a batched step writes the pool only, and the
        # mirror re-imports its window before it next drafts
        def ptrs(ts):
            return torch.tensor(
                [t.data_ptr() for t in ts], dtype=torch.int64, device=dev
            )

        self._mono_kv = (
            ptrs(L["k"] for L in dm.layers),
            ptrs(L["v"] for L in dm.layers),
        )
        self._mono_req = torch.full((1,), -1, dtype=torch.int32, device=dev)
        self._mono_len = torch.zeros(1, dtype=torch.int32, device=dev)
        self._no_req = torch.full((1,), -2, dtype=torch.int32, device=dev)
        # a verify's context runs in the draft's graph, after the verdict
        # (``ATOM_DFLASH2_ROUND_GRAPH=0``: right after the verify, eagerly)
        self._mono_round_graph = os.environ.get(
            "ATOM_DFLASH2_ROUND_GRAPH", "1"
        ) != "0" and not os.environ.get("ATOM_DFLASH2_EAGER")
        logger.info(
            "DFlash 2 single-sequence drafter: DFlash2Mono (aot %s, max_pos %d)",
            aot,
            min(fit),
        )
        return dm

    def _aux_capture_spec(self, target_model: nn.Module) -> AuxCaptureSpec:
        hidden = getattr(
            self.config.hf_config, "text_config", self.config.hf_config
        ).hidden_size
        return AuxCaptureSpec(
            layer_ids=self.model.layer_ids,
            hidden_size=hidden,
            extract=_residual_after,
            capture="output",
        )

    def _pool_ready(self) -> bool:
        k = getattr(self.model.attn[0], "k_cache", None)
        return k is not None and k.numel() > 0

    def compute_draft_kv(self, positions, hidden_states, next_token_ids) -> None:
        """The context of every row the target just ran, at its slot."""
        del next_token_ids
        fwd = get_forward_context()
        if fwd.context.is_dummy_run or not self._pool_ready():
            return
        n = hidden_states.shape[0]
        aux = (
            None
            if self.mono_aux_source is None
            else self.mono_aux_source.mono_aux(n, self.aux_layer_ids)
        )
        if aux is None:
            aux = self.aux_for(hidden_states)
        if aux is None:
            return
        if positions.dim() == 2:
            positions = positions[0]
        mono = (
            getattr(self, "_mono", None) is not None and fwd.context.scheduled_bs == 1
        )
        slots = fwd.attn_metadata.slot_mapping[:n]
        self._flush_deferred_context()
        if not mono:
            self.model.context(torch.cat(aux, -1), positions[:n], slots)
            if getattr(self, "_mono", None) is not None:
                self._mono_len.zero_()  # the mirror missed these rows
            return
        tables = fwd.attn_metadata.block_tables
        if (
            n == 8
            and tables is not None
            and not fwd.context.is_prefill
            and self._mono_round_graph
        ):
            # a single-sequence verify's context runs in the draft's graph
            # (``_propose_mono``), after the verdict the host waits on
            self._deferred_ctx = (positions[:n], aux, slots, tables)
            return
        self._mono_context(positions[:n].long(), aux, slots, tables)

    def _flush_deferred_context(self) -> None:
        """A deferred context that no draft took (a step without propose):
        run it now, before any later rows."""
        ctx = getattr(self, "_deferred_ctx", None)
        if ctx is not None:
            self._deferred_ctx = None
            pos, aux, slots, tables = ctx
            self._mono_context(pos.long(), aux, slots, tables)

    def _mono_context(self, pos, aux, slots, tables) -> None:
        """A single sequence's context rows into DFlash2Mono's cache, after the
        window it lacks (from the paged pool), and on into the paged pool."""
        dm = self._mono
        mk, mv = self._mono_kv
        pk, pv = self._paged_kv()
        kc = self.model.attn[0].k_cache
        bs, x = kc.shape[3], kc.shape[4]
        geometry = {"P": dm.P, "NKV": dm.nkv, "HD": dm.hd, "BS": bs, "X": x}
        layers = len(dm.layers)
        req = self._no_req if tables is None else tables[0, :1]
        if tables is not None:
            _mono_import_kernel[(layers, dm.nkv, 16)](
                mk, mv, pk, pv, tables[0], req, self._mono_req, self._mono_len, pos[:1],
                WIN=dm.win, PROGRAMS=16, num_warps=4, **geometry,
            )  # fmt: skip
        n = pos.numel()
        for i in range(0, n, 1024):  # bounded [rows, layers, H] staging
            dm.context(
                pos[i : i + 1024], torch.stack([a[i : i + 1024] for a in aux], 1)
            )
        _mono_export_kernel[(layers, n, dm.nkv)](
            mk, mv, pk, pv, pos, slots, req, self._mono_req, self._mono_len, num_warps=1, **geometry,
        )  # fmt: skip

    def _paged_kv(self):
        if getattr(self, "_paged_ptrs", None) is None:
            dev = self.device
            self._paged_ptrs = tuple(
                torch.tensor(
                    [getattr(a, c).data_ptr() for a in self.model.attn],
                    dtype=torch.int64,
                    device=dev,
                )
                for c in ("k_cache", "v_cache")
            )
        return self._paged_ptrs

    def propose(
        self,
        target_token_ids,
        target_positions,
        target_hidden_states,
        num_reject_tokens,
        next_token_ids,
        last_token_indices,
    ) -> torch.Tensor:
        fwd = get_forward_context()
        bs = next_token_ids.shape[0]
        if target_positions.dim() == 2:
            target_positions = target_positions[0]
        if not self._pool_ready():  # warmup before the pool exists
            return torch.zeros(
                bs, self.mtp_k, dtype=next_token_ids.dtype, device=next_token_ids.device
            )
        if bs == 1 and getattr(self, "_mono", None) is not None:
            return self._propose_mono(
                target_positions, next_token_ids, last_token_indices
            )
        tables = fwd.attn_metadata.block_tables[:bs]
        g = self._draft_graph(bs, tables, next_token_ids.dtype)
        # the anchor (the bonus token) sits one past the last accepted row
        g["anchors"].copy_(next_token_ids[:bs])
        g["p0"].copy_(
            target_positions.index_select(0, last_token_indices[:bs].long()) + 1
        )
        g["tables"].copy_(tables)
        if g["graph"] is None and not g["eager"]:
            self._capture(g)
        with record_function(f"dflash2_draft[bs={bs}]"):
            if g["graph"] is None:
                self.model.draft(g["anchors"], g["p0"], g["tables"], g["out"])
            else:
                g["graph"].replay()
        return g["out"]

    def _propose_mono(
        self, target_positions, next_token_ids, last_token_indices
    ) -> torch.Tensor:
        import os

        dm = self._mono
        _anchor_kernel[(1,)](
            next_token_ids,
            target_positions,
            last_token_indices,
            self._ids8,
            self._base,
            num_warps=1,
        )
        if self._mono_out is None:
            self._mono_out = torch.zeros(
                1, self.mtp_k, dtype=next_token_ids.dtype, device=self.device
            )
        # this verify's context, deferred by ``compute_draft_kv``: it reads
        # static buffers (the runner's positions / slots / tables, the kernel's
        # residual streams), so one graph per set of their addresses holds it
        ctx = getattr(self, "_deferred_ctx", None)
        self._deferred_ctx = None

        def draft():
            if ctx is not None:
                pos, aux, slots, tables = ctx
                self._mono_context(pos.long(), aux, slots, tables)
            # the mailbox tag is (base, step): a repeated base (a new request
            # at the same length) must not read an earlier draft's mailboxes
            dm.step.add_(1)
            dm.draft_into(self._ids8, self._base)
            self._mono_out[0].copy_(self._ids8[1 : 1 + self.mtp_k])

        key = None if ctx is None else _addresses(ctx)
        graphs = self.__dict__.setdefault("_mono_graphs", {})
        graph = graphs.get(key)
        if graph is None and len(graphs) >= 4:
            # inputs that move between steps: the graphs would pile up
            if not getattr(self, "_warned_moving", False):
                logger.warning(
                    "DFlash 2 single-sequence draft: context inputs move between steps; running eager"
                )
                self._warned_moving = True
            draft()
            return self._mono_out
        if graph is None and not os.environ.get("ATOM_DFLASH2_EAGER"):
            # warm on a side stream with this step's inputs, then record; the
            # replay below runs the context again (the same rows) and drafts
            # again under a new step
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                draft()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                draft()
            graphs[key] = graph
            logger.info(
                "DFlash 2 single-sequence %s captured",
                "draft" if ctx is None else "context + draft",
            )
        with record_function("dflash2_draft_mono"):
            if graph is None:
                draft()
            else:
                graph.replay()
        return self._mono_out

    def _draft_graph(self, bs: int, tables: torch.Tensor, dtype) -> dict:
        """The block pass for ``bs`` requests as one HIP graph over static
        buffers (eager with ``ATOM_DFLASH2_EAGER=1``), captured on first use."""
        import os

        key = (bs, tables.shape[1])
        graphs = self.__dict__.setdefault("_graphs", {})
        g = graphs.get(key)
        if g is not None:
            return g
        dev = self.device
        g = {
            "anchors": torch.zeros(bs, dtype=torch.int64, device=dev),
            "p0": torch.zeros(bs, dtype=torch.int64, device=dev),
            "tables": torch.zeros_like(tables),
            "out": torch.zeros(bs, self.mtp_k, dtype=dtype, device=dev),
            "graph": None,
        }
        g["eager"] = bool(os.environ.get("ATOM_DFLASH2_EAGER"))
        graphs[key] = g
        return g

    def _capture(self, g: dict) -> None:
        """Record the block pass. The warm-up runs this step's real inputs (the
        replay that follows recomputes the same block), so it writes no KV a
        request does not own."""
        args = (g["anchors"], g["p0"], g["tables"], g["out"])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.model.draft(*args)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.model.draft(*args)
        g["graph"] = graph
        logger.info("DFlash 2 block pass captured for batch %d", g["anchors"].shape[0])
