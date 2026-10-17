# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""EAGLE-4 drafting (``--method eagle3`` with a ``LlamaForCausalLMEagle4``
draft): ``EagleProposer``'s serial chain, with

- the aux hidden states tapped from the target's decoder-layer OUTPUTS (the
  residual stream after layers ``eagle_aux_hidden_state_layer_ids``, the
  EAGLE-4 checkpoints' convention) by the base drafter's hooks, so any target
  with a ``.layers`` stack works unmodified;
- 1-D positions for the draft even when the target runs MRoPE (a text-only
  stream has the same position in every section), which keeps the draft's
  mid-steps graph-capturable;
- the low-rank candidate head (``Eagle4LlamaModel.compute_draft_ids``);
- the whole round from one CUDA graph of fused kernels (``eagle4_fast``,
  ``ATOM_EAGLE4_FAST_DRAFT``): a chain for one request or a batch, or, with
  ``ATOM_SPEC_TREE``, each request's dynamic draft tree.
"""

import logging

import torch
import triton
import triton.language as tl

from atom.spec_decode.drafter import AuxCaptureSpec, Drafter
from atom.spec_decode.eagle_proposer import EagleProposer
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


EAGLE4_ARCHITECTURES = ("LlamaForCausalLMEagle4", "Eagle4LlamaModel")


@triton.jit
def _stage_round_kernel(
    TOK_IN,
    NEXT_IN,
    POS_IN,
    SLOT_IN,
    LAST_IN,
    TABLE_IN,
    TOK,
    NEXT,
    POS,
    SLOT,
    LAST,
    TABLE,
    W,
    Q: tl.constexpr,
    QP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The fused round's inputs into its graph's fixed buffers, one launch."""
    r = tl.arange(0, QP)
    ok = r < Q
    tl.store(TOK + r, tl.load(TOK_IN + r, mask=ok).to(tl.int64), mask=ok)
    tl.store(POS + r, tl.load(POS_IN + r, mask=ok).to(tl.int64), mask=ok)
    tl.store(SLOT + r, tl.load(SLOT_IN + r, mask=ok).to(tl.int64), mask=ok)
    tl.store(NEXT, tl.load(NEXT_IN).to(tl.int64))
    tl.store(LAST, tl.load(LAST_IN).to(tl.int64))
    for i in range(0, W, BLOCK):
        o = i + tl.arange(0, BLOCK)
        tl.store(TABLE + o, tl.load(TABLE_IN + o, mask=o < W), mask=o < W)


@triton.jit
def _stage_batch_kernel(
    TOK_IN,
    NEXT_IN,
    POS_IN,
    SLOT_IN,
    LAST_IN,
    TABLE_IN,
    TOK,
    NEXT,
    POS,
    SLOT,
    LAST,
    TABLE,
    W,
    N: tl.constexpr,
    NP: tl.constexpr,
    B: tl.constexpr,
    BP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The batched round's inputs into its graph's fixed buffers, one launch."""
    r = tl.arange(0, NP)
    ok = r < N
    tl.store(TOK + r, tl.load(TOK_IN + r, mask=ok).to(tl.int64), mask=ok)
    tl.store(POS + r, tl.load(POS_IN + r, mask=ok).to(tl.int64), mask=ok)
    tl.store(SLOT + r, tl.load(SLOT_IN + r, mask=ok).to(tl.int64), mask=ok)
    b = tl.arange(0, BP)
    okb = b < B
    tl.store(NEXT + b, tl.load(NEXT_IN + b, mask=okb).to(tl.int64), mask=okb)
    tl.store(LAST + b, tl.load(LAST_IN + b, mask=okb).to(tl.int64), mask=okb)
    for i in range(0, W, BLOCK):
        o = i + tl.arange(0, BLOCK)
        tl.store(TABLE + o, tl.load(TABLE_IN + o, mask=o < W), mask=o < W)


def _capture(round_) -> "torch.cuda.CUDAGraph":
    """Warm ``round_`` on the capture stream, then record it, inside the TP
    group's capture context (its all-reduces, at TP > 1, need it)."""
    from aiter.dist.parallel_state import graph_capture

    with graph_capture() as ctx:
        stream = ctx.stream
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            round_()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            round_()
    return graph


def _residual_after(output, _module) -> torch.Tensor:
    """A decoder layer's ``(hidden, residual)`` -> the residual stream after it."""
    hidden, residual = output[0], output[1]
    return hidden + residual


class Eagle4Proposer(EagleProposer):
    def _aux_capture_spec(self, target_model):
        ids = tuple(int(i) for i in self.speculative_config.eagle3_aux_layer_ids)
        if not ids:
            return None
        return AuxCaptureSpec(
            layer_ids=ids,
            hidden_size=getattr(
                self.config.hf_config, "text_config", self.config.hf_config
            ).hidden_size,
            extract=_residual_after,
            capture="output",
        )

    def arm_aux_capture(self, target_model) -> None:
        # the hook-based capture of the base drafter, not EAGLE3's
        # model-side ``set_aux_hidden_state_layers``
        Drafter.arm_aux_capture(self, target_model)

    def _declare_draft_graphs(self):
        use_mrope = self.runner.use_mrope
        self.runner.use_mrope = False
        try:
            return super()._declare_draft_graphs()
        finally:
            self.runner.use_mrope = use_mrope

    # ---- one sequence: the fused round (``eagle4_fast``) ----------------

    def _fast_eligible(self, target_token_ids) -> bool:
        if not envs.ATOM_EAGLE4_FAST_DRAFT:
            return False
        fwd = get_forward_context()
        ctx = fwd.context
        if (
            ctx is None
            or ctx.is_prefill
            or ctx.scheduled_bs != 1
            or not self._captures_aux
        ):
            return False
        q = target_token_ids.shape[0]
        if getattr(self.runner, "tree_verify", None) is not None:
            return False  # draft trees come from the batched round, one request or more
        return 0 < q <= self.mtp_k + 1 and self._fast_draft() is not None

    def _fast_draft(self):
        if getattr(self, "_fast", None) is None and not getattr(
            self, "_fast_refused", False
        ):
            from atom.spec_decode.eagle4_fast import Eagle4FastDraft

            try:
                self._fast = Eagle4FastDraft(
                    self.model, self.config.max_model_len, self.mtp_k, self.device,
                    mxfp4=envs.ATOM_EAGLE4_DRAFT_MXFP4,
                )  # fmt: skip
                self._fast_buffers = {}
            except ValueError as why:
                logger.warning("EAGLE-4 fused draft off: %s", why)
                self._fast_refused = True
                self._fast = None
        return self._fast

    def _fast_propose(
        self, target_token_ids, target_positions, next_token_ids, last_token_indices
    ):
        fast = self._fast
        fwd = get_forward_context()
        md = fwd.attn_metadata
        q = target_token_ids.shape[0]
        layer = fwd.kv_cache_data[
            f"layer_{self.model.midlayer.self_attn.attn.layer_num}"
        ]
        buf = self._fast_buffers.get(q)
        dev = self.device
        if buf is None:
            buf = {
                "tokens": torch.zeros(q, dtype=torch.int64, device=dev),
                "positions": torch.zeros(q, dtype=torch.int64, device=dev),
                "slots": torch.zeros(q, dtype=torch.int64, device=dev),
                "last": torch.zeros(1, dtype=torch.int64, device=dev),
                "table": torch.zeros_like(md.block_tables[0]),
                "out": torch.zeros(
                    1, self.mtp_k, dtype=next_token_ids.dtype, device=dev
                ),
            }
            self._fast_buffers[q] = buf
            buf["next"] = torch.zeros(1, dtype=torch.int64, device=dev)
        table = md.block_tables[0]
        _stage_round_kernel[(1,)](
            target_token_ids, next_token_ids, target_positions, md.slot_mapping, last_token_indices, table,
            buf["tokens"], buf["next"], buf["positions"], buf["slots"], buf["last"], buf["table"], table.numel(),
            Q=q, QP=triton.next_power_of_2(q), BLOCK=1024, num_warps=4,
        )  # fmt: skip
        aux = [a[:q] for a in self._aux_buffers]
        args = (
            buf["tokens"],
            buf["positions"],
            buf["slots"],
            buf["last"],
            aux,
            buf["table"],
            layer.k_cache,
            layer.v_cache,
            buf["out"],
        )

        def round_():
            # the anchor: the bonus token at the last accepted row
            buf["tokens"].scatter_(0, buf["last"], buf["next"])
            fast.round(*args)

        graph = fast.graphs.get(q)
        if graph is None:
            graph = fast.graphs[q] = _capture(round_)
            logger.info("EAGLE-4 fused draft round captured for %d verified rows", q)
        graph.replay()
        return buf["out"]

    # ---- several sequences: the batched fused round ----------------------

    def _batch_eligible(self, target_token_ids, next_token_ids) -> bool:
        """A decode step of B > 1 requests verifying the same number of rows
        each (all of the chain, or the narrower width of ``verify_len``)."""
        if not envs.ATOM_EAGLE4_FAST_DRAFT:
            return False
        fwd = get_forward_context()
        ctx = fwd.context
        if ctx is None or ctx.is_prefill or ctx.is_dummy_run or not self._captures_aux:
            return False
        b = next_token_ids.shape[0]
        q = int(fwd.attn_metadata.max_seqlen_q)
        trees = getattr(self.runner, "tree_verify", None) is not None
        return (
            (b > 1 or trees) and 1 < q <= self.mtp_k + 1 and target_token_ids.shape[0] == b * q
            and self._fast_draft() is not None
        )  # fmt: skip

    def _batch_depth(self, b: int) -> int:
        """Drafts worth making for a batch of ``b``: what its next step
        verifies (``ATOM_SPEC_VERIFY_LEN``), all of them without a policy."""
        policy = self.runner._verify_len_policy()
        if not policy:
            return self.mtp_k
        from atom.spec_decode.verify_len import drafts_for

        return drafts_for(policy, b, self.mtp_k)

    def _batch_propose(
        self, target_token_ids, target_positions, next_token_ids, last_token_indices
    ):
        """The extension and the chain of B requests as one graph a (B, q)."""
        fast = self._fast
        md = get_forward_context().attn_metadata
        b = next_token_ids.shape[0]
        n = target_token_ids.shape[0]
        layer = get_forward_context().kv_cache_data[
            f"layer_{self.model.midlayer.self_attn.attn.layer_num}"
        ]
        depth = self._batch_depth(b)
        tv = getattr(
            self.runner, "tree_verify", None
        )  # tree speculation (ATOM_SPEC_TREE)
        tree_in = tv is not None and tv.walked is md
        key = ("batch", b, n, depth, tree_in)
        bufs = self.__dict__.setdefault("_batch_buffers", {})
        buf = bufs.get(key)
        dev = self.device
        if buf is None:
            buf = {
                "tokens": torch.zeros(n, dtype=torch.int64, device=dev),
                "positions": torch.zeros(n, dtype=torch.int64, device=dev),
                "slots": torch.zeros(n, dtype=torch.int64, device=dev),
                "last": torch.zeros(b, dtype=torch.int64, device=dev),
                "next": torch.zeros(b, dtype=torch.int64, device=dev),
                "table": torch.zeros_like(md.block_tables[:b]),
                "rows": torch.zeros(3, self.mtp_k, b, dtype=torch.int64, device=dev),
                "drafts": torch.zeros(self.mtp_k, b, dtype=torch.int64, device=dev),
                "out": torch.zeros(
                    b, self.mtp_k, dtype=next_token_ids.dtype, device=dev
                ),
            }
            if tv is not None:
                # the draft tree's beam keys / values
                steps, width, _ = tv.spec_for(b)
                entries = b * steps * width
                buf["scratch"] = tuple(
                    torch.zeros(
                        entries, fast.H, fast.HD, dtype=torch.bfloat16, device=dev
                    )
                    for _ in range(2)
                )
            bufs[key] = buf
        table = md.block_tables[:b]
        _stage_batch_kernel[(1,)](
            target_token_ids, next_token_ids, target_positions, md.slot_mapping, last_token_indices, table,
            buf["tokens"], buf["next"], buf["positions"], buf["slots"], buf["last"], buf["table"], table.numel(),
            N=n, NP=triton.next_power_of_2(n), B=b, BP=triton.next_power_of_2(b), BLOCK=1024, num_warps=4,
        )  # fmt: skip
        aux = [a[:n] for a in self._aux_buffers]

        def round_():
            if tree_in:
                # each request's path rows, in path order: target tokens and aux
                tv.reorder_aux(self._aux_buffers, b)
                buf["tokens"].copy_(tv.toks[:b, : n // b].reshape(-1))
            # each request's anchor: the bonus token at its last accepted row
            buf["tokens"].scatter_(0, buf["last"], buf["next"])
            args = (
                buf["tokens"], buf["positions"], buf["slots"], buf["last"], aux, buf["table"],
                layer.k_cache, layer.v_cache, buf["rows"], buf["drafts"],
            )  # fmt: skip
            if tv is not None:
                steps, width, nodes = tv.spec_for(b)
                drafts, parents, depths = fast.dyn_tree_round(
                    *args[:-1], buf["scratch"], steps, width, nodes
                )
                buf["out"][:, : nodes - 1].copy_(drafts)
                # drafts past this tree (a next verify that takes more): the root's best child
                buf["out"][:, nodes - 1 :].copy_(
                    drafts[:, :1].expand(-1, self.mtp_k - nodes + 1)
                )
                # under each request's state slot: the next verify reads it there
                tv.publish(
                    self.runner,
                    md.gdn_metadata.spec_state_indices_tensor[:b, 0].long(),
                    parents,
                    depths,
                )
                return
            fast.batch_round(*args, depth)
            buf["out"].copy_(buf["drafts"].t())

        graph = fast.graphs.get(key)
        if graph is None:
            graph = fast.graphs[key] = _capture(round_)
            logger.info(
                "EAGLE-4 fused draft round captured for %d requests of %d verified rows, %d drafts",
                b, n // b, depth,
            )  # fmt: skip
        graph.replay()
        return buf["out"]

    def compute_draft_kv(self, positions, hidden_states, next_token_ids) -> None:
        if positions.dim() == 2:
            positions = positions[0]
        return super().compute_draft_kv(positions, hidden_states, next_token_ids)

    def propose(
        self,
        target_token_ids,
        target_positions,
        target_hidden_states,
        num_reject_tokens,
        next_token_ids,
        last_token_indices,
    ):
        if target_positions.dim() == 2:
            target_positions = target_positions[0]
        if self._fast_eligible(target_token_ids):
            out = self._fast_propose(
                target_token_ids, target_positions, next_token_ids, last_token_indices
            )
        elif self._batch_eligible(target_token_ids, next_token_ids):
            out = self._batch_propose(
                target_token_ids, target_positions, next_token_ids, last_token_indices
            )
        else:
            out = super().propose(
                target_token_ids,
                target_positions,
                target_hidden_states,
                num_reject_tokens,
                next_token_ids,
                last_token_indices,
            )
        return out
