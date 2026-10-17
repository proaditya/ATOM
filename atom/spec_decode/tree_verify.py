# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tree speculation on the per-op path (``ATOM_SPEC_TREE``).

Each request's verify rows are the nodes of its draft tree in topological
order (node 0 the root, the last accepted token): the EAGLE-2 dynamic tree
the drafter grew for it (``dyn:STEPS:WIDTH:NODES``, NODES rows), published
under the request's GDN state slot and loaded into the verify's tables. To the scheduler and the runner the step is an ordinary
rectangular verify of N rows; the tree lives in the layers and around the
sampler:

- full-attention layers see, past the context, only each row's
  ancestor-or-self rows (the split-KV verify's ``tree_mask``), RoPE'd at the
  context end + the row's depth (``rope_delta``);
- GDN layers continue every node from its parent (``gdn_tree``), writing
  replay records instead of state;
- the sampler walks the tree (``tree_commit.tree_walk``) and hands ATOM the
  path as a chain verdict; the commit then makes the pools look as if the
  path had been verified as a chain (``tree_commit.commit``);
- the drafter extends over the path: its rows' target tokens (``toks``) and
  aux hidden states, put in path order here.

``ATOM_SPEC_TREE``: ``dyn:STEPS:WIDTH:NODES`` (--num-speculative-tokens
NODES - 1), or a tree a batch size, ``B1=dyn:...,B2=dyn:...`` (from B
requests on, the first from 1; a step verifies its tree's nodes).
"""

import logging

import torch
import triton
import triton.language as tl

from atom.spec_decode import tree_commit as tc

logger = logging.getLogger("atom")

_RING = 8


def parse_dynamic(spec: str) -> tuple[int, int, int]:
    """``dyn:STEPS:WIDTH:NODES`` -> (steps, width, nodes)."""
    kind, *args = spec.split(":")
    if kind != "dyn" or len(args) != 3:
        raise ValueError(f"ATOM_SPEC_TREE={spec!r}: dyn:STEPS:WIDTH:NODES")
    steps, width, nodes = (int(a) for a in args)
    if not (
        steps >= 1
        and width in (1, 2, 4, 8)
        and steps * width <= 64
        and 2 <= nodes <= min(64, width + (steps - 1) * width * width + 1)
    ):
        raise ValueError(
            f"ATOM_SPEC_TREE={spec!r}: needs WIDTH 1 / 2 / 4 / 8, STEPS * WIDTH <= 64 and NODES - 1 <= the nodes scored"
        )
    return steps, width, nodes


def parse_dynamic_policy(spec: str) -> tuple[tuple[int, tuple[int, int, int]], ...]:
    """``dyn:S:W:N`` or ``B1=dyn:S:W:N,B2=dyn:...`` (from B requests on) ->
    ((min batch, (steps, width, nodes)), ...), the first from batch 1."""
    if "=" not in spec:
        return ((1, parse_dynamic(spec)),)
    out = {}
    for part in spec.split(","):
        bs, _, tree = part.strip().partition("=")
        out[int(bs)] = parse_dynamic(tree)
    if 1 not in out:
        raise ValueError(
            f"ATOM_SPEC_TREE={spec!r}: the policy needs an entry from batch 1 (1=dyn:...)"
        )
    return tuple(sorted(out.items()))


@triton.jit
def _tables_kernel(
    SLOTS,
    PAR_ST,
    DEP_ST,
    s_st,
    PAR,
    MASK,
    s_tab,
    DELTA,
    N: tl.constexpr,
    NP: tl.constexpr,
):
    """Request b's tree, published by the drafter under its state slot, into
    the verify's tables (its first N nodes): parents, ancestor-or-self masks,
    RoPE offsets (flat, N rows a request)."""
    b = tl.program_id(0)
    r = tl.arange(0, NP)
    rm = r < N
    slot = tl.load(SLOTS + b).to(tl.int64)
    par = tl.load(PAR_ST + slot * s_st + r, mask=rm & (slot >= 0), other=-1)
    dep = tl.load(DEP_ST + slot * s_st + r, mask=rm & (slot >= 0), other=0)
    par = tl.where(slot >= 0, par, r.to(par.dtype) - 1)  # a padded row: a chain
    dep = tl.where(slot >= 0, dep, r.to(dep.dtype))
    m = tl.where(r == 0, 1, 0).to(tl.int64)
    for i in tl.static_range(1, N):
        p = tl.sum(tl.where(r == i, par, 0))
        mp = tl.sum(tl.where(r == p, m, 0))
        m = tl.where(r == i, mp | (1 << i), m)
    tl.store(PAR + b * s_tab + r, par.to(tl.int32), mask=rm)
    tl.store(MASK + b * s_tab + r, m, mask=rm)
    tl.store(DELTA + b * N + r, dep.to(tl.int64) - r, mask=rm)


class TreeVerify:
    def __init__(self, runner, spec: str) -> None:
        # the drafter's own tree a request, published each round
        self.policy = parse_dynamic_policy(spec)
        nodes = max(d[2] for _, d in self.policy)
        parents = list(range(-1, nodes - 1))  # until a request's first round: a chain
        n = len(parents)
        mtp_k = runner.drafter.mtp_k
        if mtp_k != n - 1:
            raise ValueError(
                f"ATOM_SPEC_TREE={spec} has {n - 1} drafts, --num-speculative-tokens is {mtp_k}"
            )
        self.n = n  # the most nodes a verify takes (the buffers' width)
        self.n_step = n  # this step's (a batch policy's nodes for its size)
        self.spec = spec
        dev = runner.device
        self.device = dev
        max_bs = max(
            runner.config.max_num_seqs,
            max(getattr(runner, "capture_sizes", [1]) or [1]),
        )
        self.max_bs = max_bs
        depth = []
        for p in parents:
            depth.append(0 if p < 0 else depth[p] + 1)
        masks = []
        for i in range(n):
            m, cur = 1 << i, i
            while parents[cur] >= 0:
                cur = parents[cur]
                m |= 1 << cur
            masks.append(m)
        self.parents = (
            torch.tensor(parents, dtype=torch.int32, device=dev)
            .repeat(max_bs, 1)
            .contiguous()
        )
        self.mask = (
            torch.tensor(masks, dtype=torch.int64, device=dev)
            .repeat(max_bs, 1)
            .contiguous()
        )
        # read by every target forward (zero but on a verify): as long as the longest
        tokens = max(max_bs * n, runner.config.max_num_batched_tokens)
        self.rope_pattern = (
            torch.tensor(depth, device=dev) - torch.arange(n, device=dev)
        ).repeat(-(-tokens // n))
        self.rope_delta = torch.zeros_like(self.rope_pattern)
        self._verify_mode = False
        self.path = torch.zeros(max_bs, n, dtype=torch.int64, device=dev)
        self.kout = torch.zeros(max_bs, dtype=torch.int32, device=dev)
        self.toks = torch.zeros(max_bs, n, dtype=torch.int64, device=dev)
        self.yv = torch.zeros(max_bs, n, dtype=torch.int64, device=dev)
        self.ids = torch.zeros(max_bs, n, dtype=torch.int64, device=dev)
        self.outs = torch.zeros(_RING, max_bs * n, dtype=torch.int32, device=dev)
        self.verdicts = torch.zeros(_RING, 3, max_bs, dtype=torch.int32, device=dev)
        self._ring = 0
        self.records = self.raw = None
        self.gdn_index = {}
        self._kv = None
        # the step whose verify was walked (its attention metadata)
        self.walked = None
        self._store = None  # each state slot's tree (parents, depths)
        # each request's first verify row (n_step apart) and its index, for the
        # FlyDSL GDN chunk
        self.req_rows = torch.arange(max_bs, dtype=torch.int32, device=dev)
        self.cu_rows = torch.arange(max_bs + 1, dtype=torch.int32, device=dev) * n
        self._cu_n = n
        logger.info("Tree speculation %s: %d nodes, parents %s", spec, n, parents)

    # ---- install --------------------------------------------------------------

    def attach(self, model) -> None:
        """Point the model's layers at the tree tables, and size the record
        buffers from its GDN layers."""
        from atom.model_ops.attention_gdn import GatedDeltaNet
        from atom.model_ops.attention_mha import PagedAttentionImpl

        gdn = [m for m in model.modules() if isinstance(m, GatedDeltaNet)]
        for i, m in enumerate(sorted(gdn, key=lambda m: m.layer_num)):
            m.tree = self
            self.gdn_index[m.layer_num] = i
        self.attn_layers = []
        for m in model.modules():
            if isinstance(m, PagedAttentionImpl):
                if not m._verify_splitkv_ok():
                    # only the split-KV verify kernel takes the ancestor mask
                    raise ValueError(
                        f"ATOM_SPEC_TREE: attention layer {m.layer_num} cannot run the split-KV verify "
                        "(needs a bf16 KV cache, no sinks / sliding window, ATOM_USE_UNIFIED_ATTN=0)"
                    )
                m.tree_mask = self.mask
                self.attn_layers.append(f"layer_{m.layer_num}")
            if (
                hasattr(m, "rotary_emb")
                and hasattr(m, "attn")
                and hasattr(m, "qkv_proj")
            ):
                m.tree_rope_delta = self.rope_delta
        if gdn:
            g = gdn[0]
            hv = g.num_v_heads // g.tp_size
            dim = g.conv1d.weight.size(0)
            self.records, self.raw = tc.record_buffers(
                len(gdn), self.max_bs, self.n, hv, dim, self.device
            )
        logger.info(
            "Tree verify on %d GDN layers: records %.2f GB, inputs %.2f GB", len(gdn),
            self.records.numel() * 4 / 2**30 if gdn else 0, self.raw.numel() * 2 / 2**30 if gdn else 0,
        )  # fmt: skip

    def set_width(self, n: int) -> None:
        """This step's verify width (``n_step``) and the row starts it implies."""
        self.n_step = n
        if n != self._cu_n:
            torch.mul(
                torch.arange(
                    self.cu_rows.shape[0], dtype=torch.int32, device=self.device
                ),
                n,
                out=self.cu_rows,
            )
            self._cu_n = n

    def spec_for(self, bs: int) -> tuple[int, int, int]:
        """The tree (steps, width, nodes) a batch of ``bs`` drafts."""
        tree = self.policy[0][1]
        for threshold, d in self.policy:
            if bs >= threshold:
                tree = d
        return tree

    def set_mode(self, verify: bool) -> None:
        """Before a target forward: RoPE offsets for a verify's rows, none
        for a prefill's (the buffer is read by every target forward)."""
        if verify != self._verify_mode:
            self.rope_delta.copy_(
                self.rope_pattern if verify else torch.zeros_like(self.rope_pattern)
            )
            self._verify_mode = verify

    # ---- the drafter's trees, per request -------------------------------------

    def _tables(self, runner):
        """Each state slot's published tree: (parents, depths) [slots, N]."""
        if self._store is None:
            slots = runner.mamba_v_cache.shape[1]
            chain = torch.arange(self.n, device=self.device)
            self._chain = (chain - 1, chain)
            self._store = (
                (chain - 1).repeat(slots, 1).contiguous(),
                chain.repeat(slots, 1).contiguous(),
            )
        return self._store

    def reset(self, runner, slots: torch.Tensor) -> None:
        """A prefill's requests: a chain until their drafter publishes a tree."""
        par, dep = self._tables(runner)
        par.index_copy_(0, slots, self._chain[0].expand(slots.shape[0], -1))
        dep.index_copy_(0, slots, self._chain[1].expand(slots.shape[0], -1))

    def publish(
        self, runner, slots: torch.Tensor, parents: torch.Tensor, depths: torch.Tensor
    ) -> None:
        """The drafter's trees [B, nodes] under the B requests' state slots;
        past ``nodes`` (a next verify that takes more), the root's children."""
        par, dep = self._tables(runner)
        pad = self.n - parents.shape[1]
        if pad:
            parents = torch.nn.functional.pad(parents, (0, pad), value=0)
            depths = torch.nn.functional.pad(depths, (0, pad), value=1)
        par.index_copy_(0, slots, parents)
        dep.index_copy_(0, slots, depths)

    def load(self, runner, slots: torch.Tensor, n: int) -> None:
        """Before a verify of ``n`` rows a request: each request's tree (its
        first ``n`` nodes) into the verify's tables."""
        par, dep = self._tables(runner)
        bs = slots.shape[0]
        self.set_width(n)
        _tables_kernel[(bs,)](
            slots, par, dep, par.stride(0), self.parents, self.mask, self.parents.stride(0), self.rope_delta,
            N=n, NP=max(16, triton.next_power_of_2(n)), num_warps=1,
        )  # fmt: skip
        self._verify_mode = True

    # ---- the sampler -------------------------------------------------------

    def verdict(self, metadata, target_logits, bonus_token_ids, target_token_ids):
        """The rejection sampler's (output tokens, verdict) for a tree verify."""
        from atom.utils.forward_context import get_forward_context

        n = self.n_step
        nspec = (
            metadata.num_spec_steps
        )  # what the KV reservation holds, rejected against
        bs = bonus_token_ids.shape[0]
        yv, ids = self.yv[:bs, :n], self.ids[:bs, :n]
        if target_token_ids is None:
            yv[:, : n - 1].copy_(target_logits.argmax(-1).view(bs, n - 1))
        else:
            yv[:, : n - 1].copy_(target_token_ids.view(bs, n - 1))
        yv[:, n - 1].copy_(bonus_token_ids.view(-1))
        ids.copy_(metadata.input_ids[: bs * n].view(bs, n))
        i = self._ring = (self._ring + 1) % _RING
        out, verdict = (
            self.outs[i, : bs * (nspec + 1)].view(bs, nspec + 1),
            self.verdicts[i, :, :bs],
        )
        tc.tree_walk(
            yv, ids, self.parents[:bs, :n], self.path[:bs, :n], self.kout[:bs], self.toks[:bs, :n], out, verdict, nspec,
        )  # fmt: skip
        self.walked = get_forward_context().attn_metadata
        return out, verdict

    # ---- the commit -------------------------------------------------------

    def _attention_caches(self, fwd):
        if self._kv is None:
            ks, vs = [], []
            # the target's attention layers (not the drafter's): one shape
            for name in self.attn_layers:
                layer = fwd.kv_cache_data[name]
                k = layer.k_cache
                if k.dim() != 5 or k.dtype != torch.bfloat16 or k.shape[4] != 8:
                    raise ValueError(
                        f"ATOM_SPEC_TREE: {name}'s K cache {tuple(k.shape)} {k.dtype} is not bf16 [blocks, kvh, hd/8, bs, 8]"
                    )
                if ks and k.shape != ks[0].shape:
                    raise ValueError(
                        f"ATOM_SPEC_TREE: {name}'s K cache shape differs from the first layer's"
                    )
                ks.append(k)
                vs.append(layer.v_cache)
            ptr = lambda ts: torch.tensor(
                [t.data_ptr() for t in ts], dtype=torch.int64, device=self.device
            )
            self._kv = (ptr(ks), ptr(vs), ks[0].shape[3] if ks else 16, ks[0].shape[1] if ks else 1,
                        ks[0].shape[2] * ks[0].shape[4] if ks else 128)  # fmt: skip
        return self._kv

    def commit(self, metadata, runner) -> None:
        """After the walk: the path into the state, conv and K / V pools."""
        from atom.utils.forward_context import get_forward_context

        gdn = getattr(metadata, "gdn_metadata", None)
        if (
            gdn is None
            or gdn.spec_state_indices_tensor is None
            or metadata is not self.walked
        ):
            return
        bs = gdn.num_spec_decodes
        fwd = get_forward_context()
        k_ptrs, v_ptrs, block, kvh, hd = self._attention_caches(fwd)
        n = self.n_step
        tc.commit(
            runner.mamba_v_cache, runner.mamba_k_cache, gdn.spec_state_indices_tensor[:bs, 0].contiguous(),
            self.records[:, :bs], self.raw[:, :bs], self.path[:bs, :n],
            self.kout[:bs], k_ptrs, v_ptrs, metadata.slot_mapping[: bs * n].view(bs, n), block, kvh, hd,
        )  # fmt: skip

    # ---- the drafter's next extension ---------------------------------------------

    def reorder_aux(self, buffers, bs: int) -> None:
        """The drafter's aux rows of each request into path order."""
        n = self.n_step
        rows = (
            torch.arange(bs, device=self.device)[:, None] * n + self.path[:bs, :n]
        ).reshape(-1)
        for buf in buffers:
            buf[: bs * n].copy_(buf.index_select(0, rows))


def install_tree_verify(runner, spec: str) -> TreeVerify:
    """Tree speculation for this runner: the tables, the layers pointed at
    them, and the sampler / commit / RoPE hooks around the step."""
    from functools import wraps

    tv = runner.tree_verify = TreeVerify(runner, spec)
    tv.attach(runner.model)

    rejection = runner.rejection_sampler
    chain = rejection.forward

    @wraps(chain)
    def forward(metadata, target_logits, bonus_token_ids, *, target_token_ids=None):
        bs = bonus_token_ids.shape[0]
        if rejection.synthetic_acceptance_rates is None and target_logits.shape[
            0
        ] == bs * (tv.n_step - 1):
            return tv.verdict(
                metadata, target_logits, bonus_token_ids, target_token_ids
            )
        return chain(
            metadata, target_logits, bonus_token_ids, target_token_ids=target_token_ids
        )

    rejection.forward = forward

    builder = runner.attn_metadata_builder
    commit = builder.commit_speculative_state

    @wraps(commit)
    def commit_speculative_state(metadata, last_token_indices):
        tv.commit(metadata, runner)
        return commit(metadata, last_token_indices)

    builder.commit_speculative_state = commit_speculative_state

    prefill, decode = builder.prepare_prefill, builder.prepare_decode

    @wraps(prefill)
    def prepare_prefill(*args, **kwargs):
        tv.set_mode(False)
        out = prefill(*args, **kwargs)
        gdn = getattr(out[0], "gdn_metadata", None)
        if gdn is not None and gdn.non_spec_state_indices_tensor is not None:
            tv.reset(
                runner,
                gdn.non_spec_state_indices_tensor[
                    : gdn.num_prefills + gdn.num_decodes
                ].long(),
            )
        return out

    @wraps(decode)
    def prepare_decode(batch, running_bs, running_tokens, max_seqlen_q):
        out = decode(batch, running_bs, running_tokens, max_seqlen_q)
        gdn = getattr(out[0], "gdn_metadata", None)
        if (
            max_seqlen_q > 1
            and gdn is not None
            and gdn.spec_state_indices_tensor is not None
        ):
            tv.load(
                runner,
                gdn.spec_state_indices_tensor[: gdn.num_spec_decodes, 0].contiguous(),
                max_seqlen_q,
            )
        else:
            tv.set_width(tv.n)
            tv.set_mode(max_seqlen_q > 1)
        return out

    builder.prepare_prefill, builder.prepare_decode = prepare_prefill, prepare_decode
    capture = builder.build_for_cudagraph_capture

    @wraps(capture)
    def build_for_cudagraph_capture(*args, **kwargs):
        q = kwargs.get("max_q_len")
        tv.set_mode(q is None or q > 1)
        # the verify width the graph is traced for (its GDN layers read it)
        tv.set_width(tv.n if q is None or q <= 1 else q)
        return capture(*args, **kwargs)

    builder.build_for_cudagraph_capture = build_for_cudagraph_capture
    return tv
