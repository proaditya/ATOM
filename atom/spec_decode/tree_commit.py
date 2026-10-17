# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A batch's tree verify, sampled and committed.

``tree_walk`` picks each request's accepted path (from the root, the child
whose token is the target's token at the current node) and writes ATOM's
chain verdict of it, so the scheduler and the drafter see the path as an
ordinary chain of accepted drafts. ``commit`` then makes the pools say so:

- GDN state: the path's replay records (``gdn_tree``) replayed into each
  request's state slot, in place;
- GDN conv: the window a chain of the path's inputs would have left, at
  rows 0..2 (where the next tree step reads it);
- attention K / V: the path's rows moved into consecutive slots (row i <-
  row path[i]; a row is never a source after it was a target, since path[i]
  >= i).
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.gdn_tree import RECORD_FLOATS


@triton.jit
def _walk_kernel(
    YV,
    IDS,
    PAR,
    s_n,
    PATH,
    KOUT,
    TOKS,
    OUT,
    s_out,
    VERDICT,
    s_v,
    N: tl.constexpr,
    NP: tl.constexpr,
    NSPEC: tl.constexpr,
    OUTP: tl.constexpr,
):
    """One request: the greedy walk -> PATH (path nodes, the last repeated),
    KOUT (nodes on it), TOKS (the target's token after each path node: what
    the draft extends over), OUT (emitted tokens, then -1), VERDICT columns
    (accepted drafts, rejected drafts against NSPEC, the last emitted token's
    flat row in path order)."""
    b = tl.program_id(0)
    r = tl.arange(0, NP)
    rm = r < N
    yv = tl.load(YV + b * s_n + r, mask=rm, other=-1).to(tl.int64)
    ids = tl.load(IDS + b * s_n + r, mask=rm, other=-2).to(tl.int64)
    par = tl.load(PAR + b * s_n + r, mask=rm & (r > 0), other=-3).to(tl.int64)
    cur = tl.full((), 0, tl.int64)
    k = tl.full((), 1, tl.int64)
    path = tl.zeros([NP], dtype=tl.int64)
    for _ in tl.static_range(N - 1):
        ycur = tl.sum(tl.where(r == cur, yv, 0))
        cand = (par == cur) & (ids == ycur) & (r > 0) & rm
        c = tl.min(tl.where(cand, r, NP)).to(tl.int64)
        found = c < NP
        path = tl.where((r == k) & found, c, path)
        k = k + found.to(tl.int64)
        cur = tl.where(found, c, cur)
    last = tl.sum(tl.where(r == k - 1, path, 0))
    path = tl.where(r < k, path, last)
    toks = tl.sum(tl.where(path[:, None] == r[None, :], yv[None, :], 0), 1)
    tl.store(PATH + b * s_n + r, path, mask=rm)
    tl.store(KOUT + b, k.to(tl.int32))
    tl.store(TOKS + b * s_n + r, toks, mask=rm)
    o = tl.arange(0, OUTP)
    emitted = tl.sum(tl.where(o[:, None] == r[None, :], toks[None, :], 0), 1)
    tl.store(
        OUT + b * s_out + o,
        tl.where(o < k, emitted, -1).to(tl.int32),
        mask=o < NSPEC + 1,
    )
    tl.store(VERDICT + b, (k - 1).to(tl.int32))
    tl.store(VERDICT + s_v + b, (NSPEC - (k - 1)).to(tl.int32))
    tl.store(VERDICT + 2 * s_v + b, (b * N + k - 1).to(tl.int32))


def tree_walk(yv, ids, parents, path, k_out, toks, out, verdict, nspec: int) -> None:
    """``yv`` / ``ids`` / ``parents`` [bs, N] (the target's token at each
    node, the node's own token, its parent); ``path`` / ``toks`` [bs, N]
    int64, ``k_out`` [bs] int32; ``out`` [bs, nspec + 1] int32, ``verdict``
    [3, bs] int32 (ATOM's rejection-sampler outputs)."""
    bs, n = yv.shape
    assert (
        ids.stride(0)
        == parents.stride(0)
        == path.stride(0)
        == toks.stride(0)
        == yv.stride(0)
    )
    _walk_kernel[(bs,)](
        yv, ids, parents, yv.stride(0), path, k_out, toks, out, out.stride(0), verdict, verdict.stride(0),
        N=n, NP=max(16, triton.next_power_of_2(n)), NSPEC=nspec, OUTP=triton.next_power_of_2(nspec + 1), num_warps=1,
    )  # fmt: skip


# ---- commit ------------------------------------------------------------------------------


@triton.jit
def _state_commit_kernel(
    S,
    s_layer,
    s_slot,
    SLOT,
    REC,
    s_rec_layer,
    s_rec_req,
    PATH,
    s_path,
    KOUT,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    MAXN: tl.constexpr,
    RECF: tl.constexpr,
):
    """S[slot] <- the state after the path's k nodes, from their records."""
    lh = tl.program_id(0)
    vb = tl.program_id(1)
    b = tl.program_id(2)
    layer = lh // HV
    hv = lh % HV
    k = tl.load(KOUT + b)
    slot = tl.load(SLOT + b).to(tl.int64)
    if (slot >= 0) & (k > 0):
        vs = vb * BV + tl.arange(0, BV)
        ks = tl.arange(0, K)
        tile = hv * K * V + vs[:, None] * K + ks[None, :]
        base = S + layer.to(tl.int64) * s_layer + slot * s_slot
        st = tl.load(base + tile).to(
            tl.float32
        )  # replayed in fp32 whatever the pool's dtype
        rb = REC + layer.to(tl.int64) * s_rec_layer + b.to(tl.int64) * s_rec_req
        for t in tl.static_range(MAXN):
            if t < k:
                node = tl.load(PATH + b * s_path + t)
                r = rb + (node * HV + hv) * RECF
                st = (
                    tl.load(r + V + K) * st
                    + tl.load(r + vs)[:, None] * tl.load(r + V + ks)[None, :]
                )
        tl.store(base + tile, st.to(S.dtype.element_ty))


@triton.jit
def _window_row(base, s_tok, rb, s_raw_node, PATH, fm, j):
    """Row j of [h0, h1, h2, x_path[0], ...]."""
    hist = tl.load(base + tl.minimum(j, 2) * s_tok, mask=fm & (j < 3), other=0.0)
    node = tl.load(PATH + tl.maximum(j - 3, 0))
    x = tl.load(rb + node * s_raw_node, mask=fm & (j >= 3), other=0.0)
    return tl.where(j < 3, hist, x)


@triton.jit
def _conv_commit_kernel(
    C,
    s_layer,
    s_slot,
    s_tok,
    s_dim,
    SLOT,
    RAW,
    s_raw_layer,
    s_raw_req,
    s_raw_node,
    PATH,
    s_path,
    KOUT,
    DIM,
    BLOCK: tl.constexpr,
):
    """The next window at rows 0..2: the last three of [h0, h1, h2, x_path[0]
    .. x_path[k-1]] (h the window the verify read, rows 0..2)."""
    layer = tl.program_id(0)
    f = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    b = tl.program_id(2)
    fm = f < DIM
    k = tl.load(KOUT + b).to(tl.int64)
    slot = tl.load(SLOT + b).to(tl.int64)
    if (slot >= 0) & (k > 0):
        base = C + layer.to(tl.int64) * s_layer + slot * s_slot + f * s_dim
        rb = RAW + layer.to(tl.int64) * s_raw_layer + b.to(tl.int64) * s_raw_req + f
        pb = PATH + b * s_path
        w0 = _window_row(base, s_tok, rb, s_raw_node, pb, fm, k)
        w1 = _window_row(base, s_tok, rb, s_raw_node, pb, fm, k + 1)
        w2 = _window_row(base, s_tok, rb, s_raw_node, pb, fm, k + 2)
        tl.store(base, w0, mask=fm)
        tl.store(base + s_tok, w1, mask=fm)
        tl.store(base + 2 * s_tok, w2, mask=fm)


@triton.jit
def _kv_compact_kernel(
    K_PTRS,
    V_PTRS,
    SLOTS,
    s_slots,
    PATH,
    s_path,
    KOUT,
    BS: tl.constexpr,
    X: tl.constexpr,
    HD: tl.constexpr,
    KVH: tl.constexpr,
    MAXN: tl.constexpr,
):
    layer = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    k = tl.load(KOUT + b)
    kc = tl.load(K_PTRS + layer).to(tl.pointer_type(tl.bfloat16))
    vc = tl.load(V_PTRS + layer).to(tl.pointer_type(tl.bfloat16))
    dd = tl.arange(0, HD)
    for i in tl.static_range(1, MAXN):
        if i < k:
            src = tl.load(PATH + b * s_path + i)
            ss = tl.load(SLOTS + b * s_slots + src).to(tl.int64)
            sd = tl.load(SLOTS + b * s_slots + i).to(tl.int64)
            if (src != i) & (ss >= 0) & (sd >= 0):
                hs = (ss // BS) * KVH + h
                hd = (sd // BS) * KVH + h
                kv_k = tl.load(
                    kc + ((hs * (HD // X) + dd // X) * BS + ss % BS) * X + dd % X
                )
                kv_v = tl.load(
                    vc
                    + ((hs * (BS // X) + (ss % BS) // X) * HD + dd) * X
                    + (ss % BS) % X
                )
                tl.store(
                    kc + ((hd * (HD // X) + dd // X) * BS + sd % BS) * X + dd % X, kv_k
                )
                tl.store(
                    vc
                    + ((hd * (BS // X) + (sd % BS) // X) * HD + dd) * X
                    + (sd % BS) % X,
                    kv_v,
                )


def commit(
    state,
    conv,
    slots,
    records,
    raw,
    path,
    k_out,
    k_ptrs,
    v_ptrs,
    row_slots,
    block_size: int,
    kv_heads: int,
    head_dim: int,
) -> None:
    """``state`` [L, slots, HV, K, V] (VK memory), ``conv`` [L, slots, tok,
    dim] (its window at rows 0..2); ``slots`` [bs] each request's state slot; ``records`` [L, bs, N, HV, RECORD_FLOATS],
    ``raw`` [L, bs, N, dim] the nodes' pre-conv inputs; ``path`` [bs, N],
    ``k_out`` [bs]; ``k_ptrs`` / ``v_ptrs`` the attention layers' caches,
    ``row_slots`` [bs, N] each row's cache slot."""
    L, _, HV, K, V = state.shape
    bs, n = path.shape
    assert slots.is_contiguous() and k_out.is_contiguous()
    assert (
        state.stride(-2) == 1 and state.stride(-1) == K
    ), "the state pool must be VK in memory"
    _state_commit_kernel[(L * HV, V // 32, bs)](
        state, state.stride(0), state.stride(1), slots, records, records.stride(0), records.stride(1),
        path, path.stride(0), k_out, HV=HV, K=K, V=V, BV=32, MAXN=n, RECF=records.shape[-1], num_warps=4,
    )  # fmt: skip
    dim = conv.shape[-1]
    _conv_commit_kernel[(conv.shape[0], triton.cdiv(dim, 1024), bs)](
        conv, conv.stride(0), conv.stride(1), conv.stride(2), conv.stride(3), slots, raw, raw.stride(0),
        raw.stride(1), raw.stride(2), path, path.stride(0), k_out, dim, BLOCK=1024, num_warps=4,
    )  # fmt: skip
    if k_ptrs is not None and k_ptrs.numel():
        _kv_compact_kernel[(k_ptrs.numel(), kv_heads, bs)](
            k_ptrs, v_ptrs, row_slots, row_slots.stride(0), path, path.stride(0), k_out,
            BS=block_size, X=8, HD=head_dim, KVH=kv_heads, MAXN=n, num_warps=4,
        )  # fmt: skip


def record_buffers(layers: int, bs: int, n: int, hv: int, dim: int, device):
    """(records, raw) for a tree verify of up to ``bs`` requests of ``n`` nodes."""
    rec = torch.zeros(
        layers, bs, n, hv, RECORD_FLOATS, dtype=torch.float32, device=device
    )
    raw = torch.zeros(layers, bs, n, dim, dtype=torch.bfloat16, device=device)
    return rec, raw
