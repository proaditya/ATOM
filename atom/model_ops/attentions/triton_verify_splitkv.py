# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Split-KV paged attention for speculative verify (query lengths the paged
decode kernels do not take).

A verify step runs a few query rows per sequence -- the anchor and its drafts,
already written into the paged cache -- each attending causally to every key up
to its own position. The gluon paged-decode kernel stops at 4 query rows, and
past it ATOM fell back to the unified kernel, which walks the whole context
once per (sequence, head) and, at long context, leaves the memory system idle
(1.2 ms a layer at 32K). This is the flash-decode shape for it, after the
``verify_splitkv`` of SGLang's EAGLE-4 ROCm port (its GQA-packed prefix stage), on
ATOM's paged layouts:

- ``_verify_split_kernel``: one program per (sequence, KV head, split) holds
  every query row of the KV head's group -- query rows x heads per group,
  packed into one MFMA tile -- and runs online softmax over its share of the
  keys, each row masked at its own causal end;
- ``_verify_combine_kernel``: merges a row's splits (log-sum-exp).

The number of splits is fixed per launch shape (graph-safe); each program
sizes its share from the sequence's real length, so splits past the end do
nothing.

K cache [blocks, kv_heads, head_dim / X, block_size, X], V cache [blocks,
kv_heads, block_size / X, head_dim, X] (ATOM's shuffled bf16 layouts).
"""

import torch
import triton
import triton.language as tl

MIN_KEYS_PER_SPLIT = 64
TARGET_PROGRAMS = 512
MAX_SPLITS = 16
# one request's verify (the whole GPU on one context; a split past the keys exits at once):
# MI355X, 16 tree rows (two row blocks), 4K / 8K / 16K keys 36 / 62 / 114 us a layer at
# 16 splits, 26 / 34 / 49 at 64, graph-timed with the keys cache-hot (in a round, 8K: 103 us
# at 16 splits)
MAX_SPLITS_ONE = 64
# query rows a program (of a KV head's group), warps, keys a tile, unpipelined:
# MI355X, 24 / 4 heads of 256, 2K keys, graph-timed: 1 x 16 rows 32 us,
# 16 x 10 rows 75 us (was 162), 32 x 6 rows 107 us (was 302)
ROW_BLOCK = 64
NUM_WARPS = 4
BLOCK_N = 32
NUM_STAGES = 1


@triton.jit
def _verify_split_kernel(
    Q,
    sq_t,
    sq_h,
    KC,
    VC,
    BT,
    sbt,
    CTX,
    QSTART,
    PO,
    PML,
    scale,
    MASK,
    s_mask,
    GROUP: tl.constexpr,
    KVH: tl.constexpr,
    HD: tl.constexpr,
    BS: tl.constexpr,
    X: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_KEYS: tl.constexpr,
    TREE: tl.constexpr,
    BM: tl.constexpr,
):
    # program (sequence, BM-row block of its BLOCK_M group rows), kv head, split
    s = tl.program_id(0) // (BLOCK_M // BM)
    mb = tl.program_id(0) % (BLOCK_M // BM)
    kvh = tl.program_id(1)
    sp = tl.program_id(2)
    q0 = tl.load(QSTART + s)
    qlen = tl.load(QSTART + s + 1) - q0
    ctx = tl.load(CTX + s)
    # rows: query row i of head g of this group at m = i * GROUP + g
    m = mb * BM + tl.arange(0, BM)
    qi = m // GROUP
    g = m % GROUP
    row_ok = qi < qlen
    row_end = ctx - qlen + qi  # last key position row m sees
    if TREE:
        # a tree verify: row i sees the context and the rows set in its
        # ancestor-or-self mask, not every earlier row
        win0 = ctx - qlen
        rmask = tl.load(MASK + s * s_mask + qi, mask=row_ok, other=0)
    # this split's share of the keys [0, ctx)
    per = tl.maximum(tl.cdiv(ctx, SPLITS), MIN_KEYS)
    per = tl.cdiv(per, BLOCK_N) * BLOCK_N
    k_lo = sp * per
    k_hi = tl.minimum(k_lo + per, ctx)
    d = tl.arange(0, HD)
    h = kvh * GROUP + g
    q = tl.load(
        Q + (q0 + qi)[:, None].to(tl.int64) * sq_t + h[:, None] * sq_h + d[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )
    m_i = tl.full((BM,), float("-inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, HD), tl.float32)
    bt = BT + s * sbt
    n = tl.arange(0, BLOCK_N)
    gk = tl.arange(0, HD // X)
    e = tl.arange(0, X)
    vg = tl.arange(0, BLOCK_N // X)
    for st in range(k_lo, k_hi, BLOCK_N):
        kp = st + n
        kin = kp < k_hi
        blk = tl.load(bt + kp // BS, mask=kin, other=0).to(tl.int64)
        hb = blk * KVH + kvh
        koff = (
            (hb[:, None, None] * (HD // X) + gk[None, :, None]) * BS
            + (kp % BS)[:, None, None]
        ) * X + e[None, None, :]
        k = tl.reshape(
            tl.load(KC + koff, mask=kin[:, None, None], other=0.0), (BLOCK_N, HD)
        )
        sc = tl.dot(q, tl.trans(k)) * scale
        ok = kin[None, :] & (kp[None, :] <= row_end[:, None]) & row_ok[:, None]
        if TREE:
            j = kp - win0
            seen = ((rmask[:, None] >> tl.maximum(j, 0)[None, :].to(tl.int64)) & 1) != 0
            ok = ok & ((j < 0)[None, :] | seen)
        sc = tl.where(ok, sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(sc - m_safe[:, None])
        kv0 = st + vg * X
        vblk = tl.load(bt + kv0 // BS, mask=kv0 < k_hi, other=0).to(tl.int64)
        vb = vblk * KVH + kvh
        voff = (
            (vb[:, None, None] * (BS // X) + ((kv0 % BS) // X)[:, None, None]) * HD
            + d[None, :, None]
        ) * X + e[None, None, :]
        v3 = tl.load(VC + voff, mask=(kv0 < k_hi)[:, None, None], other=0.0)
        v = tl.reshape(tl.permute(v3, (0, 2, 1)), (BLOCK_N, HD))
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new
    base = ((s * KVH + kvh) * SPLITS + sp) * BLOCK_M + m
    tl.store(
        PO + base[:, None].to(tl.int64) * HD + d[None, :], acc, mask=row_ok[:, None]
    )
    tl.store(PML + base * 2, m_i, mask=row_ok)
    tl.store(PML + base * 2 + 1, l_i, mask=row_ok)


@triton.jit
def _verify_combine_kernel(
    PO,
    PML,
    O,
    so_t,
    so_h,
    QSTART,
    MAXQ: tl.constexpr,
    GROUP: tl.constexpr,
    KVH: tl.constexpr,
    HD: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    s = pid // MAXQ
    qi = pid % MAXQ
    q0 = tl.load(QSTART + s)
    if qi < tl.load(QSTART + s + 1) - q0:
        kvh = h // GROUP
        m = qi * GROUP + h % GROUP
        sp = tl.arange(0, SPLITS)
        base = ((s * KVH + kvh) * SPLITS + sp) * BLOCK_M + m
        mx = tl.load(PML + base * 2)
        ln = tl.load(PML + base * 2 + 1)
        top = tl.max(mx, 0)
        w = tl.where(mx == float("-inf"), 0.0, tl.exp(mx - top))
        d = tl.arange(0, HD)
        acc = tl.load(PO + base[:, None].to(tl.int64) * HD + d[None, :])
        out = tl.sum(w[:, None] * acc, 0) / tl.sum(w * ln, 0)
        tl.store(
            O + (q0 + qi).to(tl.int64) * so_t + h * so_h + d, out.to(O.dtype.element_ty)
        )


def num_splits(num_seqs: int, num_kv_heads: int, row_blocks: int = 1) -> int:
    """Power-of-two split count filling ~TARGET_PROGRAMS programs; static
    shapes only (graph-safe)."""
    n = max(1, TARGET_PROGRAMS // max(1, num_seqs * num_kv_heads * row_blocks))
    n = 1 << (n.bit_length() - 1)
    return max(1, min(MAX_SPLITS_ONE if num_seqs == 1 else MAX_SPLITS, n))


def verify_attention(
    q: torch.Tensor,  # [tokens, heads, head_dim]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,  # [seqs, max_blocks]
    context_lens: torch.Tensor,  # [seqs] keys per sequence, its query rows included
    cu_seqlens_q: torch.Tensor,  # [seqs + 1]
    max_seqlen_q: int,
    scale: float,
    out: torch.Tensor | None = None,
    tree_mask: (
        torch.Tensor | None
    ) = None,  # [seqs, >= max_seqlen_q] int64: each row's ancestor-or-self rows
) -> torch.Tensor:
    _, heads, hd = q.shape
    seqs = context_lens.shape[0]
    kvh = k_cache.shape[1]
    group = heads // kvh
    bs, x = k_cache.shape[3], k_cache.shape[4]
    block_m = max(16, triton.next_power_of_2(max_seqlen_q * group))
    bm = min(block_m, ROW_BLOCK)
    splits = num_splits(seqs, kvh, block_m // bm)
    po = torch.empty(
        seqs * kvh * splits * block_m, hd, dtype=torch.float32, device=q.device
    )
    pml = torch.empty(
        seqs * kvh * splits * block_m, 2, dtype=torch.float32, device=q.device
    )
    _verify_split_kernel[(seqs * (block_m // bm), kvh, splits)](
        q, q.stride(0), q.stride(1), k_cache, v_cache, block_tables, block_tables.stride(0),
        context_lens, cu_seqlens_q, po, pml, scale,
        tree_mask if tree_mask is not None else context_lens, tree_mask.stride(0) if tree_mask is not None else 0,
        GROUP=group, KVH=kvh, HD=hd, BS=bs, X=x, SPLITS=splits, BLOCK_M=block_m, BLOCK_N=BLOCK_N,
        MIN_KEYS=MIN_KEYS_PER_SPLIT, TREE=tree_mask is not None, BM=bm, num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )  # fmt: skip
    if out is None:
        out = torch.empty_like(q)
    _verify_combine_kernel[(seqs * max_seqlen_q, heads)](
        po, pml, out, out.stride(0), out.stride(1), cu_seqlens_q, MAXQ=max_seqlen_q,
        GROUP=group, KVH=kvh, HD=hd, SPLITS=splits, BLOCK_M=block_m, num_warps=4,
    )  # fmt: skip
    return out
