# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""An EAGLE-4 draft round as one CUDA graph.

``EagleProposer.propose`` drafts a round as a Python loop of model forwards,
each rebuilding attention metadata, which is a large share of a speculative
round. ``Eagle4FastDraft`` runs the same round -- the extension over the
verified rows, then the drafts -- from fused Triton kernels and the draft's
own weights, entirely on the device, and replays it from a graph per batch
size and verify width: a chain for one request (``round``) or a batch
(``batch_round``), or an EAGLE-2 dynamic tree a request (``dyn_tree_round``).
It reads and writes the draft's KV in ATOM's pool, at the slots and RoPE
positions ``EagleProposer`` uses (a row's key at its target row's slot,
rotated to that position + 1), so either path can draft any round of a
sequence.
"""

from typing import ClassVar

import torch
import triton
import triton.language as tl

WINDOW = 4096


SPLITS = 8  # a split-K attention's default split count (the layer picks attn_splits)
MULTI_ROWS = (
    2  # rows a request from which the layer's attention reads its keys once for them
)


def attn_splits(reqs: int, tree: bool = False) -> int:
    """The draft attention's split count for ``reqs`` key sets (requests, or
    rows read one at a time). MI355X, graph-timed with the caches flushed
    (the round streams the target's weights between the draft's layers):
    1 / 4 / 16 / 32 requests of 4 rows at 4K / 4K / 4K / 2K keys: 33 / 96 /
    326 / 349 us at 16 / 4 / 16 / 16 splits; the keys are the cost (40 heads
    of 128), 16 splits keep enough of them in flight from 16 requests on.
    Only the tree beam takes 16 at batch: with 16 the chain round at 64
    requests (extension and single-row steps) faulted late in SPEED-Bench
    runs (4 of 4; 0 of 3 at 4 splits), cause not found yet."""
    return 16 if reqs == 1 or (tree and reqs >= 16) else 4


@triton.jit
def _draft_rows_kernel(
    POS,
    LAST,
    TABLE,
    sbt,
    SP,
    ROPE,
    SLOT,
    B,
    D: tl.constexpr,
    DP: tl.constexpr,
    BS: tl.constexpr,
):
    """Request b's draft depth i (i = 1 .. D): slot position base + i (base =
    its last accepted row's), RoPE'd at base + i + 1, at its cache slot;
    stored depth-major ([D, B]), so a depth's rows are contiguous."""
    b = tl.program_id(0)
    i = tl.arange(0, DP)
    ok = i < D
    sp = tl.load(POS + tl.load(LAST + b)).to(tl.int64) + 1 + i
    tl.store(SP + i * B + b, sp, mask=ok)
    tl.store(ROPE + i * B + b, sp + 1, mask=ok)
    blk = tl.load(TABLE + b * sbt + sp // BS, mask=ok, other=0).to(tl.int64)
    tl.store(SLOT + i * B + b, blk * BS + sp % BS, mask=ok)


@triton.jit
def _rms_kernel(
    X,
    sx,
    R,
    sr,
    RI,
    W,
    OUT,
    so,
    RES,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_RES: tl.constexpr,
    HAS_RI: tl.constexpr,
    eps,
):
    """OUT[row] = rms_norm(X[row] (+ R[row], or R[RI[row]], rounded to bf16 into RES)) * W."""
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    m = c < N
    x = tl.load(X + row * sx + c, mask=m, other=0.0).to(tl.float32)
    if HAS_RES:
        if HAS_RI:
            rrow = tl.load(RI + row)
        else:
            rrow = row
        x = (x + tl.load(R + rrow * sr + c, mask=m, other=0.0).to(tl.float32)).to(
            tl.bfloat16
        )
        tl.store(RES + row * N + c, x, mask=m)
        x = x.to(tl.float32)
    r = 1.0 / tl.sqrt(tl.sum(x * x, 0) / N + eps)
    y = (x * r).to(tl.bfloat16).to(tl.float32) * tl.load(W + c, mask=m, other=0.0).to(
        tl.float32
    )
    tl.store(OUT + row * so + c, y.to(tl.bfloat16), mask=m)


@triton.jit
def _input_norms_kernel(
    E,
    TOK,
    Hd,
    sh,
    HI,
    W1,
    W2,
    OUT,
    Z,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_HI: tl.constexpr,
    Z0: tl.constexpr,
    Z1: tl.constexpr,
    ZROWS: tl.constexpr,
    eps,
):
    """The layer input [n, 2N]: rms(embed[tok]) * W1 | rms(h) * W2 (h's row
    HI[row] with HAS_HI), one program a (row, half). With Z0 > 0 it also
    zeroes the row's split-K accumulators in the arena Z: [ZROWS, Z0] (half 0)
    and two [ZROWS, Z1] after it (half 1)."""
    row = tl.program_id(0)
    half = tl.program_id(1)
    c = tl.arange(0, BLOCK)
    m = c < N
    if half == 0:
        x = tl.load(E + tl.load(TOK + row).to(tl.int64) * N + c, mask=m, other=0.0).to(
            tl.float32
        )
        w = tl.load(W1 + c, mask=m, other=0.0).to(tl.float32)
        if Z0 > 0:
            for z in tl.static_range(0, Z0, BLOCK):
                tl.store(
                    Z + row * Z0 + z + c,
                    tl.zeros((BLOCK,), tl.float32),
                    mask=z + c < Z0,
                )
    else:
        if HAS_HI:
            hrow = tl.load(HI + row)
        else:
            hrow = row
        x = tl.load(Hd + hrow * sh + c, mask=m, other=0.0).to(tl.float32)
        w = tl.load(W2 + c, mask=m, other=0.0).to(tl.float32)
        if Z0 > 0:
            for i in tl.static_range(2):
                for z in tl.static_range(0, Z1, BLOCK):
                    at = ZROWS * Z0 + i * ZROWS * Z1 + row * Z1 + z
                    tl.store(
                        Z + at + c, tl.zeros((BLOCK,), tl.float32), mask=z + c < Z1
                    )
    r = 1.0 / tl.sqrt(tl.sum(x * x, 0) / N + eps)
    y = (x * r).to(tl.bfloat16).to(tl.float32) * w
    tl.store(OUT + row * 2 * N + half * N + c, y.to(tl.bfloat16), mask=m)


def rms_into(x, w, out, eps, res_in=None, res_out=None, res_rows=None):
    """``_rms_kernel``; ``res_rows``: the residual's row for each row (else the same row)."""
    n = x.shape[0]
    _rms_kernel[(n,)](
        x, x.stride(0), res_in if res_in is not None else x,
        res_in.stride(0) if res_in is not None else 0, res_rows if res_rows is not None else x,
        w, out, out.stride(0), res_out if res_out is not None else out, N=x.shape[1],
        BLOCK=triton.next_power_of_2(x.shape[1]), HAS_RES=res_in is not None,
        HAS_RI=res_rows is not None, eps=eps, num_warps=8,
    )  # fmt: skip
    return out


@triton.jit
def _rope_kv_kernel(
    QKV,
    POS,
    COS,
    SIN,
    QO,
    KC,
    VC,
    SLOTS,
    H: tl.constexpr,
    HD: tl.constexpr,
    BS: tl.constexpr,
    X: tl.constexpr,
    SK=None,
    SV=None,
    SIDX=None,
    TREE: tl.constexpr = False,
):
    """neox RoPE (full head) of one (row, head) at POS[row]; q -> QO
    [n, H, HD]; k, v -> the paged cache at SLOTS[row] (skipped below 0), or
    with ``TREE`` -> the draft tree's scratch entry SIDX[row]."""
    r = tl.program_id(0)
    h = tl.program_id(1)
    half: tl.constexpr = HD // 2
    d = tl.arange(0, half)
    pos = tl.load(POS + r)
    c = tl.load(COS + pos * HD + d)
    s = tl.load(SIN + pos * HD + d)
    base = QKV + r * (3 * H * HD) + h * HD
    q1 = tl.load(base + d).to(tl.float32)
    q2 = tl.load(base + half + d).to(tl.float32)
    qo = QO + (r * H + h) * HD
    tl.store(qo + d, (q1 * c - q2 * s).to(tl.bfloat16))
    tl.store(qo + half + d, (q2 * c + q1 * s).to(tl.bfloat16))
    if TREE:
        k1 = tl.load(base + H * HD + d).to(tl.float32)
        k2 = tl.load(base + H * HD + half + d).to(tl.float32)
        out = (tl.load(SIDX + r).to(tl.int64) * H + h) * HD
        tl.store(SK + out + d, (k1 * c - k2 * s).to(tl.bfloat16))
        tl.store(SK + out + half + d, (k2 * c + k1 * s).to(tl.bfloat16))
        dd = tl.arange(0, HD)
        tl.store(SV + out + dd, tl.load(base + 2 * H * HD + dd))
        return
    slot = tl.load(SLOTS + r).to(tl.int64)
    if slot >= 0:
        k1 = tl.load(base + H * HD + d).to(tl.float32)
        k2 = tl.load(base + H * HD + half + d).to(tl.float32)
        blk = slot // BS
        t = slot % BS
        hb = blk * H + h
        ka = (k1 * c - k2 * s).to(tl.bfloat16)
        kb = (k2 * c + k1 * s).to(tl.bfloat16)
        tl.store(KC + ((hb * (HD // X) + d // X) * BS + t) * X + d % X, ka)
        tl.store(
            KC + ((hb * (HD // X) + (d + half) // X) * BS + t) * X + (d + half) % X, kb
        )
        dd = tl.arange(0, HD)
        v = tl.load(base + 2 * H * HD + dd)
        tl.store(VC + ((hb * (BS // X) + t // X) * HD + dd) * X + t % X, v)


@triton.jit
def _attn_split_kernel(
    Q,
    KC,
    VC,
    SPOS,
    FIRST,
    sf,
    BT,
    sbt,
    PO,
    PML,
    scale,
    H: tl.constexpr,
    HD: tl.constexpr,
    BS: tl.constexpr,
    X: tl.constexpr,
    WIN: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    RPR: tl.constexpr,
    NSLOT: tl.constexpr,
):
    """Split-K partial of one (head, row, split): the row's keys at slot
    positions [f - WIN, s] (s = SPOS[row], f = FIRST[b * sf] of the row's
    request b = row // RPR, whose block table is BT + b * sbt), chunk
    ``split`` of the SPLITS BLOCK-rounded chunks they divide into. The
    window is frozen, as EAGLE-4 is trained: for the rows the round extends
    over, f is the first of them (the WIN positions before, then the earlier
    rows); for draft rows, f is the first draft row (the WIN positions
    through the last accepted one, then the row's draft ancestors).

    Tiles are BLOCK-aligned so a tile's keys sit in whole cache blocks, and
    both caches are read in their X-element runs (16 B): K [block, head,
    HD / X, BS, X] a key's X dims at a time, V [block, head, BS / X, HD, X] X
    keys' values of a dim at a time."""
    h = tl.program_id(0)
    r = tl.program_id(1)
    sp = tl.program_id(2)
    d = tl.arange(0, HD)
    s_pos = tl.load(SPOS + r)
    req = r // RPR
    BT = BT + req * sbt
    lo = tl.maximum(tl.load(FIRST + req * sf) - WIN, 0)
    ch = tl.cdiv(tl.cdiv(s_pos + 1 - lo, SPLITS), BLOCK) * BLOCK
    st0 = lo + sp * ch
    end = tl.minimum(st0 + ch, s_pos + 1)
    q = tl.load(Q + (r * H + h) * HD + d).to(tl.float32) * scale
    m_i = tl.full((), float("-inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((HD,), tl.float32)
    kk = tl.arange(0, BLOCK)
    g = tl.arange(0, HD // X)
    e = tl.arange(0, X)
    vg = tl.arange(0, BLOCK // X)
    for st in range(st0 // BLOCK * BLOCK, end, BLOCK):
        kp = st + kk
        ok = (kp >= st0) & (kp < end)
        blk = tl.load(BT + kp // BS, mask=kp < end, other=0).to(tl.int64)
        hb = blk * H + h
        koff = (
            (hb[:, None, None] * (HD // X) + g[None, :, None]) * BS
            + (kp % BS)[:, None, None]
        ) * X + e[None, None, :]
        k = tl.reshape(
            tl.load(KC + koff, mask=(kp < end)[:, None, None], other=0.0), (BLOCK, HD)
        ).to(tl.float32)
        sc = tl.where(ok, tl.sum(k * q[None, :], 1), float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 0))
        alpha = tl.exp(m_i - m_new)
        pr = tl.where(ok, tl.exp(sc - m_new), 0.0)
        kv0 = st + vg * X
        vblk = tl.load(BT + kv0 // BS, mask=kv0 < end, other=0).to(tl.int64)
        vb = vblk * H + h
        voff = (
            (vb[:, None, None] * (BS // X) + ((kv0 % BS) // X)[:, None, None]) * HD
            + d[None, :, None]
        ) * X + e[None, None, :]
        v3 = tl.load(VC + voff, mask=(kv0 < end)[:, None, None], other=0.0)
        v = tl.reshape(tl.permute(v3, (0, 2, 1)), (BLOCK, HD)).to(tl.float32)
        acc = acc * alpha + tl.sum(pr[:, None] * v, 0)
        l_i = l_i * alpha + tl.sum(pr, 0)
        m_i = m_new
    slot = (r * H + h) * NSLOT + sp
    tl.store(PO + slot * HD + d, acc)
    tl.store(PML + slot * 2, m_i)
    tl.store(PML + slot * 2 + 1, l_i)


@triton.jit
def _attn_split_rows_kernel(
    Q,
    KC,
    VC,
    SPOS,
    FIRST,
    sf,
    BT,
    sbt,
    PO,
    PML,
    scale,
    H: tl.constexpr,
    HD: tl.constexpr,
    BS: tl.constexpr,
    X: tl.constexpr,
    WIN: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    RPR: tl.constexpr,
    RP: tl.constexpr,
    NSLOT: tl.constexpr,
    FP32: tl.constexpr = True,
    SK=None,
    SV=None,
    TMASK=None,
    E: tl.constexpr = 0,
    EP: tl.constexpr = 16,
):
    """``_attn_split_kernel`` for all RPR rows of a request at once (one
    program a head, request and split): each key block is read once for the
    request's rows, not once a row. Same partials, same slots."""
    h = tl.program_id(0)
    req = tl.program_id(1)
    sp = tl.program_id(2)
    d = tl.arange(0, HD)
    rr = tl.arange(0, RP)
    rok = rr < RPR
    r = req * RPR + rr
    s_pos = tl.load(SPOS + r, mask=rok, other=-1)
    BT = BT + req * sbt
    lo = tl.maximum(tl.load(FIRST + req * sf) - WIN, 0)
    hi = tl.max(s_pos, 0) + 1
    ch = tl.cdiv(tl.cdiv(hi - lo, SPLITS), BLOCK) * BLOCK
    st0 = lo + sp * ch
    end = tl.minimum(st0 + ch, hi)
    # fp32 throughout, as the single-row kernel: the rows' drafts match it
    q = tl.load(
        Q + (r[:, None] * H + h) * HD + d[None, :], mask=rok[:, None], other=0.0
    ).to(tl.float32)
    m_i = tl.full((RP,), float("-inf"), tl.float32)
    l_i = tl.zeros((RP,), tl.float32)
    acc = tl.zeros((RP, HD), tl.float32)
    kk = tl.arange(0, BLOCK)
    g = tl.arange(0, HD // X)
    e = tl.arange(0, X)
    vg = tl.arange(0, BLOCK // X)
    for st in range(st0 // BLOCK * BLOCK, end, BLOCK):
        kp = st + kk
        kin = kp < end
        blk = tl.load(BT + kp // BS, mask=kin, other=0).to(tl.int64)
        hb = blk * H + h
        koff = (
            (hb[:, None, None] * (HD // X) + g[None, :, None]) * BS
            + (kp % BS)[:, None, None]
        ) * X + e[None, None, :]
        k = tl.reshape(
            tl.load(KC + koff, mask=kin[:, None, None], other=0.0), (BLOCK, HD)
        ).to(tl.float32)
        if FP32:
            sc = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        else:
            sc = tl.dot(q.to(tl.bfloat16), tl.trans(k.to(tl.bfloat16))) * scale
        ok = (
            kin[None, :]
            & (kp[None, :] >= st0)
            & (kp[None, :] <= s_pos[:, None])
            & rok[:, None]
        )
        sc = tl.where(ok, sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        pr = tl.where(ok, tl.exp(sc - m_safe[:, None]), 0.0)
        kv0 = st + vg * X
        vblk = tl.load(BT + kv0 // BS, mask=kv0 < end, other=0).to(tl.int64)
        vb = vblk * H + h
        voff = (
            (vb[:, None, None] * (BS // X) + ((kv0 % BS) // X)[:, None, None]) * HD
            + d[None, :, None]
        ) * X + e[None, None, :]
        v3 = tl.load(VC + voff, mask=(kv0 < end)[:, None, None], other=0.0)
        v = tl.reshape(tl.permute(v3, (0, 2, 1)), (BLOCK, HD)).to(tl.float32)
        if FP32:
            acc = acc * alpha[:, None] + tl.dot(pr, v, input_precision="ieee")
        else:
            acc = acc * alpha[:, None] + tl.dot(pr.to(tl.bfloat16), v.to(tl.bfloat16))
        l_i = l_i * alpha + tl.sum(pr, 1)
        m_i = m_new
    if E > 0:  # noqa: SIM102 -- E is a constexpr: the block compiles out
        if sp == 0:
            # the rows' draft-tree ancestors (scratch [B * E, H, HD], each row's
            # ancestor-or-self mask over its request's E entries), split 0's share
            te = tl.arange(0, EP)
            tm = tl.load(TMASK + r, mask=rok, other=0)
            t_seen = (
                (te[None, :] < E)
                & (((tm[:, None] >> te[None, :].to(tl.int64)) & 1) != 0)
                & rok[:, None]
            )
            t_off = (
                (req * E + te).to(tl.int64)[:, None] * (H * HD) + h * HD + d[None, :]
            )
            t_k = tl.load(SK + t_off, mask=(te < E)[:, None], other=0.0)
            t_sc = tl.dot(q.to(tl.bfloat16), tl.trans(t_k)).to(tl.float32) * scale
            t_sc = tl.where(t_seen, t_sc, float("-inf"))
            t_m = tl.maximum(m_i, tl.max(t_sc, 1))
            t_ms = tl.where(t_m == float("-inf"), 0.0, t_m)
            t_alpha = tl.exp(m_i - t_ms)
            t_p = tl.where(t_seen, tl.exp(t_sc - t_ms[:, None]), 0.0)
            t_v = tl.load(SV + t_off, mask=(te < E)[:, None], other=0.0)
            acc = acc * t_alpha[:, None] + tl.dot(t_p.to(tl.bfloat16), t_v)
            l_i = l_i * t_alpha + tl.sum(t_p, 1)
            m_i = t_m
    slot = (r * H + h) * NSLOT + sp
    tl.store(PO + slot[:, None] * HD + d[None, :], acc, mask=rok[:, None])
    tl.store(PML + slot * 2, m_i, mask=rok)
    tl.store(PML + slot * 2 + 1, l_i, mask=rok)


@triton.jit
def _attn_tree_kernel(
    Q,
    SK,
    SV,
    MASK,
    PO,
    PML,
    scale,
    H: tl.constexpr,
    HD: tl.constexpr,
    E: tl.constexpr,
    EP: tl.constexpr,
    RPR: tl.constexpr,
    NSLOT: tl.constexpr,
    SP: tl.constexpr,
):
    """The partial (as split SP) of one (head, row) over its draft-tree
    ancestors: the request's scratch keys [E, H, HD] set in the row's mask."""
    h = tl.program_id(0)
    r = tl.program_id(1)
    req = r // RPR
    d = tl.arange(0, HD)
    e = tl.arange(0, EP)
    mask = tl.load(MASK + r)
    seen = (e < E) & (((mask >> e.to(tl.int64)) & 1) != 0)
    base = (req * E + e).to(tl.int64)[:, None] * (H * HD) + h * HD + d[None, :]
    q = tl.load(Q + (r * H + h) * HD + d).to(tl.float32) * scale
    k = tl.load(SK + base, mask=seen[:, None], other=0.0).to(tl.float32)
    sc = tl.where(seen, tl.sum(k * q[None, :], 1), float("-inf"))
    m = tl.max(sc, 0)
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    p = tl.where(seen, tl.exp(sc - m_safe), 0.0)
    v = tl.load(SV + base, mask=seen[:, None], other=0.0).to(tl.float32)
    slot = (r * H + h) * NSLOT + SP
    tl.store(PO + slot * HD + d, tl.sum(p[:, None] * v, 0))
    tl.store(PML + slot * 2, m)
    tl.store(PML + slot * 2 + 1, tl.sum(p, 0))


@triton.jit
def _attn_combine_kernel(
    PO, PML, OUT, H: tl.constexpr, HD: tl.constexpr, SPLITS: tl.constexpr
):
    h = tl.program_id(0)
    r = tl.program_id(1)
    d = tl.arange(0, HD)
    sp = tl.arange(0, triton.next_power_of_2(SPLITS))
    ok = sp < SPLITS
    slot = (r * H + h) * SPLITS + sp
    m = tl.load(PML + slot * 2, mask=ok, other=float("-inf"))
    ln = tl.load(PML + slot * 2 + 1, mask=ok, other=0.0)
    w = tl.where(m == float("-inf"), 0.0, tl.exp(m - tl.max(m, 0)))
    acc = tl.load(PO + slot[:, None] * HD + d[None, :], mask=ok[:, None], other=0.0)
    tl.store(
        OUT + r * H * HD + h * HD + d,
        (tl.sum(w[:, None] * acc, 0) / tl.sum(w * ln, 0)).to(tl.bfloat16),
    )


@triton.jit
def _silu_mul_kernel(GU, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(GU + row * 2 * N + c).to(tl.float32)
    u = tl.load(GU + row * 2 * N + N + c).to(tl.float32)
    a = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + row * N + c, (a * u).to(tl.bfloat16))


# ---- the candidate head: the low-rank top-C by a block-maximum threshold ----


@triton.jit
def _ord_key(score, idx):
    """fp32 score, int64 index -> an int64 key ordered by score, ties to the
    smaller index."""
    b = score.to(tl.int32, bitcast=True)
    o = b ^ ((b >> 31) & 0x7FFFFFFF)
    return (o.to(tl.int64) << 32) | (0xFFFFFFFF - idx)


@triton.jit
def _head_scores(HL, E, SC, BM, V, R: tl.constexpr, BV: tl.constexpr):
    """Scores of BV vocabulary rows (E [V, R] . hl) -> SC, the block maximum -> BM."""
    pid = tl.program_id(0)
    rv = pid * BV + tl.arange(0, BV)
    ok = rv < V
    hl = tl.load(HL + tl.arange(0, R)).to(tl.float32)
    e = tl.load(
        E + rv[:, None] * R + tl.arange(0, R)[None, :], mask=ok[:, None], other=0.0
    ).to(tl.float32)
    sc = tl.where(ok, tl.sum(e * hl[None, :], 1), float("-inf"))
    tl.store(SC + rv, sc, mask=ok)
    tl.store(BM + pid, tl.max(sc, 0))


@triton.jit
def _head_thresh(BM, T_, CNT, NB, NBP: tl.constexpr, C: tl.constexpr):
    """The C-th largest block maximum: no top-C score is below it."""
    r = tl.arange(0, NBP)
    bm = tl.load(BM + r, mask=r < NB, other=float("-inf"))
    tl.store(T_, tl.min(tl.topk(bm, C), 0))
    tl.store(CNT, 0)


@triton.jit
def _head_compact(SC, T_, CNT, KEYS, V, BV: tl.constexpr, CAP: tl.constexpr):
    """The keys of every score >= the threshold (at most CAP)."""
    pid = tl.program_id(0)
    rv = pid * BV + tl.arange(0, BV)
    ok = rv < V
    sc = tl.load(SC + rv, mask=ok, other=float("-inf"))
    sel = ok & (sc >= tl.load(T_))
    n = tl.sum(sel.to(tl.int32), 0)
    if n > 0:
        b = tl.atomic_add(CNT, n)
        slot = b + tl.cumsum(sel.to(tl.int32), 0) - 1
        tl.store(KEYS + slot, _ord_key(sc, rv.to(tl.int64)), mask=sel & (slot < CAP))


@triton.jit
def _head_top(
    KEYS, CNT, OUT, CAP: tl.constexpr, C: tl.constexpr, SMALL: tl.constexpr = 512
):
    """The best C of the compacted keys; a SMALL-wide top-k when they fit
    (the usual case: the threshold keeps a few times C)."""
    cnt = tl.minimum(tl.load(CNT), CAP)
    if cnt <= SMALL:
        rs = tl.arange(0, SMALL)
        top = tl.topk(tl.load(KEYS + rs, mask=rs < cnt, other=-9223372036854775807), C)
    else:
        r = tl.arange(0, CAP)
        top = tl.topk(tl.load(KEYS + r, mask=r < cnt, other=-9223372036854775807), C)
    tl.store(OUT + tl.arange(0, C), top)


@triton.jit
def _head_exact(KEYS, H_, LM, LOG, CAND, D: tl.constexpr, DP: tl.constexpr):
    """Candidate c's exact logit lm_head[idx] . h (fp32)."""
    c = tl.program_id(0)
    idx = 0xFFFFFFFF - (tl.load(KEYS + c) & 0xFFFFFFFF)
    d = tl.arange(0, DP)
    ok = d < D
    h = tl.load(H_ + d, mask=ok, other=0.0).to(tl.float32)
    w = tl.load(LM + idx * D + d, mask=ok, other=0.0).to(tl.float32)
    tl.store(LOG + c, tl.sum(h * w, 0))
    tl.store(CAND + c, idx)


@triton.jit
def _head_pick(LOG, CAND, TOK, C: tl.constexpr):
    """The best candidate (ties to the smaller token id) -> TOK."""
    c = tl.arange(0, C)
    key = _ord_key(tl.load(LOG + c), tl.load(CAND + c))
    k1 = tl.max(key, 0)
    tl.store(TOK, 0xFFFFFFFF - (k1 & 0xFFFFFFFF))


@triton.jit
def _bhead_scores(
    HL, E, SC, BM, V, NB, N, NP: tl.constexpr, R: tl.constexpr, BV: tl.constexpr
):
    """``_head_scores`` for N rows at once: one read of BV vocabulary rows of
    E for every row (SC [N, V], the block maxima BM [N, NB])."""
    pid = tl.program_id(0)
    rv = pid * BV + tl.arange(0, BV)
    ok = rv < V
    rn = tl.arange(0, NP)
    hl = tl.load(
        HL + rn[:, None] * R + tl.arange(0, R)[None, :],
        mask=(rn < N)[:, None],
        other=0.0,
    )
    e = tl.load(
        E + rv[:, None] * R + tl.arange(0, R)[None, :], mask=ok[:, None], other=0.0
    )
    sc = tl.dot(hl.to(tl.bfloat16), tl.trans(e.to(tl.bfloat16)))  # [NP, BV], fp32
    sc = tl.where(ok[None, :], sc, float("-inf"))
    m = (rn < N)[:, None] & ok[None, :]
    tl.store(SC + rn[:, None].to(tl.int64) * V + rv[None, :], sc, mask=m)
    tl.store(BM + rn * NB + pid, tl.max(sc, 1), mask=rn < N)


@triton.jit
def _bhead_thresh(BM, T_, CNT, NB, NBP: tl.constexpr, C: tl.constexpr):
    """``_head_thresh`` of row r."""
    r = tl.program_id(0)
    nb = tl.arange(0, NBP)
    bm = tl.load(BM + r * NB + nb, mask=nb < NB, other=float("-inf"))
    tl.store(T_ + r, tl.min(tl.topk(bm, C), 0))
    tl.store(CNT + r, 0)


@triton.jit
def _bhead_compact(SC, T_, CNT, KEYS, V, BV: tl.constexpr, CAP: tl.constexpr):
    """``_head_compact`` of row r over BV scores a program, branch-free: one
    vector load, one atomic (a per-block test serialized on its latency)."""
    r = tl.program_id(0)
    rv = tl.program_id(1) * BV + tl.arange(0, BV)
    ok = rv < V
    th = tl.load(T_ + r)
    sc = tl.load(SC + r.to(tl.int64) * V + rv, mask=ok, other=float("-inf"))
    sel = ok & (sc >= th)
    n = tl.sum(sel.to(tl.int32), 0)
    b = tl.atomic_add(CNT + r, n)
    slot = b + tl.cumsum(sel.to(tl.int32), 0) - 1
    tl.store(
        KEYS + r * CAP + slot, _ord_key(sc, rv.to(tl.int64)), mask=sel & (slot < CAP)
    )


@triton.jit
def _bhead_top(
    KEYS, CNT, TOP, CAP: tl.constexpr, C: tl.constexpr, SMALL: tl.constexpr = 512
):
    """``_head_top`` of row r, as vocabulary ids."""
    r = tl.program_id(0)
    cnt = tl.minimum(tl.load(CNT + r), CAP)
    if cnt <= SMALL:
        ks = tl.arange(0, SMALL)
        top = tl.topk(
            tl.load(KEYS + r * CAP + ks, mask=ks < cnt, other=-9223372036854775807), C
        )
    else:
        k = tl.arange(0, CAP)
        top = tl.topk(
            tl.load(KEYS + r * CAP + k, mask=k < cnt, other=-9223372036854775807), C
        )
    tl.store(TOP + r * C + tl.arange(0, C), 0xFFFFFFFF - (top & 0xFFFFFFFF))


@triton.jit
def _bhead_exact(
    TOP,
    H_,
    LM,
    LOG,
    C: tl.constexpr,
    CB: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    """Row r's exact logits lm_head[cand] . h for CB of its C candidates."""
    r = tl.program_id(0)
    cb = tl.program_id(1)
    c = cb * CB + tl.arange(0, CB)
    idx = tl.load(TOP + r * C + c).to(tl.int64)
    acc = tl.zeros((CB,), tl.float32)
    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        dm = d < D
        h = tl.load(H_ + r * D + d, mask=dm, other=0.0).to(tl.float32)
        w = tl.load(LM + idx[:, None] * D + d[None, :], mask=dm[None, :], other=0.0).to(
            tl.float32
        )
        acc += tl.sum(w * h[None, :], 1)
    tl.store(LOG + r * C + c, acc)


@triton.jit
def _bhead_topk(LOG, TOP, TOK, LP, C: tl.constexpr, K: tl.constexpr):
    """Row r's best K candidates and their log probabilities over the C
    candidates (best first, ties to the smaller token id)."""
    r = tl.program_id(0)
    c = tl.arange(0, C)
    lg = tl.load(LOG + r * C + c)
    cand = tl.load(TOP + r * C + c).to(tl.int64)
    m = tl.max(lg, 0)
    lse = m + tl.log(tl.sum(tl.exp(lg - m), 0))
    key = _ord_key(lg, cand)
    if K == 1:
        best = tl.max(key[None, :], 1)
    else:
        best = tl.topk(key, K)
    tok = 0xFFFFFFFF - (best & 0xFFFFFFFF)
    sel = tok[:, None] == cand[None, :]
    k = tl.arange(0, K)
    tl.store(TOK + r * K + k, tok)
    tl.store(LP + r * K + k, tl.sum(tl.where(sel, lg[None, :], 0.0), 1) - lse)


@triton.jit
def _dyn_prep_kernel(
    SP0, POS, SIDX, RB, HROW, B, S: tl.constexpr, K: tl.constexpr, KP: tl.constexpr
):
    """A dynamic tree round's beam bookkeeping for request b, once: each beam
    row's slot position (the last accepted one, SP0[b] - 1), RoPE position
    at every depth [S - 1, B * K], scratch entry at every depth, and its
    first hidden-state row (the request's)."""
    b = tl.program_id(0)
    kk = tl.arange(0, KP)
    ok = kk < K
    base = tl.load(SP0 + b).to(tl.int64) - 1
    row = b * K + kk
    tl.store(RB + row, base + tl.zeros_like(kk).to(tl.int64), mask=ok)
    tl.store(HROW + row, b + tl.zeros_like(kk).to(tl.int64), mask=ok)
    for j in tl.static_range(1, S):
        tl.store(POS + (j - 1) * B * K + row, base + j + 1, mask=ok)
        tl.store(
            SIDX + (j - 1) * B * K + row,
            (b * (S * K) + (j - 1) * K + kk).to(tl.int64),
            mask=ok,
        )


@triton.jit
def _beam_step_kernel(
    TOK,
    LP,
    BSCORE,
    BNODE,
    BMASK,
    BTOK,
    CTOK,
    CSCORE,
    CPAR,
    CDEP,
    s_c,
    HROW,
    at,
    depth,
    own0,
    K: tl.constexpr,
    KK: tl.constexpr,
    KKP: tl.constexpr,
    LAST: tl.constexpr,
):
    """One request's beam step: its K beam nodes' K children each, scored by
    path log probability, recorded as candidates at ``at``; unless ``LAST``,
    the best K of them become the beam (in place: scores, node ids, ancestor
    masks with their own scratch entry ``own0 + kk``; BTOK their tokens and
    HROW the layer row each continues from)."""
    b = tl.program_id(0)
    c = tl.arange(0, KKP)
    ok = c < KK
    par = c // K
    tok = tl.load(TOK + b * KK + c, mask=ok, other=0)
    score = tl.load(BSCORE + b * K + par, mask=ok, other=0.0) + tl.load(
        LP + b * KK + c, mask=ok, other=0.0
    )
    node = tl.load(BNODE + b * K + par, mask=ok, other=0)
    cb = b * s_c + at + c
    tl.store(CTOK + cb, tok, mask=ok)
    tl.store(CSCORE + cb, score, mask=ok)
    tl.store(CPAR + cb, node, mask=ok)
    tl.store(CDEP + cb, depth + tl.zeros_like(c).to(tl.int64), mask=ok)
    if not LAST:
        key = _ord_key(tl.where(ok, score, float("-inf")), c.to(tl.int64))
        if K == 1:
            top = tl.max(key[None, :], 1)  # tl.topk would reduce to a scalar
        else:
            top = tl.topk(key, K)  # best first, ties to the smaller child
        win = (0xFFFFFFFF - (top & 0xFFFFFFFF)).to(tl.int32)
        kk = tl.arange(0, K)
        wpar = win // K
        # gather the winners' fields by matching child ids
        sel = win[:, None] == c[None, :]
        tl.store(BTOK + b * K + kk, tl.sum(tl.where(sel, tok[None, :], 0), 1))
        new_score = tl.sum(tl.where(sel, score[None, :], 0.0), 1)
        pmask = tl.load(BMASK + b * K + wpar)
        tl.store(BSCORE + b * K + kk, new_score)
        tl.store(BNODE + b * K + kk, at + win.to(tl.int64))
        tl.store(
            BMASK + b * K + kk,
            pmask | (tl.full([K], 1, tl.int64) << (own0 + kk).to(tl.int64)),
        )
        tl.store(HROW + b * K + kk, (b * K + wpar).to(tl.int64))


@triton.jit
def _dyn_select_kernel(
    CSCORE,
    CTOK,
    CPAR,
    CDEP,
    s_c,
    total,
    DRAFTS,
    PARENTS,
    DEPTHS,
    N: tl.constexpr,
    NP: tl.constexpr,
    TP: tl.constexpr,
):
    """Request b's tree from its scored candidates: the best N - 1 (ties to
    the earlier node, so a node's parent always precedes it) in creation
    order as rows 1..N-1 after the root: their tokens, parent rows, depths."""
    b = tl.program_id(0)
    c = tl.arange(0, TP)
    sc = tl.load(CSCORE + b * s_c + c, mask=c < total, other=float("-inf"))
    top = tl.topk(_ord_key(sc, c.to(tl.int64)), NP)  # best first
    i = tl.arange(0, NP)
    sel = tl.where(i < N - 1, 0xFFFFFFFF - (top & 0xFFFFFFFF), 0x7FFFFFFF)
    sel = tl.sort(sel)  # creation order, the unused slots last
    ok = i < N - 1
    tok = tl.load(CTOK + b * s_c + sel, mask=ok, other=0)
    par = tl.load(CPAR + b * s_c + sel, mask=ok, other=-1)
    dep = tl.load(CDEP + b * s_c + sel, mask=ok, other=0)
    hit = (par[:, None] == sel[None, :]) & ok[None, :]
    prow = tl.where(par < 0, 0, tl.sum(tl.where(hit, i[None, :] + 1, 0), 1))
    tl.store(DRAFTS + b * (N - 1) + i, tok, mask=ok)
    tl.store(PARENTS + b * N + 1 + i, prow.to(tl.int64), mask=ok)
    tl.store(DEPTHS + b * N + 1 + i, dep, mask=ok)
    tl.store(PARENTS + b * N, -1)
    tl.store(DEPTHS + b * N, 0)


class CandidateHead:
    """``Eagle4LlamaModel.compute_draft_ids`` for one row in six kernels: the
    low-rank scores and each 256-row block's maximum, the C-th largest block
    maximum (no top-C score falls below it), compaction of the scores at or
    above it, their top-C, the exact logits of the C candidates, the best.
    The same candidate set as ``torch.topk``, without its radix sort."""

    def __init__(self, model, device) -> None:
        self.down = model.efficient_lm_head_down_proj.weight
        self.eff = model.efficient_lm_head.weight
        whole = getattr(model, "_whole_lm_head", None)
        self.lm = whole() if whole is not None else model.lm_head.weight
        self.C = model.num_candidates
        self.V, self.R = self.eff.shape
        self.D = self.lm.shape[1]
        self.blocks = triton.cdiv(self.V, 256)
        if self.blocks > 2048 or self.C > self.blocks:
            raise ValueError("candidate head shape")
        f32 = torch.float32
        self.sc = torch.empty(self.V, dtype=f32, device=device)
        self.bm = torch.empty(self.blocks, dtype=f32, device=device)
        self.th = torch.empty(1, dtype=f32, device=device)
        self.cnt = torch.zeros(1, dtype=torch.int32, device=device)
        self.cap = 4096
        self.small = max(
            512, triton.next_power_of_2(self.C)
        )  # _head_top's narrow top-k
        self.keys = torch.empty(self.cap, dtype=torch.int64, device=device)
        self.top = torch.empty(self.C, dtype=torch.int64, device=device)
        self.lg = torch.empty(self.C, dtype=f32, device=device)
        self.cd = torch.empty(self.C, dtype=torch.int64, device=device)

    def __call__(self, h: torch.Tensor, out: torch.Tensor) -> None:
        """h [1, D] (normed) -> out (1-element int64 view): the drafted token."""
        hl = (h @ self.down.t()).reshape(-1)
        _head_scores[(self.blocks,)](
            hl, self.eff, self.sc, self.bm, self.V, R=self.R, BV=256, num_warps=8
        )
        _head_thresh[(1,)](
            self.bm,
            self.th,
            self.cnt,
            self.blocks,
            NBP=triton.next_power_of_2(self.blocks),
            C=self.C,
            num_warps=8,
        )
        _head_compact[(self.blocks,)](
            self.sc,
            self.th,
            self.cnt,
            self.keys,
            self.V,
            BV=256,
            CAP=self.cap,
            num_warps=4,
        )
        _head_top[(1,)](
            self.keys,
            self.cnt,
            self.top,
            CAP=self.cap,
            C=self.C,
            SMALL=self.small,
            num_warps=8,
        )
        _head_exact[(self.C,)](
            self.top,
            h,
            self.lm,
            self.lg,
            self.cd,
            D=self.D,
            DP=triton.next_power_of_2(self.D),
            num_warps=4,
        )
        _head_pick[(1,)](self.lg, self.cd, out, C=self.C, num_warps=1)


# ---- MXFP4 draft GEMMs (ATOM_EAGLE4_DRAFT_MXFP4) ----------------------------


def quant_mxfp4(w: torch.Tensor):
    """[N, K] -> (e2m1 codes [N, K/2], low nibble the even k; e8m0 scales
    [N, K/32]): scale 2^(floor(log2 amax) - 2), values rounded to the grid."""
    n, k = w.shape
    x = w.float().reshape(n, k // 32, 32)
    e = torch.floor(torch.log2(x.abs().amax(-1, keepdim=True).clamp_min(1e-30))) - 2
    y = x / torch.exp2(e)
    a = y.abs()
    mag = torch.where(
        a < 2,
        torch.round(a * 2) / 2,
        torch.where(a < 4, torch.round(a), torch.round(a / 2) * 2),
    ).clamp(max=6)
    idx = torch.where(
        mag < 2, mag * 2, torch.where(mag < 5, mag + 2, torch.full_like(mag, 7))
    ).to(torch.uint8)
    code = (idx | ((y < 0) & (idx > 0)).to(torch.uint8) << 3).reshape(n, k)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous(), (
        e.squeeze(-1) + 127
    ).clamp(0, 254).to(torch.uint8).contiguous()


@triton.jit
def _mx_gemm_kernel(
    X,
    NROW,
    Q,
    S,
    Y,
    YB,
    N,
    K,
    KS,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT: tl.constexpr,
    BM: tl.constexpr = 16,
):
    """Y[<=BM, N] (+)= X @ W^T: the bf16 rows quantized to MXFP8 here, MXFP4
    weights [N, K], gfx950 scaled MFMA; program (n block, K split): the
    weights read once for all the rows."""
    pid = tl.program_id(0)
    ks = tl.program_id(1)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.arange(0, BM)
    acc = tl.zeros([BM, BLOCK_N], tl.float32)
    for k0 in range(ks * KS, ks * KS + KS, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            X + rm[:, None] * K + rk[None, :], mask=(rm < NROW)[:, None], other=0.0
        ).to(tl.float32)
        x3 = tl.reshape(x, (BM, BLOCK_K // 32, 32))
        e = tl.floor(tl.log2(tl.maximum(tl.max(tl.abs(x3), 2), 1e-30))) - 8
        xq = tl.reshape(
            tl.minimum(tl.maximum(x3 * tl.exp2(-e)[:, :, None], -448.0), 448.0),
            (BM, BLOCK_K),
        ).to(tl.float8e4nv)
        xs = tl.minimum(tl.maximum(e + 127, 0), 254).to(tl.uint8)
        sc = tl.load(
            S
            + rn[:, None] * (K // 32)
            + (k0 // 32 + tl.arange(0, BLOCK_K // 32))[None, :]
        )
        w = tl.load(
            Q + rn[:, None] * (K // 2) + (k0 // 2 + tl.arange(0, BLOCK_K // 2))[None, :]
        )
        acc = tl.dot_scaled(xq, xs, "e4m3", tl.trans(w), sc, "e2m1", acc)
    if SPLIT:
        tl.atomic_add(
            Y + rm[:, None] * N + rn[None, :],
            acc,
            sem="relaxed",
            mask=(rm < NROW)[:, None],
        )
    else:
        tl.store(
            YB + rm[:, None] * N + rn[None, :],
            acc.to(tl.bfloat16),
            mask=(rm < NROW)[:, None],
        )


# the most rows an MXFP4 draft projection takes in one launch: past 32 the bf16
# GEMMs win (MI355X, the five draft projections at 4 / 32 / 64 rows: MXFP4 112 /
# 186 / 255 us, bf16 194 / 216 / 232)
MX_ROWS = 32


class MxLinear:
    """A draft projection in MXFP4 (``ATOM_EAGLE4_DRAFT_MXFP4``): one fused
    launch for up to MX_ROWS rows, split-K with fp32 atomics for the narrow ones."""

    # (N, K) -> (block_n, block_k, split, warps, stages), tuned on MI355X at
    # 4 / 8 rows (graph-timed: qkv 28.0, o 11.6, gate_up 23.8, down 20.1 us)
    CFG: ClassVar[dict[tuple[int, int], tuple[int, int, int, int, int]]] = {
        (15360, 10240): (64, 256, 2, 4, 2), (5120, 5120): (64, 256, 4, 4, 2),
        (32768, 5120): (64, 256, 1, 4, 2), (5120, 16384): (64, 256, 8, 4, 2),
        (5120, 15360): (32, 256, 4, 4, 3),
    }  # fmt: skip

    def __init__(self, w: torch.Tensor, y: torch.Tensor | None = None) -> None:
        self.N, self.K = w.shape
        self.q, self.s = quant_mxfp4(w)
        self.block_n, self.block_k, self.split, self.warps, self.stages = self.CFG.get(
            (self.N, self.K), (32, 256, 1, 4, 2)
        )
        # the split-K accumulator: the caller's (zeroed with others, ``raw``) or its own
        self.y = (
            y
            if y is not None
            else torch.zeros(MX_ROWS, self.N, dtype=torch.float32, device=w.device)
        )
        self.yb = torch.zeros(MX_ROWS, self.N, dtype=torch.bfloat16, device=w.device)

    def __call__(self, x: torch.Tensor, raw: bool = False) -> torch.Tensor:
        """bf16 [n, N]; with ``raw`` a split projection returns its fp32
        accumulator as is, which the caller has zeroed."""
        n = x.shape[0]
        if n > MX_ROWS:
            raise ValueError(f"MxLinear takes at most {MX_ROWS} rows")
        if self.split > 1 and not raw:
            self.y.zero_()
        bm = 16 if n <= 16 else 32
        _mx_gemm_kernel[(self.N // self.block_n, self.split)](
            x, n, self.q, self.s, self.y, self.yb, self.N, self.K, self.K // self.split,
            BLOCK_N=self.block_n, BLOCK_K=self.block_k, SPLIT=self.split > 1, BM=bm,
            num_warps=self.warps, num_stages=self.stages,
        )  # fmt: skip
        if self.split > 1:
            return self.y[:n] if raw else self.y[:n].to(torch.bfloat16)
        return self.yb[:n]


class _Dense:
    def __init__(self, w: torch.Tensor) -> None:
        self.w = w

    def __call__(self, x: torch.Tensor, raw: bool = False) -> torch.Tensor:
        return x @ self.w.t()


def _tp_group():
    """The TP group when there is more than one rank, else None (also
    outside an initialized engine)."""
    from aiter.dist.parallel_state import get_tp_group

    try:
        group = get_tp_group()
    except AssertionError:
        return None
    return group if group.world_size > 1 else None


class Eagle4FastDraft:
    """The draft's weights (borrowed from the loaded ``Eagle4LlamaModel``)
    and the fused round."""

    def __init__(
        self, model, max_positions: int, mtp_k: int, device, mxfp4: bool = False
    ) -> None:
        cfg = model.config
        layer = model.midlayer
        attn = layer.self_attn
        self.model = model
        self.H, self.HD = attn.num_heads, attn.head_dim
        if attn.num_kv_heads != self.H:
            raise ValueError("the fused draft assumes as many KV heads as query heads")
        # at TP > 1 each rank runs its share of the heads and of the MLP (the
        # draft's own sharding), all-reducing after o_proj and down_proj; the
        # candidate head reads the whole LM head
        self.tp = _tp_group()
        self.hidden = cfg.hidden_size
        self.eps = cfg.rms_norm_eps
        self.mtp_k = mtp_k
        self.window = getattr(cfg, "sliding_window", WINDOW) or WINDOW

        def w(lin):
            t = lin.weight
            if t.dtype != torch.bfloat16 or t.dim() != 2 or not t.is_contiguous():
                raise ValueError(
                    f"draft weight {tuple(t.shape)} {t.dtype}: not a plain bf16 matrix"
                )
            return t

        # each draft depth's slot position, RoPE position and cache slot
        self.rows = torch.zeros(3, max(mtp_k, 1), dtype=torch.int64, device=device)
        self.mxfp4 = mxfp4
        if mxfp4:
            # one arena for the split-K accumulators, which the layer's input-norm kernel zeroes
            sizes = [
                w(x).shape[0] for x in (attn.qkv_proj, attn.o_proj, layer.mlp.down_proj)
            ]
            if sizes[1] != sizes[2]:
                raise ValueError(
                    "the arena layout assumes o_proj and down_proj of one width"
                )
            self.arena_widths = (sizes[0], sizes[1])  # zeroed by _input_norms_kernel
            self.arena = torch.zeros(
                MX_ROWS * sum(sizes), dtype=torch.float32, device=device
            )
            ys, at = [], 0
            for n_ in sizes:
                ys.append(self.arena[at : at + MX_ROWS * n_].view(MX_ROWS, n_))
                at += MX_ROWS * n_
            self.qkv = MxLinear(w(attn.qkv_proj), ys[0])
            self.o = MxLinear(w(attn.o_proj), ys[1])
            self.down = MxLinear(w(layer.mlp.down_proj), ys[2])
            self.gu = MxLinear(w(layer.mlp.gate_up_proj))
            self.fc_proj = MxLinear(w(model.fc))
        else:
            self.qkv, self.o = _Dense(w(attn.qkv_proj)), _Dense(w(attn.o_proj))
            self.gu, self.down = _Dense(w(layer.mlp.gate_up_proj)), _Dense(
                w(layer.mlp.down_proj)
            )
            self.fc_proj = _Dense(w(model.fc))
        # bf16 projections on the model's own weights: past 16 rows (batched
        # rounds) the MXFP4 GEMMs stop paying
        self.dense = tuple(
            _Dense(w(lin))
            for lin in (
                attn.qkv_proj,
                attn.o_proj,
                layer.mlp.gate_up_proj,
                layer.mlp.down_proj,
                model.fc,
            )
        )
        self.cand_head = CandidateHead(model, device)
        self._check_head(device)
        self.embed = model.embed_tokens.weight
        self.norms = {
            "inp": layer.input_layernorm.weight,
            "hid": layer.hidden_norm.weight,
            "post": layer.post_attention_layernorm.weight,
            "out": model.norm.weight,
            "fc": [n.weight for n in model.fc_norm],
        }
        theta = float(getattr(cfg, "rope_theta", 10000.0))
        inv = 1.0 / (
            theta
            ** (
                torch.arange(0, self.HD, 2, device=device, dtype=torch.float32)
                / self.HD
            )
        )
        t = torch.arange(max_positions + mtp_k + 2, device=device, dtype=torch.float32)
        fr = torch.outer(t, inv)
        self.cos = torch.cat([fr, fr], -1).cos().contiguous()
        self.sin = torch.cat([fr, fr], -1).sin().contiguous()
        self.scale = self.HD**-0.5
        self.graphs = {}
        self._neg = torch.full((1024,), -1, dtype=torch.int64, device=device)

    # ---- the layer ----------------------------------------------------------

    def fc(self, aux: list[torch.Tensor]) -> torch.Tensor:
        n = aux[0].shape[0]
        x = torch.empty(
            n, len(aux) * self.hidden, dtype=torch.bfloat16, device=aux[0].device
        )
        for i, a in enumerate(aux):
            rms_into(
                a,
                self.norms["fc"][i],
                x[:, i * self.hidden : (i + 1) * self.hidden],
                self.eps,
            )
        return (self.fc_proj if not self.mxfp4 or n <= MX_ROWS else self.dense[4])(x)

    def layer(
        self,
        tokens,
        h,
        rope_pos,
        slots,
        slot_pos,
        first,
        table,
        kc,
        vc,
        rows_per_req=0,
        tree=None,
        h_rows=None,
    ) -> torch.Tensor:
        """One draft layer over ``n`` rows -> the normed output (``norm_output``).
        ``first``: the slot position its attention window is frozen at (the
        first verified row for the extension, the first draft row for drafts;
        ``_attn_split_kernel``). One request (``table`` its block table row),
        or with ``rows_per_req`` consecutive rows a request: ``table`` [B, W]
        and ``first`` [B] (a strided view is fine).

        ``tree`` = (scratch K, scratch V [B * E, H, HD], each row's entry
        [n], each row's ancestor-or-self mask over its request's E entries
        [n] int64): draft-tree rows. Their K / V go to the scratch, not the
        cache (``slots`` unused); they see the cache through ``slot_pos``
        (the last accepted position) and the masked scratch entries.
        ``h_rows``: row i's hidden state is ``h[h_rows[i]]`` (else ``h[i]``)."""
        n, H, HD, hid = tokens.shape[0], self.H, self.HD, self.hidden
        mx = self.mxfp4 and n <= MX_ROWS
        qkv_p, o_p, gu_p, down_p = (
            (self.qkv, self.o, self.gu, self.down) if mx else self.dense[:4]
        )
        if rows_per_req:
            sf, sbt, rpr = first.stride(0), table.stride(0), rows_per_req
        else:
            sf, sbt, rpr = 0, 0, 1
        dev = h.device
        bf = torch.bfloat16
        x = torch.empty(n, 2 * hid, dtype=bf, device=dev)
        # (with MXFP4 it also zeroes the rows' split-K accumulators)
        z0, z1 = self.arena_widths if mx else (0, 0)
        _input_norms_kernel[(n, 2)](
            self.embed, tokens, h, h.stride(0), h_rows if h_rows is not None else tokens, self.norms["inp"],
            self.norms["hid"], x, self.arena if mx else x, N=hid, BLOCK=triton.next_power_of_2(hid),
            HAS_HI=h_rows is not None, Z0=z0, Z1=z1, ZROWS=MX_ROWS, eps=self.eps, num_warps=8,
        )  # fmt: skip
        # the split projections hand their fp32 accumulators on (``raw``)
        qkv = qkv_p(x, raw=True)
        q = torch.empty(n, H, HD, dtype=bf, device=dev)
        bs, xx = kc.shape[3], kc.shape[4]
        together = rpr if rows_per_req else n
        multi = together >= MULTI_ROWS  # a request's rows through the multi-row kernel
        rows_tree = tree is not None and multi  # the tree part inside it
        splits = attn_splits(n // together if multi else n, tree is not None)
        nslot = splits + (tree is not None and not rows_tree)
        if tree is not None:
            sk, sv, sidx, tmask = tree
            _rope_kv_kernel[(n, H)](
                qkv, rope_pos, self.cos, self.sin, q, kc, vc, sidx, H=H, HD=HD, BS=bs, X=xx,
                SK=sk, SV=sv, SIDX=sidx, TREE=True, num_warps=1,
            )  # fmt: skip
        else:
            _rope_kv_kernel[(n, H)](
                qkv,
                rope_pos,
                self.cos,
                self.sin,
                q,
                kc,
                vc,
                slots,
                H=H,
                HD=HD,
                BS=bs,
                X=xx,
                num_warps=1,
            )
        po = torch.empty(n * H * nslot, HD, dtype=torch.float32, device=dev)
        pml = torch.empty(n * H * nslot, 2, dtype=torch.float32, device=dev)
        # a request's rows together (one request's: all n), each key block read
        # once for them: 33 / 47 us for 4 / 16 rows at 1K / 8K keys (the
        # row-at-a-time kernel: 35 / 156); at batch the reads are the cost
        # (40 heads of 128 a key), so from two rows on
        if multi:
            tree_args = {}
            if rows_tree:
                e_ = sk.shape[0] // table.shape[0]
                tree_args = {
                    "SK": sk,
                    "SV": sv,
                    "TMASK": tmask,
                    "E": e_,
                    "EP": max(16, triton.next_power_of_2(e_)),
                }
            _attn_split_rows_kernel[(H, n // together, splits)](
                q, kc, vc, slot_pos, first, sf, table, sbt, po, pml, self.scale, H=H, HD=HD, BS=bs, X=xx,
                WIN=self.window, SPLITS=splits, BLOCK=64, RPR=together,
                RP=max(16, triton.next_power_of_2(together)), NSLOT=nslot, FP32=False, **tree_args,
                # (the AMD software pipeliner miscompiles its loop past two
                # iterations: NaN partials)
                num_warps=4, num_stages=1,
            )  # fmt: skip
        else:
            _attn_split_kernel[(H, n, splits)](
                q, kc, vc, slot_pos, first, sf, table, sbt, po, pml, self.scale, H=H, HD=HD, BS=bs, X=xx,
                WIN=self.window, SPLITS=splits, BLOCK=64, RPR=rpr, NSLOT=nslot, num_warps=4,
            )  # fmt: skip
        if tree is not None and not rows_tree:
            e = sk.shape[0] // table.shape[0]
            _attn_tree_kernel[(H, n)](
                q, sk, sv, tmask, po, pml, self.scale, H=H, HD=HD, E=e, EP=triton.next_power_of_2(e),
                RPR=rpr, NSLOT=nslot, SP=splits, num_warps=4,
            )  # fmt: skip
        o = torch.empty(n, H * HD, dtype=bf, device=dev)
        _attn_combine_kernel[(H, n)](po, pml, o, H=H, HD=HD, SPLITS=nslot, num_warps=4)
        o = self._reduce(o_p(o, raw=True))
        res = torch.empty(n, hid, dtype=bf, device=dev)
        hn = torch.empty_like(res)
        rms_into(
            o, self.norms["post"], hn, self.eps, res_in=h, res_out=res, res_rows=h_rows
        )
        gu = gu_p(hn)
        inter = gu.shape[1] // 2
        act = torch.empty(n, inter, dtype=bf, device=dev)
        _silu_mul_kernel[(n, inter // 1024)](gu, act, N=inter, BLOCK=1024, num_warps=4)
        m = self._reduce(down_p(act, raw=True))
        out = torch.empty_like(res)
        return rms_into(
            m,
            self.norms["out"],
            out,
            self.eps,
            res_in=res,
            res_out=torch.empty_like(res),
        )

    def _no_slots(self, n: int, dev) -> torch.Tensor:
        """``n`` cache slots of -1 (rows that write no paged K / V); made
        up front, not under a graph capture, which would not fill it."""
        if n > self._neg.shape[0]:
            raise ValueError(f"{n} draft-tree rows > {self._neg.shape[0]}")
        return self._neg[:n]

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        """A row-parallel projection's partial sums, summed over the TP group."""
        if self.tp is None:
            return x
        return self.tp.all_reduce(
            x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
        )

    def _check_head(self, device) -> None:
        """The threshold head must pick what ``compute_draft_ids`` picks."""
        g = torch.Generator(device=device).manual_seed(0)
        for _ in range(8):
            h = torch.randn(1, self.hidden, generator=g, device=device).to(
                torch.bfloat16
            )
            want = self.model.compute_draft_ids(
                h, out=torch.empty(1, dtype=torch.int64, device=device)
            )
            got = torch.empty(1, dtype=torch.int64, device=device)
            self.cand_head(h, got)
            if int(got) != int(want):
                raise ValueError(
                    "the threshold candidate head disagrees with compute_draft_ids"
                )

    def batch_head(self, h: torch.Tensor, out: torch.Tensor) -> None:
        """``Eagle4LlamaModel.compute_draft_ids`` for [B, D] rows, on the
        borrowed weights: the low-rank scores' top-C, then the best exact logit."""
        ch = self.cand_head
        cand = self.batch_candidates(h)
        logits = torch.einsum("nh,nch->nc", h.float(), ch.lm[cand].float())
        out.copy_(cand.gather(1, logits.argmax(1, keepdim=True)).squeeze(1))

    def batch_candidates(self, h: torch.Tensor) -> torch.Tensor:
        """Each row's top-C low-rank candidates [n, C] (the threshold head's
        set, for up to 64 rows at once): one pass over the low-rank head
        for every row, then the threshold / compaction / top-C a row."""
        ch = self.cand_head
        n = h.shape[0]
        if n > 64:
            return (
                ((h @ ch.down.t()) @ ch.eff.t())
                .topk(ch.C, dim=-1, sorted=False)
                .indices
            )
        dev = h.device
        hl = (h @ ch.down.t()).contiguous()
        sc = torch.empty(n, ch.V, dtype=torch.float32, device=dev)
        bm = torch.empty(n, ch.blocks, dtype=torch.float32, device=dev)
        _bhead_scores[(ch.blocks,)](
            hl, ch.eff, sc, bm, ch.V, ch.blocks, n, NP=max(16, triton.next_power_of_2(n)), R=ch.R, BV=256, num_warps=8,
        )  # fmt: skip
        th = torch.empty(n, dtype=torch.float32, device=dev)
        cnt = torch.empty(n, dtype=torch.int32, device=dev)
        keys = torch.empty(n, ch.cap, dtype=torch.int64, device=dev)
        top = torch.empty(n, ch.C, dtype=torch.int64, device=dev)
        _bhead_thresh[(n,)](
            bm,
            th,
            cnt,
            ch.blocks,
            NBP=triton.next_power_of_2(ch.blocks),
            C=ch.C,
            num_warps=8,
        )
        _bhead_compact[(n, triton.cdiv(ch.V, 4096))](
            sc, th, cnt, keys, ch.V, BV=4096, CAP=ch.cap, num_warps=8
        )
        _bhead_top[(n,)](
            keys, cnt, top, CAP=ch.cap, C=ch.C, SMALL=ch.small, num_warps=8
        )
        return top

    def batch_head_topk(self, h: torch.Tensor, k: int):
        """``batch_head``'s best ``k`` candidates of each row and their log
        probabilities over the candidate set (SGLang EAGLE-4's softmax over the
        exact candidate logits): (tokens, logp), [n, k] each."""
        ch = self.cand_head
        cand = self.batch_candidates(h)
        n = h.shape[0]
        if n > 64 or k & (k - 1):
            logits = torch.einsum("nh,nch->nc", h.float(), ch.lm[cand].float())
            top = logits.log_softmax(-1).topk(k, dim=-1)
            return cand.gather(1, top.indices), top.values
        dev = h.device
        logits = torch.empty(n, ch.C, dtype=torch.float32, device=dev)
        _bhead_exact[(n, ch.C // 16)](
            cand,
            h.contiguous(),
            ch.lm,
            logits,
            C=ch.C,
            CB=16,
            D=ch.D,
            BD=512,
            num_warps=4,
        )
        tok = torch.empty(n, k, dtype=torch.int64, device=dev)
        lp = torch.empty(n, k, dtype=torch.float32, device=dev)
        _bhead_topk[(n,)](logits, cand, tok, lp, C=ch.C, K=k, num_warps=4)
        return tok, lp

    def head(self, h: torch.Tensor) -> torch.Tensor:
        out = torch.empty(1, dtype=torch.int64, device=h.device)
        self.cand_head(h, out)
        return out

    # ---- the round ---------------------------------------------------------

    def round(self, tokens, positions, slots, last, aux, table, kc, vc, out) -> None:
        """Extend over the ``q`` verified rows (tokens already carry the
        bonus at row ``last``), then the chain: ``out[0, i]`` <- draft i.
        Row j sits at slot ``slots[j]`` = slot position ``positions[j]``,
        rotated to ``positions[j] + 1``; draft i's key goes to the slot of
        position ``positions[last] + i``."""
        bs = kc.shape[3]
        x = self.fc(aux)
        first = positions[:1]
        h = self.layer(tokens, x, positions + 1, slots, positions, first, table, kc, vc)
        h = h.index_select(0, last)
        sp, rope, slot = self._draft_rows(positions, last, table, bs)
        for i in range(self.mtp_k):
            self.cand_head(h, out[0, i : i + 1])
            if i + 1 < self.mtp_k:
                h = self.layer(
                    out[0, i : i + 1],
                    h,
                    rope[i : i + 1],
                    slot[i : i + 1],
                    sp[i : i + 1],
                    sp[:1],
                    table,
                    kc,
                    vc,
                )

    def _draft_rows(self, positions, last, table, bs, rows=None):
        """Every draft depth's (slot position, RoPE position, cache slot),
        [depth] for one request, [depth, B] for ``table`` [B, W] (into ``rows``)."""
        rows = self.rows if rows is None else rows
        sp, rope, slot = rows
        d = sp.shape[0]
        b = 1 if table.dim() == 1 else table.shape[0]
        _draft_rows_kernel[(b,)](
            positions, last, table, table.stride(0) if table.dim() == 2 else 0, sp, rope, slot, b,
            D=d, DP=triton.next_power_of_2(d), BS=bs, num_warps=1,
        )  # fmt: skip
        return sp, rope, slot

    def batch_round(
        self,
        tokens,
        positions,
        slots,
        last,
        aux,
        table,
        kc,
        vc,
        rows,
        drafts,
        depth=None,
    ) -> None:
        """``round`` for B requests of q verified rows each (rows of a request
        consecutive, ``last`` [B] each one's last accepted row, flat; tokens
        already carry the bonus there; ``table`` [B, W]): the extension over
        the B * q rows, then the chain for the B requests at once, ``drafts``
        [mtp_k, B] <- draft i of request b. ``rows``: [3, mtp_k, B] buffers.
        ``depth`` < mtp_k drafts that many (a batch that verifies fewer,
        ``verify_len``) and repeats the last into the rest."""
        bs = kc.shape[3]
        B = table.shape[0]
        q = tokens.shape[0] // B
        x = self.fc(aux)
        h = self.layer(
            tokens,
            x,
            positions + 1,
            slots,
            positions,
            positions[::q],
            table,
            kc,
            vc,
            rows_per_req=q,
        )
        h = h.index_select(0, last)
        sp, rope, slot = self._draft_rows(positions, last, table, bs, rows)
        depth = self.mtp_k if depth is None else depth
        for i in range(depth):
            self.batch_head(h, drafts[i])
            if i + 1 < depth:
                h = self.layer(
                    drafts[i],
                    h,
                    rope[i],
                    slot[i],
                    sp[i],
                    sp[0],
                    table,
                    kc,
                    vc,
                    rows_per_req=1,
                )
        if depth < self.mtp_k:
            drafts[depth:].copy_(
                drafts[depth - 1 : depth].expand(self.mtp_k - depth, -1)
            )

    def dyn_tree_round(
        self,
        tokens,
        positions,
        slots,
        last,
        aux,
        table,
        kc,
        vc,
        rows,
        scratch,
        steps,
        width,
        nodes,
    ):
        """``batch_round``'s extension, then an EAGLE-2 dynamic tree for each
        request (the SGLang EAGLE-4 harness's ``speculative-eagle-topk``):
        a beam of ``width`` nodes grows ``steps`` deep, each step expanding
        every beam node's best ``width`` children and keeping the best
        ``width`` of them by path log probability; the tree is the best
        ``nodes - 1`` of every node scored, in creation order (a node's
        parent always precedes it and scores at least as high).

        ``scratch`` = (K, V [B * steps * width, H, HD]): the beam rows' keys,
        which the deeper rows attend to through their ancestor masks.
        Returns (drafts [B, nodes - 1], parents [B, nodes], depths [B,
        nodes]) in tree row order (row 0 the root)."""
        bs = kc.shape[3]
        B = table.shape[0]
        q = tokens.shape[0] // B
        K, S, N = width, steps, nodes
        dev = tokens.device
        x = self.fc(aux)
        h = self.layer(
            tokens,
            x,
            positions + 1,
            slots,
            positions,
            positions[::q],
            table,
            kc,
            vc,
            rows_per_req=q,
        )
        h = h.index_select(0, last)  # [B, D]
        sp, _, _ = self._draft_rows(positions, last, table, bs, rows)  # [D, B]
        total = K + (S - 1) * K * K
        c_tok = torch.empty(B, total, dtype=torch.int64, device=dev)
        c_score = torch.empty(B, total, dtype=torch.float32, device=dev)
        c_par = torch.empty(B, total, dtype=torch.int64, device=dev)
        c_depth = torch.empty(B, total, dtype=torch.int64, device=dev)
        ar = torch.arange(K, device=dev)
        tok, lp = self.batch_head_topk(h, K)  # the root's children
        c_tok[:, :K], c_score[:, :K] = tok, lp
        c_par[:, :K], c_depth[:, :K] = -1, 1
        # the beam's own buffers, which _beam_step_kernel rewrites in place
        # (clones: at B = 1 an expand is already contiguous and would alias)
        beam_tok, beam_score = tok.reshape(-1), lp.clone()
        beam_node = ar.expand(B, K).clone()
        beam_mask = (1 << ar).expand(B, K).clone()  # its own scratch entry, j * K + kk
        # per beam row: slot position, RoPE position / scratch entry at each
        # depth, and the row of the last layer output it continues from
        rows_base = torch.empty(B * K, dtype=torch.int64, device=dev)
        pos = torch.empty(S - 1, B * K, dtype=torch.int64, device=dev)
        sidx = torch.empty_like(pos)
        hrow = torch.empty(B * K, dtype=torch.int64, device=dev)
        _dyn_prep_kernel[(B,)](
            sp[0],
            pos,
            sidx,
            rows_base,
            hrow,
            B,
            S=S,
            K=K,
            KP=max(2, triton.next_power_of_2(K)),
            num_warps=1,
        )
        btok = torch.empty(B * K, dtype=torch.int64, device=dev)
        beam_h = h
        sk, sv = scratch
        for j in range(1, S):
            # the beam (depth j) through the layer at slot position base + j
            hj = self.layer(
                beam_tok, beam_h, pos[j - 1], None, rows_base, sp[0], table, kc, vc, rows_per_req=K,
                tree=(sk, sv, sidx[j - 1], beam_mask.reshape(-1)), h_rows=hrow,
            )  # fmt: skip
            tok, lp = self.batch_head_topk(hj, K)  # [B * K, K]
            at = K + (j - 1) * K * K
            _beam_step_kernel[(B,)](
                tok, lp, beam_score, beam_node, beam_mask, btok, c_tok, c_score, c_par, c_depth, c_tok.stride(0),
                hrow, at, j + 1, j * K, K=K, KK=K * K, KKP=max(16, triton.next_power_of_2(K * K)),
                LAST=j + 1 == S, num_warps=1,
            )  # fmt: skip
            beam_tok, beam_h = btok, hj
        drafts = torch.empty(B, N - 1, dtype=torch.int64, device=dev)
        parents = torch.empty(B, N, dtype=torch.int64, device=dev)
        depths = torch.empty(B, N, dtype=torch.int64, device=dev)
        _dyn_select_kernel[(B,)](
            c_score, c_tok, c_par, c_depth, c_tok.stride(0), total, drafts, parents, depths, N=N,
            NP=max(2, triton.next_power_of_2(N - 1)), TP=triton.next_power_of_2(total), num_warps=4,
        )  # fmt: skip
        return drafts, parents, depths
