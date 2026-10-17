# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gated DeltaNet layers under a tree verify, on ATOM's state pools.

A tree verify feeds each request N rows (node 0 the root, the last accepted
token; every other node a draft whose parent precedes it). Each node's conv
window and recurrent state continue from its PARENT's, not from the row
before it. The verify does not touch the request's state: it writes, per
node, the replay record of the gated delta rule (``delta`` [V], ``key`` [K],
``decay``: S_node = decay * S_parent + delta key^T) and keeps the node's
pre-conv input. Once the walk has picked the accepted path, the commit
replays the path's records into the request's state slot in place, writes
the conv rolling buffer as a chain of the path's inputs would have, and moves
the path's K / V into consecutive slots (``tree_commit``).

The recurrence is the ancestor-masked chunk of SGLang's EAGLE-4 ROCm port
(``_gdn_tree_chunk_kernel``): the gated delta rule's
chunked form with the tree's ancestor mask in place of the causal one, no
per-node state, fp32 throughout. Its replay records are
``[row][v head][delta | key | decay]`` (``RECORD_FLOATS`` a row and head),
which ``tree_commit`` replays along the accepted path.

Layouts: the conv pool ``[slots, state_len, dim]`` (any strides), a tree
step's window at rows 0..2 (where a prefill leaves it, and where the commit
puts the next one: tree steps never read ``num_accepted``); the state pool ``[slots, HV, K, V]`` indexed, VK in memory
(``[hv][v][k]``).
"""

import torch
import triton
import triton.language as tl

RECORD_FLOATS = 260  # delta 128 | key 128 | decay, padded


# ---- the tree -----------------------------------------------------------------


@triton.jit
def _ancestor(par, cols, a):
    """Each row's parent table entry at index a (a < 0: past the root, -1 the
    newest history column, -2 the one before, ...)."""
    hit = tl.sum(tl.where(cols[None, :] == a[:, None], par[None, :], 0), 1)
    return tl.where(a >= 0, hit, a - 1)


# ---- conv ------------------------------------------------------------------------


@triton.jit
def _conv_input(xb, sx, a, m2, h0, h1, h2):
    """The conv input at ancestor a of each row: a node's row, or for a < 0
    the history column (-1 the newest)."""
    x = tl.load(
        xb + tl.maximum(a, 0)[:, None] * sx, mask=m2 & (a >= 0)[:, None], other=0.0
    ).to(tl.float32)
    hv = tl.where(
        (a == -1)[:, None],
        h2[None, :],
        tl.where((a == -2)[:, None], h1[None, :], h0[None, :]),
    )
    return tl.where((a >= 0)[:, None], x, hv)


@triton.jit
def _tree_conv_kernel(
    X,
    sx,
    CONV,
    s_slot,
    s_tok,
    s_dim,
    SLOT,
    ACC,
    PAR,
    s_par,
    W,
    B,
    OUT,
    so,
    RAW,
    s_rb,
    sr,
    DIM,
    N: tl.constexpr,
    NP: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    HAS_ACC: tl.constexpr,
    HAS_RAW: tl.constexpr,
):
    """One request's N nodes through the width-4 conv, each over its own
    ancestors; history from the request's rolling buffer at accepted - 1
    (at row 0 without ``ACC``)."""
    b = tl.program_id(0)
    f = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    fm = f < DIM
    r = tl.arange(0, NP)
    rm = r < N
    slot = tl.load(SLOT + b).to(tl.int64)
    # a padded row (slot -1) reads no history
    hm = fm & (slot >= 0)
    off = tl.full((), 0, tl.int64)
    if HAS_ACC:
        off = tl.maximum(tl.load(ACC + b).to(tl.int64) - 1, 0)
    hist = CONV + tl.maximum(slot, 0) * s_slot + f * s_dim
    h0 = tl.load(hist + off * s_tok, mask=hm, other=0.0).to(tl.float32)
    h1 = tl.load(hist + (off + 1) * s_tok, mask=hm, other=0.0).to(tl.float32)
    h2 = tl.load(hist + (off + 2) * s_tok, mask=hm, other=0.0).to(tl.float32)
    par = tl.load(PAR + b * s_par + r, mask=rm & (r > 0), other=-1).to(tl.int32)
    a1 = par
    a2 = _ancestor(par, r, a1)
    a3 = _ancestor(par, r, a2)
    xb = X + (b * N).to(tl.int64) * sx + f[None, :]
    m2 = rm[:, None] & fm[None, :]
    xs = tl.load(xb + r[:, None] * sx, mask=m2, other=0.0)
    if HAS_RAW:  # the pre-conv inputs, kept for the commit
        tl.store(
            RAW + b.to(tl.int64) * s_rb + r[:, None] * sr + f[None, :], xs, mask=m2
        )
    xs = xs.to(tl.float32)

    w0 = tl.load(W + f * 4 + 0, mask=fm, other=0.0).to(tl.float32)
    w1 = tl.load(W + f * 4 + 1, mask=fm, other=0.0).to(tl.float32)
    w2 = tl.load(W + f * 4 + 2, mask=fm, other=0.0).to(tl.float32)
    w3 = tl.load(W + f * 4 + 3, mask=fm, other=0.0).to(tl.float32)
    if HAS_BIAS:
        acc = (
            tl.load(B + f, mask=fm, other=0.0).to(tl.float32)[None, :]
            + xs * w3[None, :]
        )
    else:
        acc = xs * w3[None, :]
    acc += _conv_input(xb, sx, a1, m2, h0, h1, h2) * w2[None, :]
    acc += _conv_input(xb, sx, a2, m2, h0, h1, h2) * w1[None, :]
    acc += _conv_input(xb, sx, a3, m2, h0, h1, h2) * w0[None, :]
    if SILU:
        acc = acc / (1.0 + tl.exp(-acc))
    tl.store(
        OUT + ((b * N).to(tl.int64) + r[:, None]) * so + f[None, :],
        acc.to(OUT.dtype.element_ty),
        mask=m2,
    )


def tree_conv(
    x,
    conv_state,
    slots,
    num_accepted,
    parents,
    weight,
    bias,
    activation,
    out=None,
    raw=None,
):
    """``x`` [bs * N, dim] pre-conv rows (request-major); ``conv_state`` the
    layer's pool, indexed [slot, tok, dim]; ``slots`` [bs] each request's
    conv slot, ``num_accepted`` [bs] (its window at num_accepted - 1; None:
    at row 0, as ``tree_commit.commit`` leaves it);
    ``parents`` [bs, >= N] int32 (node 0's ignored); ``weight`` [dim, 4];
    ``raw`` [bs, N, dim] (any request / row strides) <- ``x`` (the commit's
    copy of the inputs)."""
    bs = slots.shape[0]
    assert slots.is_contiguous() and (
        num_accepted is None or num_accepted.is_contiguous()
    )
    n = x.shape[0] // bs
    dim = x.shape[1]
    out = torch.empty_like(x) if out is None else out
    # 64 features a program and few warps (MI355X, dim 10240, graph-timed: 1 x 16
    # rows 4.7 us, 16 x 10 rows 8.4 us; 128 / 4 warps: 7.0 / 26.8)
    block = 64
    _tree_conv_kernel[(bs, triton.cdiv(dim, block))](
        x, x.stride(0), conv_state, conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        slots, num_accepted if num_accepted is not None else slots, parents, parents.stride(0), weight,
        bias if bias is not None else weight, out, out.stride(0), raw if raw is not None else out,
        raw.stride(0) if raw is not None else 0, raw.stride(1) if raw is not None else 0, dim, N=n, NP=triton.next_power_of_2(n),
        BLOCK=block, HAS_BIAS=bias is not None, SILU=activation in ("silu", "swish", True),
        HAS_ACC=num_accepted is not None, HAS_RAW=raw is not None, num_warps=2 if bs < 4 else 1,
    )  # fmt: skip
    return out


# ---- the gated delta rule -----------------------------------------------------------


@triton.jit
def _tree_chunk_kernel(
    Q,
    K_,
    sqk,
    V_,
    sv,
    A,
    B,
    sab,
    A_LOG,
    DT_BIAS,
    O,
    so,
    S,
    s_slot,
    SLOT,
    PAR,
    s_par,
    REC,
    s_rec,
    scale,
    N: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    VPB: tl.constexpr,
    NP: tl.constexpr,
    LOG2N: tl.constexpr,
    RECF: tl.constexpr,
):
    """The tree as one chunk of the gated delta rule, per (request, v head,
    VPB blocks of BV values of it) (SGLang's EAGLE-4 ROCm port's ``_gdn_tree_chunk_kernel``):
      R = strict-ancestor closure of the parent links, G_t = prod of exp(g)
      along the root path of t, D = (I + diag(beta) (R o G_t/G_j o K K^T))^-1
      diag(beta) (V - diag(G) K S0), O = diag(G) Q S0 + ((R + I) o G_t/G_j o
      Q K^T) D.
    D is each node's record ``delta`` (its update against its parent's
    state), written with its key and exp(g); S0 stays untouched. The gates
    (``fused_gdn_gating``) and the l2 norms of q and k
    (``use_qk_l2norm_in_kernel``) are computed here, by every value block of
    a head alike: the N x N part is small next to the state's."""
    b = tl.program_id(0)
    hv = tl.program_id(1)
    vb = tl.program_id(2)
    h = hv // (HV // H)
    rn = tl.arange(0, NP)
    cn = tl.arange(0, NP)
    ok = tl.arange(0, K)
    m_n = rn < N
    tok = (b * N + rn).to(tl.int64)
    qk_off = tok[:, None] * sqk + h * K + ok[None, :]
    Qt = tl.load(Q + qk_off, mask=m_n[:, None], other=0.0).to(tl.float32)
    Kt = tl.load(K_ + qk_off, mask=m_n[:, None], other=0.0).to(tl.float32)
    Qt = Qt / tl.sqrt(tl.sum(Qt * Qt, 1) + 1e-6)[:, None] * scale
    Kt = Kt / tl.sqrt(tl.sum(Kt * Kt, 1) + 1e-6)[:, None]
    x = tl.load(A + tok * sab + hv, mask=m_n, other=0.0).to(tl.float32) + tl.load(
        DT_BIAS + hv
    ).to(tl.float32)
    sp = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    eg = tl.where(m_n, tl.exp(-tl.exp(tl.load(A_LOG + hv).to(tl.float32)) * sp), 1.0)
    bb = tl.load(B + tok * sab + hv, mask=m_n, other=0.0).to(tl.float32)
    # beta rounded to the input dtype, as fused_gdn_gating leaves it
    beta = tl.where(
        m_n, (1.0 / (1.0 + tl.exp(-bb))).to(B.dtype.element_ty).to(tl.float32), 0.0
    )
    par = tl.load(PAR + b * s_par + rn, mask=m_n & (rn > 0), other=-1)
    R = (cn[None, :] == par[:, None]).to(tl.float32)
    for _ in tl.static_range(LOG2N):
        R = tl.minimum(R + tl.dot(R, R, input_precision="ieee"), 1.0)
    eye = (rn[:, None] == cn[None, :]).to(tl.float32)
    M = R + eye
    logG = tl.sum(M * tl.log(tl.maximum(eg, 1e-30))[None, :], 1)
    G = tl.exp(logG)
    ratio = tl.where(M > 0, tl.exp(logG[:, None] - logG[None, :]), 0.0)
    KK = tl.dot(Kt, tl.trans(Kt), input_precision="ieee")
    QK = tl.dot(Qt, tl.trans(Kt), input_precision="ieee")
    L = beta[:, None] * (R * ratio * KK)
    # (I + L)^-1 = (I - L)(I + L^2)(I + L^4)...: L is nilpotent (strict ancestors)
    Tinv = eye - L
    P = tl.dot(L, L, input_precision="ieee")
    for _ in tl.static_range(LOG2N - 1):
        Tinv = tl.dot(Tinv, eye + P, input_precision="ieee")
        P = tl.dot(P, P, input_precision="ieee")
    Wm = ratio * QK
    slot = tl.load(SLOT + b).to(tl.int64)
    rec = REC + b.to(tl.int64) * s_rec + (rn[:, None] * HV + hv) * RECF
    for i in tl.static_range(VPB):
        ov = (vb * VPB + i) * BV + tl.arange(0, BV)
        # the state's [hv][v][k] memory read k-contiguous ([BV, K]), then turned
        S0 = tl.trans(
            tl.load(
                S + slot * s_slot + hv * K * V + ov[:, None] * K + ok[None, :],
                mask=slot >= 0,
                other=0.0,
            ).to(tl.float32)
        )
        KS = tl.dot(Kt, S0, input_precision="ieee")
        QS = tl.dot(Qt, S0, input_precision="ieee")
        Vt = tl.load(
            V_ + tok[:, None] * sv + hv * V + ov[None, :], mask=m_n[:, None], other=0.0
        )
        rhs = beta[:, None] * (Vt.to(tl.float32) - G[:, None] * KS)
        D = tl.dot(Tinv, rhs, input_precision="ieee")
        Ob = G[:, None] * QS + tl.dot(Wm, D, input_precision="ieee")
        tl.store(
            O + (tok[:, None] * HV + hv) * so + ov[None, :],
            Ob.to(O.dtype.element_ty),
            mask=m_n[:, None],
        )
        tl.store(rec + ov[None, :], D, mask=m_n[:, None])
    if vb == 0:
        tl.store(rec + V + ok[None, :], Kt, mask=m_n[:, None])
        tl.store(rec + V + K, eg[:, None], mask=m_n[:, None])


def tree_gated_delta_rule(
    q, k, v, a, b, A_log, dt_bias, state, slots, parents, records, out
):
    """``q`` / ``k`` [T, H, K], ``v`` [T, HV, V] (T = bs * N, request-major,
    any row strides), ``a`` / ``b`` [T, HV] gate inputs; ``state`` the layer's
    pool, ``slots`` [bs]; ``parents`` [bs, >= N] int32; ``records``
    [bs, N, HV, RECORD_FLOATS] fp32 <- each node's replay record; ``out``
    [T, HV, V] <- the attention output."""
    T, H, K = q.shape
    HV, V = v.shape[1], v.shape[2]
    bs = slots.shape[0]
    assert slots.is_contiguous()
    assert (
        q.stride(0) == k.stride(0)
        and q.stride(1) == k.stride(1) == K
        and q.stride(2) == k.stride(2) == 1
    )
    assert (
        v.stride(1) == V
        and v.stride(2) == 1
        and a.stride(0) == b.stride(0)
        and a.stride(1) == b.stride(1) == 1
    )
    n = T // bs
    np2 = max(16, triton.next_power_of_2(n))
    bv = 32
    # value blocks a program: enough programs to fill the GPU without
    # repeating the N x N part (each program redoes it) more than needed
    # (MI355X, H 16 / HV 48 / K = V = 128: 14 / 28 / 98 us at batch 1 / 16 / 64;
    # the kernel is latency-bound, so few warps a program)
    progs = bs * HV
    vsplit = 4 if progs < 256 else 2 if progs < 1024 else 1
    _tree_chunk_kernel[(bs, HV, vsplit)](
        q, k, q.stride(0), v, v.stride(0), a, b, a.stride(0), A_log, dt_bias, out, out.stride(1), state,
        state.stride(0), slots, parents, parents.stride(0), records, records.stride(0), K**-0.5,
        N=n, H=H, HV=HV, K=K, V=V, BV=bv, VPB=V // bv // vsplit, NP=np2, LOG2N=max(1, (np2 - 1).bit_length()),
        RECF=records.shape[-1], num_warps=2 if progs < 256 else 1,
    )  # fmt: skip
    return out


def flydsl_chunk_ok(state: torch.Tensor, n: int) -> bool:
    """Whether the FlyDSL chunk (``gdn_tree_flydsl``) takes this verify: a
    bf16 VK pool, trees of at most 16 nodes, FlyDSL importable, and
    ``ATOM_GDN_TREE_FLYDSL`` not 0."""
    from atom.utils import envs

    if not envs.ATOM_GDN_TREE_FLYDSL or state.dtype != torch.bfloat16 or n > 16:
        return False
    if state.stride(-2) != 1:
        return False
    global _FLYDSL_OK
    if _FLYDSL_OK is None:
        try:
            import atom.model_ops.gdn_tree_flydsl  # noqa: F401

            _FLYDSL_OK = True
        except ImportError:
            _FLYDSL_OK = False
    return _FLYDSL_OK


_FLYDSL_OK = None
