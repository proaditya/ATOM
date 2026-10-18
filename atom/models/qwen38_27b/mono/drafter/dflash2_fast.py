# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""fused Triton kernels for the DFlash2 draft (the reference drafter's math, far fewer launches).

Per draft layer: rms(+residual), kernel_projection GEMM, conv-prepare, QKV GEMM, one q/k-norm +
RoPE + KV-record kernel, split-window GQA attention + combine, o_proj GEMM, conv-finish, rms(+residual),
kernel_projection GEMM, conv-prepare, gate/up GEMM, SiLU*mul, down GEMM, conv-finish. GEMMs are
common's MXFP8 x MXFP4 `MxLinear`. The head: LM-head GEMM (7 rows), a two-stage row top-16, the
selector projection, and one selector-walk kernel that writes the 7 drafts into the launch ids.
The context update: fc GEMM, hidden_norm, one fused K/V GEMM for all five layers, one
k-norm + RoPE + record kernel over (rows, kv heads, layers).
"""

import torch
import triton
import triton.language as tl

from .common import MxLinear, _ord_key, _silu_mul_kernel, rms_into
from .dflash2_base import DFlash2, H


@triton.jit
def _conv_kernel(X, CO, BASE, OUT, PART: tl.constexpr, BLOCK: tl.constexpr):
    """out[t, c] = (b0[c] + d0[t, c // 16]) x[t, c] + [t > 0] (b1[c] + d1[t, c // 16]) x[t - 1, c]."""
    t = tl.program_id(0)
    cb = tl.program_id(1)
    c = cb * BLOCK + tl.arange(0, BLOCK)
    co = CO + t * 1280 + PART * 640
    d0 = tl.load(co + c // 16).to(tl.float32)
    d1 = tl.load(co + 320 + c // 16).to(tl.float32)
    x = tl.load(X + t * 5120 + c).to(tl.float32)
    o = (tl.load(BASE + c).to(tl.float32) + d0) * x
    if t > 0:
        o += (tl.load(BASE + 5120 + c).to(tl.float32) + d1) * tl.load(
            X + (t - 1) * 5120 + c
        ).to(tl.float32)
    tl.store(OUT + t * 5120 + c, o.to(tl.bfloat16))


@triton.jit
def _qkv_norm_rope_kernel(
    QKV, sq, POS, QN, KN, COS, SIN, QO, KPTR, VPTR, P, NQ: tl.constexpr, eps
):
    """Row r, head h: q heads (h < NQ) -> rms * q_norm, RoPE -> QO; k heads -> rms * k_norm, RoPE -> K cache;
    v heads -> V cache. Layer l (program axis 2) selects the cache and k_norm (context: 5 layers at once).
    """
    r = tl.program_id(0)
    h = tl.program_id(1)
    l = tl.program_id(2)
    d = tl.arange(0, 64)
    pos = tl.load(POS + r)
    base = QKV + r * sq + l * (NQ + 16) * 128
    if h < NQ + 8:
        x1 = tl.load(base + h * 128 + d).to(tl.float32)
        x2 = tl.load(base + h * 128 + 64 + d).to(tl.float32)
        inv = 1.0 / tl.sqrt((tl.sum(x1 * x1, 0) + tl.sum(x2 * x2, 0)) / 128 + eps)
        if h < NQ:
            w1 = tl.load(QN + d).to(tl.float32)
            w2 = tl.load(QN + 64 + d).to(tl.float32)
        else:
            w1 = tl.load(KN + l * 128 + d).to(tl.float32)
            w2 = tl.load(KN + l * 128 + 64 + d).to(tl.float32)
        x1 = (x1 * inv).to(tl.bfloat16).to(tl.float32) * w1
        x2 = (x2 * inv).to(tl.bfloat16).to(tl.float32) * w2
        x1 = x1.to(tl.bfloat16).to(tl.float32)
        x2 = x2.to(tl.bfloat16).to(tl.float32)
        c = tl.load(COS + pos * 128 + d)
        s = tl.load(SIN + pos * 128 + d)
        y1 = (x1 * c - x2 * s).to(tl.bfloat16)
        y2 = (x2 * c + x1 * s).to(tl.bfloat16)
        if h < NQ:
            tl.store(QO + (r * NQ + h) * 128 + d, y1)
            tl.store(QO + (r * NQ + h) * 128 + 64 + d, y2)
        else:
            kc = (
                tl.load(KPTR + l).to(tl.pointer_type(tl.bfloat16))
                + ((h - NQ) * P + pos).to(tl.int64) * 128
            )
            tl.store(kc + d, y1)
            tl.store(kc + 64 + d, y2)
    else:
        vc = (
            tl.load(VPTR + l).to(tl.pointer_type(tl.bfloat16))
            + ((h - NQ - 8) * P + pos).to(tl.int64) * 128
        )
        dd = tl.arange(0, 128)
        tl.store(vc + dd, tl.load(base + h * 128 + dd))


@triton.jit
def _gqa_split_kernel(
    Q,
    KC,
    VC,
    TERM,
    PO,
    PML,
    P,
    WIN: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """(q head, row, split): keys (term - WIN, term] -- every block row sees the whole window."""
    h = tl.program_id(0)
    r = tl.program_id(1)
    sp = tl.program_id(2)
    d = tl.arange(0, 128)
    term = tl.load(TERM)
    lo = tl.maximum(term - WIN + 1, 0)
    ch = WIN // SPLITS
    st0 = lo + sp * ch
    end = tl.minimum(st0 + ch, term + 1)
    q = tl.load(Q + (r * 32 + h) * 128 + d).to(tl.float32) * 0.08838834764831845
    g = h // 4
    kb = KC + g.to(tl.int64) * P * 128
    vb = VC + g.to(tl.int64) * P * 128
    m_i = tl.full((), float("-inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((128,), tl.float32)
    for st in range(st0, end, BLOCK):
        kp = st + tl.arange(0, BLOCK)
        ok = kp < end
        k = tl.load(
            kb + kp[:, None] * 128 + d[None, :], mask=ok[:, None], other=0.0
        ).to(tl.float32)
        sc = tl.where(ok, tl.sum(k * q[None, :], 1), float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 0))
        alpha = tl.exp(m_i - m_new)
        pr = tl.exp(sc - m_new)
        v = tl.load(
            vb + kp[:, None] * 128 + d[None, :], mask=ok[:, None], other=0.0
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(pr[:, None] * v, 0)
        l_i = l_i * alpha + tl.sum(pr, 0)
        m_i = m_new
    slot = (r * 32 + h) * SPLITS + sp
    tl.store(PO + slot * 128 + d, acc)
    tl.store(PML + slot * 2, m_i)
    tl.store(PML + slot * 2 + 1, l_i)


@triton.jit
def _gqa_block_kernel(
    Q,
    KC,
    VC,
    TERM,
    PO,
    PML,
    P,
    WIN: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """(kv head g, split): all 32 queries of the group (8 rows x 4 heads) against one key chunk,
    K / V loaded once, scores and P.V on MFMA (flash-decode partials)."""
    g = tl.program_id(0)
    sp = tl.program_id(1)
    term = tl.load(TERM)
    lo = tl.maximum(term - WIN + 1, 0)
    ch = WIN // SPLITS
    st0 = lo + sp * ch
    end = tl.minimum(st0 + ch, term + 1)
    qi = tl.arange(0, 32)
    d = tl.arange(0, 128)
    qrow = (qi // 4) * 32 + g * 4 + qi % 4  # (row, head) of query qi
    q = tl.load(Q + qrow[:, None] * 128 + d[None, :])  # [32, 128] bf16
    kb = KC + g.to(tl.int64) * P * 128
    vb = VC + g.to(tl.int64) * P * 128
    m_i = tl.full((32,), float("-inf"), tl.float32)
    l_i = tl.zeros((32,), tl.float32)
    acc = tl.zeros((32, 128), tl.float32)
    for st in range(st0, end, BLOCK):
        kp = st + tl.arange(0, BLOCK)
        ok = kp < end
        k = tl.load(kb + kp[:, None] * 128 + d[None, :], mask=ok[:, None], other=0.0)
        sc = tl.dot(q, tl.trans(k)) * 0.08838834764831845  # [32, BLOCK]
        sc = tl.where(ok[None, :], sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        alpha = tl.exp(m_i - m_new)
        pr = tl.exp(sc - m_new[:, None])
        v = tl.load(vb + kp[:, None] * 128 + d[None, :], mask=ok[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(pr.to(tl.bfloat16), v)
        l_i = l_i * alpha + tl.sum(pr, 1)
        m_i = m_new
    slot = (g * SPLITS + sp) * 32 + qi
    tl.store(PO + slot[:, None] * 128 + d[None, :], acc)
    tl.store(PML + slot * 2, m_i)
    tl.store(PML + slot * 2 + 1, l_i)


@triton.jit
def _gqa_block_combine(PO, PML, O, SPLITS: tl.constexpr):
    g = tl.program_id(0)
    qi = tl.program_id(1)
    d = tl.arange(0, 128)
    sp = tl.arange(0, SPLITS)
    slot = (g * SPLITS + sp) * 32 + qi
    m = tl.load(PML + slot * 2)
    l = tl.load(PML + slot * 2 + 1)
    w = tl.where(l > 0, tl.exp(m - tl.max(m, 0)), 0.0)
    acc = tl.load(PO + slot[:, None] * 128 + d[None, :])
    r = qi // 4
    h = g * 4 + qi % 4
    tl.store(
        O + r * 4096 + h * 128 + d,
        (tl.sum(w[:, None] * acc, 0) / tl.sum(w * l, 0)).to(tl.bfloat16),
    )


@triton.jit
def _gqa_combine_kernel(PO, PML, O, SPLITS: tl.constexpr):
    h = tl.program_id(0)
    r = tl.program_id(1)
    d = tl.arange(0, 128)
    sp = tl.arange(0, SPLITS)
    slot = (r * 32 + h) * SPLITS + sp
    m = tl.load(PML + slot * 2)
    l = tl.load(PML + slot * 2 + 1)
    w = tl.where(l > 0, tl.exp(m - tl.max(m, 0)), 0.0)
    acc = tl.load(PO + slot[:, None] * 128 + d[None, :])
    tl.store(
        O + r * 4096 + h * 128 + d,
        (tl.sum(w[:, None] * acc, 0) / tl.sum(w * l, 0)).to(tl.bfloat16),
    )


@triton.jit
def _top16_s1(LG, K1, V, NB: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    b = tl.program_id(1)
    c = b * BLOCK + tl.arange(0, BLOCK)
    ok = c < V
    x = tl.load(LG + r.to(tl.int64) * V + c, mask=ok, other=0.0).to(tl.float32)
    key = tl.where(ok, _ord_key(x, c.to(tl.int64)), -9223372036854775807)
    tl.store(K1 + (r * NB + b) * 16 + tl.arange(0, 16), tl.topk(key, 16))


@triton.jit
def _top16_s2(K1, K2, NB: tl.constexpr, NBP: tl.constexpr):
    r = tl.program_id(0)
    i = tl.arange(0, NBP * 16)
    key = tl.load(K1 + r * NB * 16 + i, mask=i < NB * 16, other=-9223372036854775807)
    tl.store(K2 + r * 16 + tl.arange(0, 16), tl.topk(key, 16))


@triton.jit
def _key_score(key):
    o = (key >> 32).to(tl.int32)
    b = o ^ ((o >> 31) & 0x7FFFFFFF)
    return b.to(tl.float32, bitcast=True)


@triton.jit
def _selector_kernel(K2, PROJ, PRED, SUCC, IDS):
    """Greedy coherent path: prev = anchor; per step: unary + succ[cand] . (pred[prev] * proj) -> argmax."""
    j = tl.arange(0, 16)
    k = tl.arange(0, 256)
    prev = tl.load(IDS)
    for s in tl.static_range(7):
        key = tl.load(K2 + s * 16 + j)
        cand = 0xFFFFFFFF - (key & 0xFFFFFFFF)
        unary = _key_score(key)
        pr = tl.load(PRED + prev * 256 + k).to(tl.float32) * tl.load(
            PROJ + s * 256 + k
        ).to(tl.float32)
        pr = pr.to(tl.bfloat16).to(tl.float32)
        su = tl.load(SUCC + cand[:, None] * 256 + k[None, :]).to(tl.float32)
        score = unary + tl.sum(su * pr[None, :], 1).to(tl.bfloat16).to(tl.float32)
        best = tl.max(score, 0)
        choice = tl.min(tl.where(score == best, j, 16), 0)
        prev = tl.sum(tl.where(j == choice, cand, 0), 0)
        tl.store(IDS + 1 + s, prev)


class DFlash2Fast(DFlash2):
    def __init__(self, path, embed, lm_head, max_pos, prec=None, attn_impl="block"):
        super().__init__(path, embed, lm_head, max_pos, mx=True, prec=prec)
        self.attn_impl = attn_impl
        w = self.w

        def _kv(t):
            if self.prec["ctx"] == "bf16":
                return lambda x, t=t: x @ t.T
            if self.prec["ctx"] == "fly8":
                from .common import FlyLinear

                return FlyLinear(t)
            if self.prec["ctx"] == "aot8":
                from . import aot_runtime

                return aot_runtime.PackedMx8(t)
            return MxLinear(t, w8=self.prec["ctx"] == "mx8")

        self.kvall = _kv(
            torch.cat(
                [
                    torch.cat(
                        [
                            w[f"layers.{i}.self_attn.k_proj.weight"],
                            w[f"layers.{i}.self_attn.v_proj.weight"],
                        ],
                        0,
                    )
                    for i in range(5)
                ],
                0,
            )
        )
        self.kn_all = torch.stack([L["kn"] for L in self.layers]).contiguous()
        self.kptr = torch.tensor(
            [L["k"].data_ptr() for L in self.layers], dtype=torch.int64, device="cuda"
        )
        self.vptr = torch.tensor(
            [L["v"].data_ptr() for L in self.layers], dtype=torch.int64, device="cuda"
        )
        for i, L in enumerate(self.layers):
            L["kp1"] = self.kptr[i : i + 1]
            L["vp1"] = self.vptr[i : i + 1]
        bf = torch.bfloat16
        self.buf = {
            "x": torch.empty(8, H, dtype=bf, device="cuda"),
            "res": torch.empty(8, H, dtype=bf, device="cuda"),
            "res2": torch.empty(8, H, dtype=bf, device="cuda"),
            "cv": torch.empty(8, H, dtype=bf, device="cuda"),
            "q": torch.empty(8, 32, 128, dtype=bf, device="cuda"),
            "o": torch.empty(8, 4096, dtype=bf, device="cuda"),
            "act": torch.empty(8, 17408, dtype=bf, device="cuda"),
            "ch": torch.empty(8, H, dtype=bf, device="cuda"),
            "po": torch.empty(8 * 32 * 4, 128, dtype=torch.float32, device="cuda"),
            "pml": torch.empty(8 * 32 * 4, 2, dtype=torch.float32, device="cuda"),
            "po2": torch.empty(8 * 16 * 32, 128, dtype=torch.float32, device="cuda"),
            "pml2": torch.empty(8 * 16 * 32, 2, dtype=torch.float32, device="cuda"),
            "k1": torch.empty(7 * 122 * 16, dtype=torch.int64, device="cuda"),
            "k2": torch.empty(7 * 16, dtype=torch.int64, device="cuda"),
        }

    def context(self, pos, aux):
        n = pos.numel()
        if n > 8:
            return super().context(pos, aux)
        ch = self.buf["ch"][:n]
        rms_into(self.fc(aux.reshape(n, 5 * H)), self.w["hidden_norm.weight"], ch)
        kv = self.kvall(ch)  # [n, 5 * 2048]
        _qkv_norm_rope_kernel[(n, 16, 5)](
            kv,
            kv.stride(0),
            pos,
            self.kn_all,
            self.kn_all,
            self.cos,
            self.sin,
            kv,
            self.kptr,
            self.vptr,
            self.P,
            NQ=0,
            eps=1e-6,
            num_warps=1,
        )

    def draft_into(self, ids8, base):
        b = self.buf
        pos = base + self.ar8
        term = base + 7
        ids = torch.cat([ids8[:1], self.mask_ids])
        hid = self.embed.index_select(0, ids)
        cur, spare = b["res"], b["res2"]  # running residual / free buffer
        for li, L in enumerate(self.layers):
            if li == 0:
                cur.copy_(hid)
                rms_into(hid, L["ln1"], b["x"])
            else:
                rms_into(hid, L["ln1"], b["x"], res_in=cur, res_out=spare)
                cur, spare = spare, cur
            co = L["akp"](b["x"])
            _conv_kernel[(8, 5)](
                b["x"], co, L["abase"][0], b["cv"], PART=0, BLOCK=1024, num_warps=4
            )
            qkv = L["qkv"](b["cv"])
            _qkv_norm_rope_kernel[(8, 48, 1)](
                qkv,
                qkv.stride(0),
                pos,
                L["qn"],
                L["kn"],
                self.cos,
                self.sin,
                b["q"],
                L["kp1"],
                L["vp1"],
                self.P,
                NQ=32,
                eps=1e-6,
                num_warps=1,
            )
            if self.attn_impl == "block":
                _gqa_block_kernel[(8, 16)](
                    b["q"],
                    L["k"],
                    L["v"],
                    term,
                    b["po2"],
                    b["pml2"],
                    self.P,
                    WIN=self.win,
                    SPLITS=16,
                    BLOCK=64,
                    num_warps=4,
                )
                _gqa_block_combine[(8, 32)](
                    b["po2"], b["pml2"], b["o"], SPLITS=16, num_warps=2
                )
            else:
                _gqa_split_kernel[(32, 8, 4)](
                    b["q"],
                    L["k"],
                    L["v"],
                    term,
                    b["po"],
                    b["pml"],
                    self.P,
                    WIN=self.win,
                    SPLITS=4,
                    BLOCK=64,
                    num_warps=4,
                )
                _gqa_combine_kernel[(32, 8)](
                    b["po"], b["pml"], b["o"], SPLITS=4, num_warps=4
                )
            a = L["o"](b["o"])
            _conv_kernel[(8, 5)](
                a, co, L["abase"][1], b["cv"], PART=1, BLOCK=1024, num_warps=4
            )
            rms_into(b["cv"], L["ln2"], b["x"], res_in=cur, res_out=spare)
            cur, spare = spare, cur
            cm = L["mkp"](b["x"])
            _conv_kernel[(8, 5)](
                b["x"], cm, L["mbase"][0], b["cv"], PART=0, BLOCK=1024, num_warps=4
            )
            gu = L["gu"](b["cv"])
            _silu_mul_kernel[(8, 17)](gu, b["act"], N=17408, BLOCK=1024, num_warps=4)
            m = L["d"](b["act"])
            hid = torch.empty(8, H, dtype=torch.bfloat16, device="cuda")
            _conv_kernel[(8, 5)](
                m, cm, L["mbase"][1], hid, PART=1, BLOCK=1024, num_warps=4
            )
        final = torch.empty(8, H, dtype=torch.bfloat16, device="cuda")
        rms_into(hid, self.w["norm.weight"], final, res_in=cur, res_out=spare)
        sample = final[1:]
        lg = self.lm(final)[1:]  # [7, V] bf16 (row 0 unused)
        _top16_s1[(7, 122)](lg, b["k1"], 248320, NB=122, BLOCK=2048, num_warps=8)
        _top16_s2[(7,)](b["k1"], b["k2"], NB=122, NBP=128, num_warps=8)
        proj = sample @ self.hp.T
        _selector_kernel[(1,)](
            b["k2"],
            proj,
            self.w["candidate_selector.predecessor_codebook"],
            self.w["candidate_selector.successor_codebook"],
            ids8,
            num_warps=4,
        )
