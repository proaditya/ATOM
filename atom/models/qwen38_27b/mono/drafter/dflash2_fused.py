# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""the DFlash2 draft (the fast drafter) with every GEMM producer emitting MXFP8 directly.

the fast drafter with FlyDSL GEMMs (prec fly8) spends 123 launches per draft, 31 of them standalone MXFP8
quantize kernels and ~60 small elementwise kernels of ~3.3 us each. Here the producers write the
FlyDSL `mxfp8_linear` activation operands themselves (row-major E4M3 rows + preshuffled E8M0
words, ceil scales -- the vendored quantizer's exact arithmetic) next to their bf16 outputs, and the
GEMMs run `project_quantized` only:
- pre-norm: [layer 0: embedding gather | conv-finish of the previous block + residual add] + RMSNorm
  -> bf16 (conv input) + MXFP8 (kernel-projection GEMM input); the final norm is the same kernel;
- conv-prepare -> MXFP8 only (QKV / gate-up GEMM input);
- GQA combine -> MXFP8 (o_proj input); SiLU * up -> MXFP8 (down input);
- attention conv-finish fused into the post-attention norm.
Per layer 14 launches (the fast drafter: 24). Numerics are the fast drafter's (fly8) bit for bit.
"""

import torch
import triton
import triton.language as tl

from .dflash2_base import H
from .dflash2_fast import (
    DFlash2Fast,
    _gqa_block_kernel,
    _qkv_norm_rope_kernel,
    _selector_kernel,
    _top16_s1,
    _top16_s2,
)


@triton.jit
def _q8(x, r, c0, AQ, ASC, K: tl.constexpr, G: tl.constexpr):
    """x [G * 32] f32 (bf16-valued): row r, columns c0 .. c0 + 32 G (those < K) -> E4M3 bytes + E8M0 scale
    bytes in mxfp8_linear's layout (rows < 16: byte (sc >> 3) * 256 + (sc & 3) * 64 + r * 4 + ((sc >> 2) & 1) * 2).
    """
    x2 = tl.reshape(x, (G, 32))
    amax = tl.maximum(tl.max(tl.abs(x2), 1), 1e-10)
    inv448 = tl.full((G,), 0x3B124925, tl.int32).to(tl.float32, bitcast=True)
    w = (amax * inv448).to(tl.int32, bitcast=True)
    e = (w >> 23) & 0xFF
    e = tl.minimum(e + ((w & 0x7FFFFF) != 0).to(tl.int32), 255)
    qs = ((254 - e) << 23).to(tl.float32, bitcast=True)
    y = (
        tl.minimum(tl.maximum(x2 * qs[:, None], -448.0), 448.0)
        .to(tl.float8e4nv)
        .to(tl.uint8, bitcast=True)
    )
    cc = c0 + tl.arange(0, G * 32)
    tl.store(AQ + r * K + cc, tl.reshape(y, (G * 32,)), mask=cc < K)
    sc = c0 // 32 + tl.arange(0, G)
    tl.store(
        ASC + (sc >> 3) * 256 + (sc & 3) * 64 + r * 4 + ((sc >> 2) & 1) * 2,
        e.to(tl.uint8),
        mask=sc < K // 32,
    )


@triton.jit
def _prenorm_kernel(
    SRC, IDS, CO, BASE, RES_IN, RES_OUT, W, OUT, AQ, ASC, MODE: tl.constexpr, eps
):
    """Row t. MODE 0: x = SRC[IDS[t]] (embedding); residual = x. MODE 1: x = conv-finish(SRC, CO part 1,
    BASE) (the previous block's output); residual = bf16(x + RES_IN). RES_OUT = residual;
    OUT = rms(residual) * W (bf16) and its MXFP8 operand."""
    t = tl.program_id(0)
    c = tl.arange(0, 8192)
    m = c < 5120
    if MODE == 0:
        x = tl.load(SRC + tl.load(IDS + t) * 5120 + c, mask=m, other=0.0).to(tl.float32)
    else:
        co = CO + t * 1280 + 640
        d0 = tl.load(co + c // 16, mask=m, other=0.0).to(tl.float32)
        d1 = tl.load(co + 320 + c // 16, mask=m, other=0.0).to(tl.float32)
        xs = tl.load(SRC + t * 5120 + c, mask=m, other=0.0).to(tl.float32)
        o = (tl.load(BASE + c, mask=m, other=0.0).to(tl.float32) + d0) * xs
        if t > 0:
            o += (
                tl.load(BASE + 5120 + c, mask=m, other=0.0).to(tl.float32) + d1
            ) * tl.load(SRC + (t - 1) * 5120 + c, mask=m, other=0.0).to(tl.float32)
        x = o.to(tl.bfloat16).to(tl.float32)
        x = (
            (x + tl.load(RES_IN + t * 5120 + c, mask=m, other=0.0).to(tl.float32))
            .to(tl.bfloat16)
            .to(tl.float32)
        )
    tl.store(RES_OUT + t * 5120 + c, x.to(tl.bfloat16), mask=m)
    r = 1.0 / tl.sqrt(tl.sum(x * x, 0) / 5120 + eps)
    y = (
        (x * r).to(tl.bfloat16).to(tl.float32)
        * tl.load(W + c, mask=m, other=0.0).to(tl.float32)
    ).to(tl.bfloat16)
    tl.store(OUT + t * 5120 + c, y, mask=m)
    _q8(tl.where(m, y.to(tl.float32), 0.0), t, 0, AQ, ASC, K=5120, G=256)


@triton.jit
def _conv0_q_kernel(X, CO, BASE, AQ, ASC):
    """conv-prepare (part 0 taps of the kernel projection) -> MXFP8 only."""
    t = tl.program_id(0)
    cb = tl.program_id(1)
    c = cb * 1024 + tl.arange(0, 1024)
    co = CO + t * 1280
    d0 = tl.load(co + c // 16).to(tl.float32)
    d1 = tl.load(co + 320 + c // 16).to(tl.float32)
    x = tl.load(X + t * 5120 + c).to(tl.float32)
    o = (tl.load(BASE + c).to(tl.float32) + d0) * x
    if t > 0:
        o += (tl.load(BASE + 5120 + c).to(tl.float32) + d1) * tl.load(
            X + (t - 1) * 5120 + c
        ).to(tl.float32)
    _q8(o.to(tl.bfloat16).to(tl.float32), t, cb * 1024, AQ, ASC, K=5120, G=32)


@triton.jit
def _gqa_combine_q_kernel(PO, PML, AQ, ASC, SPLITS: tl.constexpr):
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
    o = (tl.sum(w[:, None] * acc, 0) / tl.sum(w * l, 0)).to(tl.bfloat16).to(tl.float32)
    _q8(o, r, h * 128, AQ, ASC, K=4096, G=4)


@triton.jit
def _silu_mul_q_kernel(GU, AQ, ASC):
    row = tl.program_id(0)
    c = tl.program_id(1) * 1024 + tl.arange(0, 1024)
    g = tl.load(GU + row * 34816 + c).to(tl.float32)
    u = tl.load(GU + row * 34816 + 17408 + c).to(tl.float32)
    a = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    _q8(
        (a * u).to(tl.bfloat16).to(tl.float32),
        row,
        tl.program_id(1) * 1024,
        AQ,
        ASC,
        K=17408,
        G=32,
    )


class DFlash2Fused(DFlash2Fast):
    """with producer-side MXFP8 (requires prec layers=fly8, lm=fly8)."""

    def __init__(self, path, embed, lm_head, max_pos, prec=None, attn_impl="block"):
        super().__init__(path, embed, lm_head, max_pos, prec=prec, attn_impl=attn_impl)
        assert self.prec["layers"] in ("fly8", "aot8") and (
            self.prec["lm"] in ("fly8", "aot8") or self.prec["lm"].startswith("mx4r")
        ), "the fused drafter needs layers=fly8 and lm=fly8 (the persistent drafter also takes lm=mx4rK)"
        u8 = {"dtype": torch.uint8, "device": "cuda"}
        self.q = {
            k: (torch.zeros(32, k, **u8), torch.zeros(32 * k // 32, **u8))
            for k in (5120, 4096, 17408)
        }
        self.q5b = (
            torch.zeros(32, 5120, **u8),
            torch.zeros(32 * 5120 // 32, **u8),
        )  # second K=5120 operand (conv-prepare)
        b = self.buf
        b["final"] = torch.empty(8, H, dtype=torch.bfloat16, device="cuda")

    @staticmethod
    def _gemm(fl, q, out=None):
        y = fl.y if out is None else out
        fl.lin.project_quantized(q[0], q[1], y)
        return y

    def _prenorm(self, mode, src, co, base, res_in, res_out, w, out, ids=None):
        aq, asc = self.q[5120]
        _prenorm_kernel[(8,)](
            src,
            ids if ids is not None else src,
            co,
            base,
            res_in,
            res_out,
            w,
            out,
            aq,
            asc,
            MODE=mode,
            eps=1e-6,
            num_warps=8,
        )

    def draft_into(self, ids8, base):
        b = self.buf
        pos = base + self.ar8
        term = base + 7
        ids = torch.cat([ids8[:1], self.mask_ids])
        cur, spare = b["res"], b["res2"]
        q5, q4, q17, q5b = self.q[5120], self.q[4096], self.q[17408], self.q5b
        prev_m = prev_cm = prev_base = None
        for li, L in enumerate(self.layers):
            if li == 0:
                self._prenorm(
                    0, self.embed, cur, cur, cur, cur, L["ln1"], b["x"], ids=ids
                )
            else:
                self._prenorm(
                    1, prev_m, prev_cm, prev_base, cur, spare, L["ln1"], b["x"]
                )
                cur, spare = spare, cur
            co = self._gemm(L["akp"], q5)
            _conv0_q_kernel[(8, 5)](
                b["x"], co, L["abase"][0], q5b[0], q5b[1], num_warps=4
            )
            qkv = self._gemm(L["qkv"], q5b)
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
            _gqa_combine_q_kernel[(8, 32)](
                b["po2"], b["pml2"], q4[0], q4[1], SPLITS=16, num_warps=2
            )
            a = self._gemm(L["o"], q4)
            self._prenorm(1, a, co, L["abase"][1], cur, spare, L["ln2"], b["x"])
            cur, spare = spare, cur
            cm = self._gemm(L["mkp"], q5)
            _conv0_q_kernel[(8, 5)](
                b["x"], cm, L["mbase"][0], q5b[0], q5b[1], num_warps=4
            )
            gu = self._gemm(L["gu"], q5b)
            _silu_mul_q_kernel[(8, 17)](gu, q17[0], q17[1], num_warps=4)
            prev_m = self._gemm(L["d"], q17)
            prev_cm = cm
            prev_base = L["mbase"][1]
        final = b["final"]
        self._prenorm(
            1, prev_m, prev_cm, prev_base, cur, spare, self.w["norm.weight"], final
        )
        lg = self._gemm(self.lm, q5)[1:]  # [7, V] bf16 (row 0 unused)
        _top16_s1[(7, 122)](lg, b["k1"], 248320, NB=122, BLOCK=2048, num_warps=8)
        _top16_s2[(7,)](b["k1"], b["k2"], NB=122, NBP=128, num_warps=8)
        proj = final[1:] @ self.hp.T
        _selector_kernel[(1,)](
            b["k2"],
            proj,
            self.w["candidate_selector.predecessor_codebook"],
            self.w["candidate_selector.successor_codebook"],
            ids8,
            num_warps=4,
        )
