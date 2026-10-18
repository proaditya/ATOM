# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared helpers of the DFlash2 drafter: RMSNorm, SiLU*up, MX (micro-scaled) quantization and the
MXFP8 / MXFP4 linear layers (Triton ``tl.dot_scaled`` and FlyDSL ``mxfp8_linear``)."""

import os
import sys
from typing import ClassVar

import torch
import triton
import triton.language as tl

VOCAB = 248320
H = 5120


def rms(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(
        torch.bfloat16
    ) * w


@triton.jit
def _rms_kernel(
    X,
    sx,
    R,
    sr,
    W,
    OUT,
    so,
    RES,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_RES: tl.constexpr,
    eps,
):
    """OUT[row] = rms_norm(X[row] (+ R[row], rounded to bf16 and stored to RES)) * W, as the eager rms()."""
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    m = c < N
    x = tl.load(X + row * sx + c, mask=m, other=0.0).to(tl.float32)
    if HAS_RES:
        x = (x + tl.load(R + row * sr + c, mask=m, other=0.0).to(tl.float32)).to(
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
def _silu_mul_kernel(GU, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(GU + row * 2 * N + c).to(tl.float32)
    u = tl.load(GU + row * 2 * N + N + c).to(tl.float32)
    a = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + row * N + c, (a * u).to(tl.bfloat16))


def rms_into(x, w, out, res_in=None, res_out=None):
    """rows of x (any row stride) -> out rows (any row stride); optional residual add."""
    n = x.shape[0]
    _rms_kernel[(n,)](
        x,
        x.stride(0),
        res_in if res_in is not None else x,
        res_in.stride(0) if res_in is not None else 0,
        w,
        out,
        out.stride(0),
        res_out if res_out is not None else out,
        N=x.shape[1],
        BLOCK=8192,
        HAS_RES=res_in is not None,
        eps=1e-6,
        num_warps=8,
    )
    return out


MX_CEIL = (
    os.environ.get("MX_SCALE", "ceil") == "ceil"
)  # ceil: no clipping (the target kernel's convention); floor: OCP


def fq_mxfp4(w):
    """Fake-quantize to MXFP4 (e2m1, one e8m0 scale per 32 along the last dim), back to bf16."""
    x = w.float().reshape(-1, 32)
    amax = x.abs().amax(1, keepdim=True).clamp_min(1e-30)
    scale = torch.exp2(
        torch.ceil(torch.log2(amax / 6))
        if MX_CEIL
        else torch.floor(torch.log2(amax)) - 2
    )
    y = x / scale
    a = y.abs()
    q = torch.where(
        a < 2,
        torch.round(a * 2) / 2,
        torch.where(a < 4, torch.round(a), torch.round(a / 2) * 2),
    ).clamp(max=6)
    return (torch.sign(y) * q * scale).reshape(w.shape).to(w.dtype)


def fq_mxfp8(w):
    """Fake-quantize to MXFP8 (e4m3, one e8m0 scale per 32 along the last dim), back to the input dtype."""
    x = w.float().reshape(-1, 32)
    amax = x.abs().amax(1, keepdim=True).clamp_min(1e-30)
    scale = torch.exp2(
        torch.ceil(torch.log2(amax / 448))
        if MX_CEIL
        else torch.floor(torch.log2(amax)) - 8
    )
    y = (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (y * scale).reshape(w.shape).to(w.dtype)


def quant_mxfp4(w):
    """[N, K] -> (codes uint8 [N, K/2], low nibble = even k; scales e8m0 uint8 [N, K/32]); same grid as fq_mxfp4."""
    N, K = w.shape
    x = w.float().reshape(N, K // 32, 32)
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    e = (
        torch.ceil(torch.log2(amax / 6))
        if MX_CEIL
        else torch.floor(torch.log2(amax)) - 2
    )
    y = x / torch.exp2(e)
    a = y.abs()
    mag = torch.where(
        a < 2,
        torch.round(a * 2) / 2,
        torch.where(a < 4, torch.round(a), torch.round(a / 2) * 2),
    ).clamp(max=6)
    idx = torch.where(
        mag < 2, mag * 2, torch.where(mag < 5, mag + 2, torch.full_like(mag, 7))
    ).to(
        torch.uint8
    )  # 0,.5,1,1.5 -> 0..3; 2,3,4 -> 4,5,6; 6 -> 7
    code = (idx | ((y < 0) & (idx > 0)).to(torch.uint8) << 3).reshape(N, K)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    scales = (e.squeeze(-1) + 127).clamp(0, 254).to(torch.uint8).contiguous()
    return packed, scales


@triton.jit
def _mx_quant_x_kernel(X, X8, XS, K, CEIL: tl.constexpr = True):
    """MXFP8 (e4m3, e8m0 per 32) activations: one program per (row, 32-block)."""
    r = tl.program_id(0)
    b = tl.program_id(1)
    c = b * 32 + tl.arange(0, 32)
    x = tl.load(X + r * K + c).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 0), 1e-30)
    if CEIL:
        e = tl.ceil(tl.log2(amax / 448.0))
    else:
        e = tl.floor(tl.log2(amax)) - 8
    y = tl.minimum(tl.maximum(x * tl.exp2(-e), -448.0), 448.0)
    tl.store(X8 + r * K + c, y.to(tl.float8e4nv).to(tl.uint8, bitcast=True))
    tl.store(
        XS + r * (K // 32) + b, tl.minimum(tl.maximum(e + 127, 0), 254).to(tl.uint8)
    )


@triton.jit
def _mx_gemm_kernel(
    X,
    XS,
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
    W8: tl.constexpr = False,
):
    """Y[16, N] (+)= X[16, K] @ W^T: MXFP8 activations x MXFP4 weights [N, K] (gfx950 scaled MFMA);
    program (n block, K split)."""
    pid = tl.program_id(0)
    ks = tl.program_id(1)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.arange(0, 16)
    acc = tl.zeros([16, BLOCK_N], tl.float32)
    for k0 in range(ks * KS, ks * KS + KS, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(X + rm[:, None] * K + rk[None, :])
        xs = tl.load(
            XS
            + rm[:, None] * (K // 32)
            + (k0 // 32 + tl.arange(0, BLOCK_K // 32))[None, :]
        )
        sc = tl.load(
            S
            + rn[:, None] * (K // 32)
            + (k0 // 32 + tl.arange(0, BLOCK_K // 32))[None, :]
        )
        if W8:  # MXFP8 weights
            w = tl.load(Q + rn[:, None] * K + rk[None, :])
            acc = tl.dot_scaled(x, xs, "e4m3", tl.trans(w), sc, "e4m3", acc)
        else:
            w = tl.load(
                Q
                + rn[:, None] * (K // 2)
                + (k0 // 2 + tl.arange(0, BLOCK_K // 2))[None, :]
            )
            acc = tl.dot_scaled(x, xs, "e4m3", tl.trans(w), sc, "e2m1", acc)
    if SPLIT:
        tl.atomic_add(Y + rm[:, None] * N + rn[None, :], acc, sem="relaxed")
    else:  # bf16 straight out
        tl.store(YB + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16))


def quant_mxfp8w(w):
    """[N, K] -> (e4m3 codes uint8 [N, K], e8m0 scales [N, K/32]); same grid as fq_mxfp8."""
    N, K = w.shape
    x = w.float().reshape(N, K // 32, 32)
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    e = (
        torch.ceil(torch.log2(amax / 448))
        if MX_CEIL
        else torch.floor(torch.log2(amax)) - 8
    )
    q = (
        (x / torch.exp2(e))
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .reshape(N, K)
        .contiguous()
    )
    return q, (e.squeeze(-1) + 127).clamp(0, 254).to(torch.uint8).contiguous()


@triton.jit
def _mx_gemm_fused_kernel(
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
    W8: tl.constexpr,
    CEIL: tl.constexpr,
):
    """As _mx_gemm_kernel, with the MXFP8 activation quantization of the (<= 16) bf16 rows done in-kernel."""
    pid = tl.program_id(0)
    ks = tl.program_id(1)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.arange(0, 16)
    acc = tl.zeros([16, BLOCK_N], tl.float32)
    for k0 in range(ks * KS, ks * KS + KS, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            X + rm[:, None] * K + rk[None, :], mask=(rm < NROW)[:, None], other=0.0
        ).to(tl.float32)
        x3 = tl.reshape(x, (16, BLOCK_K // 32, 32))
        amax = tl.maximum(tl.max(tl.abs(x3), 2), 1e-30)
        if CEIL:
            e = tl.ceil(tl.log2(amax / 448.0))
        else:
            e = tl.floor(tl.log2(amax)) - 8
        xq = tl.reshape(
            tl.minimum(tl.maximum(x3 * tl.exp2(-e)[:, :, None], -448.0), 448.0),
            (16, BLOCK_K),
        ).to(tl.float8e4nv)
        xs = tl.minimum(tl.maximum(e + 127, 0), 254).to(tl.uint8)
        sc = tl.load(
            S
            + rn[:, None] * (K // 32)
            + (k0 // 32 + tl.arange(0, BLOCK_K // 32))[None, :]
        )
        if W8:
            w = tl.load(Q + rn[:, None] * K + rk[None, :])
            acc = tl.dot_scaled(xq, xs, "e4m3", tl.trans(w), sc, "e4m3", acc)
        else:
            w = tl.load(
                Q
                + rn[:, None] * (K // 2)
                + (k0 // 2 + tl.arange(0, BLOCK_K // 2))[None, :]
            )
            acc = tl.dot_scaled(xq, xs, "e4m3", tl.trans(w), sc, "e2m1", acc)
    if SPLIT:
        tl.atomic_add(Y + rm[:, None] * N + rn[None, :], acc, sem="relaxed")
    else:
        tl.store(YB + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16))


class FlyLinear:
    """MXFP8 x MXFP8 projection on the FlyDSL small-batch kernel (FlyDSL kernels/monokernel/mxfp8_linear.py,
    vendored with ceil activation scales): fixed 8 input rows, output [8, N] bf16 in a persistent buffer.
    Needs FLY_KERNELS_ROOT on sys.path (the vendored FlyDSL `kernels/` tree)."""

    def __init__(self, w, rows=8):
        import os

        root = os.environ.get("FLY_KERNELS_ROOT")
        if root and root not in sys.path:
            sys.path.insert(0, root)
        from kernels.monokernel.mxfp8_linear import Mxfp8Linear

        self.N, self.K = w.shape
        self.rows = rows
        q, sc = quant_mxfp8w(w)
        self.ref = fq_mxfp8(w)
        self.lin = Mxfp8Linear(q, sc, rows)
        self.y = torch.zeros(rows, self.N, dtype=torch.bfloat16, device="cuda")
        self.xpad = torch.zeros(rows, self.K, dtype=torch.bfloat16, device="cuda")

    def __call__(self, x):
        n = x.shape[0]
        if n > self.rows:
            return fq_mxfp8(x) @ self.ref.T
        if n < self.rows or not x.is_contiguous():
            self.xpad[:n].copy_(x)
            x = self.xpad
        self.lin(x, self.y)
        return self.y[:n]


class MxLinear:
    fused = (
        True  # in-kernel activation quantization (one launch per GEMM when not split)
    )
    cfg: ClassVar[dict] = {
        (15360, 10240): (32, 512, 2, 4, 2),
        (5120, 5120): (32, 512, 2, 4, 2),
        (32768, 5120): (32, 512, 1, 4, 2),
        (5120, 16384): (32, 512, 4, 4, 2),
        (5120, 15360): (32, 256, 4, 4, 3),  # tuned on MI355X; tune() refreshes
        # DFlash2 shapes, fused kernel, MXFP8 weights (w8=True) and MXFP4
        (1280, 5120, True): (32, 256, 4, 4, 2),
        (6144, 5120, True): (32, 512, 1, 4, 2),
        (5120, 4096, True): (32, 512, 1, 4, 2),
        (34816, 5120, True): (64, 256, 1, 4, 2),
        (5120, 17408, True): (64, 256, 4, 4, 2),
        (248320, 5120, True): (64, 512, 1, 4, 2),
        (5120, 25600, True): (64, 256, 4, 4, 2),
        (10240, 5120, True): (32, 512, 1, 4, 2),
        (1280, 5120, False): (32, 256, 4, 4, 2),
        (6144, 5120, False): (32, 512, 1, 4, 2),
        (5120, 4096, False): (32, 512, 1, 4, 2),
        (34816, 5120, False): (32, 512, 1, 4, 2),
        (5120, 17408, False): (64, 256, 4, 4, 2),
        (248320, 5120, False): (64, 512, 1, 4, 2),
        (5120, 25600, False): (64, 256, 4, 4, 2),
        (10240, 5120, False): (64, 512, 1, 4, 2),
    }

    def __init__(self, w, block_n=None, split=None, w8=False):
        self.N, self.K = w.shape
        self.w8 = w8
        self.q, self.s = quant_mxfp8w(w) if w8 else quant_mxfp4(w)
        self.ref = (
            fq_mxfp8(w) if w8 else fq_mxfp4(w)
        )  # same values in bf16 (parity checks)
        self.block_k = 256
        self.block_n = block_n or (64 if self.N >= 15360 else 32)
        self.split = split or 1
        while (
            not split
            and (self.N // self.block_n) * self.split < 400
            and (self.K // self.block_k) % (self.split * 2) == 0
        ):
            self.split *= 2
        self.warps, self.stages = 4, 2
        if (self.N, self.K, w8) in MxLinear.cfg:
            self.block_n, self.block_k, self.split, self.warps, self.stages = (
                MxLinear.cfg[(self.N, self.K, w8)]
            )
        elif not w8 and (self.N, self.K) in MxLinear.cfg:
            self.block_n, self.block_k, self.split, self.warps, self.stages = (
                MxLinear.cfg[(self.N, self.K)]
            )
        self.x16 = torch.zeros(16, self.K, dtype=torch.bfloat16, device="cuda")
        self.x8 = torch.zeros(16, self.K, dtype=torch.uint8, device="cuda")
        self.xs = torch.full((16, self.K // 32), 127, dtype=torch.uint8, device="cuda")
        self.y = torch.zeros(16, self.N, dtype=torch.float32, device="cuda")
        self.yb = torch.zeros(16, self.N, dtype=torch.bfloat16, device="cuda")

    def tune(self, timeit):
        """Pick the fastest (block_n, block_k, split, warps, stages) for one row."""
        x1 = torch.randn(1, self.K, device="cuda").to(torch.bfloat16)
        best = None
        ref = self(x1).float()
        for bn in (32, 64, 128):
            for bk in (256, 512):
                for sp in (1, 2, 4, 8, 16):
                    if (
                        (self.K // bk) % sp
                        or (self.N // bn) * sp < 128
                        or (self.N // bn) * sp > 4096
                    ):
                        continue
                    for wp in (4, 8):
                        for st in (2, 3):
                            (
                                self.block_n,
                                self.block_k,
                                self.split,
                                self.warps,
                                self.stages,
                            ) = (bn, bk, sp, wp, st)
                            try:
                                y = self(x1).float()
                                if float((y - ref).norm() / ref.norm()) > 1e-3:
                                    continue
                                t = timeit(lambda: self(x1))
                            # a tile config the GEMM rejects
                            except Exception:  # noqa: BLE001, S112
                                continue
                            if best is None or t < best[0]:
                                best = (t, (bn, bk, sp, wp, st))
        self.block_n, self.block_k, self.split, self.warps, self.stages = best[1]
        MxLinear.cfg[(self.N, self.K, self.w8)] = best[1]
        return best

    def __call__(
        self, x
    ):  # x [n, K] bf16 -> [n, N] bf16 (n > 16, e.g. the prompt extend: dequantized matmul)
        n = x.shape[0]
        if n > 16:
            return fq_mxfp8(x) @ self.ref.T
        if MxLinear.fused:
            if self.split > 1:
                self.y.zero_()
            _mx_gemm_fused_kernel[(self.N // self.block_n, self.split)](
                x,
                n,
                self.q,
                self.s,
                self.y,
                self.yb,
                self.N,
                self.K,
                self.K // self.split,
                BLOCK_N=self.block_n,
                BLOCK_K=self.block_k,
                SPLIT=self.split > 1,
                W8=self.w8,
                CEIL=MX_CEIL,
                num_warps=self.warps,
                num_stages=self.stages,
            )
            return self.y[:n].to(torch.bfloat16) if self.split > 1 else self.yb[:n]
        _mx_quant_x_kernel[(n, self.K // 32)](
            x, self.x8, self.xs, self.K, CEIL=MX_CEIL, num_warps=1
        )
        if self.split > 1:
            self.y.zero_()
        _mx_gemm_kernel[(self.N // self.block_n, self.split)](
            self.x8,
            self.xs,
            self.q,
            self.s,
            self.y,
            self.yb,
            self.N,
            self.K,
            self.K // self.split,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
            SPLIT=self.split > 1,
            W8=self.w8,
            num_warps=self.warps,
            num_stages=self.stages,
        )
        return self.y[:n].to(torch.bfloat16) if self.split > 1 else self.yb[:n]


@triton.jit
def _ord_key(score, idx):
    """float32 score, int64 index -> int64 key ordered by score, ties to the smaller index."""
    b = score.to(tl.int32, bitcast=True)
    o = b ^ ((b >> 31) & 0x7FFFFFFF)
    return (o.to(tl.int64) << 32) | (0xFFFFFFFF - idx)
