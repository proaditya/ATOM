# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP4 weights in the layout the Qwen3.8-27B megakernel reads.

The kernel was qualified against a reference compile of the checkpoint:
a dense ``[N, K]`` weight is quantized to E2M1 with one E8M0 scale per 32
values along K, then preshuffled into 16-row x 128-column tiles. These
functions reproduce that compile byte for byte (``weights.check_native``
holds them to the recorded SHA-256 of every tensor), in torch, on any device:

- ``quantize``: the GPU quantizer (scale = 2^ceil(log2(amax / 6)), values
  rounded to nearest-even by ``v_cvt_scalef32_pk_fp4_bf16``);
- ``quantize_nearest_first``: the CPU quantizer used for the small
  DeltaNet ``b`` / ``a`` projections (ties toward the lower code);
- ``preshuffle_weight``, ``preshuffle_scales_small`` and ``pack_scales``: the
  CDNA4 B-operand layouts (``mxfp4_preshuffle_b_16x16`` and the tN = 16 / 64 /
  128 / 256 scale layouts).
"""

import torch

BLOCK = 32
# E2M1 magnitudes by code
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# the midpoints between neighbouring codes; a tie rounds to the even code,
# which is the upper one at 0.75, 1.75 and 3.5
_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_TIE_UP = (False, True, False, True, False, True, False)


def _e8m0_ceil(x: torch.Tensor) -> torch.Tensor:
    """The E8M0 exponent of the smallest power of two >= x (fp32, x >= 0),
    clamped to [1, 254] as the reference quantizers do."""
    bits = x.contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    exp = exp + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return exp.clamp(1, 254)


def _pow2(exp: torch.Tensor) -> torch.Tensor:
    return (exp.to(torch.int32) << 23).view(torch.float32)


def _pack(codes: torch.Tensor) -> torch.Tensor:
    """[..., K] uint8 codes -> [..., K / 2] bytes, the even k in the low nibble."""
    codes = codes.to(torch.uint8)
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``rocket::mxfp4_quantize_weights_gpu``: bf16 [N, K] -> (packed [N, K/2]
    uint8, scales [N, K/32] uint8 E8M0)."""
    n, k = weight.shape
    if k % BLOCK:
        raise ValueError(f"K={k} is not a multiple of {BLOCK}")
    x = weight.to(torch.float32).view(n, k // BLOCK, BLOCK)
    amax = x.abs().amax(-1)
    # the kernel multiplies by the fp32 constant 1/6, it does not divide
    ratio = amax * torch.tensor(1.0 / 6.0, dtype=torch.float32, device=x.device)
    exp = _e8m0_ceil(ratio)
    y = x / _pow2(exp)[..., None]
    a = y.abs()
    code = torch.zeros_like(a, dtype=torch.int32)
    for mid, up in zip(_MIDPOINTS, _TIE_UP):
        code += (a >= mid if up else a > mid).to(torch.int32)
    code |= torch.signbit(y).to(torch.int32) << 3
    return _pack(code.view(n, k)), exp.to(torch.uint8)


def quantize_nearest_first(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference CPU quantizer (DeltaNet b / a): scale = e8m0_ceil(max(amax / 6, 1e-12)); each value takes the first
    of the 16 codes (+0 .. +6, -0 .. -6) at the least distance."""
    n, k = weight.shape
    x = weight.to(torch.float32).view(n, k // BLOCK, BLOCK)
    amax = x.abs().amax(-1)
    exp = _e8m0_ceil(torch.clamp_min(amax / 6.0, 1e-12))
    y = x / _pow2(exp)[..., None]
    table = torch.tensor(
        FP4_VALUES + tuple(-v for v in FP4_VALUES), dtype=torch.float32, device=x.device
    )
    # argmin returns the first minimum: the C loop keeps a strictly better code
    code = (y[..., None] - table).abs().argmin(-1)
    return _pack(code.view(n, k)), exp.to(torch.uint8)


def k_iters(k: int) -> int:
    """128-column steps a row is stored in, rounded up to an even count."""
    return (k + 255) // 256 * 2


def preshuffle_weight(packed: torch.Tensor) -> torch.Tensor:
    """``mxfp4_preshuffle_b_16x16``: [N, K/2] -> [N/16, K/128, 64 lanes, 16 B],
    lane t holding row 16 n_block + t % 16, columns 32 (t / 16) .. + 31."""
    n, half = packed.shape
    k = half * 2
    if n % 16 or k % 256:
        raise ValueError(f"[{n}, {k}] is not 16 x 256 aligned")
    return (
        packed.view(n // 16, 16, k // 128, 4, 16)
        .permute(0, 2, 3, 1, 4)
        .reshape(-1)
        .contiguous()
    )


def preshuffle_scales_small(scales: torch.Tensor) -> torch.Tensor:
    """The tN = 16 scale layout: [N/16, K/128, 64], lane t the scale of row
    16 n_block + t % 16, block 4 k_iter + t / 16."""
    n, blocks = scales.shape
    return (
        scales.view(n // 16, 16, blocks // 4, 4)
        .permute(0, 2, 3, 1)
        .reshape(-1)
        .contiguous()
    )


def pack_scales(scales: torch.Tensor, tile_n: int) -> torch.Tensor:
    """The packed tN = 64 / 128 / 256 scale layout
    (``mxfp4_preshuffle_scales_b_packed``): [K/128, N / tN, 64 lanes, tN / 64
    atoms]."""
    atoms = tile_n // 64
    if tile_n not in (64, 128, 256):
        raise ValueError(f"tile_n={tile_n}")
    n, blocks = scales.shape
    if n % (atoms * 16) or blocks % 8:
        raise ValueError(f"scales [{n}, {blocks}] do not tile at tN={tile_n}")
    return (
        scales.view(n // (atoms * 16), atoms, 16, blocks // 4, 4)
        .permute(3, 0, 4, 2, 1)
        .reshape(-1)
        .contiguous()
    )


def dequantize(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Canonical [N, K/2] codes and [N, K/32] scales -> fp32 [N, K] (tests)."""
    n, half = packed.shape
    table = torch.tensor(
        FP4_VALUES + tuple(-v for v in FP4_VALUES),
        dtype=torch.float32,
        device=packed.device,
    )
    codes = torch.stack(((packed & 15).long(), (packed >> 4).long()), -1).view(
        n, half * 2
    )
    values = table[codes].view(n, -1, BLOCK)
    return (values * _pow2(scales.to(torch.int32))[..., None]).view(n, half * 2)
