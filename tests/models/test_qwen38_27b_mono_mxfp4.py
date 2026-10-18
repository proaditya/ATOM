# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The megakernel's MXFP4 conversion against element-by-element reference
implementations: the GPU weight quantizer, the CPU quantizer of the small
DeltaNet b / a projections, and the B-operand and scale preshuffles. CPU
only; the byte-exact check of a whole converted checkpoint is
``tools/qwen38_27b_mono_check_weights.py``.
"""

import math
import struct

import pytest
import torch

from atom.models.qwen38_27b.mono import mxfp4 as mx

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _e8m0_ceil(x: float) -> int:
    b = _bits(x)
    e = ((b >> 23) & 0xFF) + ((b & 0x7FFFFF) != 0)
    return min(max(e, 1), 254)


def _rne_code(y: float) -> int:
    """Round-to-nearest-even onto E2M1 (the hardware conversion)."""
    a = abs(y)
    best = min(range(8), key=lambda c: (abs(a - E2M1[c]), c & 1))
    return best | (8 if math.copysign(1.0, y) < 0 else 0)


def _ref_quantize(w: torch.Tensor):
    n, k = w.shape
    packed = torch.zeros(n, k // 2, dtype=torch.uint8)
    scales = torch.zeros(n, k // 32, dtype=torch.uint8)
    inv6 = struct.unpack("<f", struct.pack("<f", 1.0 / 6.0))[0]
    for r in range(n):
        for b in range(k // 32):
            vals = [float(v) for v in w[r, b * 32 : b * 32 + 32].float()]
            ratio = struct.unpack(
                "<f", struct.pack("<f", max(abs(v) for v in vals) * inv6)
            )[0]
            e = _e8m0_ceil(ratio)
            scales[r, b] = e
            s = 2.0 ** (e - 127)
            codes = [_rne_code(v / s) for v in vals]
            for i in range(16):
                packed[r, b * 16 + i] = codes[2 * i] | (codes[2 * i + 1] << 4)
    return packed, scales


def test_quantize_matches_reference():
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(16, 256, generator=g) * 0.02).to(torch.bfloat16)
    w[0, :32] = 0  # an all-zero block
    w[1, 5] = 3 * 0.25 * 2.0**-7  # exact ties
    got_p, got_s = mx.quantize(w)
    ref_p, ref_s = _ref_quantize(w)
    assert torch.equal(got_s, ref_s)
    assert torch.equal(got_p, ref_p)


def test_nearest_first_ties_toward_lower_code():
    # 0.75 * scale sits between 0.5 and 1.0: the CPU rule keeps 0.5
    w = torch.zeros(1, 32)
    w[0, 0] = 6.0
    w[0, 2] = 0.75
    w[0, 3] = -0.25
    p, s = mx.quantize_nearest_first(w)
    assert int(s[0, 0]) == 127
    codes = [(int(b) & 15, int(b) >> 4) for b in p[0]]
    assert codes[0] == (7, 0)
    assert codes[1] == (1, 0)  # 0.75 -> 0.5; -0.25 -> +0


def _ref_preshuffle(packed: torch.Tensor) -> torch.Tensor:
    n, half = packed.shape
    k = half * 2
    k_iters = (k + 255) // 256 * 2
    out = torch.zeros((n + 15) // 16 * k_iters * 64 * 16, dtype=torch.uint8)
    for idx in range(out.numel()):
        byte = idx % 16
        lane = (idx // 16) % 64
        k_iter = (idx // (16 * 64)) % k_iters
        nb = idx // (16 * 64 * k_iters)
        row = nb * 16 + lane % 16
        kp = (k_iter * 128 + (lane // 16) * 32 + byte * 2) // 2
        if row < n and kp < half:
            out[idx] = packed[row, kp]
    return out


def _ref_scales(scales: torch.Tensor, tile_n: int) -> torch.Tensor:
    n, blocks = scales.shape
    k = blocks * 32
    k_iters = (k + 255) // 256 * 2
    if tile_n == 16:
        out = torch.full(((n + 15) // 16 * k_iters * 64,), 127, dtype=torch.uint8)
        for idx in range(out.numel()):
            lane, k_iter, nb = idx % 64, (idx // 64) % k_iters, idx // (64 * k_iters)
            row, kb = nb * 16 + lane % 16, k_iter * 4 + lane // 16
            if row < n and kb < blocks:
                out[idx] = scales[row, kb]
        return out
    atoms = tile_n // 64
    groups = (n + atoms * 16 - 1) // (atoms * 16)
    out = torch.full((k_iters * groups * atoms * 64,), 127, dtype=torch.uint8)
    for idx in range(out.numel()):
        pos = idx % (atoms * 64)
        atom, lane = pos % atoms, pos // atoms
        blk = idx // (atoms * 64)
        grp, k_iter = blk % groups, blk // groups
        row = grp * atoms * 16 + atom * 16 + lane % 16
        kb = k_iter * 4 + lane // 16
        if row < n and kb < blocks:
            out[idx] = scales[row, kb]
    return out


def test_preshuffle_weight():
    g = torch.Generator().manual_seed(1)
    packed = torch.randint(0, 256, (32, 128), generator=g, dtype=torch.uint8)
    assert torch.equal(mx.preshuffle_weight(packed), _ref_preshuffle(packed))


@pytest.mark.parametrize("tile_n", [16, 64, 128, 256])
def test_scale_layouts(tile_n):
    g = torch.Generator().manual_seed(2)
    scales = torch.randint(100, 140, (64, 16), generator=g, dtype=torch.uint8)
    ref = _ref_scales(scales, tile_n)
    got = (
        mx.preshuffle_scales_small(scales)
        if tile_n == 16
        else mx.pack_scales(scales, tile_n)
    )
    assert torch.equal(got, ref)


def test_dequantize_roundtrip():
    g = torch.Generator().manual_seed(3)
    w = (torch.randn(16, 64, generator=g)).to(torch.bfloat16)
    p, s = mx.quantize(w)
    deq = mx.dequantize(p, s)
    # each value within half an E2M1 step of the scaled grid
    assert (deq - w.float()).abs().max() <= 2.0 ** (s.int().max().item() - 127)
