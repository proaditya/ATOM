# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
# FlyDSL kernel source, kept as written: it is traced at build time, where closures over loop
# variables and the nesting of dynamic conditions are deliberate (they shape the generated control flow).
"""MXFP8 x MXFP4 small-batch projection (FlyDSL) -- the DFlash2 draft's LM head at half the bytes.

FlyDSL's mxfp8_linear project tile with the weight operand in E2M1: the same gfx950 scaled MFMA
16x16x128 (A = E4M3, cbsz 0; B = E2M1, blgp 4), the same activation staging (row-major E4M3 rows +
preshuffled E8M0 words in LDS) and the same per-32 E8M0 weight scales (pack_mxfp8_scale). Weights: each lane's
32 E2M1 codes of a 128-k step are its natural 32-K block (see pack_mxfp4). One wave per 16-row tile over the full K, with a
``depth``-step weight lookahead (5 VGPRs per step); several CTAs per CU (LDS 45 KB).
"""
import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64, Stream, T
from kernels.common import buffer_ops as bo
from kernels.monokernel.ops import rsrc
from kernels.monokernel.packing import pack_mxfp8_scale

THREADS = 256
WAVES = 4


def pack_mxfp4(codes_packed: torch.Tensor, scales: torch.Tensor):
    """(common.quant_mxfp4 output: codes [N, K/2] low nibble = even k, scales [N, K/32]) ->
    (weights uint8 [N/16 * K/128 * 64 * 16], preshuffled scales).

    Operand layouts of the scaled MFMA 16x16x128 (probed with one-hot activations): an E4M3 lane
    (group g = lane // 16) holds hardware K [16 g, 16 g + 16) in its first 16 bytes and [64 + 16 g, + 16)
    in its last 16 -- mxfp8_linear's activation loads put memory K there unchanged; an E2M1 lane holds
    hardware K [32 g, 32 g + 32) contiguously (low nibble first). So the weight lane (g, row) gets the
    32 codes of row K [128 u + 32 g, + 32) in natural order; scales stay per natural 32-group."""
    n, kh = codes_packed.shape
    k = kh * 2
    w = codes_packed.view(n // 16, 16, k // 128, 4, 16)                          # [tile, row, k128, g, 16 bytes = 32 codes]
    w4 = w.permute(0, 2, 3, 1, 4).contiguous().view(-1)                          # [tile, k128, lane = g * 16 + row, 16]
    return w4, pack_mxfp8_scale(scales)


@functools.cache
def build_mxfp4_project(rows: int, n: int, k: int, depth: int = 8):
    assert rows <= 8 and n % 32 == 0 and k % 256 == 0
    n_tiles = n // 16
    blocks = (n_tiles + WAVES - 1) // WAVES
    k128s = k // 128
    act_words = rows * k // 4
    sc_words = (k // 256) * 64

    @fx.struct
    class Smem:
        a: fx.Array[fx.Float32, act_words + sc_words, 16]

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def project_kernel(act: Int64, act_scale: Int64, w4: Int64, s4: Int64, out: Int64):
        bid = gpu.block_idx.x
        tid = gpu.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        lane_div16 = lane // 16
        lane_mod16 = lane % 16
        v4f = fx.Vector.make_type(4, fx.Float32)
        smem = fx.SharedAllocator().allocate(Smem).peek().a.ptr
        scl = fx.recast_iter(fx.Int32, smem + act_words)
        r_a = rsrc(act)
        r_as = rsrc(act_scale)
        r_w = rsrc(w4)
        r_s = rsrc(s4)
        t = fx.min(bid * WAVES + wave, n_tiles - 1)
        live_t = (bid * WAVES + wave) < n_tiles

        def unit(u):            # 128-k step u of tile t: 4 dwords of E2M1 + the scale word
            sw = fx.Int32(bo.buffer_load(r_s, ((t // 2) * (k // 256) + u // 2) * 64 + lane_div16 * 16 + lane_mod16, vec_width=1, dtype=T.i32))
            sw = ((t % 2) != 0).select(sw.shrui(fx.Int32(8)), sw)
            wv = fx.Vector(bo.buffer_load(r_w, ((t * k128s + u) * 64 + lane) * 4, vec_width=4, dtype=T.i32))
            return sw, wv

        q_ = [unit(u) for u in range(min(depth, k128s))]        # issued before the activation staging
        for i in range_constexpr((act_words + 4 * THREADS - 1) // (4 * THREADS)):
            wd = (tid + i * THREADS) * 4
            if wd < act_words:
                fx.ptr_store(fx.Vector(bo.buffer_load(r_a, wd, vec_width=4, dtype=T.i32)).bitcast(fx.Float32), smem + wd)
        for i in range_constexpr((sc_words + THREADS - 1) // THREADS):
            wd = tid + i * THREADS
            if wd < sc_words:
                fx.ptr_store(fx.Int32(bo.buffer_load(r_as, wd, vec_width=1, dtype=T.i32)), scl + wd)
        gpu.barrier()
        valid_row = lane_mod16 < rows
        input_row = fx.min(lane_mod16, rows - 1)
        scale_lane = lane_div16 * 16 + lane_mod16
        c = fx.Vector.filled(4, 0.0, fx.Float32)
        for u in range_constexpr(k128s):
            if const_expr(u + depth < k128s):
                q_.append(unit(u + depth))
            k_base = u * 128 + lane_div16 * 16
            halves = []
            for q in range_constexpr(2):
                ld = fx.Vector(fx.ptr_load(smem + (input_row * k + k_base + q * 64) // 4, result_type=v4f)).bitcast(fx.Int32)
                halves.append([valid_row.select(ld[i], fx.Int32(0)) for i in range(4)])
            a = fx.Vector.from_elements(halves[0] + halves[1], fx.Int32)
            s_in = valid_row.select(fx.ptr_load(scl + (u // 2) * 64 + scale_lane), fx.Int32(0x7F007F))
            sw, wv = q_[u]
            b = fx.Vector.from_elements([wv[i] for i in range(4)] + [fx.Int32(0)] * 4, fx.Int32)
            sel = (u % 2) * 2
            c = fx.Vector(rocdl.mfma_scale_f32_16x16x128_f8f6f4(T.vec(4, T.f32), [a, b, c, 0, 4, sel, s_in, sel, sw]))
        r_o = rsrc(out)
        if live_t:
            for e in range_constexpr(4):
                row = lane_div16 * 4 + e
                if row < rows:
                    bo.buffer_store(c[e].to(fx.BFloat16), r_o, row * n + t * 16 + lane_mod16)

    @flyc.jit
    def launch(act: Int64, act_scale: Int64, w4: Int64, s4: Int64, out: Int64, stream: Stream = Stream(None)):
        project_kernel(act, act_scale, w4, s4, out, value_attrs={"rocdl.flat_work_group_size": f"{THREADS},{THREADS}"}).launch(
            grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)

    launch.func.__name__ = f"mxfp4_project_r{rows}_n{n}_k{k}_d{depth}"
    return launch


@functools.cache
def build_mxfp8_bigk(rows: int, n: int, k: int, ks: int = 4, depth: int = 6):
    """MXFP8 x MXFP8 projection for a K too large for mxfp8_linear's LDS activation stage (the DFlash2
    context fc, 5120 x 25600): one CTA per 16-column tile, ``ks`` waves splitting K, activation fragments
    and weights read from global (L2 / HBM) with a ``depth``-step lookahead, partials reduced through LDS.
    Operands: mxfp8_linear's quantize_input buffers (row-major E4M3 + preshuffled E8M0) and
    pack_mxfp8_weight / pack_mxfp8_scale weights."""
    assert rows <= 8 and n % 16 == 0 and (k // 128) % ks == 0 and (k // 128 // ks) % 2 == 0   # (even steps per wave: static opsel)
    tiles = n // 16
    threads = 64 * ks
    k128s = k // 128
    upw = k128s // ks                         # 128-k steps per wave
    kc = k // 64

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, ks * 64 * 4, 16]

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def bigk_kernel(act: Int64, act_scale: Int64, w: Int64, s: Int64, out: Int64):
        t = gpu.block_idx.x
        tid = gpu.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        lane_div16 = lane // 16
        lane_mod16 = lane % 16
        v4 = fx.Vector.make_type(4, fx.Float32)
        red = fx.SharedAllocator().allocate(Smem).peek().red.ptr
        r_a = rsrc(act)
        r_as = rsrc(act_scale)
        r_w = rsrc(w)
        r_s = rsrc(s)
        valid_row = lane_mod16 < rows
        input_row = fx.min(lane_mod16, rows - 1)
        scale_lane = lane_div16 * 16 + lane_mod16

        def unit(u):
            k128 = wave * upw + u
            k256 = k128 // 2
            k_base = k128 * 128 + lane_div16 * 16
            a_ = [fx.Vector(bo.buffer_load(r_a, (input_row * k + k_base + q * 64) // 4, vec_width=4, dtype=T.i32)) for q in range(2)]
            sa = fx.Int32(bo.buffer_load(r_as, k256 * 64 + scale_lane, vec_width=1, dtype=T.i32))
            sw = fx.Int32(bo.buffer_load(r_s, ((t // 2) * (k // 256) + k256) * 64 + scale_lane, vec_width=1, dtype=T.i32))
            w_ = [fx.Vector(bo.buffer_load(r_w, (((t * kc + k128 * 2 + q) * 4 + lane_div16) * 16 + lane_mod16) * 4, vec_width=4, dtype=T.i32))
                  for q in range(2)]
            return a_, sa, sw, w_

        q_ = [unit(u) for u in range(min(depth, upw))]
        c = fx.Vector.filled(4, 0.0, fx.Float32)
        for u in range_constexpr(upw):
            if const_expr(u + depth < upw):
                q_.append(unit(u + depth))
            a_, sa, sw, w_ = q_[u]
            a = fx.Vector.from_elements([valid_row.select(a_[q][i], fx.Int32(0)) for q in range(2) for i in range(4)], fx.Int32)
            b = fx.Vector.from_elements([w_[q][i] for q in range(2) for i in range(4)], fx.Int32)
            sw = ((t % 2) != 0).select(sw.shrui(fx.Int32(8)), sw)
            sa = valid_row.select(sa, fx.Int32(0x7F007F))
            sel = (u % 2) * 2
            c = fx.Vector(rocdl.mfma_scale_f32_16x16x128_f8f6f4(T.vec(4, T.f32), [a, b, c, 0, 0, sel, sa, sel, sw]))
        fx.ptr_store(c, red + (wave * 64 + lane) * 4)
        gpu.barrier()
        if wave == 0:
            tot = [fx.ptr_load(red + lane * 4 + e) for e in range(4)]
            for ww in range_constexpr(1, ks):
                for e in range_constexpr(4):
                    tot[e] = tot[e] + fx.ptr_load(red + (ww * 64 + lane) * 4 + e)
            r_o = rsrc(out)
            for e in range_constexpr(4):
                row = lane_div16 * 4 + e
                if row < rows:
                    bo.buffer_store(tot[e].to(fx.BFloat16), r_o, row * n + t * 16 + lane_mod16)

    @flyc.jit
    def launch(act: Int64, act_scale: Int64, w: Int64, s: Int64, out: Int64, stream: Stream = Stream(None)):
        bigk_kernel(act, act_scale, w, s, out, value_attrs={"rocdl.flat_work_group_size": f"{threads},{threads}"}).launch(
            grid=(tiles, 1, 1), block=(threads, 1, 1), stream=stream)

    launch.func.__name__ = f"mxfp8_bigk_r{rows}_n{n}_k{k}_ks{ks}_d{depth}"
    return launch
