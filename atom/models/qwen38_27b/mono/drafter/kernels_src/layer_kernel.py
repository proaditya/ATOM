# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
# FlyDSL kernel source, kept as written: it is traced at build time, where closures over loop
# variables and the nesting of dynamic conditions are deliberate (they shape the generated control flow).
"""one DFlash2 draft layer (8 rows: anchor + 7 masks) as one persistent FlyDSL launch.

Same math as dflash2_fused.py (the Triton kernels with producer-side MXFP8), one launch of 256 CTAs x
512 threads, stages handing off through tagged-pair mailboxes (tag = step * 8 + layer):

  P0  rows (CTA r < 8): [layer 0: embedding | conv-finish(previous m, cm) + residual] -> RMSNorm(ln1)
      -> x (bf16 pairs) + xq (E4M3 words + E8M0 scales); residual stored plainly
  A1  kernel_projection(attention) GEMM 1280 x 5120 on xq -> co (bf16 pairs)
  A3  conv-prepare(x, co part 0) + MXFP8 in every QKV CTA's LDS -> QKV GEMM 6144 x 5120 -> qkv (bf16 pairs)
  A5  attention, task (kv head g, key split s of 16 x 128 keys): q heads 4g..4g+3 rms * q_norm + RoPE
      in the CTA; split 15 also k / v head g (k_norm, RoPE) -> K / V cache (plain) and its last 8 keys;
      scores and P.V on bf16 MFMA 16x16x32 -> partials (acc, m, l) as f32 pairs
  A5b combine, task (g, row) -> o (bf16) -> MXFP8 words + scales
  A6  o_proj GEMM 5120 x 4096 -> a (bf16 pairs)
  A7  rows: conv-finish(a, co part 1) + residual -> RMSNorm(ln2) -> x2 (bf16 pairs) + x2q
  M1  kernel_projection(mlp) GEMM 1280 x 5120 on x2q -> cm (bf16 pairs; also plain for the next layer)
  M3  conv-prepare(x2, cm part 0) + MXFP8 in LDS -> gate/up (K split over wave pairs) -> SiLU * up ->
      E8M0 per 32 -> act (E4M3 words + scales)
  M4  down GEMM 5120 x 17408 -> m (plain bf16, for the next layer's P0)

GEMM stages: N <= 256 tiles -> one 16-column tile per CTA, K split over 8 waves; N > 256 -> CTAs
lb < N - 256 own two tiles (4 waves each). lb = (block - stage offset) mod 256 spreads the stages.
Weights stream at 128-k granularity with a ``depth``-unit lookahead; ``pre`` units issued before
each stage's wait. Timeline: s_memrealtime at the end of each stage a CTA takes part in.
"""
import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int64, Stream, T
from kernels.common import buffer_ops as bo
from kernels.monokernel.layout import CM_DEV
from kernels.monokernel.ops import exp, mem_realtime, rcp, rsq, rsrc, spin_pause, uniform, xred, xshfl

ROWS = 8
BLOCKS = 256
THREADS = 512
WAVES = 8
H = 5120
NKP = 1280
NQKV = 6144
NO = 4096
NI = 17408
HD = 128
WIN = 2048
SPLITS = 16
KPS = WIN // SPLITS                   # 128 keys per split
MASK_ID = 248070
TL = 16

# mailbox regions (pairs)
_MB = [('xb', 8 * H // 2), ('xq', 8 * H // 4), ('xs', 8 * H // 32), ('co', 8 * NKP // 2), ('qkv', 8 * NQKV // 2),
       ('ap', 128 * 32 * 130), ('oq', 8 * NO // 4), ('os', 8 * NO // 32), ('ab', 8 * H // 2),
       ('x2b', 8 * H // 2), ('x2q', 8 * H // 4), ('x2s', 8 * H // 32), ('cm', 8 * NKP // 2),
       ('act', 8 * NI // 4), ('acts', 8 * NI // 32), ('cv', 8 * H // 4), ('cvs', 8 * H // 32), ('cv2', 8 * H // 4), ('cv2s', 8 * H // 32), ('mb', 8 * H // 2)]
MB = {}
_o = 0
for _n, _s in _MB:
    MB[_n] = _o
    _o += _s
MB_PAIRS = _o

# pointer table entries
P_EMBED, P_IDS, P_RES, P_M, P_CM, P_MBASE1_PREV, P_LN1, P_LN2, P_AKP_W, P_AKP_S, P_ABASE0, P_ABASE1, \
    P_QKV_W, P_QKV_S, P_QN, P_KN, P_COS, P_SIN, P_KC, P_VC, P_BASE, P_O_W, P_O_S, P_MKP_W, P_MKP_S, P_MBASE0, \
    P_GU_W, P_GU_S, P_D_W, P_D_S, P_M_OUT, P_CM_OUT, P_MB, P_STEP, P_TL, P_LAYER, \
    P_MBASE1, P_NORM, P_LMQ, P_LMS, P_FINAL = range(41)
N_PTRS = 41


@functools.cache
def build_layer(max_pos: int, layer0: bool, timeline: bool = True, depth: int = 4, pre: int = 2, depth_gu: int = 3, pre_gu: int = 2, pre_big: int = 4,
                   conv_stage: bool = True, pre_d: int = 2, final_norm: bool = False, fuse_kp: bool = False, fuse_kp_m: bool = None):
    """final_norm (the last layer): after the down projection, rows CTAs compute the draft's final
    RMSNorm (conv-finish(m, cm part 1) + residual) -> final (bf16) and the LM head's MXFP8 operand
    (mxfp8_linear's row-major E4M3 rows + preshuffled E8M0 bytes) -- the head's first launch folded in."""
    P = max_pos
    fuse_kp_a = fuse_kp
    fuse_kp_m = fuse_kp if fuse_kp_m is None else fuse_kp_m
    ARENA = ROWS * NI // 4 + (NI // 256) * 64          # 39168 words: the down stage's activation + scales
    # attention LDS (words): Q [32][QS], K / V [128][KS], scores [32][SS] f32, P [32][PS] bf16 pairs
    QS = HD // 2 + 4
    KS = HD // 2 + 4
    SS = KPS + 4
    PS = KPS // 2 + 4
    A_Q = 0
    A_K = A_Q + 32 * QS
    A_V = A_K + KPS * KS
    A_S = A_V + KPS * KS
    A_P = A_S + 32 * SS
    A_ML = A_P + 32 * PS
    assert A_ML + 64 <= ARENA
    # gate/up stage LDS: conv coefficients f32 [8][640], act E4M3 [8][5120] + scales, K-half partials
    G_CM = 0
    G_ACT = G_CM + 8 * 640
    G_SCL = G_ACT + 8 * H // 4
    G_X = G_SCL + (H // 256) * 64                       # x2 staged as bf16 pair words [8][2560] (conv phase only)
    G_PART = G_X                                         # K-half partials reuse it after the conv phase
    G_E = G_X + ROWS * H // 2
    assert G_E + ROWS * 160 <= ARENA
    # QKV stage LDS: co part 0 f32 [8][640], act, scales
    Q_CO = 0
    Q_ACT = Q_CO + 8 * 640
    Q_SCL = Q_ACT + 8 * H // 4
    Q_X = Q_SCL + (H // 256) * 64
    Q_E = Q_X + ROWS * H // 2                            # E8M0 codes [8][160]
    assert Q_E + ROWS * 160 <= ARENA
    RED_ROW = ARENA - 64                                 # row stages: per-wave partial sums

    @fx.struct
    class Smem:
        a: fx.Array[fx.Float32, ARENA, 16]

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def layer_kernel(table: Int64):
        tid = gpu.thread_idx.x
        bid = gpu.block_idx.x
        lane = tid % 64
        wave = tid // 64
        lane_div16 = lane // 16
        lane_mod16 = lane % 16
        smem = fx.SharedAllocator().allocate(Smem).peek().a.ptr
        smem_i = fx.recast_iter(fx.Int32, smem)
        r_tab = rsrc(table)
        v4f = fx.Vector.make_type(4, fx.Float32)

        def ptr(i):
            pv = fx.Vector(bo.buffer_load(r_tab, i * 2, vec_width=2, dtype=T.i32))
            return (fx.Int64(uniform(pv[1])) << 32) | fx.Int64(fx.Uint32(uniform(pv[0])))

        layer = uniform(bo.buffer_load(r_tab, P_LAYER * 2, vec_width=1, dtype=T.i32))
        mbp = ptr(P_MB)
        r_mb = rsrc(mbp)
        tlp = ptr(P_TL)
        bpv = fx.Vector(bo.buffer_load(rsrc(ptr(P_BASE)), 0, vec_width=2, dtype=T.i32))
        base = uniform(bpv[0])                           # draft block position (< 2^31)
        # tag: unique per (draft, layer) when either the block position (every decode round) or the
        # step counter (tests drafting repeatedly at one position) changes
        step_ = uniform(bo.buffer_load(rsrc(ptr(P_STEP)), 0, vec_width=1, dtype=T.i32))
        tag = ((base & fx.Int32(0xFFFFF)) << fx.Int32(8)) | ((step_ & fx.Int32(0x1F)) << fx.Int32(3)) | layer | fx.Int32(0x40000000)   # never 0 (fresh mailboxes)
        term = base + 7
        lo = term - (WIN - 1)
        valid_row = lane_mod16 < ROWS
        input_row = fx.min(lane_mod16, ROWS - 1)
        scale_lane = lane_div16 * 16 + lane_mod16
        atoms = [fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN, opsel_a=o, opsel_b=o)) for o in (0, 2)]

        def stamp(col):
            if const_expr(timeline):
                if tid == 0:
                    fx.generic_store(fx.inttoptr(fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8),
                                                 tlp + fx.Int64(bid * TL + col) * 8), mem_realtime())

        stamp(0)

        def bf16r(v):
            return fx.Float32(fx.Float32(v).to(fx.BFloat16))

        def bfpair(a, b):
            return fx.Vector.from_elements([a, b], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]

        def unpack(w):
            return (w << fx.Int32(16)).bitcast(fx.Float32), (w & fx.Int32(-65536)).bitcast(fx.Float32)

        def put(off_pair, words):
            """Pairs off_pair .. + len(words): (word, tag) each, one store (len 1 or 2)."""
            vals = []
            for w in words:
                vals += [fx.Int32(w), tag]
            bo.buffer_store(fx.Vector.from_elements(vals, fx.Int32), r_mb, off_pair * 2, cache_modifier=CM_DEV)

        def poll(offs, width):
            """Tagged pairs at pair offsets ``offs`` (width 1 or 2 pairs each), loaded together and re-loaded
            until every tag is this launch's. Returns the payload words."""
            def load_all():
                words = []
                for o in offs:
                    v = fx.Vector(bo.buffer_load(r_mb, o * 2, vec_width=2 * width, dtype=T.i32, cache_modifier=CM_DEV))
                    words += [v[e] for e in range(2 * width)]
                return fx.Vector.from_elements(words, fx.Int32)
            nw = len(offs) * 2 * width

            def pending(v):
                bad = v[1] != tag
                for e in range_constexpr(3, nw, 2):
                    bad = bad | (v[e] != tag)
                return bad
            v = load_all()
            while pending(v):
                spin_pause()
                v = load_all()
            return [v[e] for e in range(0, nw, 2)]

        def poll_many(offs, width, batch=12):
            out = []
            for b0 in range_constexpr(0, len(offs), batch):
                out += poll(offs[b0:b0 + batch], width)
            return out

        # ---------------------------------------------------------------- MXFP8 helpers
        def e8m0(amax):
            amax = fx.max(amax, fx.Float32(1e-10))
            bits = (amax * fx.Int32(0x3B124925).bitcast(fx.Float32)).bitcast(fx.Int32)
            e = (bits >> fx.Int32(23)) & fx.Int32(0xFF)
            e = fx.min(e + ((bits & fx.Int32(0x7FFFFF)) != fx.Int32(0)).select(fx.Int32(1), fx.Int32(0)), fx.Int32(255))
            return e, ((fx.Int32(254) - e) << fx.Int32(23)).bitcast(fx.Float32)

        def q8(v, inv):
            return fx.min(fx.max(v * inv, fx.Float32(-448.0)), fx.Float32(448.0))

        def pack4(q):
            w = rocdl.cvt_pk_fp8_f32(T.i32, q[0], q[1], fx.Int32(0), False)
            return fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q[2], q[3], w, True))

        def quant32_lds(vals, row, sc, k, act):
            """32 f32 (one thread) -> 8 E4M3 words in LDS act (row, group sc); returns E8M0."""
            amax = fx.Float32(0.0)
            for v in vals:
                amax = fx.max(amax, fmath.absf(v))
            e, inv = e8m0(amax)
            for w in range_constexpr(8):
                fx.ptr_store(pack4([q8(vals[w * 4 + i], inv) for i in range(4)]).bitcast(fx.Float32), act + (row * k + sc * 32) // 4 + w)
            return e

        def stage_fp8(wq, ws, k, act, scl):
            """Poll an E4M3 [8, k] activation (word pairs at wq, scale pairs at ws) into LDS: row-major words +
            the preshuffled scale words (k256 * 64 + col_lane * 16 + row = e(8 k256 + col_lane) | e(+4) << 16)."""
            nwp = ROWS * k // 4 // 2 // THREADS                  # two-pair loads per thread
            idx = [tid + i * THREADS for i in range(nwp)]
            got = poll_many([wq + 2 * p2 for p2 in idx], 2, 17)
            for j in range_constexpr(nwp):
                fx.ptr_store(fx.Vector.from_elements([got[2 * j], got[2 * j + 1]], fx.Int32).bitcast(fx.Float32), act + idx[j] * 2)
            units = ROWS * (k // 256) * 4
            per_u = (units + THREADS - 1) // THREADS
            us = [fx.min(tid + it * THREADS, units - 1) for it in range(per_u)]
            offs = []
            for u in us:
                row = u % ROWS
                rest = u // ROWS
                for gg in range(2):
                    offs.append(ws + row * (k // 32) + (rest // 4) * 8 + rest % 4 + gg * 4)
            got = poll_many(offs, 1)
            for it in range_constexpr(per_u):
                u = us[it]
                row = u % ROWS
                rest = u // ROWS
                fx.ptr_store(got[2 * it] | (got[2 * it + 1] << fx.Int32(16)), fx.recast_iter(fx.Int32, scl) + (rest // 4) * 64 + (rest % 4) * 16 + row)

        # ---------------------------------------------------------------- GEMM machinery
        def w_unit(r_w, r_s, t, k128, kk):
            """One 128-k step of a 16-row tile: (scale word, two 16 B halves) -- 9 VGPRs per lane."""
            kc = kk // 64
            sw = fx.Int32(bo.buffer_load(r_s, ((t // 2) * (kk // 256) + k128 // 2) * 64 + scale_lane, vec_width=1, dtype=T.i32))
            sw = ((t % 2) != 0).select(sw.shrui(fx.Int32(8)), sw)
            wh = [fx.Vector(bo.buffer_load(r_w, (((t * kc + k128 * 2 + q) * 4 + lane_div16) * 16 + lane_mod16) * 4,
                                           vec_width=4, dtype=T.i32)) for q in range(2)]
            return sw, wh

        def a_frag(act, kk, k128, live):
            k_base = k128 * 128 + lane_div16 * 16
            ok = valid_row & live
            halves = []
            for q in range_constexpr(2):
                ld = fx.Vector(fx.ptr_load(act + (input_row * kk + k_base + q * 64) // 4, result_type=v4f)).bitcast(fx.Int32)
                halves.append(fx.Vector.from_elements([ok.select(ld[i], fx.Int32(0)) for i in range(4)], fx.Int32))
            af = fx.make_rmem_tensor(8, fx.Int32)
            af.store(halves[0].shuffle(halves[1], list(range(8))))
            return af

        def a_scale(scl, k256):
            s = fx.ptr_load(fx.recast_iter(fx.Int32, scl) + k256 * 64 + scale_lane)
            return valid_row.select(s, fx.Int32(0x7F007F))

        def mfma(acc, af, unit, h, s_in):
            sw, wh = unit
            wf = fx.make_rmem_tensor(8, fx.Int32)
            wf.store(wh[0].shuffle(wh[1], list(range(8))))
            fx.gemm(atoms[h], acc, af, wf, acc, scale_a=s_in, scale_b=sw)

        def new_acc():
            acc = fx.make_rmem_tensor(4, fx.Float32)
            acc.store(fx.Vector.filled(4, 0.0, fx.Float32))
            return acc

        def gemm_geom(n, kk, r_w, r_s, ofs, t0=0):
            """Tile geometry of a GEMM stage (n output columns, K = kk; tiles t0 .. t0 + n / 16 of the weight):
            see the module docstring."""
            G = dict(n=n, kk=kk, r_w=r_w, r_s=r_s, nk=kk // 256, ntiles=n // 16, t0=t0)
            G['lb'] = (bid + (BLOCKS - ofs)) % BLOCKS
            G['cpw1'] = (G['nk'] + 7) // 8
            if const_expr(G['ntiles'] > BLOCKS):
                G['two'] = G['lb'] < (G['ntiles'] - BLOCKS)
                G['t'] = G['two'].select(G['lb'] + (wave // 4) * BLOCKS, G['lb'])
                G['cpw2'] = (G['nk'] + 3) // 4
                G['c_first'] = G['two'].select((wave % 4) * G['cpw2'], wave * G['cpw1'])
                G['active'] = bid >= 0
            else:
                G['two'] = None
                G['t'] = fx.min(G['lb'], G['ntiles'] - 1)
                G['c_first'] = wave * G['cpw1']
                G['active'] = G['lb'] < G['ntiles']
            return G

        def gemm_unit(G, u):                                     # u: this wave's 128-k step index
            k128 = fx.min(2 * G['c_first'] + u, 2 * G['nk'] - 1)
            return w_unit(G['r_w'], G['r_s'], G['t'] + G['t0'], k128, G['kk'])

        def gemm_prefetch(G, n=None):
            return [gemm_unit(G, u) for u in range(pre if n is None else n)]

        def gemm_loop(G, pre_units, act, scl, cpw, acc):
            U = 2 * cpw
            q_ = list(pre_units) + [gemm_unit(G, u) for u in range(len(pre_units), min(depth, U))]
            for u in range_constexpr(U):
                if const_expr(u + depth < U and u + depth >= len(q_)):
                    q_.append(gemm_unit(G, u + depth))
                k128 = 2 * G['c_first'] + u
                live = k128 < 2 * G['nk']
                k128c = fx.min(k128, 2 * G['nk'] - 1)
                s_in = a_scale(scl, k128c // 2)
                mfma(acc, a_frag(act, G['kk'], k128c, live), q_[u], u % 2, s_in)

        def gemm_red(nw, w0, tile, epi):
            tot = [fx.Float32(0.0) for _ in range(4)]
            for j in range_constexpr(nw):
                v_ = fx.Vector(fx.ptr_load(smem + ((w0 + j) * 64 + lane) * 4, result_type=v4f))
                for e in range_constexpr(4):
                    tot[e] = tot[e] + v_[e]
            epi(tile, tot)

        def gemm_run(G, pre_units, act, scl, epi):
            """MFMAs, then (after a barrier: the arena is reused) the per-tile K reduction; epi(tile, vals)
            runs on the reducing wave with the MFMA output layout (rows lane_div16 * 4 + e)."""
            acc = new_acc()
            if G['active']:
                if const_expr(G['two'] is not None):
                    if G['two']:
                        gemm_loop(G, pre_units, act, scl, G['cpw2'], acc)
                    else:
                        gemm_loop(G, pre_units, act, scl, G['cpw1'], acc)
                else:
                    gemm_loop(G, pre_units, act, scl, G['cpw1'], acc)
            vals = acc.load()
            gpu.barrier()
            fx.ptr_store(vals, smem + (wave * 64 + lane) * 4)
            gpu.barrier()
            if G['active']:
                if const_expr(G['two'] is not None):
                    if G['two']:
                        if wave < 2:
                            gemm_red(4, wave * 4, G['lb'] + wave * BLOCKS + G['t0'], epi)
                    else:
                        if wave == 0:
                            gemm_red(8, 0, G['lb'] + G['t0'], epi)
                else:
                    if wave == 0:
                        gemm_red(8, 0, G['lb'] + G['t0'], epi)
            gpu.barrier()

        def epi_bf(mb_name, n, plain=None):
            """Publish a tile's 8 x 16 outputs as bf16 pairs (and optionally a plain bf16 copy)."""
            def f(t, vals):
                for e in range_constexpr(4):
                    v = vals[e]
                    nb = v.shuffle_xor(1, 64)
                    row = lane_div16 * 4 + e
                    if row < ROWS:
                        if const_expr(plain is not None):
                            bo.buffer_store(v.to(fx.BFloat16), plain, row * n + t * 16 + lane_mod16)
                        if lane_mod16 % 2 == 0:
                            put(MB[mb_name] + row * (n // 2) + (t * 16 + lane_mod16) // 2, [bfpair(v, nb)])
            return f

        # ---------------------------------------------------------------- row stages (P0, A7)
        RCOLS = H // 16                                          # 320 threads x 16 columns

        def load_bf16x16(r, elem_off):
            vals = []
            for c in range_constexpr(2):
                raw = fx.Vector(bo.buffer_load(r, elem_off // 2 + c * 4, vec_width=4, dtype=T.i32))
                vals += [v for v in raw.bitcast(fx.BFloat16).to(fx.Float32)]
            return vals

        def half_of(w, odd):
            a_, b_ = unpack(w)
            return odd.select(b_, a_)

        def row_norm(xin, res_prev, g, out_b, out_q, out_s, res_out):
            """CTA r, threads < 320 (16 columns each): res = bf16(xin + res_prev) (or xin),
            y = bf16(bf16(res / sqrt(mean(res^2) + eps)) * gamma); publish y as bf16 pairs + E4M3 words + E8M0
            scales (a 32-group = two adjacent lanes); res stored plainly (read back by the same thread in the
            next row stage)."""
            r = bid
            t16 = fx.min(tid, RCOLS - 1)
            c0 = t16 * 16
            active = tid < RCOLS
            res = xin
            if const_expr(res_prev is not None):
                res = [bf16r(xin[i] + res_prev[i]) for i in range(16)]
            ss = fx.Float32(0.0)
            for i in range_constexpr(16):
                ss = ss + res[i] * res[i]
            ss = active.select(ss, fx.Float32(0.0))
            for off in (32, 16, 8, 4, 2, 1):
                ss = xred(ss, off, lambda a_, b_: a_ + b_)
            if lane == 0:
                fx.ptr_store(ss, smem + RED_ROW + wave)
            gpu.barrier()
            tot = fx.Float32(0.0)
            for w in range_constexpr(WAVES):
                tot = tot + fx.ptr_load(smem + RED_ROW + w)
            rinv = fx.Float32(1.0) / fmath.sqrt(tot * (1.0 / H) + 1e-6)
            y = [bf16r(bf16r(res[i] * rinv) * g[i]) for i in range(16)]
            amax = fx.Float32(0.0)
            for v in y:
                amax = fx.max(amax, fmath.absf(v))
            amax = xred(amax, 1, fx.max)
            e, inv = e8m0(amax)
            if active:
                r_res = rsrc(res_out)
                for c in range_constexpr(2):
                    bo.buffer_store(fx.Vector.from_elements([bfpair(res[c * 8 + 2 * j], res[c * 8 + 2 * j + 1]) for j in range(4)], fx.Int32),
                                    r_res, (r * H + c0 + c * 8) // 2)
                for j in range_constexpr(4):
                    put(out_b + (r * H + c0) // 2 + 2 * j, [bfpair(y[4 * j], y[4 * j + 1]), bfpair(y[4 * j + 2], y[4 * j + 3])])
                for w in range_constexpr(2):
                    put(out_q + (r * H + c0) // 4 + 2 * w,
                        [pack4([q8(y[w * 8 + i], inv) for i in range(4)]), pack4([q8(y[w * 8 + 4 + i], inv) for i in range(4)])])
                if tid % 2 == 0:
                    put(out_s + r * (H // 32) + c0 // 32, [e])
            gpu.barrier()

        def load_bf32(r, elem_off):
            vals = []
            for c in range_constexpr(4):
                raw = fx.Vector(bo.buffer_load(r, elem_off // 2 + c * 4, vec_width=4, dtype=T.i32))
                vals += [v for v in raw.bitcast(fx.BFloat16).to(fx.Float32)]
            return vals

        def conv16(cur, prev, d0, d1, b0, b1, has_prev):
            """bf16((b0 + d0) cur + [row > 0] (b1 + d1) prev) over 16 columns (one conv group)."""
            return [bf16r((b0[i] + d0) * cur[i] + ((b1[i] + d1) * prev[i]) * has_prev) for i in range(16)]

        # ======================================================== P0: input norm (CTAs 0..7)
        if bid < ROWS:
            r = bid
            t16 = fx.min(tid, RCOLS - 1)
            c0 = t16 * 16
            g1 = load_bf16x16(rsrc(ptr(P_LN1)), c0)
            if const_expr(layer0):
                r_ids = rsrc(ptr(P_IDS))
                idv = fx.Vector(bo.buffer_load(r_ids, 0, vec_width=2, dtype=T.i32))
                tok = (r == 0).select(uniform(idv[0]), fx.Int32(MASK_ID))
                xin = load_bf16x16(rsrc(ptr(P_EMBED) + fx.Int64(tok) * fx.Int64(H * 2)), c0)
                row_norm(xin, None, g1, MB['xb'], MB['xq'], MB['xs'], ptr(P_RES))
            else:
                res_prev = load_bf16x16(rsrc(ptr(P_RES)), r * H + c0)
                r_m = rsrc(ptr(P_M))
                cur = load_bf16x16(r_m, r * H + c0)
                prv = load_bf16x16(r_m, fx.max(r - 1, 0) * H + c0)
                r_cm = rsrc(ptr(P_CM))
                odd = (t16 % 2) == 1
                d0 = half_of(fx.Int32(bo.buffer_load(r_cm, (r * NKP + 640 + t16) // 2, vec_width=1, dtype=T.i32)), odd)
                d1 = half_of(fx.Int32(bo.buffer_load(r_cm, (r * NKP + 960 + t16) // 2, vec_width=1, dtype=T.i32)), odd)
                r_b = rsrc(ptr(P_MBASE1_PREV))
                xin = conv16(cur, prv, d0, d1, load_bf16x16(r_b, c0), load_bf16x16(r_b, H + c0), (r > 0).select(fx.Float32(1.0), fx.Float32(0.0)))
                row_norm(xin, res_prev, g1, MB['xb'], MB['xq'], MB['xs'], ptr(P_RES))
            stamp(1)

        def conv_stage_run(x_mb, co_mb, base_ptr, out_q, out_s, ofs):
            """Conv-prepare once per (row, 256-column chunk) task (160 tasks, CTAs lb < 160): 64 threads x 4
            columns; x rows r, r - 1 and the co coefficients polled, base taps loaded first; E8M0 per 32 columns
            (xor over 8 lanes) -> E4M3 words + scales published (the GEMM CTAs stage them with stage_fp8)."""
            lb = (bid + (BLOCKS - ofs)) % BLOCKS
            if lb < ROWS * (H // 256):
                if tid < 64:
                    r = lb // (H // 256)
                    col = (lb % (H // 256)) * 256 + tid * 4
                    r_b = rsrc(base_ptr)
                    b0 = fx.Vector(bo.buffer_load(r_b, col // 2, vec_width=2, dtype=T.i32)).bitcast(fx.BFloat16).to(fx.Float32)
                    b1 = fx.Vector(bo.buffer_load(r_b, (H + col) // 2, vec_width=2, dtype=T.i32)).bitcast(fx.BFloat16).to(fx.Float32)
                    w_ = poll([x_mb + (r * H + col) // 2, x_mb + (fx.max(r - 1, 0) * H + col) // 2,
                               co_mb + r * (NKP // 2) + col // 32, co_mb + r * (NKP // 2) + 160 + col // 32], 2)
                    cur = list(unpack(w_[0])) + list(unpack(w_[1]))
                    prv = list(unpack(w_[2])) + list(unpack(w_[3]))
                    odd = ((col // 16) % 2) == 1
                    d0 = (odd).select(unpack(w_[4])[1], unpack(w_[4])[0])
                    d1 = (odd).select(unpack(w_[6])[1], unpack(w_[6])[0])
                    hp = (r > 0).select(fx.Float32(1.0), fx.Float32(0.0))
                    o = [bf16r((b0[i] + d0) * cur[i] + ((b1[i] + d1) * prv[i]) * hp) for i in range(4)]
                    amax = fx.max(fx.max(fmath.absf(o[0]), fmath.absf(o[1])), fx.max(fmath.absf(o[2]), fmath.absf(o[3])))
                    for off in (1, 2, 4):
                        amax = xred(amax, off, fx.max)
                    e, inv = e8m0(amax)
                    put(out_q + (r * H + col) // 4, [pack4([q8(v, inv) for v in o])])
                    if tid % 8 == 0:
                        put(out_s + r * (H // 32) + col // 32, [e])

        def kpconv(xq_mb, xs_mb, x_mb, r_w, r_s, base_ptr, out_q, out_s, ofs):
            """Kernel projection part 0 fused with conv-prepare: task j < 20 (CTA lb = j) computes tiles j (tap-0
            coefficients d0 of columns 256 j .. + 256) and 20 + j (tap 1, d1) -- waves 0-3 / 4-7, K split 4 ways,
            all weights issued before the operand wait -- then the conv of those 256 columns for all 8 rows
            (64 threads per row x 4 columns), E8M0 per 32 -> E4M3 words + scales published (as conv_stage_run)."""
            lb = (bid + (BLOCKS - ofs)) % BLOCKS
            if lb < 20:
                j = lb
                tt = (wave // 4) * 20 + j
                c0w = (wave % 4) * 5
                units = [w_unit(r_w, r_s, tt, 2 * c0w + u, H) for u in range(10)]
                row = tid // 64
                col = j * 256 + (tid % 64) * 4
                r_b = rsrc(base_ptr)
                b0 = fx.Vector(bo.buffer_load(r_b, col // 2, vec_width=2, dtype=T.i32)).bitcast(fx.BFloat16).to(fx.Float32)
                b1 = fx.Vector(bo.buffer_load(r_b, (H + col) // 2, vec_width=2, dtype=T.i32)).bitcast(fx.BFloat16).to(fx.Float32)
                stage_fp8(xq_mb, xs_mb, H, smem, smem + ROWS * H // 4)
                gpu.barrier()
                acc = new_acc()
                for u in range_constexpr(10):
                    k128 = 2 * c0w + u
                    mfma(acc, a_frag(smem, H, k128, k128 >= 0), units[u], u % 2, a_scale(smem + ROWS * H // 4, k128 // 2))
                vals = acc.load()
                gpu.barrier()
                fx.ptr_store(vals, smem + (wave * 64 + lane) * 4)
                gpu.barrier()
                if wave < 2:                                   # wave g: tile group g (waves 4g .. 4g + 3) -> D[g][row][16]
                    tot = [fx.Float32(0.0) for _ in range(4)]
                    for w_ in range_constexpr(4):
                        v_ = fx.Vector(fx.ptr_load(smem + ((wave * 4 + w_) * 64 + lane) * 4, result_type=v4f))
                        for e in range_constexpr(4):
                            tot[e] = tot[e] + v_[e]
                    for e in range_constexpr(4):
                        rr = lane_div16 * 4 + e
                        if rr < ROWS:
                            fx.ptr_store(bf16r(tot[e]), smem + 4096 + (wave * ROWS + rr) * 16 + lane_mod16)
                gpu.barrier()
                xw = poll([x_mb + (row * H + col) // 2, x_mb + (fx.max(row - 1, 0) * H + col) // 2], 2)
                cur = list(unpack(xw[0])) + list(unpack(xw[1]))
                prv = list(unpack(xw[2])) + list(unpack(xw[3]))
                cc = (tid % 64) * 4 // 16
                d0 = fx.ptr_load(smem + 4096 + row * 16 + cc)
                d1 = fx.ptr_load(smem + 4096 + (ROWS + row) * 16 + cc)
                hp = (row > 0).select(fx.Float32(1.0), fx.Float32(0.0))
                o = [bf16r((b0[i] + d0) * cur[i] + ((b1[i] + d1) * prv[i]) * hp) for i in range(4)]
                amax = fx.max(fx.max(fmath.absf(o[0]), fmath.absf(o[1])), fx.max(fmath.absf(o[2]), fmath.absf(o[3])))
                for off in (1, 2, 4):
                    amax = xred(amax, off, fx.max)
                e, inv = e8m0(amax)
                put(out_q + (row * H + col) // 4, [pack4([q8(v, inv) for v in o])])
                if tid % 8 == 0:
                    put(out_s + row * (H // 32) + col // 32, [e])
                gpu.barrier()

        # ======================================================== A1: attention kernel projection
        if const_expr(fuse_kp_a):   # part 1 tiles (40..79, conv-finish coefficients) as a GEMM; part 0 fused into the conv
            g_akp = gemm_geom(NKP // 2, H, rsrc(ptr(P_AKP_W)), rsrc(ptr(P_AKP_S)), 28, t0=NKP // 32)
        else:
            g_akp = gemm_geom(NKP, H, rsrc(ptr(P_AKP_W)), rsrc(ptr(P_AKP_S)), 8)
        pre_akp = gemm_prefetch(g_akp, 6)      # all of a wave's 3 chunks
        if g_akp['active']:
            stage_fp8(MB['xq'], MB['xs'], H, smem, smem + ROWS * H // 4)
        gpu.barrier()
        gemm_run(g_akp, pre_akp, smem, smem + ROWS * H // 4, epi_bf('co', NKP))
        if const_expr(fuse_kp_a):
            kpconv(MB['xq'], MB['xs'], MB['xb'], rsrc(ptr(P_AKP_W)), rsrc(ptr(P_AKP_S)), ptr(P_ABASE0), MB['cv'], MB['cvs'], 8)
        stamp(2)

        # ======================================================== A3: conv-prepare + QKV
        g_qkv = gemm_geom(NQKV, H, rsrc(ptr(P_QKV_W)), rsrc(ptr(P_QKV_S)), 0)
        pre_qkv = gemm_prefetch(g_qkv, pre_big)

        def conv_prepare_lds(x_mb, co_mb, base_ptr, co_lds, act, scl, x_lds, e_lds):
            """Thread (sc = tid % 160, rows 4 (tid // 160) .. + 4), threads < 320: its 32-column group's base taps
            are loaded first; x [8, 5120] (bf16 pairs, one bulk poll) -> LDS words; co part 0 (cols 0..639)
            -> LDS f32; conv-prepare of its 4 rows -> E4M3 into act, E8M0 codes -> e_lds [8][160]; then the
            preshuffled scale words (k256 * 64 + col_lane * 16 + row = e(8 k256 + col_lane) | e(+4) << 16)."""
            sc = tid % 160
            rh = fx.min(tid // 160, 1)
            cact = tid < 320
            r_b = rsrc(base_ptr)
            bb0 = load_bf32(r_b, sc * 32)
            bb1 = load_bf32(r_b, H + sc * 32)
            nwp = ROWS * H // 2 // 2 // THREADS                  # two-pair loads per thread: 20
            idx = [tid + i * THREADS for i in range(nwp)]
            n_pairs = ROWS * 320
            per = (n_pairs + THREADS - 1) // THREADS
            ps = [fx.min(tid + i * THREADS, n_pairs - 1) for i in range(per)]
            got = poll([co_mb + (p // 320) * (NKP // 2) + p % 320 for p in ps], 1)
            for i in range_constexpr(per):
                p = ps[i]
                a_, b_ = unpack(got[i])
                fx.ptr_store(a_, co_lds + (p // 320) * 640 + 2 * (p % 320))
                fx.ptr_store(b_, co_lds + (p // 320) * 640 + 2 * (p % 320) + 1)
            gx = poll_many([x_mb + 2 * p2 for p2 in idx], 2, 10)
            for j in range_constexpr(nwp):
                fx.ptr_store(fx.Vector.from_elements([gx[2 * j], gx[2 * j + 1]], fx.Int32).bitcast(fx.Float32), x_lds + idx[j] * 2)
            gpu.barrier()

            def x32(row):
                vals = []
                for c in range_constexpr(4):
                    w4 = fx.Vector(fx.ptr_load(x_lds + (row * H + sc * 32) // 2 + c * 4, result_type=v4f)).bitcast(fx.Int32)
                    for e in range_constexpr(4):
                        a_, b_ = unpack(w4[e])
                        vals += [a_, b_]
                return vals
            if cact:
                prev = x32(fx.max(rh * 4 - 1, 0))
                for rr in range_constexpr(4):
                    row = rh * 4 + rr
                    cur = x32(row)
                    d0 = [fx.ptr_load(co_lds + row * 640 + sc * 2 + h) for h in range(2)]
                    d1 = [fx.ptr_load(co_lds + row * 640 + 320 + sc * 2 + h) for h in range(2)]
                    hp = (row > 0).select(fx.Float32(1.0), fx.Float32(0.0))
                    o = [bf16r((bb0[i] + d0[i // 16]) * cur[i] + ((bb1[i] + d1[i // 16]) * prev[i]) * hp) for i in range(32)]
                    fx.ptr_store(quant32_lds(o, row, sc, H, act), fx.recast_iter(fx.Int32, e_lds) + row * 160 + sc)
                    prev = cur
            gpu.barrier()
            units = ROWS * (H // 256) * 4
            for it in range_constexpr((units + THREADS - 1) // THREADS):
                u = tid + it * THREADS
                if u < units:
                    row = u % ROWS
                    rest = u // ROWS
                    g0 = (rest // 4) * 8 + rest % 4
                    ei = fx.recast_iter(fx.Int32, e_lds)
                    fx.ptr_store(fx.ptr_load(ei + row * 160 + g0) | (fx.ptr_load(ei + row * 160 + g0 + 4) << fx.Int32(16)),
                                 fx.recast_iter(fx.Int32, scl) + (rest // 4) * 64 + (rest % 4) * 16 + row)
            gpu.barrier()

        if const_expr(conv_stage):
            if const_expr(not fuse_kp_a):
                conv_stage_run(MB['xb'], MB['co'], ptr(P_ABASE0), MB['cv'], MB['cvs'], 88)
            stage_fp8(MB['cv'], MB['cvs'], H, smem + Q_ACT, smem + Q_SCL)
            gpu.barrier()
        else:
            conv_prepare_lds(MB['xb'], MB['co'], ptr(P_ABASE0), smem + Q_CO, smem + Q_ACT, smem + Q_SCL, smem + Q_X, smem + Q_E)
        stamp(11)
        gemm_run(g_qkv, pre_qkv, smem + Q_ACT, smem + Q_SCL, epi_bf('qkv', NQKV))
        stamp(3)

        # ======================================================== A5: q / k norm + RoPE + attention partials
        a_lb = (bid + (BLOCKS - 128)) % BLOCKS                    # CTAs 128..255
        if a_lb < 128:
            g = a_lb // SPLITS
            s = a_lb % SPLITS
            r_cos = rsrc(ptr(P_COS))
            r_sin = rsrc(ptr(P_SIN))

            def rope_val(x, partner, pos, d):
                """neox RoPE of one element: d < 64: x c - partner s; else x c + partner s (tables [P, 128])."""
                c = fx.Float32(bo.buffer_load(r_cos, pos * HD + d, vec_width=1, dtype=T.f32))
                sn = fx.Float32(bo.buffer_load(r_sin, pos * HD + d, vec_width=1, dtype=T.f32))
                sgn = (d < 64).select(fx.Float32(-1.0), fx.Float32(1.0))
                return x * c + sgn * partner * sn

            # this split's cached keys / values: issued first (they do not depend on this launch)
            r_kc = rsrc(ptr(P_KC))
            r_vc = rsrc(ptr(P_VC))
            nfresh = (s == SPLITS - 1).select(fx.Int32(8), fx.Int32(0))
            kv_pre = []
            for it in range_constexpr(KPS * 16 // THREADS):     # 128 keys x 16 four-word pieces
                pidx = tid + it * THREADS
                j = pidx // 16
                wq = (pidx % 16) * 4
                kp = lo + s * KPS + j
                ok = (kp >= 0) & (j < KPS - nfresh)
                kpc = fx.max(kp, 0)
                for r_c in (r_kc, r_vc):
                    v = fx.Vector(bo.buffer_load(r_c, ((g * P + kpc) * HD) // 2 + wq, vec_width=4, dtype=T.i32))
                    kv_pre.append(fx.Vector.from_elements([ok.select(v[e], fx.Int32(0)) for e in range(4)], fx.Int32))
            # q heads 4g .. 4g+3, rows 0..7: thread -> row = tid // 64, 8 values at q col (tid % 64) * 8
            qrow = tid // 64
            qc = (tid % 64) * 8                                  # within the 512 q columns of this group
            qw = poll([MB['qkv'] + (qrow * NQKV + g * 512 + qc) // 2 + 2 * j for j in range(2)], 2)
            xs_ = []
            for j in range_constexpr(4):
                a_, b_ = unpack(qw[j])
                xs_ += [a_, b_]
            ss = fx.Float32(0.0)
            for v in xs_:
                ss = ss + v * v
            for off in (1, 2, 4, 8):
                ss = xred(ss, off, lambda a_, b_: a_ + b_)
            inv = fx.Float32(1.0) / (fmath.sqrt(ss * (1.0 / HD) + 1e-6))
            d0_ = qc % HD
            r_qn = rsrc(ptr(P_QN))
            qn = fx.Vector(bo.buffer_load(r_qn, d0_ // 2, vec_width=4, dtype=T.i32)).bitcast(fx.BFloat16).to(fx.Float32)
            xn = [bf16r(bf16r(xs_[i] * inv) * qn[i]) for i in range(8)]
            pq = [xshfl(v, 8) for v in xn]                       # partner d +- 64
            pos = base + qrow
            qv = [bf16r(rope_val(xn[i], pq[i], pos, d0_ + i)) for i in range(8)]
            qi = qrow * 4 + qc // HD
            for j in range_constexpr(4):
                fx.ptr_store(bfpair(qv[2 * j], qv[2 * j + 1]).bitcast(fx.Float32), smem + A_Q + qi * QS + d0_ // 2 + j)
            # the block's own keys (split 15): k head g (norm + RoPE) and v head g, rows 0..7
            if s == SPLITS - 1:
                kind = tid // 256                                # 0: k, 1: v
                krow = (tid % 256) // 32
                kd = (tid % 32) * 4
                kw = poll([MB['qkv'] + (krow * NQKV + NO + kind * 1024 + g * HD + kd) // 2 + j for j in range(2)], 1)
                kv_ = []
                for j in range_constexpr(2):
                    a_, b_ = unpack(kw[j])
                    kv_ += [a_, b_]
                ss = fx.Float32(0.0)
                for v in kv_:
                    ss = ss + v * v
                for off in (1, 2, 4, 8, 16):
                    ss = xred(ss, off, lambda a_, b_: a_ + b_)
                inv = fx.Float32(1.0) / (fmath.sqrt(ss * (1.0 / HD) + 1e-6))
                r_kn = rsrc(ptr(P_KN))
                kn = fx.Vector(bo.buffer_load(r_kn, kd // 2, vec_width=2, dtype=T.i32)).bitcast(fx.BFloat16).to(fx.Float32)
                kn_ = [bf16r(bf16r(kv_[i] * inv) * kn[i]) for i in range(4)]
                pk = [v.shuffle_xor(16, 64) for v in kn_]
                kpos = base + krow
                kr = [bf16r(rope_val(kn_[i], pk[i], kpos, kd + i)) for i in range(4)]
                outv = [(kind == 0).select(kr[i], kv_[i]) for i in range(4)]
                words = [bfpair(outv[0], outv[1]), bfpair(outv[2], outv[3])]
                cache = (kind == 0).select(ptr(P_KC), ptr(P_VC))
                bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), rsrc(cache), ((g * P + kpos) * HD + kd) // 2)
                lds_t = (kind == 0).select(fx.Int32(A_K), fx.Int32(A_V))
                j_ = KPS - 8 + krow
                for j in range_constexpr(2):
                    fx.ptr_store(words[j].bitcast(fx.Float32), smem + lds_t + j_ * KS + kd // 2 + j)
            # the split's cached keys / values (loaded before the q wait) -> LDS (K and V [128 keys][KS words])
            for it in range_constexpr(KPS * 16 // THREADS):
                j = (tid + it * THREADS) // 16
                wq = ((tid + it * THREADS) % 16) * 4
                for kind_, dst in ((0, A_K), (1, A_V)):
                    if j < KPS - nfresh:
                        fx.ptr_store(kv_pre[it * 2 + kind_].bitcast(fx.Float32), smem + dst + j * KS + wq)
            gpu.barrier()
            # scores S[q][key] = Q K^T * scale: wave w -> key tile w, both query tiles
            for qt in range_constexpr(2):
                c = fx.Vector.filled(4, 0.0, fx.Float32)
                for kst in range_constexpr(HD // 32):
                    a = fx.ptr_load(smem + A_K + (wave * 16 + lane_mod16) * KS + kst * 16 + lane_div16 * 4, result_type=v4f).bitcast(fx.BFloat16)
                    b = fx.ptr_load(smem + A_Q + (qt * 16 + lane_mod16) * QS + kst * 16 + lane_div16 * 4, result_type=v4f).bitcast(fx.BFloat16)
                    c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
                for e in range_constexpr(4):
                    fx.ptr_store(c[e], smem + A_S + (qt * 16 + lane_mod16) * SS + wave * 16 + lane_div16 * 4 + e)
            gpu.barrier()
            # softmax over the split: thread -> query tid // 16, keys (tid % 16) * 8 .. + 8
            sq = tid // 16
            kb = (tid % 16) * 8
            sv = []
            for i in range_constexpr(8):
                kp = lo + s * KPS + kb + i
                raw = fx.ptr_load(smem + A_S + sq * SS + kb + i) * 0.08838834764831845
                sv.append((kp >= 0).select(raw, fx.Float32(-1.0e30)))
            m = sv[0]
            for v in sv[1:]:
                m = fx.max(m, v)
            for off in (1, 2, 4, 8):
                m = xred(m, off, fx.max)
            pv = [(sv[i] > -1.0e29).select(exp(sv[i] - m), fx.Float32(0.0)) for i in range(8)]
            l = fx.Float32(0.0)
            for v in pv:
                l = l + v
            for off in (1, 2, 4, 8):
                l = xred(l, off, lambda a_, b_: a_ + b_)
            for j in range_constexpr(4):
                fx.ptr_store(bfpair(pv[2 * j], pv[2 * j + 1]).bitcast(fx.Float32), smem + A_P + sq * PS + kb // 2 + j)
            if tid % 16 == 0:
                fx.ptr_store(m, smem + A_ML + sq)
                fx.ptr_store(l, smem + A_ML + 32 + sq)
            gpu.barrier()
            # O = P V: wave w -> dims w * 16 .. + 16, both query tiles; V gathered key-strided
            dcol = wave * 16 + lane_mod16
            task = g * SPLITS + s
            for qt in range_constexpr(2):
                c = fx.Vector.filled(4, 0.0, fx.Float32)
                for kst in range_constexpr(KPS // 32):
                    a = fx.ptr_load(smem + A_P + (qt * 16 + lane_mod16) * PS + kst * 16 + lane_div16 * 4, result_type=v4f).bitcast(fx.BFloat16)
                    ws_ = [fx.ptr_load(smem + A_V + (kst * 32 + lane_div16 * 8 + i) * KS + dcol // 2).bitcast(fx.Int32) for i in range(8)]
                    odd = (dcol % 2) == 1
                    hv = [odd.select(fx.Int32(fx.Uint32(w_) >> 16), w_ & fx.Int32(0xFFFF)) for w_ in ws_]
                    bw = [hv[2 * i] | (hv[2 * i + 1] << fx.Int32(16)) for i in range(4)]
                    b = fx.Vector.from_elements(bw, fx.Int32).bitcast(fx.BFloat16)
                    c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
                for e in range_constexpr(4):
                    q = qt * 16 + lane_div16 * 4 + e
                    put(MB['ap'] + (task * 32 + q) * 130 + dcol, [c[e].bitcast(fx.Int32)])
            if tid < 32:
                put(MB['ap'] + (task * 32 + tid) * 130 + 128,
                    [fx.ptr_load(smem + A_ML + tid).bitcast(fx.Int32), fx.ptr_load(smem + A_ML + 32 + tid).bitcast(fx.Int32)])
            gpu.barrier()
            stamp(4)

        # ======================================================== A5b: split combine -> MXFP8 o (64 tasks: CTAs 0..63)
        if bid < 64:
            g = bid // ROWS
            r = bid % ROWS
            hh = tid // HD
            d = tid % HD
            q = r * 4 + hh
            got = poll_many([MB['ap'] + ((g * SPLITS + s_) * 32 + q) * 130 + d for s_ in range(SPLITS)], 1)
            mls = poll_many([MB['ap'] + ((g * SPLITS + s_) * 32 + q) * 130 + 128 for s_ in range(SPLITS)], 2, 8)
            ms = [mls[2 * s_].bitcast(fx.Float32) for s_ in range(SPLITS)]
            ls = [mls[2 * s_ + 1].bitcast(fx.Float32) for s_ in range(SPLITS)]
            mx = ms[0]
            for v in ms[1:]:
                mx = fx.max(mx, v)
            num = fx.Float32(0.0)
            den = fx.Float32(0.0)
            for s_ in range_constexpr(SPLITS):
                w_ = (ls[s_] > 0.0).select(exp(ms[s_] - mx), fx.Float32(0.0))
                num = num + w_ * got[s_].bitcast(fx.Float32)
                den = den + w_ * ls[s_]
            o = bf16r(num / den)
            amax = fmath.absf(o)
            for off in (1, 2, 4, 8, 16):
                amax = xred(amax, off, fx.max)
            e, inv = e8m0(amax)
            qv = q8(o, inv)
            word = pack4([qv, qv.shuffle_xor(1, 64), qv.shuffle_xor(2, 64), qv.shuffle_xor(3, 64)])
            col = (g * 4 + hh) * HD + d
            if lane % 4 == 0:
                put(MB['oq'] + r * (NO // 4) + col // 4, [word])
            if lane % 32 == 0:
                put(MB['os'] + r * (NO // 32) + col // 32, [e])
            stamp(5)

        # ======================================================== A6: o_proj
        g_o = gemm_geom(H, NO, rsrc(ptr(P_O_W)), rsrc(ptr(P_O_S)), 64)
        pre_o = gemm_prefetch(g_o, pre_big)
        stage_fp8(MB['oq'], MB['os'], NO, smem, smem + ROWS * NO // 4)
        gpu.barrier()
        stamp(14)
        gemm_run(g_o, pre_o, smem, smem + ROWS * NO // 4, epi_bf('ab', H))
        stamp(6)

        # ======================================================== A7: conv-finish + residual + post-attention norm (CTAs 0..7)
        if bid < ROWS:
            r = bid
            t16 = fx.min(tid, RCOLS - 1)
            c0 = t16 * 16
            g2 = load_bf16x16(rsrc(ptr(P_LN2)), c0)
            res_prev = load_bf16x16(rsrc(ptr(P_RES)), r * H + c0)
            r_b = rsrc(ptr(P_ABASE1))
            b0_, b1_ = load_bf16x16(r_b, c0), load_bf16x16(r_b, H + c0)
            odd = (t16 % 2) == 1
            aw = poll([MB['ab'] + (r * H + c0) // 2 + 2 * j for j in range(4)] +
                      [MB['ab'] + (fx.max(r - 1, 0) * H + c0) // 2 + 2 * j for j in range(4)], 2)
            cw = poll([MB['co'] + r * (NKP // 2) + 320 + t16 // 2, MB['co'] + r * (NKP // 2) + 480 + t16 // 2], 1)   # (single pairs:
            aw = aw + [cw[0], cw[0], cw[1], cw[1]]                                  #  part 0 may be unpublished)
            cur, prv = [], []
            for j in range_constexpr(8):
                a_, b_ = unpack(aw[j])
                cur += [a_, b_]
                a_, b_ = unpack(aw[8 + j])
                prv += [a_, b_]
            d0 = half_of(aw[16], odd)
            d1 = half_of(aw[18], odd)
            xin = conv16(cur, prv, d0, d1, b0_, b1_, (r > 0).select(fx.Float32(1.0), fx.Float32(0.0)))
            row_norm(xin, res_prev, g2, MB['x2b'], MB['x2q'], MB['x2s'], ptr(P_RES))
            stamp(7)

        # ======================================================== M1: mlp kernel projection
        if const_expr(fuse_kp_m):
            g_mkp = gemm_geom(NKP // 2, H, rsrc(ptr(P_MKP_W)), rsrc(ptr(P_MKP_S)), 28, t0=NKP // 32)
        else:
            g_mkp = gemm_geom(NKP, H, rsrc(ptr(P_MKP_W)), rsrc(ptr(P_MKP_S)), 8)
        pre_mkp = gemm_prefetch(g_mkp, 6)
        if g_mkp['active']:
            stage_fp8(MB['x2q'], MB['x2s'], H, smem, smem + ROWS * H // 4)
        gpu.barrier()
        gemm_run(g_mkp, pre_mkp, smem, smem + ROWS * H // 4, epi_bf('cm', NKP, plain=rsrc(ptr(P_CM_OUT))))
        if const_expr(fuse_kp_m):
            kpconv(MB['x2q'], MB['x2s'], MB['x2b'], rsrc(ptr(P_MKP_W)), rsrc(ptr(P_MKP_S)), ptr(P_MBASE0), MB['cv2'], MB['cv2s'], 8)
        stamp(8)

        # ======================================================== M3: conv-prepare + gate / up + SiLU + MXFP8 (544 tasks)
        GU_TASKS = NI // 32
        gu_lb = (bid + (BLOCKS - 96)) % BLOCKS
        slot = wave // 2
        kh = wave % 2
        gtask = gu_lb + slot * BLOCKS
        glive = (slot < 3) & (gtask < GU_TASKS)
        ga = fx.min(gtask, GU_TASKS - 1)
        gtiles = [2 * ga, 2 * ga + 1, NI // 16 + 2 * ga, NI // 16 + 2 * ga + 1]
        r_wg = rsrc(ptr(P_GU_W))
        r_sg = rsrc(ptr(P_GU_S))
        KH = H // 256 // 2                                      # 10 chunks per K half
        U2 = 2 * KH
        c0g = kh * KH
        pre_g = [[w_unit(r_wg, r_sg, tt, 2 * c0g + u, H) for tt in gtiles] for u in range(pre_gu)]
        if const_expr(conv_stage):
            if const_expr(not fuse_kp_m):
                conv_stage_run(MB['x2b'], MB['cm'], ptr(P_MBASE0), MB['cv2'], MB['cv2s'], 88)
            stage_fp8(MB['cv2'], MB['cv2s'], H, smem + G_ACT, smem + G_SCL)
            gpu.barrier()
        else:
            conv_prepare_lds(MB['x2b'], MB['cm'], ptr(P_MBASE0), smem + G_CM, smem + G_ACT, smem + G_SCL, smem + G_X, smem + G_E)
        stamp(12)
        accs = [new_acc() for _ in range(4)]
        if glive:
            q_ = list(pre_g) + [[w_unit(r_wg, r_sg, tt, 2 * c0g + u, H) for tt in gtiles] for u in range(pre_gu, min(depth_gu, U2))]
            for u in range_constexpr(U2):
                if const_expr(u + depth_gu < U2 and u + depth_gu >= len(q_)):
                    q_.append([w_unit(r_wg, r_sg, tt, 2 * c0g + u + depth_gu, H) for tt in gtiles])
                k128 = 2 * c0g + u
                s_in = a_scale(smem + G_SCL, k128 // 2)
                af = a_frag(smem + G_ACT, H, k128, k128 >= 0)
                for j in range_constexpr(4):
                    mfma(accs[j], af, q_[u][j], u % 2, s_in)
            if kh == 1:
                for j in range_constexpr(4):
                    fx.ptr_store(accs[j].load(), smem + G_PART + ((slot * 4 + j) * 64 + lane) * 4)
        gpu.barrier()
        if glive & (kh == 0):
            tot = []
            for j in range_constexpr(4):
                v_ = accs[j].load()
                o_ = fx.Vector(fx.ptr_load(smem + G_PART + ((slot * 4 + j) * 64 + lane) * 4, result_type=v4f))
                tot.append([v_[e] + o_[e] for e in range(4)])
            for e in range_constexpr(4):
                row = lane_div16 * 4 + e
                vs = []
                for hh in range_constexpr(2):
                    gv = bf16r(tot[hh][e])
                    uv = bf16r(tot[2 + hh][e])
                    s_ = bf16r(gv / (fx.Float32(1.0) + fmath.exp(fx.Float32(0.0) - gv)))
                    vs.append(bf16r(s_ * uv))
                amax = fx.max(fmath.absf(vs[0]), fmath.absf(vs[1]))
                for off in (1, 2, 4, 8):
                    amax = xred(amax, off, fx.max)
                ex, inv = e8m0(amax)
                for hh in range_constexpr(2):
                    qv = q8(vs[hh], inv)
                    word = pack4([qv, qv.shuffle_xor(1, 64), qv.shuffle_xor(2, 64), qv.shuffle_xor(3, 64)])
                    if (row < ROWS) & (lane_mod16 % 4 == 0):
                        put(MB['act'] + row * (NI // 4) + ga * 8 + hh * 4 + lane_mod16 // 4, [word])
                if (row < ROWS) & (lane_mod16 == 0):
                    put(MB['acts'] + row * GU_TASKS + ga, [ex])
        gpu.barrier()
        stamp(9)

        # ======================================================== M4: down projection -> m (plain)
        g_d = gemm_geom(H, NI, rsrc(ptr(P_D_W)), rsrc(ptr(P_D_S)), 128)
        pre_dn = gemm_prefetch(g_d, pre_d)
        stage_fp8(MB['act'], MB['acts'], NI, smem, smem + ROWS * NI // 4)
        gpu.barrier()
        stamp(13)
        r_mo = rsrc(ptr(P_M_OUT))

        def epi_m(t, vals):
            for e in range_constexpr(4):
                v = vals[e]
                row = lane_div16 * 4 + e
                if const_expr(final_norm):
                    nb = v.shuffle_xor(1, 64)
                if row < ROWS:
                    bo.buffer_store(v.to(fx.BFloat16), r_mo, row * H + t * 16 + lane_mod16)
                    if const_expr(final_norm):
                        if lane_mod16 % 2 == 0:
                            put(MB['mb'] + row * (H // 2) + (t * 16 + lane_mod16) // 2, [bfpair(v, nb)])
        gemm_run(g_d, pre_dn, smem, smem + ROWS * NI // 4, epi_m)
        stamp(10)

        # ======================================================== F: final norm -> final (bf16) + the LM head's MXFP8 operand
        if const_expr(final_norm):
            if bid < ROWS:
                r = bid
                t16 = fx.min(tid, RCOLS - 1)
                c0 = t16 * 16
                active = tid < RCOLS
                gN = load_bf16x16(rsrc(ptr(P_NORM)), c0)
                res_prev = load_bf16x16(rsrc(ptr(P_RES)), r * H + c0)
                r_b = rsrc(ptr(P_MBASE1))
                bp0, bp1 = load_bf16x16(r_b, c0), load_bf16x16(r_b, H + c0)
                odd = (t16 % 2) == 1
                mw = poll([MB['mb'] + (r * H + c0) // 2 + 2 * j for j in range(4)] +
                          [MB['mb'] + (fx.max(r - 1, 0) * H + c0) // 2 + 2 * j for j in range(4)], 2)
                cw = poll([MB['cm'] + r * (NKP // 2) + 320 + t16 // 2, MB['cm'] + r * (NKP // 2) + 480 + t16 // 2], 1)
                mw = mw + [cw[0], cw[0], cw[1], cw[1]]
                cur, prv = [], []
                for j in range_constexpr(8):
                    a_, b_ = unpack(mw[j])
                    cur += [a_, b_]
                    a_, b_ = unpack(mw[8 + j])
                    prv += [a_, b_]
                xin = conv16(cur, prv, half_of(mw[16], odd), half_of(mw[18], odd), bp0, bp1, (r > 0).select(fx.Float32(1.0), fx.Float32(0.0)))
                res = [bf16r(xin[i] + res_prev[i]) for i in range(16)]
                ss = fx.Float32(0.0)
                for i in range_constexpr(16):
                    ss = ss + res[i] * res[i]
                ss = active.select(ss, fx.Float32(0.0))
                for off in (32, 16, 8, 4, 2, 1):
                    ss = xred(ss, off, lambda a_, b_: a_ + b_)
                if lane == 0:
                    fx.ptr_store(ss, smem + RED_ROW + wave)
                gpu.barrier()
                tot = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    tot = tot + fx.ptr_load(smem + RED_ROW + w)
                rinv = fx.Float32(1.0) / fmath.sqrt(tot * (1.0 / H) + 1e-6)
                y = [bf16r(bf16r(res[i] * rinv) * gN[i]) for i in range(16)]
                amax = fx.Float32(0.0)
                for v in y:
                    amax = fx.max(amax, fmath.absf(v))
                amax = xred(amax, 1, fx.max)
                e, inv = e8m0(amax)
                if active:
                    r_f = rsrc(ptr(P_FINAL))
                    for c in range_constexpr(2):
                        bo.buffer_store(fx.Vector.from_elements([bfpair(y[c * 8 + 2 * j], y[c * 8 + 2 * j + 1]) for j in range(4)], fx.Int32),
                                        r_f, (r * H + c0 + c * 8) // 2)
                    r_q = rsrc(ptr(P_LMQ))
                    bo.buffer_store(fx.Vector.from_elements([pack4([q8(y[w * 4 + i], inv) for i in range(4)]) for w in range(4)], fx.Int32),
                                    r_q, (r * H + c0) // 4)
                    if tid % 2 == 0:
                        sc = c0 // 32
                        boff = (sc >> 3) * 256 + (sc & 3) * 64 + r * 4 + ((sc >> 2) & 1) * 2
                        bo.buffer_store(e.to(fx.Uint8), rsrc(ptr(P_LMS)), boff, offset_is_bytes=True)
                gpu.barrier()
            stamp(15)

    @flyc.jit
    def launch(table: Int64, stream: Stream = Stream(None)):
        layer_kernel(table, value_attrs={"rocdl.flat_work_group_size": f"{THREADS},{THREADS}"}).launch(
            grid=(BLOCKS, 1, 1), block=(THREADS, 1, 1), stream=stream)

    launch.func.__name__ = f"layer560_{'l0' if layer0 else 'ln'}_P{P}_d{depth}_p{pre}_g{depth_gu}_{pre_gu}_b{pre_big}_dn{pre_d}{'_cs' if conv_stage else ''}{'_fn' if final_norm else ''}{'_kpa' if fuse_kp_a else ''}{'_kpm' if fuse_kp_m else ''}{'_tl' if timeline else ''}"
    return launch
