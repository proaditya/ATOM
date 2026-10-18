# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Triton kernels moving the model's state between ATOM's pools and the
megakernel's layout. Every one reads its sizes and slots from device tensors,
so a step's sequence of them is graph-capturable.

ATOM's layouts (``attentions.gdn_attn`` / ``mha_kv_pool``):

- conv state ``[layer, slot, state_len, 10240]`` bf16, the window of the
  current step at ``num_accepted - 1`` (the spec-decode rolling buffer of
  ``causal_conv1d_update``);
- recurrent state ``[layer, slot, 48, 128 v, 128 k]`` fp32, the current one in
  column ``num_accepted - 1`` of the request's slots;
- paged KV of 16-token blocks, K ``[block, head, 256 / 8, 16, 8]`` and V
  ``[block, head, 16 / 8, 256, 8]`` bf16.

The kernel's: conv history ``[layer][10240][4]`` (entries 1..3 the three most
recent inputs), KV pages of 32 positions ``[page][head][K | V][32][256]``.
"""

import triton
import triton.language as tl

from atom.models.qwen38_27b.mono import dims as d

# ---- conv ------------------------------------------------------------------


@triton.jit
def _conv_gather_kernel(
    conv,
    s_layer,
    s_slot,
    s_tok,
    slot_in,
    accepted,
    hist,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """hist[layer, ch, 1 + j] = conv[layer, slot_in, accepted - 1 + j, ch]."""
    layer = tl.program_id(0)
    ch = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(slot_in).to(tl.int64)
    off = tl.load(accepted).to(tl.int64) - 1
    src = conv + layer.to(tl.int64) * s_layer + slot * s_slot + ch
    dst = hist + (layer * DIM + ch).to(tl.int64) * 4
    for j in tl.static_range(3):
        tl.store(dst + 1 + j, tl.load(src + (off + j) * s_tok))


@triton.jit
def _conv_writeback_kernel(
    conv,
    s_layer,
    s_slot,
    s_tok,
    slot_out,
    hist,
    raw,
    s_raw_layer,
    rows,
    path,
    gate,
    gate2,
    DIM: tl.constexpr,
    ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """ATOM's state after a verify of ``rows`` inputs, whatever is accepted
    (``causal_conv1d_update``'s rolling buffer): [h2, h3, x_path[0] ..
    x_path[rows-1]] (the launch's rows in order).
    Nothing unless both gates are non-zero."""
    if (tl.load(gate) != 0) & (tl.load(gate2) != 0):
        layer = tl.program_id(0)
        ch = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        slot = tl.load(slot_out).to(tl.int64)
        n = tl.load(rows)
        dst = conv + layer.to(tl.int64) * s_layer + slot * s_slot + ch
        src = hist + (layer * DIM + ch).to(tl.int64) * 4
        tl.store(dst, tl.load(src + 2))
        tl.store(dst + s_tok, tl.load(src + 3))
        x = raw + layer.to(tl.int64) * s_raw_layer + ch
        for t in range(n):
            tl.store(dst + (2 + t) * s_tok, tl.load(x + tl.load(path + t) * ROW))


def conv_gather(conv, slot_in, accepted, hist) -> None:
    """``conv`` ATOM's conv pool, ``hist`` the kernel's [48, 10240 * 4]."""
    layers = conv.shape[0]
    _conv_gather_kernel[(layers, d.CONV_DIM // 1024)](
        conv, conv.stride(0), conv.stride(1), conv.stride(2), slot_in, accepted, hist,
        DIM=d.CONV_DIM, BLOCK=1024, num_warps=4,
    )  # fmt: skip


def conv_writeback(conv, slot_out, hist, raw, rows, path, gate, gate2) -> None:
    """``raw`` the kernel's pre-conv rows [48, rows * PROJECTED]; ``rows`` an
    int32 [1] device tensor, ``path`` the rows' order (int64); runs when both
    ``gate`` and ``gate2`` (int32 [1]) are non-zero."""
    layers = conv.shape[0]
    _conv_writeback_kernel[(layers, d.CONV_DIM // 1024)](
        conv, conv.stride(0), conv.stride(1), conv.stride(2), slot_out, hist, raw,
        raw.stride(0), rows, path, gate, gate2, DIM=d.CONV_DIM, ROW=d.PROJECTED, BLOCK=1024,
        num_warps=4,
    )  # fmt: skip


# ---- recurrent state ---------------------------------------------------------


@triton.jit
def _state_commit_kernel(
    pool,
    s_layer,
    s_slot,
    slot_src,
    slot_dst,
    rec,
    s_rec_layer,
    path,
    accepted,
    pending,
    MAX_ROWS: tl.constexpr,
    BV: tl.constexpr,
):
    """The recurrent state after the pending launch's first k = accepted rows
    along ``path`` (the rows in order): S <- d_t S +
    delta_t key_t^T, from the launch's initial state in ``slot_src`` into
    ``slot_dst``. One program a (layer, head, value block); nothing when no
    launch is pending."""
    if tl.load(pending) != 0:
        lh = tl.program_id(0)
        vb = tl.program_id(1)
        layer = lh // 48
        head = lh % 48
        k = tl.load(accepted)
        vs = vb * BV + tl.arange(0, BV)
        ks = tl.arange(0, 128)
        tile = head * 16384 + vs[:, None] * 128 + ks[None, :]
        base = pool + layer.to(tl.int64) * s_layer
        st = tl.load(base + tl.load(slot_src).to(tl.int64) * s_slot + tile)
        for t in tl.static_range(MAX_ROWS):
            if t < k:
                row = tl.load(path + t)
                r = rec + layer.to(tl.int64) * s_rec_layer + (row * 48 + head) * 260
                dk = tl.load(r + 256)
                dl = tl.load(r + vs)
                kk = tl.load(r + 128 + ks)
                st = dk * st + dl[:, None] * kk[None, :]
        tl.store(base + tl.load(slot_dst).to(tl.int64) * s_slot + tile, st)


def state_commit(
    pool, slot_src, slot_dst, rec, path, accepted, pending, max_rows
) -> None:
    """``pool`` ATOM's recurrent pool [48, slots, 48, 128, 128] fp32 (VK);
    ``rec`` the records [48, max_rows * 48 * 260]; ``path`` int64 [max_rows]."""
    layers = pool.shape[0]
    _state_commit_kernel[(layers * 48, 4)](
        pool, pool.stride(0), pool.stride(1), slot_src, slot_dst, rec,
        rec.stride(0), path, accepted, pending, MAX_ROWS=max_rows, BV=32, num_warps=4,
    )  # fmt: skip


@triton.jit
def _kv_import_kernel(
    k_ptrs,
    v_ptrs,
    table,
    start_pos,
    end_pos,
    mirror_ptrs,
    BS: tl.constexpr,
    X: tl.constexpr,
    HD: tl.constexpr,
    KVH: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    """Positions [start_pos, end_pos) of one (layer, head), ATOM's cache ->
    the mirror pages: each program walks every PROGRAMS-th block."""
    layer = tl.program_id(0)
    p = tl.program_id(1)
    h = tl.program_id(2)
    start = tl.load(start_pos)
    end = tl.load(end_pos)
    kc = tl.load(k_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
    vc = tl.load(v_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
    mc = tl.load(mirror_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
    t = tl.arange(0, BS)
    dd = tl.arange(0, HD)
    for blk_i in range(start // BS + p, (end + BS - 1) // BS, PROGRAMS):
        blk = tl.load(table + blk_i).to(tl.int64)
        pos = blk_i * BS + t
        ok = ((pos >= start) & (pos < end))[:, None]
        hb = blk * KVH + h
        kv_k = tl.load(
            kc
            + ((hb * (HD // X) + dd[None, :] // X) * BS + t[:, None]) * X
            + dd[None, :] % X,
            mask=ok,
        )
        kv_v = tl.load(
            vc
            + ((hb * (BS // X) + t[:, None] // X) * HD + dd[None, :]) * X
            + t[:, None] % X,
            mask=ok,
        )
        page = (pos // 32).to(tl.int64)
        dst = mc + ((page * KVH + h) * 2 * 32 + pos % 32)[:, None] * HD + dd[None, :]
        tl.store(dst, kv_k, mask=ok)
        tl.store(dst + 32 * HD, kv_v, mask=ok)


@triton.jit
def _kv_export_kernel(
    k_ptrs,
    v_ptrs,
    slots,
    base,
    rows,
    mirror_ptrs,
    BS: tl.constexpr,
    X: tl.constexpr,
    HD: tl.constexpr,
    KVH: tl.constexpr,
):
    """This launch's rows' K / V (mirror pages) -> ATOM's cache slots."""
    layer = tl.program_id(0)
    r = tl.program_id(1)
    h = tl.program_id(2)
    if r < tl.load(rows):
        slot = tl.load(slots + r).to(tl.int64)
        if slot >= 0:
            kc = tl.load(k_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
            vc = tl.load(v_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
            mc = tl.load(mirror_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
            pos = (tl.load(base) + r).to(tl.int64)
            dd = tl.arange(0, HD)
            src = mc + ((pos // 32 * KVH + h) * 2 * 32 + pos % 32) * HD + dd
            blk = slot // BS
            t = slot % BS
            hb = blk * KVH + h
            tl.store(
                kc + ((hb * (HD // X) + dd // X) * BS + t) * X + dd % X, tl.load(src)
            )
            tl.store(
                vc + ((hb * (BS // X) + t // X) * HD + dd) * X + t % X,
                tl.load(src + 32 * HD),
            )


def kv_import(
    k_ptrs, v_ptrs, table, start_pos, end_pos, mirror_ptrs, block_size, programs=32
) -> None:
    """Positions [start_pos, end_pos) (int32 [1] device tensors) of the
    request whose block table row is ``table`` into the mirror; nothing when
    the range is empty, so a step can always run it."""
    _kv_import_kernel[(len(d.ATTENTION_LAYERS), programs, d.KV_HEADS)](
        k_ptrs, v_ptrs, table, start_pos, end_pos, mirror_ptrs,
        BS=block_size, X=8, HD=d.HEAD_DIM, KVH=d.KV_HEADS, PROGRAMS=programs,
        num_warps=4,
    )  # fmt: skip


def kv_export(
    k_ptrs, v_ptrs, slots, base, rows, mirror_ptrs, max_rows, block_size
) -> None:
    _kv_export_kernel[(len(d.ATTENTION_LAYERS), max_rows, d.KV_HEADS)](
        k_ptrs, v_ptrs, slots, base, rows, mirror_ptrs,
        BS=block_size, X=8, HD=d.HEAD_DIM, KVH=d.KV_HEADS, num_warps=2,
    )  # fmt: skip
