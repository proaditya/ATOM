# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A step's bookkeeping around the megakernel launch, one Triton launch each.

Every value is a handful of words, but as torch ops a step took ~60 kernels
(~230 us of a ~5.4 ms DFlash 2 round, every one a graph node of its own):

- ``step_inputs``: ATOM's metadata -> the step's words (``StepWords``) and
  each row's cache slot; gates the in-step commit on the pending request;
- ``launch_begin``: the recurrent state the step reads into the pointer
  table, the KV mirror's import range, and the launch inputs (token ids,
  positions and pages, fresh control block, command epoch);
- ``launch_arm``: after the launch, what its commit needs (pending, source
  slot), and the mirror's new extent.
"""

import torch
import triton
import triton.language as tl

from atom.models.qwen38_27b.mono import dims as d

# StepWords layout (int32)
W_BASE, W_ROWS, W_STATE, W_CONV_IN, W_WINDOW, W_CONV_OUT, W_ACCEPTED = range(7)
STEP_WORDS = 8


class StepWords:
    """A step's scalars as views of one int32 buffer the inputs kernel writes."""

    def __init__(self, max_rows: int, device) -> None:
        self.words = torch.zeros(STEP_WORDS, dtype=torch.int32, device=device)
        self.kv_slots = torch.zeros(max_rows, dtype=torch.int64, device=device)
        self.token_ids = torch.zeros(max_rows, dtype=torch.int64, device=device)

    def word(self, i: int) -> torch.Tensor:
        return self.words[i : i + 1]


@triton.jit
def _step_inputs_kernel(
    IDS,
    POS,
    TABLE,
    TABLE_N,
    SPEC,
    NUM_ACC,
    NS_IN,
    NS_OUT,
    PENDING_SLOT0,
    PENDING,
    W,
    KV_SLOTS,
    TOK,
    Q: tl.constexpr,
    ROWS: tl.constexpr,
    BS: tl.constexpr,
    SPECULATIVE: tl.constexpr,
):
    r = tl.arange(0, ROWS)
    ids = tl.load(IDS + tl.minimum(r, Q - 1)).to(tl.int64)  # pad with the last id
    tl.store(TOK + r, ids)
    base = tl.load(POS).to(tl.int32)
    if SPECULATIVE:
        accepted = tl.load(NUM_ACC).to(tl.int32)
        state = tl.load(SPEC + accepted - 1).to(tl.int32)
        slot0 = tl.load(SPEC).to(tl.int32)
        conv_in = slot0
        conv_out = slot0
        window = accepted
        # the in-step commit settles only this request's pending launch
        hit = (slot0 == tl.load(PENDING_SLOT0)).to(tl.int32)
        tl.store(PENDING, tl.load(PENDING) * hit)
    else:
        accepted = tl.full((), 1, tl.int32)
        state = tl.load(NS_IN).to(tl.int32)
        conv_in = state
        conv_out = tl.load(NS_OUT).to(tl.int32)
        window = tl.full((), 1, tl.int32)
    tl.store(W + 0, base)
    tl.store(W + 1, tl.full((), Q, tl.int32))
    tl.store(W + 2, state)
    tl.store(W + 3, conv_in)
    tl.store(W + 4, window)
    tl.store(W + 5, conv_out)
    tl.store(W + 6, accepted)
    # each row's cache slot from the block table (``slot_mapping`` is not
    # sized for the rows of the metadata a graph is captured with)
    pos = base.to(tl.int64) + r
    blk = tl.minimum(pos // BS, TABLE_N - 1)
    tl.store(KV_SLOTS + r, tl.load(TABLE + blk).to(tl.int64) * BS + pos % BS)


def step_inputs(
    words: StepWords,
    ids,
    positions,
    table,
    rows: int,
    block_size: int,
    spec=None,
    num_accepted=None,
    pending_slot0=None,
    pending=None,
    ns_in=None,
    ns_out=None,
) -> None:
    """``ids`` the step's q token ids (padded to ``rows`` with the last),
    ``positions`` row 0 first, ``table`` the sequence's block table row.
    Speculative (``spec`` the request's state slots): window and state slot
    from the accepted count, and ``pending`` cleared unless the pending launch
    is this request's; else the non-spec slots ``ns_in`` / ``ns_out``."""
    speculative = spec is not None
    dummy = words.words
    _step_inputs_kernel[(1,)](
        ids, positions, table, table.numel(),
        spec if speculative else dummy, num_accepted if speculative else dummy,
        dummy if speculative else ns_in, dummy if speculative else ns_out,
        pending_slot0 if speculative else dummy, pending if speculative else dummy,
        words.words, words.kv_slots, words.token_ids,
        Q=ids.numel(), ROWS=rows, BS=block_size, SPECULATIVE=speculative, num_warps=1,
    )  # fmt: skip


@triton.jit
def _launch_begin_kernel(
    # the recurrent state the step reads -> the table's initial_state cells
    TABLE,
    TABLE_STRIDE,
    DELTA_ROWS,
    STATE_PTR,
    S_LAYER,
    S_SLOT,
    STATE_SLOT,
    # the KV mirror
    MIRROR_SLOT,
    MIRROR_LEN,
    REQUEST,
    IMPORT_START,
    IMPORT_END,
    # the launch
    TOK_SRC,
    TOK_DST,
    BASE,
    POSITIONS,
    CONTEXT_BLOCKS,
    BLOCK_MAP,
    NATIVE,
    CONTROLS,
    COMMAND,
    STATE_COL: tl.constexpr,
    NDELTA: tl.constexpr,
    NDELTA_P2: tl.constexpr,
    ROWS: tl.constexpr,
    PAGE: tl.constexpr,
    NATIVE_TOTAL: tl.constexpr,
    CONTROL_N: tl.constexpr,
    COMMAND_WORD: tl.constexpr,
    MAX_EPOCH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    j = tl.arange(0, NDELTA_P2)
    jm = j < NDELTA
    row = tl.load(DELTA_ROWS + j, mask=jm, other=0)
    slot = tl.load(STATE_SLOT).to(tl.int64)
    cell = STATE_PTR + j.to(tl.int64) * S_LAYER + slot * S_SLOT
    tl.store(TABLE + row * TABLE_STRIDE + STATE_COL, cell, mask=jm)
    # import what ATOM's cache holds that the mirror does not: from its extent
    # if it holds this request, else from 0
    base = tl.load(BASE).to(tl.int32)
    held = tl.load(MIRROR_SLOT) == tl.load(REQUEST)
    tl.store(IMPORT_START, tl.where(held, tl.minimum(tl.load(MIRROR_LEN), base), 0))
    tl.store(IMPORT_END, base)
    r = tl.arange(0, ROWS)
    tl.store(TOK_DST + r, tl.load(TOK_SRC + r))
    pos = base + r
    tl.store(POSITIONS + r, pos)
    tl.store(CONTEXT_BLOCKS + r, tl.load(BLOCK_MAP + pos // PAGE))
    # a fresh control block every launch, as the qualified drivers do
    z = tl.zeros([BLOCK], dtype=tl.int32)
    for i in range(0, NATIVE_TOTAL, BLOCK):
        o = i + tl.arange(0, BLOCK)
        tl.store(NATIVE + o, z, mask=o < NATIVE_TOTAL)
    for i in range(0, CONTROL_N, BLOCK):
        o = i + tl.arange(0, BLOCK)
        tl.store(CONTROLS + o, z, mask=o < CONTROL_N)
    tl.debug_barrier()
    # the command epoch: non-zero, below MAX_EPOCH, advancing every launch
    epoch = tl.load(COMMAND) % (MAX_EPOCH - 1) + 1
    tl.store(COMMAND, epoch)
    tl.store(NATIVE + COMMAND_WORD, epoch)


def launch_begin(
    binding,
    delta_rows,
    state_col: int,
    pools,
    inp,
    mirror_slot,
    mirror_len,
    import_start,
    import_end,
) -> None:
    state = pools.state
    nbytes = state.element_size()
    b = binding
    _launch_begin_kernel[(1,)](
        b.table, b.table.stride(0), delta_rows, state.data_ptr(), state.stride(0) * nbytes,
        state.stride(1) * nbytes, inp.state_slot,
        mirror_slot, mirror_len, inp.request, import_start, import_end,
        inp.token_ids, b.endpoints["token_ids"], inp.base, b.positions, b.context_blocks, b.block_map,
        b.native, b.controls, b.command,
        STATE_COL=state_col, NDELTA=delta_rows.numel(),
        NDELTA_P2=triton.next_power_of_2(delta_rows.numel()), ROWS=b.rows, PAGE=d.PAGE_TOKENS,
        NATIVE_TOTAL=b.native.numel(), CONTROL_N=b.controls.numel(),
        COMMAND_WORD=d.COMMAND_WORD, MAX_EPOCH=d.MAX_COMMAND_EPOCH,
        BLOCK=4096, num_warps=8,
    )  # fmt: skip


@triton.jit
def _launch_arm_kernel(
    PENDING,
    PENDING_SRC,
    STATE_SLOT,
    BASE,
    MIRROR_SLOT,
    MIRROR_LEN,
    REQUEST,
    PENDING_SLOT0,
):
    # the mirror now holds positions < base (this launch's rows beyond the
    # accepted ones are overwritten by the next launch)
    tl.store(MIRROR_SLOT, tl.load(REQUEST).to(tl.int32))
    tl.store(MIRROR_LEN, tl.load(BASE).to(tl.int32))
    tl.store(PENDING_SRC, tl.load(STATE_SLOT).to(tl.int32))
    # the pending launch is named by its request
    tl.store(PENDING_SLOT0, tl.load(REQUEST).to(tl.int32))
    tl.store(PENDING, tl.full((), 1, tl.int32))


def launch_arm(eng, inp) -> None:
    _launch_arm_kernel[(1,)](
        eng.pending, eng.pending_src, inp.state_slot, inp.base, eng.mirror_slot, eng.mirror_len, inp.request,
        eng.pending_slot0, num_warps=1,
    )  # fmt: skip


@triton.jit
def _commit_inputs_kernel(LAST, SPEC, PENDING_SLOT0, PENDING, W):
    accepted = tl.load(LAST).to(tl.int32) + 1
    tl.store(W + 0, accepted)
    tl.store(W + 1, tl.load(SPEC + accepted - 1).to(tl.int32))
    hit = (tl.load(SPEC).to(tl.int32) == tl.load(PENDING_SLOT0)).to(tl.int32)
    tl.store(PENDING, tl.load(PENDING) * hit)


def commit_inputs(
    words: torch.Tensor,
    last: torch.Tensor,
    spec_row: torch.Tensor,
    pending_slot0,
    pending,
) -> None:
    """A sampled verify's commit inputs: ``words`` int32 [2] <- (accepted =
    ``last`` + 1, the request's state slot ``spec_row[accepted - 1]``), and
    ``pending`` cleared unless the pending launch is this request's."""
    _commit_inputs_kernel[(1,)](
        last, spec_row, pending_slot0, pending, words, num_warps=1
    )
