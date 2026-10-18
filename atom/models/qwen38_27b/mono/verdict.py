# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A greedy chain verify, sampled from the megakernel's logits.

The launch's rows are the anchor (the last committed token) and the drafts
after it. The target's greedy token at each row (``RowArgmax``, two launches)
accepts drafts while they match; ``chain_verdict`` (one launch) then writes
ATOM's verify result for one request: the emitted tokens and the counts the
rejection sampler would have produced.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _chain_verdict_kernel(
    YV, IDS, OUT, VERDICT, NSPEC: tl.constexpr, M: tl.constexpr, OUTP: tl.constexpr
):
    """Drafts ids[1..] are accepted while ids[r + 1] == yv[r]. -> OUT[0..NSPEC]
    the emitted tokens (the target's at rows 0..k) then -1, VERDICT (accepted
    drafts, rejected drafts, the last emitted token's row)."""
    r = tl.arange(0, M)
    yv = tl.load(YV + r).to(tl.int64)
    nxt = tl.load(IDS + r + 1, mask=r + 1 < M, other=-1).to(tl.int64)
    miss = (nxt != yv) | (r >= NSPEC)
    k = (
        tl.min(tl.where(miss, r, M)) + 1
    )  # rows emitted: the matches + the target's token after them
    o = tl.arange(0, OUTP)
    emitted = tl.sum(tl.where(o[:, None] == r[None, :], yv[None, :], 0), 1)
    tl.store(OUT + o, tl.where(o < k, emitted, -1).to(tl.int32), mask=o < NSPEC + 1)
    tl.store(VERDICT + 0, (k - 1).to(tl.int32))
    tl.store(VERDICT + 1, (NSPEC - (k - 1)).to(tl.int32))
    tl.store(VERDICT + 2, (k - 1).to(tl.int32))


def chain_verdict(yv, ids, out, verdict, nspec: int) -> None:
    """``yv`` the target's token at each row, ``ids`` each row's own token
    (int [rows] each, rows >= nspec + 1); ``out`` int32 [nspec + 1],
    ``verdict`` int32 [3]."""
    rows = yv.shape[0]
    _chain_verdict_kernel[(1,)](
        yv, ids, out, verdict, NSPEC=nspec, M=rows, OUTP=triton.next_power_of_2(nspec + 1), num_warps=1,
    )  # fmt: skip


# ---- the target's greedy token per row -----------------------------------------

ARGMAX_BLOCK = 8192


@triton.jit
def _argmax_partial_kernel(L, s_row, V, PV, PI, NCH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1)
    o = c * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(L + row.to(tl.int64) * s_row + o, mask=o < V, other=float("-inf"))
    i = tl.argmax(x, 0, tie_break_left=True)
    tl.store(PV + row * NCH + c, tl.max(x, 0))
    tl.store(PI + row * NCH + c, (c * BLOCK + i).to(tl.int32))


@triton.jit
def _argmax_combine_kernel(
    PV, PI, YV, ROWS: tl.constexpr, NCH: tl.constexpr, NCH_P2: tl.constexpr
):
    r = tl.arange(0, ROWS)
    c = tl.arange(0, NCH_P2)
    ok = (c < NCH)[None, :]
    v = tl.load(PV + r[:, None] * NCH + c[None, :], mask=ok, other=float("-inf"))
    i = tl.load(PI + r[:, None] * NCH + c[None, :], mask=ok, other=2147483647)
    m = tl.max(v, 1)
    # ties to the lowest index, as ``torch.argmax``
    best = tl.min(tl.where(v == m[:, None], i, 2147483647), 1)
    tl.store(YV + r, best.to(tl.int64))


class RowArgmax:
    """The greedy token of each of a launch's first ``rows`` logits rows, in
    two launches (a block max per 8K columns, then the rows' combine)."""

    def __init__(self, rows: int, vocab: int, device) -> None:
        self.rows, self.vocab = rows, vocab
        self.nch = triton.cdiv(vocab, ARGMAX_BLOCK)
        self.pv = torch.empty(rows, self.nch, dtype=torch.float32, device=device)
        self.pi = torch.empty(rows, self.nch, dtype=torch.int32, device=device)

    def __call__(self, logits: torch.Tensor, n: int, yv: torch.Tensor) -> None:
        """``logits`` [>= n, vocab] fp32 rows; ``yv`` int64 [rows] (rows >= n
        are written from whatever ``logits`` holds there)."""
        assert logits.shape[-1] == self.vocab and logits.stride(-1) == 1
        _argmax_partial_kernel[(n, self.nch)](
            logits, logits.stride(0), self.vocab, self.pv, self.pi, NCH=self.nch, BLOCK=ARGMAX_BLOCK, num_warps=8,
        )  # fmt: skip
        _argmax_combine_kernel[(1,)](
            self.pv, self.pi, yv, ROWS=self.rows, NCH=self.nch, NCH_P2=triton.next_power_of_2(self.nch), num_warps=4,
        )  # fmt: skip
