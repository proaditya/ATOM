# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""How many drafts a decode step verifies, by batch size (``ATOM_SPEC_VERIFY_LEN``).

A verify step's cost is flat in its rows while the weights dominate and grows
with them past that, so the chain that pays shrinks as the batch grows: on
Qwen3.8-27B (MI355X, TP1, SPEED-Bench) EAGLE-4 at 64 conversations gives
3438 tok/s verifying 3 drafts and 2440 verifying 5 (EAGLE-Harness's
``chain_intensity`` mode makes the same choice from cost data).

The drafter still drafts ``mtp_k`` tokens and KV stays reserved for ``mtp_k + 1``
rows; a step of B requests verifies only the first ``drafts(B)`` of each
(rectangular, every request alike), on the graph captured for that width. The
rest are re-drafted next step, so the output does not change.

Policy: ``"BS:DRAFTS,..."``, the entry with the largest ``BS <= B`` applying,
e.g. ``"1:7,24:3,48:2"``. Batches below the first entry verify everything.
"""

import numpy as np


def parse_policy(spec: str | None, mtp_k: int) -> tuple[tuple[int, int], ...]:
    """``"1:7,24:3"`` -> ((1, 7), (24, 3)), drafts clamped to [1, mtp_k]."""
    if not spec or mtp_k <= 0:
        return ()
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        bs, drafts = part.split(":")
        out[int(bs)] = max(1, min(int(drafts), mtp_k))
    return tuple(sorted(out.items()))


def drafts_for(policy: tuple[tuple[int, int], ...], bs: int, mtp_k: int) -> int:
    """Drafts a step of ``bs`` requests verifies."""
    drafts = mtp_k
    for threshold, d in policy:
        if bs >= threshold:
            drafts = d
    return drafts


def q_buckets(policy: tuple[tuple[int, int], ...], mtp_k: int) -> list[int]:
    """Every verify width (anchor + drafts) a step can run: one graph set each."""
    return sorted({d + 1 for _, d in policy} | {mtp_k + 1})


def apply_verify_len(
    batch, policy: tuple[tuple[int, int], ...], mtp_k: int
) -> int | None:
    """Shrink a pure-decode, rectangular verify step to the policy's width.

    Returns the new per-request width q, or None when the step keeps
    ``mtp_k + 1``. Rewrites only the worker's copy of the batch: each request's
    tokens keep their first q (anchor, then drafts), and its context length
    drops by the rows it no longer forwards, so positions and slots of the
    rows it does forward are unchanged.
    """
    if not policy or batch.total_tokens_num_prefill > 0:
        return None
    if int(getattr(batch, "num_spec_step", 0) or 0) != mtp_k:
        return None
    bs = batch.total_seqs_num_decode
    if bs <= 0:
        return None
    full_q = mtp_k + 1
    q = drafts_for(policy, bs, mtp_k) + 1
    if q >= full_q:
        return None
    old_nst = np.asarray(batch.num_scheduled_tokens)
    if not np.all(old_nst[:bs] == full_q):
        return None  # only a rectangular step shrinks
    sched = np.asarray(batch.scheduled_tokens)
    batch.scheduled_tokens = np.ascontiguousarray(
        sched[: bs * full_q].reshape(bs, full_q)[:, :q].reshape(-1)
    )
    nst = old_nst.copy()
    nst[:bs] = q
    batch.num_scheduled_tokens = nst
    batch.total_tokens_num_decode = bs * q
    batch.total_tokens_num = bs * q
    ctx = np.array(batch.context_lens, dtype=np.int64, copy=True)
    ctx[:bs] -= full_q - q
    batch.context_lens = ctx
    spec = batch.scheduled_spec_decode_tokens
    if spec is not None and getattr(spec, "size", 0) > 0:
        batch.scheduled_spec_decode_tokens = np.ascontiguousarray(spec[:, : q - 1])
    return q
