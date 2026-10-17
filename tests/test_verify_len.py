# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_SPEC_VERIFY_LEN: the policy and the step shrink (CPU only)."""

from types import SimpleNamespace

import numpy as np

from atom.spec_decode.verify_len import (
    apply_verify_len,
    drafts_for,
    parse_policy,
    q_buckets,
)


def _batch(bs, mtp_k, ctx):
    q = mtp_k + 1
    return SimpleNamespace(
        total_tokens_num_prefill=0,
        total_seqs_num_decode=bs,
        num_spec_step=mtp_k,
        num_scheduled_tokens=np.full(bs, q, dtype=np.int32),
        scheduled_tokens=np.arange(bs * q, dtype=np.int32),
        total_tokens_num_decode=bs * q,
        total_tokens_num=bs * q,
        context_lens=np.asarray(ctx, dtype=np.int32),
        scheduled_spec_decode_tokens=np.arange(bs * mtp_k, dtype=np.int32).reshape(
            bs, mtp_k
        ),
    )


def test_policy():
    p = parse_policy("24:3, 1:7,48:2,64:9", mtp_k=5)
    assert p == ((1, 5), (24, 3), (48, 2), (64, 5))
    assert [drafts_for(p, b, 5) for b in (1, 23, 24, 47, 48, 64)] == [5, 5, 3, 3, 2, 5]
    assert drafts_for((), 32, 5) == 5
    assert q_buckets(p, 5) == [3, 4, 6]
    assert parse_policy("", 5) == () and parse_policy("1:3", 0) == ()


def test_shrink_keeps_anchor_and_first_drafts():
    b = _batch(3, 5, [100, 200, 300])
    q = apply_verify_len(b, parse_policy("2:2", 5), 5)
    assert q == 3
    assert b.scheduled_tokens.tolist() == [0, 1, 2, 6, 7, 8, 12, 13, 14]
    assert b.num_scheduled_tokens.tolist() == [3, 3, 3]
    assert b.total_tokens_num == b.total_tokens_num_decode == 9
    # the rows it keeps keep their positions: context - q + i
    assert (np.asarray(b.context_lens) - 3).tolist() == [94, 194, 294]
    assert b.scheduled_spec_decode_tokens.tolist() == [[0, 1], [5, 6], [10, 11]]


def test_no_shrink():
    policy = parse_policy("4:2", 5)
    assert (
        apply_verify_len(_batch(3, 5, [9, 9, 9]), policy, 5) is None
    )  # below the first entry
    b = _batch(4, 5, [9] * 4)
    b.total_tokens_num_prefill = 7
    assert apply_verify_len(b, policy, 5) is None  # a prefill in the step
    b = _batch(4, 5, [9] * 4)
    b.num_scheduled_tokens[1] = 4
    assert apply_verify_len(b, policy, 5) is None  # not rectangular
