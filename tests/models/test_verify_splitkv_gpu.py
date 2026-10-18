# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Split-KV verify attention over ATOM's shuffled paged cache (GPU only)
against a dense causal reference, Qwen3.8's shape (24 q / 4 kv heads, head dim
256), ragged query lengths and a shuffled block table."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from atom.model_ops.attentions.triton_verify_splitkv import verify_attention

H, KVH, HD, BS, X = 24, 4, 256, 16, 8


@pytest.mark.parametrize(
    "ctxs,qlens", [((40,), (8,)), ((1500, 33000, 700), (8, 5, 16)), ((5000, 9), (4, 4))]
)
def test_verify_matches_reference(ctxs, qlens):
    gen = torch.Generator(device="cuda").manual_seed(0)
    seqs = len(ctxs)
    per = max(ctxs) // BS + 2
    blocks = seqs * per + 1
    k = torch.randn(blocks, KVH, HD // X, BS, X, generator=gen, device="cuda").to(
        torch.bfloat16
    )
    v = torch.randn(blocks, KVH, BS // X, HD, X, generator=gen, device="cuda").to(
        torch.bfloat16
    )
    tables = (
        torch.randperm(seqs * per, generator=gen, device="cuda")
        .to(torch.int32)
        .view(seqs, per)
    )
    cu = torch.tensor(
        [0] + list(torch.tensor(qlens).cumsum(0)), dtype=torch.int32, device="cuda"
    )
    tokens = int(cu[-1])
    q = torch.randn(tokens, H, HD, generator=gen, device="cuda").to(torch.bfloat16)
    ctx = torch.tensor(ctxs, dtype=torch.int32, device="cuda")
    scale = HD**-0.5
    out = verify_attention(q, k, v, tables, ctx, cu, max(qlens), scale)
    for s, (c, ql) in enumerate(zip(ctxs, qlens)):
        pos = torch.arange(c, device="cuda")
        blk = tables[s].long()[pos // BS]
        t = pos % BS
        kk = k[blk, :, :, t, :].reshape(c, KVH, HD).float()
        vv = v[blk, :, t // X, :, t % X].float()  # [c, KVH, HD]
        for i in range(ql):
            end = c - ql + i
            qq = q[int(cu[s]) + i].float().view(KVH, H // KVH, HD)
            sc = torch.einsum("kgd,nkd->kgn", qq, kk[: end + 1]) * scale
            ref = torch.einsum("kgn,nkd->kgd", sc.softmax(-1), vv[: end + 1]).reshape(
                H, HD
            )
            got = out[int(cu[s]) + i].float()
            assert (got - ref).abs().max() < 3e-2, (
                s,
                i,
                float((got - ref).abs().max()),
            )
