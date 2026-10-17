# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tree verify of the Gated DeltaNet layers and its commit (GPU only), against
a node-by-node reference that continues every node from its parent."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from atom.model_ops import gdn_tree as gt
from atom.spec_decode import tree_commit as tc

dev = torch.device("cuda")


def _tree(n, gen):
    """A random tree in topological order: node 0 the root."""
    par = [-1] + [int(torch.randint(0, i, (1,), generator=gen)) for i in range(1, n)]
    return par


def _parents(trees, n):
    p = torch.full((len(trees), n), -1, dtype=torch.int32)
    for b, t in enumerate(trees):
        p[b, : len(t)] = torch.tensor(t, dtype=torch.int32)
    return p.to(dev)


def _chain_inputs(par, i, hist):
    """Conv inputs of node i: [great-grandparent, grandparent, parent, self]
    as node indices, history columns as -1 (newest), -2, -3."""
    out, cur = [i], i
    for _ in range(3):
        cur = par[cur] if cur >= 0 else cur - 1
        out.append(cur)
    return out[::-1]


@pytest.mark.parametrize("n", [5, 12])
def test_tree_conv(n):
    gen = torch.Generator().manual_seed(n)
    bs, dim, state_len = 3, 300, 8
    trees = [_tree(n, gen) for _ in range(bs)]
    x = torch.randn(bs * n, dim, generator=gen).to(torch.bfloat16).to(dev)
    conv = torch.randn(7, state_len, dim, generator=gen).to(torch.bfloat16).to(dev)
    slots = torch.tensor([4, 0, 6], dtype=torch.int32, device=dev)
    acc = torch.tensor([1, 3, 2], dtype=torch.int32, device=dev)
    w = torch.randn(dim, 4, generator=gen).to(torch.bfloat16).to(dev)
    bias = torch.randn(dim, generator=gen).to(torch.bfloat16).to(dev)
    raw = torch.zeros(
        bs, n + 3, dim, dtype=x.dtype, device=dev
    )  # wider than the verify
    out = gt.tree_conv(
        x, conv, slots, acc, _parents(trees, n), w, bias, "silu", raw=raw[:, :n]
    )
    assert torch.equal(raw[:, :n].reshape(bs * n, dim), x)
    for b, par in enumerate(trees):
        off = int(acc[b]) - 1
        hist = conv[int(slots[b]), off : off + 3].float()  # h0, h1, h2
        for i in range(n):
            cols = _chain_inputs(par, i, hist)
            vals = [x[b * n + c].float() if c >= 0 else hist[3 + c] for c in cols]
            ref = sum(v * w[:, j].float() for j, v in enumerate(vals)) + bias.float()
            ref = ref * torch.sigmoid(ref)
            assert torch.allclose(out[b * n + i].float(), ref, atol=2e-2, rtol=2e-2), (
                b,
                i,
            )


def _reference(q, k, v, a, bb, A_log, dt_bias, s0, par):
    """fp64 node-by-node gated delta rule: (outputs [n, HV, V], states [n, HV, V, K])."""
    n, H, K = q.shape
    HV = v.shape[1]
    g = -torch.exp(A_log.double()) * torch.nn.functional.softplus(
        a.double() + dt_bias.double()
    )
    beta = (
        torch.sigmoid(bb.double()).to(bb.dtype).double()
    )  # the kernels round beta to the input dtype
    qd, kd = q.double(), k.double()
    qn = qd / torch.sqrt((qd * qd).sum(-1, keepdim=True) + 1e-6) * K**-0.5
    kn = kd / torch.sqrt((kd * kd).sum(-1, keepdim=True) + 1e-6)
    rep = HV // H
    states, outs = [], []
    for i in range(n):
        s = s0 if par[i] < 0 else states[par[i]]  # [HV, V, K]
        d = torch.exp(g[i])[:, None, None]
        kk = kn[i].repeat_interleave(rep, 0)  # [HV, K]
        qq = qn[i].repeat_interleave(rep, 0)
        sd = d * s
        delta = beta[i][:, None] * (v[i].double() - torch.einsum("hvk,hk->hv", sd, kk))
        s_new = sd + delta[:, :, None] * kk[:, None, :]
        states.append(s_new)
        outs.append(torch.einsum("hvk,hk->hv", s_new, qq))
    return torch.stack(outs), torch.stack(states)


def test_tree_gated_delta_rule_and_commit():
    gen = torch.Generator().manual_seed(7)
    bs, n, H, HV, K, V = 2, 9, 2, 4, 128, 128
    trees = [_tree(n, gen) for _ in range(bs)]
    T = bs * n
    q = torch.randn(T, H, K, generator=gen).to(torch.bfloat16).to(dev)
    k = torch.randn(T, H, K, generator=gen).to(torch.bfloat16).to(dev)
    v = torch.randn(T, HV, V, generator=gen).to(torch.bfloat16).to(dev)
    a = torch.randn(T, HV, generator=gen).to(torch.bfloat16).to(dev)
    bb = torch.randn(T, HV, generator=gen).to(torch.bfloat16).to(dev)
    A_log = torch.randn(HV, generator=gen).to(dev)
    dt_bias = torch.randn(HV, generator=gen).to(dev)
    # the pool [layers=1, slots, HV, K, V] indexed, [hv][v][k] in memory
    mem = torch.randn(1, 5, HV, V, K, generator=gen).to(dev) * 0.1
    state = mem.transpose(-1, -2)
    slots = torch.tensor([3, 1], dtype=torch.int32, device=dev)
    parents = _parents(trees, n)
    rec = torch.zeros(1, bs, n, HV, gt.RECORD_FLOATS, device=dev)
    out = torch.empty(T, HV, V, dtype=torch.bfloat16, device=dev)
    gt.tree_gated_delta_rule(
        q, k, v, a, bb, A_log, dt_bias, state[0], slots, parents, rec[0], out
    )
    path = torch.zeros(bs, n, dtype=torch.int64, device=dev)
    kout = torch.zeros(bs, dtype=torch.int32, device=dev)
    refs = []
    for b, par in enumerate(trees):
        s0 = mem[0, int(slots[b])].double()
        o_ref, s_ref = _reference(
            q[b * n : (b + 1) * n], k[b * n : (b + 1) * n], v[b * n : (b + 1) * n],
            a[b * n : (b + 1) * n], bb[b * n : (b + 1) * n], A_log, dt_bias, s0, par,
        )  # fmt: skip
        err = (
            out[b * n : (b + 1) * n].double() - o_ref
        ).abs().max() / o_ref.abs().max()
        assert err < 2e-2, err
        # a root-to-leaf path: the deepest node's ancestors
        leaf = n - 1
        chain = [leaf]
        while par[chain[-1]] >= 0:
            chain.append(par[chain[-1]])
        chain = chain[::-1]
        path[b, : len(chain)] = torch.tensor(chain, device=dev)
        path[b, len(chain) :] = chain[-1]
        kout[b] = len(chain)
        refs.append(s_ref[leaf])
    conv = torch.randn(1, 5, 12, 64, generator=gen).to(torch.bfloat16).to(dev)
    raw = torch.randn(1, bs, n, 64, generator=gen).to(torch.bfloat16).to(dev)
    kout[1] = 2  # a short path: the window keeps a history row
    path[1, 2:] = path[1, 1]
    want_conv = []
    for b in range(bs):
        k_b = int(kout[b])
        seq = [conv[0, int(slots[b]), j] for j in range(3)] + [
            raw[0, b, int(path[b, i])] for i in range(k_b)
        ]
        want_conv.append(torch.stack(seq[-3:]))
    tc.commit(
        mem.transpose(-1, -2),
        conv,
        slots,
        rec,
        raw,
        path,
        kout,
        None,
        None,
        None,
        16,
        1,
        128,
    )
    for b in range(bs):
        assert torch.equal(conv[0, int(slots[b]), :3], want_conv[b]), b
        if b == 1:
            continue  # its state took a shorter path than the reference's
        got = mem[0, int(slots[b])].double()
        err = (got - refs[b]).abs().max() / refs[b].abs().max()
        assert err < 1e-3, (b, err)


def test_walk():
    gen = torch.Generator().manual_seed(3)
    bs, n = 4, 10
    trees = [_tree(n, gen) for _ in range(bs)]
    parents = _parents(trees, n)
    ids = torch.randint(0, 5, (bs, n), generator=gen).to(dev)
    yv = torch.randint(0, 5, (bs, n), generator=gen).to(dev)
    path = torch.zeros(bs, n, dtype=torch.int64, device=dev)
    toks = torch.zeros_like(path)
    kout = torch.zeros(bs, dtype=torch.int32, device=dev)
    out = torch.zeros(bs, n, dtype=torch.int32, device=dev)
    verdict = torch.zeros(3, bs, dtype=torch.int32, device=dev)
    tc.tree_walk(yv, ids, parents, path, kout, toks, out, verdict, n - 1)
    for b, par in enumerate(trees):
        cur, walk = 0, [0]
        while True:
            kids = [
                c
                for c in range(1, n)
                if par[c] == cur and int(ids[b, c]) == int(yv[b, cur])
            ]
            if not kids:
                break
            cur = min(kids)
            walk.append(cur)
        kk = len(walk)
        assert int(kout[b]) == kk
        assert path[b, :kk].tolist() == walk
        assert out[b, :kk].tolist() == [int(yv[b, c]) for c in walk]
        assert out[b, kk:].tolist() == [-1] * (n - kk)
        assert verdict[:, b].tolist() == [kk - 1, n - 1 - (kk - 1), b * n + kk - 1]


def test_tree_policy():
    from atom.spec_decode.tree_verify import parse_dynamic, parse_dynamic_policy

    assert parse_dynamic("dyn:8:4:16") == (8, 4, 16)
    assert parse_dynamic_policy("dyn:5:4:10") == ((1, (5, 4, 10)),)
    assert parse_dynamic_policy("1=dyn:8:4:16,4=dyn:5:4:10, 16=dyn:3:1:4") == (
        (1, (8, 4, 16)), (4, (5, 4, 10)), (16, (3, 1, 4)),
    )  # fmt: skip
    for bad in ("spine:3", "dyn:8:3:16", "dyn:2:2:20", "4=dyn:5:4:10", "dyn:8:4"):
        with pytest.raises(ValueError):
            parse_dynamic_policy(bad)


@pytest.mark.parametrize("bs,n", [(1, 16), (3, 9), (16, 4)])
def test_flydsl_chunk_matches_triton(bs, n):
    """The FlyDSL chunk (SGLang's EAGLE-4 ROCm port) against the Triton chunk on a bf16 VK pool."""
    flydsl = pytest.importorskip("atom.model_ops.gdn_tree_flydsl")
    gen = torch.Generator().manual_seed(bs * 31 + n)
    H, HV, K, V = 16, 48, 128, 128
    T = bs * n
    trees = [_tree(n, gen) for _ in range(bs)]
    dim = 2 * H * K + HV * V
    conv = (
        torch.randn(T, dim, generator=gen).to(torch.bfloat16).to(dev)
    )  # q / k / v as views, as in ATOM
    q = conv[:, : H * K].view(T, H, K)
    k = conv[:, H * K : 2 * H * K].view(T, H, K)
    v = conv[:, 2 * H * K :].view(T, HV, V)
    ab = torch.randn(T, 2 * HV, generator=gen).to(torch.bfloat16).to(dev)
    a, bb = ab[:, :HV], ab[:, HV:]
    A_log = torch.randn(HV, generator=gen).to(dev)
    dt_bias = torch.randn(HV, generator=gen).to(dev)
    mem = (
        (torch.randn(bs + 2, HV, V, K, generator=gen) * 0.1).to(torch.bfloat16).to(dev)
    )
    state = mem.transpose(-1, -2)
    slots = torch.randperm(bs + 2, generator=gen)[:bs].to(torch.int32).to(dev)
    parents = _parents(trees, n)
    outs, recs = [], []
    for fn in (gt.tree_gated_delta_rule, flydsl.tree_chunk_flydsl):
        rec = torch.zeros(bs, n, HV, gt.RECORD_FLOATS, device=dev)
        out = torch.empty(T, HV, V, dtype=torch.bfloat16, device=dev)
        fn(q, k, v, a, bb, A_log, dt_bias, state, slots, parents, rec, out)
        outs.append(out.float())
        recs.append(rec)
    rel = (outs[0] - outs[1]).abs().max() / outs[0].abs().max()
    assert rel < 2e-2, rel
    d = (recs[0][..., :V] - recs[1][..., :V]).abs().max() / recs[0][..., :V].abs().max()
    assert d < 1e-3, d  # the deltas feed the committed state
    assert torch.allclose(
        recs[0][..., V : V + K + 1], recs[1][..., V : V + K + 1], atol=1e-5
    )
