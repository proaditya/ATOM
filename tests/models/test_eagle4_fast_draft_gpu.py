# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The fused EAGLE-4 draft's attention over ATOM's paged cache (GPU only):
split-K partials + combine against a gathered reference, with a shuffled
block table and the window frozen at the round's first row."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from atom.spec_decode import eagle4_fast as ef


def _cache(blocks, h, hd, bs, x, gen):
    k = torch.randn(blocks, h, hd // x, bs, x, generator=gen, device="cuda").to(
        torch.bfloat16
    )
    v = torch.randn(blocks, h, bs // x, hd, x, generator=gen, device="cuda").to(
        torch.bfloat16
    )
    return k, v


def _logical(k, v, table, n):
    """The first n positions of the sequence, [n, h, hd] each."""
    bs = k.shape[3]
    pos = torch.arange(n, device="cuda")
    blk = table[pos // bs].long()
    t = pos % bs
    kk = k[blk, :, :, t, :].reshape(n, k.shape[1], -1)  # [n, h, hd/x * x]
    vv = v[blk, :, t // 8, :, t % 8]  # [n, h, hd]
    return kk.float(), vv.float()


@pytest.mark.parametrize(
    "s_pos,window", [(37, 4096), (1500, 4096), (5000, 4096), (700, 256)]
)
def test_paged_window_attention(s_pos, window):
    gen = torch.Generator(device="cuda").manual_seed(0)
    h, hd, bs, x = 4, 128, 16, 8
    blocks = (s_pos + 64) // bs + 4
    k, v = _cache(blocks, h, hd, bs, x, gen)
    table = torch.randperm(blocks, generator=gen, device="cuda").to(torch.int32)
    q = torch.randn(1, h, hd, generator=gen, device="cuda").to(torch.bfloat16)
    spos = torch.tensor([s_pos], dtype=torch.int64, device="cuda")
    # the row is the round's third new row: its window is frozen at the first
    first = max(0, s_pos - 2)
    firsts = torch.tensor([first], dtype=torch.int64, device="cuda")
    splits = ef.SPLITS
    po = torch.empty(h * splits, hd, dtype=torch.float32, device="cuda")
    pml = torch.empty(h * splits, 2, dtype=torch.float32, device="cuda")
    scale = hd**-0.5
    ef._attn_split_kernel[(h, 1, splits)](
        q, k, v, spos, firsts, 0, table, 0, po, pml, scale, H=h, HD=hd, BS=bs, X=x,
        WIN=window, SPLITS=splits, BLOCK=64, RPR=1, NSLOT=splits, num_warps=4,
    )  # fmt: skip
    out = torch.empty(1, h * hd, dtype=torch.bfloat16, device="cuda")
    ef._attn_combine_kernel[(h, 1)](
        po, pml, out, H=h, HD=hd, SPLITS=splits, num_warps=4
    )
    kk, vv = _logical(k, v, table, s_pos + 1)
    lo = max(0, first - window)
    kk, vv = kk[lo:], vv[lo:]
    att = torch.einsum("hd,nhd->hn", q[0].float(), kk) * scale
    ref = torch.einsum("hn,nhd->hd", att.softmax(-1), vv).reshape(1, -1)
    assert (out.float() - ref).abs().max() < 2e-2


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _fake_draft(hid=512, heads=4, hd=128, inter=1024, vocab=40000, gen=None):
    """A small Eagle4LlamaModel stand-in carrying exactly what the fused draft reads."""
    bf = torch.bfloat16

    def w(*shape, s=0.05):
        return (torch.randn(*shape, generator=gen, device="cuda") * s).to(bf)

    def norm(n):
        return (1 + 0.1 * torch.randn(n, generator=gen, device="cuda")).to(bf)

    attn = _NS(
        num_heads=heads, num_kv_heads=heads, head_dim=hd,
        qkv_proj=_NS(weight=w(3 * heads * hd, 2 * hid)), o_proj=_NS(weight=w(hid, heads * hd)),
        attn=_NS(layer_num=0),
    )  # fmt: skip
    layer = _NS(
        self_attn=attn,
        mlp=_NS(gate_up_proj=_NS(weight=w(2 * inter, hid)), down_proj=_NS(weight=w(hid, inter))),
        input_layernorm=_NS(weight=norm(hid)), hidden_norm=_NS(weight=norm(hid)),
        post_attention_layernorm=_NS(weight=norm(hid)),
    )  # fmt: skip
    model = _NS(
        config=_NS(hidden_size=hid, rms_norm_eps=1e-6, sliding_window=4096, rope_theta=1e7),
        midlayer=layer, fc=_NS(weight=w(hid, 3 * hid)), embed_tokens=_NS(weight=w(vocab, hid, s=1.0)),
        norm=_NS(weight=norm(hid)), fc_norm=[_NS(weight=norm(hid)) for _ in range(3)],
        efficient_lm_head_down_proj=_NS(weight=w(128, hid)), efficient_lm_head=_NS(weight=w(vocab, 128)),
        lm_head=_NS(weight=w(vocab, hid)), num_candidates=128,
    )  # fmt: skip
    return model


def _rms(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(
        torch.bfloat16
    ).float() * w.float()


def _ref_layer(m, fast, tokens, h, rope_pos, slot_pos, keys, values):
    """Reference over the logical cache (keys/values [P, heads, hd] after this
    step's writes); the window frozen at the first row: the ``fast.window``
    positions before it, then the rows up to each row itself."""
    L = m.midlayer
    x = torch.cat(
        [
            _rms(m.embed_tokens.weight[tokens], L.input_layernorm.weight),
            _rms(h, L.hidden_norm.weight),
        ],
        -1,
    )
    qkv = x.to(torch.bfloat16).float() @ L.self_attn.qkv_proj.weight.float().t()
    heads, hd = fast.H, fast.HD
    q = qkv[:, : heads * hd].view(-1, heads, hd)
    cos, sin = fast.cos[rope_pos], fast.sin[rope_pos]
    q1, q2 = q[..., : hd // 2], q[..., hd // 2 :]
    q = torch.cat(
        [
            q1 * cos[:, None, : hd // 2] - q2 * sin[:, None, : hd // 2],
            q2 * cos[:, None, : hd // 2] + q1 * sin[:, None, : hd // 2],
        ],
        -1,
    )
    q = q.to(torch.bfloat16).float()
    outs = []
    for r in range(tokens.shape[0]):
        s = int(slot_pos[r])
        lo = max(0, int(slot_pos[0]) - fast.window)
        att = torch.einsum("hd,nhd->hn", q[r], keys[lo : s + 1]) * fast.scale
        outs.append(
            torch.einsum("hn,nhd->hd", att.softmax(-1), values[lo : s + 1]).reshape(-1)
        )
    o = (
        torch.stack(outs).to(torch.bfloat16).float()
        @ L.self_attn.o_proj.weight.float().t()
    )
    res = (o + h.float()).to(torch.bfloat16)
    hn = _rms(res, L.post_attention_layernorm.weight).to(torch.bfloat16)
    gu = hn.float() @ L.mlp.gate_up_proj.weight.float().t()
    g, u = gu.chunk(2, -1)
    act = (torch.nn.functional.silu(g).to(torch.bfloat16).float() * u).to(
        torch.bfloat16
    )
    mm = act.float() @ L.mlp.down_proj.weight.float().t()
    return _rms((mm + res.float()).to(torch.bfloat16), m.norm.weight)


@pytest.mark.parametrize("mxfp4", [False, True])
def test_fused_layer_matches_reference(mxfp4):
    gen = torch.Generator(device="cuda").manual_seed(1)
    m = _fake_draft(gen=gen)
    # the candidate-head self check runs on the real head; skip it here
    ef.Eagle4FastDraft._check_head = lambda self, device: None
    fast = ef.Eagle4FastDraft(m, 4096, 3, torch.device("cuda"), mxfp4=mxfp4)
    fast.window = 128  # shorter than the context, so the frozen window shows
    heads, hd, bs, x = fast.H, fast.HD, 16, 8
    P = 300
    blocks = P // bs + 4
    k, v = _cache(blocks, heads, hd, bs, x, gen)
    table = torch.randperm(blocks, generator=gen, device="cuda").to(torch.int32)
    n = 5
    slot_pos = torch.arange(P - n, P, device="cuda")
    slots = table[slot_pos // bs].long() * bs + slot_pos % bs
    tokens = torch.randint(0, 40000, (n,), generator=gen, device="cuda")
    h = torch.randn(n, 512, generator=gen, device="cuda").to(torch.bfloat16)
    out = fast.layer(
        tokens, h, slot_pos + 1, slots, slot_pos, slot_pos[:1], table, k, v
    )
    keys, values = _logical(k, v, table, P)  # after the layer's own K / V writes
    if mxfp4:
        # the reference on the same MXFP4 weights (its activations stay bf16)
        L = m.midlayer
        for lin in (
            L.self_attn.qkv_proj,
            L.self_attn.o_proj,
            L.mlp.gate_up_proj,
            L.mlp.down_proj,
        ):
            lin.weight = dequantize(*ef.quant_mxfp4(lin.weight)).to(torch.bfloat16)
    ref = _ref_layer(m, fast, tokens, h, slot_pos + 1, slot_pos, keys, values)
    rel = float((out.float() - ref).norm() / ref.norm())
    assert rel < (0.05 if mxfp4 else 0.02), rel


def dequantize(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """``quant_mxfp4``'s [N, K/2] e2m1 codes and [N, K/32] e8m0 scales -> fp32 [N, K]."""
    n, half = packed.shape
    mags = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    table = torch.tensor(
        mags + tuple(-v for v in mags), dtype=torch.float32, device=packed.device
    )
    codes = torch.stack(((packed & 15).long(), (packed >> 4).long()), -1).view(
        n, half * 2
    )
    scale = (scales.to(torch.int32) << 23).view(torch.float32)  # 2 ** (e - 127)
    return (table[codes].view(n, -1, 32) * scale[..., None]).view(n, half * 2)


def test_batch_round_matches_single_rounds():
    """B requests in one batched round draft what B single rounds draft, and
    write the same draft K / V."""
    gen = torch.Generator(device="cuda").manual_seed(3)
    m = _fake_draft(gen=gen)
    ef.Eagle4FastDraft._check_head = lambda self, device: None
    k_draft = 3
    fast = ef.Eagle4FastDraft(m, 4096, k_draft, torch.device("cuda"), mxfp4=False)
    heads, hd, bs, x = fast.H, fast.HD, 16, 8
    B, q, W = 3, k_draft + 1, 32
    blocks = B * W + 4
    k, v = _cache(blocks, heads, hd, bs, x, gen)
    perm = torch.randperm(blocks, generator=gen, device="cuda").to(torch.int32)
    table = perm[: B * W].view(B, W).contiguous()
    lens = [200, 37, 311]  # each request's first verified row
    positions = torch.cat([torch.arange(p, p + q, device="cuda") for p in lens])
    req = torch.arange(B, device="cuda").repeat_interleave(q)
    slots = table[req, positions // bs].long() * bs + positions % bs
    tokens = torch.randint(0, 40000, (B * q,), generator=gen, device="cuda")
    aux = [
        torch.randn(B * q, 512, generator=gen, device="cuda").to(torch.bfloat16)
        for _ in range(3)
    ]
    last = torch.tensor([b * q + j for b, j in enumerate([1, 3, 0])], device="cuda")
    # single rounds, each on its own copy of the cache
    want, caches = [], []
    for b in range(B):
        kb, vb = k.clone(), v.clone()
        rows = slice(b * q, (b + 1) * q)
        out = torch.zeros(1, k_draft, dtype=torch.int64, device="cuda")
        fast.round(
            tokens[rows].clone(), positions[rows], slots[rows], last[b : b + 1] - b * q,
            [a[rows] for a in aux], table[b], kb, vb, out,
        )  # fmt: skip
        want.append(out[0])
        caches.append((kb, vb))
    rows3 = torch.zeros(3, k_draft, B, dtype=torch.int64, device="cuda")
    drafts = torch.zeros(k_draft, B, dtype=torch.int64, device="cuda")
    fast.batch_round(
        tokens.clone(), positions, slots, last, aux, table, k, v, rows3, drafts
    )
    assert torch.equal(drafts.t(), torch.stack(want))
    for b, (kb, vb) in enumerate(caches):
        used = table[b].long()
        # (a batch splits the draft attention differently: an ulp apart)
        assert torch.allclose(k[used].float(), kb[used].float(), atol=2e-2, rtol=2e-2)
        assert torch.allclose(v[used].float(), vb[used].float(), atol=2e-2, rtol=2e-2)


def _round_inputs(fast, gen, B, q, W, lens):
    bs = 16
    blocks = B * W + 4
    k, v = _cache(blocks, fast.H, fast.HD, bs, 8, gen)
    table = (
        torch.randperm(blocks, generator=gen, device="cuda")
        .to(torch.int32)[: B * W]
        .view(B, W)
        .contiguous()
    )
    positions = torch.cat([torch.arange(p, p + q, device="cuda") for p in lens])
    req = torch.arange(B, device="cuda").repeat_interleave(q)
    slots = table[req, positions // bs].long() * bs + positions % bs
    tokens = torch.randint(0, 40000, (B * q,), generator=gen, device="cuda")
    aux = [
        torch.randn(B * q, 512, generator=gen, device="cuda").to(torch.bfloat16)
        for _ in range(3)
    ]
    return k, v, table, positions, slots, tokens, aux


@pytest.mark.parametrize("B", [1, 2])
def test_dyn_tree_round(B):
    """Width 1 is the chain ``batch_round`` drafts; a wider tree is a valid
    tree (parents before children, depths along them) whose root children
    are the root's best candidates."""
    gen = torch.Generator(device="cuda").manual_seed(9)
    m = _fake_draft(gen=gen)
    ef.Eagle4FastDraft._check_head = lambda self, device: None
    mtp_k = 6
    fast = ef.Eagle4FastDraft(m, 4096, mtp_k, torch.device("cuda"), mxfp4=False)
    q, W = 4, 32
    k, v, table, positions, slots, tokens, aux = _round_inputs(
        fast, gen, B, q, W, [70, 301][:B]
    )
    last = torch.tensor([1, q + 3][:B], device="cuda")
    rows3 = torch.zeros(3, mtp_k, B, dtype=torch.int64, device="cuda")
    drafts = torch.zeros(mtp_k, B, dtype=torch.int64, device="cuda")
    fast.batch_round(
        tokens.clone(),
        positions,
        slots,
        last,
        aux,
        table,
        k.clone(),
        v.clone(),
        rows3,
        drafts,
    )

    def scratch(steps, width):
        return tuple(
            torch.zeros(
                B * steps * width, fast.H, fast.HD, dtype=torch.bfloat16, device="cuda"
            )
            for _ in range(2)
        )

    chain, parents, depths = fast.dyn_tree_round(
        tokens.clone(), positions, slots, last, aux, table, k.clone(), v.clone(), rows3, scratch(mtp_k, 1), mtp_k, 1, mtp_k + 1,
    )  # fmt: skip
    assert torch.equal(chain, drafts.t())
    assert parents.tolist() == [list(range(-1, mtp_k))] * B
    steps, width, nodes = 4, 4, 9
    tree, parents, depths = fast.dyn_tree_round(
        tokens.clone(), positions, slots, last, aux, table, k.clone(), v.clone(), rows3, scratch(steps, width), steps, width, nodes,
    )  # fmt: skip
    assert tree.shape == (B, nodes - 1)
    for b in range(B):
        par, dep = parents[b].tolist(), depths[b].tolist()
        assert par[0] == -1 and dep[0] == 0
        for i in range(1, nodes):
            assert 0 <= par[i] < i and dep[i] == dep[par[i]] + 1
        # the root's best child is the chain's first draft
        assert int(tree[b, 0]) == int(drafts[0, b]) and par[1] == 0


@pytest.mark.parametrize("width", [2, 4])
def test_dyn_tree_multi_row_attention(width):
    """The tree beam's rows through the multi-row attention (each request's
    keys read once for its rows, the tree part fused, bf16 products) draft
    the tree the row-at-a-time kernels (fp32) draft, but for a near tie."""
    gen = torch.Generator(device="cuda").manual_seed(11)
    m = _fake_draft(gen=gen)
    ef.Eagle4FastDraft._check_head = lambda self, device: None
    B, q, W, steps, nodes = 2, 4, 32, 4, 7
    fast = ef.Eagle4FastDraft(m, 4096, nodes - 1, torch.device("cuda"), mxfp4=False)
    k, v, table, positions, slots, tokens, aux = _round_inputs(
        fast, gen, B, q, W, [70, 301]
    )
    last = torch.tensor([1, q + 3], device="cuda")
    rows3 = torch.zeros(3, nodes - 1, B, dtype=torch.int64, device="cuda")
    out = []
    for multi in (ef.MULTI_ROWS, 1 << 30):
        saved, ef.MULTI_ROWS = ef.MULTI_ROWS, multi
        try:
            scratch = tuple(
                torch.zeros(
                    B * steps * width,
                    fast.H,
                    fast.HD,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                for _ in range(2)
            )
            out.append(fast.dyn_tree_round(
                tokens.clone(), positions, slots, last, aux, table, k.clone(), v.clone(), rows3, scratch, steps, width, nodes,
            ))  # fmt: skip
        finally:
            ef.MULTI_ROWS = saved
    (multi, _, _), (single, _, _) = out
    assert torch.equal(multi[:, 0], single[:, 0])
    assert (multi == single).float().mean() >= 0.8, (multi, single)


def test_batch_candidates_match_topk():
    """The batched threshold head picks torch.topk's candidate set, row by row."""
    gen = torch.Generator(device="cuda").manual_seed(4)
    m = _fake_draft(gen=gen)
    ef.Eagle4FastDraft._check_head = lambda self, device: None
    fast = ef.Eagle4FastDraft(m, 4096, 3, torch.device("cuda"), mxfp4=False)
    ch = fast.cand_head
    for n in (1, 4, 13):
        h = torch.randn(n, fast.hidden, generator=gen, device="cuda").to(torch.bfloat16)
        got = fast.batch_candidates(h)
        # fp32 scores of the bf16 low-rank projection, as the kernel computes them
        want = (
            ((h @ ch.down.t()).float() @ ch.eff.float().t()).topk(ch.C, dim=-1).indices
        )
        for r in range(n):
            common = len(set(got[r].tolist()) & set(want[r].tolist()))
            assert common >= ch.C - 2, (n, r, common)  # an edge candidate may tie


@pytest.mark.parametrize("k", [1, 4])
def test_batch_head_topk(k):
    """The fused exact-logit top-k matches torch on the same candidates."""
    gen = torch.Generator(device="cuda").manual_seed(6)
    m = _fake_draft(gen=gen)
    ef.Eagle4FastDraft._check_head = lambda self, device: None
    fast = ef.Eagle4FastDraft(m, 4096, 3, torch.device("cuda"), mxfp4=False)
    ch = fast.cand_head
    h = torch.randn(5, fast.hidden, generator=gen, device="cuda").to(torch.bfloat16)
    tok, lp = fast.batch_head_topk(h, k)
    cand = fast.batch_candidates(h)
    logits = torch.einsum("nh,nch->nc", h.float(), ch.lm[cand].float())
    top = logits.log_softmax(-1).topk(k, dim=-1)
    assert torch.equal(tok, cand.gather(1, top.indices))
    assert torch.allclose(lp, top.values, atol=1e-4)
