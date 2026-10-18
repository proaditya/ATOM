# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The DFlash 2 draft over ATOM's paged draft KV (GPU only) against a dense
reference (per-position K / V, the window gathered in torch), on a small
random draft with a shuffled block table."""

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from atom.models import qwen3_dflash2 as dm

H, NH, NKV, HD, INTER, V, WIN, LAYERS, T = 256, 4, 2, 128, 512, 1000, 64, 2, 8


class _Attn:
    def __init__(self, blocks, bs=16, x=8):
        self.k_cache = torch.zeros(
            blocks, NKV, HD // x, bs, x, dtype=torch.bfloat16, device="cuda"
        )
        self.v_cache = torch.zeros(
            blocks, NKV, bs // x, HD, x, dtype=torch.bfloat16, device="cuda"
        )


def _model(gen, blocks):
    bf = torch.bfloat16

    def r(*s, sc=0.05):
        return (torch.randn(*s, generator=gen, device="cuda") * sc).to(bf)

    def nw(n):
        return (1 + 0.1 * torch.randn(n, generator=gen, device="cuda")).to(bf)

    m = object.__new__(dm.DFlash2DraftModel)
    torch.nn.Module.__init__(m)
    m.layer_ids, m.block, m.mask_id, m.top_k, m.group, m.taps = (
        (0, 1, 2),
        T,
        V - 1,
        16,
        16,
        2,
    )
    m.hidden, m.nh, m.nkv, m.hd, m.window, m.eps, m.theta = (
        H,
        NH,
        NKV,
        HD,
        WIN,
        1e-6,
        1e7,
    )
    m.max_positions, m.num_layers = 4096, LAYERS
    m.attn = [_Attn(blocks) for _ in range(LAYERS)]
    w = {"fc": r(H, 3 * H), "hidden_norm": nw(H), "norm": nw(H), "hp": r(64, H),
         "pred_cb": r(V, 64, sc=1.0), "succ_cb": r(V, 64, sc=1.0)}  # fmt: skip
    for i in range(LAYERS):
        q, k, v = r(NH * HD, H), r(NKV * HD, H), r(NKV * HD, H)
        w[i] = {
            "ln1": nw(H), "ln2": nw(H), "qn": nw(HD), "kn": nw(HD), "qkv": torch.cat([q, k, v]), "kv": torch.cat([k, v]),
            "o": r(H, NH * HD), "gu": r(2 * INTER, H), "d": r(H, INTER), "akp": r(4 * H // 16, H), "abase": r(2, 2, H, sc=0.5),
            "mkp": r(4 * H // 16, H), "mbase": r(2, 2, H, sc=0.5),
        }  # fmt: skip
    m.w = w
    m.embed, m.lm_head = r(V, H, sc=1.0), r(V, H, sc=1.0)
    inv = 1.0 / (
        m.theta ** (torch.arange(0, HD, 2, device="cuda", dtype=torch.float32) / HD)
    )
    fr = torch.outer(
        torch.arange(m.max_positions, device="cuda", dtype=torch.float32), inv
    )
    emb = torch.cat([fr, fr], -1)
    m.cos, m.sin = emb.cos().contiguous(), emb.sin().contiguous()
    m.mask_ids = torch.full((T - 1,), m.mask_id, dtype=torch.int64, device="cuda")
    return m


class _Ref:
    """The reference math over dense per-position K / V."""

    def __init__(self, m, P):
        self.m = m
        self.k = [
            torch.zeros(NKV, P, HD, dtype=torch.bfloat16, device="cuda")
            for _ in range(LAYERS)
        ]
        self.v = [torch.zeros_like(k) for k in self.k]
        self.P = P

    def rope(self, x, pos):
        c = self.m.cos[pos][:, None, :]
        s = self.m.sin[pos][:, None, :]
        x1, x2 = x[..., : HD // 2].float(), x[..., HD // 2 :].float()
        return (x.float() * c + torch.cat([-x2, x1], -1) * s).to(torch.bfloat16)

    def context(self, aux, pos):
        m = self.m
        ch = dm.rms(aux @ m.w["fc"].T, m.w["hidden_norm"])
        for i in range(LAYERS):
            kv = ch @ m.w[i]["kv"].T
            k = self.rope(
                dm.rms(kv[:, : NKV * HD].reshape(-1, NKV, HD), m.w[i]["kn"]), pos
            )
            self.k[i][:, pos] = k.transpose(0, 1)
            self.v[i][:, pos] = kv[:, NKV * HD :].reshape(-1, NKV, HD).transpose(0, 1)

    def attend(self, q, i, terminal):
        kp = terminal - WIN + 1 + torch.arange(WIN, device="cuda")
        idx = kp.clamp(0, self.P - 1)
        K = self.k[i].index_select(1, idx).float()
        Vv = self.v[i].index_select(1, idx).float()
        qg = q.float().view(T, NKV, NH // NKV, HD)
        sc = torch.einsum("rkgd,ktd->rkgt", qg, K) * HD**-0.5
        sc = sc.masked_fill((kp < 0)[None, None, None, :], float("-inf"))
        return (
            torch.einsum("rkgt,ktd->rkgd", torch.softmax(sc, -1), Vv)
            .reshape(T, NH * HD)
            .to(torch.bfloat16)
        )

    def draft(self, anchor, p0, steps):
        m = self.m
        ids = torch.cat([anchor.view(1), m.mask_ids])
        pos = p0 + torch.arange(T, device="cuda")
        hidden, residual = m.embed[ids], None
        for i in range(LAYERS):
            L = m.w[i]
            residual = (
                hidden
                if residual is None
                else (hidden.float() + residual.float()).to(torch.bfloat16)
            )
            x = dm.rms(residual, L["ln1"])
            co = (x @ L["akp"].T).view(T, 2, 2, -1)
            xc = _conv(x, co[:, 0], L["abase"][0])
            qkv = xc @ L["qkv"].T
            q = self.rope(dm.rms(qkv[:, : NH * HD].reshape(T, NH, HD), L["qn"]), pos)
            k = self.rope(
                dm.rms(qkv[:, NH * HD : (NH + NKV) * HD].reshape(T, NKV, HD), L["kn"]),
                pos,
            )
            self.k[i][:, pos] = k.transpose(0, 1)
            self.v[i][:, pos] = (
                qkv[:, (NH + NKV) * HD :].reshape(T, NKV, HD).transpose(0, 1)
            )
            a = _conv(self.attend(q, i, p0 + T - 1) @ L["o"].T, co[:, 1], L["abase"][1])
            residual = (a.float() + residual.float()).to(torch.bfloat16)
            x = dm.rms(residual, L["ln2"])
            cm = (x @ L["mkp"].T).view(T, 2, 2, -1)
            g, u = (_conv(x, cm[:, 0], L["mbase"][0]) @ L["gu"].T).chunk(2, -1)
            hidden = _conv(
                (F.silu(g) * u).contiguous() @ L["d"].T, cm[:, 1], L["mbase"][1]
            )
        final = dm.rms(
            (hidden.float() + residual.float()).to(torch.bfloat16), m.w["norm"]
        )
        sample = final[1:]
        unary, cand = (sample @ m.lm_head.T).float().topk(16, -1)
        proj = sample @ m.w["hp"].T
        prev, out = anchor.view(1), []
        for s in range(steps):
            pr = m.w["pred_cb"].index_select(0, prev) * proj[s : s + 1]
            edge = m.w["succ_cb"].index_select(0, cand[s]) @ pr[0]
            prev = cand[s].index_select(0, (unary[s] + edge.float()).argmax().view(1))
            out.append(int(prev))
        return final, out


def _conv(x, delta, base):
    c = base[None].float() + delta.float().repeat_interleave(16, -1)
    xf = x.float()
    out = c[:, 0] * xf
    out[1:] += c[1:, 1] * xf[:-1]
    return out.to(torch.bfloat16)


@pytest.mark.parametrize("p0s", [(37,), (300, 90)])
def test_draft_matches_dense_reference(p0s):
    gen = torch.Generator(device="cuda").manual_seed(0)
    bs_ = 16
    blocks_per = (max(p0s) + T + bs_) // bs_ + 1
    B = len(p0s)
    m = _model(gen, B * blocks_per + 2)
    perm = torch.randperm(B * blocks_per, generator=gen, device="cuda").to(torch.int32)
    tables = perm.view(B, blocks_per)
    anchors = torch.randint(0, V - 1, (B,), generator=gen, device="cuda")
    refs = []
    for b, p0 in enumerate(p0s):
        ref = _Ref(m, blocks_per * bs_)
        pos = torch.arange(p0, device="cuda")
        aux = (torch.randn(p0, 3 * H, generator=gen, device="cuda")).to(torch.bfloat16)
        slots = tables[b].long()[pos // bs_] * bs_ + pos % bs_
        m.context(aux, pos, slots)
        ref.context(aux, pos)
        refs.append(ref)
    out = torch.zeros(B, T - 1, dtype=torch.int64, device="cuda")
    m.draft(anchors, torch.tensor(p0s, device="cuda"), tables, out)
    for b, p0 in enumerate(p0s):
        final, toks = refs[b].draft(anchors[b], p0, T - 1)
        got = m.last_final.view(B, T, H)[b].float()
        rel = float((got - final.float()).norm() / final.float().norm())
        assert rel < 0.02, rel
        assert out[b].tolist() == toks
