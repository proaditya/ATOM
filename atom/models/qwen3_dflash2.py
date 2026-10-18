# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""DFlash 2 block drafter (``DFlash2DraftModel``, e.g. incoai/Qwen3.8-27B-DFlash2).

DFlash 2 drafts a whole block in one pass. Per request and round:

- context: the target's residual stream after layers ``target_layer_ids``
  (5 / 19 / 33 / 47 / 61 for Qwen3.8-27B, HF ``hidden_states[id + 1]``) of every
  position the target ran goes through ``fc`` -> ``hidden_norm`` -> each draft
  layer's k / v projection (``k_norm``, RoPE) into that layer's KV;
- block: ``[anchor, mask x 7]`` at positions ``p0 .. p0 + 7`` (``p0`` the
  anchor's) through five layers, non-causal inside the block: every row attends
  to the keys at positions ``(p0 + 7 - window, p0 + 7]``, the block's own rows
  included. Each layer wraps attention and MLP in block-local dynamic grouped
  convolutions (``out[t] = c0[t] x[t] + c1[t] x[t - 1]``, ``c = base +
  kernel_projection(x)`` per 16-channel group);
- head: final norm, the TARGET's LM head on rows 1..7, the top-16 of each, and
  a low-rank selector that walks the greedy coherent path from the anchor
  (``unary + successor[cand] . (predecessor[prev] * hidden_projection(h))``).

The draft's KV lives in ATOM's draft pool (``draft_kv_builder``), riding the
target's block ids, so a position's draft K / V sit at its target slot. The
block rows overwrite the context rows at ``p0 .. p0 + 7``: those positions are
re-written with the target's context once the next verify has run them, before
anything reads them.

The weights are read straight from the checkpoint (``load_checkpoint``); the
embedding and LM head are the target's, bf16.

"""

from pathlib import Path

import torch
import triton
import triton.language as tl
from torch import nn

from atom.model_ops.base_attention import Attention

SPLITS = 8  # split-K attention over the window


@triton.jit
def _norm_rope_store_kernel(
    X,
    sx,
    POS,
    QN,
    KN,
    COS,
    SIN,
    QO,
    KC,
    VC,
    SLOTS,
    NQ: tl.constexpr,
    NKV: tl.constexpr,
    HD: tl.constexpr,
    BS: tl.constexpr,
    XV: tl.constexpr,
    eps,
):
    """Row r, head h of a [q | k | v] projection (q heads h < NQ, absent when
    NQ = 0): q -> rms * q_norm, RoPE -> QO [n, NQ, HD]; k -> rms * k_norm, RoPE
    -> the paged K cache at SLOTS[r]; v -> the paged V cache (slot < 0 skipped).
    Rounding as the reference: rms to bf16, times the weight to bf16, RoPE in
    fp32 to bf16."""
    r = tl.program_id(0)
    h = tl.program_id(1)
    half: tl.constexpr = HD // 2
    d = tl.arange(0, half)
    pos = tl.load(POS + r).to(tl.int64)
    base = X + r * sx
    if h < NQ + NKV:
        x1 = tl.load(base + h * HD + d).to(tl.float32)
        x2 = tl.load(base + h * HD + half + d).to(tl.float32)
        inv = 1.0 / tl.sqrt((tl.sum(x1 * x1, 0) + tl.sum(x2 * x2, 0)) / HD + eps)
        if h < NQ:
            w1 = tl.load(QN + d).to(tl.float32)
            w2 = tl.load(QN + half + d).to(tl.float32)
        else:
            w1 = tl.load(KN + d).to(tl.float32)
            w2 = tl.load(KN + half + d).to(tl.float32)
        x1 = (
            ((x1 * inv).to(tl.bfloat16).to(tl.float32) * w1)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        x2 = (
            ((x2 * inv).to(tl.bfloat16).to(tl.float32) * w2)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        c1 = tl.load(COS + pos * HD + d)
        c2 = tl.load(COS + pos * HD + half + d)
        s1 = tl.load(SIN + pos * HD + d)
        s2 = tl.load(SIN + pos * HD + half + d)
        o1 = (x1 * c1 - x2 * s1).to(tl.bfloat16)
        o2 = (x2 * c2 + x1 * s2).to(tl.bfloat16)
        if h < NQ:
            qo = QO + (r * NQ + h) * HD
            tl.store(qo + d, o1)
            tl.store(qo + half + d, o2)
        else:
            slot = tl.load(SLOTS + r).to(tl.int64)
            if slot >= 0:
                hb = (slot // BS) * NKV + (h - NQ)
                t = slot % BS
                tl.store(KC + ((hb * (HD // XV) + d // XV) * BS + t) * XV + d % XV, o1)
                tl.store(
                    KC
                    + ((hb * (HD // XV) + (d + half) // XV) * BS + t) * XV
                    + (d + half) % XV,
                    o2,
                )
    else:
        slot = tl.load(SLOTS + r).to(tl.int64)
        if slot >= 0:
            dd = tl.arange(0, HD)
            hb = (slot // BS) * NKV + (h - NQ - NKV)
            t = slot % BS
            v = tl.load(base + h * HD + dd)
            tl.store(VC + ((hb * (BS // XV) + t // XV) * HD + dd) * XV + t % XV, v)


@triton.jit
def _gqa_attn_split_kernel(
    Q,
    KC,
    VC,
    END,
    BT,
    sbt,
    PO,
    PML,
    scale,
    ROWS: tl.constexpr,
    H: tl.constexpr,
    KVH: tl.constexpr,
    HD: tl.constexpr,
    BS: tl.constexpr,
    X: tl.constexpr,
    WIN: tl.constexpr,
    SPLITS: tl.constexpr,
    CH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Split-K partial of one (q head, row, split): the keys at positions
    (e - WIN, e] (e = END[request], the same for the ROWS rows of a request),
    chunk ``split`` of them, through the request's block table. The paged
    layouts are ATOM's (K [block, kvh, HD / X, BS, X], V [block, kvh, BS / X,
    HD, X])."""
    h = tl.program_id(0)
    r = tl.program_id(1)
    sp = tl.program_id(2)
    req = r // ROWS
    kvh = h // (H // KVH)
    bt = BT + req * sbt
    d = tl.arange(0, HD)
    e_pos = tl.load(END + req)
    st0 = tl.maximum(e_pos - WIN + 1, 0) + sp * CH
    end = tl.minimum(st0 + CH, e_pos + 1)
    q = tl.load(Q + (r * H + h) * HD + d).to(tl.float32) * scale
    m_i = tl.full((), float("-inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((HD,), tl.float32)
    kk = tl.arange(0, BLOCK)
    g = tl.arange(0, HD // X)
    e = tl.arange(0, X)
    vg = tl.arange(0, BLOCK // X)
    lo = tl.where(st0 < end, st0 // BLOCK * BLOCK, end)
    for st in range(lo, end, BLOCK):
        kp = st + kk
        ok = (kp >= st0) & (kp < end)
        blk = tl.load(bt + kp // BS, mask=kp < end, other=0).to(tl.int64)
        hb = blk * KVH + kvh
        koff = (
            (hb[:, None, None] * (HD // X) + g[None, :, None]) * BS
            + (kp % BS)[:, None, None]
        ) * X + e[None, None, :]
        k = tl.reshape(
            tl.load(KC + koff, mask=(kp < end)[:, None, None], other=0.0), (BLOCK, HD)
        ).to(tl.float32)
        sc = tl.where(ok, tl.sum(k * q[None, :], 1), float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 0))
        # a tile with no key of this split leaves the running max at -inf
        alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp(m_i - m_new))
        pr = tl.where(ok, tl.exp(sc - m_new), 0.0)
        kv0 = st + vg * X
        vblk = tl.load(bt + kv0 // BS, mask=kv0 < end, other=0).to(tl.int64)
        vb = vblk * KVH + kvh
        voff = (
            (vb[:, None, None] * (BS // X) + ((kv0 % BS) // X)[:, None, None]) * HD
            + d[None, :, None]
        ) * X + e[None, None, :]
        v3 = tl.load(VC + voff, mask=(kv0 < end)[:, None, None], other=0.0)
        v = tl.reshape(tl.permute(v3, (0, 2, 1)), (BLOCK, HD)).to(tl.float32)
        acc = acc * alpha + tl.sum(pr[:, None] * v, 0)
        l_i = l_i * alpha + tl.sum(pr, 0)
        m_i = m_new
    slot = (r * H + h) * SPLITS + sp
    tl.store(PO + slot * HD + d, acc)
    tl.store(PML + slot * 2, m_i)
    tl.store(PML + slot * 2 + 1, l_i)


@triton.jit
def _attn_combine_kernel(
    PO, PML, O, H: tl.constexpr, HD: tl.constexpr, SPLITS: tl.constexpr
):
    h = tl.program_id(0)
    r = tl.program_id(1)
    d = tl.arange(0, HD)
    sp = tl.arange(0, SPLITS)
    slot = (r * H + h) * SPLITS + sp
    m = tl.load(PML + slot * 2)
    ln = tl.load(PML + slot * 2 + 1)
    w = tl.where(m == float("-inf"), 0.0, tl.exp(m - tl.max(m, 0)))
    acc = tl.load(PO + slot[:, None] * HD + d[None, :])
    tl.store(
        O + r * H * HD + h * HD + d,
        (tl.sum(w[:, None] * acc, 0) / tl.sum(w * ln, 0)).to(tl.bfloat16),
    )


@triton.jit
def _rms_kernel(
    X, R, W, OUT, RES, N: tl.constexpr, BLOCK: tl.constexpr, HAS_RES: tl.constexpr, eps
):
    """OUT[row] = rms(x) * W, x = X[row] (+ R[row], rounded to bf16 into RES);
    the reference's rounding (normalized to bf16, times the weight to bf16)."""
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK)
    m = c < N
    x = tl.load(X + row * N + c, mask=m, other=0.0).to(tl.float32)
    if HAS_RES:
        x = (x + tl.load(R + row * N + c, mask=m, other=0.0).to(tl.float32)).to(
            tl.bfloat16
        )
        tl.store(RES + row * N + c, x, mask=m)
        x = x.to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(x * x, 0) / N + eps)
    y = (x * inv).to(tl.bfloat16).to(tl.float32) * tl.load(W + c, mask=m, other=0.0).to(
        tl.float32
    )
    tl.store(OUT + row * N + c, y.to(tl.bfloat16), mask=m)


@triton.jit
def _conv_kernel(
    X,
    CO,
    BASE,
    OUT,
    PART: tl.constexpr,
    T: tl.constexpr,
    N: tl.constexpr,
    G: tl.constexpr,
    GS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Block-local dynamic grouped conv, part PART (0 before, 1 after) of
    ``kernel_projection``'s [n, 2 parts, 2 taps, G groups] coefficients:
    out[r] = (b0 + d0[r]) x[r] + [r % T > 0] (b1 + d1[r]) x[r - 1]."""
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    co = CO + r * (4 * G) + PART * 2 * G
    d0 = tl.load(co + c // GS).to(tl.float32)
    d1 = tl.load(co + G + c // GS).to(tl.float32)
    b0 = tl.load(BASE + PART * 2 * N + c).to(tl.float32)
    b1 = tl.load(BASE + PART * 2 * N + N + c).to(tl.float32)
    x = tl.load(X + r * N + c).to(tl.float32)
    o = (b0 + d0) * x
    if r % T > 0:
        o += (b1 + d1) * tl.load(X + (r - 1) * N + c).to(tl.float32)
    tl.store(OUT + r * N + c, o.to(tl.bfloat16))


@triton.jit
def _silu_mul_kernel(GU, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(GU + row * 2 * N + c).to(tl.float32)
    u = tl.load(GU + row * 2 * N + N + c).to(tl.float32)
    a = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + row * N + c, (a * u).to(tl.bfloat16))


@triton.jit
def _ord_key(score, idx):
    """fp32 score, int index -> an int64 key ordered by score, ties to the
    smaller index."""
    b = score.to(tl.int32, bitcast=True)
    o = b ^ ((b >> 31) & 0x7FFFFFFF)
    return (o.to(tl.int64) << 32) | (0xFFFFFFFF - idx.to(tl.int64))


@triton.jit
def _topk_chunk_kernel(
    L, KEYS, V, NCH: tl.constexpr, CH: tl.constexpr, K: tl.constexpr
):
    """Top-K keys of one CH-wide chunk of a logits row."""
    r = tl.program_id(0).to(tl.int64)
    ch = tl.program_id(1)
    i = ch * CH + tl.arange(0, CH)
    ok = i < V
    sc = tl.load(L + r * V + i, mask=ok, other=float("-inf")).to(tl.float32)
    key = tl.where(ok, _ord_key(sc, i), -9223372036854775807)
    tl.store(KEYS + (r * NCH + ch) * K + tl.arange(0, K), tl.topk(key, K))


@triton.jit
def _select_walk_kernel(
    KEYS,
    NKEYS: tl.constexpr,
    PROJ,
    PRED,
    SUCC,
    ANCHOR,
    OUT,
    so,
    STEPS: tl.constexpr,
    TS: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
):
    """One request: the top-K of each block row from the chunk keys, then the
    selector's greedy walk from the anchor: score = unary + succ[cand] .
    (pred[prev] * proj) (bf16 roundings as the reference), argmax (ties to the
    better unary)."""
    b = tl.program_id(0).to(tl.int64)
    prev = tl.load(ANCHOR + b).to(tl.int64)
    rr = tl.arange(0, R)
    kk = tl.arange(0, K)
    for s in tl.static_range(STEPS):
        row = b * TS + s
        keys = tl.load(KEYS + row * NKEYS + tl.arange(0, NKEYS))
        top = tl.topk(keys, K)  # sorted, best first
        o = (top >> 32).to(tl.int32)
        bits = o ^ ((o >> 31) & 0x7FFFFFFF)
        unary = bits.to(tl.float32, bitcast=True)
        cand = 0xFFFFFFFF - (top & 0xFFFFFFFF)
        pj = tl.load(PROJ + row * R + rr).to(tl.float32)
        pc = tl.load(PRED + prev * R + rr).to(tl.float32)
        pr = (pc * pj).to(tl.bfloat16).to(tl.float32)
        succ = tl.load(SUCC + cand[:, None] * R + rr[None, :]).to(tl.float32)
        edge = tl.sum(succ * pr[None, :], 1).to(tl.bfloat16).to(tl.float32)
        score = unary + edge
        best = tl.max(score, 0)
        pick = tl.min(tl.where(score == best, kk, K), 0)
        prev = tl.sum(tl.where(kk == pick, cand, 0), 0)
        tl.store(OUT + b * so + s, prev)


def rms(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Qwen3 RMSNorm (plain weight), the reference's rounding."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(
        torch.bfloat16
    ) * w


class DFlash2DraftModel(nn.Module):
    """The DFlash 2 draft: weights, its KV holders and the two passes."""

    def __init__(self, atom_config, layer_offset: int = 0) -> None:
        super().__init__()
        cfg = atom_config.hf_config
        dc = cfg.dflash_config
        self.config = cfg
        self.layer_ids = tuple(int(i) for i in dc["target_layer_ids"])
        self.block = int(dc["block_size"])
        self.mask_id = int(dc["mask_token_id"])
        self.top_k = int(dc["selector_top_k"])
        self.group = int(dc["conv_group_size"])
        self.taps = int(dc["conv_kernel_size"])
        if self.taps != 2:
            raise ValueError(
                f"DFlash 2 convs of {self.taps} taps: only 2 are implemented"
            )
        self.hidden = cfg.hidden_size
        self.nh, self.nkv, self.hd = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        )
        self.window = int(cfg.sliding_window)
        self.eps = cfg.rms_norm_eps
        self.theta = float(
            (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta", 1e7)
        )
        self.max_positions = int(atom_config.max_model_len) + self.block + 1
        self.num_layers = cfg.num_hidden_layers
        # KV holders only: ModelRunner binds each to its rows of the draft pool
        # (``k_cache`` / ``v_cache``); this model's own kernels read and write them.
        self.attn = nn.ModuleList(
            Attention(
                self.nh, self.hd, self.hd**-0.5, num_kv_heads=self.nkv,
                kv_cache_dtype=atom_config.kv_cache_dtype, layer_num=layer_offset + i,
                prefix=f"layers.{i}.self_attn.attn", per_layer_sliding_window=self.window,
            )
            for i in range(self.num_layers)
        )  # fmt: skip
        self.w = {}
        self.embed = None
        self.lm_head = None
        self.cos = self.sin = None

    # ---- weights ----------------------------------------------------------

    def load_checkpoint(
        self, path: str, embed: torch.Tensor, lm_head: torch.Tensor, device
    ) -> None:
        """The draft's tensors from ``path``/model.safetensors (bf16, fused per
        layer the way the passes use them); ``embed`` / ``lm_head`` are the
        target's [vocab, hidden] weights."""
        from safetensors.torch import load_file

        raw = {
            k: v.to(device)
            for k, v in load_file(str(Path(path) / "model.safetensors")).items()
        }
        w = {
            "fc": raw["fc.weight"], "hidden_norm": raw["hidden_norm.weight"], "norm": raw["norm.weight"],
            "hp": raw["candidate_selector.hidden_projection.weight"],
            "pred_cb": raw["candidate_selector.predecessor_codebook"],
            "succ_cb": raw["candidate_selector.successor_codebook"],
        }  # fmt: skip
        for i in range(self.num_layers):
            p = f"layers.{i}."
            a = p + "self_attn."
            w[i] = {
                "ln1": raw[p + "input_layernorm.weight"], "ln2": raw[p + "post_attention_layernorm.weight"],
                "qn": raw[a + "q_norm.weight"], "kn": raw[a + "k_norm.weight"],
                "qkv": torch.cat([raw[a + "q_proj.weight"], raw[a + "k_proj.weight"], raw[a + "v_proj.weight"]]),
                "kv": torch.cat([raw[a + "k_proj.weight"], raw[a + "v_proj.weight"]]),
                "o": raw[a + "o_proj.weight"],
                "gu": torch.cat([raw[p + "mlp.gate_proj.weight"], raw[p + "mlp.up_proj.weight"]]),
                "d": raw[p + "mlp.down_proj.weight"],
                "akp": raw[p + "attention_conv.kernel_projection.weight"],
                "abase": raw[p + "attention_conv.base_kernel"],
                "mkp": raw[p + "mlp_conv.kernel_projection.weight"],
                "mbase": raw[p + "mlp_conv.base_kernel"],
            }  # fmt: skip
        self.w = w
        self.embed, self.lm_head = embed, lm_head
        inv = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.hd, 2, device=device, dtype=torch.float32)
                / self.hd
            )
        )
        fr = torch.outer(
            torch.arange(self.max_positions, device=device, dtype=torch.float32), inv
        )
        emb = torch.cat([fr, fr], -1)
        self.cos, self.sin = emb.cos().contiguous(), emb.sin().contiguous()
        self.mask_ids = torch.full(
            (self.block - 1,), self.mask_id, dtype=torch.int64, device=device
        )

    # ---- pieces ------------------------------------------------------------

    def _store(self, x, pos, slots, layer, q_out=None) -> None:
        """Norm + RoPE the [q |] k | v rows ``x`` at ``pos``; q -> ``q_out``,
        k / v -> layer ``layer``'s cache at ``slots``."""
        a = self.attn[layer]
        kc, vc = a.k_cache, a.v_cache
        nq = 0 if q_out is None else self.nh
        L = self.w[layer]
        _norm_rope_store_kernel[(x.shape[0], nq + 2 * self.nkv)](
            x, x.stride(0), pos, L["qn"], L["kn"], self.cos, self.sin,
            q_out if q_out is not None else x, kc, vc, slots,
            NQ=nq, NKV=self.nkv, HD=self.hd, BS=kc.shape[3], XV=kc.shape[4], eps=self.eps, num_warps=1,
        )  # fmt: skip

    def _conv(self, x, delta, base):
        """Block-local dynamic grouped conv over [B * block, H] rows: row t of a
        block mixes x[t] and x[t - 1] of the same block."""
        n = x.shape[0]
        c = base[None].float() + delta.float().repeat_interleave(
            self.group, -1
        )  # [n, taps, H]
        xf = x.float()
        out = c[:, 0] * xf
        prev = torch.zeros_like(xf)
        prev[1:] = xf[:-1]
        first = (torch.arange(n, device=x.device) % self.block) == 0
        prev[first] = 0
        out = out + c[:, 1] * prev
        return out.to(torch.bfloat16)

    def _attend(self, q, layer, end, tables):
        """q [B * block, nh, hd] -> [B * block, nh * hd]: every row of request b
        attends to keys (end[b] - window, end[b]] of its own sequence."""
        a = self.attn[layer]
        kc, vc = a.k_cache, a.v_cache
        n = q.shape[0]
        po = torch.empty(
            n * self.nh * SPLITS, self.hd, dtype=torch.float32, device=q.device
        )
        pml = torch.empty(n * self.nh * SPLITS, 2, dtype=torch.float32, device=q.device)
        _gqa_attn_split_kernel[(self.nh, n, SPLITS)](
            q, kc, vc, end, tables, tables.stride(0), po, pml, self.hd**-0.5,
            ROWS=self.block, H=self.nh, KVH=self.nkv, HD=self.hd, BS=kc.shape[3], X=kc.shape[4],
            WIN=self.window, SPLITS=SPLITS, CH=triton.cdiv(self.window, SPLITS), BLOCK=64, num_warps=4,
        )  # fmt: skip
        o = torch.empty(n, self.nh * self.hd, dtype=torch.bfloat16, device=q.device)
        _attn_combine_kernel[(self.nh, n)](
            po, pml, o, H=self.nh, HD=self.hd, SPLITS=SPLITS, num_warps=4
        )
        return o

    # ---- the two passes ------------------------------------------------------

    def context(
        self, aux: torch.Tensor, pos: torch.Tensor, slots: torch.Tensor
    ) -> None:
        """Record the target features ``aux`` [n, layers * H] of positions
        ``pos`` [n] into every layer's KV at ``slots`` [n]."""
        ch = rms(aux @ self.w["fc"].T, self.w["hidden_norm"], self.eps)
        for i in range(self.num_layers):
            kv = ch @ self.w[i]["kv"].T
            self._store(kv, pos, slots, i)

    def draft(
        self,
        anchors: torch.Tensor,
        p0: torch.Tensor,
        tables: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """One block per request: anchor ``anchors[b]`` at position ``p0[b]``
        (block table ``tables[b]``) -> its drafts ``out[b, :]`` (at most
        block - 1)."""
        B, T, H = anchors.shape[0], self.block, self.hidden
        dev = anchors.device
        ids = torch.cat(
            [anchors.view(B, 1).to(torch.int64), self.mask_ids.expand(B, T - 1)], 1
        ).view(-1)
        ar = torch.arange(T, device=dev)
        pos = (p0.view(B, 1).to(torch.int64) + ar).view(-1)
        bs = self.attn[0].k_cache.shape[3]
        slots = (
            tables.gather(1, (pos.view(B, T) // bs)).to(torch.int64).view(-1) * bs
            + pos % bs
        )
        end = p0.to(torch.int64) + T - 1
        n = B * T
        bf = torch.bfloat16
        hidden = self.embed[ids]
        residual = torch.empty(n, H, dtype=bf, device=dev)
        x = torch.empty(n, H, dtype=bf, device=dev)
        hb = triton.next_power_of_2(H)
        G = H // self.group
        for i in range(self.num_layers):
            L = self.w[i]
            # residual (+)= hidden; x = rms(residual) * ln1
            _rms_kernel[(n,)](
                hidden, residual, L["ln1"], x, residual, N=H, BLOCK=hb, HAS_RES=i > 0, eps=self.eps, num_warps=8,
            )  # fmt: skip
            if i == 0:
                residual.copy_(hidden)
            co = x @ L["akp"].T
            xc = self._conv_fused(x, co, L["abase"], 0, G)
            qkv = xc @ L["qkv"].T
            q = torch.empty(n, self.nh, self.hd, dtype=bf, device=dev)
            self._store(qkv, pos, slots, i, q_out=q)
            a = self._conv_fused(
                self._attend(q, i, end, tables) @ L["o"].T, co, L["abase"], 1, G
            )
            _rms_kernel[(n,)](
                a,
                residual,
                L["ln2"],
                x,
                residual,
                N=H,
                BLOCK=hb,
                HAS_RES=True,
                eps=self.eps,
                num_warps=8,
            )
            cm = x @ L["mkp"].T
            gu = self._conv_fused(x, cm, L["mbase"], 0, G) @ L["gu"].T
            inter = gu.shape[1] // 2
            act = torch.empty(n, inter, dtype=bf, device=dev)
            sb = min(1024, inter)
            _silu_mul_kernel[(n, inter // sb)](gu, act, N=inter, BLOCK=sb, num_warps=4)
            hidden = self._conv_fused(act @ L["d"].T, cm, L["mbase"], 1, G)
        final = torch.empty(n, H, dtype=bf, device=dev)
        _rms_kernel[(n,)](
            hidden,
            residual,
            self.w["norm"],
            final,
            residual,
            N=H,
            BLOCK=hb,
            HAS_RES=True,
            eps=self.eps,
            num_warps=8,
        )
        self.last_final = final  # for tests
        sample = final.view(B, T, H)[:, 1:].reshape(B * (T - 1), H)
        logits = sample @ self.lm_head.T  # [B * (T - 1), V] bf16
        V = logits.shape[1]
        ch = 8192
        nch = triton.cdiv(V, ch)
        nkeys = triton.next_power_of_2(nch * self.top_k)
        keys = torch.full(
            (B * (T - 1), nkeys), -(2**63) + 1, dtype=torch.int64, device=dev
        )
        _topk_chunk_kernel[(B * (T - 1), nch)](
            logits, keys, V, NCH=nkeys // self.top_k, CH=ch, K=self.top_k, num_warps=8
        )
        proj = sample @ self.w["hp"].T  # [B * (T - 1), R]
        _select_walk_kernel[(B,)](
            keys, NKEYS=nkeys, PROJ=proj, PRED=self.w["pred_cb"], SUCC=self.w["succ_cb"], ANCHOR=anchors, OUT=out,
            so=out.stride(0), STEPS=out.shape[1], TS=T - 1, K=self.top_k, R=proj.shape[1], num_warps=4,
        )  # fmt: skip

    def _conv_fused(self, x, co, base, part: int, G: int):
        out = torch.empty_like(x)
        cb = min(1024, self.hidden)
        _conv_kernel[(x.shape[0], self.hidden // cb)](
            x, co, base, out, PART=part, T=self.block, N=self.hidden, G=G, GS=self.group, BLOCK=cb, num_warps=4,
        )  # fmt: skip
        return out
