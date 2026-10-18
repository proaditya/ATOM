# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DFlash2 draft head (incoai/Qwen3.8-27B-DFlash2) in PyTorch, for the FlyDSL megakernel verifier.

The reference implementation:
- context: the target residual stream after layers 5/19/33/47/61 of every committed position ->
  fc -> hidden_norm -> each draft layer's k_proj / v_proj (k_norm, RoPE) into that layer's KV cache;
- draft: one block [anchor, mask x 7] at positions p0 .. p0 + 7 through five non-causal layers
  (every row sees the window [p0 + 7 - 2047, p0 + 7]), each with block-local dynamic grouped
  convolutions around attention and MLP; final norm; target LM head on rows 1..7 -> top-16
  candidates; a low-rank selector walks the most coherent candidate path from the anchor.
Norm weights are plain (Qwen3 RMSNorm); embeddings / LM head are the target's.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

H = 5120


def rms(x, w, eps=1e-6):  # Qwen3 RMSNorm (plain weight)
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(
        torch.bfloat16
    ) * w


class DFlash2:
    def __init__(self, path, embed, lm_head, max_pos, mx=False, prec=None):
        from safetensors.torch import load_file

        w = {
            k: v.to("cuda")
            for k, v in load_file(str(Path(path) / "model.safetensors")).items()
        }
        self.w = w
        self.embed = embed
        self.P = max_pos
        self.mx = mx
        self.nh, self.nkv, self.hd, self.win, self.block = 32, 8, 128, 2048, 8
        self.mask_id = 248070
        # prec: {'layers' | 'lm' | 'ctx': 'bf16' | 'mx4' | 'mx8'}; mx=True is mx4 everywhere
        prec = dict(prec or {})
        [prec.setdefault(k_, "mx4" if mx else "bf16") for k_ in ("layers", "lm", "ctx")]
        prec.setdefault(
            "mlp", prec["layers"]
        )  # gate/up + down (+ the MLP conv's kernel projection); 'layers' = attention side
        # lm 'mx4rK' (offline only): MXFP4 LM head screens the top K, the bf16 head rescores those K -> top-16
        self.prec = prec
        from .common import MxLinear

        def mk(kind):
            if prec[kind] == "bf16":
                return lambda t: (lambda x, t=t: x @ t.T)
            if prec[kind] == "fly8":  # FlyDSL MXFP8 x MXFP8 kernel (8 rows)
                from .common import FlyLinear

                return lambda t: (
                    FlyLinear(t) if t.shape[1] <= 20480 else MxLinear(t, w8=True)
                )
            if (
                prec[kind] == "aot8"
            ):  # the same kernels as exported code objects (aot; no FlyDSL import)
                from . import aot_runtime

                return lambda t: aot_runtime.PackedMx8(t, keep_ref=(kind == "ctx"))
            return lambda t: MxLinear(t, w8=prec[kind] == "mx8")

        lin = mk("layers")
        lin_mlp = mk("mlp")
        lin_ctx = mk("ctx")
        self.rescreen = int(prec["lm"][4:]) if prec["lm"].startswith("mx4r") else 0
        lin_lm = (
            (lambda t: None)
            if self.rescreen and getattr(self, "skip_lm4_triton", False)
            else (lambda t: MxLinear(t, w8=False)) if self.rescreen else mk("lm")
        )
        self.layers = []
        for i in range(5):
            p = f"layers.{i}."
            L = {
                "ln1": w[p + "input_layernorm.weight"],
                "ln2": w[p + "post_attention_layernorm.weight"],
                "qn": w[p + "self_attn.q_norm.weight"],
                "kn": w[p + "self_attn.k_norm.weight"],
                "qkv": lin(
                    torch.cat([w[p + f"self_attn.{x}_proj.weight"] for x in "qkv"], 0)
                ),
                "kv": lin_ctx(
                    torch.cat(
                        [
                            w[p + "self_attn.k_proj.weight"],
                            w[p + "self_attn.v_proj.weight"],
                        ],
                        0,
                    )
                ),
                "o": lin(w[p + "self_attn.o_proj.weight"]),
                "gu": lin_mlp(
                    torch.cat(
                        [w[p + "mlp.gate_proj.weight"], w[p + "mlp.up_proj.weight"]], 0
                    )
                ),
                "d": lin_mlp(w[p + "mlp.down_proj.weight"]),
                "akp": lin(w[p + "attention_conv.kernel_projection.weight"]),
                "abase": w[p + "attention_conv.base_kernel"],
                "mkp": lin_mlp(w[p + "mlp_conv.kernel_projection.weight"]),
                "mbase": w[p + "mlp_conv.base_kernel"],
                "k": torch.zeros(
                    self.nkv, max_pos, self.hd, dtype=torch.bfloat16, device="cuda"
                ),
                "v": torch.zeros(
                    self.nkv, max_pos, self.hd, dtype=torch.bfloat16, device="cuda"
                ),
            }
            self.layers.append(L)
        self.fc = lin_ctx(w["fc.weight"])
        self.lm = lin_lm(lm_head)
        self.lm_bf16 = lm_head
        self.hp = w["candidate_selector.hidden_projection.weight"]
        inv = 1.0 / (
            1e7 ** (torch.arange(0, 128, 2, device="cuda", dtype=torch.float32) / 128)
        )
        fr = torch.outer(torch.arange(max_pos, device="cuda", dtype=torch.float32), inv)
        emb = torch.cat([fr, fr], -1)
        self.cos, self.sin = emb.cos(), emb.sin()
        self.ar8 = torch.arange(8, device="cuda")
        self.arw = torch.arange(self.win, device="cuda")
        self.mask_ids = torch.full((7,), self.mask_id, dtype=torch.int64, device="cuda")

    # ---- pieces
    def rope(self, x, pos):  # x [n, heads, 128]
        c = self.cos[pos][:, None, :]
        s = self.sin[pos][:, None, :]
        x1, x2 = x[..., :64].float(), x[..., 64:].float()
        return (x.float() * c + torch.cat([-x2, x1], -1) * s).to(torch.bfloat16)

    @staticmethod
    def conv(x, delta, base):
        """Block-local dynamic grouped conv: out[t] = c0[t] x[t] + c1[t] x[t-1]; c = base + delta (per group of 16)."""
        c = base[None].float() + delta.float().repeat_interleave(16, -1)  # [n, taps, H]
        xf = x.float()
        out = c[:, 0] * xf
        out[1:] += c[1:, 1] * xf[:-1]
        return out.to(torch.bfloat16)

    def context(self, pos, aux):
        """Record the target features of committed positions pos [n] (aux [n, 5, H]) into every layer's KV."""
        n = pos.numel()
        ch = rms(self.fc(aux.reshape(n, 5 * H)), self.w["hidden_norm.weight"])
        for L in self.layers:
            kv = L["kv"](ch)
            k = self.rope(rms(kv[:, :1024].reshape(n, 8, 128), L["kn"]), pos)
            v = kv[:, 1024:].reshape(n, 8, 128)
            L["k"].index_copy_(1, pos, k.transpose(0, 1))
            L["v"].index_copy_(1, pos, v.transpose(0, 1))

    def attend(
        self, q, L, terminal
    ):  # q [8, 32, 128]; every row sees keys (terminal - win, terminal] (static shape)
        kp = terminal - self.win + 1 + self.arw
        idx = kp.clamp(0, self.P - 1)
        K = L["k"].index_select(1, idx).float()
        V = L["v"].index_select(1, idx).float()  # [8, win, 128]
        qg = q.float().view(8, 8, 4, 128)
        sc = torch.einsum("rkgd,ktd->rkgt", qg, K) * (128**-0.5)
        sc = sc.masked_fill((kp < 0)[None, None, None, :], float("-inf"))
        o = torch.einsum("rkgt,ktd->rkgd", torch.softmax(sc, -1), V)
        return o.reshape(8, 4096).to(torch.bfloat16)

    def draft_into(self, ids8, base):
        """Graph-safe: anchor ids8[0] at position base[0] -> 7 drafts written to ids8[1:8]."""
        ids = torch.cat([ids8[:1], self.mask_ids])
        pos = base + self.ar8
        terminal = base + 7
        hidden = self.embed[ids]
        residual = None
        for L in self.layers:
            residual = (
                hidden
                if residual is None
                else (hidden.float() + residual.float()).to(torch.bfloat16)
            )
            x = rms(residual, L["ln1"])
            co = L["akp"](x).reshape(8, 2, 2, 320)
            xc = self.conv(x, co[:, 0], L["abase"][0])
            qkv = L["qkv"](xc)
            q = self.rope(rms(qkv[:, :4096].reshape(8, 32, 128), L["qn"]), pos)
            k = self.rope(rms(qkv[:, 4096:5120].reshape(8, 8, 128), L["kn"]), pos)
            v = qkv[:, 5120:].reshape(8, 8, 128)
            L["k"].index_copy_(1, pos, k.transpose(0, 1))
            L["v"].index_copy_(1, pos, v.transpose(0, 1))
            a = self.conv(L["o"](self.attend(q, L, terminal)), co[:, 1], L["abase"][1])
            residual = (a.float() + residual.float()).to(torch.bfloat16)
            x = rms(residual, L["ln2"])
            cm = L["mkp"](x).reshape(8, 2, 2, 320)
            g, u = L["gu"](self.conv(x, cm[:, 0], L["mbase"][0])).chunk(2, -1)
            hidden = self.conv(
                L["d"]((F.silu(g) * u).contiguous()), cm[:, 1], L["mbase"][1]
            )
        final = rms(
            (hidden.float() + residual.float()).to(torch.bfloat16),
            self.w["norm.weight"],
        )
        sample = final[1:].contiguous()  # [7, H]
        if self.rescreen:  # MXFP4 screen -> bf16 rescoring of the screened candidates
            scr = self.lm(sample).float().topk(self.rescreen, -1).indices  # [7, K]
            ex = torch.einsum("rh,rkh->rk", sample.float(), self.lm_bf16[scr].float())
            u_, j_ = ex.topk(16, -1)
            unary, cand = u_, scr.gather(1, j_)
        else:
            unary, cand = self.lm(sample).float().topk(16, -1)
        proj = sample @ self.hp.T  # [7, 256]
        pred_cb = self.w["candidate_selector.predecessor_codebook"]
        succ_cb = self.w["candidate_selector.successor_codebook"]
        prev = ids8[:1]
        self.alts = (
            []
        )  # per step, the selector's runner-up tokens given the spine predecessor (tree drafts)
        for s in range(7):
            pr = pred_cb.index_select(0, prev) * proj[s : s + 1]  # [1, 256]
            edge = succ_cb.index_select(0, cand[s]) @ pr[0]  # [16]
            score = unary[s] + edge.float()
            choice = score.argmax().view(1)
            self.alts.append(cand[s][score.topk(3).indices[1:]])
            prev = cand[s].index_select(0, choice)
            ids8[1 + s : 2 + s].copy_(prev)

    def draft(self, anchor, p0):  # eager convenience (offline evaluation)
        ids8 = torch.zeros(8, dtype=torch.int64, device="cuda")
        ids8[0] = anchor
        self.draft_into(ids8, torch.tensor([p0], device="cuda"))
        return ids8[1:]
