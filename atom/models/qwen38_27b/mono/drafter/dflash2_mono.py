# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""the DFlash2 draft with each layer as one persistent FlyDSL launch (kernels_src/layer_kernel.py)
and the fused drafter's head (final norm, LM-head GEMM, top-16, selector walk). A drop-in for the fused drafter: same weights, KV
caches, context path and draft_into(ids8, base) semantics. JIT mode needs FLY_KERNELS_ROOT and the kernel sources
on sys.path (PROTO_ROOT, default kernels_src/); AOT mode needs neither.
"""

import os
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from .common import _ord_key, quant_mxfp4, rms_into
from .dflash2_fast import (
    _key_score,
    _qkv_norm_rope_kernel,
    _selector_kernel,
    _top16_s1,
    _top16_s2,
)
from .dflash2_fused import DFlash2Fused

_kr = os.environ.get("FLY_KERNELS_ROOT")
if _kr and _kr not in sys.path:
    sys.path.insert(0, _kr)
_proto = os.environ.get(
    "PROTO_ROOT", str(Path(__file__).resolve().parent / "kernels_src")
)
if _proto not in sys.path:
    sys.path.insert(0, _proto)


class _Abi:
    """layer_kernel's table / mailbox constants: from the module (JIT) or from an AOT manifest (aot)."""

    def __init__(self, d):
        self.__dict__.update(d)


def _abi_jit():
    import layer_kernel as M

    d = {
        k: getattr(M, k)
        for k in dir(M)
        if k.startswith("P_") or k in ("N_PTRS", "MB_PAIRS", "BLOCKS", "TL")
    }
    return _Abi(d)


@triton.jit
def _selector_pre_kernel(K2, PROJ, PRED, SUCC, IDS):
    """the fast drafter's greedy selector walk with every gather issued up front: the successor rows of all 7 x 16
    candidates and the predecessor rows of the anchor and of every step's candidates do not depend on
    the choices; step s picks its predecessor row from step s - 1's 16 by a masked sum (exact).
    """
    j = tl.arange(0, 16)
    k = tl.arange(0, 256)
    anchor = tl.load(IDS)
    pp = tl.load(PRED + anchor * 256 + k).to(tl.float32)
    prevrows = pp[None, :] + tl.zeros((16, 256), tl.float32)
    choice = 0
    for s in tl.static_range(7):
        key = tl.load(K2 + s * 16 + j)
        cand = 0xFFFFFFFF - (key & 0xFFFFFFFF)
        unary = _key_score(key)
        su = tl.load(SUCC + cand[:, None] * 256 + k[None, :]).to(tl.float32)
        pn = tl.load(PRED + cand[:, None] * 256 + k[None, :]).to(tl.float32)
        if s > 0:
            pp = tl.sum(tl.where((j == choice)[:, None], prevrows, 0.0), 0)
        pr = (
            (pp * tl.load(PROJ + s * 256 + k).to(tl.float32))
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        score = unary + tl.sum(su * pr[None, :], 1).to(tl.bfloat16).to(tl.float32)
        best = tl.max(score, 0)
        choice = tl.min(tl.where(score == best, j, 16), 0)
        tl.store(IDS + 1 + s, tl.sum(tl.where(j == choice, cand, 0), 0))
        prevrows = pn


@triton.jit
def _topk_s2(K1, KO, NB: tl.constexpr, NBP: tl.constexpr, KK: tl.constexpr):
    """Row r: top KK of the NB * 16 stage-1 keys."""
    r = tl.program_id(0)
    i = tl.arange(0, NBP * 16)
    key = tl.load(K1 + r * NB * 16 + i, mask=i < NB * 16, other=-9223372036854775807)
    tl.store(KO + r * KK + tl.arange(0, KK), tl.topk(key, KK))


@triton.jit
def _rescore_kernel(K3, FINAL, LMH, SC, HP, PROJ, KK: tl.constexpr, PER: tl.constexpr):
    """(row s, block cb): cb < KK / PER: exact f32 dot of final[1 + s] (bf16) with PER candidates' bf16
    LM-head rows -> keys; else: PER outputs of the selector projection final[1 + s] @ hp^T (bf16 out,
    as the torch matmul it replaces) -- one launch instead of two."""
    s = tl.program_id(0)
    cb = tl.program_id(1)
    NB: tl.constexpr = KK // PER
    if cb < NB:
        j = cb * PER + tl.arange(0, PER)
        key = tl.load(K3 + s * KK + j)
        idx = (0xFFFFFFFF - (key & 0xFFFFFFFF)).to(tl.int64)
    else:
        j = (cb - NB) * PER + tl.arange(0, PER)
        idx = j.to(tl.int64)
    acc = tl.zeros((PER,), tl.float32)
    for c0 in range(0, 5120, 512):
        c = c0 + tl.arange(0, 512)
        x = tl.load(FINAL + (1 + s) * 5120 + c).to(tl.float32)
        if cb < NB:
            w = tl.load(LMH + idx[:, None] * 5120 + c[None, :]).to(tl.float32)
        else:
            w = tl.load(HP + idx[:, None] * 5120 + c[None, :]).to(tl.float32)
        acc += tl.sum(w * x[None, :], 1)
    if cb < NB:
        tl.store(SC + s * KK + j, _ord_key(acc, idx))
    else:
        tl.store(PROJ + s * 256 + j, acc.to(tl.bfloat16))


@triton.jit
def _top16_of(SC, K2, KK: tl.constexpr):
    s = tl.program_id(0)
    tl.store(
        K2 + s * 16 + tl.arange(0, 16),
        tl.topk(tl.load(SC + s * KK + tl.arange(0, KK)), 16),
    )


@triton.jit
def _select_of_kernel(SC, PROJ, PRED, SUCC, IDS, KK: tl.constexpr):
    """the fast drafter's selector walk, each step's top-16 taken from the KK rescored keys in the same program."""
    j = tl.arange(0, 16)
    k = tl.arange(0, 256)
    prev = tl.load(IDS)
    for s in tl.static_range(7):
        key = tl.topk(tl.load(SC + s * KK + tl.arange(0, KK)), 16)
        cand = 0xFFFFFFFF - (key & 0xFFFFFFFF)
        unary = _key_score(key)
        pr = tl.load(PRED + prev * 256 + k).to(tl.float32) * tl.load(
            PROJ + s * 256 + k
        ).to(tl.float32)
        pr = pr.to(tl.bfloat16).to(tl.float32)
        su = tl.load(SUCC + cand[:, None] * 256 + k[None, :]).to(tl.float32)
        score = unary + tl.sum(su * pr[None, :], 1).to(tl.bfloat16).to(tl.float32)
        best = tl.max(score, 0)
        choice = tl.min(tl.where(score == best, j, 16), 0)
        prev = tl.sum(tl.where(j == choice, cand, 0), 0)
        tl.store(IDS + 1 + s, prev)


class DFlash2Mono(DFlash2Fused):
    def __init__(
        self,
        path,
        embed,
        lm_head,
        max_pos,
        prec=None,
        attn_impl="block",
        timeline=False,
        aot=None,
    ):
        """aot: a directory exported by aot_runtime (code objects + manifest): every FlyDSL kernel then runs from
        it and nothing imports FlyDSL (use prec layers / ctx = 'aot8'; max_pos must be an exported P).
        """
        self.skip_lm4_triton = (
            True  # (the screened head runs on the FlyDSL / exported MXFP4 kernel)
        )
        self.aot = None
        if aot is not None:
            from . import aot_runtime

            aot_runtime.REG = aot_runtime.Registry(aot)
            self.aot = aot_runtime.REG
            assert (
                max_pos in self.aot.abi["Ps"]
            ), f"max_pos {max_pos} not exported ({self.aot.abi['Ps']})"
        super().__init__(path, embed, lm_head, max_pos, prec=prec, attn_impl=attn_impl)
        M = (
            _Abi(
                {
                    **self.aot.abi["ptr"],
                    **{
                        k: self.aot.abi[k]
                        for k in ("N_PTRS", "MB_PAIRS", "BLOCKS", "TL")
                    },
                }
            )
            if self.aot
            else _abi_jit()
        )
        self.M = M
        bf, i32 = torch.bfloat16, {"dtype": torch.int32, "device": "cuda"}
        self.mb = torch.zeros(M.MB_PAIRS * 2, **i32)
        self.step = torch.zeros(1, **i32)
        self.res = torch.zeros(8, 5120, dtype=bf, device="cuda")
        self.m = torch.zeros(8, 5120, dtype=bf, device="cuda")
        self.cm = torch.zeros(8, 1280, dtype=bf, device="cuda")
        self.tl = torch.zeros(5, M.BLOCKS, M.TL, dtype=torch.int64, device="cuda")
        self.keep = []
        self.tables = {}
        if self.aot:
            self.k0, self.kn, self.kf = (
                self.aot[f"layer_{t_}_P{self.P}"] for t_ in ("l0", "ln", "fn")
            )
        else:
            import layer_kernel

            self.k0 = layer_kernel.build_layer(self.P, True, timeline)
            self.kn = layer_kernel.build_layer(self.P, False, timeline)
            self.kf = layer_kernel.build_layer(
                self.P, False, timeline, final_norm=True
            )  # the last layer + the final norm
        # context fc (5120 x 25600, too large a K for mxfp8_linear): big-K MXFP8 projection (bit-exact with MxLinear)
        from . import aot_runtime
        from .common import quant_mxfp8w

        qf, sf = quant_mxfp8w(self.w["fc.weight"])
        self.fc_w, self.fc_s = aot_runtime.pack_mxfp8_weight(
            qf
        ), aot_runtime.pack_mxfp8_scale(sf)
        del qf, sf
        K_ = self.w["fc.weight"].shape[1]
        self.fc_act = torch.zeros(32, K_, dtype=torch.uint8, device="cuda")
        self.fc_as = torch.zeros(32 * K_ // 32, dtype=torch.uint8, device="cuda")
        if self.aot:
            self.fc_q, self.fc_k = self.aot[f"q8_{K_}"], self.aot[f"bk8_5120_{K_}"]
        else:
            import head_kernels
            from kernels.monokernel.mxfp8_linear import build_mxfp8_quantize

            self.fc_q = build_mxfp8_quantize(8, K_)
            self.fc_k = head_kernels.build_mxfp8_bigk(8, 5120, K_, 5, 8)
        self.fc_y = torch.zeros(8, 5120, dtype=torch.bfloat16, device="cuda")
        if (
            self.rescreen
        ):  # lm 'mx4rK': the MXFP8 x MXFP4 head screens the top K, the bf16 head rescores them
            q4, s4 = quant_mxfp4(lm_head)
            self.lm4_w, self.lm4_s = aot_runtime.pack_mxfp4(q4, s4)
            del q4, s4
            if self.aot:
                self.lm4 = self.aot["lm4_{}_{}".format(*lm_head.shape)]
            else:
                import head_kernels

                self.lm4 = head_kernels.build_mxfp4_project(
                    8, lm_head.shape[0], lm_head.shape[1], 8
                )
            self.lm4_y = torch.zeros(
                8, lm_head.shape[0], dtype=torch.bfloat16, device="cuda"
            )
            self.k3 = torch.empty(7 * self.rescreen, dtype=torch.int64, device="cuda")
            self.sc3 = torch.empty(7 * self.rescreen, dtype=torch.int64, device="cuda")
            self.proj = torch.empty(7, 256, dtype=torch.bfloat16, device="cuda")

    def context(self, pos, aux):
        n = pos.numel()
        if n != 8:
            return super().context(pos, aux)
        st = torch.cuda.current_stream()
        x = aux.reshape(8, -1)
        x = x if x.is_contiguous() else x.contiguous()
        self.fc_q(
            x.data_ptr(), self.fc_act.data_ptr(), self.fc_as.data_ptr(), stream=st
        )
        self.fc_k(
            self.fc_act.data_ptr(),
            self.fc_as.data_ptr(),
            self.fc_w.data_ptr(),
            self.fc_s.data_ptr(),
            self.fc_y.data_ptr(),
            stream=st,
        )
        ch = self.buf["ch"][:8]
        rms_into(self.fc_y, self.w["hidden_norm.weight"], ch)
        kv = self.kvall(ch)
        _qkv_norm_rope_kernel[(8, 16, 5)](
            kv,
            kv.stride(0),
            pos,
            self.kn_all,
            self.kn_all,
            self.cos,
            self.sin,
            kv,
            self.kptr,
            self.vptr,
            self.P,
            NQ=0,
            eps=1e-6,
            num_warps=1,
        )

    def _c(self, t):
        t = t.contiguous()
        self.keep.append(t)
        return t.data_ptr()

    def _tables(self, ids8, base):
        key = (ids8.data_ptr(), base.data_ptr())
        if key not in self.tables:
            tabs = []
            M = self.M
            for li, L in enumerate(self.layers):
                Lp = self.layers[li - 1] if li else L
                e = [0] * M.N_PTRS
                e[M.P_EMBED] = self.embed.data_ptr()
                e[M.P_IDS] = ids8.data_ptr()
                e[M.P_RES] = self.res.data_ptr()
                e[M.P_M] = self.m.data_ptr()
                e[M.P_CM] = self.cm.data_ptr()
                e[M.P_MBASE1_PREV] = self._c(Lp["mbase"][1])
                e[M.P_LN1] = self._c(L["ln1"])
                e[M.P_LN2] = self._c(L["ln2"])
                for nm, pw, ps in (
                    ("akp", M.P_AKP_W, M.P_AKP_S),
                    ("qkv", M.P_QKV_W, M.P_QKV_S),
                    ("o", M.P_O_W, M.P_O_S),
                    ("mkp", M.P_MKP_W, M.P_MKP_S),
                    ("gu", M.P_GU_W, M.P_GU_S),
                    ("d", M.P_D_W, M.P_D_S),
                ):
                    e[pw] = L[nm].lin.weight.data_ptr()
                    e[ps] = L[nm].lin.scale.data_ptr()
                e[M.P_ABASE0] = self._c(L["abase"][0])
                e[M.P_ABASE1] = self._c(L["abase"][1])
                e[M.P_MBASE0] = self._c(L["mbase"][0])
                e[M.P_QN] = self._c(L["qn"])
                e[M.P_KN] = self._c(L["kn"])
                e[M.P_COS] = self._c(self.cos)
                e[M.P_SIN] = self._c(self.sin)
                e[M.P_KC] = L["k"].data_ptr()
                e[M.P_VC] = L["v"].data_ptr()
                e[M.P_BASE] = base.data_ptr()
                e[M.P_M_OUT] = self.m.data_ptr()
                e[M.P_CM_OUT] = self.cm.data_ptr()
                e[M.P_MB] = self.mb.data_ptr()
                e[M.P_STEP] = self.step.data_ptr()
                e[M.P_TL] = self.tl[li].data_ptr()
                e[M.P_LAYER] = li
                e[M.P_MBASE1] = self._c(L["mbase"][1])
                e[M.P_NORM] = self._c(self.w["norm.weight"])
                e[M.P_LMQ] = self.q[5120][0].data_ptr()
                e[M.P_LMS] = self.q[5120][1].data_ptr()
                e[M.P_FINAL] = self.buf["final"].data_ptr()
                tabs.append(torch.tensor(e, dtype=torch.int64, device="cuda"))
            self.tables[key] = tabs
        return self.tables[key]

    def layers_into(self, ids8, base):
        st = torch.cuda.current_stream()
        for li, tab in enumerate(self._tables(ids8, base)):
            (self.k0 if li == 0 else self.kf if li == 4 else self.kn)(
                tab.data_ptr(), stream=st
            )

    def draft_into(self, ids8, base):
        """base must not repeat between drafts unless self.step is bumped (the mailbox tag uses both)."""
        self.layers_into(
            ids8, base
        )  # (the last layer also writes final + the LM-head operand)
        b = self.buf
        final = b["final"]
        if self.rescreen:
            aq, asc = self.q[5120]
            self.lm4(
                aq.data_ptr(),
                asc.data_ptr(),
                self.lm4_w.data_ptr(),
                self.lm4_s.data_ptr(),
                self.lm4_y.data_ptr(),
                stream=torch.cuda.current_stream(),
            )
            lg = self.lm4_y[1:]
            _top16_s1[(7, 122)](lg, b["k1"], 248320, NB=122, BLOCK=2048, num_warps=8)
            KK = self.rescreen
            _topk_s2[(7,)](b["k1"], self.k3, NB=122, NBP=128, KK=KK, num_warps=8)
            _rescore_kernel[(7, KK // 16 + 256 // 16)](
                self.k3,
                final,
                self.lm_bf16,
                self.sc3,
                self.hp,
                self.proj,
                KK=KK,
                PER=16,
                num_warps=4,
            )
            _top16_of[(7,)](
                self.sc3, b["k2"], KK=KK, num_warps=4
            )  # (folding this into the walk: 25 vs ~17 us)
            _selector_kernel[(1,)](
                b["k2"],
                self.proj,
                self.w["candidate_selector.predecessor_codebook"],
                self.w["candidate_selector.successor_codebook"],
                ids8,
                num_warps=4,
            )
            return
        else:
            lg = self._gemm(self.lm, self.q[5120])[1:]
            _top16_s1[(7, 122)](lg, b["k1"], 248320, NB=122, BLOCK=2048, num_warps=8)
            _top16_s2[(7,)](b["k1"], b["k2"], NB=122, NBP=128, num_warps=8)
        proj = final[1:] @ self.hp.T
        _selector_kernel[(1,)](
            b["k2"],
            proj,
            self.w["candidate_selector.predecessor_codebook"],
            self.w["candidate_selector.successor_codebook"],
            ids8,
            num_warps=4,
        )

    def draft(
        self, anchor, p0
    ):  # offline evaluation: persistent ids / base buffers (one table set), step bumped
        if not hasattr(self, "_ids"):
            self._ids = torch.zeros(8, dtype=torch.int64, device="cuda")
            self._base = torch.zeros(1, dtype=torch.int64, device="cuda")
        self._ids.zero_()
        self._ids[0] = anchor
        self._base.fill_(int(p0))
        self.step.add_(1)
        self.draft_into(self._ids, self._base)
        self.alts = [
            self._ids[1 + s : 2 + s].clone() for s in range(7)
        ]  # (no runner-ups on this path)
        return self._ids[1:].clone()
