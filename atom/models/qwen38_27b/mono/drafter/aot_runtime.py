#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""AOT: the DFlash2 drafter's FlyDSL kernels as exported code objects, run without FlyDSL.

Export (needs FlyDSL + the vendored kernels; compile only, no launches -- the CLI sets COMPILE_ONLY=1,
ARCH=gfx950 and hides the GPUs):

  FLY_KERNELS_ROOT=DIR PROTO_ROOT=DIR python aot_runtime.py export --out DIR --P 40968,131136,8256

writes DIR/<name>.hsaco and DIR/manifest.json: for every kernel its symbol, grid, block, static LDS,
private (scratch) bytes, kernarg size and argument names (every argument is one 64-bit address or
integer), plus the persistent layer kernel's table / mailbox ABI (pointer-table indices, sizes).

Run (no FlyDSL import anywhere on this path):

  dfl = DFlash2Mono(path, embed, lm_head, max_pos, prec=dict(layers='aot8', lm='mx4r128', ctx='aot8'), aot='DIR')

max_pos must equal one of the exported P (the layer kernel's KV / RoPE tables are sized by it).
Kernels: the per-layer persistent kernel (layer 0, middle layers, last layer + final norm) per P; the
MXFP8 x MXFP4 LM head; the context fc (big-K MXFP8) and its activation quantizer; mxfp8_linear's
quantizer + projection for the fused K/V GEMM (8 rows). Triton kernels stay JIT (Triton is in the image).
"""

import ctypes as ct
import json
import math
import os
import sys
from pathlib import Path

import torch

# ----------------------------------------------------------------------------- packing (torch only)


def pack_mxfp8_weight(q):
    """FlyDSL mxfp8_linear's preshuffle of row-major E4M3 [N, K] (kernels/monokernel/packing.py)."""
    rows, k = q.shape
    return (
        q.view(torch.uint8)
        .reshape(rows // 16, 16, k // 64, 4, 16)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(-1)
    )


def pack_mxfp8_scale(scale):
    """FlyDSL mxfp8_linear's preshuffle of per-32 E8M0 scales [N, K / 32] (kernels/monokernel/packing.py)."""
    scale = scale.view(torch.uint8)
    rows, groups = scale.shape
    padded_rows = (rows + 255) // 256 * 256
    padded = torch.zeros(padded_rows, groups, dtype=torch.uint8, device=scale.device)
    padded[:rows] = scale
    return (
        padded.reshape(padded_rows // 32, 2, 16, groups // 8, 2, 4)
        .permute(0, 3, 5, 2, 4, 1)
        .contiguous()
        .view(-1)
    )


def pack_mxfp4(codes_packed, scales):
    """head_kernels.pack_mxfp4: each lane's 32 E2M1 codes of a 128-k step are its natural 32-K block."""
    n, kh = codes_packed.shape
    k = kh * 2
    w4 = (
        codes_packed.view(n // 16, 16, k // 128, 4, 16)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(-1)
    )
    return w4, pack_mxfp8_scale(scales)


# ----------------------------------------------------------------------------- HIP module launches

_P = ct.c_void_p
_SIG = {
    "hipModuleLoadData": [ct.POINTER(_P), _P],
    "hipModuleGetFunction": [ct.POINTER(_P), _P, ct.c_char_p],
    "hipModuleLaunchKernel": [_P]
    + [ct.c_uint] * 7
    + [_P, ct.POINTER(_P), ct.POINTER(_P)],
    "hipModuleLaunchCooperativeKernel": [_P] + [ct.c_uint] * 7 + [_P, ct.POINTER(_P)],
}
_hip = None


def _runtime_library():
    """The HIP runtime torch uses (``kernels/hip.py``: a wheel-installed ROCm can
    hold a second libamdhip64)."""
    from atom.models.qwen38_27b.mono.kernels.hip import runtime_library

    return runtime_library()


def _lib():
    global _hip
    if _hip is None:
        lib = ct.CDLL(_runtime_library())
        for n, a in _SIG.items():
            getattr(lib, n).argtypes = a
            getattr(lib, n).restype = ct.c_int
        _hip = lib
    return _hip


def _check(name, *args):
    rc = getattr(_lib(), name)(*args)
    if rc:
        raise RuntimeError(f"{name}: HIP error {rc}")


class HipKernel:
    """One exported kernel: launch(*int args, stream=None). Persistent kernels (all workgroups resident)
    launch cooperatively outside a graph capture and as ordinary kernel nodes inside one.
    """

    def __init__(self, blob: bytes, meta: dict):
        self.meta = meta
        self._blob = ct.create_string_buffer(
            blob, len(blob)
        )  # (kept alive: the module references it)
        self._module = _P()
        _check("hipModuleLoadData", ct.byref(self._module), ct.cast(self._blob, _P))
        self._fn = _P()
        _check(
            "hipModuleGetFunction",
            ct.byref(self._fn),
            self._module,
            meta["symbol"].encode(),
        )
        n = len(meta["args"])
        self._cells = (ct.c_uint64 * n)()
        base = ct.addressof(self._cells)
        self._frame = (_P * n)(*[_P(base + 8 * i) for i in range(n)])
        self.grid, self.block, self.persistent = (
            meta["grid"],
            meta["block"],
            meta.get("persistent", False),
        )

    def __call__(self, *args, stream=None):
        assert len(args) == len(self._cells), (self.meta["name"], len(args))
        for i, v in enumerate(args):
            self._cells[i] = int(v) & 0xFFFFFFFFFFFFFFFF
        st = stream if stream is not None else torch.cuda.current_stream()
        sp = _P(st.cuda_stream)
        g, b = self.grid, self.block
        if self.persistent and not torch.cuda.is_current_stream_capturing():
            _check(
                "hipModuleLaunchCooperativeKernel",
                self._fn,
                g[0],
                g[1],
                g[2],
                b[0],
                b[1],
                b[2],
                0,
                sp,
                self._frame,
            )
        else:
            _check(
                "hipModuleLaunchKernel",
                self._fn,
                g[0],
                g[1],
                g[2],
                b[0],
                b[1],
                b[2],
                0,
                sp,
                self._frame,
                None,
            )


class Registry:
    """The exported kernels of one manifest directory, loaded on first use."""

    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self._k = {}

    @property
    def abi(self):
        return self.manifest["abi"]

    def has(self, name):
        return name in self.manifest["kernels"]

    def __getitem__(self, name):
        if name not in self._k:
            meta = self.manifest["kernels"][name]
            self._k[name] = HipKernel((self.root / meta["file"]).read_bytes(), meta)
        return self._k[name]


REG = None  # set by DFlash2Mono(aot=DIR) before its base classes build their GEMM operands


class PackedMx8:
    """FlyLinear's interface (8 rows, MXFP8 x MXFP8) on exported kernels: .lin.weight / .lin.scale (packed),
    .y (persistent [8, N] output), __call__(x) for <= 8 rows (quantize + project; the context fc's K uses the
    big-K projection), > 8 rows through the bf16 dequantized reference (kept only when keep_ref).
    """

    def __init__(self, w, rows=8, keep_ref=False):
        from .common import fq_mxfp8, quant_mxfp8w

        self.N, self.K = w.shape
        self.rows = rows
        q, s = quant_mxfp8w(w)
        self.weight, self.scale = pack_mxfp8_weight(q), pack_mxfp8_scale(s)
        del q, s
        self.lin = self
        self.ref = fq_mxfp8(w) if keep_ref else None
        self.y = torch.zeros(rows, self.N, dtype=torch.bfloat16, device="cuda")
        self.xpad = torch.zeros(rows, self.K, dtype=torch.bfloat16, device="cuda")
        self.activation = torch.zeros(32, self.K, dtype=torch.uint8, device="cuda")
        self.activation_scale = torch.zeros(
            32 * self.K // 32, dtype=torch.uint8, device="cuda"
        )

    def project_quantized(self, act, act_scale, out):
        name = (
            f"p8_{self.N}_{self.K}"
            if REG.has(f"p8_{self.N}_{self.K}")
            else f"bk8_{self.N}_{self.K}"
        )
        REG[name](
            act.data_ptr(),
            act_scale.data_ptr(),
            self.weight.data_ptr(),
            self.scale.data_ptr(),
            out.data_ptr(),
        )
        return out

    def __call__(self, x):
        n = x.shape[0]
        if n > self.rows:
            assert (
                self.ref is not None
            ), "PackedMx8 without a reference called with > 8 rows"
            from .common import fq_mxfp8

            return fq_mxfp8(x) @ self.ref.T
        if n < self.rows or not x.is_contiguous():
            self.xpad[:n].copy_(x)
            x = self.xpad
        REG[f"q8_{self.K}"](
            x.data_ptr(), self.activation.data_ptr(), self.activation_scale.data_ptr()
        )
        self.project_quantized(self.activation, self.activation_scale, self.y)
        return self.y[:n]


# ----------------------------------------------------------------------------- export (FlyDSL)


def _extract(launch, nargs, out_dir, name, grid, block, args, persistent=False):
    import hashlib
    import re

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import gpu
    from flydsl._mlir.passmanager import PassManager
    from flydsl.compiler.backends import get_backend
    from flydsl.compiler.jit_function import _create_mlir_context, _extract_isa_text

    flyc.compile[{"fast_fp_math": False, "unsafe_fp_math": False}](
        launch, *([0] * nargs), fx.Stream(None)
    )
    art = launch._last_compiled[1]
    pre = (
        "builtin.module("
        + ",".join(get_backend(arch="gfx950").pipeline_fragments(compile_hints={})[:-1])
        + ")"
    )
    with _create_mlir_context():
        module = ir.Module.parse(art._ir_text)
        objs = [
            gpu.ObjectAttr(o)
            for op in module.body.operations
            if op.operation.name == "gpu.binary"
            for o in ir.ArrayAttr(op.attributes["objects"])
        ]
        blob = bytes(objs[0].object)
        module = ir.Module.parse(art._source_ir)
        PassManager.parse(pre).run(module.operation)
        PassManager.parse("builtin.module(gpu-module-to-binary{format=isa})").run(
            module.operation
        )
        isa = _extract_isa_text(str(module))
    res = {
        k: int(v)
        for k, v in re.findall(r"^\s*\.amdhsa_(\w+)\s+(\d+)\s*$", isa, re.MULTILINE)
    }
    meta_ = {
        k: int(v)
        for k, v in re.findall(
            r"^\s*\.(vgpr_count|sgpr_count|vgpr_spill_count|agpr_count):\s*(\d+)\s*$",
            isa,
            re.MULTILINE,
        )
    }
    sym = re.findall(r"^\s*\.amdhsa_kernel\s+(\S+)", isa, re.MULTILINE)[0]
    (out_dir / (name + ".hsaco")).write_bytes(blob)
    m = {
        "name": name,
        "file": name + ".hsaco",
        "symbol": sym,
        "grid": list(grid),
        "block": list(block),
        "args": args,
        "persistent": persistent,
        "LDS": res.get("group_segment_fixed_size"),
        "private": res.get("private_segment_fixed_size"),
        "kernarg": res.get("kernarg_size"),
        "regs": meta_,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": len(blob),
    }
    print(
        name,
        json.dumps(
            {
                k: m[k]
                for k in (
                    "symbol",
                    "grid",
                    "block",
                    "LDS",
                    "private",
                    "kernarg",
                    "regs",
                )
            }
        ),
        flush=True,
    )
    return m


def export(out_dir, Ps):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, os.environ["FLY_KERNELS_ROOT"])
    sys.path.insert(0, os.environ["PROTO_ROOT"])
    import head_kernels
    import layer_kernel as M
    from kernels.monokernel.mxfp8_linear import (
        build_mxfp8_project,
        build_mxfp8_quantize,
    )

    ks = {}
    for P in Ps:
        for tag, kw in (
            ("l0", {"layer0": True}),
            ("ln", {"layer0": False}),
            ("fn", {"layer0": False, "final_norm": True}),
        ):
            launch = M.build_layer(
                P, kw["layer0"], False, final_norm=kw.get("final_norm", False)
            )
            nm = f"layer_{tag}_P{P}"
            ks[nm] = _extract(
                launch,
                1,
                out_dir,
                nm,
                (M.BLOCKS, 1, 1),
                (M.THREADS, 1, 1),
                ["table"],
                persistent=True,
            )
    N4, K4 = 248320, 5120
    ks[f"lm4_{N4}_{K4}"] = _extract(
        head_kernels.build_mxfp4_project(8, N4, K4, 8),
        5,
        out_dir,
        f"lm4_{N4}_{K4}",
        ((N4 // 16 + head_kernels.WAVES - 1) // head_kernels.WAVES, 1, 1),
        (head_kernels.THREADS, 1, 1),
        ["act", "act_scale", "w4", "s4", "out"],
    )
    ks["bk8_5120_25600"] = _extract(
        head_kernels.build_mxfp8_bigk(8, 5120, 25600, 5, 8),
        5,
        out_dir,
        "bk8_5120_25600",
        (5120 // 16, 1, 1),
        (64 * 5, 1, 1),
        ["act", "act_scale", "w", "s", "out"],
    )
    for K in (25600, 5120):
        ks[f"q8_{K}"] = _extract(
            build_mxfp8_quantize(8, K),
            3,
            out_dir,
            f"q8_{K}",
            (math.ceil(8 * K // 32 / 64), 1, 1),
            (64, 1, 1),
            ["source", "output", "scale_output"],
        )
    for N, K in ((10240, 5120),):
        ks[f"p8_{N}_{K}"] = _extract(
            build_mxfp8_project(8, N, K),
            5,
            out_dir,
            f"p8_{N}_{K}",
            ((N // 16 + 3) // 4, 1, 1),
            (256, 1, 1),
            ["activation", "activation_scale", "weight", "weight_scale", "output"],
        )
    abi = {
        "N_PTRS": M.N_PTRS,
        "MB_PAIRS": M.MB_PAIRS,
        "BLOCKS": M.BLOCKS,
        "THREADS": M.THREADS,
        "TL": M.TL,
        "Ps": list(Ps),
        "ptr": {
            k: getattr(M, k)
            for k in dir(M)
            if k.startswith("P_") and isinstance(getattr(M, k), int)
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "kernels": ks,
                "abi": abi,
                "source": "Qwen3.8-27B DFlash2 drafter (layer_kernel / head_kernels / FlyDSL mxfp8_linear)",
            },
            indent=1,
        )
    )
    print("exported", len(ks), "kernels to", out_dir)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--out", required=True)
    ap.add_argument("--P", default="40968,131136,8256")
    a = ap.parse_args()
    if a.cmd == "export":
        os.environ.update(
            COMPILE_ONLY="1",
            ARCH="gfx950",
            FLYDSL_GPU_ARCH="gfx950",
            FLYDSL_RUNTIME_ENABLE_CACHE="0",
            HIP_VISIBLE_DEVICES="",
            ROCR_VISIBLE_DEVICES="",
            CUDA_VISIBLE_DEVICES="",
        )
        export(a.out, [int(x) for x in a.P.split(",")])
