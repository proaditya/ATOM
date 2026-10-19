# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Loading a megakernel code object and launching it, through HIP's module
API (ctypes): the kernel takes one argument, the address of its pointer
table, and runs 256 workgroups of 256 lanes, one a CU, all resident at once.
"""

import ctypes as ct
import json
from pathlib import Path

import torch

GRID = BLOCK = 256

_P = ct.c_void_p
_SIGNATURES = {
    "hipModuleLoad": [ct.POINTER(_P), ct.c_char_p],
    "hipModuleUnload": [_P],
    "hipModuleGetFunction": [ct.POINTER(_P), _P, ct.c_char_p],
    "hipModuleLaunchKernel": [_P] + [ct.c_uint] * 7 + [_P, ct.POINTER(_P), ct.POINTER(_P)],
    "hipModuleLaunchCooperativeKernel": [_P] + [ct.c_uint] * 7 + [_P, ct.POINTER(_P)],
    "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor": [ct.POINTER(ct.c_int), _P, ct.c_int, ct.c_size_t],
    "hipFuncGetAttribute": [ct.POINTER(ct.c_int), ct.c_int, _P],
}  # fmt: skip
# hipFunction_attribute: private (scratch) bytes a lane
_ATTR_PRIVATE = 3

_hip = None


def runtime_library() -> str:
    """The HIP runtime torch uses. Images that ship ROCm as wheels (TheRock, e.g.
    ROCm 10) hold more than one ``libamdhip64.so``: loading another one by name
    gives the process a second HIP runtime, in which torch's streams do not exist
    (a launch fails with hipErrorContextIsDestroyed)."""
    torch.cuda.init()
    with open("/proc/self/maps") as maps:
        for line in maps:
            path = line.split()[-1]
            if "/libamdhip64.so" in path:
                return path
    return "libamdhip64.so"


def hip():
    global _hip
    if _hip is None:
        lib = ct.CDLL(runtime_library())
        for name, args in _SIGNATURES.items():
            getattr(lib, name).argtypes = args
            getattr(lib, name).restype = ct.c_int
        _hip = lib
    return _hip


def _check(name, *args) -> None:
    rc = getattr(hip(), name)(*args)
    if rc:
        raise RuntimeError(f"{name}: HIP error {rc}")


class Megakernel:
    """One loaded code object, launched over a pointer table."""

    def __init__(self, object_dir: Path) -> None:
        meta = json.loads((Path(object_dir) / "object.json").read_text())
        self.meta = meta
        self.rows = meta["rows"]
        self._module = _P()
        _check(
            "hipModuleLoad",
            ct.byref(self._module),
            str(Path(object_dir) / "kernel.hsaco").encode(),
        )
        self._function = _P()
        _check(
            "hipModuleGetFunction",
            ct.byref(self._function),
            self._module,
            meta["symbol"].encode(),
        )
        occupancy = ct.c_int()
        _check(
            "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
            ct.byref(occupancy), self._function, BLOCK, 0,
        )  # fmt: skip
        cus = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        if occupancy.value * cus < GRID:
            raise RuntimeError(
                f"megakernel needs {GRID} resident workgroups, the device holds {occupancy.value * cus}"
            )
        private = ct.c_int()
        _check("hipFuncGetAttribute", ct.byref(private), _ATTR_PRIVATE, self._function)
        if private.value:
            raise RuntimeError("megakernel spills to scratch")
        self._cell = _P()
        self._frame = (_P * 1)(ct.cast(ct.byref(self._cell), _P))

    def launch(
        self, table: torch.Tensor, stream: torch.cuda.Stream | None = None
    ) -> None:
        """Run once over ``table`` on ``stream`` (default: current).

        Cooperative outside a graph capture, which guarantees every workgroup
        is resident; inside one, an ordinary launch of the same grid (a
        graph's kernel node), resident in practice because nothing else runs
        beside a step's kernels."""
        stream = stream or torch.cuda.current_stream()
        self._cell.value = table.data_ptr()
        sp = _P(stream.cuda_stream)
        if torch.cuda.is_current_stream_capturing():
            _check(
                "hipModuleLaunchKernel",
                self._function,
                GRID,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                sp,
                self._frame,
                None,
            )
        else:
            _check(
                "hipModuleLaunchCooperativeKernel",
                self._function,
                GRID,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                sp,
                self._frame,
            )
