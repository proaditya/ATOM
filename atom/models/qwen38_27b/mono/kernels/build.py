# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Building the megakernel's code objects.

The kernels are FlyDSL programs qualified with one pinned FlyDSL frontend
(git 19418894 over the ``flydsl==0.4.0.dev963`` native wheel). A build runs in a
subprocess whose ``PYTHONPATH`` is that frontend (``ATOM_QWEN38_MK_FLYDSL``), compile
only, and leaves a code object (``kernel.hsaco``) and its metadata in a cache
directory keyed by every source byte. Serving loads the code object through
HIP (``hip.py``) and needs no FlyDSL at all.

A variant is a kernel source plus the generators applied to it, in order
(``gen_replay_records.py``: the GDN replay records a speculative commit
replays).
"""

import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger("atom")

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Generator:
    """A source transform: ``python FILE SRC OUT BUILDER SYMBOL SUFFIX`` with
    ``env``, renaming the builder to ``<prefix><suffix>`` and the kernel to
    ``qwen<suffix>_<kind>``."""

    file: str
    suffix: str
    prefix: str
    kind: str
    env: tuple[tuple[str, str], ...] = ()


def _replay(suffix):
    return Generator("gen_replay_records.py", suffix, "build_replay", "replay")


@dataclass(frozen=True)
class Variant:
    source: str
    builder: str
    symbol: str
    rows: int
    generators: tuple[Generator, ...]


VARIANTS = {
    # 8 rows + replay records
    "m8": Variant("verify_m8.py", "build_verify_m8", "qwen38_verify_m8", 8, (_replay("_m8"),)),
    # 8 rows + replay records, row-octet attention above 16K positions
    "m8_long": Variant("verify_m8_long.py", "build_verify_m8_long", "qwen38_verify_m8_long", 8, (_replay("_m8_long"),)),
    # 16 rows + replay records
    "m16": Variant("verify_m16.py", "build_verify_m16", "qwen38_verify_m16", 16, (_replay("_m16"),)),
}  # fmt: skip

# the build recipe: one Uint64 pointer argument (the table), no fast math
_WORKER = r"""
import hashlib, importlib.util, json, re, sys
from pathlib import Path
spec, out = json.loads(sys.argv[1]), Path(sys.argv[2])
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import gpu
from flydsl.compiler.jit_function import _create_mlir_context
from flydsl.utils import env
assert env.compile.compile_only and env.compile.arch == 'gfx950'
m = importlib.util.spec_from_file_location('qwen38_mk', spec['path'])
mod = importlib.util.module_from_spec(m); m.loader.exec_module(mod)
launch = getattr(mod, spec['builder'])(64)
ptr = [flyc.from_c_void_p(fx.Uint64, None, assumed_align=16)]
flyc.compile[{'fast_fp_math': False, 'unsafe_fp_math': False}](launch, *ptr, fx.Stream(None))
art = launch._last_compiled[1]
with _create_mlir_context():
    module = ir.Module.parse(art._ir_text)
    objs = [gpu.ObjectAttr(o) for op in module.body.operations if op.operation.name == 'gpu.binary'
            for o in ir.ArrayAttr(op.attributes['objects'])]
    binary = bytes(objs[0].object)
(out / 'kernel.hsaco').write_bytes(binary)
kds = sorted(set(re.findall(rb'([A-Za-z_][A-Za-z0-9_$.]*)\.kd\x00', binary)))
assert len(kds) == 1, kds
print(json.dumps(dict(sha256=hashlib.sha256(binary).hexdigest(), bytes=len(binary), symbol=kds[0].decode())))
"""


def _source_key(variant: Variant) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(asdict(variant), sort_keys=True).encode())
    for name in (variant.source, *(g.file for g in variant.generators)):
        h.update((HERE / name).read_bytes())
    h.update(_WORKER.encode())
    return h.hexdigest()[:16]


def cache_root() -> Path:
    return Path(
        os.environ.get("ATOM_QWEN38_MK_CACHE")
        or (Path.home() / ".cache/atom/qwen38_27b_mono")
    )


def generate(variant: Variant, workdir: Path) -> tuple[Path, str, str]:
    """Apply the variant's generators -> (source path, builder, symbol)."""
    src = HERE / variant.source
    builder, symbol = variant.builder, variant.symbol
    for g in variant.generators:
        out = workdir / f"{src.stem}_{g.suffix}.py"
        subprocess.run(
            [sys.executable, str(HERE / g.file), str(src), str(out), builder, symbol, g.suffix],
            check=True, capture_output=True, text=True, env=dict(os.environ, **dict(g.env)),
        )  # fmt: skip
        src, builder, symbol = out, f"{g.prefix}{g.suffix}", f"qwen{g.suffix}_{g.kind}"
    return src, builder, symbol


def build(name: str, flydsl_path: str | None = None) -> Path:
    """The cached object directory of variant ``name``, built if missing."""
    variant = VARIANTS[name]
    target = cache_root() / f"{name}-{_source_key(variant)}"
    if (target / "object.json").exists():
        return target
    flydsl_path = flydsl_path or os.environ.get("ATOM_QWEN38_MK_FLYDSL")
    if not flydsl_path:
        raise RuntimeError(
            f"megakernel {name} is not built ({target}) and ATOM_QWEN38_MK_FLYDSL "
            "names no pinned FlyDSL frontend to build it with"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        work = Path(tmp)
        src, builder, _ = generate(variant, work)
        env = dict(
            os.environ,
            COMPILE_ONLY="1", ARCH="gfx950", FLYDSL_GPU_ARCH="gfx950",
            FLYDSL_RUNTIME_ENABLE_CACHE="0", PYTHONDONTWRITEBYTECODE="1",
            HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="",
            PYTHONPATH=flydsl_path,
        )  # fmt: skip
        spec = {"path": str(src), "builder": builder}
        logger.info("building Qwen3.8-27B megakernel %s (a few minutes)", name)
        run = subprocess.run(
            [sys.executable, "-c", _WORKER, json.dumps(spec), str(work)],
            env=env, capture_output=True, text=True, timeout=3600, check=False,
        )  # fmt: skip
        if run.returncode:
            raise RuntimeError(f"megakernel {name} build failed:\n{run.stderr[-4000:]}")
        meta = json.loads(run.stdout.strip().splitlines()[-1])
        meta.update(name=name, rows=variant.rows, source_key=target.name)
        (work / "object.json").write_text(json.dumps(meta, indent=1))
        (work / "source.py").write_bytes(src.read_bytes())
        final = work / "final"
        final.mkdir()
        for f in ("kernel.hsaco", "object.json", "source.py"):
            os.replace(work / f, final / f)
        os.replace(final, target)
    return target


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="build the Qwen3.8-27B megakernel code objects"
    )
    ap.add_argument("names", nargs="*", default=sorted(VARIANTS))
    ap.add_argument("--flydsl", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    for n in a.names:
        print(n, build(n, a.flydsl))
