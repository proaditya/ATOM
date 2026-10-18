#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Convert a Qwen3.8-27B checkpoint to the megakernel's weights and compare
every tensor with the SHA-256 of a reference conversion
(``atom/models/qwen38_27b/mono/native_sha256.json``):

    python tools/qwen38_27b_mono_check_weights.py /models/Qwen/Qwen3.8-27B/1d4bf0f2
"""

import argparse
import json
import time
from pathlib import Path

from atom.models.qwen38_27b.mono import weights as W


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--layers", default="", help="comma list (default: all)")
    a = ap.parse_args()
    ref = json.loads((Path(W.__file__).parent / "native_sha256.json").read_text())
    get = W.safetensors_source(a.model, "cuda")
    layers = [int(x) for x in a.layers.split(",") if x] or range(64)
    bad = []
    t0 = time.time()
    for i in layers:
        got = W.convert_layer(get, i)
        for name, digest in ref["layers"][i].items():
            if W._sha256(got[name]) != digest:
                bad.append(f"layer {i} {name}")
        print(
            i,
            "ok" if not any(b.startswith(f"layer {i} ") for b in bad) else "MISMATCH",
            flush=True,
        )
        del got
    ends = W.convert_endpoints(get)
    for name in ("embedding_weight", "final_norm", "head_weight", "head_scale"):
        if W._sha256(ends[name]) != ref[name]:
            bad.append(name)
    print(json.dumps({"mismatches": bad, "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
