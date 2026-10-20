#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pick hipBLASLt solutions for decode / verify bf16 GEMMs, timed with the weights cache-cold.

aiter's tuners time a GEMM back to back, so its weights sit in the 256 MB
last-level cache; in a model forward every projection's weights arrive from
HBM, and the ranking changes (the default's Qwen3.8-27B gate_up at 16 rows:
76 us cold; the warm-ranked "best" ~70 us cold; the cold-ranked best 57 us).
Here every solution hipBLASLt offers is screened with a 512 MB buffer
written between calls, the best eight are re-timed in a CUDA graph the same
way against the default, and a row is written where one wins by
``--min-gain``. Each kept row is then checked against fp32 at every M aiter's
lookup sends to it (exact M, then M padded to 16, then to a power of two).

    python tools/bf16_gemm_cold_tune.py --shapes 16480,5120 34816,5120 5120,17408 \\
        --m 16,32,48,64,96,128,160,256 --out qwen38_tp1.csv [--merge recipes/configs/qwen38_27b_bf16_tuned_gemm.csv]

Shapes are ``N,K`` of ``y = x @ W^T`` (W [N, K]: a TP shard's own shape).
Use the CSV through ``AITER_CONFIG_GEMM_BF16=<aiter's csv>:<this csv>``.
Progress is kept in ``<out>.json``: re-run the same command after a crash
(a faulting solution kills the process) and it resumes.
"""

import argparse
import csv
import json
import os

import torch
import triton

HDR = ["gfx", "cu_num", "M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle", "libtype", "solidx",
       "splitK", "us", "kernelName", "err_ratio", "tflops", "bw"]  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--shapes", nargs="+", required=True, help="N,K pairs")
    ap.add_argument("--m", default="16,32,48,64,96,128,160,256")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--merge", default="", help="an earlier CSV whose rows stay unless beaten"
    )
    ap.add_argument("--min-gain", type=float, default=0.03)
    a = ap.parse_args()

    import aiter
    from aiter.ops.gemm_op_common import get_padded_m
    from aiter.tuned_gemm import hipb_gemm, tgemm

    aiter.hipb_create_extension()
    shapes = [tuple(int(v) for v in s.split(",")) for s in a.shapes]
    ms = [int(m) for m in a.m.split(",")]
    state_path = a.out + ".json"
    done = {}
    if os.path.exists(state_path):
        with open(state_path) as f:
            done = json.load(f)

    def save():
        with open(state_path, "w") as f:
            json.dump(done, f)

    flush = torch.zeros(512 * 1024 * 1024 // 4, device="cuda")

    def cold(fn):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(10):
                flush.add_(1)
                fn()
        return triton.testing.do_bench(lambda: g.replay(), warmup=3, rep=25) * 100

    t_flush = cold(lambda: None)
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    for n, k in shapes:
        w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
        for m in ms:
            key = f"{m},{n},{k}"
            if key in done:
                continue
            done[key] = {
                "pending": True
            }  # a fault leaves the mark: the re-run skips the shape
            save()
            x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
            ref = x.float() @ w.float().t()
            screened = []
            for sol in aiter.hipb_findallsols(x, w.t()):
                try:
                    if (
                        (hipb_gemm(x, w, sol).float() - ref).norm() / ref.norm()
                    ).item() > 1e-2:
                        continue
                    flush.add_(1)
                    hipb_gemm(x, w, sol)
                    st.record()
                    for _ in range(4):
                        flush.add_(1)
                        hipb_gemm(x, w, sol)
                    en.record()
                    en.synchronize()
                    screened.append((st.elapsed_time(en) / 4 * 1000 - t_flush, sol))
                except RuntimeError:  # a solution that refuses the shape
                    continue
            screened.sort()
            default = cold(lambda x=x, w=w: tgemm.mm(x, w)) - t_flush
            us, sol = min(
                (cold(lambda x=x, w=w, s=s: hipb_gemm(x, w, s)) - t_flush, s)
                for _, s in screened[:8]
            )
            done[key] = {"default": default, "best": sol, "best_us": us}
            save()
            print(
                f"M {m} N {n} K {k}: default {default:.1f} us, best {us:.1f} us (solution {sol})",
                flush=True,
            )

    rows = {}
    if a.merge and os.path.exists(a.merge):
        with open(a.merge) as f:
            for r in csv.DictReader(f):
                rows[(int(r["M"]), int(r["N"]), int(r["K"]))] = (
                    float(r["us"]),
                    int(r["solidx"]),
                )
    for key, v in done.items():
        if "best" in v and v["best_us"] <= (1 - a.min_gain) * v["default"]:
            mnk = tuple(int(t) for t in key.split(","))
            if mnk not in rows or v["best_us"] < rows[mnk][0]:
                rows[mnk] = (v["best_us"], v["best"])

    # every kept row against fp32 at each M aiter's lookup sends to it
    def served(m0, n, k):
        def lands(m):
            if (m, n, k) in rows:
                return m == m0
            p0 = get_padded_m(m, n, k, 0)
            if p0 == m0:
                return True
            return (p0, n, k) not in rows and get_padded_m(m, n, k, 1) == m0

        return [m for m in range(1, 513) if lands(m)]

    weights = {}
    for (m0, n, k), (_, sol) in sorted(rows.items()):
        w = weights.setdefault(
            (n, k), (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
        )
        for m in served(m0, n, k):
            x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
            ref = x.float() @ w.float().t()
            err = ((hipb_gemm(x, w, sol).float() - ref).norm() / ref.norm()).item()
            if err > 1e-2:
                raise SystemExit(
                    f"solution {sol} (row M {m0} N {n} K {k}) is wrong at M {m}: rel err {err:.2e}"
                )
    with open(a.out, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(HDR)
        for (m, n, k), (us, sol) in sorted(
            rows.items(), key=lambda r: (r[0][1], r[0][2], r[0][0])
        ):
            tflops = 2 * m * n * k / (us * 1e-6) / 1e12
            bw = (n * k + m * k + m * n) * 2 / (us * 1e-6) / 1e9
            out.writerow(["gfx950", torch.cuda.get_device_properties(0).multi_processor_count, m, n, k, False,
                          "torch.bfloat16", "torch.bfloat16", False, False, "hipblaslt", sol, 0, f"{us:.1f}", "", 0.0,
                          f"{tflops:.2f}", f"{bw:.2f}"])  # fmt: skip
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
