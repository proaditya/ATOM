#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# One command, from an ATOM checkout on AMD Instinct MI355X GPUs: reproduce the ATOM rows of
# "EAGLE-4 on AMD Instinct GPUs from day 0" (Qwen3.8-27B, BF16, EAGLE-4 speculative decoding, SPEED-Bench):
#   1. install this ATOM checkout (unless it is already the one installed),
#   2. fetch the model (and the draft, given a Hugging Face id),
#   3. take the SPEED-Bench client and data from an EAGLE-4 checkout,
#   4. for each concurrency: serve with the recipe's flags for it, warm up, run the client,
#   5. print the results next to the published ones.
#
# Usage, from the ATOM checkout, in rocm/atom-dev:nightly_202610131505 (ROCm 7.2.4 with aiter, FlyDSL and
# ATOM's Triton pin; see blog_benchmark/eagle4/README.md):
#   HIP_VISIBLE_DEVICES=0 EAGLE4=<EAGLE-4 checkout> DRAFT=<EAGLE-4 Qwen3.8-27B checkpoint> bash blog_benchmark/eagle4/build_and_repro.sh
#
# Optional environment:
#   MODEL     local checkpoint directory or Hugging Face id     (default Qwen/Qwen3.8-27B)
#   TP        tensor-parallel size: 1, 2 or 4                    (default 1; the first TP visible GPUs, so
#             pick them with HIP_VISIBLE_DEVICES)
#   CONC      concurrent conversations to measure                (default "1 4 16 32")
#   WORK      checkpoints, client and results                    (default ./blog_benchmark/eagle4/work)
#   PORT      server port                                        (default 8000)

set -euo pipefail

cd "$(dirname "$0")/../.."
ATOM_ROOT=$PWD
PY="${PY:-python3}"
WORK="${WORK:-$ATOM_ROOT/blog_benchmark/eagle4/work}"
MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
TP="${TP:-1}"
CONC="${CONC:-1 4 16 32}"
PORT="${PORT:-8000}"
: "${EAGLE4:?set EAGLE4 to an EAGLE-4 repository checkout (SPEED-Bench client and data)}"
: "${DRAFT:?set DRAFT to the EAGLE-4 Qwen3.8-27B checkpoint (directory or Hugging Face id)}"
export AITER_LOG_LEVEL=WARNING
mkdir -p "$WORK"
step() { printf '\n== %s\n' "$*"; }

step "0. checks"
if command -v rocminfo > /dev/null; then rocminfo 2> /dev/null | grep gfx950 > /dev/null || { echo "needs MI355X / MI350X (gfx950)" >&2; exit 1; }; fi
"$PY" - << 'PYEOF'
import sys
import triton
if tuple(int(v) for v in triton.__version__.split("+")[0].split(".")[:2]) < (3, 7):
    sys.exit(f"Triton {triton.__version__}: ATOM needs its pin, triton 3.7.0+amd.rocm7.2.0.git89002410 "
             "(and triton_kernels) from https://pypi.amd.com/triton/release/rocm-7.2.0/simple")
for mod in ("aiter", "flydsl"):
    try:
        __import__(mod)
    except ImportError:
        sys.exit(f"no {mod}: run in the rocm/atom-dev image named in blog_benchmark/eagle4/README.md, which has "
                 "aiter and FlyDSL 0.3.4.1")
PYEOF

step "1. ATOM"
# An image may ship its own ATOM: install this checkout unless it is already the one imported.
"$PY" - "$ATOM_ROOT" 2> /dev/null << 'PYEOF' || "$PY" -m pip install --no-build-isolation --no-deps -e "$ATOM_ROOT"
import os, sys
import atom
sys.exit(0 if os.path.realpath(atom.__file__).startswith(os.path.realpath(sys.argv[1]) + os.sep) else 1)
PYEOF
"$PY" -c "import atom; print('atom', atom.__file__)"

step "2. checkpoints"
fetch() {  # fetch ID_OR_DIR -> prints a local directory
    if [ -d "$1" ]; then echo "$1"; return; fi
    local dst="$WORK/models/$(basename "$1")"
    [ -f "$dst/config.json" ] || "$PY" -c "import sys; from huggingface_hub import snapshot_download; \
snapshot_download(sys.argv[1], local_dir=sys.argv[2])" "$1" "$dst" > /dev/null
    echo "$dst"
}
MODEL=$(fetch "$MODEL")
DRAFT=$(fetch "$DRAFT")
echo "model $MODEL"
echo "draft $DRAFT"

step "3. SPEED-Bench client and data (EAGLE-4 repository)"
# inference/sglang-v0.5.20/ since EAGLE-4 7d8f2ab, inference/ before
DATA=""
for d in inference/sglang-v0.5.20 inference; do
    if [ -f "$EAGLE4/$d/dataset/speed_bench/question_multiturn.jsonl" ]; then
        DATA="$EAGLE4/$d/dataset/speed_bench/question_multiturn.jsonl"
        SRC_CLIENT="$EAGLE4/$d/scripts/client_speedbench.py"
        break
    fi
done
[ -n "$DATA" ] || { echo "no SPEED-Bench data under $EAGLE4/inference/{sglang-v0.5.20/,}dataset/speed_bench/" >&2; exit 1; }
CLIENT="$WORK/client_speedbench.py"
# The client reads token counts from SGLang's meta_info; ATOM returns the OpenAI usage field, which this
# copy maps onto it. The acceptance length comes from ATOM's /debug/mtp_stats around each run instead.
"$PY" - "$SRC_CLIENT" "$CLIENT" << 'PYEOF'
import sys
s = open(sys.argv[1]).read()
old = "        choice = data['choices'][0]\n"
new = old + ("        choice.setdefault('meta_info', {'prompt_tokens': data['usage']['prompt_tokens'],\n"
             "                                        'completion_tokens': data['usage']['completion_tokens'],\n"
             "                                        'finish_reason': choice.get('finish_reason')})\n")
assert s.count(old) == 1, "unexpected client_speedbench.py: update the meta_info mapping in this script"
open(sys.argv[2], "w").write(s.replace(old, new))
PYEOF
echo "client $CLIENT"

wait_ready() {  # wait_ready PID LOG
    for _ in $(seq 1 360); do
        curl -sf -m 3 "http://127.0.0.1:${PORT}/v1/models" > /dev/null && return 0
        kill -0 "$1" 2> /dev/null || { echo "server exited; see $2" >&2; tail -20 "$2" >&2; exit 1; }
        sleep 5
    done
    echo "server not ready after 30 min; see $2" >&2
    exit 1
}
SERVER_PID=""
stop() { [ -n "$SERVER_PID" ] && { kill "$SERVER_PID" 2> /dev/null || true; wait "$SERVER_PID" 2> /dev/null || true; }; SERVER_PID=""; }
trap stop EXIT

AITER_CSV=$("$PY" -c "import aiter, os; print(os.path.join(os.path.dirname(aiter.__file__), 'configs', 'bf16_tuned_gemm.csv'))")
export ATOM_EAGLE4_DRAFT_MXFP4=1 ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export AITER_CONFIG_GEMM_BF16="$AITER_CSV:$ATOM_ROOT/recipes/configs/qwen38_27b_bf16_tuned_gemm.csv"

step "4. SPEED-Bench, TP$TP, conversations: $CONC"
for c in $CONC; do
    out="$WORK/results/tp$TP/c$c"; mkdir -p "$out"
    # recipes/Qwen3.8-27B-EAGLE-4.md: trees up to 16 conversations, chains past
    unset ATOM_SPEC_TREE ATOM_SPEC_VERIFY_LEN
    if [ "$c" -le 16 ]; then
        if [ "$TP" -ge 4 ]; then export ATOM_SPEC_TREE=1=dyn:8:4:16,4=dyn:7:4:12,8=dyn:5:4:8
        elif [ "$c" -le 4 ]; then export ATOM_SPEC_TREE=dyn:8:4:16
        else export ATOM_SPEC_TREE=1=dyn:8:4:16,4=dyn:6:4:12,12=dyn:5:4:8
        fi
        spec=15
    else
        export ATOM_SPEC_VERIFY_LEN=1:5,16:3
        spec=5
    fi
    echo "c$c: ATOM_SPEC_TREE=${ATOM_SPEC_TREE:-} ATOM_SPEC_VERIFY_LEN=${ATOM_SPEC_VERIFY_LEN:-} --num-speculative-tokens $spec"
    "$PY" -m atom.entrypoints.openai_server --model "$MODEL" -tp "$TP" --kv_cache_dtype bf16 \
        --max-model-len 40000 --max-num-seqs "$c" --no-enable_prefix_caching --server-port "$PORT" \
        --cudagraph-capture-sizes "[1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64]" \
        --timeout-keep-alive 3600 --method eagle3 --draft-model "$DRAFT" --num-speculative-tokens "$spec" \
        > "$out/server.log" 2>&1 &
    SERVER_PID=$!
    wait_ready "$SERVER_PID" "$out/server.log"
    # The draft round's graphs are captured on first use of each batch size: c concurrent requests of
    # staggered lengths, twice, before the measurement (SGLang captures its graphs at startup).
    "$PY" - "$PORT" "$c" << 'PYEOF'
import sys
from concurrent.futures import ThreadPoolExecutor
import requests
base, c = f"http://127.0.0.1:{sys.argv[1]}", int(sys.argv[2])
model = requests.get(base + "/v1/models", timeout=60).json()["data"][0]["id"]
def one(i):
    msg = [{"role": "user", "content": f"Write a long story about the number {i}."}]
    requests.post(base + "/v1/chat/completions", timeout=1200,
                  json={"model": model, "messages": msg, "max_tokens": 48 * (i + 1), "temperature": 0})
for _ in range(2):
    with ThreadPoolExecutor(c) as ex:
        list(ex.map(one, range(c)))
PYEOF
    curl -sf "http://127.0.0.1:${PORT}/debug/mtp_stats" > "$out/mtp_before.json" || true
    "$PY" "$CLIENT" --port "$PORT" --data "$DATA" --every 10 --concurrency "$c" --max-new-tokens 4096 \
        --tag "tp${TP}_c$c" --out "$out/result.json" > "$out/client.log" 2>&1
    curl -sf "http://127.0.0.1:${PORT}/debug/mtp_stats" > "$out/mtp_after.json" || true
    stop
    echo "  c$c done"
done

step "5. results (generated tokens / wall time; tokens accepted per round)"
"$PY" - "$WORK/results/tp$TP" "$TP" << 'PYEOF'
import json, os, sys
root, tp = sys.argv[1], int(sys.argv[2])
# MI355X: the numbers the blog shows
published = {
    1: {1: 309, 4: 976, 16: 2309, 32: 3243},
    2: {1: 403},
    4: {1: 489, 4: 1347, 16: 3381, 32: 4796},
}[tp]
def load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None
print(f"{'conversations':>13s} {'tok/s':>8s} {'accepted':>9s} {'valid':>6s}   published (MI355X)")
for d in sorted(os.listdir(root), key=lambda d: int(d[1:])):
    c = int(d[1:])
    r = load(os.path.join(root, d, "result.json"))
    s0, s1 = load(os.path.join(root, d, "mtp_before.json")), load(os.path.join(root, d, "mtp_after.json"))
    acc = "-"
    if s0 and s1 and "distribution" in s1:
        steps = sum(s1["distribution"].values()) - sum(s0["distribution"].values())
        if steps:
            acc = f"{1 + (s1['total_accepted_tokens'] - s0['total_accepted_tokens']) / steps:.2f}"
    tput = f"{r['summary']['all']['wall_tput']:8.0f}" if r else "       -"
    valid = str(r["summary"]["valid"]) if r else "-"
    print(f"{c:13d} {tput} {acc:>9s} {valid:>6s}   {published.get(c, '-')}")
print("\nMI350X runs ~8-9% below MI355X up to 8 conversations and further at large batch (power cap).")
PYEOF
