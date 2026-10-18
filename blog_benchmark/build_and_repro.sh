#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# One command, from a fresh ATOM checkout on one AMD Instinct MI355X: install, build and reproduce the
# Qwen3.8-27B single-request results (DFlash2 speculative decoding, MXFP4 weights, one request at a time):
#   1. install ATOM (skipped when it already imports),
#   2. fetch the model and the DFlash2 draft (unless local paths are given),
#   3. set up the FlyDSL compiler used to build the kernels (pinned frontend and wheel),
#   4. build the megakernel and export the DFlash2 drafter code objects (skipped when already built),
#   5. export the prompt sets: GSM8K 128, HumanEval 128, MT-Bench 80,
#   6. serve the megakernel, then ATOM's regular path, and benchmark each,
#   7. optionally do the same through SGLang with the aitermk plugin,
#   8. print the results next to the published ones.
#
# Usage, from the ATOM checkout:  bash blog_benchmark/build_and_repro.sh
#
# Optional environment:
#   WORK           everything goes here (checkpoints, compiler, code objects, prompts, results)  (default ./blog_benchmark/work)
#   MODEL, DRAFT   local checkpoint directories or Hugging Face ids  (default Qwen/Qwen3.8-27B, incoai/Qwen3.8-27B-DFlash2)
#   FLYDSL_SRC     a FlyDSL checkout or source tarball at commit 19418894  (default: cloned from github.com/ROCm/FlyDSL)
#   FLYDSL_WHEEL   the flydsl 0.4.0.dev963 wheel for this Python             (default: downloaded from PyPI)
#   ARMS           "megakernel regular" or one of them                       (default both)
#   RUN_SGLANG=1   also run SGLang with the plugin; needs AITERMK=<aitermk checkout> and SGLang with DFLASH
#   PORT           server port                                               (default 8000)

set -euo pipefail

cd "$(dirname "$0")/.."
ATOM_ROOT=$PWD
PY="${PY:-python3}"
WORK="${WORK:-$ATOM_ROOT/blog_benchmark/work}"
MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
DRAFT="${DRAFT:-incoai/Qwen3.8-27B-DFlash2}"
PORT="${PORT:-8000}"
FLYDSL_COMMIT=19418894
FLYDSL_VERSION=0.4.0.dev963
SETS="gsm8k:128 humaneval:128 mt-bench:80"
export AITER_LOG_LEVEL=WARNING
mkdir -p "$WORK"
step() { printf '\n== %s\n' "$*"; }

step "0. checks"
if command -v rocminfo > /dev/null; then rocminfo 2> /dev/null | grep gfx950 > /dev/null || { echo "needs an MI355X (gfx950)" >&2; exit 1; }; fi

step "1. ATOM"
"$PY" -c "import atom" 2> /dev/null || "$PY" -m pip install -e "$ATOM_ROOT"

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

step "3. FlyDSL (build time only; serving loads the code objects)"
FLY="$WORK/flydsl"
if [ ! -f "$FLY/READY" ]; then
    rm -rf "$FLY" && mkdir -p "$FLY/src" "$FLY/wheel" "$FLY/python"
    if [ -d "${FLYDSL_SRC:-}" ]; then
        cp -r "$FLYDSL_SRC"/. "$FLY/src/"
    elif [ -f "${FLYDSL_SRC:-}" ]; then
        tar xzf "$FLYDSL_SRC" -C "$FLY/src"
    else
        git clone -q https://github.com/ROCm/FlyDSL "$FLY/src" && git -C "$FLY/src" checkout -q "$FLYDSL_COMMIT"
    fi
    if [ -f "${FLYDSL_WHEEL:-}" ]; then
        cp "$FLYDSL_WHEEL" "$FLY/wheel/"
    else
        "$PY" -m pip download -q --no-deps --only-binary=:all: "flydsl==$FLYDSL_VERSION" -d "$FLY/wheel"
    fi
    # the frontend at the pinned commit, with the wheel's compiled MLIR bindings and libraries
    "$PY" -m zipfile -e "$FLY"/wheel/flydsl-*.whl "$FLY/wheel/x"
    cp -r "$FLY/src/python/flydsl" "$FLY/python/"
    cp -r "$FLY/wheel/x/flydsl/_mlir" "$FLY/python/flydsl/"
    cp -r "$FLY/wheel/x/flydsl.libs" "$FLY/python/"
    # the drafter's MXFP8 quantizer rounds E8M0 scales up (ceil), as the megakernel does; upstream rounds to nearest
    "$PY" - "$FLY/src/kernels/monokernel/mxfp8_linear.py" << 'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = """            round_up = ((working & fx.Int32(0x400000)) != 0) & (
                ((working & fx.Int32(0x200000)) != 0) | ((working & fx.Int32(0x1FFFFF)) != 0) | (exponent > 0)
            )"""
new = """            round_up = (working & fx.Int32(0x7FFFFF)) != 0  # ceil: never clip at the FP8 maximum"""
assert s.count(old) == 1, "unexpected FlyDSL mxfp8_linear.py: is the checkout at the pinned commit?"
open(p, "w").write(s.replace(old, new))
PYEOF
    touch "$FLY/READY"
fi
echo "flydsl $FLY"

step "4. code objects"
export ATOM_QWEN38_MK_CACHE="${ATOM_QWEN38_MK_CACHE:-$WORK/kernel-cache}"
export ATOM_QWEN38_MK_FLYDSL="$FLY/python"
"$PY" -c "from atom.models.qwen38_27b.mono.kernels.build import build; print('megakernel', build('m8'))"
DFLASH2_AOT="$WORK/drafter-objects"
if [ ! -f "$DFLASH2_AOT/manifest.json" ]; then
    PYTHONPATH="$FLY/python${PYTHONPATH:+:$PYTHONPATH}" FLY_KERNELS_ROOT="$FLY/src" \
        PROTO_ROOT="$ATOM_ROOT/atom/models/qwen38_27b/mono/drafter/kernels_src" \
        "$PY" atom/models/qwen38_27b/mono/drafter/aot_runtime.py export --out "$DFLASH2_AOT" --P 40968
fi
echo "drafter $DFLASH2_AOT"
export ATOM_DFLASH2_AOT="$DFLASH2_AOT"

step "5. prompt sets"
PROMPTS="$WORK/prompts"
mkdir -p "$PROMPTS"
for s in $SETS; do
    name=${s%%:*}
    [ -s "$PROMPTS/${name}_130.jsonl" ] || "$PY" atom/benchmarks/single_request_bench.py --dataset "$name" \
        --num-prompts 130 --export-prompts "$PROMPTS/${name}_130.jsonl"
done

bench() {  # bench OUT_DIR [extra client args...]: the three prompt sets against the server on $PORT
    local out=$1; shift
    for s in $SETS; do
        local name=${s%%:*} n=${s#*:}
        "$PY" atom/benchmarks/single_request_bench.py --base-url "http://127.0.0.1:${PORT}" \
            --prompts-file "$PROMPTS/${name}_130.jsonl" --num-prompts "$n" --max-tokens 2048 \
            --chat-template-kwargs '{"reasoning_effort": "xhigh"}' --out "$out/$name.json" "$@" > "$out/$name.log"
        echo "  $name done"
    done
}
wait_ready() {  # wait_ready PID LOG
    for _ in $(seq 1 240); do
        curl -sf -m 3 "http://127.0.0.1:${PORT}/v1/models" > /dev/null && return 0
        kill -0 "$1" 2> /dev/null || { echo "server exited; see $2" >&2; tail -20 "$2" >&2; exit 1; }
        sleep 5
    done
    echo "server not ready after 20 min; see $2" >&2
    exit 1
}
SERVER_PID=""
stop() { [ -n "$SERVER_PID" ] && { kill "$SERVER_PID" 2> /dev/null || true; wait "$SERVER_PID" 2> /dev/null || true; }; SERVER_PID=""; }
trap stop EXIT

step "6. ATOM: the megakernel, then the regular path"
for arm in ${ARMS:-megakernel regular}; do
    out="$WORK/results/atom/$arm"; mkdir -p "$out"
    set -- --model "$MODEL" --kv_cache_dtype bf16 -tp 1 --max-model-len 40000 --max-num-seqs 4 --server-port "$PORT" \
        --method dflash --draft-model "$DRAFT" --num-speculative-tokens 7
    case $arm in
        megakernel) mono=1 ;;
        # the same weights, quantized to MXFP4 at load (the conv1d weight stays bf16)
        regular) mono=0; set -- "$@" --online_quant_config '{"global_quant_config": "mxfp4", "exclude_layer": ["*conv1d*"]}' ;;
        *) echo "unknown arm $arm" >&2; exit 1 ;;
    esac
    # TEMP is Python's temp-dir variable: never let a caller's TEMP reach the server
    env -u TEMP ATOM_MONO_ENABLE=$mono "$PY" -m atom.entrypoints.openai_server "$@" > "$out/server.log" 2>&1 &
    SERVER_PID=$!
    wait_ready "$SERVER_PID" "$out/server.log"
    echo "$arm:"; bench "$out"
    stop
done

if [ "${RUN_SGLANG:-0}" = 1 ]; then
    step "7. SGLang with the aitermk plugin"
    : "${AITERMK:?set AITERMK to the aitermk checkout}"
    out="$WORK/results/sglang"; mkdir -p "$out"
    "$PY" -c "import aitermk_sglang" 2> /dev/null || \
        "$PY" -m pip install "$AITERMK/aitermk/megakernels/qwen38_27b" "$AITERMK/aitermk/engine_adapters/sglang"
    env -u TEMP SGLANG_DFLASH_MEGAKERNEL_AOT="$DFLASH2_AOT" AITERMK_QWEN38_CACHE="$ATOM_QWEN38_MK_CACHE" \
        "$PY" -m sglang.launch_server --model-path "$MODEL" --tokenizer-path "$MODEL" \
        --speculative-algorithm DFLASH --speculative-draft-model-path "$DRAFT" --speculative-num-draft-tokens 8 \
        --trust-remote-code --mem-fraction-static 0.6 --max-running-requests 1 --context-length 40960 --port "$PORT" \
        --attention-backend triton --mm-attention-backend sdpa --skip-server-warmup --mamba-ssm-dtype float32 \
        --disable-radix-cache > "$out/server.log" 2>&1 &
    SERVER_PID=$!
    wait_ready "$SERVER_PID" "$out/server.log"
    echo "sglang:"; bench "$out" --stats none
    stop
fi

step "8. results (tokens/s, mean of each request's decode speed)"
"$PY" - "$WORK/results" << 'PYEOF'
import json, os, sys
root = sys.argv[1]
published = {"gsm8k": (481, 1105), "humaneval": (406, 924), "mt-bench": (368, 846)}
def summary(*parts):
    f = os.path.join(root, *parts)
    return json.load(open(f))["summary"] if os.path.exists(f) else {}
fmt = lambda v: f"{v:7.0f}" if v else "      -"
print(f"{'set':10s} {'regular':>8s} {'megakernel':>11s} {'speedup':>8s} {'SGLang':>8s}   published (regular / megakernel)")
for name, (pr, pm) in published.items():
    r = summary("atom", "regular", name + ".json").get("decode_tok_s_per_request_mean")
    m = summary("atom", "megakernel", name + ".json").get("decode_tok_s_per_request_mean")
    g = summary("sglang", name + ".json").get("decode_tok_s_per_request_mean")
    sp = f"{m / r:7.1f}x" if r and m else "       -"
    print(f"{name:10s} {fmt(r):>8s} {fmt(m):>11s} {sp:>8s} {fmt(g):>8s}   {pr} / {pm}")
print("\nThe megakernel is not bit-reproducible run to run, so speeds move by a few percent with the acceptance length.")
PYEOF
