#!/usr/bin/env bash
# One-command reproduction of the Qwen3.8-Flash-Next monokernel results on one MI355X (gfx950):
# benchmark_serving (random ISL / OSL), AgentX (aiperf), GSM8K, HumanEval, GPQA-Diamond and MMLU-Pro
# accuracy and single-request tokens/s on GSM8K / HumanEval / MT-Bench, in a pinned ROCm container.
# build_and_repro.sh runs all of it for both decode paths and every draft count, then tabulates.
# Run it on the host from a checkout of this repository; it starts the container itself, builds
# the environment once into $WORK, starts the server, runs the client, prints a summary and stops
# the server.
#
# What is compared: two decode paths in the same ATOM server, on the same MXFP4 checkpoint
# (repro.sh convert), with the same server line, prompts and client.
#   DECODE=mono    the monokernel: each decoder layer's decode step is one FlyDSL kernel
#                  (atom/models/qwen4_exp_mono). Prefill and the final mixer stay ATOM's;
#                  the embedding is one launch and lm_head runs from MXFP4 weights.
#   DECODE=native  the baseline, ATOM's regular path: ATOM's and aiter's own kernels, unchanged
#                  (aiter's FlyDSL MoE; ATOM's Triton W4A8 GEMM, GatedDeltaNet and QSA kernels).
#                  The ATOM changes it runs are the few lines that load this checkpoint, the
#                  inline-send option below and the step-overhead changes README.md lists.
# Both run with ATOM_RUNNER_INLINE_SEND=1, and with ATOM_ENABLE_REPLAYSSM=0 when MTP is on (the
# mono's verify needs it off; ATOM's own default turns it on with MTP).
#
#   MODEL=/path/to/Qwen3.8-Flash-Next-MXFP4 GPU=0 bash scripts/qwen38_flash_mono/repro.sh bench
#   MODEL=/path/to/Qwen3.8-Flash-Next-MXFP4 GPU=0 bash scripts/qwen38_flash_mono/repro.sh agentx
#
# Commands:
#   bench    server, then benchmark_serving for every CONC x ISL (x REPS), then a table; each
#            CONC x ISL starts with an untimed warmup, and every run has its own prompts
#   agentx   server with AgentX's settings, then aiperf's inferencex-agentx-mvp scenario, then X / Y
#   gsm8k    server, then lm_eval GSM8K (1,319 questions, 5-shot, chat template, up to 4,096 tokens)
#   accuracy server, then atom/benchmarks/qwen38_flash_mono/accuracy_eval.py for each TASK:
#            HumanEval pass@1 (164), GPQA-Diamond (198) and MMLU-Pro (20 per category), greedy;
#            GPQA and MMLU-Pro think at REASONING (default low)
#   speed    server, then atom/benchmarks/qwen38_flash_mono/single_request_bench.py once per set
#            and CONC: batch-1 tokens/s on the DFlash benchmark's GSM8K, HumanEval and MT-Bench
#            prompts (seed-42 order, first 128 / 128 / 80), reasoning effort xhigh, up to 2,048
#            output tokens, greedy. CONC above 1 keeps that many requests in flight (tokens/s per
#            request)
#   convert  download Qwen/Qwen3.8-Flash-Next (BF16, 335 GiB) into $WORK/models unless BF16 is set,
#            then convert it to $WORK/models/Qwen3.8-Flash-Next-MXFP4
#            (scripts/qwen38_flash_mxfp4/convert_mxfp4.py; about 500 GiB of disk with the download)
#   server   the server alone, in the foreground, published on host port $PORT
#   setup    only build the environment (every other command does this first; a few minutes, once)
#   shell    a shell in the container with the environment active
#
# Settings (environment variables):
#   MODEL        the MXFP4 checkpoint directory (repro.sh convert); required except for setup,
#                shell and convert
#   BF16         convert: an existing Qwen/Qwen3.8-Flash-Next download (not downloaded again)
#   GPU          HIP device ordinal (HIP_VISIBLE_DEVICES) to run on; default 0
#   WORK         venv, sources, caches and results; default $HOME/q38-repro (~12 GB)
#   DECODE       mono | native; default mono
#   K            MTP draft tokens, 0 = no MTP; default 2
#   AL           forced acceptance length (--spec-decode-acceptance-length), "real" = not forced;
#                default real for bench / gsm8k, InferenceX's golden length for agentx
#                (k = 1 / 2 / 3: 1.65 / 2.06 / 2.32)
#   CONC         client concurrencies, space separated; default 1 (gsm8k / accuracy / agentx: the first)
#   ISL OSL      bench input lengths, space separated (default "1024 8192"), output length (1024)
#   PROMPTS      bench prompts per run; default max(10, 4 x concurrency). speed: prompts per set (128)
#   REPS         bench runs per (concurrency, ISL); default 1
#   DURATION     agentx benchmark duration in seconds; default 900
#   SETS         speed: prompt sets, space separated; default "gsm8k humaneval mt-bench"
#   MAX_TOKENS   speed: output tokens per request; default 2048
#   REASONING    speed / accuracy: the chat template's reasoning_effort (xhigh, medium, low; accuracy
#                also none); default xhigh for speed, low for accuracy
#   TASK         accuracy: tasks, space separated; default "humaneval gpqa mmlupro"
#   LIMIT        accuracy / gsm8k: only the first LIMIT questions of each task (a quick check);
#                default all
#   PORT         server port; default 8000
#   MAX_SEQS     --max-num-seqs; default max(4, CONC) (agentx 8)
#   MAX_MODEL_LEN --max-model-len; default 10240, or more for longer ISL (agentx 262144 - K)
#
# Results go to $WORK/runs/<UTC time>-<command>-<decode>-k<K>/ (summary.txt, server.log, raw client
# output, meta.txt with the commit, image, GPU and package versions). Time on an otherwise idle GPU.
set -euo pipefail

IMAGE=${IMAGE:-rocm/pytorch@sha256:bbdaba66029f905321be3bbc95206a2d9a56a2bc5300877d146ad24d40db9a12}
AITER_COMMIT=c53e9b3f4285c36ea785e614ce5cf8b04c78b3bd
AIPERF_COMMIT=754356e9a39acc6cc6afb242d123bb57c3fb6f75
HF_MODEL=Qwen/Qwen3.8-Flash-Next HF_REVISION=de4b8e4d43b917e7706784d8bb445c9af86a3540
SELF=scripts/qwen38_flash_mono/repro.sh

die() { echo "repro: $*" >&2; exit 1; }
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }

cmd=${1:-help}
case $cmd in
  bench | agentx | gsm8k | accuracy | speed | server | convert | setup | shell) ;;
  help | -h | --help) usage; exit 0 ;;
  *) usage; exit 2 ;;
esac

# ---------------------------------------------------------------------------------------------
# Host: start the container and run this script inside it.
# ---------------------------------------------------------------------------------------------
if [ -z "${REPRO_IN_CONTAINER:-}" ]; then
  command -v docker > /dev/null || die "docker not found"
  REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
  WORK=${WORK:-$HOME/q38-repro}
  mkdir -p "$WORK"
  args=(
    --rm --init --name "q38-repro-$(id -un)-$$"
    --ulimit nofile=65535:65535 --ulimit memlock=-1:-1 --ulimit stack=67108864
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined
    --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host
    -v "$REPO":/atom -v "$(cd "$WORK" && pwd)":/work -w /atom
    -e REPRO_IN_CONTAINER=1 -e HIP_VISIBLE_DEVICES="${GPU:-0}"
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)"
  )
  # A git worktree's metadata lives in the main checkout's .git: mount it at the same path.
  common=$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir 2> /dev/null || true)
  [ -z "$common" ] || [ "$common" = "$REPO/.git" ] || args+=(-v "$common":"$common")
  case $cmd in
    setup | shell) ;;
    convert)
      if [ -n "${BF16:-}" ]; then
        [ -f "$BF16/config.json" ] || die "BF16=$BF16 has no config.json"
        args+=(-v "$(realpath "$BF16")":/bf16:ro)
      fi
      ;;
    *) [ -n "${MODEL:-}" ] || die "set MODEL=<the Qwen3.8-Flash-Next-MXFP4 checkpoint directory>" ;;
  esac
  if [ -n "${MODEL:-}" ] && [ "$cmd" != convert ]; then
    [ -f "$MODEL/config.json" ] || die "MODEL=$MODEL has no config.json"
    args+=(-v "$(realpath "$MODEL")":/model:ro)
  fi
  for v in DECODE K AL CONC ISL OSL PROMPTS REPS DURATION PORT MAX_SEQS \
    MAX_MODEL_LEN SETS MAX_TOKENS REASONING TASK LIMIT; do
    [ -z "${!v:-}" ] || args+=(-e "$v=${!v}")
  done
  [ "$cmd" != server ] || args+=(-p "${PORT:-8000}:${PORT:-8000}")
  [ ! -t 0 ] || args+=(-it)
  echo "repro: $cmd on HIP device ${GPU:-0}, work dir $WORK, image $IMAGE"
  exec docker run "${args[@]}" "$IMAGE" bash "/atom/$SELF" "$@"
fi

# ---------------------------------------------------------------------------------------------
# Container.
# ---------------------------------------------------------------------------------------------
DECODE=${DECODE:-mono} K=${K:-2} CONC=${CONC:-1} ISL=${ISL:-1024 8192} OSL=${OSL:-1024}
REPS=${REPS:-1} DURATION=${DURATION:-900}
PORT=${PORT:-8000} SETS=${SETS:-gsm8k humaneval mt-bench} MAX_TOKENS=${MAX_TOKENS:-2048}
REASONING=${REASONING-$([ "$cmd" = accuracy ] && echo low || echo xhigh)}
TASK=${TASK:-humaneval gpqa mmlupro}
case $DECODE in mono | native) ;; *) die "DECODE must be mono or native" ;; esac
case $K in 0 | 1 | 2 | 3) ;; *) die "K must be 0, 1, 2 or 3" ;; esac

export HOME=/work/home TMPDIR=/work/tmp
export PIP_CACHE_DIR=/work/cache/pip XDG_CACHE_HOME=/work/cache/xdg HF_HOME=/work/cache/hf
export TRITON_CACHE_DIR=/work/cache/triton TORCHINDUCTOR_CACHE_DIR=/work/cache/inductor
export FLYDSL_RUNTIME_CACHE_DIR=/work/cache/flydsl
export MIOPEN_USER_DB_PATH=/work/cache/miopen MIOPEN_CUSTOM_CACHE_DIR=/work/cache/miopen
export PYTHONDONTWRITEBYTECODE=1 AITER_LOG_LEVEL=WARNING HSA_COREDUMP_PATTERN=/dev/null
mkdir -p "$HOME" "$TMPDIR" /work/cache /work/logs /work/runs
git config --global --add safe.directory '*'

SERVER_PID=
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2> /dev/null; then
    kill "$SERVER_PID" 2> /dev/null || true
    for _ in $(seq 1 60); do kill -0 "$SERVER_PID" 2> /dev/null || break; sleep 2; done
    kill -9 "$SERVER_PID" 2> /dev/null || true
  fi
  # The container runs as root: hand everything it wrote back to the caller.
  chown -R "${HOST_UID:-0}:${HOST_GID:-0}" /work /atom/atom.egg-info /atom/atom/_version.py 2> /dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

step() {  # step <name> <command...>: run quietly into /work/logs/setup-<name>.log
  local name=$1 log=/work/logs/setup-$1.log rc
  shift
  echo "repro: setup $name (log $log)"
  # Not in an if: set -e would be off inside the step's functions.
  set +e
  (set -e; "$@") > "$log" 2>&1
  rc=$?
  set -e
  [ $rc = 0 ] || { tail -n 30 "$log"; die "setup $name failed"; }
}

SP=/work/venv/lib/python3.12/site-packages
OPT_SP=/opt/venv/lib/python3.12/site-packages

rocm_env() {
  export ROCM_PATH=$SP/_rocm_sdk_devel
  export HIP_PATH=$ROCM_PATH
  export CMAKE_PREFIX_PATH=$ROCM_PATH${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}
  export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib/host-math/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  export PATH=$ROCM_PATH/bin:$PATH
}

setup_rocm() {
  pip install "rocm[devel]==10.0.0" --index-url https://stable.repo.amd.com/rocm/whl-next
  python -m rocm_sdk init
  # rocm_sdk init expands relative symlinks into the core package, which lives in /opt/venv.
  ln -sfn $OPT_SP/_rocm_sdk_core $SP/_rocm_sdk_core
  ln -sfn $OPT_SP/_rocm_sdk_libraries $SP/_rocm_sdk_libraries
}

setup_aiter() {
  [ -d /work/src/aiter/.git ] \
    || git clone --filter=blob:none https://github.com/ROCm/aiter.git /work/src/aiter
  cd /work/src/aiter
  git fetch origin $AITER_COMMIT || true
  git checkout -f $AITER_COMMIT
  git submodule update --init --recursive --depth 1
  pip install -r requirements.txt
  MAX_JOBS=$(($(nproc) < 64 ? $(nproc) : 64)) PREBUILD_KERNELS=0 GPU_ARCHS=gfx950 \
    python setup.py develop
}

setup() {
  # An overlay venv: its own packages first, then the image's (torch, triton, ROCm runtime).
  if [ ! -f /work/venv/.repro-venv ]; then
    rm -rf /work/venv
    /opt/venv/bin/python3 -m venv /work/venv
    echo $OPT_SP > $SP/zz_opt_venv.pth
    touch /work/venv/.repro-venv
  fi
  # shellcheck disable=SC1091
  source /work/venv/bin/activate
  export PIP_CONSTRAINT=/atom/scripts/qwen38_flash_mono/repro-constraints.txt
  [ -f /work/venv/.repro-rocm ] || { step rocm-sdk setup_rocm && touch /work/venv/.repro-rocm; }
  rocm_env
  [ -f /work/venv/.repro-aiter ] || { step aiter setup_aiter && touch /work/venv/.repro-aiter; }
  cd /atom
  if [ ! -f /work/venv/.repro-atom ]; then
    step atom pip install -e /atom "lm-eval[api]==0.4.13" msgpack
    touch /work/venv/.repro-atom
  fi
  unset PIP_CONSTRAINT
  local arch
  arch=$(python -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName.split(':')[0])")
  [ "$arch" = gfx950 ] || die "HIP device ${HIP_VISIBLE_DEVICES} is $arch; the monokernel needs gfx950 (MI355X)"
  echo "repro: environment ready ($(python -c 'import torch; print(torch.cuda.get_device_name(0))'), $arch)"
}

setup_aiperf() {
  [ -d /work/src/aiperf/.git ] || git clone https://github.com/SemiAnalysisAI/aiperf.git /work/src/aiperf
  git -C /work/src/aiperf fetch origin $AIPERF_COMMIT || true
  git -C /work/src/aiperf checkout -f $AIPERF_COMMIT
  [ -x /work/aiperf-venv/bin/aiperf ] || {
    /opt/venv/bin/python3 -m venv /work/aiperf-venv
    /work/aiperf-venv/bin/python -m pip install --upgrade pip
    /work/aiperf-venv/bin/python -m pip install -e /work/src/aiperf
  }
  /work/aiperf-venv/bin/aiperf --version
}

# --- server ----------------------------------------------------------------------------------
max() { local m=$1 x; for x in "$@"; do [ "$x" -gt "$m" ] && m=$x; done; echo "$m"; }

server_cmd() {  # server_cmd <command>: fills SERVER_ENV and SERVER_ARGS
  local mode=$1 al=${AL:-} seqs mml
  declare -A golden=([1]=1.65 [2]=2.06 [3]=2.32)
  SERVER_ENV=(ATOM_RUNNER_INLINE_SEND=1)
  if [ "$DECODE" = mono ]; then
    SERVER_ENV+=(ATOM_MONO_ENABLE=1)
  else
    SERVER_ENV+=(ATOM_MONO_ENABLE=0)
  fi
  if [ "$mode" = agentx ]; then
    # AgentX: decode before queued prefills, 8 sequences, the full 262144-token window
    # (ATOM reserves K tokens of it for the drafts), InferenceX's golden acceptance length.
    SERVER_ENV+=(ATOM_PREFILL_DECODE_INTERVAL=1000000)
    seqs=${MAX_SEQS:-8} mml=${MAX_MODEL_LEN:-$((262144 - K))}
    [ -n "$al" ] || [ "$K" = 0 ] || al=${golden[$K]}
  else
    seqs=${MAX_SEQS:-$(max 4 $CONC)}
    local need=$(($(max $ISL) + OSL + K + 1))
    mml=${MAX_MODEL_LEN:-$(max 10240 $(((need + 1023) / 1024 * 1024)))}
  fi
  SERVER_ARGS=(--model /model --trust-remote-code -tp 1 --server-port "$PORT"
    --max-num-seqs "$seqs" --max-model-len "$mml" --gpu-memory-utilization 0.9
    --hf-overrides '{"_multimodal_config": null}')
  if [ "$K" != 0 ]; then
    # The mono's verify uses ATOM's speculative state, which ReplaySSM replaces.
    SERVER_ENV+=(ATOM_ENABLE_REPLAYSSM=0)
    SERVER_ARGS+=(--method mtp --num-speculative-tokens "$K")
    [ -z "$al" ] || [ "$al" = real ] || SERVER_ARGS+=(--spec-decode-acceptance-length "$al")
  fi
}

start_server() {  # start_server <command> <log>
  server_cmd "$1"
  echo "repro: server: ${SERVER_ENV[*]} python -m atom.entrypoints.openai_server ${SERVER_ARGS[*]}"
  echo "${SERVER_ENV[*]} python -m atom.entrypoints.openai_server ${SERVER_ARGS[*]}" >> "$RUN/meta.txt"
  rm -rf "$HOME/.cache/atom"
  env "${SERVER_ENV[@]}" python -m atom.entrypoints.openai_server "${SERVER_ARGS[@]}" > "$2" 2>&1 &
  SERVER_PID=$!
  echo "repro: waiting for the server (the first start JIT-compiles aiter and the 48 layer kernels; log $2)"
  local t=0
  until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
    kill -0 "$SERVER_PID" 2> /dev/null || { tail -n 30 "$2"; die "the server exited"; }
    [ $t -lt 5400 ] || { tail -n 30 "$2"; die "the server was not ready after 90 min"; }
    sleep 10
    t=$((t + 10))
  done
  grep -ao 'Qwen3.8 mono [a-z]* o[nf]*[^[]*' "$2" | sort -u | sed 's/^/repro: /' || true
  local reply
  reply=$(curl -s "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model": "/model", "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}], "max_tokens": 512, "temperature": 0}' \
    | python -c 'import json, sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"][-200:])' 2>&1 || true)
  echo "repro: server ready after ${t}s; sanity reply: $(echo "$reply" | tr '\n' ' ' | tail -c 120)"
}

stop_server() {
  kill "$SERVER_PID" 2> /dev/null || true
  for _ in $(seq 1 60); do kill -0 "$SERVER_PID" 2> /dev/null || break; sleep 2; done
  kill -9 "$SERVER_PID" 2> /dev/null || true
  SERVER_PID=
}

new_run() {
  RUN=/work/runs/$(date -u +%Y%m%dT%H%M%SZ)-$1-$DECODE-k$K
  mkdir -p "$RUN"
  {
    echo "command: $1"
    if [ "$DECODE" = mono ]; then
      echo "path: mono, the monokernel decode (prefill and the final mixer stay ATOM's)"
    else
      echo "path: native, ATOM's regular path, the baseline (ATOM's and aiter's own kernels)"
    fi
    echo "date: $(date -u)"
    echo "atom: $(git -C /atom rev-parse HEAD) ($(git -C /atom status --porcelain | wc -l) changed files)"
    echo "image: $IMAGE"
    echo "aiter: $AITER_COMMIT"
    echo "gpu: HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES $(python -c 'import torch; p = torch.cuda.get_device_properties(0); print(p.name, p.gcnArchName)')"
    echo "settings: DECODE=$DECODE K=$K AL=${AL:-} CONC=$CONC ISL=$ISL OSL=$OSL PROMPTS=${PROMPTS:-} REPS=$REPS DURATION=$DURATION SETS=$SETS MAX_TOKENS=$MAX_TOKENS REASONING=$REASONING TASK=$TASK LIMIT=${LIMIT:-}"
  } > "$RUN/meta.txt"
  pip list --format=freeze > "$RUN/pip-freeze.txt" 2> /dev/null
  echo "repro: results in $RUN"
}

mtp_stats() { curl -sf "http://localhost:$PORT/debug/mtp_stats" > "$1" || echo '{}' > "$1"; }

mtp_summary() {  # mtp_summary <run dir>: draft acceptance between mtp_before.json and mtp_after.json
  python - "$1" << 'PY'
import json, sys

run = sys.argv[1]
a = json.load(open(f"{run}/mtp_after.json")).get("distribution", {})
b = json.load(open(f"{run}/mtp_before.json")).get("distribution", {})
h = {int(j): v - b.get(j, 0) for j, v in a.items()}
steps = sum(h.values())
if steps:
    # p_i: of the steps that accepted i - 1 drafts or more, the share that accepted i or more.
    at_least = [sum(v for j, v in h.items() if j >= i) for i in range(max(h) + 1)]
    p = " ".join(f"{at_least[i] / at_least[i - 1]:.3f}" for i in range(1, len(at_least)))
    print(f"MTP: {1 + sum(j * v for j, v in h.items()) / steps:.2f} tokens per verify, p1..pk {p}")
PY
}

# --- clients ---------------------------------------------------------------------------------
bench_serving() {  # bench_serving <log> <concurrency> <ISL> <OSL> <prompts> <seed> [args...]
  local log=$1 c=$2 isl=$3 osl=$4 n=$5 seed=$6
  shift 6
  python -m atom.benchmarks.benchmark_serving --model=/model --backend=vllm \
    --base-url="http://localhost:$PORT" --dataset-name=random \
    --random-input-len="$isl" --random-output-len="$osl" --random-range-ratio=1.0 \
    --num-prompts="$n" --max-concurrency="$c" --request-rate=inf --ignore-eos --seed="$seed" \
    --percentile-metrics=ttft,tpot,itl,e2el "$@" > "$log" 2>&1 \
    || { tail -n 20 "$log"; die "benchmark_serving failed"; }
}

run_bench() {
  local c isl r n tag
  for c in $CONC; do
    for isl in $ISL; do
      # Untimed: the first steps at a new batch shape JIT-compile kernels on a fresh cache.
      echo "repro: warmup, concurrency $c, ISL $isl"
      bench_serving "$RUN/warmup_c${c}_isl$isl.log" "$c" "$isl" 64 $((2 * c)) $((1000000 + isl * 64 + c))
      for r in $(seq 1 "$REPS"); do
        n=${PROMPTS:-$(max 10 $((4 * c)))}
        tag=c${c}_isl${isl}_osl${OSL}_r$r
        echo "repro: benchmark_serving concurrency $c, ISL $isl, OSL $OSL, $n prompts (run $r)"
        mtp_stats "$RUN/mtp_${tag}_before.json"
        # A seed per run: no run repeats another's prompts, so none hits the prefix cache;
        # every server gets the same prompts for the same (concurrency, ISL, run).
        bench_serving "$RUN/bench_$tag.log" "$c" "$isl" "$OSL" "$n" $(((isl * 64 + c) * 64 + r)) \
          --save-result --result-dir="$RUN" --result-filename="bench_$tag.json"
        mtp_stats "$RUN/mtp_${tag}_after.json"
      done
    done
  done
  python - "$RUN" "$K" << 'PY' | tee "$RUN/summary.txt"
import glob, json, os, re, sys

run, k = sys.argv[1], int(sys.argv[2])


def acceptance(tag):
    # Steps by accepted drafts, this run only (the server's counters are cumulative).
    try:
        a = json.load(open(f"{run}/mtp_{tag}_after.json")).get("distribution", {})
        b = json.load(open(f"{run}/mtp_{tag}_before.json")).get("distribution", {})
    except (OSError, ValueError):
        return ""
    d = {int(j): v - b.get(j, 0) for j, v in a.items()}
    steps = sum(d.values())
    if not steps:
        return ""
    al = 1 + sum(j * v for j, v in d.items()) / steps
    # p_i: of the steps that accepted i - 1 drafts or more, the share that accepted i or more.
    ge = [sum(v for j, v in d.items() if j >= i) for i in range(k + 1)]
    ps = " ".join(f"{ge[i] / ge[i - 1]:.2f}" if ge[i - 1] else "-" for i in range(1, k + 1))
    return f"{al:5.2f}  {ps}"


print(f"{os.path.basename(run)}")
print("conc   ISL   OSL  prompts  TPOT mean/med ms  ITL med ms  TTFT mean ms  out tok/s  total tok/s  AL  p1..pk")
runs = []
for f in glob.glob(f"{run}/bench_*.json"):
    tag = os.path.basename(f)[len("bench_") : -len(".json")]
    c, isl, osl, r = map(int, re.fullmatch(r"c(\d+)_isl(\d+)_osl(\d+)_r(\d+)", tag).groups())
    runs.append(((c, isl, r), tag, osl, json.load(open(f))))
for (c, isl, r), tag, osl, d in sorted(runs):
    print(
        f"{c:>4} {isl:>5} {osl:>5} {d['completed']:>8}  {d['mean_tpot_ms']:7.2f} / {d['median_tpot_ms']:5.2f}"
        f"  {d['median_itl_ms']:10.2f}  {d['mean_ttft_ms']:12.0f}  {d['output_throughput']:9.1f}"
        f"  {d['total_token_throughput']:11.1f}  {acceptance(tag)}"
    )
PY
}

run_agentx() {
  step aiperf setup_aiperf
  local c=${CONC%% *}
  echo "repro: AgentX at concurrency $c for ${DURATION}s (first run also downloads and tokenizes the 393 traces)"
  mtp_stats "$RUN/mtp_before.json"
  (
    export AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300 AIPERF_HTTP_TCP_USER_TIMEOUT=900000 \
      AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0 AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800 \
      AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
    /work/aiperf-venv/bin/aiperf profile --scenario inferencex-agentx-mvp \
      --url "http://localhost:$PORT" --endpoint /v1/chat/completions \
      --endpoint-type chat --streaming \
      --model /model --tokenizer /model --tokenizer-trust-remote-code \
      --concurrency "$c" --benchmark-duration "$DURATION" \
      --stats-interval 30 --random-seed 42 --failed-request-threshold 0.10 \
      --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
      --warmup-requests-per-lane 5 --trace-idle-gap-cap-seconds 300 \
      --agentic-warmup-grace-period 1800 \
      --use-server-token-count --no-gpu-telemetry \
      --num-dataset-entries 393 --slice-duration 1.0 \
      --server-metrics "http://localhost:$PORT/metrics" \
      --public-dataset semianalysis_cc_traces_weka_062126_256k \
      --ui simple --output-artifact-dir "$RUN/aiperf"
  ) > "$RUN/aiperf.log" 2>&1 || { tail -n 30 "$RUN/aiperf.log"; die "aiperf failed"; }
  mtp_stats "$RUN/mtp_after.json"
  python - "$RUN" << 'PY' | tee "$RUN/summary.txt"
import json, os, sys

run = sys.argv[1]
d = json.load(open(f"{run}/aiperf/profile_export_aiperf.json"))


def stat(key, s):
    v = d.get(key)
    return v.get(s) if isinstance(v, dict) else v


meta = d.get("metadata") or {}
dur = stat("benchmark_duration", "avg")
itl90 = stat("inter_token_latency", "p90")
print(os.path.basename(run))
print(f"  submission valid {meta.get('submission_valid')} {meta.get('submission_invalid_reasons', '')}")
print(f"  requests {stat('request_count', 'avg')}, errors {d.get('error_summary')}")
print(f"  ISL mean {stat('input_sequence_length', 'avg'):.0f}, OSL mean {stat('output_sequence_length', 'avg'):.0f}")
print("  ITL ms mean / p50 / p90 / p99 " + " / ".join(f"{stat('inter_token_latency', s):.2f}" for s in ("avg", "p50", "p90", "p99")))
print("  TTFT ms mean / p50 / p90 " + " / ".join(f"{stat('time_to_first_token', s):.0f}" for s in ("avg", "p50", "p90")))
print(f"  X = 1000 / p90 ITL = {1000 / itl90:.1f} tok/s/user")
print(f"  Y = total tokens / duration = {stat('total_usage_total_tokens', 'avg') / dur:.0f} tok/s/GPU ({dur:.0f} s)")
PY
  { mtp_summary "$RUN" || true; } | sed 's/^/  /' | tee -a "$RUN/summary.txt"
}

run_gsm8k() {
  local c=${CONC%% *} n=1319 limit=()
  [ "${LIMIT:-0}" = 0 ] || { n=$LIMIT; limit=(--limit "$LIMIT"); }
  echo "repro: GSM8K ($n questions, 5-shot, chat template) at concurrency $c"
  mtp_stats "$RUN/mtp_before.json"
  lm_eval --model local-chat-completions --tasks gsm8k --num_fewshot 5 "${limit[@]}" \
    --apply_chat_template --fewshot_as_multiturn \
    --model_args "model=/model,base_url=http://localhost:$PORT/v1/chat/completions,num_concurrent=$c,max_retries=3,timeout=900,tokenized_requests=False,trust_remote_code=True" \
    --gen_kwargs 'max_gen_toks=4096,until=<|im_end|>' \
    --output_path "$RUN/gsm8k" --log_samples > "$RUN/gsm8k.log" 2>&1 \
    || { tail -n 30 "$RUN/gsm8k.log"; die "lm_eval failed"; }
  mtp_stats "$RUN/mtp_after.json"
  { grep -E "^\|(Tasks|gsm8k| )" "$RUN/gsm8k.log"; mtp_summary "$RUN" || true; } | tee "$RUN/summary.txt"
}

run_accuracy() {
  local c=${CONC%% *} t
  echo "repro: accuracy on $TASK at concurrency $c, reasoning ${REASONING:-low} on GPQA / MMLU-Pro"
  for t in $TASK; do
    echo "repro: accuracy_eval $t"
    python -m atom.benchmarks.qwen38_flash_mono.accuracy_eval --base-url "http://localhost:$PORT" \
      --model /model --task "$t" --concurrency "$c" --reasoning-effort "${REASONING:-low}" \
      --limit "${LIMIT:-0}" --out "$RUN/accuracy_$t.json" > "$RUN/accuracy_$t.log" 2>&1 \
      || { tail -n 30 "$RUN/accuracy_$t.log"; die "accuracy_eval failed on $t"; }
  done
  # shellcheck disable=SC2086
  python - "$RUN" "$DECODE k=$K" $TASK << 'PY' | tee "$RUN/summary.txt"
import json, sys

run, label, tasks = sys.argv[1], sys.argv[2], sys.argv[3:]
print(f"accuracy_eval {label}")
print("task       questions  right  accuracy  stderr  at cap  out tokens  reasoning  seconds")
for t in tasks:
    s = json.load(open(f"{run}/accuracy_{t}.json"))["summary"]
    print(
        f"{t:<10} {s['n']:>9}  {s['correct']:>5}  {s['accuracy']:8.4f}  {s['stderr']:6.4f}"
        f"  {s['truncated']:>6}  {s['output_tokens'] / s['n']:10.0f}  {s['reasoning_effort'] or '-':>9}  {s['seconds']:7.0f}"
    )
PY
}

run_speed() {
  local n=${PROMPTS:-128} s c kwargs=()
  [ -z "$REASONING" ] || kwargs=(--chat-template-kwargs "{\"reasoning_effort\": \"$REASONING\"}")
  echo "repro: single-request tokens/s on $SETS ($n prompts per set, up to $MAX_TOKENS tokens, reasoning ${REASONING:-default}), concurrency $CONC"
  mkdir -p "$RUN/speed"
  for c in $CONC; do
    for s in $SETS; do
      echo "repro: single_request_bench $s, concurrency $c"
      python -m atom.benchmarks.qwen38_flash_mono.single_request_bench --base-url "http://localhost:$PORT" \
        --model /model --dataset "$s" --num-prompts "$n" --max-tokens "$MAX_TOKENS" "${kwargs[@]}" \
        --concurrency "$c" --out "$RUN/speed/${s}_c$c.json" > "$RUN/speed/${s}_c$c.log" 2>&1 \
        || { tail -n 30 "$RUN/speed/${s}_c$c.log"; die "single_request_bench failed on $s"; }
    done
  done
  # shellcheck disable=SC2086
  python - "$RUN/speed" "$DECODE k=$K" "$CONC" $SETS << 'PY' | tee "$RUN/summary.txt"
import json, sys

out, label, concs, sets = sys.argv[1], sys.argv[2], sys.argv[3].split(), sys.argv[4:]
print(f"single_request_bench {label}")
print("set        conc  requests  tok/s mean  median  pooled  total tok/s  TTFT ms  out tokens  at cap  tok/step  ms/step  GSM8K right")
for c in concs:
    for name in sets:
        s = json.load(open(f"{out}/{name}_c{c}.json"))["summary"]
        tps = f"{s['acceptance']:.2f}" if "acceptance" in s else "-"
        ms = f"{s['ms_per_round']:.2f}" if "ms_per_round" in s else "-"
        right = f"{s['correct']}/{s['checked']}" if "checked" in s else ""
        print(
            f"{name:<10} {c:>4}  {s['prompts']:>8}  {s['decode_tok_s_per_request_mean']:10.1f}"
            f"  {s['decode_tok_s_per_request_median']:6.1f}  {s['decode_tok_s_token_weighted']:6.1f}"
            f"  {s['end_to_end_tok_s']:11.1f}  {s['mean_ttft_ms']:7.0f}  {s['mean_output_tokens']:10.0f}"
            f"  {s['at_cap']:>6}  {tps:>8}  {ms:>7}  {right}"
        )
PY
}

run_convert() {
  local hf=/bf16 out=/work/models/Qwen3.8-Flash-Next-MXFP4
  if [ ! -f $hf/config.json ]; then
    hf=/work/models/Qwen3.8-Flash-Next
    echo "repro: downloading $HF_MODEL at $HF_REVISION (335 GiB) into \$WORK/models"
    python -c "from huggingface_hub import snapshot_download as d; d('$HF_MODEL', revision='$HF_REVISION', local_dir='$hf')"
  fi
  echo "repro: converting $hf to MXFP4 in \$WORK/models/$(basename $out)"
  python /atom/scripts/qwen38_flash_mxfp4/convert_mxfp4.py --hf $hf --out $out
}

setup
case $cmd in
  setup) ;;
  shell) bash -i || true ;;
  convert) run_convert ;;
  server)
    server_cmd server
    echo "repro: ${SERVER_ENV[*]} python -m atom.entrypoints.openai_server ${SERVER_ARGS[*]}"
    env "${SERVER_ENV[@]}" python -m atom.entrypoints.openai_server "${SERVER_ARGS[@]}"
    ;;
  bench | agentx | gsm8k | accuracy | speed)
    new_run "$cmd"
    start_server "$cmd" "$RUN/server.log"
    "run_$cmd"
    stop_server
    ;;
esac
