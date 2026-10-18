#!/usr/bin/env bash
# Qwen3.8-Flash-Next on one MI355X (gfx950): ATOM's monokernel decode against ATOM's regular decode,
# from the public BF16 checkpoint to one results table. Every run goes through repro.sh (the pinned
# ROCm container, one server per run); this script picks the runs and tabulates them.
#
# What is compared: two decode paths in the same ATOM server, on the same MXFP4 checkpoint (built
# by the model step), with the same server line, prompts and client.
#   mono     the monokernel: each decoder layer's decode step is one FlyDSL kernel
#            (atom/models/qwen4_exp_mono). Prefill, the embedding, the final mixer and lm_head
#            stay ATOM's.
#   regular  the baseline, ATOM's regular path (DECODE=native): ATOM's and aiter's own kernels,
#            unchanged (aiter's FlyDSL MoE; ATOM's Triton W4A8 GEMM, GatedDeltaNet and QSA
#            kernels). The only ATOM changes it runs are the few lines that load this checkpoint
#            and the inline-send option below.
# Both run with ATOM_RUNNER_INLINE_SEND=1, and with ATOM_ENABLE_REPLAYSSM=0 when MTP is on (the
# mono's verify needs it off; ATOM's own default turns it on with MTP). The baseline is ATOM at
# this branch's base, not later ATOM releases or other checkpoints.
#
#   GPU=0 bash scripts/qwen38_flash_mono/build_and_repro.sh
#   GPU=0 MODEL=/path/to/Qwen3.8-Flash-Next-MXFP4 STEPS="speed table" KS="0 3" \
#     bash scripts/qwen38_flash_mono/build_and_repro.sh
#   GPU=0 SMOKE=1 bash scripts/qwen38_flash_mono/build_and_repro.sh    # every run, a few requests
#
# Steps (STEPS, run in this order; default all):
#   model     Qwen/Qwen3.8-Flash-Next converted to MXFP4 in $WORK/models (repro.sh convert;
#             downloaded unless BF16 is set); skipped when MODEL is set
#   speed     tokens/s per request on the DFlash benchmark's GSM8K / HumanEval / MT-Bench prompts
#             (128 / 128 / 80), reasoning effort xhigh, up to 2,048 output tokens, at each of CONCS
#             requests in flight (repro.sh speed)
#   accuracy  GSM8K (lm_eval, 5-shot, chat template; repro.sh gsm8k), HumanEval, GPQA-Diamond and
#             MMLU-Pro (repro.sh accuracy), greedy, 4 requests in flight
#   context   one request at a time, 1,024 output tokens after random 1K and 8K prompts (10 prompts,
#             3 runs) and 128K and 250K prompts (3 prompts) (repro.sh bench)
#   serving   CONCS requests in flight, 1,024 output tokens after random 1K and 8K prompts, 2 runs
#             (repro.sh bench)
#   table     every result in one table, $RESULTS/table.txt: regular path, monokernel, speedup
# speed, accuracy, context and serving run once per decode path (ARMS) and draft count (KS).
#
# Settings (environment variables):
#   GPU      HIP device ordinal; default 0
#   WORK     repro.sh's environment, caches and checkpoints; default $HOME/q38-repro
#   RESULTS  one directory per run, and the table; default $WORK/results ($WORK/results-smoke)
#   MODEL    an MXFP4 checkpoint to use (skips the model step); BF16: a BF16 download to convert
#   ARMS     decode paths; default "mono native"
#   KS       MTP draft tokens, 0 = no MTP; default "0 1 2 3"
#   CONCS    requests in flight for speed and serving; default "1 2 3 4"
#   STEPS    default "model speed accuracy context serving table"
#   SMOKE    1: every run with a few short requests, to check the setup (minutes per run)
#   DRY_RUN  1: list the runs only
# A run with a summary in $RESULTS is not repeated: rerun to continue, delete its directory to redo it.
# About 20 h on one GPU for everything; time on an otherwise idle GPU.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export GPU=${GPU:-0} WORK=${WORK:-$HOME/q38-repro}
SMOKE=${SMOKE:-} DRY_RUN=${DRY_RUN:-}
RESULTS=${RESULTS:-$WORK/results${SMOKE:+-smoke}}
ARMS=${ARMS:-mono native} KS=${KS:-0 1 2 3} CONCS=${CONCS:-1 2 3 4}
STEPS=${STEPS:-model speed accuracy context serving table}
mkdir -p "$RESULTS"
FAILED=()

want() { [[ " $STEPS " == *" $1 "* ]]; }
smoke() { if [ -n "$SMOKE" ]; then echo "$1"; else echo "$2"; fi; }  # smoke <smoke value> <value>

run() {  # run <name> <repro.sh command> [VAR=value ...]: one repro.sh run into $RESULTS/<name>
  local name=$1 cmd=$2 log dir
  shift 2
  [ ! -f "$RESULTS/$name/summary.txt" ] || { echo "== $name: done"; return 0; }
  echo "== $name: repro.sh $cmd $*"
  [ -z "$DRY_RUN" ] || return 0
  rm -rf "${RESULTS:?}/$name"
  log=$RESULTS/$name.log
  if env "$@" bash "$HERE/repro.sh" "$cmd" < /dev/null 2>&1 | tee "$log"; then :; else
    FAILED+=("$name")
  fi
  dir=$(sed -n 's|^repro: results in /work/runs/||p' "$log" | tail -n 1)
  if [ -n "$dir" ] && [ -d "$WORK/runs/$dir" ]; then
    mv "$WORK/runs/$dir" "$RESULTS/$name"
  else
    mkdir -p "$RESULTS/$name"
  fi
  mv "$log" "$RESULTS/$name/repro.out"
}

if want model && [ -z "${MODEL:-}" ]; then
  MODEL=$WORK/models/Qwen3.8-Flash-Next-MXFP4
  if [ -f "$MODEL/model.safetensors.index.json" ]; then
    echo "== model: $MODEL"
  else
    echo "== model: repro.sh convert into $MODEL"
    [ -n "$DRY_RUN" ] || bash "$HERE/repro.sh" convert < /dev/null
  fi
fi
export MODEL=${MODEL:-$WORK/models/Qwen3.8-Flash-Next-MXFP4}

for arm in $ARMS; do
  for k in $KS; do
    v=(DECODE="$arm" K="$k")
    ! want speed || run "speed-$arm-k$k" speed "${v[@]}" CONC="$CONCS" \
      PROMPTS="$(smoke 4 128)" MAX_TOKENS="$(smoke 64 2048)"
    if want accuracy; then
      run "gsm8k-$arm-k$k" gsm8k "${v[@]}" CONC=4 LIMIT="$(smoke 4 0)"
      run "accuracy-$arm-k$k" accuracy "${v[@]}" CONC=4 LIMIT="$(smoke 4 0)"
    fi
    if want context; then
      run "context-$arm-k$k" bench "${v[@]}" CONC=1 ISL="1024 8192" OSL="$(smoke 64 1024)" \
        PROMPTS="$(smoke 1 10)" REPS="$(smoke 1 3)"
      # ATOM keeps K positions of the model's 262,144 for the drafts.
      run "context-long-$arm-k$k" bench "${v[@]}" CONC=1 ISL="131072 250000" \
        OSL="$(smoke 64 1024)" PROMPTS="$(smoke 1 3)" MAX_MODEL_LEN=$((262144 - k))
    fi
    ! want serving || run "serving-$arm-k$k" bench "${v[@]}" CONC="$CONCS" ISL="1024 8192" \
      OSL="$(smoke 64 1024)" REPS="$(smoke 1 2)" ${SMOKE:+PROMPTS=4}
  done
done

if want table && [ -z "$DRY_RUN" ]; then
  python3 - "$RESULTS" << 'PY' | tee "$RESULTS/table.txt"
import glob, json, os, re, statistics, sys

res = sys.argv[1]


def load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def acceptance(run, tag):
    """Tokens per verify round between a bench run's MTP counter snapshots."""
    a = load(f"{run}/mtp_{tag}_after.json") or {}
    b = load(f"{run}/mtp_{tag}_before.json") or {}
    d = {int(j): v - b.get("distribution", {}).get(j, 0) for j, v in a.get("distribution", {}).items()}
    steps = sum(d.values())
    return 1 + sum(j * v for j, v in d.items()) / steps if steps else None


def runs(step):
    """{(arm, k): run dir} for a step."""
    out = {}
    for d in glob.glob(f"{res}/{step}-*-k*"):
        m = re.fullmatch(rf"{step}-(mono|native)-k(\d)", os.path.basename(d))
        if m:
            out[m.group(1), int(m.group(2))] = d
    return out


def f(x, spec):  # a missing value is "-", at the same width
    return format("-", ">" + spec.split(".")[0]) if x is None else format(x, spec)


def speedup(native, mono, higher_is_better=True):
    if not native or not mono:
        return "-"
    return f"{(mono / native if higher_is_better else native / mono):.2f}x"


print("Qwen3.8-Flash-Next, one MI355X: ATOM's regular decode (regular) against the monokernel (mono)")
print("Same ATOM server, MXFP4 checkpoint, server line and client; regular = ATOM's and aiter's own kernels;")
print("both with ATOM_RUNNER_INLINE_SEND=1, and ATOM_ENABLE_REPLAYSSM=0 when MTP is on")

# speed: per-request decode tokens/s
rows = {}
for (arm, k), d in runs("speed").items():
    for p in glob.glob(f"{d}/speed/*_c*.json"):
        name, c = re.fullmatch(r"(.+)_c(\d+)\.json", os.path.basename(p)).groups()
        rows.setdefault((name, k, int(c)), {})[arm] = load(p)["summary"]
if rows:
    print("\nSpeed: decode tokens/s per request (mean), DFlash prompts, reasoning xhigh, greedy")
    print(f"{'set':<10} {'K':>2} {'conc':>4}  {'regular':>8}  {'mono':>8}  {'speedup':>7}"
          f"  {'tok/step reg':>12}  {'tok/step mono':>13}  {'total tok/s reg':>15}  {'total tok/s mono':>16}")
    for (name, k, c), r in sorted(rows.items(), key=lambda x: (x[0][1], x[0][0], x[0][2])):
        n, m = r.get("native", {}), r.get("mono", {})
        tn, tm = n.get("decode_tok_s_per_request_mean"), m.get("decode_tok_s_per_request_mean")
        print(f"{name:<10} {k:>2} {c:>4}  {f(tn, '8.1f')}  {f(tm, '8.1f')}  {speedup(tn, tm):>7}"
              f"  {f(n.get('acceptance'), '12.2f')}  {f(m.get('acceptance'), '13.2f')}"
              f"  {f(n.get('end_to_end_tok_s'), '15.1f')}  {f(m.get('end_to_end_tok_s'), '16.1f')}")

# accuracy
rows = {}
for (arm, k), d in runs("gsm8k").items():
    for p in glob.glob(f"{d}/gsm8k/**/results_*.json", recursive=True):
        r = load(p)
        g = r["results"]["gsm8k"]
        rows.setdefault(("gsm8k", k), {})[arm] = {"n": r["n-samples"]["gsm8k"]["effective"],
                                                  "accuracy": g["exact_match,flexible-extract"],
                                                  "stderr": g["exact_match_stderr,flexible-extract"]}
for (arm, k), d in runs("accuracy").items():
    for p in glob.glob(f"{d}/accuracy_*.json"):
        s = load(p)["summary"]
        rows.setdefault((s["task"], k), {})[arm] = s
if rows:
    print("\nAccuracy, greedy, 4 requests in flight (GSM8K flexible extract; GPQA / MMLU-Pro reasoning low)")
    print(f"{'task':<10} {'K':>2}  {'questions':>9}  {'regular':>8}  {'mono':>8}  {'stderr':>6}")
    for (task, k), r in sorted(rows.items(), key=lambda x: (x[0][1], x[0][0])):
        n, m = r.get("native"), r.get("mono")
        s = m or n
        pct = [f(x and 100 * x["accuracy"], ".1f") + "%" * bool(x) for x in (n, m)]
        print(f"{task:<10} {k:>2}  {s['n']:>9}  {pct[0]:>8}  {pct[1]:>8}  {100 * s['stderr']:6.1f}")


# context and serving: benchmark_serving TPOT
def bench(step):
    rows = {}
    for (arm, k), d in list(runs(step).items()) + list(runs(f"{step}-long").items()):
        for p in glob.glob(f"{d}/bench_*.json"):
            tag = os.path.basename(p)[len("bench_"):-len(".json")]
            c, isl = map(int, re.fullmatch(r"c(\d+)_isl(\d+)_osl\d+_r\d+", tag).groups())
            s = load(p)
            r = rows.setdefault((k, c, isl), {}).setdefault(arm, {"tpot": [], "tput": [], "al": []})
            r["tpot"].append(s["mean_tpot_ms"])
            r["tput"].append(s["output_throughput"])
            al = acceptance(d, tag)
            if al:
                r["al"].append(al)
    return rows


for step, title in (("context", "Context: one request at a time, 1,024 output tokens"),
                    ("serving", "Serving: requests in flight, 1,024 output tokens")):
    rows = bench(step)
    if not rows:
        continue
    print(f"\n{title}; TPOT ms (mean over runs), output tokens/s summed over requests")
    print(f"{'ISL':>6} {'K':>2} {'conc':>4}  {'TPOT reg':>8}  {'TPOT mono':>9}  {'speedup':>7}"
          f"  {'tok/step reg':>12}  {'tok/step mono':>13}  {'tok/s reg':>9}  {'tok/s mono':>10}  runs")
    for (k, c, isl), r in sorted(rows.items(), key=lambda x: (x[0][0], x[0][2], x[0][1])):
        n, m = r.get("native", {}), r.get("mono", {})
        avg = lambda x, key: statistics.mean(x[key]) if x.get(key) else None  # noqa: E731
        print(f"{isl:>6} {k:>2} {c:>4}  {f(avg(n, 'tpot'), '8.2f')}  {f(avg(m, 'tpot'), '9.2f')}"
              f"  {speedup(avg(n, 'tpot'), avg(m, 'tpot'), False):>7}"
              f"  {f(avg(n, 'al'), '12.2f')}  {f(avg(m, 'al'), '13.2f')}"
              f"  {f(avg(n, 'tput'), '9.1f')}  {f(avg(m, 'tput'), '10.1f')}"
              f"  {len(n.get('tpot', []))}/{len(m.get('tpot', []))}")
PY
  echo "table: $RESULTS/table.txt"
fi
[ ${#FAILED[@]} = 0 ] || { echo "failed (logs in $RESULTS/<run>/repro.out): ${FAILED[*]}"; exit 1; }
