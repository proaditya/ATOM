# Qwen3.8-Flash-Next on the monokernel (TP1, MI355X)

Qwen3.8-Flash-Next decodes on one MI355X (gfx950) with one FlyDSL kernel per decoder layer. The
kernel runs MXFP4 weights and MXFP8 activations. Prefill and the final mixer stay ATOM's; a decode
step's embedding is one launch and its `lm_head` runs from MXFP4 weights. The mono path serves:
- decode steps of up to 4 requests;
- MTP verify and draft steps with 1–3 draft tokens.

ATOM's default path (`atom/models/qwen4_exp.py`) runs every other step, and both paths share ATOM's KV
cache and recurrent-state pools. `scripts/qwen38_flash_mono/README.md` explains the kernel and the
ATOM changes that let the default path serve the same MXFP4 checkpoint. ATOM's own guide for the
model, in BF16 and FP8, is [Qwen3.8-Flash-Next.md](Qwen3.8-Flash-Next.md).

## The checkpoint

`scripts/qwen38_flash_mxfp4/convert_mxfp4.py` quantizes `Qwen/Qwen3.8-Flash-Next` (BF16, 335 GiB) at
revision `de4b8e4d` into the MXFP4 checkpoint both paths load (163 GiB). It needs `torch` and
`safetensors`, and no GPU:

```bash
hf download Qwen/Qwen3.8-Flash-Next --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 --local-dir Qwen3.8-Flash-Next
python scripts/qwen38_flash_mxfp4/convert_mxfp4.py --hf Qwen3.8-Flash-Next --out Qwen3.8-Flash-Next-MXFP4
cd Qwen3.8-Flash-Next-MXFP4 && sha256sum -c /path/to/ATOM/scripts/qwen38_flash_mxfp4/Qwen3.8-Flash-Next-MXFP4.sha256
```

- **Checksums.** The conversion is deterministic. `Qwen3.8-Flash-Next-MXFP4.sha256` lists the 63
  files of the checkpoint every result below ran on, and a fresh conversion matches all of them.
- **Disk.** About 500 GiB: the 335 GiB download and the 163 GiB result.
- **Time and memory.** On a 256-core host with the BF16 files in RAM, it took 41 s and at most
  53 GiB of memory with the default 16 workers, or 50 s and 13 GiB with `--workers 4`. The
  converter maps the BF16 files instead of reading them into memory. From disk, reading the
  335 GiB takes most of the time.
- **In the container.** `repro.sh convert` runs the same in the pinned container:
  ```bash
  GPU=0 BF16=/path/to/Qwen3.8-Flash-Next bash scripts/qwen38_flash_mono/repro.sh convert
  ```
  Without `BF16`, it downloads revision `de4b8e4d` into `$WORK/models`. The model repository is
  public and ungated, so no token is needed. The script sets up its environment first, which needs
  an MI355X; on other hosts, convert by hand as above.
- **Not AMD's Quark checkpoint.** `amd/Qwen3.8-Flash-Next-Quark-MXFP4` is a different checkpoint
  and no result here used it. It has MXFP4 weights only in the decoder layers' experts; the
  attention, GatedDeltaNet and PLE projections stay BF16, and it declares FP4 activations. The mono
  path needs this converter's layout. At its first step, it checks every weight's shape and dtype.
  If one differs, it logs `Qwen3.8 mono decode off: …` and the default path serves. We have not run
  this branch on the Quark checkpoint.

## Serving

```bash
ATOM_RUNNER_INLINE_SEND=1 ATOM_ENABLE_REPLAYSSM=0 \
python -m atom.entrypoints.openai_server --model Qwen3.8-Flash-Next-MXFP4 --trust-remote-code -tp 1 \
  --max-num-seqs 4 --max-model-len 10240 --hf-overrides '{"_multimodal_config": null}' \
  --method mtp --num-speculative-tokens 3          # drop these two flags for no MTP
```

| Variable | Default | Effect |
|---|---|---|
| `ATOM_MONO_ENABLE` | 1 | `0` serves ATOM's default path. |
| `ATOM_RUNNER_INLINE_SEND` | 0 | `1` sends each step's output from the worker's main thread. Both paths use it below. |
| `ATOM_ENABLE_REPLAYSSM` | on with MTP | Must be `0` for the mono path with MTP. |

The mono path needs:
- TP1 and a text-only deployment;
- the converted checkpoint;
- the bf16 KV cache with block size 16;
- `--max-model-len` up to 262,144 (less the draft count with MTP);
- no EP, DP, PP or context parallelism.

A deployment it does not serve logs the reason once and runs the default path. The full list is in
`docs/environment_variables.md`, "Mono decode routing".

## Reproducing the results

Every result below comes from one `scripts/qwen38_flash_mono/repro.sh` command. The script:
1. starts a pinned `rocm/pytorch` container;
2. builds its environment once into `$WORK`;
3. starts the server, runs the client and prints a summary;
4. stops the server.

Each run writes `$WORK/runs/<time>-<command>-<decode>-k<K>/`, which holds the summary, the logs and
the settings. `DECODE=mono` or `native` picks the path, and `K` the MTP draft count (0 = no MTP).
`repro.sh help` lists every setting. Run on an otherwise idle GPU; the script does not check for
other processes.

```bash
export MODEL=/path/to/Qwen3.8-Flash-Next-MXFP4 GPU=0
bash scripts/qwen38_flash_mono/repro.sh setup        # once, a few minutes
```

The first server start JIT-compiles aiter and the layer kernels, which takes about 6 minutes. Later
starts reuse the cache.

**Setup of the runs below:**

| | |
|---|---|
| GPU | one MI355X (gfx950), TP1 |
| Container | `rocm/pytorch@sha256:bbdaba66…` (ROCm 10.0, PyTorch 2.13.0), aiter `c53e9b3f` |
| ATOM | `a80c8ada`; the default path runs ATOM's own code at that commit |
| Server | the line above with `--gpu-memory-utilization 0.9`; `--max-model-len 262144` for long context |
| Environment | both paths `ATOM_RUNNER_INLINE_SEND=1`, plus `ATOM_ENABLE_REPLAYSSM=0` with MTP; default path `ATOM_MONO_ENABLE=0` |

The runs used the development branch. This branch carries a cleaned copy of the same kernels; its
single-request GSM8K speed is within 1 % of the numbers below at every draft count.

- **Tokens per step:** 1 + the mean drafts accepted per verify step, from the server's MTP counters.
- **TPOT:** `benchmark_serving`'s mean over requests of (end-to-end time − TTFT) / (output tokens − 1).

## Results

Every table shows the mono path against the default path. Speedup is the default path's time over
the mono path's, or the mono path's tok/s over the default path's.

### Single-request speed: GSM8K, HumanEval, MT-Bench

```bash
DECODE=mono K=3 bash scripts/qwen38_flash_mono/repro.sh speed    # and DECODE=native, K=0..3
```

The client is `atom/benchmarks/qwen38_flash_mono/single_request_bench.py`. It runs one request at a
time, greedy, with the chat template and `reasoning_effort` xhigh, up to 2,048 output tokens. It uses
the DFlash benchmark's first 128 GSM8K, 128 HumanEval and 80 MT-Bench prompts (seed-42 order). Each
value is tok/s per request: (output tokens − 1) / (last token − first token), averaged.

| MTP drafts | GSM8K | HumanEval | MT-Bench | Tokens per step (GSM8K) |
|---|---|---|---|---|
| none | 333.3 / 107.5 (3.10×) | 331.0 / 107.3 (3.08×) | 333.1 / 107.5 (3.10×) | 1 |
| 1 | 472.1 / 169.0 (2.79×) | 450.1 / 162.6 (2.77×) | 447.4 / 160.6 (2.79×) | 1.84 / 1.84 |
| 2 | 547.5 / 191.9 (2.85×) | 503.2 / 178.0 (2.83×) | 497.6 / 173.9 (2.86×) | 2.52 / 2.51 |
| 3 | 590.4 / 221.9 (2.66×) | 520.8 / 197.2 (2.64×) | 499.1 / 189.7 (2.63×) | 3.04 / 3.02 |

- **Step cost.** Without MTP, a decode step takes 3.02 ms on the mono path against 9.32 ms on the
  default path. A verify step with 3 drafts takes 5.41 ms against 14.27 ms.
- **Same elsewhere.** Acceptance and TTFT (~170 ms) are the same on both paths.

The same prompts at 2–4 requests in flight, no MTP, in tok/s per request:

```bash
DECODE=mono K=0 CONC="2 3 4" bash scripts/qwen38_flash_mono/repro.sh speed    # and DECODE=native
```

| Requests | GSM8K | HumanEval | MT-Bench |
|---|---|---|---|
| 2 | 271.8 / 96.3 (2.82×) | 278.8 / 97.4 (2.86×) | 280.5 / 97.4 (2.88×) |
| 3 | 228.3 / 79.8 (2.86×) | 243.0 / 81.8 (2.97×) | 241.7 / 81.4 (2.97×) |
| 4 | 194.9 / 76.8 (2.54×) | 215.5 / 79.4 (2.72×) | 214.6 / 79.1 (2.71×) |

### Serving: 1K/1K and 8K/1K at 1–4 requests

```bash
DECODE=mono K=3 CONC="1 2 3 4" REPS=3 bash scripts/qwen38_flash_mono/repro.sh bench     # K=0..3; ISL 1024 and 8192, OSL 1024
DECODE=native K=3 CONC="1 2 3 4" REPS=2 bash scripts/qwen38_flash_mono/repro.sh bench
```

The client is `benchmark_serving` on random prompts, with EOS ignored. Each run uses 10, 10, 12 or 16
prompts at 1–4 requests in flight. There are 3 runs per point on the mono path and 2 on the default
path, each with its own seed, after an untimed warmup. Values are mean TPOT in ms.

| Requests | MTP drafts | 1K/1K | Speedup | 8K/1K | Speedup | Tokens per step (1K) |
|---|---|---|---|---|---|---|
| 1 | none | 3.05 / 9.38 | 3.07× | 3.18 / 9.45 | 2.97× | 1 |
| 1 | 1 | 2.25 / 6.28 | 2.79× | 2.37 / 6.35 | 2.68× | 1.78 / 1.77 |
| 1 | 2 | 1.92 / 5.48 | 2.85× | 2.02 / 5.54 | 2.74× | 2.53 / 2.47 |
| 1 | 3 | 1.89 / 5.08 | 2.69× | 2.06 / 5.21 | 2.53× | 2.93 / 2.80 |
| 2 | none | 3.51 / 10.11 | 2.88× | 3.70 / 10.22 | 2.76× | 1 |
| 2 | 1 | 2.84 / 7.41 | 2.61× | 3.41 / 7.75 | 2.27× | 1.80 / 1.78 |
| 2 | 2 | 2.74 / 5.84 | 2.14× | 3.30 / 6.46 | 1.96× | 2.40 / 2.47 |
| 2 | 3 | 2.69 / 5.72 | 2.13× | 3.10 / 5.88 | 1.89× | 2.89 / 2.77 |
| 3 | none | 3.95 / 11.91 | 3.02× | 4.16 / 12.01 | 2.89× | 1 |
| 3 | 1 | 3.56 / 7.79 | 2.19× | 4.53 / 8.67 | 1.91× | 1.80 / 1.79 |
| 3 | 2 | 3.33 / 6.39 | 1.92× | 4.38 / 7.34 | 1.68× | 2.45 / 2.47 |
| 3 | 3 | 3.53 / 6.02 | 1.70× | 4.58 / 6.94 | 1.51× | 2.79 / 2.80 |
| 4 | none | 4.33 / 12.16 | 2.81× | 4.70 / 12.47 | 2.65× | 1 |
| 4 | 1 | 4.25 / 8.38 | 1.97× | 5.60 / 9.58 | 1.71× | 1.77 / 1.77 |
| 4 | 2 | 4.08 / 6.73 | 1.65× | 5.59 / 8.51 | 1.52× | 2.46 / 2.50 |
| 4 | 3 | 4.32 / 6.41 | 1.48× | 5.91 / 7.83 | 1.32× | 2.86 / 2.81 |

- **Without MTP**, the mono path is 2.65–3.07× faster at every point.
- **With MTP**, the gap narrows as requests and drafts grow. The default path's verify step costs
  relatively less than its decode step, so it gains more from MTP.
- **TTFT** is the same on both paths, since prefill is ATOM's on both.

### Long context

```bash
DECODE=mono K=0 ISL="131072 250000" PROMPTS=3 bash scripts/qwen38_flash_mono/repro.sh bench    # and DECODE=native
```

These runs use one request at a time, 1,024 output tokens and 3 prompts. `--max-model-len` grows to
fit.

| Input | TPOT (ms) | Speedup | TTFT (s) |
|---|---|---|---|
| 128K | 3.61 / 9.91 | 2.74× | 7.6 / 7.7 |
| 250K | 3.82 / 10.30 | 2.70× | 14.7 / 14.8 |

### Accuracy

```bash
DECODE=mono K=3 CONC=4 bash scripts/qwen38_flash_mono/repro.sh gsm8k      # K=0, 2, 3; DECODE=native K=0
DECODE=mono K=0 CONC=4 bash scripts/qwen38_flash_mono/repro.sh accuracy   # HumanEval, GPQA-Diamond, MMLU-Pro; and DECODE=native
```

**GSM8K** uses `lm_eval` `local-chat-completions` on all 1,319 questions. It runs 5-shot with the
chat template and multi-turn few-shot, greedy, with up to 4,096 output tokens and 4 requests in
flight. The table gives flexible-extract accuracy; the p value is McNemar's exact test per question,
against the default path without MTP.

| Run | Accuracy | p |
|---|---|---|
| default, no MTP | 96.2 % ± 0.5 | – |
| mono, no MTP | 96.0 % ± 0.5 | 0.72 |
| mono, 2 drafts | 96.0 % ± 0.5 | 0.74 |
| mono, 3 drafts | 96.4 % ± 0.5 | 0.84 |

**HumanEval, GPQA-Diamond and MMLU-Pro** are run by `atom/benchmarks/qwen38_flash_mono/accuracy_eval.py`.
It keeps the Qwen3.8-27B evaluation's prompts, caps, stop strings and scoring. Runs are greedy, with
4 requests in flight and no MTP.

| Eval | Questions | Reasoning, cap | Mono | Default | p |
|---|---|---|---|---|---|
| HumanEval pass@1 | 164 | –, 768 tokens | 77.4 % ± 3.3 | 75.6 % ± 3.4 | 0.66 |
| GPQA-Diamond | 198 | low, 6,144 tokens | 58.1 % ± 3.5 | 57.1 % ± 3.5 | 0.87 |
| MMLU-Pro | 280 (20 per category) | low, 6,144 tokens | 78.2 % ± 2.5 | 77.1 % ± 2.5 | 0.65 |

- **Within one standard error.** Every difference is inside one standard error, and no per-question
  difference is significant.
- **GPQA cap.** About a third of GPQA replies reach the 6,144-token cap on both paths.
- **No BF16 reference.** The BF16 model needs two GPUs, so every comparison here uses the same MXFP4
  weights.

### Everything in one command

```bash
GPU=0 SMOKE=1 bash scripts/qwen38_flash_mono/build_and_repro.sh   # every run, a few requests each
GPU=0 bash scripts/qwen38_flash_mono/build_and_repro.sh           # everything, about 20 h
```

The script builds the checkpoint unless `MODEL` is set. It then runs the steps below for both paths
(`ARMS="mono native"`) and every draft count (`KS="0 1 2 3"`), and writes one table to
`$WORK/results/table.txt`. `STEPS`, `ARMS`, `KS` and `CONCS` select a subset. A rerun skips the runs
that already have a summary.

| Step | What it runs |
|---|---|
| `speed` | the single-request speed above, at 1–4 requests in flight |
| `accuracy` | GSM8K, HumanEval, GPQA-Diamond and MMLU-Pro |
| `context` | one request after 1K, 8K, 128K and 250K prompts |
| `serving` | 1–4 requests after 1K and 8K prompts |
