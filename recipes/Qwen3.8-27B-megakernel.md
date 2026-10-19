# Qwen3.8-27B on the megakernel (TP1, MI355X)

Qwen3.8-27B (`Qwen3_5ForConditionalGeneration`, dense hybrid: 64 layers, 48
Gated DeltaNet + 16 full attention, hidden 5120) decodes one sequence at a time
through a persistent megakernel: the whole forward of up to 8 consecutive
positions -- embedding, 64 layers, final norm, LM head -- in one launch of 256
workgroups, with MXFP4 weights and MXFP8 activations on the scaled MFMA. It
verifies speculative drafts the same way. Everything else (prefill, batches of
several sequences) runs the per-op model, and either path can take over at any
step: they share ATOM's KV cache and recurrent-state pools.

- `atom/models/qwen38_27b/mono/` -- weights (`weights.py`, `mxfp4.py`), the
  kernel's pointer table (`binding.py`, `dims.py`), code objects
  (`kernels/build.py`, `kernels/hip.py`), the step over ATOM's pools
  (`engine.py`, `launch_kernels.py`, `state_kernels.py`), a greedy verify's
  sampling (`verdict.py`), routing (`dispatch.py`, `config.py`).

## The kernel

The kernels are FlyDSL programs in `mono/kernels/`:

| build | rows | source |
|---|---:|---|
| `m8` (default) | 8 | `verify_m8.py` |
| `m8_long` | 8 | `verify_m8_long.py`: row-octet attention above 16K positions |
| `m16` | 16 | `verify_m16.py` |

Each build adds the GDN replay records a speculative commit replays
(`gen_replay_records.py`). `ATOM_QWEN38_MK_KERNELS` picks the builds a
deployment loads, one per width (e.g. `m8,m16`).

They compile with FlyDSL frontend `19418894` over the `flydsl==0.4.0.dev963`
wheel, compile-only, into a cache:

```bash
export ATOM_QWEN38_MK_FLYDSL=/path/to/frontend-19418894/python   # flydsl/ with the dev963 native libs linked in
export ATOM_QWEN38_MK_CACHE=/path/to/cache                       # default ~/.cache/atom/qwen38_27b_mono
python -c "from atom.models.qwen38_27b.mono.kernels.build import build; print(build('m8'))"
```

Serving loads `kernel.hsaco` through HIP; it needs no FlyDSL. A missing
object is built at startup when `ATOM_QWEN38_MK_FLYDSL` is set.

## Weights

The kernel reads MXFP4 weights (E2M1, an E8M0 scale per 32) in the CDNA4
preshuffled layout, with MXFP8 activations. `weights.py` converts the Hugging
Face checkpoint at load time (~15 s on the GPU); check the conversion with

```bash
python tools/qwen38_27b_mono_check_weights.py /models/Qwen/Qwen3.8-27B
```

(every tensor of all 64 layers and the head against `native_sha256.json`).
The per-op path keeps the bf16 checkpoint: prefill and batched decode run in
bf16, single-sequence decode in MXFP4.

## Serving

```bash
python -m atom.entrypoints.openai_server --model Qwen/Qwen3.8-27B \
    -tp 1 --kv_cache_dtype bf16 --max-model-len 40000 --max-num-seqs 4
```

The megakernel serves a deployment when (`config.config_refusal`):

- TP1, no PP / DP, `--kv_cache_dtype bf16`, KV block size a multiple of 8;
- `--max-model-len` <= 131064 (the 8-row build's split layout's limit);
- no speculation, or a chain of at most 7 drafted tokens (15 with `m16`);
- ReplaySSM not forced on (`ATOM_ENABLE_REPLAYSSM=1`): the kernel commits into
  per-token state slots;
- `ATOM_MONO_ENABLE` is not `0`.

It then keeps the recurrent state in fp32 in the layout the kernel reads
(`[head][v][k]`, the "VK" transposed view ATOM's recurrent kernels accept).

On ROCm 10 images (ROCm as TheRock wheels), serve with `GPU_MAX_HW_QUEUES=2`: with
HIP's default of four hardware queues per device every kernel of a megakernel
round runs ~8 us longer (6.1 instead of 5.5 ms per round on an MI350X). ROCm 7.2
is unaffected, and so is the per-op path.
With speculation every request holds `num_speculative_tokens + 1` fp32 state
slots (~150 MB each), so keep `--max-num-seqs` small.

### DFlash 2

```bash
python -m atom.entrypoints.openai_server --model Qwen/Qwen3.8-27B \
    -tp 1 --kv_cache_dtype bf16 --max-model-len 40000 --max-num-seqs 4 \
    --method dflash --draft-model /path/to/Qwen3.8-27B-DFlash2 --num-speculative-tokens 7
```

DFlash 2 (`DFlash2DraftModel`, `atom/models/qwen3_dflash2.py`,
`atom/spec_decode/dflash2_proposer.py`) drafts a block of 7 tokens in one
pass of five layers over the target's context, and the target verifies them
as a chain. The draft's context is the target's residual stream after layers
5 / 19 / 33 / 47 / 61 of every position the target ran, kept in a paged draft
KV pool riding the target's block ids. The block pass for a batch is one HIP
graph of fused Triton kernels (`ATOM_DFLASH2_EAGER=1` runs it eagerly). On
the megakernel, a single-sequence verify of the 8 rows is one launch, and its
residual streams feed the draft's context; past 4 query rows the per-op
verify's attention runs a split-KV kernel (`triton_verify_splitkv.py`).

### DFlash 2 on the megakernel: the persistent drafter

```bash
ATOM_DFLASH2_AOT=/path/to/drafter-objects python -m atom.entrypoints.openai_server ... \
    --method dflash --draft-model /path/to/Qwen3.8-27B-DFlash2 --num-speculative-tokens 7
```

With `ATOM_DFLASH2_AOT`, single-sequence steps draft on DFlash2Mono
(`mono/drafter/`): each draft layer one persistent FlyDSL launch with MXFP8
GEMMs, and an LM head that screens the whole vocabulary at MXFP4 and rescores
its top 128 a row with the exact bf16 rows before the top-16 and the selector.
It reads the megakernel's residual streams in place, and a verify's context
update runs inside the draft's HIP graph, after the verdict the host waits on
(`ATOM_DFLASH2_ROUND_GRAPH=0`: right after the verify instead). Its KV mirrors
the paged draft pool, which stays the record, so requests move between
single-sequence and batched steps at any time; batches keep the Triton draft.

The drafter's kernels are exported ahead of time as code objects plus a
manifest; serving loads them through HIP and imports no FlyDSL:

```bash
FLY_KERNELS_ROOT=/path/to/FlyDSL/kernels-tree PROTO_ROOT=atom/models/qwen38_27b/mono/drafter/kernels_src \
    python atom/models/qwen38_27b/mono/drafter/aot_runtime.py export --out /path/to/drafter-objects --P 8256,40968,131136
```

`--P` lists the drafter's maximum positions to build; the serving
`--max-model-len` + 8 must fit one of them.

## How a step runs

1. If a verify is still pending, its accepted prefix is committed (GDN state
   replayed from the records into the slot ATOM reads next). Normally that
   happens right after sampling, in `commit_speculative_state`.
2. The table points each DeltaNet layer at the request's state slot; the conv
   window is gathered from ATOM's rolling buffer.
3. The kernel's KV mirror (32-position pages, one sequence) imports whatever
   ATOM's cache holds beyond what it saw (all of it after a step it did not run).
4. One launch over the verify rows (padded to the build's 8 or 16).
5. The conv buffer and this launch's K / V are written back to ATOM's pools, so
   the per-op path can run the next step.

The bookkeeping between these is a handful of Triton launches
(`launch_kernels`), not dozens of torch ops. A greedy verify that ran on the
kernel is sampled from its logits in three launches (row argmax, then the
accept walk, `verdict.py`).

## Reproducing the single-request numbers

`blog_benchmark/build_and_repro.sh` builds everything (pinned FlyDSL, the megakernel, the drafter code
objects) and serves DFlash2 twice on one MI355X: on the megakernel, and on ATOM's regular path with the same
weights quantized to MXFP4 at load. Each arm runs the DFlash benchmark's prompt sets one request at a time:
GSM8K and HumanEval (each test split shuffled with seed 42, the first 128) and MT-Bench (80 first turns),
reasoning effort xhigh, greedy, up to 2,048 tokens. See `blog_benchmark/README.md` for its options.

```bash
bash blog_benchmark/build_and_repro.sh
```

The client, `atom/benchmarks/single_request_bench.py`, works against any OpenAI-compatible server (ATOM,
vLLM, SGLang). It reports each request's decode speed (output tokens after the first over the time after the
first) and its mean, the acceptance length and the time per round from ATOM's `/debug/mtp_stats`. Measured
on one MI355X (2026-10-14):

| set | regular path, MXFP4 | megakernel | ms per round |
|---|---:|---:|---|
| GSM8K (128) | 481 tok/s | 1,105 tok/s | 11.33 / 5.07 |
| HumanEval (128) | 406 tok/s | 924 tok/s | 11.39 / 5.09 |
| MT-Bench (80) | 368 tok/s | 846 tok/s | 11.36 / 5.08 |

The megakernel is not bit-reproducible run to run (see Validation), so speeds move with the acceptance
length by a few percent between runs; the time per round is the stable figure.

## Validation

- `tools/qwen38_27b_mono_check_weights.py` -- converted weights against the
  reference hashes.
- `tools/qwen38_27b_mono_smoke.py` -- the kernel alone over stand-in pools;
  `--teacher-force OUT.json` runs a generation's prompt + completion through it
  and compares every token with the kernel's argmax, twice, giving the kernel's
  own run-to-run noise (it is not bit-reproducible: atomic split-K + MXFP8;
  single logits move by up to ~10 between identical runs, so judge a stream by
  its mismatch count against the self flips, not by one large gap).
- `tools/qwen38_27b_mono_generate.py` -- batch-1 tok/s, TTFT and, with
  speculation, the accepted length and steps per prompt.
- `tools/qwen38_27b_eval.py` -- GSM8K, HumanEval, GPQA-Diamond, MMLU-Pro.
- `tests/models/test_qwen38_27b_mono_mxfp4.py` -- the MXFP4 conversion (CPU).
