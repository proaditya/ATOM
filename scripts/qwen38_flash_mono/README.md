# Qwen3.8-Flash-Next monokernel decode in ATOM

This branch serves Qwen3.8-Flash-Next on one AMD Instinct MI355X (gfx950) two ways, in the same ATOM
server, on the same MXFP4 checkpoint:

- **Default path:** ATOM's own model (`atom/models/qwen4_exp.py`), one kernel per op: ATOM's Triton
  GEMM, GatedDeltaNet and QSA kernels, aiter's FlyDSL MoE, norms and `lm_head` GEMM.
- **Mono path:** the same model, but each decoder layer's decode step is one FlyDSL kernel
  (`atom/models/qwen4_exp_mono`). Prefill and the final mixer stay ATOM's; a decode step's
  embedding is one launch and its `lm_head` runs from MXFP4 weights.

## The monokernel

A decode step of one decoder layer is a single launch of 256 workgroups × 512 threads. The launch runs
all of the layer's stages in order:

1. the attention (QSA sparse attention or GatedDeltaNet);
2. the PLE embedding mix;
3. the MoE with its fused shared expert;
4. the hyper-connection streams.

Stages hand data to each other through mailboxes in device memory instead of kernel launches. The
weights are MXFP4 and the activations MXFP8.

The mono path reads and writes ATOM's own KV cache and recurrent-state pools, so either path can run any
step. It serves:
- decode steps of up to 4 requests;
- MTP with 1, 2 or 3 draft tokens: the verify step and the drafter's decoder layer.

ATOM's default path runs every other step. The kernel started as a standalone FlyDSL monokernel and was
ported into ATOM.

## How we brought it up

At the time (ATOM `a80c8ada`), ATOM served Qwen3.8-Flash-Next with BF16 or FP8 weights only
(`recipes/Qwen3.8-Flash-Next.md`). There was no MXFP4 path for this model. The BF16 checkpoint
(335 GiB) does not fit one MI355X (288 GB). The monokernel computes MXFP4 weights × MXFP8 activations,
so we gave the default path the same MXFP4 checkpoint and numerics:

1. **The checkpoint.** `scripts/qwen38_flash_mxfp4/convert_mxfp4.py` quantizes the public BF16
   checkpoint into the Quark MXFP4 layout ATOM loads (163 GiB): E2M1 values with one E8M0 scale per
   32. `lm_head` and the hyper-connection mix weights are rounded to MXFP4 values and kept in BF16.
2. **ATOM's default path on it.**
   - GatedDeltaNet accepts MXFP4 QKV/Z projections (`atom/models/qwen4_exp.py`).
   - The MoE runs MXFP4 weights on MXFP8 activations with aiter's FlyDSL kernels (`qwen4_exp.py`).
   - The dense linears run W4A8 through ATOM's existing W4A8 group-32 linear (`atom/quant_spec.py`).
     ATOM's default for Quark MXFP4 is FP4 activations, which lost 3.3 GSM8K points against FP8.
   - Fewer launches and host stalls a step (`hyperconnection.py`, `ops/qsa.py`, `qwen4_exp_attn.py`):
     the final hyper-connection mix runs the two fused kernels up to 8 rows, a draft step advances its
     lengths and positions inside its QSA metadata launch, and the accepted-token counts reach the GPU
     in one copy from pinned memory.
3. **The mono path,** `atom/models/qwen4_exp_mono`, is routed by `ATOM_MONO_ENABLE` (on by
   default). A deployment it does not serve logs the reason once and runs the default path.
4. **`ATOM_RUNNER_INLINE_SEND=1`** (new, both paths). ATOM's model runner sends each step's output
   from its main thread instead of handing it to another thread.

## Running it

You need one MI355X. The scripts also need Docker: they pull a pinned `rocm/pytorch` image (ROCm
10.0, PyTorch 2.13).

**1. The checkpoint**, on CPU, from the BF16 model at revision `de4b8e4d`:

```bash
hf download Qwen/Qwen3.8-Flash-Next --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 --local-dir Qwen3.8-Flash-Next
python scripts/qwen38_flash_mxfp4/convert_mxfp4.py --hf Qwen3.8-Flash-Next --out Qwen3.8-Flash-Next-MXFP4
# or in the container on an MI355X; downloads that revision unless BF16 points at a copy:
GPU=0 BF16=/path/to/Qwen3.8-Flash-Next bash scripts/qwen38_flash_mono/repro.sh convert
```

The [recipe](../../recipes/Qwen3.8-Flash-Next-monokernel.md#the-checkpoint) lists the disk and
memory it needs and the checksums of the result. The mono path needs this checkpoint. AMD's
`amd/Qwen3.8-Flash-Next-Quark-MXFP4` is a different one: it keeps the attention and GatedDeltaNet
projections in BF16.

**2. Serve it.** The mono path is on by default; `ATOM_MONO_ENABLE=0` gives the default path:

```bash
ATOM_RUNNER_INLINE_SEND=1 ATOM_ENABLE_REPLAYSSM=0 \
python -m atom.entrypoints.openai_server --model Qwen3.8-Flash-Next-MXFP4 --trust-remote-code -tp 1 \
  --max-num-seqs 4 --max-model-len 10240 --hf-overrides '{"_multimodal_config": null}' \
  --method mtp --num-speculative-tokens 3          # drop these two flags for no MTP
```

The mono path needs:
- TP1 and text only (the `--hf-overrides` above);
- the converter's MXFP4 checkpoint;
- the bf16 KV cache with block size 16;
- `--max-model-len` of at most 262,144 (less the draft count with MTP);
- `ATOM_ENABLE_REPLAYSSM=0` with MTP.

At the first step it serves, the log says `Qwen3.8 mono decode on`. The first start JIT-compiles
aiter and the layer kernels, which takes about 6 minutes; later starts reuse the cache.

**3. Measure it.** `repro.sh` runs one test in the pinned container. `DECODE=mono` or `native` picks
the path and `K` the draft count. `build_and_repro.sh` runs everything, for both paths, and writes one
table:

```bash
export MODEL=/path/to/Qwen3.8-Flash-Next-MXFP4 GPU=0
DECODE=mono K=3 ISL="1024 8192" PROMPTS=10 REPS=3 bash scripts/qwen38_flash_mono/repro.sh bench
bash scripts/qwen38_flash_mono/repro.sh help                      # every command and setting
GPU=0 SMOKE=1 bash scripts/qwen38_flash_mono/build_and_repro.sh   # every test, a few requests each
```

## Result

Mean TPOT in ms at one request at a time, with random 1K and 8K prompts and 1,024 output tokens
(`repro.sh bench`). Speedup is the default path's TPOT over the mono path's.

| MTP drafts | 1K/1K mono | 1K/1K default | Speedup | 8K/1K mono | 8K/1K default | Speedup |
|---|---|---|---|---|---|---|
| none | 3.05 | 9.38 | 3.07× | 3.18 | 9.45 | 2.97× |
| 1 | 2.25 | 6.28 | 2.79× | 2.37 | 6.35 | 2.68× |
| 2 | 1.92 | 5.48 | 2.85× | 2.02 | 5.54 | 2.74× |
| 3 | 1.89 | 5.08 | 2.69× | 2.06 | 5.21 | 2.53× |

Both paths accept the same number of tokens per step (1.8 / 2.5 / 2.8–2.9 with 1 / 2 / 3 drafts).
The default path was measured at ATOM `a80c8ada`; later ATOM releases keep improving it.

[recipes/Qwen3.8-Flash-Next-monokernel.md](../../recipes/Qwen3.8-Flash-Next-monokernel.md) has every
result with the command that reproduces it: single-request speed, serving at 1–4 requests, long
context and accuracy.

## Code

| Path | What |
|---|---|
| `atom/models/qwen4_exp_mono/` | the mono path: routing (`dispatch.py`), ATOM's caches and weights (`state.py`, `weights.py`), the single-request, grouped and draft runners |
| `atom/models/qwen4_exp_mono/kernels/` | the FlyDSL layer kernel: `kernel.py` builds a layer from the stages in `stages/` |
| `scripts/qwen38_flash_mxfp4/convert_mxfp4.py` | BF16 → MXFP4 checkpoint |
| `scripts/qwen38_flash_mono/` | `repro.sh`, `build_and_repro.sh`, the pinned pip constraints |
| `atom/benchmarks/qwen38_flash_mono/` | the per-request speed client and the accuracy evals |
| `tests/test_qwen4_exp_mono_*.py` | routing, cache binding, grouped steps, build keys |
