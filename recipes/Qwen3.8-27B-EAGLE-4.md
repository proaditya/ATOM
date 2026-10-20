# Qwen3.8-27B with EAGLE-4 speculative decoding

[EAGLE-4](https://github.com/eagle-hq/EAGLE-4) drafts for Qwen3.8-27B (BF16,
dense, hybrid full attention + Gated DeltaNet) with one trained draft layer
and a low-rank candidate head. ATOM serves it through `--method eagle3` with
an EAGLE-4 checkpoint (`LlamaForCausalLMEagle4`), drafting either a chain or
an EAGLE-2 dynamic tree per round:

- the whole draft round is one CUDA graph of fused kernels a batch size
  (`spec_decode/eagle4_fast.py`), at any TP;
- a tree is verified in one target forward: full-attention rows see their
  ancestors only (split-KV verify kernel with a tree mask, rows RoPE'd at the
  context end + their depth), the GDN layers run the tree as one
  ancestor-masked chunk of the gated delta rule (the FlyDSL kernel of
  SGLang's EAGLE-4 ROCm port), and the accepted path's state, conv window and K / V are
  committed into place (`spec_decode/tree_verify.py`, `tree_commit.py`);
- the tree shape is fixed per batch size (`ATOM_SPEC_TREE`), and past ~24
  requests chains with a batch-dependent verify width win
  (`ATOM_SPEC_VERIFY_LEN`).

## Environment

`rocm/atom-dev:nightly_202610131505` (ROCm 7.2.4, torch 2.10, Python 3.12, aiter,
FlyDSL 0.3.4.1 and ATOM's Triton pin), with this checkout installed over
the image's ATOM (`pip install --no-build-isolation --no-deps -e .`). Elsewhere:
aiter, FlyDSL 0.3.4.1 and the Triton pin
(`triton==3.7.0+amd.rocm7.2.0.git89002410` and `triton_kernels`, with
`--extra-index-url https://pypi.amd.com/triton/release/rocm-7.2.0/simple`).
Tree mode needs a bf16 KV cache.

## Serving

```bash
MODEL=Qwen/Qwen3.8-27B
DRAFT=/path/to/EAGLE4-Qwen3.8-27B
export AITER_LOG_LEVEL=WARNING
export ATOM_EAGLE4_DRAFT_MXFP4=1 ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export AITER_CONFIG_GEMM_BF16=$AITER/aiter/configs/bf16_tuned_gemm.csv:recipes/configs/qwen38_27b_bf16_tuned_gemm.csv
ATOM_SPEC_TREE=1=dyn:8:4:16,4=dyn:6:4:12,12=dyn:5:4:8 \
python -m atom.entrypoints.openai_server --model $MODEL -tp 1 --kv_cache_dtype bf16 \
    --max-model-len 40000 --max-num-seqs 16 --no-enable_prefix_caching \
    --cudagraph-capture-sizes [1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64] \
    --method eagle3 --draft-model $DRAFT --num-speculative-tokens 15
```

Speculation by expected concurrency C (MI355X; re-derive for other GPUs, TP
sizes or kernels, below):

| C | TP1 / TP2 | TP4 |
|---|---|---|
| 1-4 | `ATOM_SPEC_TREE=dyn:8:4:16`, `--num-speculative-tokens 15` | `1=dyn:8:4:16,4=dyn:7:4:12,8=dyn:5:4:8`, 15 |
| up to 16 | `1=dyn:8:4:16,4=dyn:6:4:12,12=dyn:5:4:8`, 15 | as above |
| 32-64 | chains: no `ATOM_SPEC_TREE`, `--num-speculative-tokens 5`, `ATOM_SPEC_VERIFY_LEN=1:5,16:3` | as TP1 |

- `ATOM_SPEC_TREE=dyn:STEPS:WIDTH:NODES`: the draft grows a beam of WIDTH
  nodes STEPS deep, each step expanding every beam node's best WIDTH
  children by path log probability, and keeps the best NODES - 1 of all
  nodes scored (`--num-speculative-tokens` = NODES - 1, at most 15).
  `B1=dyn:...,B2=dyn:...` picks a tree per batch size (from B requests on,
  the first from 1); the verify width follows the tree.
- `ATOM_SPEC_VERIFY_LEN=BS:DRAFTS,...`: chains verify fewer drafts as the
  batch grows (the verify is weight-bound up to ~128 rows).
- `ATOM_EAGLE4_DRAFT_MXFP4=1`: the draft layer's projections in MXFP4 up to
  16 rows (weights quantized at load; the target stays bf16).
- `ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1`: the full-attention
  layers' q / k norm, partial RoPE and cache write in one kernel.
- `AITER_CONFIG_GEMM_BF16`: aiter merges the listed CSVs; ours adds
  hipBLASLt solutions for the verify's bf16 projections at 16-256 rows (TP1,
  TP2 and TP4 shards), picked with the weights cache-cold: the batch-1 round
  15.9 -> 14.8 ms. Processes with different settings must not start at once
  in one container (they share aiter's merged file in `/tmp/aiter_configs/`).
- `--no-enable_prefix_caching`: as SGLang's `--disable-radix-cache`; with it
  the hybrid model's state checkpoints split each prefill into ~3 steps.
- `--cudagraph-capture-sizes`: the default list jumps 16 -> 32 -> 48, so 17-31
  requests would run padded to 32.
- At TP > 1 the all-reduce after each row-parallel projection is fused into
  the next RMSNorm (`ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION`, on by default).

The draft round's graphs are captured on first use of each batch size: warm a
server with C concurrent requests before measuring it.

## Retuning

SGLang's EAGLE-4 harness decides each round's tree on the host from offline
cost tables and the draft's scores. ATOM fixes the tree per batch size
instead (the round is one CUDA graph a batch size; a per-round host decision
would need a device-to-host read mid-round). When kernels, GEMM tuning, TP or
the GPU change, re-derive the GEMM table and the tree policy:

```bash
# bf16 GEMM solutions for the TP shard's projections, timed cache-cold (resumable)
python tools/bf16_gemm_cold_tune.py --shapes 16480,5120 34816,5120 14336,5120 5120,6144 5120,17408 \
    --m 16,32,48,64,96,128,160,256 --out gemm.csv --merge recipes/configs/qwen38_27b_bf16_tuned_gemm.csv
# the tree a batch size (one engine a candidate, ~100 s each)
AITER_CONFIG_GEMM_BF16=$AITER/aiter/configs/bf16_tuned_gemm.csv:gemm.csv \
python tools/qwen38_27b_tree_tune.py --model $MODEL --draft-model $DRAFT --questions question_multiturn.jsonl \
    --tp 1 --batch-sizes 1,2,4,8,12,16,24,32 -- --kv_cache_dtype bf16 --level 3
```

Shard shapes at TP t: GDN in_proj (16480 / t, 5120), attention qkv + gate
(14336 / t, 5120), gate_up (34816 / t, 5120), o / out_proj (5120, 6144 / t),
down (5120, 17408 / t). Confirm a policy on SPEED-Bench.

## Results

SPEED-Bench (the EAGLE-4 paper's multi-turn protocol: 88 conversations, every
10th of each category, replayed turn by turn, up to 4096 tokens a turn,
thinking on, greedy), MI355X, `--level 3`, the flags above; total generated
tokens / wall time, mean of repeated runs (min-max), accepted tokens per
round in brackets for TP1. SGLang: the EAGLE-4 fork with its harness, best
configuration, same client and data.

| TP | conversations | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 1 | ATOM (3 runs) | 311 (309-312) | 979 (976-984) | 1475 (1470-1482) | 2239 (2140-2309) | 3253 (3243-3261) | 4048 (4020-4062) |
| 1 | tokens / round | 4.45 | 4.52 | 4.11 | 3.75 | 2.80 | 2.77 |
| 1 | SGLang | 294 | 966 | 1436 | 2169 | 2932 | 3566 |
| 2 | ATOM (3 runs; 8: 2) | 400 (397-403) | 1128 (1110-1143) | 1638 (1620-1657) | 2552 (2531-2589) | 3676 (3664-3698) | 4472 (4294-4603) |
| 2 | SGLang | 345 | 1101 | 1645 | 2504 | 3384 | 4325 |
| 4 | ATOM (2 runs) | 489 (488-490) | 1347 (1345-1349) | 2130 (2094-2165) | 3381 (3376-3385) | 4796 (4757-4835) | 6290 (6144-6435) |
| 4 | SGLang | 473 | 1390 | 2249 | 3245 | 4765 | 6015 |

At TP4 with 4-8 conversations ATOM's round costs what SGLang's does (12.4 vs
12.5 ms at 4) but accepts slightly fewer tokens (4.2 vs 4.3): the harness
sizes each round's tree.

MI350X (1000 W, one run, TP1, the same flags and trees): 285 / 902 / 1337 /
1972 / 2562 / 2737 at 1 / 4 / 8 / 16 / 32 / 64 conversations (tokens a round
4.46 / 4.48 / 4.13 / 3.71 / 2.79 / 2.83, as on MI355X). That is 8-9% below
MI355X up to 8 conversations and 12% / 21% / 32% below at 16 / 32 / 64,
where the GPU runs near its 1000 W cap (935 W and ~2.0 GHz seen at 8).

The client is the EAGLE-4 repository's
`inference/sglang-v0.5.20/scripts/client_speedbench.py` (`inference/scripts/`
before EAGLE-4 `7d8f2ab`),
reading token counts from the response's `usage` (ATOM's server returns no
SGLang `meta_info`); the acceptance length from `/debug/mtp_stats` before and
after a run.

Losslessness (greedy): on 32 SPEED-Bench first turns of 512 tokens (TP1),
tree speculation's outputs first differ from plain decoding at a median
151-154 tokens (batch 1 and 16), as late as plain decoding at batch 16
differs from plain decoding at batch 1 (144): trees change bf16 rounding the
way batching does. GSM8K (`tools/qwen38_27b_gsm8k.py`, 250 questions,
5-shot, 256 tokens, standard error +-2.9%): chains at TP1 68.4%; trees at TP1
68.4% (batch 16, MI350X), at TP2 68.8% / 67.6% (batch 1 / 16), at TP4 72%
(100 questions, batch 1) / 69.2%.

GPU tests: `tests/models/test_eagle4_fast_draft_gpu.py` (the fused draft
layer against a reference, bf16 and MXFP4; batched rounds against single
rounds; dynamic trees; the candidate head), `tests/models/test_gdn_tree_gpu.py`
(tree conv, chunk, commit and walk against fp64 references; FlyDSL against
Triton), `tests/models/test_verify_splitkv_gpu.py` (split-KV verify and its
tree mask).
