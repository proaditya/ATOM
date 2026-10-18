# Qwen3.8-27B single-request benchmark

One command reproduces the Qwen3.8-27B results of "Bringing ultra-fast inference to AMD Instinct GPUs" on one AMD
Instinct MI355X: DFlash2 speculative decoding on the megakernel against ATOM's regular path, both with MXFP4
weights, on GSM8K (128 prompts), HumanEval (128) and MT-Bench (80), one request at a time.

```bash
bash blog_benchmark/build_and_repro.sh
```

It installs ATOM if needed, fetches Qwen3.8-27B and the DFlash2 draft, sets up the pinned FlyDSL compiler (frontend
19418894 with the flydsl 0.4.0.dev963 wheel), builds the megakernel and exports the drafter code objects once, runs
both paths and prints:

```text
set         regular  megakernel  speedup   SGLang   published (regular / megakernel)
gsm8k           481        1105     2.3x        -   481 / 1105
...
```

- `MODEL` / `DRAFT`: local checkpoints instead of downloads; `FLYDSL_SRC` / `FLYDSL_WHEEL`: local FlyDSL copies.
- `RUN_SGLANG=1 AITERMK=<aitermk checkout>`: also run SGLang with the plugin (needs SGLang main with DFLASH).
- Everything goes under `blog_benchmark/work/` (override with `WORK`); reruns skip what is already built.

## Measuring your own setup

The client, `atom/benchmarks/single_request_bench.py`, measures single-request decode against any OpenAI-compatible
server (ATOM, SGLang or vLLM) on the same prompt sets:

```bash
# export the prompt set once (needs network access)
python atom/benchmarks/single_request_bench.py --dataset gsm8k --num-prompts 130 --export-prompts gsm8k_130.jsonl
# run it against a server
python atom/benchmarks/single_request_bench.py --base-url http://127.0.0.1:8000 \
    --prompts-file gsm8k_130.jsonl --num-prompts 128 --max-tokens 2048 --out result.json
```

It reports each request's decode speed (output tokens after the first over the time after the first) and their mean,
and, from the server's counters where it has them, the acceptance length and the time per speculative round.
`--help` lists the other prompt sets and the synthetic prompt.
