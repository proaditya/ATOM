# EAGLE-4 SPEED-Bench benchmark

One command reproduces the ATOM rows of "EAGLE-4 on AMD Instinct GPUs from day 0": Qwen3.8-27B in BF16 with
EAGLE-4 speculative decoding on SPEED-Bench (the EAGLE-4 paper's multi-turn protocol: 88 conversations, every
10th of each category, thinking on, greedy, up to 4,096 tokens a turn), at 1, 4, 16 and 32 concurrent
conversations.

In the measured image, from this checkout:

```bash
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16g \
    -v "$PWD":/workspace/ATOM -v /path/to/EAGLE-4:/workspace/EAGLE-4 -v /path/to/models:/models -w /workspace/ATOM \
    rocm/atom-dev:nightly_202610131505@sha256:b78300a6caf6c1fec3b1f27b4f7333326c9d34911bfc7ae229d13d2ebbc43884
# inside the container:
HIP_VISIBLE_DEVICES=0 EAGLE4=/workspace/EAGLE-4 DRAFT=<EAGLE-4 Qwen3.8-27B checkpoint> MODEL=/models/Qwen3.8-27B \
    bash blog_benchmark/eagle4/build_and_repro.sh
```

It installs this ATOM checkout (unless it is already the installed ATOM), fetches Qwen3.8-27B, takes the SPEED-Bench client and data from the EAGLE-4
repository, and for each concurrency serves the model with the flags of
[`recipes/Qwen3.8-27B-EAGLE-4.md`](../../recipes/Qwen3.8-27B-EAGLE-4.md) (draft trees up to 16 conversations,
chains past), warms the server up, runs the client and prints:

```text
conversations    tok/s  accepted  valid   published (MI355X)
            1      309      4.45   True   309
...
```

- Environment: the measured one is `rocm/atom-dev:nightly_202610131505@sha256:b78300a6caf6c1fec3b1f27b4f7333326c9d34911bfc7ae229d13d2ebbc43884`
  (ROCm 7.2.4, torch 2.10, aiter, FlyDSL 0.3.4.1 and ATOM's Triton pin). Elsewhere, ATOM needs aiter, FlyDSL
  0.3.4.1 and its Triton pin (`triton==3.7.0+amd.rocm7.2.0.git89002410` and `triton_kernels`, with
  `--extra-index-url https://pypi.amd.com/triton/release/rocm-7.2.0/simple`); the script checks for them.
- `EAGLE4`: a checkout of the EAGLE-4 repository (github.com/eagle-hq/EAGLE-4), which has the SPEED-Bench client
  and data; `DRAFT`: the EAGLE-4 checkpoint for Qwen3.8-27B, as the EAGLE team releases it on Hugging Face
  (`EAGLE-Corp/EAGLE4-Qwen3.8-27B`) or a local directory.
- `MODEL`: without it the script downloads Qwen/Qwen3.8-27B (~54 GB) into `WORK/models`; a local directory skips
  the download.
- The servers run on the first visible GPUs: pick them with `HIP_VISIBLE_DEVICES` (one GPU at `TP=1`).
- `TP=2` or `TP=4`: tensor parallel (the TP4 trees are the recipe's).
- `CONC="1 4 8 16 32 64"`: other concurrencies (the published column covers the post's 1, 4, 16 and 32).
- Everything goes under `blog_benchmark/eagle4/work/` (override with `WORK`); each run keeps its server log,
  the client's JSON and the server's acceptance counters.
- The client is the EAGLE-4 repository's `inference/sglang-v0.5.20/scripts/client_speedbench.py` (checked at
  EAGLE-4 `2d1fa94`; older checkouts keep it under `inference/scripts/`), copied with one change:
  token counts come from the OpenAI `usage` field (ATOM returns no SGLang `meta_info`). The tokens accepted per
  round come from ATOM's `/debug/mtp_stats` around each run.
- A run at 1 conversation takes ~13 minutes, 4 about 5, 16 and 32 about 2-3, plus ~2 minutes of server start
  each. An MI350X measures ~8-9% below the published MI355X numbers up to 8 conversations and more at larger
  batches (see the recipe).

To serve EAGLE-4 for your own workload, take the flags from the recipe; to retune trees or GEMMs for another
GPU, TP size or kernel change, see its "Retuning" section.
