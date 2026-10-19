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
gsm8k           535        1091     2.0x        -   535 / 1091
...
```

The published pair is MI355X, ROCm 10.0, GPU_MAX_HW_QUEUES=1 on both paths (the script's default): 535 / 1,091, 443 / 913, 412 / 824
tok/s (regular / megakernel on GSM8K, HumanEval, MT-Bench; 2.0-2.1x). On ROCm 7.2.4 at HIP's defaults: 481 / 1,105,
406 / 924, 368 / 846 (2.3x); on ROCm 10 the hardware-queue setting also sped up the regular path on that MI355X.

- `MODEL` / `DRAFT`: local checkpoints instead of downloads; `FLYDSL_SRC` / `FLYDSL_WHEEL`: local FlyDSL copies.
- `RUN_SGLANG=1 AITERMK=<aitermk checkout>`: also run SGLang with the plugin (needs SGLang main with DFLASH).
- Everything goes under `blog_benchmark/work/` (override with `WORK`); reruns skip what is already built.
- The servers run with `GPU_MAX_HW_QUEUES=1`: on ROCm 10 HIP's default of four hardware queues makes every kernel of
  a megakernel round slower (6.1 instead of 5.5 ms per round on an MI350X), and one queue is also the regular path's
  faster setting. ROCm 7.2 is unaffected.

## Docker

`Dockerfile` here builds the ROCm 10.0 environment of the measured runs: the PyTorch image
`rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.13.0` pinned by digest (torch 2.13.0+rocm10.0.0), ROCm's
devel wheels, aiter `c53e9b3f`, ATOM from the checkout (scripts/qwen38_flash_mono/repro.sh's setup and constraints,
in the image's own Python; no git history in the image), FlyDSL set up for that Python, and the megakernel and drafter
code objects, prebuilt (`SETUP_ONLY=1` runs the script's steps 1, 3 and 4, which need no GPU). It sets
`GPU_MAX_HW_QUEUES=1`. No weights. From the checkout:

```bash
docker build -f blog_benchmark/Dockerfile -t atom-qwen38-27b .
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16g \
    -v /path/to/models:/models -e MODEL=/models/Qwen3.8-27B -e DRAFT=/models/Qwen3.8-27B-DFlash2 \
    -v "$PWD/work:/work" atom-qwen38-27b
```

The container runs `build_and_repro.sh`: the prompt sets, the results and any checkpoint given as a Hugging Face id
(`MODEL` / `DRAFT` default to the ids) go to the `/work` volume; FlyDSL and the code objects stay in the image
(`FLYDSL_DIR`, `DRAFTER_OBJECTS`, `ATOM_QWEN38_MK_CACHE`). Checked on an MI350X (2026-10-18), from a fresh clone: a
50.8 GB image (the base's 31 GB plus ROCm's devel wheels and aiter) built in 328 s, then 998 / 844 / 752 tok/s on
the megakernel at 5.61-5.64 ms per round and 485 / 400 / 371 on the regular path at 11.2-11.3 ms, as the same
checkout outside Docker on that GPU (about 3 minutes to start each server, 33 for the whole run). On another MI350X
(2026-10-17) the megakernel's outputs first differed from its own greedy decode at median token 46 / 48 on GSM8K /
MT-Bench, against 43 between two identical runs. The image keeps the Triton torch pins (aiter's build would otherwise
replace it with a ROCm 7.2 wheel, which slows ATOM's Triton kernels; the build checks it).

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
