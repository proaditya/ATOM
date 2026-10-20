#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pick ``ATOM_SPEC_TREE``'s per-batch tree policy for this machine by measurement.

The SGLang EAGLE-4 harness picks a tree for each round from offline cost data
(per-batch verify / draft costs) and the draft's own scores. ATOM fixes one
tree per batch size instead (``1=dyn:8:4:16,4=dyn:6:4:12,...``), so when
kernels, GEMM tuning, TP or the GPU change, re-run this to re-derive it:

    python tools/qwen38_27b_tree_tune.py --model Qwen/Qwen3.8-27B \\
        --draft-model /path/to/EAGLE4-Qwen3.8-27B --questions question_multiturn.jsonl \\
        --batch-sizes 1,2,4,8,12,16,24,32 \\
        --candidates dyn:8:4:16,dyn:6:4:12,dyn:5:4:10,dyn:5:4:8,dyn:4:2:6,dyn:3:2:4 \\
        --tp 1 --out tree_tune.json -- --kv_cache_dtype bf16 --level 3

For each candidate tree it starts one engine (the tree set through the
environment, as a deployment sets it) and decodes waves of ``b`` concurrent
requests at every batch size ``b`` (chat-templated first turns of the
questions, thinking on, greedy, ``--max-tokens`` each, at least
``--min-requests`` a size), measuring generated tokens over wall time. The
policy takes each batch size's fastest candidate, keeping the previous size's
unless another is ``--margin`` faster. It prints the policy and the table and
writes both to ``--out``. About 100 s a candidate at TP4.

Engine arguments after ``--`` pass through (e.g. ``--kv_cache_dtype bf16``,
``--level 3``); export the deployment's other variables first
(``ATOM_EAGLE4_DRAFT_MXFP4``, ``AITER_CONFIG_GEMM_BF16``, ...). Large batches
(past ~24 requests at TP1) usually run faster on the chain path
(``ATOM_SPEC_VERIFY_LEN``, no ``ATOM_SPEC_TREE``): compare with its SPEED-Bench
numbers before choosing tree mode for them.
"""

import argparse
import json
import os
import subprocess
import sys
import time


def parse_dyn(spec: str) -> int:
    """A ``dyn:STEPS:WIDTH:NODES`` tree's node count."""
    kind, *nums = spec.split(":")
    if kind != "dyn" or len(nums) != 3:
        raise ValueError(f"candidate {spec!r}: expected dyn:STEPS:WIDTH:NODES")
    return int(nums[2])


def load_prompts(tok, path: str, n: int) -> list[list[int]]:
    """The first turns of the questions (SPEED-Bench jsonl: a ``turns`` list or a ``question``), chat-wrapped."""
    out = []
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        row = json.loads(line)
        turns = row.get("turns") or [row.get("question") or row.get("prompt")]
        text = tok.apply_chat_template(
            [{"role": "user", "content": turns[0]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        out.append(tok(text, add_special_tokens=False)["input_ids"])
        if len(out) == n:
            break
    if len(out) < n:
        out = (out * (n // max(1, len(out)) + 1))[:n]
    return out


def measure(a, engine_args: list[str]) -> None:
    """One candidate (``ATOM_SPEC_TREE`` already set): tok/s at each batch size, one JSON line."""
    from transformers import AutoTokenizer

    from atom import SamplingParams
    from atom.model_engine.arg_utils import EngineArgs
    from atom.utils.arg_parser import FlexibleArgumentParser

    sizes = [int(b) for b in a.batch_sizes.split(",")]
    nodes = parse_dyn(os.environ["ATOM_SPEC_TREE"])
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    argv =["--model", a.model, "-tp", str(a.tp), "--max-num-seqs", str(max(sizes)),
            "--cudagraph-capture-sizes", "[" + ",".join(map(str, sizes)) + "]", "--method", "eagle3", "--draft-model", a.draft_model,
            "--num-speculative-tokens", str(nodes - 1), "--no-enable_prefix_caching"] + engine_args  # fmt: skip
    ea = parser.parse_args(argv)
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = EngineArgs.from_cli_args(ea).create_engine()
    waves = {b: -(-a.min_requests // b) for b in sizes}
    prompts = load_prompts(tok, a.questions, max(w * b for b, w in waves.items()))
    params = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, ignore_eos=False)
    # every batch size once first: the draft round's graphs are captured on first use
    for b in sizes:
        llm.generate(
            prompts[:b], SamplingParams(temperature=0.0, max_tokens=24, ignore_eos=True)
        )
    result = {}
    for b in sizes:
        tokens, wall = 0, 0.0
        for w in range(waves[b]):
            t0 = time.time()
            outs = llm.generate(prompts[w * b : (w + 1) * b], params)
            wall += time.time() - t0
            tokens += sum(o["num_tokens_output"] for o in outs)
        result[b] = tokens / wall
    llm.close()
    print("TUNE_RESULT " + json.dumps(result), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft-model", required=True)
    ap.add_argument(
        "--questions",
        required=True,
        help="jsonl of questions (SPEED-Bench question_multiturn.jsonl)",
    )
    ap.add_argument("--batch-sizes", default="1,2,4,8,12,16,24,32")
    ap.add_argument(
        "--candidates",
        default="dyn:8:4:16,dyn:6:4:12,dyn:5:4:10,dyn:5:4:8,dyn:4:2:6,dyn:3:2:4",
    )
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument(
        "--min-requests",
        type=int,
        default=8,
        help="requests a batch size decodes at least (in waves of b; small batches need more waves)",
    )
    ap.add_argument(
        "--margin",
        type=float,
        default=0.03,
        help="a bigger batch keeps the previous size's tree unless another is this much faster",
    )
    ap.add_argument("--out", default="tree_tune.json")
    ap.add_argument("--measure", action="store_true", help=argparse.SUPPRESS)
    argv = sys.argv[1:]
    engine_args = []
    if "--" in argv:
        i = argv.index("--")
        argv, engine_args = argv[:i], argv[i + 1 :]
    a = ap.parse_args(argv)
    if a.measure:
        measure(a, engine_args)
        return
    table = {}
    for spec in a.candidates.split(","):
        parse_dyn(spec)
        env = dict(os.environ, ATOM_SPEC_TREE=spec)
        cmd = [sys.executable, __file__, "--measure"] + sys.argv[1:]
        if "--" not in cmd:
            cmd.append("--")
        t0 = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("TUNE_RESULT ")]
        if proc.returncode != 0 or not line:
            print(
                f"{spec}: failed (exit {proc.returncode}); last output:\n{proc.stderr[-2000:]}",
                file=sys.stderr,
            )
            continue
        table[spec] = {
            int(b): v for b, v in json.loads(line[0][len("TUNE_RESULT ") :]).items()
        }
        print(f"{spec}: " + " ".join(f"b{b} {v:.0f}" for b, v in table[spec].items()) + f"  ({time.time() - t0:.0f} s)",
              flush=True)  # fmt: skip
    if not table:
        sys.exit("no candidate ran")
    sizes = sorted({b for row in table.values() for b in row})
    # each batch size's fastest tree, but a batch keeps the previous size's tree
    # unless another beats it by --margin (the measurement's noise is a few %)
    best, last = {}, None
    for b in sizes:
        top = max(table, key=lambda s: table[s].get(b, 0.0))
        if last is not None and table[top].get(b, 0.0) < (1 + a.margin) * table[
            last
        ].get(b, 0.0):
            top = last
        best[b] = last = top
    # one entry each time the tree changes, from that batch size on
    policy, last = [], None
    for b in sizes:
        if best[b] != last:
            policy.append(f"{b}={best[b]}")
            last = best[b]
    policy[0] = "1=" + policy[0].split("=", 1)[1]
    spec = ",".join(policy)
    nodes = max(parse_dyn(s) for s in best.values())
    print(f"\nATOM_SPEC_TREE={spec}  (--num-speculative-tokens {nodes - 1})")
    with open(a.out, "w") as f:
        json.dump(
            {"policy": spec, "num_speculative_tokens": nodes - 1, "tok_s": table},
            f,
            indent=1,
        )


if __name__ == "__main__":
    main()
