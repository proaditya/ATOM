#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""GSM8K (5-shot, greedy, flexible-extract) through the ATOM engine, offline.

Questions go in batches of ``--batch`` (1: one at a time). Engine arguments
(speculation included) pass through; export the deployment's variables first.

    python tools/qwen38_27b_gsm8k.py --model Qwen/Qwen3.8-27B --kv_cache_dtype bf16 \
        --max-model-len 8192 --data /path/to/gsm8k --limit 250 \
        --method eagle3 --draft-model /path/to/EAGLE4-Qwen3.8-27B --num-speculative-tokens 15
"""

import json
import re
import time

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from atom.utils.arg_parser import FlexibleArgumentParser

STOP = ["Question:", "</s>", "<|im_end|>", "\n\n\n"]


def read_parquet(path):
    import pandas as pd

    df = pd.read_parquet(path)
    return [{"question": q, "answer": a} for q, a in zip(df["question"], df["answer"])]


def target(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def extract(text: str) -> str | None:
    """lm-eval's flexible-extract: the last number in the completion."""
    found = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    if not found:
        return None
    last = next(x for x in found[-1] if x)
    return last.strip(".").replace(",", "").replace("$", "")


def main() -> None:
    parser = FlexibleArgumentParser(description=__doc__)
    EngineArgs.add_cli_args(parser)
    parser.add_argument(
        "--data", required=True, help="dir with gsm8k_{train,test}.parquet"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--out", default="")
    a = parser.parse_args()
    a.cudagraph_capture_sizes = (
        "[1]" if a.batch == 1 else str(sorted({1, 2, 4, 8, 16, 32, 64, a.batch}))
    )
    train = read_parquet(f"{a.data}/gsm8k_train.parquet")
    test = read_parquet(f"{a.data}/gsm8k_test.parquet")
    if a.limit:
        test = test[: a.limit]
    shots = "".join(
        f"Question: {s['question']}\nAnswer: {s['answer'].replace('####', 'The answer is')}\n\n"
        for s in train[: a.shots]
    )
    prompts = [f"{shots}Question: {t['question']}\nAnswer:" for t in test]
    llm = EngineArgs.from_cli_args(a).create_engine()
    params = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, stop_strings=STOP)
    t0 = time.time()
    outputs = []
    for i in range(0, len(prompts), a.batch):
        outputs += llm.generate(prompts[i : i + a.batch], params)
    elapsed = time.time() - t0
    rows, correct = [], 0
    for t, o in zip(test, outputs):
        got, want = extract(o["text"]), target(t["answer"])
        ok = (
            got is not None and abs(float(got) - float(want)) < 1e-6
            if got and re.fullmatch(r"-?[0-9.]+", got)
            else False
        )
        correct += ok
        rows.append({"want": want, "got": got, "ok": ok, "text": o["text"]})
    summary = {
        "n": len(test),
        "accuracy": correct / len(test),
        "seconds": round(elapsed, 1),
        "batch": a.batch,
    }
    print("SUMMARY", json.dumps(summary), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": summary, "rows": rows}, f)
    llm.close()


if __name__ == "__main__":
    main()
