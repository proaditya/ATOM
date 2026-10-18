#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Accuracy evals of Qwen3.8-27B through the ATOM engine, offline, greedy.

Tasks (``--data`` holds their files):

- ``gsm8k``: 5-shot completion, flexible-extract (``gsm8k_{train,test}.parquet``);
- ``humaneval``: the function prompt completed, pass@1 by running each
  problem's tests in a subprocess (``humaneval.parquet``);
- ``gpqa``: GPQA-Diamond, chat template, choices shuffled per question with a
  fixed seed (``gpqa_diamond.csv``, OpenAI simple-evals' copy);
- ``mmlupro``: MMLU-Pro, the first ``--per-category`` questions of each
  category, chat template (``mmlu_pro_test.parquet``).

The multiple-choice tasks think at ``--reasoning-effort`` and are scored on
the last "Answer: X" after the thinking. ``--batch 1`` submits one prompt at
a time (the Qwen3.8-27B megakernel's single-sequence decode); a larger batch
hands every prompt to the engine at once (the per-op model).

    python tools/qwen38_27b_eval.py --model /models/Qwen/Qwen3.8-27B/1d4bf0f2 \
        --kv_cache_dtype bf16 --max-model-len 16384 --task gpqa --data /path/to/evals
"""

import json
import math
import multiprocessing
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from atom.utils.arg_parser import FlexibleArgumentParser

LETTERS = "ABCDEFGHIJ"


# ---- tasks: (prompts, scorer, sampling defaults) ---------------------------------------


def _gsm8k(a, tok):
    import pandas as pd

    sys.path.insert(0, os.path.dirname(__file__))
    from qwen38_27b_gsm8k import STOP, extract, target

    train = pd.read_parquet(f"{a.data}/gsm8k_train.parquet")
    test = pd.read_parquet(f"{a.data}/gsm8k_test.parquet")
    shots = "".join(
        f"Question: {q}\nAnswer: {ans.replace('####', 'The answer is')}\n\n"
        for q, ans in zip(train["question"][:5], train["answer"][:5])
    )
    items = [
        {"prompt": f"{shots}Question: {q}\nAnswer:", "want": target(ans)}
        for q, ans in zip(test["question"], test["answer"])
    ]

    def score(item, text):
        got = extract(text)
        ok = bool(
            got
            and re.fullmatch(r"-?[0-9.]+", got)
            and abs(float(got) - float(item["want"])) < 1e-6
        )
        return ok, got

    return items, score, {"max_tokens": 1024, "stop_strings": STOP}


def _strip_special(text: str) -> str:
    """The decoded completion without the end-of-text / end-of-turn tokens
    the engine leaves in it."""
    for t in ("<|endoftext|>", "<|im_end|>"):
        text = text.replace(t, "")
    return text


def _run_tests(program: str, timeout: float) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run(
            [sys.executable, "-I", path],
            capture_output=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
            check=False,
        )
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


def _humaneval(a, tok):
    import pandas as pd

    df = pd.read_parquet(f"{a.data}/humaneval.parquet")
    items = [
        {"prompt": p, "test": t, "entry": e, "task_id": i}
        for p, t, e, i in zip(
            df["prompt"], df["test"], df["entry_point"], df["task_id"]
        )
    ]

    stop = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\nassert "]

    def score(item, text):
        text = _strip_special(text)
        # the body: cut at the first top-level statement after it (the
        # engine's stop strings, applied here too)
        cut = min(
            [i for i in (text.find(s_) for s_ in stop) if i >= 0], default=len(text)
        )
        program = f"{item['prompt']}{text[:cut]}\n\n{item['test']}\n\ncheck({item['entry']})\n"
        return _run_tests(program, 15.0), None

    return items, score, {"max_tokens": 768, "stop_strings": stop}


def _chat(tok, user: str, effort: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
        enable_thinking=effort != "none", **({"reasoning_effort": effort} if effort != "none" else {}),
    )  # fmt: skip


def _mc_prompt(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices))
    return (
        f"{question}\n\n{opts}\n\nAnswer the multiple-choice question above. "
        f"The last line of your response must be exactly 'Answer: X', X one of {LETTERS[: len(choices)]}."
    )


def _mc_extract(text: str, n: int) -> str | None:
    body = text.split("</think>")[-1]
    found = re.findall(r"Answer\s*[:：]\s*\**\(?([A-J])\)?", body)
    if not found:
        found = re.findall(r"\b\(?([A-J])\)?\s*$", body.strip())
    got = found[-1] if found else None
    return got if got is not None and LETTERS.index(got) < n else None


def _mc_score(item, text):
    got = _mc_extract(text, item["n"])
    return got == item["want"], got


def _gpqa(a, tok):
    import csv

    with open(f"{a.data}/gpqa_diamond.csv") as fh:
        rows = list(csv.DictReader(fh))
    items = []
    for i, r in enumerate(rows):
        choices = [
            r["Correct Answer"],
            r["Incorrect Answer 1"],
            r["Incorrect Answer 2"],
            r["Incorrect Answer 3"],
        ]
        choices = [c.strip() for c in choices]
        order = list(range(4))
        random.Random(i).shuffle(order)
        shuffled = [choices[j] for j in order]
        items.append({"prompt": _chat(tok, _mc_prompt(r["Question"].strip(), shuffled), a.reasoning_effort),
                          "want": LETTERS[order.index(0)], "n": 4})  # fmt: skip
    return items, _mc_score, {"max_tokens": a.mc_max_tokens}


def _mmlupro(a, tok):
    import pandas as pd

    df = pd.read_parquet(f"{a.data}/mmlu_pro_test.parquet")
    items = []
    for cat, g in df.groupby("category", sort=True):
        for _, r in g.head(a.per_category).iterrows():
            opts = [str(o) for o in r["options"]]
            items.append({"prompt": _chat(tok, _mc_prompt(r["question"], opts), a.reasoning_effort),
                              "want": r["answer"], "n": len(opts), "category": cat})  # fmt: skip
    return items, _mc_score, {"max_tokens": a.mc_max_tokens}


TASKS = {"gsm8k": _gsm8k, "humaneval": _humaneval, "gpqa": _gpqa, "mmlupro": _mmlupro}


def main() -> None:
    parser = FlexibleArgumentParser(description=__doc__)
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        help="xhigh / medium / low / none (no thinking)",
    )
    parser.add_argument("--mc-max-tokens", type=int, default=6144)
    parser.add_argument("--per-category", type=int, default=20)
    parser.add_argument(
        "--max-tokens", type=int, default=0, help="override the task's output budget"
    )
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--rescore", default="", help="score an --out JSON's texts again, no engine"
    )
    a = parser.parse_args()
    a.cudagraph_capture_sizes = (
        "[1]" if a.batch == 1 else str(sorted({1, 2, 4, 8, 16, 32, 64, a.batch}))
    )
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    items, score, sp = TASKS[a.task](a, tok)
    if a.max_tokens:
        sp["max_tokens"] = a.max_tokens
    if a.limit:
        items = items[: a.limit]
    if a.rescore:
        old = json.loads(Path(a.rescore).read_text())
        outputs = [
            {"text": r["text"], "token_ids": [0] * r["tokens"]} for r in old["rows"]
        ]
        elapsed = old["summary"]["seconds"]
    else:
        llm = EngineArgs.from_cli_args(a).create_engine()
        params = SamplingParams(temperature=0.0, **sp)
        prompts = [it["prompt"] for it in items]
        t0 = time.time()
        if a.batch == 1:
            outputs = []
            for p in prompts:
                outputs += llm.generate([p], params)
        else:
            outputs = llm.generate(prompts, params)
        elapsed = time.time() - t0
        llm.close()
    texts = [o["text"] for o in outputs]
    # score after the engine is down: HumanEval runs subprocesses
    rows, correct, tokens, truncated = [], 0, 0, 0
    for it, o, text in zip(items, outputs, texts):
        ok, got = score(it, text)
        correct += ok
        n_out = len(o.get("token_ids") or ()) or len(tok.encode(text))
        tokens += n_out
        truncated += n_out >= sp["max_tokens"]
        rows.append(
            {
                "want": it.get("want"),
                "got": got,
                "ok": bool(ok),
                "tokens": n_out,
                "category": it.get("category"),
                "text": text,
            }
        )
    n = len(items)
    acc = correct / n
    summary = {
        "task": a.task, "n": n, "accuracy": round(acc, 4), "stderr": round(math.sqrt(acc * (1 - acc) / n), 4),
        "truncated": truncated, "max_tokens": sp["max_tokens"], "output_tokens": tokens, "seconds": round(elapsed, 1), "batch": a.batch,
        "reasoning_effort": a.reasoning_effort if a.task in ("gpqa", "mmlupro") else None,
    }  # fmt: skip
    print("SUMMARY", json.dumps(summary), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": summary, "rows": rows}, f)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
