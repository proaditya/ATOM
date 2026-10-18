# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Accuracy evals against an OpenAI-compatible server, greedy.

Each prompt is sent already formatted to a running server's
``/v1/completions``, ``--concurrency`` at a time, with fixed prompts, output
caps, stop strings and scoring:

- ``humaneval``: the function prompt completed, pass@1 by running each
  problem's tests in a subprocess (164 problems, up to 768 tokens);
- ``gpqa``: GPQA-Diamond, chat template, choices shuffled per question with a
  fixed seed (198 questions, OpenAI simple-evals' copy);
- ``mmlupro``: MMLU-Pro, the first ``--per-category`` questions of each
  category, chat template (20 x 14 = 280 questions).

GSM8K is not here: Qwen3.8-Flash-Next ends a raw 5-shot completion prompt at
once, so it runs through lm_eval with the chat template instead
(``repro.sh gsm8k``).

The multiple-choice tasks think at ``--reasoning-effort`` (default low), up to
``--mc-max-tokens`` (6,144), and are scored on the last "Answer: X" after the
thinking. The data loads from Hugging Face (openai/openai_humaneval,
TIGER-Lab/MMLU-Pro) and the simple-evals URL; files named as theirs in
``--data`` (``humaneval.parquet``, ``gpqa_diamond.csv``,
``mmlu_pro_test.parquet``) are used instead when present.

    python -m atom.benchmarks.qwen38_flash_mono.accuracy_eval --base-url http://127.0.0.1:8000 \
        --model /model --task gpqa --concurrency 4 --out gpqa.json
"""

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LETTERS = "ABCDEFGHIJ"
GPQA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"


def _frame(a, name: str, repo: str):
    """``--data``/``name`` if present, else the HF dataset's test split."""
    import pandas as pd

    path = Path(a.data, name)
    if path.exists():
        return pd.read_parquet(path)
    from datasets import load_dataset

    return load_dataset(repo, split="test").to_pandas()


# ---- tasks: (prompts, scorer, sampling defaults) ---------------------------------------


def _strip_special(text: str) -> str:
    """The completion without the end-of-text / end-of-turn tokens."""
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
    df = _frame(a, "humaneval.parquet", "openai/openai_humaneval")
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
        # server's stop strings, applied here too)
        cut = min(
            [i for i in (text.find(s_) for s_ in stop) if i >= 0], default=len(text)
        )
        program = f"{item['prompt']}{text[:cut]}\n\n{item['test']}\n\ncheck({item['entry']})\n"
        return _run_tests(program, 15.0), None

    return items, score, {"max_tokens": 768, "stop": stop}


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

    path = Path(a.data, "gpqa_diamond.csv")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(GPQA_URL, path)
    with open(path) as fh:
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
    df = _frame(a, "mmlu_pro_test.parquet", "TIGER-Lab/MMLU-Pro")
    items = []
    for cat, g in df.groupby("category", sort=True):
        for _, r in g.head(a.per_category).iterrows():
            opts = [str(o) for o in r["options"]]
            items.append({"prompt": _chat(tok, _mc_prompt(r["question"], opts), a.reasoning_effort),
                              "want": r["answer"], "n": len(opts), "category": cat})  # fmt: skip
    return items, _mc_score, {"max_tokens": a.mc_max_tokens}


TASKS = {"humaneval": _humaneval, "gpqa": _gpqa, "mmlupro": _mmlupro}


# ---- client ----------------------------------------------------------------------------


def _complete(a, prompt: str, sp: dict) -> dict:
    body = {"model": a.model, "prompt": prompt, "temperature": 0.0, **sp}
    req = urllib.request.Request(
        f"{a.base_url.rstrip('/')}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=a.timeout) as r:
        d = json.load(r)
    choice = d["choices"][0]
    return {
        "text": choice.get("text") or "",
        "finish_reason": choice.get("finish_reason"),
        "tokens": (d.get("usage") or {}).get("completion_tokens"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True, help="served name; also the tokenizer")
    p.add_argument("--tokenizer", default="", help="default: --model")
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument(
        "--data",
        default=os.path.expanduser("~/.cache/qwen38_flash_mono_evals"),
        help="local data files, and where the GPQA CSV is cached",
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument(
        "--reasoning-effort",
        default="low",
        help="xhigh / medium / low / none (no thinking)",
    )
    p.add_argument("--mc-max-tokens", type=int, default=6144)
    p.add_argument("--per-category", type=int, default=20)
    p.add_argument(
        "--max-tokens", type=int, default=0, help="override the task's output budget"
    )
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--out", default="")
    p.add_argument(
        "--rescore", default="", help="score an --out JSON's texts again, no server"
    )
    a = p.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model, trust_remote_code=True)
    items, score, sp = TASKS[a.task](a, tok)
    if a.max_tokens:
        sp["max_tokens"] = a.max_tokens
    if a.limit:
        items = items[: a.limit]
    if a.rescore:
        old = json.loads(Path(a.rescore).read_text())
        outputs = [{"text": r["text"], "tokens": r["tokens"]} for r in old["rows"]]
        elapsed = old["summary"]["seconds"]
    else:
        t0 = time.time()
        with ThreadPoolExecutor(a.concurrency) as pool:
            outputs = list(pool.map(lambda it: _complete(a, it["prompt"], sp), items))
        elapsed = time.time() - t0
    # score after every reply is in: HumanEval runs subprocesses
    rows, correct, tokens, truncated = [], 0, 0, 0
    for it, o in zip(items, outputs):
        text = o["text"]
        ok, got = score(it, text)
        correct += ok
        n_out = o.get("tokens") or len(tok.encode(text, add_special_tokens=False))
        tokens += n_out
        truncated += n_out >= sp["max_tokens"]
        rows.append(
            {
                "want": it.get("want"),
                "got": got,
                "ok": bool(ok),
                "tokens": n_out,
                "finish_reason": o.get("finish_reason"),
                "category": it.get("category"),
                "task_id": it.get("task_id"),
                "text": text,
            }
        )
    n = len(items)
    acc = correct / n
    summary = {
        "task": a.task, "n": n, "correct": correct, "accuracy": round(acc, 4), "stderr": round(math.sqrt(acc * (1 - acc) / n), 4),
        "truncated": truncated, "max_tokens": sp["max_tokens"], "output_tokens": tokens, "seconds": round(elapsed, 1),
        "concurrency": a.concurrency, "reasoning_effort": a.reasoning_effort if a.task in ("gpqa", "mmlupro") else None,
    }  # fmt: skip
    print("SUMMARY", json.dumps(summary), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": summary, "rows": rows}, f)


if __name__ == "__main__":
    main()
