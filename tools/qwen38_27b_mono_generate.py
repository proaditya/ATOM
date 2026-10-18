#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Batch-1 greedy generation through the ATOM engine, one prompt at a time,
for Qwen3.8-27B on the megakernel (``ATOM_MONO_ENABLE=1``, the default) or
the per-op model (``ATOM_MONO_ENABLE=0``). Prints one JSON line a prompt:
tokens, text, TTFT and TPOT, and a summary.

    python tools/qwen38_27b_mono_generate.py --model /models/Qwen/Qwen3.8-27B/1d4bf0f2 \
        --kv_cache_dtype bf16 --max-model-len 40000 --ctx 1024 --prompts 4 --max-tokens 256

``--ctx N`` builds prompts of about N tokens from a text corpus
(``--text-glob``, default the Python standard library): a random window,
then "Explain in detail what the code above does.", chat-wrapped.
"""

import glob
import json
import random
import statistics
import time
from pathlib import Path

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from atom.utils.arg_parser import FlexibleArgumentParser


def make_prompts(tok, a) -> list[list[int]]:
    if a.prompt_ids:
        rows = json.loads(Path(a.prompt_ids).read_text())["rows"]
        return [r["prompt_ids"] for r in rows][: a.prompts]
    if a.prompt:
        text = tok.apply_chat_template(
            [{"role": "user", "content": a.prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        return [tok(text, add_special_tokens=False)["input_ids"]] * a.prompts
    rng = random.Random(a.seed)
    files = sorted(glob.glob(a.text_glob))
    corpus = tok("\n\n".join(Path(f).read_text(errors="replace") for f in files))[
        "input_ids"
    ]
    out = []
    for _ in range(a.prompts):
        start = rng.randrange(0, len(corpus) - a.ctx)
        chat = [
            {
                "role": "user",
                "content": tok.decode(corpus[start : start + a.ctx - 64])
                + "\n\nExplain in detail what the code above does.",
            }
        ]
        text = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        out.append(tok(text, add_special_tokens=False)["input_ids"])
    return out


def main() -> None:
    parser = FlexibleArgumentParser(description=__doc__)
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--prompt", default="")
    parser.add_argument(
        "--stop-at-eos",
        action="store_true",
        help="stop at EOS / stop ids (tau counted up to it)",
    )
    parser.add_argument(
        "--prompt-ids",
        default="",
        help="reuse the prompts of an --out JSON (its rows' prompt_ids)",
    )
    parser.add_argument("--ctx", type=int, default=1024)
    parser.add_argument("--prompts", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--text-glob", default="/usr/lib/python3.10/*.py")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0, help="0: greedy")
    parser.add_argument(
        "--together",
        action="store_true",
        help="one generate call over all prompts (batched until the shorter ones finish)",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--profile-prompt",
        type=int,
        default=-1,
        help="torch-profile this prompt's generation (with --together, any value >= 0 profiles "
        "the one call; needs --torch-profiler-dir)",
    )
    a = parser.parse_args()
    if a.cudagraph_capture_sizes == EngineArgs.cudagraph_capture_sizes:
        a.cudagraph_capture_sizes = "[1]"  # one sequence at a time unless asked
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    llm = EngineArgs.from_cli_args(a).create_engine()
    params = SamplingParams(
        temperature=a.temperature, max_tokens=a.max_tokens, ignore_eos=not a.stop_at_eos
    )
    prompts = make_prompts(tok, a)
    for p in prompts[: a.warmup]:
        llm.generate(
            [p], SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
        )
    spec = a.method is not None

    def mtp():
        if not spec:
            return None
        st = llm.get_mtp_statistics()
        return (
            st["total_draft_tokens"],
            st["total_accepted_tokens"],
            sum(st["distribution"].values()),
        )

    if a.together:
        before = mtp()
        if a.profile_prompt >= 0:
            llm.start_profile()
        t0 = time.time()
        outs = llm.generate(prompts, params)
        wall = time.time() - t0
        if a.profile_prompt >= 0:
            llm.stop_profile()
        after = mtp()
        tokens = sum(o["num_tokens_output"] for o in outs)
        summary = {
            "prompts": len(prompts),
            "output_tokens": tokens,
            "wall_s": wall,
            "tok_s": tokens / wall,
        }
        if spec:
            steps = after[2] - before[2]
            summary["tau"] = 1 + (after[1] - before[1]) / steps if steps else None
        print("TOGETHER", json.dumps(summary), flush=True)
        llm.close()
        return
    rows = []
    for i, p in enumerate(prompts):
        before = mtp()
        if i == a.profile_prompt:
            llm.start_profile()
        out = llm.generate([p], params)[0]
        if i == a.profile_prompt:
            llm.stop_profile()
        after = mtp()
        row = {
            "prompt": i, "prompt_tokens": len(p), "output_tokens": out["num_tokens_output"],
            "ttft": out["ttft"], "tpot": out["tpot"],
            "decode_tok_s": 1.0 / out["tpot"] if out["tpot"] else None,
            "prompt_ids": list(p), "token_ids": out["token_ids"], "text": out["text"][:400],
        }  # fmt: skip
        if spec:
            steps = after[2] - before[2]
            row["tau"] = 1 + (after[1] - before[1]) / steps if steps else None
            row["steps"] = steps
        rows.append(row)
        print(
            json.dumps(
                {k: v for k, v in row.items() if k not in ("token_ids", "prompt_ids")}
            ),
            flush=True,
        )
    summary = {
        "prompts": len(rows),
        "mean_prompt_tokens": statistics.mean(r["prompt_tokens"] for r in rows),
        "mean_decode_tok_s": statistics.mean(
            r["decode_tok_s"] for r in rows if r["decode_tok_s"]
        ),
        "mean_ttft": statistics.mean(r["ttft"] for r in rows),
    }
    if spec:
        summary["mean_tau"] = statistics.mean(r["tau"] for r in rows if r["tau"])
    print("SUMMARY", json.dumps(summary), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": summary, "rows": rows}, f)
    llm.close()


if __name__ == "__main__":
    main()
