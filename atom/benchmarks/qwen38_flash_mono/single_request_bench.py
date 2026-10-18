# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
r"""Single-request (batch 1) decode speed against any OpenAI-compatible server:
ATOM, vLLM or SGLang, with or without speculative decoding.

One request at a time (``--concurrency N``: N at a time), streamed. For every
request it records the time to first token, the decode time after it and the
output tokens; with speculation, the verify rounds and the acceptance length
(accepted drafts per round + 1, the verified token included), read from the
server's own counters. From those it reports the in-engine time per round, and
the speed any assumed acceptance length would give at that round time
(``--project-acceptance``).

Two workloads:

* a benchmark dataset (``--dataset``): the prompts, formatting and selection
  of the DFlash benchmark (z-lab/dflash: the HF test split shuffled with seed
  42, the first N; chat, first turn only), greedy, stop at end of turn;
* a synthetic prompt (``--input-len`` / ``--output-len``): Inferact's recipe,
  ``"hi"`` for one input token (else N copies of ``" word"``), EOS ignored.

Speculative counters, picked by ``--stats`` (default: whichever the server has):
ATOM ``/debug/mtp_stats``, vLLM ``/metrics`` (``vllm:spec_decode_*``). SGLang
reports none here: its per-request ``meta_info`` is refused with streaming.

Export the prompts once (where the network is), then pass the file::

    python -m atom.benchmarks.qwen38_flash_mono.single_request_bench --dataset gsm8k \
        --num-prompts 128 --export-prompts gsm8k_128.jsonl
    python -m atom.benchmarks.qwen38_flash_mono.single_request_bench \
        --base-url http://127.0.0.1:8000 --prompts-file gsm8k_128.jsonl \
        --chat-template-kwargs '{"reasoning_effort": "xhigh"}' --project-acceptance 6 \
        --out result.json
    python -m atom.benchmarks.qwen38_flash_mono.single_request_bench \
        --base-url http://127.0.0.1:8000 --input-len 1 --output-len 1024 --num-prompts 12 \
        --project-acceptance 6

``scripts/qwen38_flash_mono/repro.sh speed`` runs it once per set. Beyond the
timing (none of this changes what is timed or how the speeds are computed):

* each request's reply text and ``finish_reason`` (kept in ``--out``); the
  summary counts the replies that reached ``--max-tokens`` (``at_cap``);
* GSM8K: each reply's last ``\boxed{}`` (or else its last number) against the
  reference answer (``correct``), a sanity check only, since a reply cut at
  the cap often has no answer yet. ``--export-prompts`` writes the reference
  next to each GSM8K prompt, and ``--prompts-file`` reads it back;
* the median of the per-request speeds next to their mean;
* ``--concurrency``: N requests in flight. The speculative counters are then
  read around the whole run, so the acceptance length and the time per round
  are the run's (rounds counted per request), not each request's; the
  end-to-end figure is the aggregate throughput.
"""

import argparse
import json
import random
import re
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# The DFlash benchmark's datasets and prompt formats (z-lab/dflash, benchmark.py).
DATASETS = {
    "gsm8k": (
        ("openai/gsm8k", "main"),
        "test",
        lambda x: x["question"]
        + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
    ),
    "math500": (
        ("HuggingFaceH4/MATH-500",),
        "test",
        lambda x: x["problem"]
        + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
    ),
    "humaneval": (
        ("openai/openai_humaneval",),
        "test",
        lambda x: "Write a solution to the following problem and make sure that it "
        "passes the tests:\n```python\n" + x["prompt"] + "\n```",
    ),
    "mbpp": (
        ("google-research-datasets/mbpp", "sanitized"),
        "test",
        lambda x: x["prompt"],
    ),
    # first turn only, as the DFlash server benchmark sends it
    "mt-bench": (
        ("HuggingFaceH4/mt_bench_prompts",),
        "train",
        lambda x: x["prompt"][0],
    ),
}

# The reference answer of a row, for the sets whose replies are checked.
ANSWERS = {
    "gsm8k": lambda x: x["answer"].split("####")[-1].strip().replace(",", ""),
}


def _dataset_rows(name: str, count: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    args, split, _ = DATASETS[name]
    rows = load_dataset(*args, split=split)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return [rows[i] for i in order[: min(count, len(order))]]


def dataset_prompts(name: str, count: int, seed: int = 42) -> list[str]:
    """The first ``count`` prompts of the split shuffled with ``seed``: the
    DFlash benchmark's selection, so the two report the same prompts."""
    return [DATASETS[name][2](row) for row in _dataset_rows(name, count, seed)]


def dataset_answers(name: str, count: int, seed: int = 42) -> list[str | None]:
    """The reference answers of ``dataset_prompts``'s prompts, in the same
    order; None for a set without a check."""
    if name not in ANSWERS:
        return [None] * count
    return [ANSWERS[name](row) for row in _dataset_rows(name, count, seed)]


def gsm8k_correct(text: str, reference: str) -> bool:
    """The reply's last ``\\boxed{}``, or else its last number, equals the
    reference."""
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    cand = boxed[-1] if boxed else (re.findall(r"-?\d[\d,]*\.?\d*", text) or [""])[-1]
    cand = cand.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(cand) == float(reference)
    except ValueError:
        return False


def _get(url: str, timeout: float = 10.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode()
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return None


def _prom_sum(text: str, name: str) -> float:
    """A Prometheus counter summed over its label sets."""
    total = 0.0
    for line in text.splitlines():
        m = re.match(rf"^{re.escape(name)}(?:{{[^}}]*}})?\s+(\S+)", line)
        if m:
            total += float(m.group(1))
    return total


class SpecCounters:
    """Cumulative (verify rounds, accepted draft tokens) from the server, read
    before and after each request; None when the server has no speculation."""

    def __init__(self, base: str, kind: str):
        self.base, self.kind = base, kind
        if kind == "auto":
            self.kind = self._detect()

    def _detect(self) -> str:
        atom = _get(self.base + "/debug/mtp_stats")
        if atom is not None:
            try:
                if json.loads(atom).get("enabled"):
                    return "atom"
            except json.JSONDecodeError:
                pass
        metrics = _get(self.base + "/metrics") or ""
        if "vllm:spec_decode_num_drafts" in metrics:
            return "vllm"
        return "sglang"  # meta_info on the response, if the server sends it

    def read(self) -> tuple[float, float] | None:
        if self.kind == "atom":
            s = json.loads(_get(self.base + "/debug/mtp_stats") or "{}")
            if not s.get("enabled"):
                return None
            return sum(s.get("distribution", {}).values()), s["total_accepted_tokens"]
        if self.kind == "vllm":
            m = _get(self.base + "/metrics") or ""
            return (
                _prom_sum(m, "vllm:spec_decode_num_drafts_total"),
                _prom_sum(m, "vllm:spec_decode_num_accepted_tokens_total"),
            )
        return None


def stream_one(base, model, prompt, a) -> dict:
    """One streamed request: (ttft, decode seconds, output tokens, meta_info,
    reply text, finish reason)."""
    if a.chat:
        url = base + "/v1/chat/completions"
        body = {"messages": [{"role": "user", "content": prompt}]}
        if a.chat_template_kwargs:
            body["chat_template_kwargs"] = a.chat_template_kwargs
    else:
        url = base + "/v1/completions"
        body = {"prompt": prompt}
    body.update(
        model=model,
        max_tokens=a.max_tokens,
        temperature=a.temperature,
        top_p=a.top_p,
        stream=True,
        stream_options={"include_usage": True},
        ignore_eos=a.ignore_eos,
    )
    if a.top_k > 0:
        body["top_k"] = a.top_k
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    t_first = t_last = None
    usage, meta, chunks = {}, {}, 0
    text, finish = [], None
    try:
        r = urllib.request.urlopen(req, timeout=a.timeout)
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"{url}: HTTP {e.code}: {e.read().decode(errors='replace')[:2000]}"
        ) from e
    with r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            usage = ev.get("usage") or usage
            meta = ev.get("meta_info") or meta
            for ch in ev.get("choices", []):
                meta = ch.get("meta_info") or meta
                d = ch.get("delta", ch)
                if any(
                    d.get(k)
                    for k in ("content", "reasoning_content", "reasoning", "text")
                ):
                    now = time.perf_counter()
                    t_first = t_first or now
                    t_last = now
                    chunks += 1
                    text.extend(
                        d.get(k) or ""
                        for k in ("reasoning_content", "reasoning", "content", "text")
                    )
                finish = ch.get("finish_reason") or finish
    end = time.perf_counter()
    return {
        "ttft": (t_first or end) - t0,
        "decode_s": (t_last or end) - (t_first or end),
        "latency_s": end - t0,
        "prompt_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "chunks": chunks,
        "meta": meta,
        "finish_reason": finish,
        "text": "".join(text),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="", help="default: the server's first model")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--dataset", choices=sorted(DATASETS))
    src.add_argument(
        "--prompts-file", help='JSONL of {"prompt": ...} (--export-prompts)'
    )
    src.add_argument("--input-len", type=int, help="synthetic prompt of about N tokens")
    p.add_argument(
        "--export-prompts", help="write the dataset's prompts to this JSONL and exit"
    )
    p.add_argument("--num-prompts", type=int, default=128)
    p.add_argument(
        "--output-len", type=int, default=1024, help="synthetic: tokens generated"
    )
    p.add_argument(
        "--max-tokens", type=int, default=2048, help="dataset: generation cap"
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0, help="0: not sent")
    p.add_argument(
        "--chat-template-kwargs",
        type=json.loads,
        default=None,
        help='e.g. \'{"reasoning_effort": "xhigh"}\'',
    )
    p.add_argument("--warmup", type=int, default=2, help="short requests before timing")
    p.add_argument(
        "--concurrency", type=int, default=1, help="requests in flight (default 1)"
    )
    p.add_argument(
        "--stats", default="auto", choices=["auto", "atom", "vllm", "sglang", "none"]
    )
    p.add_argument(
        "--project-acceptance",
        type=float,
        nargs="*",
        default=[],
        help="acceptance lengths (verified token included) to project at the measured round time",
    )
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--out", help="write the per-request rows and the summary as JSON")
    a = p.parse_args()

    if a.export_prompts:
        assert a.dataset, "--export-prompts needs --dataset"
        with open(a.export_prompts, "w") as f:
            f.writelines(
                json.dumps(
                    {"prompt": text, "dataset": a.dataset}
                    | ({"answer": ref} if ref is not None else {})
                )
                + "\n"
                for text, ref in zip(
                    dataset_prompts(a.dataset, a.num_prompts),
                    dataset_answers(a.dataset, a.num_prompts),
                )
            )
        print(f"wrote {a.num_prompts} {a.dataset} prompts to {a.export_prompts}")
        return

    base = a.base_url.rstrip("/")
    model = a.model or json.loads(_get(base + "/v1/models"))["data"][0]["id"]
    if a.input_len is not None:
        a.chat, a.ignore_eos, a.max_tokens = False, True, a.output_len
        prompt = "hi" if a.input_len <= 1 else " ".join(["word"] * a.input_len)
        prompts, warm = [prompt] * a.num_prompts, [prompt] * a.warmup
        answers = [None] * a.num_prompts
        workload = f"synthetic: {a.input_len} in, {a.output_len} out, EOS ignored"
    else:
        a.chat, a.ignore_eos = True, False
        if a.prompts_file:
            with open(a.prompts_file) as f:
                rows = [json.loads(x) for x in f if x.strip()]
            texts = [r["prompt"] for r in rows]
            answers = [r.get("answer") for r in rows]
            workload = f"{a.prompts_file} ({rows[0].get('dataset', '?')})"
        else:
            assert a.dataset, "one of --dataset, --prompts-file, --input-len"
            texts = dataset_prompts(a.dataset, a.num_prompts + a.warmup)
            answers = dataset_answers(a.dataset, a.num_prompts)
            workload = a.dataset
        prompts, warm = (
            texts[: a.num_prompts],
            texts[a.num_prompts :][: a.warmup] or texts[: a.warmup],
        )

    counters = SpecCounters(base, a.stats) if a.stats != "none" else None
    warm_args = argparse.Namespace(**{**vars(a), "max_tokens": 64})
    pool = ThreadPoolExecutor(a.concurrency)
    # at least one warmup request per lane: every batch size runs once untimed
    warm = (warm * a.concurrency)[: max(len(warm), a.concurrency)]
    list(pool.map(lambda w: stream_one(base, model, w, warm_args), warm))

    def measured(prompt):
        """One request, with the counters around it when it runs alone."""
        alone = counters and a.concurrency == 1
        before = counters.read() if alone else None
        r = stream_one(base, model, prompt, a)
        after = counters.read() if alone else None
        return r, before, after

    rows = []
    run_before = counters.read() if counters else None
    t_start = time.perf_counter()
    for i, (r, before, after) in enumerate(pool.map(measured, prompts)):
        rounds = accepted = None
        if before is not None and after is not None:
            rounds, accepted = after[0] - before[0], after[1] - before[1]
        elif r["meta"].get("spec_verify_ct"):
            rounds = r["meta"]["spec_verify_ct"]
            # every round emits its accepted drafts + 1; the prefill emits the first token
            accepted = r["output_tokens"] - 1 - rounds if r["output_tokens"] else None
        n = r["output_tokens"] or r["chunks"]
        r.update(
            prompt_index=i,
            rounds=rounds,
            acceptance=(
                (1 + accepted / rounds) if rounds and accepted is not None else None
            ),
            # the decode window holds every round after the prefill's token
            ms_per_round=(1000 * r["decode_s"] / rounds) if rounds else None,
            decode_tok_s=((n - 1) / r["decode_s"]) if r["decode_s"] > 0 and n else None,
        )
        r.pop("meta")
        if i < len(answers) and answers[i] is not None:
            r["correct"] = gsm8k_correct(r["text"], answers[i])
        rows.append(r)
        acc = (
            f" acceptance {r['acceptance']:.2f}, {r['ms_per_round']:.2f} ms/round"
            if rounds
            else ""
        )
        print(
            f"[{i + 1}/{len(prompts)}] {n} tokens, {r['decode_tok_s'] or 0:,.0f} tok/s, ttft {r['ttft'] * 1000:.0f} ms{acc}"
        )
    wall = time.perf_counter() - t_start
    run_after = counters.read() if counters else None
    pool.shutdown()

    def mean(k):
        v = [r[k] for r in rows if r[k] is not None]
        return statistics.mean(v) if v else None

    out_tokens = sum((r["output_tokens"] or r["chunks"]) for r in rows)
    decode_tokens = sum((r["output_tokens"] or r["chunks"]) - 1 for r in rows)
    decode_s = sum(r["decode_s"] for r in rows)
    all_rounds = sum(r["rounds"] or 0 for r in rows)
    summary = {
        "workload": workload,
        "model": model,
        "prompts": len(rows),
        "concurrency": a.concurrency,
        "temperature": a.temperature,
        "top_p": a.top_p,
        "top_k": a.top_k,
        "mean_prompt_tokens": mean("prompt_tokens"),
        "mean_output_tokens": out_tokens / len(rows),
        # the mean of every request's decode speed (each request counts once)
        "decode_tok_s_per_request_mean": mean("decode_tok_s"),
        # every decoded token counts once: total decode tokens / total decode time
        "decode_tok_s_token_weighted": decode_tokens / decode_s if decode_s else None,
        # the DFlash benchmark's figure: output tokens / wall time, TTFT included
        "end_to_end_tok_s": out_tokens / wall,
        "mean_ttft_ms": 1000 * mean("ttft"),
        "decode_tok_s_per_request_median": statistics.median(
            [r["decode_tok_s"] for r in rows if r["decode_tok_s"] is not None] or [0]
        ),
        "at_cap": sum(r["finish_reason"] == "length" for r in rows),
    }
    checked = [r["correct"] for r in rows if "correct" in r]
    if checked:
        summary.update(correct=sum(checked), checked=len(checked))
    if all_rounds:
        summary.update(
            acceptance=1
            + sum((r["acceptance"] - 1) * r["rounds"] for r in rows if r["rounds"])
            / all_rounds,
            ms_per_round=1000
            * sum(r["decode_s"] for r in rows if r["rounds"])
            / all_rounds,
        )
    elif run_before is not None and run_after is not None:
        # several requests in flight: the run's counters, rounds counted per request
        all_rounds = run_after[0] - run_before[0]
        if all_rounds:
            summary.update(
                acceptance=1 + (run_after[1] - run_before[1]) / all_rounds,
                ms_per_round=1000 * decode_s / all_rounds,
            )
    if all_rounds:
        summary["projected_tok_s"] = {
            f"{al:g}": 1000 * al / summary["ms_per_round"]
            for al in a.project_acceptance
        }
    print(json.dumps(summary, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(
                {
                    "summary": summary,
                    "rows": rows,
                    "args": {k: v for k, v in vars(a).items()},
                },
                f,
                indent=1,
            )


if __name__ == "__main__":
    main()
