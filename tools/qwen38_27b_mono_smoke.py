#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Qwen3.8-27B megakernel alone, over stand-ins for ATOM's pools (conv /
recurrent state slots, a paged KV cache): the weights, the binding, the
kernel and every state kernel of a mono step, without the engine.

Greedy generation:

    python tools/qwen38_27b_mono_smoke.py /models/Qwen/Qwen3.8-27B/1d4bf0f2 --prompt "1+2+3=?"

Teacher-forced check of sequences generated elsewhere (``--teacher-force``,
the JSON of ``qwen38_27b_mono_generate.py --out``): each prompt + completion
runs through the kernel in 8-row launches from a fresh state, and every
completion token is compared with the kernel's argmax there, with the logit
gap ``max - logit[token]``. Two identical passes give the kernel's own
run-to-run noise (it is not bit-reproducible: atomic split-K + MXFP8), the
yardstick a mismatch is judged by.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from atom.models.qwen38_27b.mono import dims as d
from atom.models.qwen38_27b.mono import weights as W
from atom.models.qwen38_27b.mono.engine import MonoEngine, Pools, StepInputs


class Harness:
    def __init__(self, model: str, variant: str, max_len: int) -> None:
        dev = self.dev = torch.device("cuda")
        t0 = time.time()
        weights = W.convert_model(W.safetensors_source(model, dev))
        torch.cuda.synchronize()
        print("weights", round(time.time() - t0, 1), "s", flush=True)
        self.variant = variant
        self.eng = MonoEngine(weights, max_len, (variant,), dev)
        self.rows = rows = self.eng.binding(variant).rows
        nd = len(d.DELTA_LAYERS)
        # one request with its 8 state columns, ATOM's layouts
        conv = torch.zeros(
            nd, 1, 3 + rows - 1, d.CONV_DIM, dtype=torch.bfloat16, device=dev
        )
        state = torch.zeros(
            nd, rows, d.V_HEADS, d.GDN_DIM, d.GDN_DIM, dtype=torch.float32, device=dev
        )
        state = state.transpose(-1, -2)
        bs = 16
        blocks = (max_len + rows + bs - 1) // bs
        self.kc = [
            torch.zeros(
                blocks,
                d.KV_HEADS,
                d.HEAD_DIM // 8,
                bs,
                8,
                dtype=torch.bfloat16,
                device=dev,
            )
            for _ in d.ATTENTION_LAYERS
        ]
        self.vc = [
            torch.zeros(
                blocks,
                d.KV_HEADS,
                bs // 8,
                d.HEAD_DIM,
                8,
                dtype=torch.bfloat16,
                device=dev,
            )
            for _ in d.ATTENTION_LAYERS
        ]
        self.pools = Pools(
            conv=conv, state=state,
            k_ptrs=torch.tensor([t.data_ptr() for t in self.kc], dtype=torch.int64, device=dev),
            v_ptrs=torch.tensor([t.data_ptr() for t in self.vc], dtype=torch.int64, device=dev),
            block_size=bs,
        )  # fmt: skip
        # a shuffled block table, as ATOM's allocator hands out
        g = torch.Generator(device="cpu").manual_seed(0)
        self.table = torch.randperm(blocks, generator=g).to(torch.int32).to(dev)
        self.bs = bs
        self.fresh()

    def slots(self, base: int, rows: int) -> torch.Tensor:
        pos = torch.arange(base, base + rows, device=self.dev)
        return (self.table[pos // self.bs].long() * self.bs + pos % self.bs).to(
            torch.int32
        )

    def i32(self, v):
        return torch.tensor([v], dtype=torch.int32, device=self.dev)

    def fresh(self) -> None:
        self.pools.conv.zero_()
        self.pools.state.zero_()
        for t in self.kc + self.vc:
            t.zero_()
        self.eng.invalidate_mirror()
        self.eng.pending.zero_()
        self.accepted_prev = 1

    def launch(self, tokens: list[int], base: int) -> torch.Tensor:
        """One launch over ``tokens`` (<= rows) at ``base``, all accepted;
        returns their logits (fp32 copy)."""
        rows, real = self.rows, len(tokens)
        padded = tokens + [tokens[-1]] * (rows - real)
        inp = StepInputs(
            token_ids=torch.tensor(padded, dtype=torch.int64, device=self.dev),
            base=self.i32(base), real_rows=self.i32(real),
            state_slot=self.i32(self.accepted_prev - 1), conv_in=self.i32(0),
            window=self.i32(self.accepted_prev), conv_out=self.i32(0),
            block_table=self.table,
            kv_slots=self.slots(base, rows),
            request=self.i32(0),
        )  # fmt: skip
        b = self.eng.step(self.variant, self.pools, inp)
        logits = b.logits[:real].float().clone()
        k = real
        self.eng.commit(self.pools, self.i32(k), self.i32(k - 1))
        self.accepted_prev = k
        if int(b.failed):
            raise RuntimeError("kernel failure word set")
        return logits

    def prefill(self, ids: list[int], keep_from: int = -1) -> list[torch.Tensor]:
        """All of ``ids`` in row-wide launches; the logits of positions >= keep_from."""
        out = []
        for c in range(0, len(ids), self.rows):
            chunk = ids[c : c + self.rows]
            lg = self.launch(chunk, c)
            if keep_from >= 0 and c + len(chunk) > keep_from:
                out.append((c, lg))
        return out


def teacher_force(h: Harness, rows: list[dict], reps: int, dump: str = "") -> dict:
    report, dumped = [], []
    for r in rows:
        prompt, gen = r["prompt_ids"], r["token_ids"]
        seq = prompt + gen
        L = len(prompt)
        runs = []
        for _ in range(reps):
            h.fresh()
            chunks = h.prefill(seq[:-1], keep_from=L - 1)
            lg = torch.cat([x for _, x in chunks])
            first = chunks[0][0]
            runs.append(lg[L - 1 - first : L - 1 - first + len(gen)])
        nxt = torch.tensor(gen, device=h.dev)
        lg0 = runs[0]
        gap = lg0.max(1).values - lg0.gather(1, nxt[:, None]).squeeze(1)
        mism = (gap > 0).nonzero().flatten()
        e = {
            "prompt": r.get("prompt"),
            "checked": len(gen),
            "mismatches": int(mism.numel()),
            "max_gap": float(gap.max()),
        }
        if dump:
            dumped.append(
                {
                    "gap": gap.tolist(),
                    "argmax": lg0.argmax(1).tolist(),
                    "token_ids": gen,
                }
            )
        if reps > 1:
            lg1 = runs[1]
            noise = (lg0 - lg1).abs()
            a1 = lg1.argmax(1)
            self_gap = lg0.max(1).values - lg0.gather(1, a1[:, None]).squeeze(1)
            e.update(
                self_argmax_flips=int((lg0.argmax(1) != a1).sum()),
                self_gap_max=float(self_gap.max()),
                noise_max_abs=float(noise.max()),
                gaps_at_mismatch=[
                    [round(float(gap[j]), 3), round(float(noise[j].max()), 3)]
                    for j in mism.tolist()[:12]
                ],
            )
        print(json.dumps(e), flush=True)
        report.append(e)
    if dump:
        with open(dump, "w") as f:
            json.dump(dumped, f)
    return {
        "prompts": len(report),
        "checked": sum(e["checked"] for e in report),
        "mismatches": sum(e["mismatches"] for e in report),
        "self_argmax_flips": sum(e.get("self_argmax_flips", 0) for e in report),
        "max_gap": max(e["max_gap"] for e in report),
        "self_gap_max": max(e.get("self_gap_max", 0) for e in report),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt", default="1+2+3=?")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--variant", default="m8")
    ap.add_argument("--max-len", type=int, default=40000)
    ap.add_argument("--raw", action="store_true", help="no chat template")
    ap.add_argument("--teacher-force", default="", help="generation JSON to check")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument(
        "--dump", default="", help="per-token gaps of --teacher-force (JSON)"
    )
    a = ap.parse_args()
    h = Harness(a.model, a.variant, a.max_len)
    if a.teacher_force:
        rows = json.loads(Path(a.teacher_force).read_text())["rows"]
        print("TF SUMMARY", json.dumps(teacher_force(h, rows, a.reps, a.dump)))
        return
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    if a.raw:
        ids = tok(a.prompt)["input_ids"]
    else:
        text = tok.apply_chat_template(
            [{"role": "user", "content": a.prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        ids = tok(text, add_special_tokens=False)["input_ids"]
    torch.cuda.synchronize()
    t0 = time.time()
    last = h.prefill(ids, keep_from=len(ids) - 1)[-1][1][-1]
    torch.cuda.synchronize()
    t_prefill = time.time() - t0
    out = [int(last.argmax())]
    t0 = time.time()
    for _ in range(a.tokens - 1):
        lg = h.launch([out[-1]], len(ids) + len(out) - 1)
        out.append(int(lg[0].argmax()))
    torch.cuda.synchronize()
    t_dec = time.time() - t0
    print(
        json.dumps(
            {
                "prompt_tokens": len(ids),
                "prefill_s": round(t_prefill, 3),
                "decode_ms_per_token": round(t_dec * 1000 / max(1, a.tokens - 1), 3),
            }
        )
    )
    print("COMPLETION:", repr(tok.decode(out)))


if __name__ == "__main__":
    main()
