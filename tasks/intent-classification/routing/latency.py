"""E7: single-request latency on this Mac, one request at a time, warm model.

The plan forbids quoting batch timing as production latency, so this measures what a
customer's process would see: one request, no batching, after a warm-up. Three paths:
  head:       prefill only, then one matmul on the last hidden state (the classifier tier)
  generate:   prefill plus greedy decoding of the answer (the generative path)
Reports p50 and p95 in milliseconds over N requests, with the conditions stated.

Usage: uv run python latency.py --adapter runs/banking77/adapter --rows data/banking77/test.jsonl --n 50
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler

from score_rows import MODEL_ID, last_hidden


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=48)
    args = ap.parse_args()
    rows = [json.loads(l) for l in args.rows.read_text(encoding="utf-8").splitlines() if l.strip()][: args.n + 5]
    model, tokenizer = load(MODEL_ID, adapter_path=str(args.adapter))
    tok = cast(Any, tokenizer)
    head = mx.random.normal((2560, 77))  # a stand-in head; the matmul cost is what is measured

    def prompt(text: str) -> list[int]:
        s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
        return tok.encode(s, add_special_tokens=False)

    prompts = [prompt(r["input"]) for r in rows]
    for p in prompts[:5]:  # warm-up, not timed
        last_hidden(model, p)
        generate(model, tok, p, max_tokens=args.max_tokens, sampler=make_sampler(0.0))
    head_ms, gen_ms, gen_tokens = [], [], []
    for p in prompts[5:]:
        t0 = time.perf_counter()
        h = last_hidden(model, p)
        logits = mx.array(h) @ head
        mx.eval(logits)
        head_ms.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        out = generate(model, tok, p, max_tokens=args.max_tokens, sampler=make_sampler(0.0))
        gen_ms.append((time.perf_counter() - t0) * 1000)
        gen_tokens.append(len(tok.encode(out)))
    def q(a: list[float]) -> tuple[float, float]:
        return float(np.percentile(a, 50)), float(np.percentile(a, 95))
    print(json.dumps({
        "conditions": f"{platform.machine()} Mac, one request at a time, warm, no concurrency, "
                      f"{len(head_ms)} requests, adapter {args.adapter.name}, 4-bit",
        "prompt_tokens_mean": float(np.mean([len(p) for p in prompts[5:]])),
        "head_prefill_plus_matmul_ms": {"p50": q(head_ms)[0], "p95": q(head_ms)[1]},
        "generate_ms": {"p50": q(gen_ms)[0], "p95": q(gen_ms)[1]},
        "generated_tokens_mean": float(np.mean(gen_tokens)),
    }, indent=1))


if __name__ == "__main__":
    main()
