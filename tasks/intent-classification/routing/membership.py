"""E11: did the base model memorise the benchmark during pretraining?

Membership inference by completion, the plainest form: give the untouched base model (no
adapter, no chat template) the first 60% of a test query's words and let it continue
greedily. A model that has seen the dataset reproduces the exact next words far more often
than one that has not. Reported: the share of rows whose next four words are reproduced
exactly, and the share whose continuation shares at least half its words with the truth,
with examples. There is no clean control for a public dataset, so the number is read
against the other benchmark and against plain expectation: formulaic queries can be
guessed a little, verbatim long tails cannot.

Usage: uv run python membership.py --bench banking77 --rows 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx_lm import generate, load

MODEL_ID = "mlx-community/Qwen3.5-4B-OptiQ-4bit"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="banking77")
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    import random

    rows = [json.loads(line)["input"] for line in Path(f"data/{args.bench}/test.jsonl").read_text().splitlines() if line.strip()]
    random.Random(args.seed).shuffle(rows)
    rows = [r for r in rows if len(r.split()) >= 8][: args.rows]
    model, tokenizer = load(MODEL_ID)
    exact4 = half = 0
    examples = []
    for text in rows:
        words = text.split()
        cut = max(3, int(len(words) * 0.6))
        prompt, truth = " ".join(words[:cut]), words[cut:]
        out = generate(model, tokenizer, prompt=prompt, max_tokens=16, verbose=False)
        got = out.strip().split()
        if got[:4] == truth[:4] and len(truth) >= 4:
            exact4 += 1
            examples.append((prompt, " ".join(truth), " ".join(got[:8])))
        overlap = len(set(got[: len(truth)]) & set(truth)) / max(1, len(set(truth)))
        half += overlap >= 0.5
    lines = [f"# E11 membership by completion, {args.bench}, base model, {len(rows)} test rows of 8+ words", "",
             f"Prompt = first 60% of the words, greedy 16 tokens, no adapter, no chat template.", "",
             f"| next four words reproduced exactly | continuation shares half its words with the truth |", "|---|---|",
             f"| {exact4} ({exact4 / len(rows):.1%}) | {half} ({half / len(rows):.1%}) |", ""]
    if examples:
        lines += ["Exact reproductions:", ""] + [f"- `{p}` | truth `{t}` | model `{g}`" for p, t, g in examples[:10]]
    Path(f"runs/{args.bench}/E11_membership.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
