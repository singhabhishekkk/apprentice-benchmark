"""Generated-answer accuracy of the fine-tune served by llama.cpp, on a rows file.

One chat completion per row (temperature 0, thinking off, a few tokens), the JSON answer
parsed, the intent compared with the gold label by exact match, as score_rows.py does on
MLX. Writes <stem>.jsonl with answer, correct and format_valid per row and prints the totals.

    uv run python score_llama.py --rows data/banking77/test.jsonl --stem runs/latency/llama_lora_test_scores
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from latency_bench import post

ROOT = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--max-tokens", type=int, default=24)
    args = ap.parse_args()
    rows = [
        json.loads(line)
        for line in (ROOT / args.rows).read_text().splitlines()
        if line.strip()
    ]
    out, correct, valid = [], 0, 0
    t0 = time.perf_counter()
    for i, r in enumerate(rows, 1):
        gold = json.loads(r["output"])["intent"]
        body = post(
            f"{args.url}/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": r["input"]}],
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        text = body["choices"][0]["message"]["content"]
        try:
            answer = json.loads(text)["intent"]
            ok_format = 1
        except (ValueError, KeyError, TypeError):
            answer, ok_format = None, 0
        ok = int(answer == gold)
        correct += ok
        valid += ok_format
        out.append({"answer": answer, "correct": ok, "format_valid": ok_format})
        if i % 200 == 0 or i == len(rows):
            print(
                f"{i}/{len(rows)}: {time.perf_counter() - t0:.0f}s, accuracy so far {correct / i:.4f}",
                flush=True,
            )
    with open(str(ROOT / args.stem) + ".jsonl", "w") as f:
        f.writelines(json.dumps(o) + "\n" for o in out)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "accuracy": correct / len(rows),
                "format_valid": valid / len(rows),
            }
        )
    )


if __name__ == "__main__":
    main()
