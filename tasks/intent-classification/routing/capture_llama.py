"""Capture the last-token state from a llama.cpp server for every row, in classifier_tier's layout.

The native /embeddings route with embd_normalize -1 returns the raw pooled state (after the
final norm, which is where llama.cpp pools). Writes <stem>.layers.npz with one key,
L<layers>_prompt_last, and <stem>.layers.jsonl with a null generative answer per row, so
classifier_tier.py fits and scores the head on it unchanged.

    uv run python capture_llama.py --rows data/banking77/test.jsonl --stem runs/latency/llama32_test --layers 32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from latency_bench import TEMPLATE, post

ROOT = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--rows", required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--layers", type=int, required=True)
    args = parser.parse_args()
    texts = [
        json.loads(line)["input"]
        for line in (ROOT / args.rows).read_text().splitlines()
        if line.strip()
    ]
    states = []
    t0 = time.perf_counter()
    for i, text in enumerate(texts, 1):
        body = post(
            f"{args.url}/embeddings",
            {"content": TEMPLATE.format(text=text), "embd_normalize": -1},
        )
        vec = body[0]["embedding"][0]
        states.append(np.asarray(vec, dtype=np.float32))
        if i % 200 == 0 or i == len(texts):
            print(f"{i}/{len(texts)}: {time.perf_counter() - t0:.0f}s", flush=True)
    x = np.stack(states)
    assert x.shape == (len(texts), 2560), x.shape
    stem = ROOT / args.stem
    np.savez(str(stem) + ".layers.npz", **{f"L{args.layers}_prompt_last": x})
    with open(str(stem) + ".layers.jsonl", "w") as f:
        f.writelines(
            json.dumps({"answer": None, "correct": 0, "format_valid": 0}) + "\n"
            for _ in texts
        )
    print(f"wrote {stem}.layers.npz {x.shape}")


if __name__ == "__main__":
    main()
