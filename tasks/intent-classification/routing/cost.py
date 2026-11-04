"""E7 cost: measured tokens per request beside published frontier prices. No latency claim.

Prompt tokens use the same chat template as score_rows.py (one user turn, thinking off),
tokenizer only, no weights. Completion tokens are the greedy answer's `n_tokens`.

Usage: uv run python cost.py --scores runs/banking77/test_scores.jsonl \
           --model mlx-community/Qwen3.5-4B-OptiQ-4bit --acceptance 0.80
       uv run python cost.py --self-check
"""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

OPENROUTER_POST = "https://openrouter.ai/blog/insights/jev-vs-claude-opus-5-classification/"
DECISIONS_DOCS = "https://developers.openai.com/api/docs/guides/decisions"


def load_tokenizer(model_id: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, local_files_only=True)


def prompt_tokens(tok: Any, text: str) -> int:
    s = tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return len(tok.encode(s, add_special_tokens=False))


def report(scores: Path, model_id: str, acceptance: float) -> str:
    with open(scores) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    tok = load_tokenizer(model_id)
    prompt = np.array([prompt_tokens(tok, r["input"]) for r in rows])
    completion = np.array([int(r["n_tokens"]) for r in rows])
    decisions_per_1k = 0.10 * prompt.mean() * 1000 / 1e6
    frontier = [
        ("Opus 5 (with caching)", 2.42, "$2.42 per 1k requests", OPENROUTER_POST),
        ("Jev 1.13", 0.11, "$0.11 per 1k requests", OPENROUTER_POST),
        (
            "OpenAI Decisions gpt-6-luna",
            decisions_per_1k,
            f"$0.10 per 1M uncached input tokens x our mean {prompt.mean():.1f} prompt tokens",
            DECISIONS_DOCS,
        ),
    ]
    lines = [
        f"Rows {len(rows)}, tokenizer {model_id}.",
        (
            f"Prompt tokens: mean {prompt.mean():.1f}, p95 {np.percentile(prompt, 95):.0f}. "
            f"Completion tokens: mean {completion.mean():.1f}."
        ),
        "",
        "| model | $ per 1k requests | basis | source |",
        "|---|---|---|---|",
        *(f"| {name} | {usd:.4f} | {basis} | {url} |" for name, usd, basis, url in frontier),
        f"| Small model on a Mac ({model_id}) | not metered | electricity not counted | measured here |",
        "",
        "The Decisions row uses our tokenizer's count; OpenAI's own tokenizer and request wrapper will differ.",
        "",
        f"Cascade cost (Conformal Cascade, Theorem 4): c_small + c_frontier x (1 - acceptance), acceptance {acceptance}.",
        "Frontier dollars avoided per 1k requests (estimate; assumes the small model's cost is zero):",
        "",
        "| frontier | always-frontier $/1k | cascade frontier $/1k | avoided $/1k |",
        "|---|---|---|---|",
        *(
            f"| {name} | {usd:.4f} | {usd * (1 - acceptance):.4f} | {usd * acceptance:.4f} |"
            for name, usd, _, _ in frontier
        ),
    ]
    return "\n".join(lines)


class _StubTokenizer:
    def apply_chat_template(self, messages: list[dict], **_: Any) -> str:
        return f"<user>{messages[0]['content']}</user><assistant>"

    def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(s.split())))


def _self_check() -> None:
    global load_tokenizer
    real, load_tokenizer = load_tokenizer, lambda _: _StubTokenizer()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "scores.jsonl")
            path.write_text(
                "".join(json.dumps({"input": "word " * (i + 1), "n_tokens": 10 + i}) + "\n" for i in range(5))
            )
            text = report(path, "stub", 0.80)
    finally:
        load_tokenizer = real
    print(text)
    assert "| Opus 5 (with caching) | 2.4200 | 0.4840 | 1.9360 |" in text
    assert "mean 12.0." in text and "latency" not in text.lower()
    print("cost self-check passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path)
    ap.add_argument("--model", default="mlx-community/Qwen3.5-4B-OptiQ-4bit")
    ap.add_argument("--acceptance", type=float, default=0.80)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        return _self_check()
    if not args.scores:
        ap.error("--scores is required")
    print(report(args.scores, args.model, args.acceptance))


if __name__ == "__main__":
    main()
