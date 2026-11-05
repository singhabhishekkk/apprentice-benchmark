"""Decision latency against a llama.cpp server: prefill plus the last-token state, no decoding.

Sends Banking77 test prompts one at a time (or with a fixed concurrency) and reports the
client-observed latency and, when the server returns it, its own prompt-processing time.
Two modes: `embed` asks /embeddings with pooling last (what a head reads); `chat` asks
/v1/chat/completions for one token (prefill plus one decode step, the floor for a
written answer). The chat template matches score_layers.py: one user message, thinking off.

    uv run python latency_bench.py --url http://localhost:8080 --mode embed --n 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent


API_KEY = ""  # set from --api-key; Magnitude's server wants its network key as a bearer


def post(url: str, payload: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def load_rows(path: Path, n: int) -> list[str]:
    rows = [json.loads(line)["input"] for line in path.read_text().splitlines()]
    return rows[:n]


# Qwen3.5 chat template for one user turn with thinking off, as the server's /apply-template
# renders it; the embeddings route takes raw text, so the client applies it.
TEMPLATE = (
    "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def one(url: str, mode: str, text: str, max_tokens: int = 1, model: str = "") -> dict:
    """embed: one pass, the pooled state. chat: prefill plus one token. generate: prefill plus
    max_tokens of decoding, the written-answer path for extraction tasks."""
    messages = [{"role": "user", "content": text}]
    t = time.perf_counter()
    if mode == "embed":
        body = post(f"{url}/v1/embeddings", {"input": TEMPLATE.format(text=text)})
    else:
        payload: dict = {
            "messages": messages,
            "max_tokens": 1 if mode == "chat" else max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if model:
            payload["model"] = model
        body = post(f"{url}/v1/chat/completions", payload)
    client_ms = (time.perf_counter() - t) * 1000
    timings = body.get("timings") or {}
    usage = body.get("usage") or {}
    return {
        "client_ms": client_ms,
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_ms": timings.get("prompt_ms"),
        "prompt_n": timings.get("prompt_n") or usage.get("prompt_tokens"),
    }


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, round(p * (len(s) - 1)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument(
        "--mode", choices=["embed", "chat", "generate"], default="embed"
    )
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument(
        "--model", default="", help="model name, for servers that need one"
    )
    parser.add_argument("--api-key", default="")
    parser.add_argument("--rows", default="data/banking77/test.jsonl")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    global API_KEY
    API_KEY = args.api_key

    texts = load_rows(ROOT / args.rows, args.n)
    one(args.url, args.mode, texts[0], args.max_tokens, args.model)  # warm
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(
            pool.map(
                lambda s: one(args.url, args.mode, s, args.max_tokens, args.model),
                texts,
            )
        )
    wall = time.perf_counter() - t0
    client = [r["client_ms"] for r in results]
    prompt = [r["prompt_ms"] for r in results if r["prompt_ms"] is not None]
    tokens = [r["prompt_n"] for r in results if r["prompt_n"] is not None]
    summary = {
        "label": args.label,
        "mode": args.mode,
        "rows": len(results),
        "concurrency": args.concurrency,
        "prompt_tokens_mean": statistics.mean(tokens) if tokens else None,
        "client_ms": {
            "p50": pct(client, 0.5),
            "p95": pct(client, 0.95),
            "mean": statistics.mean(client),
        },
        "server_prompt_ms": {"p50": pct(prompt, 0.5), "p95": pct(prompt, 0.95)}
        if prompt
        else None,
        "requests_per_second": len(results) / wall,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        (ROOT / args.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
