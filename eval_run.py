"""Run a task's held-out rows against any OpenAI-compatible chat endpoint and write receipts.

    python eval_run.py --task receipt-extraction --model gpt-4o-mini
    python eval_run.py --task receipt-extraction --model qwen --base-url http://localhost:8080/v1 \\
        --price-in 0.05 --price-out 0.05 --reference results/receipt-extraction/gpt-4o-mini/summary.json

Needs tasks/<task>/golden.csv (run that task's prepare_data.py first). The held-out rows are the
last 30% after random.Random(42).shuffle, the same split as gepa_optimize.py and the notebooks.
Standard library only. API key: --api-key or OPENAI_API_KEY (not needed for local servers).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import checks

ROOT = Path(__file__).parent
STALE_PRICE_DAYS = 30


def heldout_rows(task: str, seed: int) -> list[dict]:
    path = ROOT / "tasks" / task / "golden.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run tasks/{task}/prepare_data.py first")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    random.Random(seed).shuffle(rows)
    return rows[int(len(rows) * 0.7):]


def chat(base_url: str, api_key: str, body: dict) -> dict:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            raise SystemExit(f"API error {e.code}: {e.read().decode()[:500]}")
    raise SystemExit("unreachable")


def prices(args) -> tuple[dict | None, str]:
    """Return ({'input','output'} per 1M tokens, provenance) or (None, reason)."""
    if args.price_in is not None and args.price_out is not None:
        return {"input": args.price_in, "output": args.price_out}, "assumption supplied via --price-in/--price-out"
    table = json.loads((ROOT / "pricing.json").read_text())
    if args.model in table["models"]:
        age = (date.today() - date.fromisoformat(table["checked_on"])).days
        if age > STALE_PRICE_DAYS:
            print(f"WARNING: pricing.json was checked {age} days ago ({table['checked_on']}); re-check {table['sources']}")
        return table["models"][args.model], f"pricing.json checked {table['checked_on']}"
    return None, "no price: model not in pricing.json and no --price-in/--price-out given"


def cost_usd(tokens_in: int, tokens_out: int, price: dict | None) -> float | None:
    if price is None:
        return None
    return (tokens_in * price["input"] + tokens_out * price["output"]) / 1_000_000


def compare(summary: dict, reference: dict) -> dict | None:
    """'X% match at Y% of the cost' against a reference (larger model) summary."""
    if not summary.get("cost_usd") or not reference.get("cost_usd") or not reference["mean_score"]:
        return None
    return {
        "reference_model": reference["model"],
        "match_pct": round(100 * summary["mean_score"] / reference["mean_score"], 1),
        "cost_pct": round(100 * summary["cost_usd"] / reference["cost_usd"], 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(checks.TASK_KIND))
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="https://api.openai.com/v1", help="any OpenAI-compatible /v1 base")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "none"))
    p.add_argument("--seed", type=int, default=42, help="split seed and request seed")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--omit-temperature", action="store_true", help="for models that reject temperature=0")
    p.add_argument("--price-in", type=float, help="USD per 1M input tokens (overrides pricing.json)")
    p.add_argument("--price-out", type=float, help="USD per 1M output tokens (overrides pricing.json)")
    p.add_argument("--reference", help="summary.json of a larger model, for the match-at-cost figure")
    p.add_argument("--out", default=str(ROOT / "results"))
    args = p.parse_args()

    kind = checks.TASK_KIND[args.task]
    price, price_note = prices(args)
    out_dir = Path(args.out) / args.task / args.model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    answers = []
    for i, row in enumerate(heldout_rows(args.task, args.seed)):
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": row["input"]}],
            "seed": args.seed,
            "max_tokens": args.max_tokens,
        }
        if not args.omit_temperature:
            body["temperature"] = 0
        start = time.perf_counter()
        resp = chat(args.base_url, args.api_key, body)
        latency = time.perf_counter() - start
        raw = resp["choices"][0]["message"].get("content") or ""
        parsed = checks.parse_output(raw, kind)
        result = checks.score_parsed(row["output"], parsed, kind)
        usage = resp.get("usage") or {}
        answers.append({
            "row_id": i,
            "expected": row["output"],
            "raw_output": raw,
            "parsed": parsed,
            "score": result["score"],
            "field_scores": result["fields"],
            "tokens_in": usage.get("prompt_tokens", 0),
            "tokens_out": usage.get("completion_tokens", 0),
            "latency_s": round(latency, 3),
        })
        print(f"row {i}: {result['score']:.2f}")

    with open(out_dir / "answers.jsonl", "w", encoding="utf-8") as f:
        for a in answers:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    tin, tout = sum(a["tokens_in"] for a in answers), sum(a["tokens_out"] for a in answers)
    summary = {
        "task": args.task,
        "kind": kind,
        "model": args.model,
        "base_url": args.base_url,
        "rows": len(answers),
        "mean_score": round(100 * sum(a["score"] for a in answers) / len(answers), 2),
        "temperature": None if args.omit_temperature else 0,
        "seed": args.seed,
        "tokens_in": tin,
        "tokens_out": tout,
        "price_per_1m_tokens": price,
        "price_basis": price_note,
        "cost_usd": cost_usd(tin, tout, price),
        "run_date": date.today().isoformat(),
    }
    if args.reference:
        summary["vs_reference"] = compare(summary, json.loads(Path(args.reference).read_text()))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
