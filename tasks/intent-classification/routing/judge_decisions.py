"""E4: a third-party decision model as the keep-or-defer judge, measured against gold.

Asks OpenAI's Decisions API (POST /v1/decisions, model gpt-6-luna; docs read 2026-11-04 at
developers.openai.com/api/docs/guides/decisions) one predicate per row: "the proposed label
is the correct intent for this message", with the full label list in the instructions. The
returned probability is the judge's confidence; run_signals.py then treats it like any other
signal, so the comparison with our own probes is on the same rows and the same protocol.

Cost: about 200 input tokens a row at $0.10 per 1M uncached tokens; 3,080 rows is well
under one dollar. Output tokens are not charged. Requires OPENAI_API_KEY in the
environment; the key is read, never printed. Rows are sent one message at a time.

Usage: uv run python judge_decisions.py --scores runs/banking77/test_scores.jsonl \
    --labels data/banking77/labels.json --out runs/banking77/judge_conf.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

URL = "https://api.openai.com/v1/decisions"
MODEL = "gpt-6-luna"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY is not set; nothing was sent.")
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in args.scores.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    instructions = (
        "A customer support message and a proposed intent label. The valid labels are: "
        + ", ".join(labels)
        + ". Is the proposed label the correct intent for the message?"
    )
    done = 0
    if args.out.exists():
        done = sum(1 for l in args.out.read_text(encoding="utf-8").splitlines() if l.strip())
    client = httpx.Client(timeout=30, headers={"Authorization": f"Bearer {key}"})
    t0 = time.time()
    with args.out.open("a", encoding="utf-8") as f:
        for i, row in enumerate(rows[done:], start=done):
            answer = row.get("answer") or "(no valid label)"
            body = {
                "model": MODEL,
                "input": f"Message: {row['input']}\nProposed label: {answer}",
                "questions": [{"name": "correct", "type": "predicate", "instructions": instructions}],
            }
            for attempt in range(4):
                try:
                    r = client.post(URL, json=body)
                    if r.status_code == 429 or r.status_code >= 500:
                        raise httpx.HTTPStatusError("retry", request=r.request, response=r)
                    r.raise_for_status()
                    break
                except (httpx.HTTPError, httpx.TransportError):
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            data = r.json()
            ans = next(a for a in data["answers"] if a["name"] == "correct")
            prob = ans.get("probability") if ans.get("type") == "predicate" else None
            f.write(json.dumps({"judge": prob, "judge_type": ans.get("type"),
                                "usage": data.get("usage")}) + "\n")
            f.flush()
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
    print(json.dumps({"rows": len(rows), "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
