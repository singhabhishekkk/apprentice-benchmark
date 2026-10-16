"""Re-score committed answers.jsonl files with checks.py. No API key, no network.

    python verify.py results/receipt-extraction/gpt-4o-mini
    python verify.py results            # every answers.jsonl under a directory

Each row stores the gold answer and the model's raw output, so the score is recomputed from
scratch. Exits non-zero if any recomputed score differs from the committed one, or from summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import checks


def verify(answers_path: Path) -> bool:
    summary_path = answers_path.with_name("summary.json")
    summary = json.loads(summary_path.read_text())
    kind = summary["kind"]
    rows = [json.loads(line) for line in answers_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ok = True
    for r in rows:
        again = checks.score_output(r["expected"], r["raw_output"], kind)
        if again["score"] != r["score"] or again["fields"] != r["field_scores"]:
            ok = False
            print(f"  MISMATCH row {r['row_id']}: committed {r['score']}, recomputed {again['score']}")
    mean = round(100 * sum(checks.score_output(r["expected"], r["raw_output"], kind)["score"] for r in rows) / len(rows), 2)
    if mean != summary["mean_score"]:
        ok = False
    print(f"{'OK  ' if ok else 'FAIL'} {answers_path.parent}: {len(rows)} rows, recomputed {mean}, summary {summary['mean_score']}")
    return ok


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    files = [target] if target.name == "answers.jsonl" else sorted(target.rglob("answers.jsonl"))
    if not files:
        raise SystemExit(f"no answers.jsonl under {target}")
    sys.exit(0 if all([verify(f) for f in files]) else 1)


if __name__ == "__main__":
    main()
