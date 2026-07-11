"""Stream SynSQL-2.5M, seed-shuffle it, then take first N usable rows.

Sampling uses Hugging Face IterableDataset.shuffle(seed=42), whose bounded
shuffle buffer makes identical seed, row count, and dataset revision
reproducible without downloading the 2.5M-row corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]

DATASET = "seeklhy/SynSQL-2.5M"
FIELDS = ("schema", "question", "sql")
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--output", type=Path, default=Path("train.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows < 1:
        raise SystemExit("--rows must be at least 1")

    stream = load_dataset(DATASET, split="train", streaming=True)
    columns = set(stream.column_names or [])
    missing = set(FIELDS) - columns
    if missing:
        raise RuntimeError(
            f"{DATASET} missing expected columns {sorted(missing)}; found {sorted(columns)}"
        )

    written = 0
    preview: list[dict[str, int]] = []
    with args.output.open("w", encoding="utf-8") as output:
        # ponytail: fixed-size shuffle buffer avoids full download; increase only if broader mixing matters.
        for source in stream.shuffle(seed=SEED, buffer_size=10_000):
            values = {field: source[field] for field in FIELDS}
            if not all(isinstance(value, str) and value.strip() for value in values.values()):
                continue
            row: dict[str, Any] = {**values, "source": "synsql-2.5m"}
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            if len(preview) < 3:
                preview.append({field: len(values[field]) for field in FIELDS})
            written += 1
            if written == args.rows:
                break

    if written < args.rows:
        raise RuntimeError(f"requested {args.rows} usable rows, found {written}")
    print(f"wrote {written} rows to {args.output}")
    print(f"3-row length preview: {preview}")


if __name__ == "__main__":
    main()
