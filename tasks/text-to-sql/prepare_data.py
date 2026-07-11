"""Sample SynSQL-2.5M into train.jsonl.

SynSQL ships as one giant data.json (rows grouped by database; no schema
inline) plus tables.json (db_id -> DDL list). Hugging Face datasets streaming
cannot parse the single huge JSON array (pyarrow int32 block-size overflow),
so this script downloads both files once via hf_hub_download (cached) and
streams them with ijson. Rows are reservoir-sampled with seed 42:
deterministic for a fixed --rows and dataset revision, and unbiased despite
the database-grouped file order.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import ijson  # type: ignore[import-not-found,import-untyped]
from huggingface_hub import hf_hub_download

DATASET = "seeklhy/SynSQL-2.5M"
SEED = 42


def load_schema_map(tables_path: Path) -> dict[str, str]:
    with tables_path.open("rb") as handle:
        return {
            str(table["db_id"]): "\n\n".join(str(ddl) for ddl in table["ddls"])
            for table in ijson.items(handle, "item")
        }


def reservoir_sample(
    data_path: Path, schema_map: dict[str, str], rows: int, rng: random.Random
) -> tuple[list[dict[str, Any]], int]:
    sample: list[dict[str, Any]] = []
    seen = 0
    with data_path.open("rb") as handle:
        for record in ijson.items(handle, "item"):
            question = str(record.get("question") or "").strip()
            sql = str(record.get("sql") or "").strip()
            schema = schema_map.get(str(record.get("db_id") or ""))
            if not (question and sql and schema):
                continue
            row = {
                "schema": schema,
                "question": question,
                "evidence": str(record.get("external_knowledge") or "").strip(),
                "sql": sql,
                "source": "synsql-2.5m",
            }
            if len(sample) < rows:
                sample.append(row)
            else:
                slot = rng.randint(0, seen)
                if slot < rows:
                    sample[slot] = row
            seen += 1
            if seen % 250_000 == 0:
                print(f"scanned {seen} rows...")
    return sample, seen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--output", type=Path, default=Path("train.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows < 1:
        raise SystemExit("--rows must be at least 1")

    tables_path = Path(hf_hub_download(DATASET, "tables.json", repo_type="dataset"))
    data_path = Path(hf_hub_download(DATASET, "data.json", repo_type="dataset"))
    print(f"tables.json: {tables_path} ({tables_path.stat().st_size / 1e6:.0f} MB)")
    print(f"data.json: {data_path} ({data_path.stat().st_size / 1e9:.1f} GB)")

    schema_map = load_schema_map(tables_path)
    print(f"schemas: {len(schema_map)} databases")

    rng = random.Random(SEED)
    sample, seen = reservoir_sample(data_path, schema_map, args.rows, rng)
    if len(sample) < args.rows:
        raise RuntimeError(f"requested {args.rows} usable rows, found {len(sample)} of {seen} scanned")
    rng.shuffle(sample)

    preview = [
        {field: len(row[field]) for field in ("schema", "question", "sql")}
        for row in sample[:3]
    ]
    with args.output.open("w", encoding="utf-8") as output:
        for row in sample:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(sample)} rows to {args.output} (scanned {seen})")
    print(f"3-row length preview: {preview}")


if __name__ == "__main__":
    main()
