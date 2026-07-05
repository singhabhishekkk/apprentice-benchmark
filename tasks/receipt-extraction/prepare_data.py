"""Prepare the public receipt-extraction golden dataset for spike2.

Pulls Voxel51/scanned_receipts raw annotations, samples 200 usable rows with
seed 42, and writes golden.csv with input/output columns. No HF token needed.
"""

from __future__ import annotations

import csv
import json
import random
import urllib.request

SEED = 42
ROW_COUNT = 200
SOURCE_URL = "https://huggingface.co/datasets/Voxel51/scanned_receipts/resolve/main/samples.json"


def input_text(sample: dict) -> str:
    detections = sample["text_detections"]["detections"]
    lines = [d["label"].strip() for d in detections if d.get("label", "").strip()]
    return (
        "Extract receipt fields as a JSON object with keys company, address, date, total.\n\n"
        "Receipt OCR text:\n"
        + "\n".join(lines)
    )


def output_json(sample: dict) -> str:
    return json.dumps(
        {
            "company": sample["company"],
            "address": sample["address"],
            "date": sample["date"],
            "total": sample["total"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def main() -> None:
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as resp:
        data = json.load(resp)

    rows = [
        {"input": input_text(sample), "output": output_json(sample)}
        for sample in data["samples"]
        if sample.get("text_detections", {}).get("detections")
        and all(sample.get(field) for field in ("company", "address", "date", "total"))
    ]
    if len(rows) < ROW_COUNT:
        raise RuntimeError(f"expected at least {ROW_COUNT} usable rows, got {len(rows)}")

    rows = random.Random(SEED).sample(rows, ROW_COUNT)

    with open("golden.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "output"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote golden.csv with {len(rows)} rows")


if __name__ == "__main__":
    main()
