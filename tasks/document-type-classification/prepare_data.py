"""Prepare the public document-type-classification golden dataset.

Pulls anirudh1112/corrected-tobacco-dataset-with-ocr from the Hugging Face
datasets-server API. It is a corrected Tobacco3482 variant with OCR text and
10 real document-type labels, CC-BY-4.0. Samples 200 rows with seed 42 and
writes golden.csv with input/output columns. No HF token needed.
"""

from __future__ import annotations

import csv
import json
import random
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

SEED = 42
ROW_COUNT = 200
DATASET = "anirudh1112/corrected-tobacco-dataset-with-ocr"
SPLITS = {"train": 2186, "validation": 272, "test": 279}
PAGE_SIZE = 100
MAX_CONTENT_CHARS = 4000
CLASS_NAMES = ["ADVE", "Email", "Form", "Letter", "Memo", "News", "Note", "Report", "Resume", "Scientific"]
PROMPT_TEMPLATE = """I will provide you with the content and the title of a document.
Your task is to select the most appropriate document type for the document from the list of available document types I will provide.
Only select a document type from the provided list. Respond only with the selected document type name, without any additional information.
If none of the available document types fit the document, respond with an empty string.
The content is likely in English.

The data will be provided using an XML-like format for clarity:

<available_document_types>
{available_document_types}
</available_document_types>

<title>
</title>

<content>
{content}
</content>

Please select the single most appropriate English document type from the list above that best categorizes this document.
Be selective and only choose a document type if it clearly matches the document's nature (e.g., Invoice, Contract, Receipt, Letter, etc.).
"""


def fetch_split(split: str, total_rows: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for offset in range(0, total_rows, PAGE_SIZE):
        params = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": min(PAGE_SIZE, total_rows - offset),
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{params}"
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = json.load(resp)
        rows.extend(item["row"] for item in data["rows"])
    return rows


def build_input(text: str) -> str:
    # ponytail: char cap approximates paperless-gpt token truncation; switch to token count only if scores prove sensitive.
    content = text.strip()[:MAX_CONTENT_CHARS]
    return PROMPT_TEMPLATE.format(
        available_document_types=", ".join(CLASS_NAMES),
        content=content,
    )


def sample_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rng = random.Random(SEED)
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        label = row.get("label", "").strip()
        text = row.get("text", "").strip()
        if label in CLASS_NAMES and text:
            by_label[label].append(row)

    per_class = ROW_COUNT // len(CLASS_NAMES)
    sampled: list[dict[str, str]] = []
    for label in CLASS_NAMES:
        candidates = by_label[label]
        if len(candidates) < per_class:
            raise RuntimeError(f"expected at least {per_class} rows for {label}, got {len(candidates)}")
        sampled.extend(rng.sample(candidates, per_class))

    rng.shuffle(sampled)
    return [
        {
            "input": build_input(row["text"]),
            "output": row["label"].strip(),
        }
        for row in sampled
    ]


def main() -> None:
    source_rows = []
    for split, total_rows in SPLITS.items():
        source_rows.extend(fetch_split(split, total_rows))

    rows = sample_rows(source_rows)
    if len(rows) != ROW_COUNT:
        raise RuntimeError(f"expected {ROW_COUNT} rows, got {len(rows)}")

    with open("golden.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "output"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote golden.csv: {len(rows)} rows")
    print("class distribution:")
    for label, count in sorted(Counter(row["output"] for row in rows).items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
