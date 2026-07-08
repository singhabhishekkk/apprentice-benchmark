"""Prepare the public contract-clause-extraction golden dataset.

Pulls the official CUAD v1 release (Contract Understanding Atticus Dataset,
510 real commercial contracts sourced from SEC EDGAR, expert-annotated by
lawyers, CC-BY-4.0), samples 200 usable rows with seed 42, and writes
golden.csv with input/output columns. No HF token needed -- downloads the
canonical data.zip directly from the Atticus Project's GitHub release
(the HF dataset viewer for theatticusproject/cuad-qa requires running its
loader script, so this goes straight to the source it downloads from).
"""

from __future__ import annotations

import csv
import io
import json
import random
import urllib.request
import zipfile

SEED = 42
ROW_COUNT = 200
SOURCE_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"

# Five categories out of CUAD's 41: two near-universal identity fields
# (Document Name, Parties -- present in 509/510 contracts) plus three
# substantive review fields lawyers actually look for (Agreement Date,
# Governing Law, Anti-Assignment). Governing Law and Anti-Assignment answers
# sit anywhere in a contract (median offset 32,252 and unclustered), not just
# the preamble, which is why each field gets its own excerpt window below
# rather than one truncated prefix of the contract.
CATEGORIES = ["Document Name", "Parties", "Agreement Date", "Governing Law", "Anti-Assignment"]
FIELD_KEYS = {
    "Document Name": "document_name",
    "Parties": "parties",
    "Agreement Date": "agreement_date",
    "Governing Law": "governing_law",
    "Anti-Assignment": "anti_assignment",
}
WINDOW = 350  # chars of context on each side of an answer span


def extract_fields(paragraph: dict) -> dict[str, tuple[str, int]] | None:
    """One paragraph == one contract's full text + all 41 category QAs.

    Returns {category: (answer_text, answer_start)} for present categories only,
    or None if the near-universal identity fields are both missing (509/510
    contracts have both; skip the one outlier).
    """
    found: dict[str, tuple[str, int]] = {}
    for qa in paragraph["qas"]:
        for cat in CATEGORIES:
            if f'"{cat}"' in qa["question"] and qa["answers"]:
                ans = qa["answers"][0]
                found[cat] = (ans["text"].strip(), ans["answer_start"])
    if "Document Name" not in found or "Parties" not in found:
        return None
    return found


def build_row(context: str, found: dict[str, tuple[str, int]]) -> dict[str, str]:
    """Input: excerpt windows around each present category's real answer span,
    assembled in original document order (so it reads like one coherent
    excerpt, not five disjoint quotes) -- never the raw category label, so the
    model has to recognize each field's content the way it would in real text.
    Output: the fixed 5-key JSON, "" for any category this contract lacks.
    """
    spans = sorted(found.items(), key=lambda kv: kv[1][1])
    windows = []
    for _cat, (text, start) in spans:
        lo = max(0, start - WINDOW)
        hi = min(len(context), start + len(text) + WINDOW)
        windows.append(context[lo:hi].strip())
    excerpt = "\n\n".join(windows)

    output = {FIELD_KEYS[cat]: found[cat][0] if cat in found else "" for cat in CATEGORIES}
    input_text = (
        "Extract contract fields as a JSON object with keys document_name, parties, "
        "agreement_date, governing_law, anti_assignment. Use \"\" for any field not "
        "present in the excerpt.\n\n"
        "Contract excerpt:\n" + excerpt
    )
    return {
        "input": input_text,
        "output": json.dumps(output, ensure_ascii=False, sort_keys=True),
    }


def main() -> None:
    with urllib.request.urlopen(SOURCE_URL, timeout=120) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        with zf.open("CUADv1.json") as f:
            data = json.load(f)["data"]

    rows = []
    for entry in data:
        for paragraph in entry["paragraphs"]:
            found = extract_fields(paragraph)
            if found is None:
                continue
            rows.append(build_row(paragraph["context"], found))

    if len(rows) < ROW_COUNT:
        raise RuntimeError(f"expected at least {ROW_COUNT} rows, got {len(rows)}")

    # Same two-step sample-then-shuffle pattern as the other tasks in this repo:
    # prepare_data.py samples+orders the rows once; gepa_optimize.py and the
    # Colab notebooks each re-shuffle with the same seed, so the held-out split
    # is identical across every run and every model.
    rows = random.Random(SEED).sample(rows, ROW_COUNT)

    with open("golden.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "output"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote golden.csv: {len(rows)} rows")


if __name__ == "__main__":
    main()
