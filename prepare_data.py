"""Fetch the public JSON-extraction golden dataset for the Phase 0 spike.

Pulls NousResearch/json-mode-eval (100 rows: schema + text -> JSON) via the
HF datasets-server REST API and writes golden.csv with input/output columns.
No HF account or token needed.
"""

import csv
import json
import urllib.request

URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=NousResearch%2Fjson-mode-eval&config=default&split=train&offset=0&length=100"
)


def main() -> None:
    with urllib.request.urlopen(URL) as resp:
        data = json.load(resp)

    rows = []
    for item in data["rows"]:
        row = item["row"]
        # prompt = [system message with schema, user message with the text]
        text = "\n\n".join(m["content"] for m in row["prompt"])
        rows.append({"input": text, "output": row["completion"]})

    with open("golden.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "output"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote golden.csv with {len(rows)} rows")


if __name__ == "__main__":
    main()
