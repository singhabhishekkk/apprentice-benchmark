"""CLINC150 -> Apprentice task CSV and the frozen spike splits.

Source: Hugging Face dataset `clinc/clinc_oos`, `plus` configuration, license CC BY 3.0. The
repo itself is parquet-backed (plus/{train,validation,test}-00000-of-00001.parquet), so no
script loader is involved.

The out-of-scope class (label name "oos") is dropped from every split: this is a closed-set
head experiment, so the task is the 150 in-scope intents only, matching Banking77's shape.

Writes, under --out:
  train.csv            20 rows per intent (3,000) for `apprentice train --local`; the product
                       holds out 30% of these itself for its own self-score.
  probe.jsonl          10 further rows per intent (1,500), never seen by the fine-tune; used to
                       train the hidden-state probe (idea 1 and 2 in the plan).
  test.jsonl           the full in-scope test split (4,500), untouched; later cut into
                       explore / calibrate / evaluate thirds by a fixed seed.
Output format for every row: {"intent": "<label name>"} so the SDK's JSON field scorer applies.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

PER_INTENT_TRAIN = 20
PER_INTENT_PROBE = 10
SEED = 42
OOS = "oos"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="untrusted download dir")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    local = snapshot_download(
        "clinc/clinc_oos", repo_type="dataset", local_dir=args.data, allow_patterns=["plus/*.parquet"]
    )
    files = sorted((Path(local) / "plus").glob("*.parquet"))
    train_files = [f for f in files if f.name.startswith("train")]
    test_files = [f for f in files if f.name.startswith("test")]
    assert train_files and test_files, [f.name for f in files]

    # The parquet schema carries the datasets Features JSON, including the ClassLabel names in order.
    meta = pq.read_schema(train_files[0]).metadata[b"huggingface"]
    all_names: list[str] = json.loads(meta)["info"]["features"]["intent"]["names"]
    assert OOS in all_names, "expected an oos label to drop"
    names = [n for n in all_names if n != OOS]
    assert len(names) == 150, f"expected 150 in-scope labels, got {len(names)}"

    def rows(paths: list[Path]) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for p in paths:
            for r in pq.read_table(p).to_pylist():
                label = all_names[int(r["intent"])]
                if label != OOS:
                    out.append({"text": r["text"], "label": label})
        return out

    train_rows = rows(train_files)
    test_rows = rows(test_files)

    rng = random.Random(SEED)
    by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in train_rows:
        by_label[str(r["label"])].append(r)
    assert sorted(by_label) == sorted(names),"train split does not cover all 150 intents"
    train_pick: list[dict[str, object]] = []
    probe_pick: list[dict[str, object]] = []
    for label in sorted(by_label):
        bucket = by_label[label]
        assert len(bucket) >= PER_INTENT_TRAIN + PER_INTENT_PROBE, (label, len(bucket))
        rng.shuffle(bucket)
        train_pick.extend(bucket[:PER_INTENT_TRAIN])
        probe_pick.extend(bucket[PER_INTENT_TRAIN:PER_INTENT_TRAIN + PER_INTENT_PROBE])
    rng.shuffle(train_pick)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "train.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["input", "output"])
        w.writeheader()
        for r in train_pick:
            w.writerow({"input": r["text"], "output": json.dumps({"intent": r["label"]})})
    for name, data in (("probe.jsonl", probe_pick), ("test.jsonl", test_rows)):
        with (args.out / name).open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps({"input": r["text"], "output": json.dumps({"intent": r["label"]})}) + "\n")
    (args.out / "labels.json").write_text(json.dumps(names, indent=1), encoding="utf-8")
    print(f"train {len(train_pick)}  probe {len(probe_pick)}  test {len(test_rows)}  labels {len(names)}")


if __name__ == "__main__":
    main()
