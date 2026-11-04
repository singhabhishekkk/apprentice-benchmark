"""Banking77 -> Apprentice task CSV and the frozen spike splits.

Writes, under --out:
  train.csv            20 rows per intent (1,540) for `apprentice train --local`; the product
                       holds out 30% of these itself for its own self-score.
  probe.jsonl          10 further rows per intent (770), never seen by the fine-tune; used to
                       train the hidden-state probe (idea 1 and 2 in the plan).
  test.jsonl           the full public test split (3,080), untouched; later cut into
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="untrusted download dir")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    local = snapshot_download("PolyAI/banking77", repo_type="dataset", local_dir=args.data)
    files = sorted(Path(local).rglob("*.parquet"))
    train_files = [f for f in files if "train" in f.name]
    test_files = [f for f in files if "test" in f.name]
    assert train_files and test_files, [f.name for f in files]

    names: list[str] | None = None
    readme = Path(local) / "README.md"
    if readme.exists():
        # The dataset card lists the ClassLabel names in order under `names:`.
        lines = readme.read_text(encoding="utf-8").splitlines()
        start = next((i for i, l in enumerate(lines) if l.strip() == "names:"), None)
        if start is not None:
            names = []
            for l in lines[start + 1:]:
                s = l.strip()
                if not s.startswith("'") and not s.startswith("- "):
                    break
                names.append(s.split(":", 1)[-1].strip().strip("'\""))
    assert names and len(names) == 77, f"could not read 77 label names, got {names and len(names)}"

    def rows(paths: list[Path]) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for p in paths:
            t = pq.read_table(p).to_pylist()
            for r in t:
                out.append({"text": r["text"], "label": names[int(r["label"])]})
        return out

    train_rows = rows(train_files)
    test_rows = rows(test_files)
    assert len(test_rows) == 3080, len(test_rows)

    rng = random.Random(SEED)
    by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in train_rows:
        by_label[str(r["label"])].append(r)
    train_pick: list[dict[str, object]] = []
    probe_pick: list[dict[str, object]] = []
    for label in sorted(by_label):
        bucket = by_label[label]
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
