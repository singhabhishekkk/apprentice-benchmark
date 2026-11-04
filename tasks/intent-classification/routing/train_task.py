"""Fine-tune Qwen3.5-4B on a task CSV through the product's own local path.

Same function `apprentice train --local --data` calls, with fewer epochs: the CLI default
of 30 is sized for datasets of a few hundred rows; 1,540 rows at batch 1 would be 32k
iterations. Two or three epochs answer the spike's question, which is about the confidence
signal, not the last point of accuracy. The product's 30% self-holdout and self-score stay.

Usage: uv run python train_task.py <task> <train.csv> <out_dir> [epochs]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from runapprentice.local import load_csv_rows, train_mlx_local

if __name__ == "__main__":
    task, csv_path, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    out.mkdir(parents=True, exist_ok=True)
    gold = load_csv_rows(csv_path)
    result = train_mlx_local(
        gold, task=task, adapter_path=out / "adapter", epochs=epochs, stream=True, fresh=True,
    )
    (out / "train_result.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({k: result.get(k) for k in ("final_score", "adapter_path", "effort", "iters")}))
