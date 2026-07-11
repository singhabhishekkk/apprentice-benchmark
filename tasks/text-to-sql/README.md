# Text-to-SQL data preparation and BIRD evaluation

This task contains data preparation and evaluation only. It prints benchmark results from real executions; it contains no training code.

## Provenance

- [SynSQL-2.5M](https://huggingface.co/datasets/seeklhy/SynSQL-2.5M) supplies synthetic SFT data under Apache-2.0.
- [BIRD](https://bird-bench.github.io/) supplies evaluation data; see its official page for license terms.

Download BIRD dev manually from its homepage (~1.6 GB), then unzip it. Keep either `dev.json` and `dev_databases/` directly under the chosen directory, or under its `dev_20240627/` child.

## Reproduce

From repository root:

```bash
uv run python tasks/text-to-sql/prepare_data.py --rows 50000 --output tasks/text-to-sql/train.jsonl

uv run python tasks/text-to-sql/eval_bird.py \
  --bird-dir /path/to/unzipped/bird-dev \
  --predictions preds.jsonl \
  --report report.json

OPENAI_API_KEY=... uv run python tasks/text-to-sql/eval_bird.py \
  --bird-dir /path/to/unzipped/bird-dev \
  --api-base http://localhost:8000/v1 \
  --model model-name \
  --generated-output predictions.jsonl \
  --report report.json

uv run pytest tasks/text-to-sql/test_eval_bird.py -q
```

Phase-1 fine-tune: open `colab_finetune_qwen3_sql.ipynb` in Google Colab (GPU runtime) and run top to bottom. It clones this repo, builds the training subset, scores the raw Qwen3.5-4B baseline on BIRD dev, trains LoRA, rescores, and prints the gate comparison. Set `QUICK_N = 200` for a smoke pass; smoke scores are never publishable.

`preds.jsonl` and generated predictions use one JSON object per line: `{"question_id": ..., "sql": ...}`. Re-score generated SQL with `--predictions predictions.jsonl`.

## Honesty and contamination

No number is published that a run did not print.

BIRD dev and test rows must never enter any training set.
