# apprentice-benchmark

Small models learn a task from a golden dataset, then beat the bigger model that was doing it. This repo is a growing suite of use cases — different industries, data shapes, and dataset sizes — all run through the same loop. Every number comes from a run you can reproduce yourself: one shared runner, per-task data, fixed seeds.

## Use cases

| Use case | Industry / data shape | Data | Rows | Baseline (GPT-4o-mini) | GEPA-optimized | Fine-tuned Qwen3.5-4B |
|---|---|---|---|---|---|---|
| [JSON extraction](tasks/json-extraction/) | API responses, structured generation | json-mode-eval, public | 100 (30 held out) | 83.1 | 85.6 | **88.9** |
| [Receipt extraction](tasks/receipt-extraction/) | Finance / retail, noisy OCR documents | SROIE scanned receipts, real + human-annotated | 200 (60 held out) | 72.9 | 84.2 | **88.3** |

Two use cases so far, two data sizes, clean and messy data — and on both, the fine-tuned 4B beat its teacher's GEPA-optimized prompt. On the harder OCR task, prompt optimization alone gained +11.3, and the fine-tune added +4.2 on top (from a raw score of 42.5 — the verified dataset is what makes the small model work). Scores are field-level F1 on a held-out split (seed 42), 0 to 100 — never compare across tasks.

## How it works

Each use case is a self-contained folder under `tasks/` with a fixed contract:

- `prepare_data.py` — downloads the public dataset and writes `golden.csv` (deterministic seed)
- `colab_finetune_qwen.ipynb` — LoRA fine-tune of Qwen3.5-4B on the train split, evaluated with the same metric on the same held-out split
- `README.md` — results, exact reproduce commands, dataset provenance + license, caveats

The shared runner at the repo root:

```bash
uv init && uv add dspy
cd tasks/<task>
python prepare_data.py
export OPENAI_API_KEY=sk-...
python ../../gepa_optimize.py --data golden.csv
```

It prints the baseline score, runs DSPy GEPA prompt optimization (gpt-4o reflection), prints the optimized score, and saves `optimized_program.json`.

## Adding a use case

New tasks follow the same contract, so results stay comparable in method even when they are not comparable in score:

1. Pick a real public dataset (verify license and row count on the dataset page; no synthetic-only data, no fabricated rows).
2. Write `prepare_data.py` with a deterministic seed and document the exact row count.
3. Run `gepa_optimize.py` for the baseline + optimized scores; publish only what it printed.
4. Adapt the Colab notebook to the same seed, split, and metric; publish only what it printed.
5. README: results table, reproduce commands, provenance, caveats.

## Honesty rules

- Every published number comes from a real run.
- Held-out splits are small (30 and 60 rows). Treat these as directional, then run the loop on your own data.
- Task scores are not comparable to each other: different data, different difficulty.

## What this is

The seed experiments behind [Apprentice](https://github.com/singh-abhishekk/Apprentice): small models learn your task from your golden dataset, then replace the expensive model — eval-gated, with instant rollback.

Apache-2.0.
