# apprentice-benchmark

Small models learn a task from a golden dataset, then beat the bigger model that was doing it. Every number here comes from a run you can reproduce yourself — one shared runner, per-task data, fixed seeds.

## Tasks

| Task | Data | Rows | Baseline (GPT-4o-mini) | GEPA-optimized | Fine-tuned Qwen3.5-4B |
|---|---|---|---|---|---|
| [JSON extraction](tasks/json-extraction/) | json-mode-eval, public | 100 (30 held out) | 83.1 | 85.6 | **88.9** |
| [Receipt extraction](tasks/receipt-extraction/) | SROIE scanned receipts, real + human-annotated | 200 (60 held out) | 72.9 | 84.2 | **88.3** |

Two different tasks, two different data sizes, one loop — and on both, the fine-tuned 4B beat its teacher's GEPA-optimized prompt. On the harder OCR task, prompt optimization alone gained +11.3, and the fine-tune added +4.2 on top. Scores are field-level F1 on a held-out split (seed 42), 0 to 100 — never compare across tasks.

## How it works

Each task folder has:

- `prepare_data.py` — downloads the public dataset and writes `golden.csv` (deterministic seed)
- `colab_finetune_qwen.ipynb` — LoRA fine-tune of Qwen3.5-4B on the train split, evaluated with the same metric on the same held-out split
- `README.md` — results, exact reproduce commands, caveats

The shared runner at the repo root:

```bash
uv init && uv add dspy
cd tasks/<task>
python prepare_data.py
export OPENAI_API_KEY=sk-...
python ../../gepa_optimize.py --data golden.csv
```

It prints the baseline score, runs DSPy GEPA prompt optimization (gpt-4o reflection), prints the optimized score, and saves `optimized_program.json`.

## Honesty rules

- Every published number comes from a real run.
- Held-out splits are small (30 and 60 rows). Treat these as directional, then run the loop on your own data.
- Task scores are not comparable to each other: different data, different difficulty.

## What this is

The seed experiments behind [Apprentice](https://github.com/singh-abhishekk/Apprentice): small models learn your task from your golden dataset, then replace the expensive model — eval-gated, with instant rollback.

Apache-2.0.
