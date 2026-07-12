# apprentice-benchmark

Small models learn a task from a golden dataset, then beat the bigger model that was doing it. This repo is a growing suite of use cases — different industries, data shapes, and dataset sizes — all run through the same loop. Every number comes from a run you can reproduce yourself: one shared runner, per-task data, fixed seeds.

## Use cases

| Use case | Industry / data shape | Data | Rows | Baseline | GEPA-optimized | Fine-tuned Qwen3.5-4B | Fine-tuned Gemma 4 |
|---|---|---|---|---|---|---|---|
| [JSON extraction](tasks/json-extraction/) | API responses, structured generation | json-mode-eval, public | 100 (30 held out) | 83.1 (GPT-4o-mini) | 85.6 | **88.9** | — |
| [Receipt extraction](tasks/receipt-extraction/) | Finance / retail, noisy OCR documents | SROIE scanned receipts, real + human-annotated | 200 (60 held out) | 72.9 (GPT-4o-mini) | 84.2 | **89.2** | **87.1** (E4B) |
| [Contract clause extraction](tasks/contract-clause-extraction/) | Legal / compliance, long-document extraction | CUAD v1, real contracts + expert legal annotation | 200 (60 held out) | 34.0 (gpt-5.4-mini) | 36.3 | *pending* | *pending* (E4B + E2B) |
| [Document type classification](tasks/document-type-classification/) | Document automation, noisy OCR classification | Corrected Tobacco3482 OCR text, real labels | 200 (60 held out) | 78.3 (gpt-5.4-mini) | 81.7 | **80.0** | **86.7** (E4B) |

Two use cases so far, two data sizes, clean and messy data — and on both, the fine-tuned 4B beat its teacher's GEPA-optimized prompt. On the harder OCR task, prompt optimization alone gained +11.3, and the fine-tune added +5.0 on top (from a raw score of 42.5 — the verified dataset is what makes the small model work). A control run with the newer GPT-5.4-mini on the receipts task scored 72.9 baseline / 79.6 GEPA-optimized — the fine-tuned 4B (89.2) beats it by +9.6. Scores are field-level F1 on a held-out split (seed 42), 0 to 100 — never compare across tasks.

The loop is not tied to one model family: [receipt-extraction](tasks/receipt-extraction/) also fine-tunes `google/gemma-4-E4B-it` (Apache-2.0) via `colab_finetune_gemma.ipynb`, same split and metric as the Qwen run — 79.2 raw, 87.1 fine-tuned, beating every teacher prompt.

Nor is it tied to one prompt-optimizer LLM: [contract-clause-extraction](tasks/contract-clause-extraction/) ran GEPA on both **GLM 5.2** via the OpenCode gateway and **gpt-5.4-mini** (`gepa_optimize.py --api-base` now supports any OpenAI-compatible gateway, plus `--num-threads`/`--max-errors` after GLM's gateway repeatedly dropped connections mid-eval — gpt-5.4-mini completed cleanly: 34.0 baseline, 36.3 optimized). Fine-tunes on the same split are next: Qwen3.5-4B, Gemma 4 E4B, and Gemma 4 E2B.

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

The seed experiments behind [Apprentice](https://github.com/singhabhishekkk/Apprentice): small models learn your task from your golden dataset, then replace the expensive model — eval-gated, with instant rollback.

Apache-2.0.
