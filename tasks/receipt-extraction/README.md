# Task 2: Receipt extraction (200 rows)

200 examples sampled (seed 42) from [Voxel51/scanned_receipts](https://huggingface.co/datasets/Voxel51/scanned_receipts) — real scanned receipts (SROIE), human-annotated, CC-BY-4.0. Input is the receipt's OCR text; output is a JSON object with `company`, `address`, `date`, `total`. 140 train, 60 held out.

This is a deliberately harder, messier task than Task 1: real OCR noise, longer inputs, twice the data.

## Results

| Model | Score |
|---|---|
| GPT-4o-mini, plain prompt | 72.9 |
| GPT-4o-mini, GEPA-optimized prompt | 84.2 |
| Qwen3.5-4B, no fine-tune | 42.5 |
| **Qwen3.5-4B, fine-tuned on 140 examples** | **88.3** |

Prompt optimization alone gained +11.3 on the held-out set — over 4x the gain it produced on the easier Task 1. The fine-tuned 4B then beat the GEPA-optimized teacher by another +4.2, starting from a raw score of 42.5. Both runs measured on the same 60 held-out rows, GEPA on 2026-07-06 and the Colab fine-tune the same day.

## Reproduce

**The GPT-4o-mini baselines (needs an OpenAI key):**

```bash
python prepare_data.py
export OPENAI_API_KEY=sk-...
python ../../gepa_optimize.py --data golden.csv
```

**The fine-tune (free, GPU Colab):** open `colab_finetune_qwen.ipynb` in [Google Colab](https://colab.research.google.com), set runtime to GPU, run all cells. Same 200 rows, same seed, same split, same metric.

## Dataset selection

Runner-ups considered and rejected: `sandeeppanem/resume-json-extraction-5k` (LLM-generated labels), `HenriqueGodoy/extract-0` (marked synthetic), `paraloq/json_data_extraction` (research-only intent, weaker provenance). Receipts won on real data with human annotations.

## Caveats

60 held-out rows is still a modest eval. Field matching is exact-string, which is strict on OCR text (a one-character total mismatch scores zero for that field). Run the loop on your own data before believing anyone's numbers, including ours.
