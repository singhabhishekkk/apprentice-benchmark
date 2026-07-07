# Task 2: Receipt extraction (200 rows)

200 examples sampled (seed 42) from [Voxel51/scanned_receipts](https://huggingface.co/datasets/Voxel51/scanned_receipts) — real scanned receipts (SROIE), human-annotated, CC-BY-4.0. Input is the receipt's OCR text; output is a JSON object with `company`, `address`, `date`, `total`. 140 train, 60 held out.

This is a deliberately harder, messier task than Task 1: real OCR noise, longer inputs, twice the data.

## Results

| Model | Score |
|---|---|
| GPT-4o-mini, plain prompt | 72.9 |
| GPT-4o-mini, GEPA-optimized prompt | 84.2 |
| GPT-5.4-mini, plain prompt | 72.9 |
| GPT-5.4-mini, GEPA-optimized prompt | 79.6 |
| Qwen3.5-4B, no fine-tune | 42.5 |
| **Qwen3.5-4B, fine-tuned on 140 examples** | **89.2** |
| Gemma 4 E4B, no fine-tune | 79.2 |
| **Gemma 4 E4B, fine-tuned on 140 examples** | **87.1** |

Prompt optimization on GPT-4o-mini gained +11.3 on the held-out set — over 4x the gain it produced on the easier Task 1. The fine-tuned Qwen3.5-4B beat both GEPA-optimized teachers: +5.0 over GPT-4o-mini (84.2) and +9.6 over the newer GPT-5.4-mini (79.6), starting from a raw score of 42.5. All runs measured on the same 60 held-out rows, GEPA runs on 2026-07-06 and the Colab fine-tunes the same day. Exact fine-tune score: 89.17.

A second model family, Gemma 4 E4B (Apache-2.0, ~4.5B effective params), ran the identical loop: same 200 rows, same seed, same split, same metric, same teacher numbers reused (no GEPA re-run). It started far ahead of Qwen raw (79.17 vs 42.50 — Gemma 4's instruction-tuning already handles structured JSON output better zero-shot) and still gained from fine-tuning, beating every teacher prompt at 87.08 — though it landed 2.1 below the fine-tuned Qwen3.5-4B. Exact fine-tune score: 87.08.

The GPT-5.4-mini rows are a control run: same pipeline, same data, same seed, only the student model swapped (baseline coincidentally matches GPT-4o-mini's 72.92 to the second decimal; the per-row logs differ, see `spike2_run_gpt54mini.log` artifacts in the source project). The fine-tuned adapters are published: [Qwen3.5-4B](https://huggingface.co/singhabhishekkk/apprentice-qwen35-4b-lora-receipts), [Gemma 4 E4B](https://huggingface.co/singhabhishekkk/apprentice-gemma4-e4b-lora-receipts).

## Reproduce

**The GPT-4o-mini baselines (needs an OpenAI key):**

```bash
python prepare_data.py
export OPENAI_API_KEY=sk-...
python ../../gepa_optimize.py --data golden.csv
```

**The fine-tune (free, GPU Colab):** open `colab_finetune_qwen.ipynb` in [Google Colab](https://colab.research.google.com), set runtime to GPU, run all cells. Same 200 rows, same seed, same split, same metric.

**A second model family (Apache-2.0, for AMD Developer Hackathon's Best Use of Gemma track):** `colab_finetune_gemma.ipynb` runs the identical fine-tune loop on `google/gemma-4-E4B-it` — same split, same metric, same teacher numbers, no GEPA re-run. Run 2026-07-07: 79.17 raw, 87.08 fine-tuned (published above).

## Dataset selection

Runner-ups considered and rejected: `sandeeppanem/resume-json-extraction-5k` (LLM-generated labels), `HenriqueGodoy/extract-0` (marked synthetic), `paraloq/json_data_extraction` (research-only intent, weaker provenance). Receipts won on real data with human annotations.

## Caveats

60 held-out rows is still a modest eval. Field matching is exact-string, which is strict on OCR text (a one-character total mismatch scores zero for that field). Run the loop on your own data before believing anyone's numbers, including ours.
