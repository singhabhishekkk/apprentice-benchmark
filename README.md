# apprentice-benchmark

A small fine-tuned model beats GPT-4o-mini on a JSON extraction task. Run it yourself.

## Results

100 examples from [NousResearch/json-mode-eval](https://huggingface.co/datasets/NousResearch/json-mode-eval). 70 train, 30 held out. Metric: JSON validity + field-level F1. Same split (seed 42) for every run.

| Model | Score |
|---|---|
| GPT-4o-mini, plain prompt | 83.1 |
| GPT-4o-mini, GEPA-optimized prompt | 85.6 |
| Qwen3.5-4B, no fine-tune | 69.1 |
| **Qwen3.5-4B, fine-tuned on 70 examples** | **88.9** |

Training: LoRA, 3 epochs, 7.5 minutes on a free Colab T4.

## Reproduce

**The fine-tune (free, ~40 min):**
Open `colab_finetune_qwen.ipynb` in [Google Colab](https://colab.research.google.com). Set runtime to GPU. Run all cells. The notebook downloads the data, trains, and prints the score table.

**The GPT-4o-mini baselines (needs an OpenAI key, ~$2):**
```bash
uv init && uv add dspy
python prepare_data.py
export OPENAI_API_KEY=sk-...
python gepa_optimize.py --data golden.csv
```

## Trained adapter

[singhabhishekkk/apprentice-qwen35-4b-lora-jsonextract](https://huggingface.co/singhabhishekkk/apprentice-qwen35-4b-lora-jsonextract) — LoRA weights, model card with full config.

## Caveats

30 held-out rows is a small eval. The dataset is public and fairly easy. Your task will differ — that is the point: run this loop on your own data before believing anyone's numbers, including ours.

## What this is

The seed experiment behind [Apprentice](https://github.com/singh-abhishekk/Apprentice): small models learn your task from your golden dataset, then replace the expensive model — eval-gated, with instant rollback.

Apache-2.0.
