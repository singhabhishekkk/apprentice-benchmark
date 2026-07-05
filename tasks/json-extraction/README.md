# Task 1: JSON extraction (100 rows)

100 examples from [NousResearch/json-mode-eval](https://huggingface.co/datasets/NousResearch/json-mode-eval): a JSON schema plus a short text, extract the fields. 70 train, 30 held out, seed 42.

## Results

| Model | Score |
|---|---|
| GPT-4o-mini, plain prompt | 83.1 |
| GPT-4o-mini, GEPA-optimized prompt | 85.6 |
| Qwen3.5-4B, no fine-tune | 69.1 |
| **Qwen3.5-4B, fine-tuned on 70 examples** | **88.9** |

Training: LoRA, 3 epochs, 7.5 minutes on a free Colab T4.

## Reproduce

**The fine-tune (free, ~40 min):** open `colab_finetune_qwen.ipynb` in [Google Colab](https://colab.research.google.com), set runtime to GPU, run all cells.

**The GPT-4o-mini baselines (needs an OpenAI key, ~$2):**

```bash
python prepare_data.py
export OPENAI_API_KEY=sk-...
python ../../gepa_optimize.py --data golden.csv
```

## Trained adapter

[singhabhishekkk/apprentice-qwen35-4b-lora-jsonextract](https://huggingface.co/singhabhishekkk/apprentice-qwen35-4b-lora-jsonextract) — LoRA weights, model card with full config.

## Caveats

30 held-out rows is a small eval. The dataset is public and fairly easy. Your task will differ — that is the point.
