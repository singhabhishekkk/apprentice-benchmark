# Task 3: Contract clause extraction (200 rows)

200 examples sampled (seed 42) from [CUAD v1](https://github.com/TheAtticusProject/cuad) (Contract Understanding Atticus Dataset) — 510 real commercial contracts sourced from SEC EDGAR, expert-annotated by lawyers (Atticus Project), CC-BY-4.0, "free to the public for commercial and non-commercial use." Input is a bounded excerpt around 5 clause categories' real answer spans (contracts run up to ~290K chars, far past any model's context window, so each row only carries the text that actually contains the answers — not the whole contract); output is a JSON object with `document_name`, `parties`, `agreement_date`, `governing_law`, `anti_assignment`. 140 train, 60 held out.

This is the hardest task in the suite: legal language, clauses that are often genuinely absent (~15% of optional fields are `""` in the gold set, which is correct, not missing data), and answer spans that live anywhere in a document rather than the preamble.

## Prompt optimizer: two student models

This task's GEPA runs were tried on **GLM 5.2** (via the OpenCode gateway) and **gpt-5.4-mini** (OpenAI). `gepa_optimize.py` gained `--api-base`/`--api-key` flags to support any OpenAI-compatible gateway, not just OpenAI itself, plus `--num-threads`/`--max-errors` after the OpenCode gateway repeatedly dropped connections during the final held-out eval.

GLM 5.2's GEPA-optimized run failed 3 times at the same point (~53-57/60 into the final eval) with `litellm.InternalServerError: Connection error` from the OpenCode gateway, even after lowering concurrency to 4 threads and raising the error budget to 30 — a real, reproducible gateway instability, not transient bad luck. gpt-5.4-mini completed cleanly on the first try on OpenAI's own API.

## Results

| Model | Score |
|---|---|
| GLM 5.2, plain prompt | 34.67 |
| GLM 5.2, GEPA-optimized prompt | *blocked — OpenCode gateway dropped connections 3x during final eval* |
| gpt-5.4-mini, plain prompt | 34.00 |
| gpt-5.4-mini, GEPA-optimized prompt | 36.33 |
| Qwen3.5-4B, no fine-tune | *pending* |
| Qwen3.5-4B, fine-tuned on 140 examples | *pending* |
| Gemma 4 E4B, no fine-tune | *pending* |
| Gemma 4 E4B, fine-tuned on 140 examples | *pending* |
| Gemma 4 E2B, no fine-tune | *pending* |
| Gemma 4 E2B, fine-tuned on 140 examples | *pending* |

All rows measured on the same 60 held-out rows, same seed, same split. GLM 5.2 baseline run 2026-07-07; gpt-5.4-mini run 2026-07-08. No number marked "pending" is published until the corresponding notebook has actually been run.

## Reproduce

**GLM 5.2 (needs an OpenCode API key; final eval may need a retry — see note above):**

```bash
python prepare_data.py
export OPENCODE_API_KEY=sk-...
python ../../gepa_optimize.py --data golden.csv \
  --model openai/glm-5.2 --reflection-model openai/glm-5.2 \
  --api-base https://opencode.ai/zen/go/v1 --api-key $OPENCODE_API_KEY \
  --num-threads 4 --max-errors 30
```

**gpt-5.4-mini (needs an OpenAI API key):**

```bash
python prepare_data.py
export OPENAI_API_KEY=sk-...
python ../../gepa_optimize.py --data golden.csv \
  --model openai/gpt-5.4-mini --reflection-model openai/gpt-5.4-mini
```

**Three fine-tunes (free, GPU Colab):** open `colab_finetune_qwen.ipynb` (Qwen3.5-4B), `colab_finetune_gemma.ipynb` (Gemma 4 E4B), or `colab_finetune_gemma2b.ipynb` (Gemma 4 E2B) in [Google Colab](https://colab.research.google.com), set runtime to GPU, run all cells. Same 200 rows, same seed, same split, same metric across all three.

## Task design

CUAD ships 41 clause categories per contract as extractive QA (question names the category, answer is a text span or absent). Two are near-universal identity fields (Document Name, Parties — present in 509/510 contracts); the other three (Agreement Date, Governing Law, Anti-Assignment) are real review fields lawyers look for, and their answers can sit anywhere in the document (Governing Law's answer starts at a median offset of 32,252 characters — nowhere near the preamble). `prepare_data.py` builds each row's input from a ±350-character window around each present category's real answer span, assembled in original document order — never a truncated prefix that would cut off the ground truth, and never a labeled excerpt (the model has to recognize "this is the governing-law clause" from content, same as it had to recognize receipt fields from unlabeled OCR lines in Task 2).

## Dataset selection

CUAD is a long-established, widely-cited legal-NLP benchmark (Hendrycks et al. 2021, arXiv:2103.06268) with real experts labeling real contracts — the strongest provenance available for a legal-extraction task, and CC-BY-4.0 clears it for a commercial benchmark repo (unlike several customer-support-ticket alternatives considered, which were either non-commercial-only or hybrid-synthetic).

## Caveats

60 held-out rows is a modest eval, same as Task 2. Field matching is exact-string, which is strict on legal prose (a rephrased clause with identical meaning scores zero). The creators of CUAD make no representations about the underlying contracts' own copyright status beyond their public EDGAR availability. Run the loop on your own data before believing anyone's numbers, including ours.
