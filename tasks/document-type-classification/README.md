# Task 4: Document type classification (200 rows)

200 examples sampled (seed 42) from [anirudh1112/corrected-tobacco-dataset-with-ocr](https://huggingface.co/datasets/anirudh1112/corrected-tobacco-dataset-with-ocr), a corrected Tobacco3482 variant with OCR text and 10 real document-type labels. The dataset card lists CC-BY-4.0. Input is the verbatim shipped `paperless-gpt` document-type prompt with OCR text filled into `<content>`; output is the gold class name only. 140 train, 60 held out.

This task is a case study for [icereed/paperless-gpt](https://github.com/icereed/paperless-gpt), an MIT-licensed open-source project that sends OCR'd documents to LLMs to suggest document type, title, correspondent, and tags. The baseline prompt here is its shipped document-type template, filled the same way its Go code fills `Language`, `AvailableDocumentTypes`, `Title`, and truncated `Content`.

## Results

| Model | Score |
|---|---|
| gpt-5.4-mini, plain paperless-gpt prompt | 78.33 |
| gpt-5.4-mini, GEPA-optimized prompt | 81.67 |
| Qwen3.5-4B, no fine-tune | *pending* |
| Qwen3.5-4B, fine-tuned on 140 examples | *pending* |
| Gemma 4 E4B, no fine-tune | *pending* |
| Gemma 4 E4B, fine-tuned on 140 examples | *pending* |

All rows measured on the same 60 held-out rows, same seed, same split (exact match, 0 to 100). gpt-5.4-mini runs 2026-07-10. No pending number is published until the corresponding command or notebook prints it.

## Dataset stats

Printed by the real `prepare_data.py` run (2026-07-10):

```text
wrote golden.csv: 200 rows
class distribution:
  ADVE: 20
  Email: 20
  Form: 20
  Letter: 20
  Memo: 20
  News: 20
  Note: 20
  Report: 20
  Resume: 20
  Scientific: 20
```

## Reproduce

**gpt-5.4-mini (needs an OpenAI API key):**

```bash
python prepare_data.py
export OPENAI_API_KEY=sk-...
python ../../gepa_optimize.py --data golden.csv \
  --metric exact \
  --model openai/gpt-5.4-mini --reflection-model openai/gpt-5.4-mini
```

**Two fine-tunes (free, GPU Colab):** open `colab_finetune_both.ipynb` in [Google Colab](https://colab.research.google.com) to run Qwen3.5-4B and Gemma 4 E4B in one session (GPU memory freed between models), or `colab_finetune_qwen.ipynb` / `colab_finetune_gemma.ipynb` individually. Set runtime to GPU, run all cells. Same 200 rows, same seed, same split, exact-match metric.

## Dataset selection

First choice was Tobacco3482 with OCR text. The selected Hugging Face dataset ships OCR text in the `text` field, corrected class labels in `label`, and a CC-BY-4.0 license on its dataset page. It avoids local OCR and keeps this benchmark text-only, matching the paperless-gpt path after OCR has already happened.

Class labels are used as shipped: `ADVE`, `Email`, `Form`, `Letter`, `Memo`, `News`, `Note`, `Report`, `Resume`, `Scientific`. `ADVE` is the dataset's advertisement label.

## Provenance and license

- Source dataset: [anirudh1112/corrected-tobacco-dataset-with-ocr](https://huggingface.co/datasets/anirudh1112/corrected-tobacco-dataset-with-ocr)
- Source family: Tobacco3482, scanned tobacco-industry documents with document-type labels
- OCR: already included in the dataset as text
- Dataset license: CC-BY-4.0 as listed by Hugging Face
- Baseline prompt source: [icereed/paperless-gpt](https://github.com/icereed/paperless-gpt), MIT

## Case study: paperless-gpt

paperless-gpt pairs with paperless-ngx and uses LLMs over OCR text to suggest metadata for scanned documents. This benchmark isolates one shipped slice: selecting `document_type` from available types. The prompt is the verbatim document-type template from paperless-gpt, and `prepare_data.py` fills the template variables the same way its Go code does. `Content` is capped at about 4,000 characters as a small, documented simplification of the project's token-based truncation.

## Caveats

60 held-out rows is a modest eval. This task measures one field only. Tobacco3482 labels are useful public gold labels, but label-error work has found ambiguity in the original benchmark family. paperless-gpt production traffic may include different document types, OCR engines, languages, and user-specific type lists. Run the loop on your own data before believing anyone's numbers, including ours.
