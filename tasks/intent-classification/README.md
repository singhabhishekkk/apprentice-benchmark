# Task 6: Intent classification with a bounded router (Banking77 and CLINC150)

Two public intent benchmarks, one recipe, every number from a recorded run on 2026-11-04 on an
Apple-silicon Mac (36 GB). This task answers a different question from the others in this
repo: not only "does a fine-tuned small model match the frontier", but "can the small model
decide, per request, whether to answer or hand the request to the frontier, with a proven
ceiling on the error among the answers it keeps".

## Results

### Accuracy on the full public test splits

| System | Banking77 (3,080 rows, 77 intents) | CLINC150 (4,500 in-scope rows, 150 intents) | Who ran it |
|---|---|---|---|
| **Head on our fine-tune** (one matmul on the prefill state, no decoding) | **90.2** | **95.6** | this folder |
| Claude Opus 5, labels in the prompt, temperature 0, strict enum | 84.4 | not run | [OpenRouter, 2026](https://openrouter.ai/blog/insights/jev-vs-claude-opus-5-classification/) |
| Our fine-tune's generated answer, Qwen3.5-4B 4-bit LoRA, two epochs | 81.9 | 93.3 | this folder |
| Jev 1.13, Decisions API choice question | 81.0 | not run | same OpenRouter post |
| TRACER's classical surrogate (their split, OOS handling may differ) | not run | 92.4 | [arXiv 2604.14531](https://arxiv.org/abs/2604.14531) |

Exact-match accuracy, 0 to 100. Format valid on every row of both test splits. Scores compare
within a benchmark only. Full-fine-tuned encoders report low 90s on Banking77; this is not a
state-of-the-art claim.

### Routing: share of requests the head keeps at a 5% error ceiling

Selection with guaranteed risk (Geifman and El-Yaniv 2017, Theorem 3.2), target risk 5%,
confidence 95%, threshold chosen on one third of the test rows, measured on another third
the rule never saw, three different splits.

| Benchmark | Kept share across three seeds | Realised error among kept (one-sided 95% upper limit) |
|---|---|---|
| Banking77 | 70.1%, 79.2%, 81.8% | 1.0% (1.8%), 2.5% (3.6%), 2.7% (3.9%) |
| CLINC150 | 88.9%, 92.1%, 90.2% | 1.3% (1.9%), 1.4% (2.0%), 1.2% (1.8%) |

The bound held on every seed on both benchmarks. What the head defers goes to the frontier.

### Latency, one request at a time, this Mac, warm, 4-bit (`results/banking77/E7_latency.json`)

| Path | p50 | p95 |
|---|---|---|
| Head: prefill plus one matmul | 63 ms | 107 ms |
| Generated answer, 8 tokens | 400 ms | 458 ms |

Measured under these conditions only. OpenRouter's numbers for Opus 5 and Jev were measured
under theirs (8 concurrent requests, client-observed) and are not comparable.

## Method, exactly

- **Data.** Banking77: [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77),
  CC BY 4.0, 10,003 train, 3,080 test. CLINC150: [clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos)
  `plus` configuration, CC BY 3.0, the out-of-scope class dropped, 4,500 in-scope test rows.
- **Split.** Seed 42. Per intent: 20 training rows and 10 further "probe" rows the fine-tune
  never sees. The public test split is untouched by everything until the final measurement.
- **Fine-tune.** The Apprentice product path (`runapprentice.local.train_mlx_local`, mlx-lm
  LoRA on `mlx-community/Qwen3.5-4B-OptiQ-4bit`, 16 layers), two epochs, batch 1. The product
  holds out 30% of the training rows for its own self-score (81.2 on Banking77, 95.6 on
  CLINC150). Banking77 was also trained at four epochs: 82.4, half a point more.
- **Generated answer.** Greedy decoding of `{"intent": "<label>"}`; a row is correct when the
  parsed label equals gold; malformed output is wrong.
- **Head.** Hidden state at the last prompt token after layer 32 of 32 (residual stream,
  pre-norm), 2,560 floats. Logistic regression, standardised features, C chosen by 5-fold
  cross-validation on the head's training rows only (the probe rows plus the fine-tune's
  training rows: 2,310 on Banking77, 4,500 on CLINC150), then applied once to the test rows.
  Confidence is the head's max-probability.
- **Routing bound.** `routing/sgr.py`: ten fixed coverage cuts at δ/10 each (not the paper's
  binary search, which assumed risk monotone in coverage and missed feasible cuts on these
  rows), calibrated on one third of the test rows, evaluated on a disjoint third.
- **Not done.** No hyperparameter search of any kind. No prompt engineering. Comparator
  numbers are copied from their publishers with the link; none were re-run here.

## Reproduce

```bash
cd tasks/intent-classification/routing && uv sync      # mlx-lm, scikit-learn, huggingface-hub
cd ..
uv run --project routing python -I prepare_banking77.py --data /tmp/banking77_raw --out data/banking77
uv run --project routing python routing/train_task.py banking77 data/banking77/train.csv runs/banking77 2
uv run --project routing python routing/score_rows.py --adapter runs/banking77/adapter --rows data/banking77/test.jsonl --out runs/banking77/test_scores.jsonl --samples 1
uv run --project routing python routing/score_layers.py --adapter runs/banking77/adapter --rows data/banking77/probe.jsonl --out runs/banking77/probe_layers
uv run --project routing python routing/score_layers.py --adapter runs/banking77/adapter --rows data/banking77/train.jsonl --out runs/banking77/train_layers
uv run --project routing python routing/score_layers.py --adapter runs/banking77/adapter --rows data/banking77/test.jsonl --out runs/banking77/test_layers
uv run --project routing python routing/classifier_tier.py --train runs/banking77/probe_layers runs/banking77/train_layers --train-rows data/banking77/probe.jsonl data/banking77/train.jsonl --test runs/banking77/test_layers --test-rows data/banking77/test.jsonl --out runs/banking77/classifier_conf.jsonl
```

`train.jsonl` is `train.csv` as JSON lines (`{"input","output"}`). For the routing bound,
write `head_L32.jsonl` from `classifier_conf.jsonl` as one line per row with
`confidence = head_L32_prompt_last` and `correct = head_L32_prompt_last_correct`, then for
each seed in 42, 7, 2026:

```bash
uv run --project routing python routing/sgr.py runs/banking77/head_L32.jsonl --target-risk 0.05 --delta 0.05 --seed 42
```

The outputs are recorded in `results/*/E3_head_sgr.md`. Replace `banking77` with
`clinc150` for the second benchmark. Needs an Apple-silicon Mac with
mlx-lm; about an hour of training and an hour of scoring per benchmark.

The `routing/` folder is the spike's own scripts, copied as run. Their `pyproject.toml`
points at the Apprentice SDK by local path; install `runapprentice[local]` from PyPI instead
when running outside the Apprentice checkout.

## Artifacts

Adapters and heads: uploaded by the owner to Hugging Face under `singhabhishekkk/`, linked
here once the upload is done. Built with Qwen (Apache-2.0).

## Caveats

- The head was trained partly on rows the LoRA also trained on, so its cross-validation
  number may flatter; the published accuracy is on the public test split, which neither saw.
- The routing bound is a promise about traffic like the calibration rows. On a simulated
  shift of the input mix (calibrate on half the intents, evaluate on the other half) it held
  one direction and broke the other (`results/banking77/E6_robustness.md`). Two hundred
  labelled calibration rows is the floor: under that, nothing certifies.
- One model family, one laptop, two epochs, no tuning. Your intents will differ; run the loop
  on your own data before believing anyone's numbers, including ours.
- On a 25-token classification prompt a decision API costs about a quarter of a cent per
  thousand requests (`results/banking77/E7_cost.md`), so the case for a small model here is
  accuracy, latency and privacy, not price.
