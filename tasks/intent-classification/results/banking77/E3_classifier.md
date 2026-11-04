head training rows 2310, classes 77, test rows 3080
generative answer accuracy on test rows: 0.8192

| layer, position | C | head CV accuracy (train) | head accuracy (test) | generative accuracy | head confidence = max p |
|---|---|---|---|---|---|
| L16_prompt_last | 0.1 | 0.7554 | 0.7964 | 0.8192 | written |
| L24_prompt_last | 0.1 | 0.8641 | 0.8864 | 0.8192 | written |
| L32_prompt_last | 0.1 | 0.8957 | 0.9023 | 0.8192 | written |
| L8_prompt_last | 0.3 | 0.7835 | 0.8075 | 0.8192 | written |

wrote runs/banking77/classifier_conf.jsonl: per-row head confidence and head correctness per layer
