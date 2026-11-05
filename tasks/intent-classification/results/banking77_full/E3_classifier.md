head training rows 10003, classes 77, test rows 3080
generative answer accuracy on test rows: 0.9123

| layer, position | C | head CV accuracy (train) | head accuracy (test) | generative accuracy | head confidence = max p |
|---|---|---|---|---|---|
| L16_prompt_last | 0.03 | 0.8720 | 0.8844 | 0.9123 | written |
| L24_prompt_last | 0.03 | 0.9289 | 0.9286 | 0.9123 | written |
| L32_prompt_last | 0.03 | 0.9330 | 0.9357 | 0.9123 | written |
| L8_prompt_last | 0.03 | 0.8819 | 0.8948 | 0.9123 | written |

wrote runs/banking77_full/classifier_conf.jsonl: per-row head confidence and head correctness per layer
