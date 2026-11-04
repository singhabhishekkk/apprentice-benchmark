head training rows 10003, classes 77, test rows 3080
generative answer accuracy on test rows: 0.8192

| layer, position | C | head CV accuracy (train) | head accuracy (test) | generative accuracy | head confidence = max p |
|---|---|---|---|---|---|
| L16_prompt_last | 0.1 | 0.8722 | 0.8805 | 0.8192 | written |
| L24_prompt_last | 0.01 | 0.9161 | 0.9231 | 0.8192 | written |
| L32_prompt_last | 0.03 | 0.9232 | 0.9305 | 0.8192 | written |
| L8_prompt_last | 0.03 | 0.8818 | 0.8958 | 0.8192 | written |

wrote runs/banking77/classifier_full_conf.jsonl: per-row head confidence and head correctness per layer
