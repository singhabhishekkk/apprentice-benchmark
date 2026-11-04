head training rows 4500, classes 150, test rows 4500
generative answer accuracy on test rows: 0.9327

| layer, position | C | head CV accuracy (train) | head accuracy (test) | generative accuracy | head confidence = max p |
|---|---|---|---|---|---|
| L16_prompt_last | 0.1 | 0.9258 | 0.9060 | 0.9327 | written |
| L24_prompt_last | 0.1 | 0.9609 | 0.9447 | 0.9327 | written |
| L32_prompt_last | 0.03 | 0.9722 | 0.9562 | 0.9327 | written |
| L8_prompt_last | 0.1 | 0.9449 | 0.9256 | 0.9327 | written |

wrote runs/clinc150/classifier_conf.jsonl: per-row head confidence and head correctness per layer
