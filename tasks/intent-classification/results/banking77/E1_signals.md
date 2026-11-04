Rows 3080: explore 1027, calibrate 1027, evaluate 1026. Target risk 0.05, delta 0.05, seed 42.
Evaluate third: accuracy 0.822, format_valid rate 1.000.

| signal | explore AUROC | calibrate accepted/rows | calibrate bound | evaluate acceptance (Wilson 95%) | evaluate error among kept (count, rate, CP95 upper) | error among deferred | saved $/1k vs always-frontier |
|---|---|---|---|---|---|---|---|
| seq_logprob | 0.834 | 591/1027 | 0.047 | 0.580 (0.549 to 0.610) | 15, 0.025, 0.039 | 0.390 | 1.40 |
| mean_logprob | 0.838 | 591/1027 | 0.047 | 0.580 (0.549 to 0.610) | 15, 0.025, 0.039 | 0.390 | 1.40 |
| min_token_logprob | 0.827 | 591/1027 | 0.047 | 0.580 (0.549 to 0.610) | 15, 0.025, 0.039 | 0.390 | 1.40 |
| agreement | 0.765 | 0/1027 | 1.000 | 0.000 (0.000 to 0.004) | 0, n/a, 1.000 | 0.178 | 0.00 |
| modal_share | 0.763 | 0/1027 | 1.000 | 0.000 (0.000 to 0.004) | 0, n/a, 1.000 | 0.178 | 0.00 |
| probe_pre | 0.701 | 0/1027 | 1.000 | 0.000 (0.000 to 0.004) | 0, n/a, 1.000 | 0.178 | 0.00 |
| probe_post | 0.802 | 0/1027 | 1.000 | 0.000 (0.000 to 0.004) | 0, n/a, 1.000 | 0.178 | 0.00 |
| min_token_logprob_x_agreement | 0.829 | 576/1027 | 0.046 | 0.574 (0.544 to 0.604) | 15, 0.025, 0.039 | 0.384 | 1.39 |

Saved $/1k = evaluate acceptance rate x $2.42 per 1k requests (Opus 5, OpenRouter Banking77 post). Small model cost assumed zero on a Mac, electricity not counted.
