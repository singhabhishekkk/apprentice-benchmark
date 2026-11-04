Rows 3080, target risk 0.07, delta 0.05, event = correct AND format_valid.

### (a) Split seeds

| signal | seed | theta | calibrate bound | evaluate acceptance (Wilson 95%) | evaluate error among kept (errors/kept, rate, CP95 upper) |
|---|---|---|---|---|---|
| mean_logprob | 42 | -0.0125 | 0.062 | 0.628 (0.598 to 0.657) | 23/644, 0.036, CP95 0.050 |
| mean_logprob | 7 | -0.0125 | 0.065 | 0.639 (0.610 to 0.668) | 29/656, 0.044, CP95 0.060 |
| mean_logprob | 2026 | -0.0125 | 0.054 | 0.643 (0.613 to 0.672) | 29/660, 0.044, CP95 0.059 |
| min_token_logprob | 42 | 0 | 0.047 | 0.580 (0.549 to 0.610) | 15/595, 0.025, CP95 0.039 |
| min_token_logprob | 7 | 0 | 0.060 | 0.577 (0.547 to 0.607) | 19/592, 0.032, CP95 0.047 |
| min_token_logprob | 2026 | 0 | 0.039 | 0.590 (0.559 to 0.619) | 21/605, 0.035, CP95 0.050 |

- mean_logprob: evaluate acceptance across seeds, min 0.628, max 0.643
- min_token_logprob: evaluate acceptance across seeds, min 0.577, max 0.590

### (b) Calibration size (seed 42; evaluate third untouched, 1026 rows)

| signal | calibrate rows | theta | calibrate bound | evaluate acceptance (Wilson 95%) | evaluate error among kept (errors/kept, rate, CP95 upper) |
|---|---|---|---|---|---|
| mean_logprob | 1027 | -0.0125 | 0.062 | 0.628 (0.598 to 0.657) | 23/644, 0.036, CP95 0.050 |
| mean_logprob | 400 | -0.0125 | 0.068 | 0.628 (0.598 to 0.657) | 23/644, 0.036, CP95 0.050 |
| mean_logprob | 200 | 0 | 0.050 | 0.580 (0.549 to 0.610) | 15/595, 0.025, CP95 0.039 |
| mean_logprob | 100 | inf | 1.000 | 0.000 (0.000 to 0.004) | 0/0, n/a, CP95 1.000 |
| mean_logprob | 50 | inf | 1.000 | 0.000 (0.000 to 0.004) | 0/0, n/a, CP95 1.000 |
| min_token_logprob | 1027 | 0 | 0.047 | 0.580 (0.549 to 0.610) | 15/595, 0.025, CP95 0.039 |
| min_token_logprob | 400 | 0 | 0.055 | 0.580 (0.549 to 0.610) | 15/595, 0.025, CP95 0.039 |
| min_token_logprob | 200 | 0 | 0.050 | 0.580 (0.549 to 0.610) | 15/595, 0.025, CP95 0.039 |
| min_token_logprob | 100 | inf | 1.000 | 0.000 (0.000 to 0.004) | 0/0, n/a, CP95 1.000 |
| min_token_logprob | 50 | inf | 1.000 | 0.000 (0.000 to 0.004) | 0/0, n/a, CP95 1.000 |

### (c) Simulated distribution shift (not production drift)

Intents sorted and split by alternating index: A = 39 intents, 1560 rows; B = 38 intents, 1520 rows. Calibrate on every row of one half, evaluate on every row of the other.

| signal | calibrate on | evaluate on | calibrate bound | evaluate acceptance (Wilson 95%) | evaluate error among kept (errors/kept, rate, CP95 upper) | realised error within bound? |
|---|---|---|---|---|---|---|
| mean_logprob | A | B | 0.066 | 0.626 (0.601 to 0.650) | 33/951, 0.035, CP95 0.046 | yes |
| mean_logprob | B | A | 0.063 | 0.715 (0.692 to 0.737) | 82/1116, 0.073, CP95 0.088 | NO |
| min_token_logprob | A | B | 0.060 | 0.587 (0.562 to 0.611) | 20/892, 0.022, CP95 0.032 | yes |
| min_token_logprob | B | A | 0.068 | 0.734 (0.711 to 0.755) | 91/1145, 0.079, CP95 0.094 | NO |
