# experimental results

| Method | Closed field median | Open field median | Paired open/closed median | At least 2x worse | Controlled-rollout median | Complete rollouts |
|---|---:|---:|---:|---:|---:|---:|
| FIM_local2 | 0.05758045 | 0.04967039 | 0.828843 | 0/8 | 0.06425606 | 32/32 |
| FIM_full3_diagnostic | 0.16907955 | 0.09482413 | 0.640107 | 0/8 | 0.11716926 | 32/32 |
| weak_polynomial3 | 0.00027483 | 0.00025270 | 1.102993 | 0/8 | 0.00013280 | 32/32 |
| weak_rff128 | 0.00982537 | 0.00382364 | 0.381719 | 0/8 | 0.02481246 | 32/32 |

All field cases, candidate coefficients and every rollout are in outputs/RESULTS82_V1.json and the complete audit. No filtered success-only comparison is used.
