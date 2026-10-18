# C88 observed compute accounting

| Arm | Parameters | Updates | Observed elapsed seconds |
|---|---:|---:|---:|
| stock | 126932 | 1000 | 568.0665 |
| lagged | 126932 | 1000 | 541.9200 |
| acrkn | 66272 | 1000 | 368.2495 |
| gru | 128864 | 1000 | 5.3129 |

These are actual run timers under a concurrent independent campaign, not isolated GPU throughput benchmarks. Equal updates do not mean equal compute. The ordinary GRU is dramatically faster in this implementation; no efficiency novelty or general model ranking is claimed. Stock/lagged manager-index controls have the same parameter count, initialization and minibatches, which is the decisive comparison for this candidate.

The raw GRU network reacts to arbitrary replacement values at positions flagged missing. The declared preprocessing always zero-fills those positions before inference, so the perturbation is a separate descriptive robustness check, not evidence of future-observation access in the executed task. Native Kalman models ignore those invalid measurements directly.
