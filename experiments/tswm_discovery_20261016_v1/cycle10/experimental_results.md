# Cycle10 experimental evidence

All scientific jobs ran on the user's private Windows server. These are controlled source transfers, not a reproduction of official benchmark scores. No neural model or new method is trained here. Raw configs/logs/arrays/results are retained.

The unmodified pinned DoTime TemporalSCM, TemporalDAG and mechanism classes were instantiated with explicitly supplied linear parameters. The alternative permits cross-lags, outside the original diagonal-lag class except at zero scale. Five lag scales x three independent noise seeds =15 native paired runs. Full observed-process equality follows from the rational identities in `outputs/rational_proof.json`, including independent equal-variance Gaussian structural noise. Native float32 parameters in float64 buffers give maximum observed path difference 3.60736789e-07; direct float64 recurrence maximum 6.66133815e-16. Maximum independent-frequency spectral error 3.83029458e-15. Both systems are stable and no clipping occurs. Onset response is +0.6 vs -0.4 in every run.

## Existing diagonal-lag covariance-fit control

12 starts per fixed slope and scale;120 total fits. The true-slope warm start fits exactly. Opposite-slope local fits yield:

| lag scale | normalized covariance residual L2 | holdout spectral max error |
|---|---:|---:|
| 0 | 1.8238834e-20 | 4.40205916e-23 |
| 0.001 | 0.000526421615 | 8.71879796e-05 |
| 0.01 | 0.00526463109 | 0.000872040384 |
| 0.1 | 0.0530610064 | 0.00887595391 |
| 1 | 1.10565803 | 0.26584469 |

Nonzero local residual is not a global identification certificate. Zero-scale numerical equality is consistent with the exact iid witness. No claim is made that positive scales identify the effect in the original class.

## Additional-information randomized control

OLS on independent one-step hard-do A=+/-1 experiments, with stationary initial states. These trials contain extra intervention labels; they are not an input-matched predictive-model improvement. The table pools five lag scales per each of40 seeds.200 rows per cell are correlated within seed block; no200-independent-seed significance claim is valid.

| world | N per trial | correct sign | 95% interval coverage | interval excludes0 | slope RMSE |
|---|---:|---:|---:|---:|---:|
| 0 | 8 | 0.995 | 0.990 | 0.820 | 0.159325 |
| 0 | 32 | 1.000 | 0.935 | 1.000 | 0.083410 |
| 0 | 128 | 1.000 | 0.940 | 1.000 | 0.043358 |
| 1 | 8 | 0.955 | 0.960 | 0.440 | 0.200464 |
| 1 | 32 | 1.000 | 0.965 | 0.980 | 0.087298 |
| 1 | 128 | 1.000 | 0.940 | 1.000 | 0.047726 |

1200 control trials in total. The N=32 sign result is100% on this fixed finite design. It is not a general sample-complexity guarantee. Native witness elapsed 13.714s; follow-up controls 3.667s, excluding transport/verification. No GPU-hour performance comparison is claimed.
