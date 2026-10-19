# Persistent-context marginalization discriminator

## Controlled model

For fast state z and persistent context l, z_next = a*z + c*l + epsilon. Start with independent zero-mean state variance.05 and context variance s; process variance.01. Context is sampled once for a rollout. The source-native approximation adds c^2*s every step without carrying fast/context cross-covariance.

At horizon h, exact persistent marginal variance is a^(2h)*P0 + c^2*(sum_j a^j)^2*s + q*sum_j a^(2j). Independently resampling the context gives a^(2h)*P0 + (c^2*s+q)*sum_j a^(2j). Standard augmented-state Gaussian propagation reproduces the former. All means agree in this zero-mean example; only uncertainty differs.

## Native evidence

90 prescribed cell/branch rollouts,7 inspected horizons each:630 rows. Native and fresh-context marginal formulas agree to 4.26e-14; augmented and persistent formulas to 4.55e-13. There are438 one-step or negative-control rows. Negative a tests cancellation and can yield overdispersion; failure is not universally undercoverage.

| a | h | Native variance | Persistent variance | Actual coverage of native nominal95% interval |
|---:|---:|---:|---:|---:|
| 1 | 64 | 32.690000 | 2048.690000 | 19.554% |
| 0.95 | 64 | 5.223474 | 185.374447 | 25.785% |
| -0.8 | 64 | 1.416667 | 0.182099 | 100.000% |
| 0 | 64 | 0.510000 | 0.510000 | 95.000% |

These are analytic Gaussian marginal coverages for prescribed native operator parameters, not percentages from trained models on a dataset. a=0 is only a marginal-variance negative control; temporal dependence may still differ across processes. No unique full joint law is inferred just from matching marginal moments.

Independent simulations:6 preselected cases x50,000 particles,42 horizon rows. All168 simultaneous coordinate/family/horizon sample-variance confidence intervals contain the known variance, using Bonferroni familywise.05. Saved endpoints independently reproduce variances and coverage statistics. This is verification, not test-set tuning.

## Conventional-control cost

RTX5090, float32/eager, TF32 disabled,32 open-loop steps. Warm-up4, three repeats of5 rollouts. CUDA event times and synchronized wall times are both saved.

| Latent observation dim | Batch | Native eager ms/32 steps | Dense augmented ms/32 steps | Augmented peak increment MiB |
|---:|---:|---:|---:|---:|
| 1 | 1 | 54.777 | 3.004 | 0.002 |
| 1 | 32 | 54.650 | 3.482 | 0.006 |
| 30 | 1 | 55.309 | 3.409 | 0.166 |
| 30 | 32 | 61.094 | 3.335 | 5.288 |
| 128 | 1 | 58.439 | 3.463 | 3.002 |
| 128 | 32 | 61.460 | 13.525 | 96.062 |

At the native d30 configuration, full augmented dimension is120. The ordinary joint method is practical in this inference test. Relative timings compare current eager code paths with different retained information, not equally optimized algorithms. The native path recomputes several constant transforms and uses more small operations; ordinary hoisting, compilation or kernel fusion could change its relative time. These results do not establish a general algorithmic speedup. Peak allocated memory increments are inference-only, not training/backpropagation cost.

## What remains unproved

No trained native model, learned context-channel relevance, real-data interval coverage, decoded joint-path calibration, training memory/throughput, arbitrary nonlinear mechanism or cross-dataset improvement has been evaluated. A learned model may reduce context coupling or encode persistent information in its fast state. The operator witness alone cannot establish an important failure of the complete learned model.
