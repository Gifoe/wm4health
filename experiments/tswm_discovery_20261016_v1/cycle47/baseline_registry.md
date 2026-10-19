# C47 baseline registry

All four arms use the same 644 trainable parameters, initial weights per seed, TRAIN cohort, held-out observed-VAL criterion and precomputed minibatch schedules. Full NLL integrates the stochastic state with an exact finite-grid forward recursion and includes empty intervals. It is a complete joint-law baseline, not a deterministic or per-time-independent likelihood. No future hidden-state labels enter fitting or VAL selection. HyperNSDE MLP class definitions execute unchanged; no full original-paper reproduction is claimed.

Source: HyperNSDE9d3db33ccfcf79e134d69381c6284b2692a5aef6 (selected class reuse only), TESAR-CDEa57a940310ee877cb9305b588af759d572e2be2e (review only). PCF-GAN, SigTPP and deterministic neural-SDE moment propagation are prior art, not executed headline reproductions. License metadata and selected-file hashes retained; no public redistribution.

| Fit | Updates | Selected update | Observed VAL NLL/65 | Fit seconds |
|---|---:|---:|---:|---:|
| bistable__full_hybrid__seed2026471 | 1600 | 1600 | 0.6841120 | 90.405 |
| bistable__full_hybrid__seed2026472 | 1600 | 1600 | 0.6842487 | 89.017 |
| bistable__full_laplace__seed2026471 | 1600 | 1600 | 0.6844333 | 25.722 |
| bistable__full_laplace__seed2026472 | 1800 | 1800 | 0.6847256 | 28.920 |
| bistable__full_nll__seed2026471 | 1200 | 1000 | 0.6837418 | 49.645 |
| bistable__full_nll__seed2026472 | 1200 | 1000 | 0.6835987 | 49.604 |
| bistable__mean_nll__seed2026471 | 1200 | 1200 | 0.7047385 | 58.337 |
| bistable__mean_nll__seed2026472 | 1400 | 1200 | 0.7049062 | 67.089 |
| ou__full_hybrid__seed2026471 | 1200 | 400 | 0.7910505 | 66.007 |
| ou__full_hybrid__seed2026472 | 1200 | 400 | 0.7909826 | 65.057 |
| ou__full_laplace__seed2026471 | 1200 | 400 | 0.7909237 | 19.392 |
| ou__full_laplace__seed2026472 | 1200 | 400 | 0.7909190 | 19.150 |
| ou__full_nll__seed2026471 | 1200 | 600 | 0.7908974 | 47.057 |
| ou__full_nll__seed2026472 | 1200 | 600 | 0.7908858 | 47.182 |
| ou__mean_nll__seed2026471 | 1200 | 400 | 0.8002208 | 55.666 |
| ou__mean_nll__seed2026472 | 1200 | 600 | 0.8002444 | 55.757 |

Discarded qualification:7 updates ×4 methods ×2 devices =56 updates, excluded from formal totals. Actual CPU median steps were .0427/.0357/.0149/.0514s for mean NLL/full NLL/Laplace/hybrid; CUDA slower in this small grid workload. Process-local affinity/BLAS settings only, no global changes. Five of sixteen selections occur at the final update despite the prescribed plateau rule; zero hard caps are hit. This is not proof of universal optimization exhaustion. Laplace-only updates and complete fits are cheaper here, but exact matched-time future-quality frontiers, preprocessing cost and high-dimensional scaling are not measured. Empirical feature means use the entire TRAIN cohort once, unlike raw trajectory minibatches; this is identical data but different statistic reuse, not additional observations. Only selected and last fit checkpoints are retained. Confidence intervals resample 256 physical paths within each carrier while retaining policy/seed pairing; they are conditional on these two fitted initializations, not population uncertainty across retrainings or domains. Nominal90 coverage comes from finite64-draw empirical quantiles; even the known-process reference has sampling/finite-cohort deviation from .90.
