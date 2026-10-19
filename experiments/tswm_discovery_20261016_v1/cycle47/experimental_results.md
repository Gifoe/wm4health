# C47 — complete joint clock/state objective screen

UTC 2026-10-17T12:35:17.206876+00:00.

Deprioritize the current fixed-feature Laplace / hybrid recipe as a new world-model framework. In this qualified screen it does not improve complete-clock likelihood training: the bistable future-energy differences are adverse in both training seeds, and OU provides no reliable future-quality advantage. The mean-clock restriction loses useful dependence, but a complete ordinary stochastic-clock model already restores it.

Sixteen fresh fits, 21200 updates, sum of per-fit wall times 834.007s (two concurrent CPU workers; not elapsed time or GPU hours). EVAL:54 raw groups,13824 model–origin–policy scores,512 unique physical trajectories and1536 paired history contexts,64 joint futures each. Independent NumPy/SciPy verification PASS: max ES discrepancy 4.08e-07, joint NLL/65 discrepancy 2.06e-06, covariance discrepancy 1.41e-05. All frozen weight/source/data/schedule identities checked.

| Mechanism | Objective | rho | Future ES | Future MSE | Coverage90 | Joint NLL/65 |
|---|---|---:|---:|---:|---:|---:|
| ou | mean_nll | 0.25 | 0.460877 | 0.479085 | 0.8398 | 0.372555 |
| ou | mean_nll | 1 | 0.453204 | 0.460580 | 0.8367 | 1.003210 |
| ou | mean_nll | 2 | 0.444252 | 0.443035 | 0.8403 | 1.571029 |
| ou | full_nll | 0.25 | 0.452006 | 0.462071 | 0.8658 | 0.370760 |
| ou | full_nll | 1 | 0.441846 | 0.438787 | 0.8629 | 0.988227 |
| ou | full_nll | 2 | 0.438715 | 0.434483 | 0.8621 | 1.536626 |
| ou | full_laplace | 0.25 | 0.451584 | 0.461245 | 0.8597 | 0.370924 |
| ou | full_laplace | 1 | 0.442599 | 0.439141 | 0.8518 | 0.989063 |
| ou | full_laplace | 2 | 0.440025 | 0.436310 | 0.8517 | 1.537762 |
| ou | full_hybrid | 0.25 | 0.451381 | 0.461694 | 0.8645 | 0.370931 |
| ou | full_hybrid | 1 | 0.442819 | 0.440312 | 0.8592 | 0.988698 |
| ou | full_hybrid | 2 | 0.439532 | 0.435760 | 0.8589 | 1.537213 |
| ou | known_process_reference | 0.25 | 0.452329 | 0.463734 | 0.8721 | 0.370674 |
| ou | known_process_reference | 1 | 0.443261 | 0.440621 | 0.8719 | 0.988026 |
| ou | known_process_reference | 2 | 0.439126 | 0.434253 | 0.8662 | 1.536242 |
| bistable | mean_nll | 0.25 | 0.528861 | 0.921208 | 0.8892 | 0.350266 |
| bistable | mean_nll | 1 | 0.433707 | 0.549109 | 0.8934 | 0.899556 |
| bistable | mean_nll | 2 | 0.406287 | 0.447803 | 0.8863 | 1.408289 |
| bistable | full_nll | 0.25 | 0.462621 | 0.669092 | 0.8994 | 0.339426 |
| bistable | full_nll | 1 | 0.407532 | 0.453619 | 0.8863 | 0.850259 |
| bistable | full_nll | 2 | 0.390035 | 0.393552 | 0.8917 | 1.314139 |
| bistable | full_laplace | 0.25 | 0.473137 | 0.690480 | 0.9108 | 0.339809 |
| bistable | full_laplace | 1 | 0.418425 | 0.477289 | 0.8936 | 0.851106 |
| bistable | full_laplace | 2 | 0.398407 | 0.407279 | 0.8986 | 1.315390 |
| bistable | full_hybrid | 0.25 | 0.470661 | 0.688373 | 0.9058 | 0.339658 |
| bistable | full_hybrid | 1 | 0.414201 | 0.467634 | 0.8897 | 0.850816 |
| bistable | full_hybrid | 2 | 0.395952 | 0.404383 | 0.8932 | 1.314991 |
| bistable | known_process_reference | 0.25 | 0.462259 | 0.669864 | 0.8735 | 0.339205 |
| bistable | known_process_reference | 1 | 0.402083 | 0.435695 | 0.8684 | 0.849756 |
| bistable | known_process_reference | 2 | 0.385399 | 0.384480 | 0.8701 | 1.313948 |

Lower is better for ES/MSE/NLL. Learned rows average two fits; reference rows use the generating process filtered on the same observed history, never the actual hidden terminal state. First scheduled future is Z32 after32 observed count cells, without an extra transition.

| Mechanism | Candidate vs full NLL | Mean ES difference | Conditional path CI95 | Seed1 / Seed2 |
|---|---|---:|---|---|
| ou | mean_nll | +0.008589 | [+0.001956, +0.015222] | +0.009381 / +0.007796 |
| ou | full_laplace | +0.000547 | [-0.000741, +0.001853] | +0.000733 / +0.000361 |
| ou | full_hybrid | +0.000389 | [-0.000664, +0.001485] | +0.000462 / +0.000315 |
| bistable | mean_nll | +0.036222 | [+0.020034, +0.053442] | +0.038990 / +0.033455 |
| bistable | full_laplace | +0.009927 | [+0.005773, +0.013888] | +0.009676 / +0.010178 |
| bistable | full_hybrid | +0.006875 | [+0.003428, +0.010131] | +0.007031 / +0.006719 |

Positive differences mean worse than full NLL; the table averages three paired density conditions equally. These are descriptive conditional-fit intervals, not a declaration of multi-domain statistical significance.

All four arms use the same 644 trainable parameters, initial weights per seed, TRAIN cohort, held-out observed-VAL criterion and precomputed minibatch schedules. Full NLL integrates the stochastic state with an exact finite-grid forward recursion and includes empty intervals. It is a complete joint-law baseline, not a deterministic or per-time-independent likelihood. No future hidden-state labels enter fitting or VAL selection. HyperNSDE MLP class definitions execute unchanged; no full original-paper reproduction is claimed.

Five of sixteen selections occur at the final update despite the prescribed plateau rule; zero hard caps are hit. This is not proof of universal optimization exhaustion. Laplace-only updates and complete fits are cheaper here, but exact matched-time future-quality frontiers, preprocessing cost and high-dimensional scaling are not measured. Empirical feature means use the entire TRAIN cohort once, unlike raw trajectory minibatches; this is identical data but different statistic reuse, not additional observations. Only selected and last fit checkpoints are retained. Confidence intervals resample 256 physical paths within each carrier while retaining policy/seed pairing; they are conditional on these two fitted initializations, not population uncertainty across retrainings or domains. Nominal90 coverage comes from finite64-draw empirical quantiles; even the known-process reference has sampling/finite-cohort deviation from .90.

Two separately fitted one-dimensional finite-grid Markov carriers, a known Gaussian mark/emission model, known initial prior and known exogenous density multiplier. Not continuous-SDE, learned high-dimensional latent-state, real-domain, history-dependent clock, informative-EOS, unknown observation-policy or causal-intervention validation. Actual physical EVAL paths are shared across density contexts; policy contexts and trained-model scores are not independent new trajectories.

Raw samples, model bundles, configs, optimizer/last states and logs remain on the server. Compact arithmetic, coupling, mechanism diagnostics and CSV tables are mirrored locally. No EVAL-driven hyperparameter adjustment was performed.
