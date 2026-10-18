# Baseline, data and reproduction registry

All C64 models share the generated trajectory information; no future held-out outcome or interventional truth selects a model or setting. Full joint Gaussian and Markov causal AR have different capacity. This is attribution, not a fair claim of new-model performance.

| Entry | Execution and information | Status |
|---|---|---|
| Full joint Gaussian MLE | 48 related prefix fits; 12 independent training corpora; covariance jitter 1e-10 | Completed; all checkpoints hashed |
| Causal linear Gaussian AR | 48 train-only SVD fits, same records and prefix; policy regression only for observational NLL conditioning | Completed; existing causal baseline |
| Known-policy density ratio | Privileged exact logging policy; 96 fixed queries | Completed; invalid precision retained |
| Estimated-policy density ratio | 48 train-only regressions; 96 fixed queries | Completed; invalid precision retained |
| Fixed eigenvalue floor | 1e-3 fixed before scoring, applied only estimated-policy precision | Completed; no tuning |
| Frozen joint temporal factorization | Same full joint covariance; no new data/parameters/fit | 96 rows completed; added after invalid-ratio screen |
| Terminal importance sampling | Oracle Gaussian laws; 16 cells x 3 budgets x 3 MC seeds | 144 rows; no learned score |
| Annealed Gibbs SMC | 32 temperatures, 1 sweep, native FK utility, 4 cells x 2 budgets x 3 seeds | 24 rows; not full TDS |
| Affine DDIM propagation | Observed / denoised-mean plug-in / exact target scores, 64/256/1024 steps | 144 exact propagated Gaussian final-law rows |
| Exact target iid | Privileged statistical floor, same population sizes, 3 MC seeds | 144 post-screen reference rows; not deployable |
| Real nonlinear generative framework | CaTSG/ADMIT/learned sequence diffusion or autoregressive density | Not trained or evaluated in C64 |

Training corpus: each of 4 gains and 3 seeds has 16,384 length-64 training trajectories plus 4,096 independent validation trajectories. Horizons 1/4/16/64 are related prefixes. There are 48 predictive-law fit cells in each of two model classes and 48 policy regressions, not 96 independent datasets or paper reproductions. MLE uses no iterative optimizer.

Environment: Python 3.10.20, NumPy 1.26.4, SciPy 1.15.3, Torch 2.8.0+cu128; CPU only. Generated data, 48 checkpoints and raw traces are stored on G in the authorized project subtree. No original source/data paths were deleted. The upstream optional numba JIT import/decorator is removed through AST for compatibility; original Python algorithm bodies remain intact. Native tilt/resampling/ESS fixtures qualified before execution.
