# Cycle16: complete learned-model relevance gate

The native MTS3 transfer pilot and its final covariance diagnostic are complete. This is one model, one seed, one public series, and validation-only checkpoint selection. It is not original MTS3 paper reproduction, a stable model ranking, or a completed novel world-model method.

Training:1500 fixed optimizer updates, batch16, seed2021, Adam.003, clip5. Native source ab971c0ec43df7664c4b727ca69d5b8dd0cab5da and its configuration are unchanged. The selected checkpoint is step1100 with SHA2564c7f3f139d77afda3bcff8bb68f74ba0dd5cb08365158e65093b5e2fdde83452. Training took3394.504s (56.58min); termination was the predeclared pilot budget. The curve fluctuates and does not prove optimal convergence. The learned forecasts are nontrivial relative to initialization and ordinary references; no gross inability to forecast was observed.

Protocol: NP52416 hourly timestamps; source SHA25629e142d895fa3254bf138e3bec32051e081794a7393a642db36a5d21859d36e1. Train-only target normalization;151 valid history observations,75 future targets; six deterministic calendar conditions. Future raw forecast columns are excluded because issuance-time availability is unverified. Final20% test labels are not evaluated. Validation uses69 origins with disjoint75-step target blocks, allowing historical observations from preceding validation times at later rolling origins. Exact registry/config and every training origin are saved.

Ordinary validation references, with training-only residual/innovation variance:

| Reference | NLL | MAE standardized | RMSE standardized | Marginal95% coverage |
|---|---:|---:|---:|---:|
| persistence | 0.808441 | 0.327788 | 0.489163 | 96.734% |
| seasonal24 | 0.611270 | 0.263598 | 0.391766 | 97.778% |
| Ridge_ARX | 0.341222 | 0.233564 | 0.324779 | 97.778% |

References do not match neural capacity or training compute. No comparison with original WorldTS numbers is permitted because the input-information contract differs.

At the same selected weights and native mean, the final post-hoc covariance controls give:

| Control | NLL | MAE standardized | RMSE standardized | Marginal95% coverage | Mean predicted variance |
|---|---:|---:|---:|---:|---:|
| Native MTS3 | 0.231094 | 0.209789 | 0.295703 | 96.502% | 0.122063 |
| Dense fast; reset cross-scale covariance | 0.276096 | 0.209789 | 0.295703 | 90.899% | 0.061453 |
| Dense fast/context joint | 0.240164 | 0.209789 | 0.295703 | 97.101% | 0.132837 |

Dense-independent versus native changes decoded variance by49.923% on average in absolute relative terms. Dense-joint versus native changes it by9.635%, with maximum29.922%. Joint-minus-native NLL is0.009070; only14 of69 origins have lower NLL after the joint replacement. These are descriptive paired observations, not independent confirmatory tests. The same validation data selected the native checkpoint.

The median learned trace(C S C^T)/trace(Q) is0.358140, so the context covariance channel is not negligible. Both spatial covariance projection and retained cross-scale dependence affect the decoder. Their partial cancellation means the cycle15 scalar witness cannot be promoted directly to a trained undercoverage claim. Learned compensation is a possible explanation, not proven without an appropriate retrained control.

Verification: all source files checked; future observation placeholders cannot change scored forecasts; context factors remain constant and future masked updates leave priors unchanged. Native projected covariance replay error is4.81e-08. Independent NumPy closed-form powers/sums verify both dense trajectories over69x75 predictions within6.66e-16; all metrics match exactly. Initial covariance factors are positive definite in the saved cases. Original arrays, weights, optimizer/RNG states and complete raw training/diagnostic stdout/stderr are retained.

Limits: initial fast/context cross-covariance is assumed0 in both dense controls, not recovered from a true learned joint posterior. Modified covariance summaries may be outside the trained variance decoder's distribution. Neither dense control is a validated posterior correction or a new trained method. No test-set, multi-seed or cross-system generality claim is made.

Figures: [training curve](outputs/analysis/training_curve.png), [covariance diagnostic](outputs/analysis/covariance_diagnostic.png). Machine-readable analysis: outputs/analysis/results.json. Independent receipt: outputs/diagnostics/final_step001100/independent_verification.json.

Decision: no evidence supporting a new covariance-repair architecture. Stop this method line; retain the source finding and negative learned-model evidence. Overall research objective remains active and unachieved.
