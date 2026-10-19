# Experimental results — working cycle16

This cycle is not sealed. Native MTS3 training remains active; final checkpoint selection and full validation covariance diagnostic are pending. Scientific objective remains unachieved.

The successful preflight confirms native full forward/backward finite values, and zero forecast changes when future masked observation placeholders are perturbed. Correct alignment uses151 valid historical observations and source outputs150:225 for75 future targets. Native abstract action encoding uses the whole planned next slow window; the last condition is not unused padding. The earlier failed alignment preflight is retained.

Data have52416 hourly timestamps. Only training/validation targets are exported; target standardization uses training rows. Validation targets are disjoint75-step blocks; prior validation observations may be available at later rolling origins. Source SHA25629e142d895fa3254bf138e3bec32051e081794a7393a642db36a5d21859d36e1. Exact boundaries and origins are in outputs/data/registry.json.

Completed ordinary validation references:

| Model | Gaussian NLL | MAE standardized | RMSE standardized | Marginal95% coverage |
|---|---:|---:|---:|---:|
| Persistence | .808441 | .327788 | .489163 | .967343 |
| Seasonal24 | .611270 | .263598 | .391766 | .977778 |
| Ridge ARX | .341222 | .233564 | .324779 | .977778 |

Native initialized NLL.900980; step100 NLL.406876,MAE.209368,RMSE.298935,coverage.987053. These interim values do not establish convergence or stable model ranking. The frozen training protocol, not these interim observations, determines stopping and checkpoint selection. Complete validation rows and per-update sampled origins are logged on the server.

The fixed step100 diagnostic smoke uses only the first8 existing validation origins. Independent scalar recurrence checks have maximum absolute error9.09e-13; replay of the native learned projected trajectory differs by at most6.29e-9. Context constancy, no-update future masks, unchanged decoded means and horizon1 agreement of the dense controls all pass. An independent NumPy verifier uses closed-form powers and sums instead of the recursive implementation: both dense trajectories agree within1.39e-16 and recomputed metrics agree exactly. Its small slice is an implementation check, not the full scientific comparison. See outputs/diagnostics/smoke_step000100/results.json and independent_verification.json. Final diagnostic must wait for training completion and use all validation origins.

The covariance gate holds the native mean fixed. Any NLL/coverage change is a post-hoc response of a decoder trained under another covariance recursion; it is not a validated posterior correction. A trace ratio describes one latent coordinate system and covariance channel, not causal importance of the context mean or a coordinate-free epistemic quantity.
