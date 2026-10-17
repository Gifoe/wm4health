# C54: Jointly learned RP-GSSM linear audit

Status: all 12 fixed-budget fits (RP native; ordinary LGSSM EM with the explicitly qualified initial-covariance normalization correction), training-only state maps, causal heldout scoring and independent numerical checks completed. This is a paper-recipe reconstruction using pinned native components; it is not the original full hyperparameter sweep or visual benchmark reproduction.

| Hypothesis | Discriminating evidence | Current scope |
|---|---|---|
| A learned representation necessarily lacks usable transition structure | Learned native A, eigenvalues and causal forecasts; compare existing EM and oracle | Cannot be asserted from aggregate decoding scores |
| A weak long forecast means the model forgot the mechanism | Conditional-mean discrepancy versus irreducible oracle MSE | Factual and conditional errors separated |
| An apparent success comes from fitting state maps to heldout labels | Immutable training-only readout seal before heldout scoring | Training-only maps used for all forecasts |
| Native Gaussian numerical error causes the result | Independent NumPy KF/RTS with explicit source jitter and matrix-power checks | Numerical checks passed; does not validate RPM approximation |
| Two linear tasks establish a reusable new world-model research gap | Requires distinct nonlinear/real evidence and close existing controls | Rejected: current evidence is insufficient |

| Task | Model | Full smoothing R2, train-only map | Mean eigenvalue error | h=1 state MSE | h=10 state MSE | h=50 state MSE |
|---|---|---:|---:|---:|---:|---:|
|linear_small|rp|0.933015 +/- 0.001018|0.029943|0.188494|0.741257|1.047827|
|linear_small|em|0.960674 +/- 0.000000|0.000769|0.168859|0.697773|0.957503|
|linear_medium|rp|0.744632 +/- 0.007335|0.461281|0.390011|0.867067|1.019076|
|linear_medium|em|0.965215 +/- 0.000000|0.002634|0.170120|0.644126|0.942786|


## Existing native EM option: four matched learning controls

| Task | Seed | Original False R2 | Existing True R2 | False eigenvalue error | True eigenvalue error | h10 conditional-mean error delta (True-False) |
|---|---:|---:|---:|---:|---:|---:|
|linear_small|0|0.933891|0.764874|0.0289171|0.358336|0.811175|
|linear_small|1|0.933256|0.682439|0.029567|0.212945|0.391376|
|linear_medium|0|0.736170|0.531683|0.520065|0.501034|0.0121804|
|linear_medium|1|0.748544|0.523179|0.425786|0.499489|0.0370148|

The existing True option gives numerically lower h10 conditional-mean discrepancy in 0/4 initialization-matched pairs and lower transition-eigenvalue error in 1/4. These counts describe the four fixed comparisons, without significance or independent-domain claims.

The original12-fit comparison is preserved in full, including its third False initialization for each task. Four additional native True runs use the same data, model, initial arrays, Adam/batch RNG, stabilization and 10,000-update terminal budget; only the already-provided posterior-gradient option changes. This is a source-option control, not a new method. All16 models and all16 diagnostic maps were frozen before the first heldout score. The extra maps were prepared first, followed by the original12-map preparation and scoring, then extra scoring.

The frozen-gradient reference dependence is configuration-specific: differentiating through the approximate E-step and taking a fixed-posterior M-step are different derivative contracts. Its presence under the public default cannot alone establish that the original paper or all existing world models lack a capability. Encoder reference dependence and proper-RPM approximation remain separate questions. A scalar gradient improvement is not automatically a forecasting or uncertainty-calibration improvement. See the complete paired horizon metrics in `outputs/em_option_learning_v1/evaluation/summary.json` and the training-only audit receipts.

No new research direction is accepted from these two synthetic tasks. If the existing option reduces a defining error, a new-primitive necessity claim is weakened. Any remaining failure must survive competent source settings and proper-likelihood controls on independent task evidence before method design. The current data do not establish broad real-domain validity or a complete probabilistic calibration result.
