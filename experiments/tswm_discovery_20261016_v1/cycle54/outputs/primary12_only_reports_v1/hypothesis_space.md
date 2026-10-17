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
