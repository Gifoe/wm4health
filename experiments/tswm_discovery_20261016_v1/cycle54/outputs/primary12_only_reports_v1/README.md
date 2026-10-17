# C54: Jointly learned RP-GSSM linear audit

Status: all 12 fixed-budget fits (RP native; ordinary LGSSM EM with the explicitly qualified initial-covariance normalization correction), training-only state maps, causal heldout scoring and independent numerical checks completed. This is a paper-recipe reconstruction using pinned native components; it is not the original full hyperparameter sweep or visual benchmark reproduction.

| Task | Model | Full smoothing R2, train-only map | Mean eigenvalue error | h=1 state MSE | h=10 state MSE | h=50 state MSE |
|---|---|---:|---:|---:|---:|---:|
|linear_small|rp|0.933015 +/- 0.001018|0.029943|0.188494|0.741257|1.047827|
|linear_small|em|0.960674 +/- 0.000000|0.000769|0.168859|0.697773|0.957503|
|linear_medium|rp|0.744632 +/- 0.007335|0.461281|0.390011|0.867067|1.019076|
|linear_medium|em|0.965215 +/- 0.000000|0.002634|0.170120|0.644126|0.942786|

| Task | Model | h=1 conditional-mean discrepancy | h=10 discrepancy | h=50 discrepancy |
|---|---|---:|---:|---:|
|linear_small|rp|0.0322689|0.0715555|0.073247|
|linear_small|em|3.4704e-05|3.23437e-05|2.00214e-05|
|linear_medium|rp|0.248114|0.199638|0.0972055|
|linear_medium|em|0.000188373|0.000844449|0.000463326|

| Task | h=1 oracle state MSE | h=10 | h=50 |
|---|---:|---:|---:|
|linear_small|0.168970|0.696734|0.957262|
|linear_medium|0.170530|0.645280|0.945543|

The ordinary EM baseline uses the same native E-step, regression updates and 200-iteration fit loop, with only initial covariance corrected to sum_Ezz/N - outer(sum_Ez/N,sum_Ez/N). The uncorrected seed0 first M-step was indefinite; two failed 200-iteration attempts were excluded and retained. Independent training-only qualification verified the corrected moments, all other parameters identical, and increasing finite likelihood. The protocol amendment predates heldout/state-readout evaluation. No upstream package/source was edited. See plan54_em_amendment_v2.json and outputs/QUALIFICATION_EM_FIX54_V1.json.

The three seeds vary initialization on one fixed dataset per dimension; they are not independent domains or independent data draws. All state labels enter only post-training diagnostic maps. Full-sequence smoothing uses future observations relative to individual times; causal prediction uses only the first 50 observations. Auxiliary observation maps use training observations alone and are applied equally to both learned models. Long-horizon factual MSE includes irreducible process noise; conditional-mean discrepancy measures deviation from the oracle conditional expectation. No significance or broad generalization claim is made.

All source files retain their pinned SHA256. Independent NumPy Kalman/RTS checks account explicitly for the RP source's fixed 1e-5 I addition to reported smoothed covariances. This does not equate the Gaussian auxiliary surrogate with the proper normalized RPM posterior. All model and readout checkpoint hashes are unchanged after evaluation.

RP optimizer updates: 60000; native EM iterations: 1200; sum of fit wall times: 11222.150s. No GPU training is claimed. Independent posterior arithmetic maximum absolute error after the explicitly documented native covariance correction: 4.28e-09.

Raw arrays and checkpoints remain on server storage. `outputs/metrics54.csv`, `outputs/evaluation54.json`, `outputs/verification54.json` and immutable model/readout seals provide the compact evidence.

Execution contract: see `PROTOCOL.md` and `REPRODUCE.md`.
