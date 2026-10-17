# C57 complete controlled-sensor audit

Existing small-reference controls pass the fixed repair criterion in these systems: Running .1 centered.

| Sensor | Control | Fits | h20 effect MSE | Seed SD | rho absolute error | Filtered R2 |
|---|---|---:|---:|---:|---:|---:|
|linear|Native same-batch|3|0.0311884|0.00186291|0.0168401|0.862306|
|linear|Independent bank|3|0.159613|1.19222e-07|0.481515|0.68413|
|linear|Running .9 centered|3|0.0200239|0.00154992|0.0177986|0.86276|
|linear|Running .1 centered|3|0.00183539|0.000419122|0.00144091|0.867696|
|linear|Full training bank|3|0.00215571|0.000470504|0.00169932|0.867211|
|linear|Proper likelihood|1|0.00109134|N/A: deterministic|0.000223141|0.868238|
|asinh|Native same-batch|3|0.0308149|0.00177256|0.0168062|0.86256|
|asinh|Independent bank|3|0.159613|5.52649e-08|0.510051|0.690541|
|asinh|Running .9 centered|3|0.0202819|0.00126919|0.0179066|0.862717|
|asinh|Running .1 centered|3|0.00198301|0.000391282|0.00144498|0.867336|
|asinh|Full training bank|3|0.0021312|0.000447654|0.00161106|0.867618|
|asinh|Proper likelihood|1|0.00108067|N/A: deterministic|7.01629e-05|0.86824|

| Control | Nonlinear closure | All 3 seeds improve | Linear degradation | Passes fixed rule |
|---|---:|:---:|---:|:---:|
|Independent bank|-4.3317|False|0.128425|False|
|Running .9 centered|0.3542|True|-0.0111645|False|
|Running .1 centered|0.9697|True|-0.029353|True|
|Full training bank|0.9647|True|-0.0290327|True|

The protocol, numerical amendments and 80% gap-closure rule were fixed before the first heldout opening. This is not an untouched before-training preregistration: the unshifted adaptation failed in training and was explicitly replaced by separately qualified centered controls before heldout access. Every failed log/checkpoint is retained. Valid neural fits total 300,000 updates, with another 10,000 failed adaptation updates. Proper controls are two deterministic fits, not three seed replications.

Primary values are h20 changed-plan conditional-mean effect MSE, averaging zero/one/alternating plans. Neural seed spread and within-model sequence uncertainty are separate. See outputs/results57_v1.csv, outputs/EVALUATION57_V1.json and outputs/SUMMARY57_V1.json for all seeds, horizons and controls. INDEPENDENT_REFERENCE_CAVEAT_V1.md explains why the unsafe independent-bank control cannot establish failure of competent existing methods.

The evidence covers one controlled linear latent system with two paired sensors, a restricted in-model rotation prior and a privileged train-state mean diagnostic. It does not demonstrate arbitrary nonlinear latent dynamics, covariance calibration, causal validity on real data or a novel general framework. The two sensors are not two independent physical systems. No final research direction is accepted.
