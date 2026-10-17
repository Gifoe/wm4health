# C57 experimental results

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

Existing small-reference controls pass the fixed repair criterion in these systems: Running .1 centered.

All 32 model/mean-map hashes were sealed before heldout opening; terminal model -> train-only map -> heldout receipt -> scoring timestamp ordering is verified. Primary values were independently recomputed from every prediction array (maximum discrepancy 2.22e-16). The heldout set has 256 independent simulated sequences; the physical process/data draw is fixed. Paired sequence intervals condition on each fitted model, while the table's SD is across the three neural initializations. There is no validation or test checkpoint selection and no best-seed reporting.

The proper control receives the same prior structure and a small parametric invertible-sensor family, learns its parameters without state labels, and uses full-data float64 L-BFGS. The five neural controls share matched initial tensors, architecture, Adam step budget, data and training precision. Independent/running variants encode 64 sequences per update, native 32 and full reference 544; total compute is not silently equated. Their runtime and peak allocation are in the CSV. The proper and auxiliary losses have different units/definitions and are not ranked as a common objective.

The evidence covers one controlled linear latent system with two paired sensors, a restricted in-model rotation prior and a privileged train-state mean diagnostic. It does not demonstrate arbitrary nonlinear latent dynamics, covariance calibration, causal validity on real data or a novel general framework. The two sensors are not two independent physical systems. No final research direction is accepted.
