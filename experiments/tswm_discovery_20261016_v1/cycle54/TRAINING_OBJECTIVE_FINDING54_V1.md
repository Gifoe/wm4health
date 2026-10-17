# Training-only reference and EM-contract audit

All quantities below use two completed native default checkpoints on the small task and training observations only. No true-state labels, heldout observations, optimizer updates or model-selection changes were used. The original12 fitting protocol remains intact, and four matched existing-source EM-option learning controls are separately declared before heldout scoring.

| Seed | Full-bank total dL/dalpha | Balanced batch32 total | EM full-bank partial | EM balanced batch32 partial | EM gradient discrepancy |
|---|---:|---:|---:|---:|---:|
|0|1.847127|-0.541775|2.559436|2.559436|4.62e-14|
|1|2.255522|-0.220866|3.015146|3.015146|6.71e-14|

The same frozen parameters give opposite total-derivative directions when the target batch is also the reference bank. Equal target frequencies rule out target-averaging noise for the balanced control. The per-target decomposition matches the native loss and A gradient, and the balanced full-reference average matches the full gradient to numerical precision.

| Seed | Estimator | Mean dL/dalpha | Batch SD | IID MC SE when applicable |
|---|---|---:|---:|---:|
|0|native_coupled|-0.723420|0.366960|0.011468|
|0|leave_designated_diagonal_out|2.323412|0.608494|0.019015|
|0|independent_reference|2.352293|0.629217|0.019663|
|0|oracle_frozen_normalizer|0.101254|94.104571|2.940768|
|0|running_normalizer_gamma09|-62.191617|20.879774|not iid|
|0|running_normalizer_gamma01|-28.264650|74.659497|not iid|
|1|native_coupled|-0.410722|0.386921|0.012091|
|1|leave_designated_diagonal_out|2.811850|0.640184|0.020006|
|1|independent_reference|2.841279|0.664735|0.020773|
|1|oracle_frozen_normalizer|0.312625|97.586764|3.049586|
|1|running_normalizer_gamma09|-64.207936|21.714986|not iid|
|1|running_normalizer_gamma01|-28.968817|77.473771|not iid|

Dropping only the designated diagonal or using independent references reverses the scalar direction at these frozen points, but neither is demonstrated to recover the exact full-bank gradient or improve a trained model. The oracle-normalizer estimator has large conditional MC variance; its observed mean deviation is not evidence of systematic bias given its reported SE. The two running-normalizer adaptations do not recover the full-bank derivative here. They are fixed-parameter controls, not an official SogCLR reproduction or a test of its convergence while model parameters change. All1,024 recorded batch estimates per model are exported as CSV; caches and raw array hashes remain on the server.

A simpler source-provided control is decisive for the derivative attribution: `em=True` holds the inferred posterior fixed in the M-step. It leaves the forward loss value unchanged while changing a total derivative into a partial derivative. At these points, the A-gradient reference dependence disappears. An independently held-posterior M-step matches that partial derivative. Central finite differences near the covariance boundary require step halving; the original h=1e-5 failure and the full convergence grid are retained. A finite difference that recomputes the posterior would test the wrong derivative contract.

This is not proof of calibrated dynamics or of a new world-model gap. The encoder can still depend on the reference construction, and the Gaussian auxiliary approximation remains distinct from proper RPM likelihood. The already-provided EM option must be tested through matched complete learning before any new mechanism is designed. The added four fits preserve model, data, initial parameter arrays, batch RNG, Adam, stabilization and10,000-update terminal budget. All16 models/maps will be frozen before the first heldout comparison.

Current research judgment: reject the argument that the observed default A-gradient effect alone requires a new architecture. Retain a narrower unresolved question about how inference/optimization contracts affect reusable dynamics learning. Forecast consequences, uncertainty calibration, independent-data/generalization evidence and a useful method advantage remain unproven. Accepted final research directions:0.

See `LITERATURE_REFERENCE_BOUNDARY54_V1.md` for existing gradient-bias, normalization, nonlinear system-identification and negative-sampling work. The known components are not renamed as an invention.
