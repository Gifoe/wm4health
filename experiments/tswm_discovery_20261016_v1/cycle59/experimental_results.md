# C59 executed results

| History samples | Control | Coefficients | Test MSE / train H variance | Mean row NRMSE |
|---:|---|---:|---:|---:|
|100|static|5|0.118216|0.320014|
|100|play|21|0.0516982|0.174145|
|100|ema|13|0.0633208|0.204736|
|100|hybrid|29|0.0478821|0.167235|
|100|arx|6|0.0584894|0.212098|
|1252|static|5|0.0978898|0.291758|
|1252|play|21|0.0256351|0.135252|
|1252|ema|13|0.0367135|0.173924|
|1252|hybrid|29|0.0161076|0.109978|
|1252|arx|6|0.0617461|0.215846|

All ten stored prediction arrays were independently checked against original raw target rows and re-scored. Maximum discrepancy: 2.22045e-16. Training-only scales, file hashes and split -> selected-model seal -> score receipt -> final result timestamp order passed. Original validation decisions and all candidate scores remain visible. Elapsed baseline script time: 4.531 seconds on the existing CPU runtime, with no neural updates.

Standard play plus exponential memory is useful on this heldout N87 split; adding those known primitives is not a new method. This result neither establishes an unexplained residual nor proves that all neural/state-space frameworks are unnecessary.

One tutorial material, one temperature, 128 raw rows, one group-disjoint 76/25/27 split. No cross-material transfer, neural-seed stability, identified measurement noise floor, reset-history validation or full official trainer reproduction. All raw arrays were inspected for metadata before this screen; predictive test scores were first computed after the ten selected models were sealed. Overall goal remains active / unachieved; accepted final research directions: zero.
