# C61: learned existing duration controls

The learned existing HSMM passes the predeclared narrow sufficiency screen in every fitted seed on both families. Generic residual-life belief is not supported as a new-method necessity here.

| Family | Existing model | Fits | Conditional joint NLL/step | NLL above known law | Remaining-life probability MSE | Brier |
|---|---|---:|---:|---:|---:|---:|
|peaked|hsmm|3/3|1.418537|0.007370878|0.0002720672|0.04770211|
|peaked|hmm|3/3|1.52174|0.1105733|0.08988824|0.1390457|
|bimodal|hsmm|3/3|1.427477|0.006737566|0.0006568929|0.0787841|
|bimodal|hmm|3/3|1.502124|0.08138425|0.06479209|0.1422987|

Every seed and source/hash/config is retained. See outputs/results61_v1.csv, SUMMARY61_V1.json and EVALUATION61_V1.json. Overall goal remains active and unachieved; accepted final directions = 0.

Two synthetic finite-duration Gaussian-emission families, fixed known state count/support, fresh known episode boundaries, independent records and unknown fitted law parameters. No real domain, unknown observation model, unknown regime count, arbitrary left censoring, cross-system pretrained checkpoint or new architecture is evaluated. The term physical_test_records in the machine evaluation denotes independent generated paths; it does not mean real measurements.
