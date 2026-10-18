# C61 complete evidence

| Family | Existing model | Fits | Conditional joint NLL/step | NLL above known law | Remaining-life probability MSE | Brier |
|---|---|---:|---:|---:|---:|---:|
|peaked|hsmm|3/3|1.418537|0.007370878|0.0002720672|0.04770211|
|peaked|hmm|3/3|1.52174|0.1105733|0.08988824|0.1390457|
|bimodal|hsmm|3/3|1.427477|0.006737566|0.0006568929|0.0787841|
|bimodal|hmm|3/3|1.502124|0.08138425|0.06479209|0.1422987|

The learned existing HSMM passes the predeclared narrow sufficiency screen in every fitted seed on both families. Generic residual-life belief is not supported as a new-method necessity here.

Valid fixed-budget optimizer updates: 18000; separate profiling updates: 12. Known-law reference parameters and latent labels are never provided to training or validation selection. Primary joint future observation NLL and secondary remaining-life probability/Brier arithmetic are independently verified from saved arrays. Native versus independent probability-space recursion maximum conditional-NLL discrepancy: 9.3258734e-15; posterior discrepancy: 1.4321877e-14; survival discrepancy: 1.4321877e-14. Posterior inference receives only its observed prefix. Future observations enter a separate proper density score.

Bootstrap intervals resample 128 independent generated paths while keeping each path's three history lengths and five survival horizons together, conditioning on the three fixed model seeds. Seed SD is reported separately in JSON. Model seal precedes test opening and no test values choose a new fit or recipe. No model instability or missing fit is silently averaged away.

Two synthetic finite-duration Gaussian-emission families, fixed known state count/support, fresh known episode boundaries, independent records and unknown fitted law parameters. No real domain, unknown observation model, unknown regime count, arbitrary left censoring, cross-system pretrained checkpoint or new architecture is evaluated. The term physical_test_records in the machine evaluation denotes independent generated paths; it does not mean real measurements.
