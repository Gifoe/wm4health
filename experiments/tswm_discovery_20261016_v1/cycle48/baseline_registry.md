# C48 completed baseline registry

UTC 2026-10-17T13:59:36.191934+00:00.16 fresh fits,40750 formal updates;44 raw groups /1694 model-origin-reveal scores from77 chronological origins across2 real processes. Independent CSV/ZIP, windows, model and metric reconstruction PASS. Max model difference 1.49e-06; window and score arithmetic differences0. No hard-cap or final-step selections,0 numerical-refinement flags. Origins are temporally dependent.

Article-linked source model9b573c4bed5efee9fe9c6b765044dbd506b5bf07 / solver4888c63dc84e966f6c6553277adc3851a2cca4fc.98 Git blobs verified. Source MLP/GRU forward bodies and native solver unchanged.
New calendar packing, scalar residual GRU readout, H96/F48 task and optimizer protocol are adaptations. This is cross-benchmark transfer, not headline reproduction. NODE3329/fixedDDE3521/learnedDDE3523/GRU3585 trainable parameters; ridge4752 coefficients.
Two seeds; same allowed history/known clock; shared minibatch schedules per case/seed; fixed/learned DDE identical initial weights. Parameter counts, objectives and effective costs are not exactly matched across all families.

| Fit | Updates | Selected step | VAL MSE |
|---|---:|---:|---:|
| ett__fixed_dde__seed2026481 | 2000 | 250 | 0.1205108 |
| ett__fixed_dde__seed2026482 | 3000 | 2250 | 0.1464131 |
| ett__gru__seed2026481 | 3000 | 1750 | 0.0954641 |
| ett__gru__seed2026482 | 2000 | 250 | 0.0915173 |
| ett__learned_dde__seed2026481 | 2000 | 250 | 0.1141349 |
| ett__learned_dde__seed2026482 | 3000 | 2250 | 0.1421130 |
| ett__node__seed2026481 | 2000 | 250 | 0.0957119 |
| ett__node__seed2026482 | 2000 | 250 | 0.0983964 |
| jena__fixed_dde__seed2026481 | 3000 | 2500 | 0.1222514 |
| jena__fixed_dde__seed2026482 | 2250 | 750 | 0.1276801 |
| jena__gru__seed2026481 | 3000 | 2500 | 0.0970561 |
| jena__gru__seed2026482 | 3000 | 2250 | 0.1007099 |
| jena__learned_dde__seed2026481 | 3000 | 2500 | 0.1239654 |
| jena__learned_dde__seed2026482 | 2250 | 750 | 0.1273056 |
| jena__node__seed2026481 | 3000 | 2500 | 0.1408199 |
| jena__node__seed2026482 | 2250 | 750 | 0.1415038 |

Sum per-fit wall time 5416.796s, not elapsed time or GPU hours. Actual discarded CPU/GPU throughput selected2 CPU workers;24 qualification updates excluded.
Full checkpoints at every250-step validation, selected/last weights, optimizer/RNG, source, data and logs retained on server.
