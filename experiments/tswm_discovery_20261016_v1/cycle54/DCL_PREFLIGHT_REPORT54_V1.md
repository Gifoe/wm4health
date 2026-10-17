# Native DCL core: qualified executable, no formal benchmark result

The pinned53-file native core passed independent numerical checks and32 discarded native solver updates on NVIDIA GeForce RTX 5090 with Torch2.8.0+cu128. Toy fitting took 1.142935s; peak allocated CUDA memory was 18,900,480bytes. These are small-fixture resource measurements, not estimates for the paper sweep.

| Independent comparison | Maximum absolute error |
|---|---:|
|cpu native full-denominator loss versus NumPy|1.77636e-15|
|cpu reference gradient versus analytic NumPy derivative|2.498e-16|
|cuda native full-denominator loss versus NumPy|1.77636e-15|
|cuda reference gradient versus analytic NumPy derivative|4.30211e-16|
|Native linear prediction versus NumPy|4.44089e-16|
|Native reciprocal-MSE switching logits versus NumPy|0|

All2048 actual training pairs respected trial boundaries and advanced by one time point. A tracing-only loader subclass returned native batches unchanged. The actual encoder layers were5->90->90->30->3, with 11562 total encoder/dynamics parameters. The native Offset1ModelMLP ignores hidden_dim/num_layers configuration fields; no replacement architecture was introduced.

Changing only the supplied next target changes the preferred native switching mode for all7 arithmetic-fixture examples. This verifies the documented retrospective interface; it is not evidence of poor causal forecasting or an accusation of leakage in the original identification task. Existing Markov switching and ordinary HMM/SLDS mechanisms remain required controls.

Execution provenance: direct raw-GitHub requests from the server failed through both Python and native Windows clients. The connected GitHub service supplied the pinned public source through a RAM-only relay; all53 Git blobs were verified before use. No local result file was written. A separate project venv installed only lightweight typing dependencies over the existing shared Torch runtime. The RP/JAX environment, global packages and drivers were not changed. The downloaded public DCL catalog remains metadata-only; no released signal file or database service was used.

The earlier qualifier v1 was never executed; v2 tightened pair-index validation to cover all indices regardless of trailing dimensions before the single successful run. There were32 discarded updates in total,0 formal DCL fits,0 paper metrics and0 accepted research directions from this work. The32 finite toy losses and parameter changes verify execution only. They do not establish representation recovery, forecast quality, calibration or a method benefit. Runtime preparation, source/package versions, protocol, raw qualification receipt, and server artifact hashes are retained.

Next decision: await the existing16-model C54 comparison. DCL is now an executable alternative if its input and task contract makes a subsequent failure attribution informative. Do not launch a full sweep merely because the core runs.

Raw qualification summary SHA256: b23ac45f934827dc5c5a2386273ba9ec9dfd989fa43899ddffde8632ef847b2c
