# DCL native core qualification: latest source-pinned NumPy run

The latest run uses actual imported NumPy1.26.4, typeguard2.13.3, jaxtyping0.2.38 and existing Torch2.8.0+cu128 in the separate project venv. It passes the numerical and GPU checks. There are64 discarded native optimizer updates in total:32 initial qualification updates with inherited NumPy2.2.6, followed by32 with the source-required NumPy1.26.4. Both runs, weights and fixtures are preserved. These are environment qualification runs, not scientific replicates. Formal DCL fits remain0.

| Latest independent comparison | Maximum absolute error |
|---|---:|
|cpu loss versus NumPy|1.77636e-15|
|cpu reference gradient versus analytic NumPy derivative|2.498e-16|
|cuda loss versus NumPy|1.77636e-15|
|cuda reference gradient versus analytic NumPy derivative|4.30211e-16|
|Linear prediction versus NumPy|0|
|Reciprocal-MSE switching logits versus NumPy|0|

The native fitter executed32 updates on NVIDIA GeForce RTX 5090 in 0.829060s for the small toy fixture. All2048 sampled pairs stayed within trial boundaries and advanced one point. Actual layers remain5->90->90->30->3; total parameters11562. Raw losses are finite and weights change, but this short toy run does not test recovery, forecasting or calibration. No heldout C54 data, released paper signal files, database service or external tracking was used.

Runtime provenance is based on actual module paths. The earlier distribution-inventory dictionary can hide duplicate base/venv versions; it is retained as historical metadata and is not used to assert the active version. The base NumPy source hash is unchanged, and the newly installed copy resolves inside .venv_dcl. Neither global packages nor the active RP/JAX environment were changed. The earlier V1 report is a frozen32-update history, superseded for current totals by this report. The unexecuted qualifier v1, the initial executed v2 and source-NumPy executed v3 are retained separately.

All53 native core/license/configuration source hashes match before and after both runs. Service implementations and the full paper sweep are outside the qualification scope. The switching-target input demonstration confirms retrospective identification; it establishes neither a causal forecast failure nor a new contribution. The immediate next scientific decision still waits for the16 fixed C54 fits and their guarded evaluation.

Initial qualification SHA256: b23ac45f934827dc5c5a2386273ba9ec9dfd989fa43899ddffde8632ef847b2c

Latest qualification SHA256: dd1362f29281d13e18104fc5b6638fe22392305d07a51824d1be66e1fc3d193a

Runtime amendment SHA256: 90688727e68b70aec40852a53a1051e5faf1ebab25166083a6caaf10c82054ad
