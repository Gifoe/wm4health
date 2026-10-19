# Completed native discriminator, not a new-method result

All scientific jobs in this bounded audit terminated successfully. Research discovery remains incomplete.

## Protocol

Four unchanged trained checkpoints; 32 origins drawn entirely from the corresponding training region; shifts 1, 8, 24. Exact shared raw patches are verified. Comparing current versus original window statistics isolates a change of representation coordinates. It does not establish lost physical information, forecast degradation, a source bug or a causal failure.

| Dataset | H | Shift | Target relative L2 | Fixed-coordinate max error |
|---|---:|---:|---:|---:|
| NP | 24 | 1 | 0.015465 | 0.0 |
| NP | 24 | 8 | 0.074028 | 0.0 |
| NP | 24 | 24 | 0.187774 | 0.0 |
| NP | 360 | 1 | 0.000395 | 0.0 |
| NP | 360 | 8 | 0.002211 | 0.0 |
| NP | 360 | 24 | 0.005626 | 0.0 |
| PJM | 24 | 1 | 0.005342 | 0.0 |
| PJM | 24 | 8 | 0.042428 | 0.0 |
| PJM | 24 | 24 | 0.105317 | 0.0 |
| PJM | 360 | 1 | 0.000816 | 0.0 |
| PJM | 360 | 8 | 0.005413 | 0.0 |
| PJM | 360 | 24 | 0.014216 | 0.0 |

The condition encoders are audited separately in the machine-readable results. All 24 target/condition fixed-reference comparisons are exactly zero for the saved floating-point tensors. Independent verification covers 12 rows, 124 saved arrays and 128 training origins. No scientific fit was rerun.

## Timing

Eight warm-up calls, then three repetitions of 40 calls; CUDA events and synchronized wall time both saved.

| Dataset | H | Batch | Full forecast ms | Separate encoder+stats ms | Ratio |
|---|---:|---:|---:|---:|---:|
| NP | 24 | 1 | 4.505 | 1.079 | 0.239 |
| NP | 24 | 8 | 4.481 | 1.069 | 0.238 |
| NP | 360 | 1 | 4.594 | 1.055 | 0.230 |
| NP | 360 | 8 | 4.688 | 1.077 | 0.230 |
| PJM | 24 | 1 | 4.560 | 1.092 | 0.239 |
| PJM | 24 | 8 | 4.865 | 1.076 | 0.221 |
| PJM | 360 | 1 | 4.620 | 1.060 | 0.229 |
| PJM | 360 | 8 | 5.668 | 1.083 | 0.191 |

The separately timed encoder+stats component is approximately 19%-24% of full wall time. This establishes limited current component cost, not a measured encoder-removal speedup. The helper omits native finite-value validation synchronization, and separate calls are not an additive execution decomposition. Consequently the raw legacy field `encoder_elimination_ideal_speedup_upper_bound` must not be interpreted as a rigorous bound. No actual cached forecast or transport method was benchmarked. All full forecasts are about 4.5-5.7 ms in this specific environment; these measurements are not general hardware guarantees.

## Evidence integrity and limitations

Checkpoint/config/data SHA256 and complete before/after tensor fingerprints match the prior registry. Executed source bytes are archived and checked. NumPy independently reconstructs chart differences and statistics with metric tolerance 2e-6. This verifier does not independently replay checkpoint forecasts on CPU. There is no new held-out forecast score, trained method, multi-seed efficacy comparison, causal benchmark, or mechanism-uncertainty experiment in cycle 14.
