# Calibration and information scope

This records an existing protocol fact; it changes no data, model, training, split, metric, or queued source lock.

Synthetic signals use each system's separate4096-step reference trajectory (after the declared burn-in) to define fixed input/output units. For WH, the reference also defines the fixed internal nonlinearity scaling that is part of the generated plant. It is held fixed for every future branch. This avoids the original query-dependent normalization problem, but the resulting task is explicitly simulation in provided fixed-reference units. Models receive numerical signals in these units, rather than raw signals from an entirely uncalibrated physical plant. The scalar calibration statistics are not additional tokens, and no query future is used to estimate them.

Consequently, C128 is the model's per-query observed-history length, not a claim that the entire experimental construction or physical calibration used only128 observations. Independent reference generation and its physical information cost remain part of the data protocol. Report the fixed-reference family definition and source transformation when comparing papers. Do not use this experiment to claim raw physical-parameter identification, system-independent absolute units, or a total128-sample data budget.

For measured Silverbox/CascadedTanks, fixed means/stds are estimated from the official training recording only, and supplied uniformly to the frozen models and classical controls. Real neural weights are not fitted there, but scale calibration still uses that training recording. A zero-shot weight-transfer result here therefore means no neural target-data finetuning under the supplied unit calibration, not absence of any target-system training-record information. The external ICL additionally applies its declared context-only transform; this supplies no new observations.

The secondary finite-plan objective uses the benchmark's supplied output/input units consistently for predictions, true costs, and all methods. Its targets-1/0/+1 and action-cost coefficient0.01 are illustrative standardized engineering objectives. They are not universal physical setpoints or a comparison across raw physical units.

These qualifications restrict interpretation, not the validity of the same-information within-protocol architectural comparisons. Preserve all samples, coefficients, reference metadata and results; no protocol correction or training restart is implied.
