# Ordinary-control factorial attribution, frozen after the weak screen

Weak-form fitting did not repair the observed chemical dynamics/joint failures on either original seed. Before interpreting this as a new mechanism, test two remaining ordinary explanations: calibration support/trajectory length, and unnecessary changes to a transferable source field.

This is a post-screen attribution experiment for seeds2026851/2026852. Seed2026853 is newly generated after this protocol is fixed. No neural method or old model selection is changed. No proposed method or publication claim follows from this control.

Factor A: same-budget calibration allocation. Short uses the original4x64 TRAIN and2x64 VAL transitions. Long keeps the first two original TRAIN paths and extends each from64 to128 steps, dropping the other two; VAL keeps the first original path and extends it to128, dropping the second. Thus TRAIN remains256 and VAL128 transitions. Preserve every retained prefix's states, controls and observation noise exactly; extend commands using new independent namespaces. The eight static calibration pairs remain identical and available, but are unused by these ordinary field regressions. Reset costs are not modeled.

Factor B: full weak-field refit versus a source field plus an affine residual. Use the same polynomial3/RFF128 dictionaries, weak windows8/16 and source qualification as WEAK_CONTROL_PROTOCOL85_V1.md. Full refit retains its ridge/source-prior grid. The residual control freezes all source coefficients except the affine feature rows (constant, observed state and command), and fits their coefficient changes with ridge in{1e-6,1e-3,0.1,10}. This is standard restricted residual identification; the true equation or sensor mapping is not supplied to fitting.

For each allocation/model family, select a primary candidate by VAL H32 and a diagnostic candidate by the longest available VAL horizon (64 for short,128 for long). Save selection before generating EVAL; use the unchanged independent EVAL inputs, output metrics and guards. New seed2026853 is used for all four change conditions, without changing these grids or objectives based on its results.

If longer coverage or a standard residual control repairs the failure, report that explanation and do not claim a new world-model mechanism is necessary. Otherwise quantify remaining support and finite-data/model-family limitations before calling it a gap. All first-two-seed gains are descriptive attribution; the third seed is an unopened ordinary-control check, not validation of a novel method.
