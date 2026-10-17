# C52 native TSWM real-data input contract and baseline qualification

Status: source-grounded diagnostic; no accepted new research direction.

This audit follows C49's negative conditional-score screen. It asks whether a reproduced world model receives only forecast-time-available target history, and whether the demonstrated native interpolation boundary defect occurs on its original public Greenhouse data. Fixing a loader is not proposed as research novelty.

Pinned upstream: HaochenZhang717/TSWM-Official-Repo commit 2e5e48540ff0da48e8192163969a6c58df1d5a3e. Execute its unchanged native dataset and windowing code. Original public source DOI 10.4121/21960932, third Autonomous Greenhouse Challenge. Do not substitute the fourth challenge or download images. Retrieve metadata, file listing and sizes first. Preserve source IDs, response bytes, checksums and acquisition receipts.

Qualification protocol: native context 256, horizon 16, total window 272, stride 80, max_nan_ratio .1, max_nan_gap 6, data split seed 0, held-out team ratio .2, native fixed-range scaling. Enumerate every native accepted window and report per team and split. Measure target/exogenous missingness crossing the history boundary, actual differences between native full-window and history-only interpolation, normalized RMS/max effect and affected feature counts. Perturb only future target values in retained copies and verify native-history dependence; the original data and upstream files remain immutable. Non-missing boundaries serve as the null control.

Report denominator counts and overlapping-window limitations. This measures preprocessing dependence and prevalence, not predictive degradation, publication result correction, causal validity, or a research gap. Distinguish source-protocol reproduction from information-safe variants in any later model comparisons. No new model or test-selected hyperparameter search is part of this audit.

After qualification, choose the next baseline comparison from actual evidence and prior completed cycles. The overall target remains a reusable time-series world-model problem with defensible prior-art boundaries and independent minimal method validation, not Greenhouse-specific optimization.
