# C53 native TSWM subset reproduction - running snapshot

This is an unfinished experiment snapshot. Read CURRENT_STATE.md and the dated outputs/progress_snapshot_v1/progress.json before interpreting status. No new-crop result or final novel research direction is claimed.

The frozen grid has two original backbones, four native configurations and all five original seeds. Completed fits and the shared AE are on the server; incomplete resource/instrumentation attempts are retained there and excluded. RUNTIME_SUPPLEMENT_V5.md and V6.md document the adopted effective-batch memory adaptation and its numerical/dropout limits.

Active scripts are explicitly listed by the publication manifest. The server supervisor is capture_native53_v6.py; score_native53_v7.py waits for all 40 selected checkpoints; capture_eval53_v2.py monitors the actual Windows process and runs scoring/verification once. The latter is Windows-specific. Source and evidence scripts are supplied for review; this running snapshot is not yet a one-command fresh-run release. Historical repair guards must not be faked to rerun it. A fresh-run entry point will accompany the completed reproduction.

Paths are supplied through TSWM_STORAGE_CONFIG and TSWM_UPSTREAM_ROOT. storage.example.json illustrates a non-identifying layout; its data_root must contain greenhouse3_original_v1/experiment_1/TimeSeries and experiment_2/TimeSeries with the original public bytes. TSWM source pin is 2e5e48540ff0da48e8192163969a6c58df1d5a3e, and C52 records the acquisition route. No server machine paths, raw signals, weights or environments are published here.

Prior framework evidence remains in ../cycle49 and the real-data input audit in ../cycle52. This grid is source-protocol subset reproduction plus a separately identified new-crop transfer audit. Neither is a new model. New hypotheses and methods require additional falsifiable evidence.
