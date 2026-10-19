# Autonomous time-series world-model discovery: cycle9

This is a completed evidence checkpoint; the full research goal remains active.

Server directory: `D:\tswm_discovery_20261016_v1\cycle9`. Original source dependency used during DoTime execution: `D:\tswm_discovery_20261016_v1\upstream\dotime`, revision8f227406a5b17c2040cfdcc680b493729ef47d00. Byte-identical package/protocol files are retained under `sources/dotime` with a SHA256 manifest.

Use the existing project `.venv\Scripts\python.exe -B -u`. `scripts/runtime_profile.py` must precede numeric imports. Run `scripts/inverse_score_audit.py` for the new native-loss diagnostic. `scripts/evaluate_dotime_native.py` evaluates the fixed540episode subset; append `--full` for all12,150episodes. The DoTime evaluator expects the prior pinned upstream dependency and the cycle-local pfns vendor. The retained copy can restore that dependency if reproducing on another machine. No training or hyperparameter tuning is needed for the frozen model.

Checkpoint and released data are downloaded directly on the server, hash-checked and retained in `data`. Source/configs, metrics, per-episode predictions, raw scientific logs, primary PDFs and the7required research ledgers are included. Do not interpret historical native defects or a known generic repair as a novel contribution. Do not overwrite prior cycles or this checkpoint after sealing.
