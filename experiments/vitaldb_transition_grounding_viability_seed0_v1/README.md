# VitalDB transition-grounding viability, seed 0

Three matched prospective-action RSSM conditions: unchanged baseline, current-state auxiliary grounding, and predicted-rollout transition grounding. The two auxiliary heads have identical 2,179-parameter capacity and are removed at inference. Device-computed CE is reserved for post-hoc probes.

This reuses the original 495 cases and patient-disjoint split from Round 1, the 58,946-parameter prospective RSSM and training permutations from Round 2, and TRAIN thresholds and patient-balanced geometry protocol from Round 6. All λ selection uses the original validation split. The original Baseline checkpoint and prediction are reused and verified exactly.

On the server, run `python src/train.py`, then `python src/evaluate.py`, `python src/nonlinear.py`, `python src/lambda_scan.py`, and `python src/finalize.py`. `scripts/run_remaining.sh TRAIN_PID` runs the latter four after training. Paths in `config.yaml` resolve relative to this folder. No additional data download is needed. The large `outputs/arrays` cache stays on the compute server; compact CSV, plots, reports, and all six new checkpoints are versioned.

Training uses BIS/MAP forecasting plus either a current-state or predicted-rollout physiological grounding loss. The latter uses the shared head at 30, 60, 180, and 300 seconds. HR is only an auxiliary target. TCI-derived CE never enters training. The matched wrong-action analysis evaluates true, held, and wrong future actions on the same donor-eligible test windows.
