#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! grep -q TRAIN_CONDITIONS_COMPLETE outputs/training.log; then
  echo 'Training is not complete' >&2
  exit 1
fi
python -u scripts/stamp_checkpoints.py
python -u src/predict_conditions.py
python -u src/support_diagnostics.py
python -u src/evaluate_conditions.py
python -u src/finalize.py
