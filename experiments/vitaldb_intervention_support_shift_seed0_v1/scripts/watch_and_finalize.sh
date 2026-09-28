#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
while ! grep -q TRAIN_CONDITIONS_COMPLETE outputs/training.log; do
  if ! pgrep -f '^python -u src/train_conditions.py$' >/dev/null; then
    echo 'Training process stopped before completion' >&2
    exit 1
  fi
  sleep 30
done
bash scripts/run_after_training.sh
