#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
while [[ ! -s outputs/training_summary.csv ]] || [[ $(wc -l < outputs/training_summary.csv) -lt 3 ]]; do
  if ! kill -0 "${1:?training PID required}" 2>/dev/null; then
    echo 'TRAIN_FAILED_OR_EXITED_WITHOUT_SUMMARY' >&2
    exit 1
  fi
  sleep 15
done
python -u src/evaluate.py
python -u src/probes.py
python -u src/finalize.py
echo PIPELINE_COMPLETE
