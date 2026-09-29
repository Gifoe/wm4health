#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
while [[ ! -s outputs/selected_lambda.json ]]; do
  if ! kill -0 "${1:?training PID required}" 2>/dev/null; then
    echo 'TRAIN_FAILED_OR_EXITED_WITHOUT_SELECTION' >&2
    exit 1
  fi
  sleep 20
done
python -u src/evaluate.py
python -u src/nonlinear.py
python -u src/lambda_scan.py
python -u src/finalize.py
echo PIPELINE_COMPLETE
