#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
test -s outputs/model_summary.csv
"$PYTHON" -u src/evaluate.py >logs/evaluate.log 2>&1
"$PYTHON" -u src/divergence_trend.py >logs/divergence_trend.log 2>&1
"$PYTHON" -u src/informativeness.py >logs/informativeness_train.log 2>&1
"$PYTHON" -u src/informativeness.py --evaluate >logs/informativeness_evaluate.log 2>&1
"$PYTHON" -u src/tail_sensitivity.py >logs/tail_sensitivity.log 2>&1
"$PYTHON" -u src/exposure_probe.py >logs/exposure_probe.log 2>&1
"$PYTHON" -u src/report.py >logs/report.log 2>&1
echo "Computation complete; inspect evidence before outcome classification."
