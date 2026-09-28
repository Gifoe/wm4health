#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/arrays plots logs
python -u src/build_reference.py > logs/build_reference.log 2>&1
python -u src/checks.py > logs/checks.log 2>&1
python -u src/score_support.py > logs/score_support.log 2>&1
python -u src/dynamic_support.py > logs/dynamic_support.log 2>&1
python -u src/analyze.py > logs/analyze.log 2>&1
python -u src/figures.py > logs/figures.log 2>&1
python -u src/validate_outputs.py > logs/validate_outputs.log 2>&1
echo PIPELINE_COMPLETE
