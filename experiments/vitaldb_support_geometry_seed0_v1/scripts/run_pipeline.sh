#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/checkpoints outputs/arrays plots logs
python -u src/train_ensemble.py 2>&1 | tee logs/train_ensemble.log
python -u src/build_reference.py 2>&1 | tee logs/build_reference.log
python -u src/checks.py 2>&1 | tee logs/checks.log
python -u src/score_support.py 2>&1 | tee logs/score_support.log
python -u src/dynamic_support.py 2>&1 | tee logs/dynamic_support.log
python -u src/analyze.py 2>&1 | tee logs/analyze.log
python -u src/figures.py 2>&1 | tee logs/figures.log
python -u src/validate_outputs.py 2>&1 | tee logs/validate_outputs.log
