#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -u src/discover.py
python -u scripts/optimize_random_blocks.py
python -u scripts/apply_random_blocks.py
python -u src/describe_regimes.py
python -u src/control_balance.py
python -u src/train_conditions.py
python -u scripts/stamp_checkpoints.py
python -u src/predict_conditions.py
python -u src/support_diagnostics.py
python -u src/evaluate_conditions.py
python -u src/finalize.py
