#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -u src/evaluate.py | tee outputs/evaluate.log
python -u src/finalize.py | tee outputs/finalize.log
