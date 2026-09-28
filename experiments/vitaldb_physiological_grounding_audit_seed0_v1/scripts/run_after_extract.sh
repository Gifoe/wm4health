#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -u src/probes.py | tee outputs/probes.log
python -u src/nonlinear.py | tee outputs/nonlinear.log
python -u src/geometry.py | tee outputs/geometry.log
python -u src/finalize.py | tee outputs/finalize.log
