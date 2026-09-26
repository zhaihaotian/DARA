#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/vendor/verl${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests -v
