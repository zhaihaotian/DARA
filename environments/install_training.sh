#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
env_name=${1:-rdgdpo}
conda create -y -n "$env_name" python=3.10.21 pip
conda run -n "$env_name" python -m pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cu121 -r "$repo/environments/training.lock.txt"
conda run -n "$env_name" python -m pip install --no-deps 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
conda run -n "$env_name" python -m pip install --no-deps --no-build-isolation -e "$repo/vendor/verl"
conda run -n "$env_name" python -m pip check
conda run --no-capture-output -n "$env_name" bash "$repo/scripts/test.sh"
