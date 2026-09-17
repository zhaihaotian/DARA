#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
env_name=${1:-dara-inference}
conda create -y -n "$env_name" python=3.12.14 pip
conda run -n "$env_name" python -m pip install --no-deps -r "$repo/environments/inference.lock.txt"
# vLLM 0.11.0 supplies its own flash-attention backend. The separate flash_attn
# package in the historical environment is used by training, not this API server.
conda run -n "$env_name" python -c 'import torch,vllm; print(torch.__version__, vllm.__version__)'
