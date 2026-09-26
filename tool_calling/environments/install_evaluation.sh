#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
env_name=${1:-dara-bfcl-v4}
conda create -y -n "$env_name" python=3.12.14 pip
conda run -n "$env_name" python -m pip install --no-deps -r "$repo/environments/bfcl-v4.lock.txt"
source_dir="$repo/.dependencies/gorilla-v4"
mkdir -p "$(dirname "$source_dir")"
git clone https://github.com/ShishirPatil/gorilla.git "$source_dir"
git -C "$source_dir" checkout 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8
conda run -n "$env_name" python -c 'import pathlib,site,sys; (pathlib.Path(site.getsitepackages()[0])/"bfcl_source.pth").write_text(sys.argv[1]+"\n")' "$source_dir/berkeley-function-call-leaderboard"
# BFCL V4 is pinned by source commit, independently of its package version metadata.
conda run -n "$env_name" python -c 'import bfcl_eval; print("BFCL source:", bfcl_eval.__file__)'
