#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PREFIX="${HOME}/dara-math-v07"
PYTHON=python3
UV_VERSION=0.12.12
VERL_REPOSITORY=https://github.com/verl-project/verl.git
VERL_TAG=v0.7.0
VERL_COMMIT=f9c855f7cf04d603c9546bc01776c74806a879c1

usage() {
    cat <<'EOF'
Usage: scripts/bootstrap_runtime.sh [options]

Options:
  --prefix DIR       Install outside the repository (default: ~/dara-math-v07)
  --python COMMAND   Python 3.12 interpreter used to create the venv
  --help             Show this message

The target prefix must not already contain venv, verl-v0.7.0-base, or
framework-paper. Models and training outputs are intentionally not downloaded.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            PREFIX=$2
            shift 2
            ;;
        --python)
            PYTHON=$2
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

PREFIX=$(realpath -m "$PREFIX")
VENV="$PREFIX/venv"
BASE_FRAMEWORK="$PREFIX/verl-v0.7.0-base"
RUNTIME="$PREFIX/framework-paper"

for path in "$VENV" "$BASE_FRAMEWORK" "$RUNTIME"; do
    if [[ -e "$path" ]]; then
        printf 'Refusing to overwrite existing path: %s\n' "$path" >&2
        exit 2
    fi
done

command -v "$PYTHON" >/dev/null || {
    printf 'Python interpreter not found: %s\n' "$PYTHON" >&2
    exit 2
}
"$PYTHON" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))' || {
    printf 'The validated dependency lock requires Python 3.12: %s\n' "$PYTHON" >&2
    exit 2
}
command -v git >/dev/null || {
    printf 'git is required\n' >&2
    exit 2
}

mkdir -p "$PREFIX"
if command -v uv >/dev/null; then
    UV=$(command -v uv)
else
    command -v curl >/dev/null || {
        printf 'curl is required to install uv %s\n' "$UV_VERSION" >&2
        exit 2
    }
    installer=$(mktemp)
    trap 'rm -f "$installer"' EXIT
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" -o "$installer"
    UV_INSTALL_DIR="$PREFIX/bin" sh "$installer"
    UV="$PREFIX/bin/uv"
fi

"$UV" venv --seed --python "$PYTHON" "$VENV"
git clone --filter=blob:none --branch "$VERL_TAG" --depth 1 "$VERL_REPOSITORY" "$BASE_FRAMEWORK"
actual_commit=$(git -C "$BASE_FRAMEWORK" rev-parse HEAD)
if [[ "$actual_commit" != "$VERL_COMMIT" ]]; then
    printf 'Unexpected verl commit: %s (expected %s)\n' "$actual_commit" "$VERL_COMMIT" >&2
    exit 2
fi

"$UV" pip install --python "$VENV/bin/python" \
    --constraint "$ROOT/requirements-runtime-lock.txt" \
    --requirement "$ROOT/requirements-runtime.txt"
"$UV" pip install --python "$VENV/bin/python" \
    --constraint "$ROOT/requirements-runtime-lock.txt" \
    --editable "$BASE_FRAMEWORK[math]"
"$UV" pip install --python "$VENV/bin/python" \
    --constraint "$ROOT/requirements-runtime-lock.txt" packaging ninja wheel setuptools
"$UV" pip install --python "$VENV/bin/python" \
    --constraint "$ROOT/requirements-runtime-lock.txt" \
    --no-build-isolation "flash-attn==2.8.3"

"$VENV/bin/python" "$ROOT/scripts/prepare_runtime.py" \
    --base-framework "$BASE_FRAMEWORK" \
    --output "$RUNTIME"
PYTHONPATH="$RUNTIME" "$VENV/bin/python" "$ROOT/scripts/preflight.py" \
    --framework "$RUNTIME"

printf '\nRuntime ready. Configure a machine profile with:\n'
printf '  python:    %s\n' "$VENV/bin/python"
printf '  framework: %s\n' "$RUNTIME"
