#!/bin/bash
# Create (or re-sync) the three test environments: .venv (Python 3.14), .venv-py311 (3.11) and .venv-anemoi3 (3.13,
# plus anemoi-datasets >= 0.5.50 from PyPI, which writes and reads Zarr 3 stores; anemoi-datasets needs
# Python < 3.14). All have zarr-python 3 (a runtime dependency). The environments live in $ANEMOI_DATASETS_CODECS_ENVS
# if set (./.venv* are symlinks to them), else in place.
set -euo pipefail
cd "$(dirname "$0")/.."
S=${ANEMOI_DATASETS_CODECS_ENVS:-}
[ -n "$S" ] && mkdir -p "$S"
make() {  # name python [extra requirement]
  local name=$1 py=$2 req=${3:-}
  local env=$name
  if [ -n "$S" ]; then env="$S/$name"; ln -sfn "$env" "$name"; fi
  UV_PROJECT_ENVIRONMENT="$env" uv sync --python "$py" --reinstall-package anemoi-datasets-codecs
  if [ -n "$req" ]; then
    # fasteners: imported by anemoi-datasets' create path, declared only in its heavy "create" extra
    uv pip install --python "$name/bin/python" "$req" fasteners
  fi
}
make .venv 3.14
make .venv-py311 3.11
make .venv-anemoi3 3.13 "anemoi-datasets>=0.5.50"
