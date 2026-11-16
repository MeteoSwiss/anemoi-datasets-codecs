#!/bin/bash
#SBATCH --job-name=codecs-anemoi-chain
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=.local/logs/%x-%j.out
# anemoi-datasets end to end (scripts/anemoi_chain.py): `anemoi-datasets create` from the ICON-CH1 sample as an
# anemoi_dataset source, ANEMOI_CHAIN_DATES (default 3) dates, once with the chain from `physics expand` and once with
# the single icon_lattice_bytes codec, both read back with open_dataset in a fresh process (cold cache, only
# ANEMOI_DATASETS_CODECS_PATH=<store>) and checked; results in ANEMOI_CHAIN_OUT (default .local/anemoi-chain).
# Interpreter: PYTHON (default .venv-anemoi3/bin/python, anemoi-datasets with Zarr 3 support). Submit from the
# repository root.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
py=${PYTHON:-.venv-anemoi3/bin/python}
out=${ANEMOI_CHAIN_OUT:-.local/anemoi-chain}
export ANEMOI_DATASETS_CODECS_THREADS=${ANEMOI_DATASETS_CODECS_THREADS:-16} PYTHONWARNINGS=ignore
export ANEMOI_DATASETS_CODECS_CACHE=${ANEMOI_DATASETS_CODECS_CACHE:-$PWD/.local/cache}
lscpu | grep "Model name"
git log --oneline -1 || true
$py scripts/anemoi_chain.py "$out" "${ANEMOI_CHAIN_DATES:-3}" ${ANEMOI_CHAIN_ARGS:-}
