#!/bin/bash
#SBATCH --job-name=codecs-store
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=.local/logs/%x-%j.out
# Real-data store: REALDATA_DATES (default 3) ICON-CH1 sample dates written with the codec into a Zarr format 3
# store (REALDATA_OUT, default .local/data/realdata-f3.zarr, -f3 for Zarr format 3; never a store named
# *sample-f3.zarr, a name reserved for the store of a chunk baseline of scripts/baseline_chunks.py), read back with
# anemoi-datasets in a fresh process with a cold cache and only ANEMOI_DATASETS_CODECS_PATH=<store>, then
# `anemoi-datasets-codecs check`. Interpreters: WRITE_PYTHON (.venv), READ_PYTHON (.venv-anemoi3). Submit from the
# repository root.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
write_py=${WRITE_PYTHON:-.venv/bin/python}
read_py=${READ_PYTHON:-.venv-anemoi3/bin/python}
n=${REALDATA_DATES:-3}
out=${REALDATA_OUT:-.local/data/realdata-f3.zarr}
case "$out" in *sample-f3.zarr) echo "refusing to overwrite $out" >&2; exit 2 ;; esac
export ANEMOI_DATASETS_CODECS_THREADS=${ANEMOI_DATASETS_CODECS_THREADS:-16} PYTHONWARNINGS=ignore
lscpu | grep "Model name"
rm -rf "$out"
ANEMOI_DATASETS_CODECS_CACHE=$PWD/.local/cache $write_py scripts/realdata_store.py write "$out" "$n"
ANEMOI_DATASETS_CODECS_CACHE=off ANEMOI_DATASETS_CODECS_PATH=$(realpath -m "$out") \
  $read_py scripts/realdata_store.py read "$out" "$n"
ANEMOI_DATASETS_CODECS_CACHE=off $write_py -m anemoi_datasets_codecs.cli check "$out"
