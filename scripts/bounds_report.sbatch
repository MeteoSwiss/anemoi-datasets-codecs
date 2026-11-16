#!/bin/bash
#SBATCH --job-name=codecs-bounds
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=.local/logs/%x-%j.out
# Encode step of scripts/bounds_report.py: BOUNDS_DATES (date indices, default 2 evenly spaced) of BOUNDS_STORE
# (default the ICON-CH1 sample) with the bounds of BOUNDS_CONFIG (default tests/data/icon_ch1_sample.json) times
# each of BOUNDS_SCALES (default 0.5,1,2,4), into BOUNDS_OUT (default .local/bounds/encode-<job>.json).
# Interpreter: PYTHON. Submit from the repository root.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
py=${PYTHON:-.venv/bin/python}
store=${BOUNDS_STORE:-${ANEMOI_DATASETS_CODECS_SAMPLE:-.local/data/icon-ch1-sample.zarr}}
config=${BOUNDS_CONFIG:-tests/data/icon_ch1_sample.json}
out=${BOUNDS_OUT:-.local/bounds/encode-$SLURM_JOB_ID.json}
export ANEMOI_DATASETS_CODECS_CACHE=${ANEMOI_DATASETS_CODECS_CACHE:-$PWD/.local/cache}
export ANEMOI_DATASETS_CODECS_THREADS=${ANEMOI_DATASETS_CODECS_THREADS:-16} PYTHONWARNINGS=ignore
lscpu | grep "Model name"
mkdir -p "$(dirname "$out")"
$py scripts/bounds_report.py encode "$store" "$config" "$out" ${BOUNDS_DATES:+--dates "$BOUNDS_DATES"} \
  --scales "${BOUNDS_SCALES:-0.5,1,2,4}"
