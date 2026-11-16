#!/bin/bash
#SBATCH --job-name=codecs-refs
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=.local/logs/%x-%j.out
# lab refs on ICON-CH1 sample date REFS_DATE (default 6), all lossy variables (REFS_FIELDS: comma-separated subset),
# against the reference compressors at the same bound; one JSON line per variable, method and layout in
# .local/lab/refs-<job>.jsonl, the tables in the log. Needs the optional references (zfpy, pysz, numcodecs; the
# interpreter PYTHON must have them). REFS_METHODS drops slow methods (e.g. everything but xz). Submit from the
# repository root.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
py=${PYTHON:-.venv/bin/python}
date=${REFS_DATE:-6}
store=${ANEMOI_DATASETS_CODECS_SAMPLE:-.local/data/icon-ch1-sample.zarr}
export ANEMOI_DATASETS_CODECS_CACHE=$PWD/.local/cache ANEMOI_DATASETS_CODECS_THREADS=16 PYTHONWARNINGS=ignore
mkdir -p .local/lab
args=(--date "$date" --out ".local/lab/refs-$SLURM_JOB_ID.jsonl")
[ -n "${REFS_FIELDS:-}" ] && args+=(--fields "$REFS_FIELDS")
[ -n "${REFS_METHODS:-}" ] && args+=(--methods "$REFS_METHODS")
$py -m anemoi_datasets_codecs.cli lab refs tests/data/icon_ch1_sample.json "$store" "${args[@]}"
