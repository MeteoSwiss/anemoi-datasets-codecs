#!/bin/bash
#SBATCH --job-name=codecs-physics
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=.local/logs/%x-%j.out
# Worth of the physics switches on ICON-CH1 sample date PHYSICS_DATE (default 6): tree_b k = 3 and tree k = 1, one
# JSON line per setting in .local/physics/size-<job>.jsonl. Interpreter: PYTHON. Submit from the repository root.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
py=${PYTHON:-.venv/bin/python}
date=${PHYSICS_DATE:-6}
export ANEMOI_DATASETS_CODECS_CACHE=$PWD/.local/cache ANEMOI_DATASETS_CODECS_THREADS=16 PYTHONWARNINGS=ignore
mkdir -p .local/physics
out=.local/physics/size-$SLURM_JOB_ID.jsonl
$py scripts/physics_size.py "$date" tree_b 3 "$out"
$py scripts/physics_size.py "$date" tree 1 "$out"
