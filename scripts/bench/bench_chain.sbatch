#!/bin/bash
#SBATCH --job-name=codecs-bench-chain
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=.local/logs/%x-%j.out
# Single codec vs the physics-expand chain vs Blosc lz4 on ICON-CH1 sample date BENCH_DATE (default 6), through zarr
# (scripts/bench/bench_chain.py): write `tree` k = 1 and `tree_b` k = 3 stores into BENCH_CHAIN_DIR (default
# .local/bench/chain, rewritten) at 16 threads, then three interleaved rounds of zarr reads at 1 and 16 threads, each
# in its own process; JSON lines in BENCH_CHAIN_OUT (default .local/bench/chain-<job>.jsonl), then the table.
# Interpreter: PYTHON (default .venv/bin/python). Submit from the repository root.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
py=${PYTHON:-.venv/bin/python}
dir=${BENCH_CHAIN_DIR:-.local/bench/chain}
out=${BENCH_CHAIN_OUT:-.local/bench/chain-$SLURM_JOB_ID.jsonl}
date=${BENCH_DATE:-6}
export ANEMOI_DATASETS_CODECS_CACHE=${ANEMOI_DATASETS_CODECS_CACHE:-$PWD/.local/cache} PYTHONWARNINGS=ignore
mkdir -p "$(dirname "$out")"
lscpu | grep "Model name"
git log --oneline -1 || true
rm -rf "$dir"
ANEMOI_DATASETS_CODECS_THREADS=16 $py scripts/bench/bench_chain.py write "$dir" "$date" tree 1 "$out"
ANEMOI_DATASETS_CODECS_THREADS=16 $py scripts/bench/bench_chain.py write "$dir" "$date" tree_b 3 "$out"
for round in 1 2 3; do
  for t in 1 16; do
    ANEMOI_DATASETS_CODECS_THREADS=$t $py scripts/bench/bench_chain.py read "$dir" "$out"
  done
done
$py scripts/bench/summary_chain.py "$out"
