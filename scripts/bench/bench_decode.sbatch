#!/bin/bash
#SBATCH --job-name=codecs-bench
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=.local/logs/%x-%j.out
# Decode speed of this package, AVX2 and scalar kernels, on the first BENCH_DATES (default 1) chunks of a Zarr format 3
# store written by the codec (BENCH_STORE, default .local/data/realdata-f3.zarr from realdata_store.sbatch), 1 and 16
# threads, 3 interleaved rounds in separate processes; results in .local/bench/decode-<job>.jsonl and a table at the
# end of the log. Interpreter: PYTHON. Submit from the repository root.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
py=${PYTHON:-.venv/bin/python}
store=${BENCH_STORE:-.local/data/realdata-f3.zarr}
dates=${BENCH_DATES:-1}
export ANEMOI_DATASETS_CODECS_CACHE=$PWD/.local/cache
out=.local/bench/decode-$SLURM_JOB_ID.jsonl
mkdir -p .local/bench
lscpu | grep "Model name"
for round in 1 2 3; do
  $py -W ignore scripts/bench/bench_decode.py "$store" "$out" --dates "$dates" --threads 1,16 --repeats 9 --isa avx2
  $py -W ignore scripts/bench/bench_decode.py "$store" "$out" --dates "$dates" --threads 1,16 --repeats 3 --isa scalar
done
$py scripts/bench/summary.py "$out"
