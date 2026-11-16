#!/bin/bash
# Build the kernels library by hand into OUT.so (same sources and flags as setup.py: the avx2 and scalar variants
# plus the dispatch), for ANEMOI_DATASETS_CODECS_LIBRARY=OUT.so (development: time or test a kernel change, e.g. with
# bench_decode.py or baseline_chunks.py check, without reinstalling the package). Uses .venv's Python headers.
#   scripts/bench/build_lib.sh OUT.so
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
csrc=$here/src/anemoi_datasets_codecs/icon_lattice/csrc
out=$1
tmp=$(mktemp -d)
inc=$("$here/.venv/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['include'])")
flags="-O3 -fopenmp -ffp-contract=off -fPIC -fvisibility=hidden -std=gnu11 -Wall -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable"
gcc $flags -DLC_SCALAR -DLC_VARIANT=scalar -c "$csrc/variant.c" -o "$tmp/scalar.o"
gcc $flags -mavx2 -DLC_VARIANT=avx2 -c "$csrc/variant.c" -o "$tmp/avx2.o"
gcc $flags -DLC_HAVE_AVX2=1 -DPy_LIMITED_API=0x030B0000 -I"$inc" -c "$csrc/dispatch.c" -o "$tmp/dispatch.o"
gcc -shared -fopenmp -o "$out" "$tmp/dispatch.o" "$tmp/scalar.o" "$tmp/avx2.o" -lm
rm -rf "$tmp"
echo "$out"
