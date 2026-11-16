#!/bin/bash
# Build the kernels library with AddressSanitizer and UndefinedBehaviorSanitizer into OUT.so (same sources and flags as
# setup.py / scripts/bench/build_lib.sh, plus -fsanitize=address,undefined -fno-omit-frame-pointer -g -O1;
# -ffp-contract=off stays). Prints how to run Python against it.
#   scripts/lab/build_asan.sh OUT.so
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
csrc=$here/src/anemoi_datasets_codecs/icon_lattice/csrc
out=$(realpath -m "$1")
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
inc=$("$here/.venv/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['include'])")
san="-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1"
flags="-fopenmp -ffp-contract=off -fPIC -fvisibility=hidden -std=gnu11 -Wall -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable -Wno-psabi $san"
gcc $flags -DLC_SCALAR -DLC_VARIANT=scalar -c "$csrc/variant.c" -o "$tmp/scalar.o"
gcc $flags -mavx2 -DLC_VARIANT=avx2 -c "$csrc/variant.c" -o "$tmp/avx2.o"
gcc $flags -DLC_HAVE_AVX2=1 -DPy_LIMITED_API=0x030B0000 -I"$inc" -c "$csrc/dispatch.c" -o "$tmp/dispatch.o"
gcc -shared -fopenmp $san -o "$out" "$tmp/dispatch.o" "$tmp/scalar.o" "$tmp/avx2.o" -lm
asan=$(gcc -print-file-name=libasan.so)
ubsan=$(gcc -print-file-name=libubsan.so)
cat <<TXT
built $out
run Python against it (the sanitizer runtimes must be preloaded into the interpreter):
  export ANEMOI_DATASETS_CODECS_LIBRARY=$out
  export LD_PRELOAD="$asan $ubsan"
  export ASAN_OPTIONS=detect_leaks=0:abort_on_error=1
  export UBSAN_OPTIONS=print_stacktrace=1
  ANEMOI_DATASETS_CODECS_THREADS=1 uv run pytest tests/test_codec.py -q
TXT
