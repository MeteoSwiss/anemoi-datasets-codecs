#!/bin/bash
# Run the test suite in the three environments of scripts/make_envs.sh. Extra pytest arguments are passed
# through, e.g. scripts/test_all_envs.sh -m "not realdata".
set -o pipefail
cd "$(dirname "$0")/.." || exit 1
status=0
for env in .venv .venv-py311 .venv-anemoi3; do
  [ -x "$env/bin/python" ] || continue
  versions=$("$env/bin/python" -c 'import sys, zarr, numcodecs, numpy; print(sys.version.split()[0], "zarr", zarr.__version__, "numcodecs", numcodecs.__version__, "numpy", numpy.__version__)')
  echo "== $env: $versions"
  "$env/bin/python" -m pytest -q -W ignore::DeprecationWarning -p no:cacheprovider "$@" | tail -1 || status=1
done
exit $status
