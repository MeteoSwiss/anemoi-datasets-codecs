# scripts

Regression, real-data and benchmark tooling. Run everything from the repository root (`uv run python scripts/...`, or
an environment's `bin/python`).

## Data

Most of the real-data scripts, the `realdata` tests and the measurements of [docs/benchmarks.md](../docs/benchmarks.md)
need two inputs that are **not distributed with the repository**:

* the **ICON-CH1 sample**: an anemoi dataset in Zarr format 3 with 20 dates of ICON-CH1, 115 variables on 1,147,980
  cells, located through `ANEMOI_DATASETS_CODECS_SAMPLE` (default `.local/data/icon-ch1-sample.zarr` in the checkout),
  with its codec configuration `tests/data/icon_ch1_sample.json` (variables, bounds, clip, static fields, grid
  `icon-ch1`; the scripts add lossless Blosc lz4);
* the **ICON-CH1 grid file** `icon_grid_0001_R19B08_mch.nc` (`ANEMOI_DATASETS_CODECS_GRID_FILE`), for building a
  lattice only.

The `realdata` tests skip without them. ICON-CH1-EPS forecasts are published as MeteoSwiss open data; an anemoi
dataset built from them can be written and measured with these scripts too, but its numbers are its own (other dates,
other fields), and the scripts that take the sample assume its 115 variable names.

What each script needs:

| needs | scripts |
|---|---|
| nothing beyond the package | `plan_baseline.py`, `golden_chunks.py`, `fork_check.py`, `figures/*.py`, `bench/build_lib.sh`, `lab/build_asan.sh`, `make_envs.sh`, `test_all_envs.sh`; `bench/summary*.py` and `bounds_report.py draw` on the output of the others |
| a Zarr 3 store written with the single lattice codec (any data) | `baseline_chunks.py`, `bench/bench_decode.py` |
| an anemoi dataset (any) | `sample_store.py`; `bounds_report.py stats` / `encode` (with a lattice codec configuration for its variables and grid) |
| the ICON-CH1 sample | `realdata_store.py`, `physics_size.py`, `bench/bench_chain.py`, `bench/bench_encode.py`, `anemoi_chain.py` (also anemoi-datasets), the `*.sbatch` files |
| an ICON grid file | `build_bundled_grid.py` (`uv run --extra grid`) |

The scripts' default outputs (stores, results, logs) go under `.local/` in the checkout, which git ignores.

## Regression guards

The golden chunks (`tests/data/golden_chunks.json`, run by pytest) guard the decoder and are never regenerated;
`golden_chunks.py` only adds new ones. The other scripts below are byte baselines: regression guards of the encoder,
which must stay green for a change not meant to change encoder output; a deliberate encoder or default change adds or
regenerates their entries in the same pull request and lists the change in the changelog
([contributing](../docs/contributing.md#golden-chunks-and-byte-baselines)). `baseline_chunks.py` is the exception: it
checks the decoder on stored chunks and, like a golden chunk, is never regenerated.

* `golden_chunks.py add` / `sizes`: the golden chunks, small synthetic chunks on the test lattice of
  `tests/conftest.py`, each with its complete configuration, the sha256 of the chunk and of its decoded values, and the
  static fields it references in `tests/data/golden_sidecar/`. `add` writes the entries `tests/data/golden_chunks.json`
  lacks and never rewrites one; `sizes` encodes them without writing anything. Needs no data.
* `plan_baseline.py write|add|check OUT.json`: sha256 of a synthetic chunk with the 115 CH1 names under several plan
  settings; `check` against `tests/data/chunk_sha256_baseline_numpy-2.5.4.json` after any change of the encoder, the
  plan, the structures or the C kernels. Every setting spells out the options whose default has changed, so that a
  change of a default leaves the existing entries valid; `add` adds the settings the file lacks (an encoder change
  adds settings, it does not rewrite existing ones), `write` writes a new file and refuses an existing one. Needs no
  data.
* `baseline_chunks.py write STORE OUT_DIR` / `check STORE BASELINE.json`: the decoder side. `write` records the size,
  the sha256 of every chunk file of a Zarr format 3 store written by the codec and the sha256 of its decoded float32
  bytes (`OUT_DIR/<store>.chunks.json`, with the configuration, package version and kernel variant); `check`
  re-decodes every chunk with the current package and fails on a changed chunk file, a missing chunk or any decoded
  bit that differs. Run it on a store whose chunks are never rewritten (the maintainers keep two ICON-CH1 dates for
  this, not distributed), with `ANEMOI_DATASETS_CODECS_ISA=scalar` for the scalar kernels as well. A note that the
  store's codec configuration differs from the baseline's is not a failure: a store's metadata can be rewritten (for
  instance to add keys) without touching a chunk, and the chunk and decode comparison decides; the writer stamp
  `written_by` is ignored there. A store whose
  `zarr.json` lacks `rounding` / `recipe_fit` takes the reading version's defaults for new chunks; add them explicitly
  before appending.
* `bench/bench_encode.py write|check BASELINE.json [--dates 0,6,13,19] [--threads N] [--no-valid-dates]`: the encoder
  side on real data. `write` encodes sample dates with `LatticeCodec(**sample_config()).encode`, without dates (as any
  zarr write) and with the date the sample gives each one, and records per date and mode the seconds, bytes and sha256
  of the chunk, with the package and numpy versions, kernel variant, thread count and the codec's complete
  configuration; `check` re-encodes with the baseline's configuration (a baseline without one: the sample configuration
  with `rounding: grid` and `recipe_fit: false`), compares every sha256 and exits non-zero on any difference or missing
  entry. Write the baseline with the unchanged encoder (`write` refuses an existing file), check after any change of the
  encoder, the plan, the physics, the structures or the C kernels; after a deliberate encoder or default change, write a
  new one. `write` encodes with the sample configuration as it is, so with the package's current defaults. The dates of
  the ICON-CH1 sample are relabelled, not the valid dates of its fields, so there the dated encodes store the forcings
  losslessly (the recipes of those dates miss), while the undated ones fit the recipes to the fields
  (`physics.recipe_fit`, the default); recipes from true valid dates are guarded by `plan_baseline.py` (synthetic data),
  or by this script on a store with true valid dates.
* `fork_check.py [CHILD_THREADS]`: decodes the synthetic test chunk in a process forked after the parent decoded
  with 8 threads; prints whether the child hung (GNU libgomp's pool does not survive fork; the decoder clamps such a
  child to 1 thread).

## Real data

* `sample_store.py convert SRC DST` / `check SRC DST`: copies a plain anemoi store (Blosc arrays, group attributes;
  Zarr format 2 or 3) into a Zarr format 3 store with zarr-python 3, `data` chunked one date per chunk and copied a
  few dates at a time, every other array in one chunk, Blosc lz4 everywhere, attributes as they are; `check`
  compares attributes, metadata, the small arrays and some dates bit for bit. Use it to make a Zarr 3 copy of an
  existing anemoi dataset to try the codecs on.
* `realdata_store.py write OUT.zarr N` / `read OUT.zarr N`: writes the first N sample dates with the codec into a Zarr
  format 3 store laid out as anemoi does (codec `icon_lattice` as the serializer, coordinates, statistics, attributes,
  sidecar filled), and reads it back (anemoi-datasets `open_dataset` when importable, else zarr), checking every lossy
  variable against its bound, the forcing recipes against `physics.recipe_bound` and every other one for equality;
  prints MiB per date, sidecar and configuration size. `realdata_store.sbatch` (`REALDATA_DATES`, default 3;
  `REALDATA_OUT`, default `.local/data/realdata-f3.zarr`, `-f3` for Zarr format 3) writes with `.venv`, reads with
  `.venv-anemoi3` in a fresh process with `ANEMOI_DATASETS_CODECS_CACHE=off` and only `ANEMOI_DATASETS_CODECS_PATH=OUT`,
  then runs `anemoi-datasets-codecs check`; it refuses to write over a store named `*sample-f3.zarr`, the name of the
  chunk baseline store of `baseline_chunks.py`.
* `physics_size.py DATE STRUCTURE K OUT.jsonl`: chunk size, worst error / bound and mean RMSE / bound of one sample
  date (with its date, so the forcings can be recipes) for every physics setting of `SETTINGS` (default, off, each
  part off). `physics.sbatch` (`PHYSICS_DATE`, default 6) runs `tree_b` k = 3 and `tree` k = 1 into
  `.local/physics/size-<job id>.jsonl`.
* `bounds_report.py stats STORE CONFIG STATS.json` / `encode STORE CONFIG ENCODE.json` / `draw STATS.json OUT.svg
  [--encode ENCODE.json]`: are the bounds of a configuration reasonable? `stats` (seconds per date) computes per lossy
  variable the float32 ulp and the data's own step, the median difference across a lattice edge, the median change
  over one time step (from each date and the one before it; `--no-tendency` for the sample, whose dates are
  relabelled) and the standard deviation; `encode` encodes dates with all bounds times `--scales` (default
  0.5,1,2,4) and records bytes and errors per variable; `draw` places every bound among its scales and next to its
  bits per value (docs/user-guide.md "Choosing the bounds"; the example figure is
  `docs/figures/bounds_report_icon_ch1.svg`, `--png` also renders a PNG when cairosvg is installed).
  `bounds_report.sbatch` runs `encode` (`BOUNDS_STORE`, default the sample; `BOUNDS_CONFIG`; `BOUNDS_DATES`;
  `BOUNDS_SCALES`; `BOUNDS_OUT`, default `.local/bounds/encode-<job id>.json`).
* `anemoi_chain.py OUT_DIR N_DATES [--structure tree] [--k 1] [--only chain|single] [--forcing-bound B]` (in
  `.venv-anemoi3`, with anemoi-datasets >= 0.5.50): anemoi-datasets end to end. `anemoi-datasets create` from the
  sample as an `anemoi_dataset` source (no GRIB), the first N dates, twice: with the `output` section of
  `physics expand` (the chain `[field_terms, static_ref, icon_lattice_bytes]`, the margin from the sample's
  statistics) and with the single
  `icon_lattice_bytes` codec of the same configuration, each with `ANEMOI_DATASETS_CODECS_PATH=<target>`; then per store
  `open_dataset` in a fresh process (cold cache, only `ANEMOI_DATASETS_CODECS_PATH=<store>`) reads every date and
  compares it with the source (every lossy variable within its bound, the forcing recipes of the single codec within
  `physics.recipe_bound`, the others bit for bit), and `anemoi-datasets-codecs check` runs. `--forcing-bound B` builds
  the chain of `physics expand --forcing-bound B` (its forcings checked against B). Recipes, create logs and
  `results.json` (bytes per date, create and read seconds, worst error / bound, the check) go to OUT_DIR.
  `anemoi_chain.sbatch` (`ANEMOI_CHAIN_DATES`, default 3; `ANEMOI_CHAIN_OUT`, default `.local/anemoi-chain`;
  `ANEMOI_CHAIN_ARGS`, e.g. `--structure tree_b --k 3`).
* `build_bundled_grid.py GRID.nc NAME` (`uv run --extra grid`): recovers the lattice of a grid from its cell
  centres, validates it against the grid file's neighbour table and writes it with `index.json` into
  `src/anemoi_datasets_codecs/icon_lattice/data/grids` (for ICON-CH1 the sha256 must stay `2c359f6e...`).

## Benchmark (`bench/`)

* `bench_decode.py STORE OUT.jsonl [--dates 1] [--threads 1,16] [--repeats 9] [--isa auto|avx2|scalar]`: decodes
  the first chunk files of a store written with the single lattice codec (read from the store directory, codec from
  the array metadata) into a preallocated buffer; one JSON line per chunk and thread count with best / median ms and
  the sha256 of the decoded bytes. Works on any such store, not only the sample's.
* `summary.py OUT.jsonl`: table per kernel variant and thread count (best of rounds, median of medians, ratio to
  AVX2) and a check that every variant and thread count decoded each chunk to the same bits.
* `bench_decode.sbatch` (`BENCH_STORE`, default `.local/data/realdata-f3.zarr`; `BENCH_DATES`): three interleaved
  rounds of AVX2 and scalar at 1 and 16 threads, each in its own process, into
  `.local/bench/decode-<job id>.jsonl`, then the table.
* `bench_chain.py write DIR DATE STRUCTURE K OUT.jsonl [--forcing-bound B]` / `read DIR OUT.jsonl [--repeats 7]`: the
  single codec, the
  chain of `physics expand` and Blosc lz4 (written once) on one sample date, all through zarr (`a[:] = x`, no dates):
  `write` writes the stores, times the write, checks the read-back against the bounds (the single codec's forcing
  recipes against `physics.recipe_bound`) and records the chunk and sidecar bytes, and with `--forcing-bound B` also
  writes the chain of `physics expand --forcing-bound B`; `read` times `a[...]` per store at
  the thread count of `ANEMOI_DATASETS_CODECS_THREADS` (Blosc with as many threads) after two warm-up reads, best /
  median and the sha256 of the decoded bytes. `summary_chain.py OUT.jsonl`: the size and read tables and a check that
  every thread count decoded each store to the same bits. `bench_chain.sbatch` (`BENCH_DATE`, default 6;
  `BENCH_CHAIN_DIR`, default `.local/bench/chain`; `BENCH_CHAIN_OUT`): `tree` k = 1 and `tree_b` k = 3, then three
  interleaved rounds of reads at 1 and 16 threads, each in its own process.
* `bench_encode.py write|check BASELINE.json`: encode time of sample dates (with and without the sample's dates) at
  `--threads N`; `check` prints baseline and current seconds per chunk and the total (the regression guard above).
* `build_lib.sh OUT.so`: builds the kernels library by hand with the flags of `setup.py`, for
  `ANEMOI_DATASETS_CODECS_LIBRARY=OUT.so` (try a kernel change without reinstalling).

## Figures

* `figures/icon_lattice_figures.py [--out DIR] [--only NAME ...]`: writes the SVG figures of
  `docs/codecs/icon_lattice.md` into `docs/codecs/figures/icon_lattice/` (plain SVG, no plotting library). Everything
  shown is computed: the structures from the bundled ICON-CH1 lattice, the plan from `tests/data/icon_ch1_sample.json`,
  the container from the synthetic test chunk of `tests/conftest.py` (in a temporary cache directory). The output is
  deterministic: rerun it after a change of the structures, the plan, the encoder or the container and commit the
  diff.
* `figures/static_ref_figures.py [--out DIR] [--only NAME ...]`: writes the SVG figures of `docs/codecs/static_ref.md`
  into `docs/codecs/figures/static_ref/` with the helpers and palette of the lattice script. Everything shown is
  computed: the synthetic test chunk of `tests/conftest.py` on 2 dates and 2 members is written through zarr with the
  chain `physics.expand` gives for its variables, into a temporary store, and the figures read its chunk, its sidecar
  and the stage's headers. Deterministic; rerun it after a change of the stage, its header or the envelope.
* `figures/field_terms_figures.py [--out DIR] [--only NAME ...]`: writes the SVG figures of
  `docs/codecs/field_terms.md` into `docs/codecs/figures/field_terms/`, with the helpers and palette of
  `icon_lattice_figures.py`. Every number comes from the stage itself (`FieldTerms`, `field_terms.arith`) on synthetic
  slices; the header figure encodes the fixed input of `test_header_pinned_bytes`. Deterministic: rerun it after a
  change of the stage's arithmetic or header and commit the diff.
* `figures/benchmarks_figures.py [--out DIR] [--only NAME ...]`: writes the SVG figures of `docs/benchmarks.md` into
  `docs/benchmarks/figures/` (size, decode, encode, physics) with the helpers and palette of `icon_lattice_figures.py`.
  Nothing is measured: every number comes from `figures/benchmarks_data.json`, a transcription of the document's tables
  with the section, machine and conditions of each series, so the figures reproduce without the data. Each panel
  draws one run (timings are not comparable across runs). Deterministic: after a change of the document's numbers,
  update the data file, rerun and commit the diff.
* `figures/envelope_figures.py [--out DIR] [--only NAME ...]`: writes the SVG figures of `docs/codecs/envelope.md`
  into `docs/codecs/figures/envelope/` with the helpers and palette of `icon_lattice_figures.py`. It runs the chain
  `[field_terms, static_ref, icon_lattice_bytes]` of `physics.expand` on the synthetic test chunk through zarr (in a
  temporary directory), records every codec call, and draws the buffers, the stored envelope and the decode path of
  that run (asserted, so a change of the adapters fails it). Deterministic; rerun it after a change of the envelope,
  the adapters, the stages or the encoder and commit the diff.

## Lab (`lab/`)

The research toolkit itself is the command `anemoi-datasets-codecs lab` (docs/lab.md); here are its helper scripts.

* `lab/build_asan.sh OUT.so`: the kernels with AddressSanitizer and UndefinedBehaviorSanitizer, for
  `ANEMOI_DATASETS_CODECS_LIBRARY=OUT.so` (prints the `LD_PRELOAD` and `ASAN_OPTIONS` to use); run the fuzzer and the
  decode tests against it.
* `lab/refs.sbatch` (`REFS_DATE`, default 6; `REFS_FIELDS`; `REFS_METHODS`; `PYTHON` with the `lab` extra): `lab refs`
  on one sample date, all lossy variables, into `.local/lab/refs-<job id>.jsonl`.

## Environments and tests

`make_envs.sh` (re)creates the three test environments: `.venv` (Python 3.14), `.venv-py311` (3.11) and
`.venv-anemoi3` (3.13, with anemoi-datasets >= 0.5.50 from PyPI). They are created in place, or in
`$ANEMOI_DATASETS_CODECS_ENVS` with `./.venv*` symlinks to them when that is set. `test_all_envs.sh [PYTEST ARGS]` runs
the tests in each.

## Slurm examples

The `*.sbatch` files are examples for a Slurm cluster. Submit them from the repository root with a CPU partition
(`mkdir -p .local/logs; sbatch -p PARTITION scripts/<name>.sbatch`): they `cd "$SLURM_SUBMIT_DIR"`, ask for 16-32
CPUs, 64-96 GiB and at most 1 h, and write logs to `.local/logs/<job name>-<job id>.out`, results to
`.local/{bench,physics,bounds,lab}/` and stores to `.local/data/`. Each is a short shell script: without Slurm, run it
as `SLURM_SUBMIT_DIR=$PWD SLURM_JOB_ID=local bash scripts/<name>.sbatch`. They take the interpreter from `PYTHON` (or
`WRITE_PYTHON` / `READ_PYTHON` in `realdata_store.sbatch`), default `.venv/bin/python` (`.venv-anemoi3/bin/python` for
anemoi-datasets).

## Environment variables

| variable | used by | meaning |
|---|---|---|
| `ANEMOI_DATASETS_CODECS_SAMPLE` | realdata_store, physics_size, anemoi_chain, bench_chain, bench_encode, bounds_report.sbatch, lab/refs.sbatch | the ICON-CH1 sample store (default `.local/data/icon-ch1-sample.zarr`) |
| `ANEMOI_DATASETS_CODECS_GRID_FILE` | tests (lattice-building test) | the ICON-CH1 grid file (netCDF); the test skips when unset |
| `ANEMOI_DATASETS_CODECS_ENVS` | make_envs | where the test environments are created (default: in place) |
| `ANEMOI_DATASETS_CODECS_THREADS` | all decoding / encoding (bench_encode: `--threads`) | codec threads (sbatch files: 16) |
| `ANEMOI_DATASETS_CODECS_CACHE` | all | cache directory, `off` for a cold reader (sbatch files: `.local/cache`) |
| `ANEMOI_DATASETS_CODECS_PATH` | realdata_store read | sidecar search path (the store being read) |
| `ANEMOI_DATASETS_CODECS_ISA` | bench_decode (`--isa`), baseline_chunks | kernel variant: `auto`, `avx2`, `scalar` |
| `ANEMOI_DATASETS_CODECS_LIBRARY` | any (with build_lib.sh) | another build of the kernels |
| `PYTHON`, `WRITE_PYTHON`, `READ_PYTHON`, `REALDATA_DATES`, `REALDATA_OUT`, `PHYSICS_DATE`, `BENCH_STORE`, `BENCH_DATES`, `BENCH_DATE`, `BENCH_CHAIN_DIR`, `BENCH_CHAIN_OUT`, `ANEMOI_CHAIN_DATES`, `ANEMOI_CHAIN_OUT`, `ANEMOI_CHAIN_ARGS`, `BOUNDS_*`, `REFS_*` | sbatch files | see each file's header |
