# Contributing

The repository is [MeteoSwiss/anemoi-datasets-codecs](https://github.com/MeteoSwiss/anemoi-datasets-codecs). The
package is alpha (0.x): the Python API, the command line, the encoder's bytes and the defaults may change between
releases; the chunk formats change only with a new version ([goals.md](goals.md#compatibility-promise)).

## Development setup

Python >= 3.11, a C compiler with OpenMP (gcc) and [uv](https://docs.astral.sh/uv/), which is what CI uses:

```bash
git clone https://github.com/MeteoSwiss/anemoi-datasets-codecs.git
cd anemoi-datasets-codecs
uv sync                              # .venv with the package (editable), the dev group and the C kernels
uv run anemoi-datasets-codecs isa    # the kernel variants built and the one in use
```

`uv.lock` pins the dependencies; `.python-version` selects the Python of `uv sync`. Optional extras: `grid`
(scipy and netCDF4, for `grid build`) and `lab` (the reference compressors of `lab refs`, docs/lab.md):
`uv sync --extra lab`.

`scripts/make_envs.sh` creates three test environments: `.venv` (Python 3.14), `.venv-py311` (Python 3.11, the
oldest supported) and `.venv-anemoi3` (Python 3.13 with anemoi-datasets >= 0.5.50 from PyPI, for the tests that use
anemoi-datasets). `scripts/test_all_envs.sh` runs the tests in each of them that exists.

## Tests

```bash
uv run pytest -m "not realdata"
ANEMOI_DATASETS_CODECS_ISA=scalar uv run pytest -m "not realdata"     # on the portable kernel variant
```

The suite runs on synthetic data in about a minute and needs nothing outside the repository. Some tests skip
without an optional dependency: the two anemoi-datasets tests in `tests/test_zarr.py` (they run in `.venv-anemoi3`)
and the reference compressors of `tests/lab/` (the `lab` extra).

Tests marked `realdata` need data that is not distributed with the repository: an ICON-CH1 sample store (an anemoi
dataset in Zarr format 3, 20 dates; `ANEMOI_DATASETS_CODECS_SAMPLE`, configuration in
`tests/data/icon_ch1_sample.json`) and the ICON-CH1 grid file `icon_grid_0001_R19B08_mch.nc`
(`ANEMOI_DATASETS_CODECS_GRID_FILE`). They skip when these are missing. The benchmarks and most scripts need the same
data ([scripts/README.md](../scripts/README.md)). ICON-CH1-EPS forecasts are MeteoSwiss open data.

## Code style

[ruff](https://docs.astral.sh/ruff/), configured in `pyproject.toml` (line length 120; the markdown files are wrapped at
120 columns too):

```bash
uv run ruff check && uv run ruff format --check
uvx pre-commit run --all-files       # every hook of .pre-commit-config.yaml, as CI runs them
```

Source files carry no licence header; the licence is the repository's [LICENSE](../LICENSE) (BSD 3-Clause). Docs
and comments state facts about this package, in British spelling, with nothing site-specific (machine paths, cluster
names, user names).

CI (`.github/workflows/ci.yml`) runs on every pull request and on pushes to `main`: the pre-commit hooks (ruff, and
checks of YAML, TOML and JSON syntax, whitespace, large files and merge-conflict markers; `tests/data/`, which holds the
golden chunks and the baselines, is excluded), `pytest -m "not realdata"` on Python 3.11 and 3.14 (part of it again on
the scalar kernels), and the anemoi-datasets tests with anemoi-datasets >= 0.5.50 on Python 3.13. All must pass. A
second workflow checks the pull request title (see below).

## The invariants

These hold for every change. A change that would break one is a new format version or a new structure id, never a
silent change ([compatibility promise](goals.md#compatibility-promise)).

* **Formats change only with a new version.** The lattice container version 3
  ([codecs/icon_lattice.md](codecs/icon_lattice.md)), the chain envelope version 1
  ([codecs/envelope.md](codecs/envelope.md)), the `static_ref` and `field_terms` headers version 1
  ([codecs/static_ref.md](codecs/static_ref.md), [codecs/field_terms.md](codecs/field_terms.md)), the structures
  `tree`, `tree_b` and `hc2_k3`, and the sidecar files. A layout or decoder change is a new version in its document;
  a new structure gets a new id.
* **Existing chunks decode bit-identically**, with every later version, on both kernel variants. The golden chunks
  and the tests that pin the header bytes guard this (below).
* **Every decoded value is within its variable's bound**, verified per cell at encode time by the codec that codes
  it, and decoding is deterministic across machines, thread counts and the two kernel variants.
* **The codecs know no variable names.** Everything name- or physics-specific lives in
  `src/anemoi_datasets_codecs/physics/`; new kinds of terms go in `physics/terms.py`, new naming conventions are data
  (`physics/conventions.py`). The stages and the package-level modules import nothing of `icon_lattice`.
* **Configurations stay small JSON** (they are stored in `zarr.json`, complete: every option and physics key); large
  data (static fields, lattices) lives in the content-addressed sidecar. Nothing in one codec's configuration is read
  from another codec's chunk headers. A key keeps its meaning once released.
* **Zarr 3 only**: no numcodecs codec, no Zarr format 2 metadata.
* **Format-level names never change**: the codec names `icon_lattice`, `icon_lattice_bytes`, `static_ref`,
  `field_terms`, the sidecar directory `.codecs/`, the magics `LATC` and `FCHN`.

The encoder's bytes and the defaults are not invariants: they may change in a release, and every such change is
listed in the [changelog](../CHANGELOG.md).

### Golden chunks and byte baselines

Two kinds of checks, with different rules:

* **Golden chunks** guard the decoder. `tests/data/golden_chunks.json` lists chunk files stored in `tests/data/` with
  the sha256 of the chunk and of its decoded values; `test_golden_chunk_decodes_bit_identically` decodes each with both
  kernel variants. `scripts/golden_chunks.py add` writes the entries the index lacks and never rewrites one
  ([scripts/README.md](../scripts/README.md#regression-guards)). A golden chunk is never regenerated or changed: a
  failure means the decoder changed, which only a new format version may do. A new format element gets a new golden
  chunk.
* **Byte baselines** guard the encoder, as a regression check, not a promise: they say whether a change alters the
  bytes the encoder writes. They must stay green for any change that is not meant to change encoder output. A
  deliberate change of the encoder or of a default adds or regenerates the affected baseline entries in the same pull
  request and says so in the changelog under `### Encoder changes`. Never regenerate a baseline to hide an unintended
  change, and never touch a golden chunk for an encoder change.

Every contributor can run these, and should, with both kernel variants (`ANEMOI_DATASETS_CODECS_ISA=scalar` for the
portable one):

* `pytest`, which includes the golden chunks, the plan and chunk baselines (`tests/test_physics.py`:
  `test_plan_identical_to_baseline`, and `test_chunk_bytes_identical_to_baseline`, which runs with numpy 2.5.4 only,
  the version its baseline was written with; its below-ground setting also needs the OpenBLAS kernels of the
  baseline, Haswell or Zen: set `OPENBLAS_CORETYPE=Haswell` on other CPUs, as CI does), the tests that pin the
  bytes of the envelope and the stage headers (`tests/test_envelope.py`, `tests/test_static_ref.py`,
  `tests/test_field_terms.py`), the structure sha test and the AVX2 / scalar bit-identity test;
* `uv run python scripts/plan_baseline.py check tests/data/chunk_sha256_baseline_numpy-2.5.4.json`: the sha256 of a
  synthetic chunk under several plan settings. Every setting spells out the options whose default has changed, so a
  change of a default does not move the existing entries; an encoder change adds settings with
  `scripts/plan_baseline.py add` rather than rewriting existing ones.

Two more checks run on real data and need the maintainers' ICON-CH1 sample store and baselines, which are not
distributed: `scripts/baseline_chunks.py check STORE BASELINE.json` (the decoder side: chunks of a stored dataset
decode to the same bits; like a golden chunk, never regenerated) and `scripts/bench/bench_encode.py check
BASELINE.json` (the encoder side on real dates, a byte baseline). If you cannot run them, say so in the pull request;
a maintainer runs them. Their arguments are in [scripts/README.md](../scripts/README.md).

### C kernels

The kernels are in `src/anemoi_datasets_codecs/icon_lattice/csrc/`, compiled by `setup.py` at install time twice (an
AVX2 and a portable scalar variant, chosen at load time; `ANEMOI_DATASETS_CODECS_ISA` overrides). After editing a C
source, rebuild:

```bash
uv sync --reinstall-package anemoi-datasets-codecs
```

`-ffp-contract=off` is required: without fused multiply-add the dequantisation, the recipe and mask formulas and the
encoder's fit pass (`csrc/encode.c`) round exactly like numpy, so the encoder verifies the bound on the values every
build decodes. The two variants must decode bit-identically.

## Documentation

The documents are part of the change: a change that alters behaviour updates them in the same pull request.

* [user-guide.md](user-guide.md): configuration and usage; [codecs/](codecs/): how each codec works and the versioned
  formats; [design.md](design.md): architecture and decisions; [goals.md](goals.md): promises and known limitations.
* [benchmarks.md](benchmarks.md): measured numbers, with the setup they were measured on. Other documents quote them
  from there.
* Markdown is wrapped at 120 columns.

## Commits, pull requests and the changelog

* Pull requests go against `main`, one topic each, with the documentation update and, where the encoder, the plan or
  the kernels change, the results of the checks above.
* Pull requests are squash-merged: the title becomes the commit on `main` and the description its body, so the
  title follows [Conventional Commits](https://www.conventionalcommits.org/): `type: summary` or
  `type(scope): summary`, with the type one of `feat`, `fix`, `perf`, `docs`, `refactor`, `test`, `build`, `ci`,
  `chore`, and a `!` before the colon for a change that breaks the Python API, the command line or a configuration
  (`feat!: ...`). CI checks the title (`.github/workflows/pr-title.yml`). Commits on the branch are free-form.
* Every user-visible change adds an entry to [CHANGELOG.md](../CHANGELOG.md) under the topmost heading of a version
  not yet released (`## Unreleased`, or a version marked `(unreleased)`), with `### Features`, `### Fixes`,
  `### Encoder changes` or `### Formats` sections. Entries say what changed for the user. A change of the bytes the
  encoder writes for an existing configuration, or of a default, is always an entry under `### Encoder changes`; a
  new format version or structure id is always an entry under `### Formats`.
* Bugs and proposals: [issues](https://github.com/MeteoSwiss/anemoi-datasets-codecs/issues). A bug report is most
  useful with the codec configuration (`zarr.json`), the command and its output, and
  `anemoi-datasets-codecs isa`.
