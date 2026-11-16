# anemoi-datasets-codecs

Zarr 3 codecs for [anemoi](https://github.com/ecmwf/anemoi-datasets) datasets. They compress every field that has a
per-variable absolute error bound to within that bound, store every other field exactly (by default the forcings as
analytic recipes within 1e-6), and are selected in the `output` section of an anemoi-datasets recipe. On ICON-CH1
(MeteoSwiss's 1 km ICON model, 115 variables on 1.15 M cells) the lattice codec stores a date in 27.2-35.7 MiB with
its default configuration (mean 31.87 MiB over four dates) and in 31.5-44.8 MiB with grid rounding and the forcings
stored exactly (mean 39.0 MiB over 20 dates), against about 300 MiB with anemoi's default Blosc lz4 compressor, and
decodes it in about 0.34 s on one core or 41 ms on 16 ([benchmarks](docs/benchmarks.md)).

**Status: alpha (0.x).** The chunk formats are versioned and stable in practice; the unconditional
[compatibility promise](docs/goals.md#compatibility-promise) starts with 1.0. Version 0.1.0, not on PyPI: install
from the repository (a C compiler with OpenMP is needed). The Python API and the command line may change. Developed
and measured on one dataset, ICON-CH1.

## Features

* **Error-bounded**: every decoded value is within its variable's bound, checked cell by cell when encoding; variables
  without a bound and static fields (orography, latitude and longitude terms) are exact, except the forcings (local
  time, insolation), which the single codec stores by default as recipes within 1e-6 (`physics.recipe_fit: false`
  keeps them exact).
* **Deterministic**: decoding is bit-identical across machines, thread counts and the two CPU variants of the C
  kernels (AVX2 and portable).
* **Three codecs**: `icon_lattice_bytes` (and its array-to-bytes form `icon_lattice`), the ICON lattice codec for
  datasets on ICON limited-area grids, which predicts each cell from already decoded neighbours on the grid and from
  related variables; `static_ref` (static fields as references) and `field_terms` (fields that depend on the
  orography as residuals), two grid-agnostic *stages* that run ahead of any codec in a Zarr codec chain.
* **Single codec or chain**: the lattice codec alone, or the chain `[field_terms, static_ref, icon_lattice_bytes]`;
  on ICON-CH1 the chain, with a bound on the forcings, was 0.2-0.7% larger than the single codec; the chain is
  less measured ([which to choose](docs/user-guide.md#single-codec-or-chain)).
* **Self-contained stores**: the codec configurations are small JSON in the array's `zarr.json`; static fields and the
  grid's lattice live in the store's *sidecar*, the directory `<store>/.codecs/`, which the codecs find through zarr.
  Readers need only the package.
* **Fits anemoi-datasets**: `anemoi-datasets-codecs physics expand` writes the recipe's `output` section;
  `anemoi-datasets-codecs check` verifies a built store.
* **Versioned formats** with a [compatibility promise](docs/goals.md#compatibility-promise): a format never changes
  without a new version number, and a reader decodes a chunk bit-identically or fails with a clear error. Each store
  records its complete configuration and the package version that created it.

## Installation

The package is not on PyPI. Install it from the repository into the environment that has zarr-python 3 (and
anemoi-datasets, if you build datasets with it):

```bash
pip install "git+https://github.com/MeteoSwiss/anemoi-datasets-codecs"
```

or from a checkout:

```bash
git clone https://github.com/MeteoSwiss/anemoi-datasets-codecs.git
pip install ./anemoi-datasets-codecs
pip install "./anemoi-datasets-codecs[grid]"     # + scipy, netCDF4: build the lattice of a new ICON grid
anemoi-datasets-codecs isa                       # the C kernel variant in use
```

Python >= 3.11 and a C compiler with OpenMP (gcc): the C kernels are compiled at install time, as an AVX2 and a portable
scalar variant, one of which is picked when the library loads. The dependencies are numpy, zarr-python >= 3.1 and
numcodecs >= 0.14. Building anemoi datasets with the codecs needs anemoi-datasets >= 0.5.50 with zarr-python >= 3.4.1,
which anemoi-datasets installs on Python 3.12 and 3.13 ([details](docs/goals.md#integration-with-anemoi-datasets)).

## Quick start

A lattice codec configuration, `lattice.json`. It names the grid and gives the variables' error bounds (names or
regular expressions), clips and static fields; everything else has a [default](docs/user-guide.md#configuration). The
names are those of the ICON-CH1 datasets (the default `icon-ch1` name convention); other names need a matching
[convention](docs/user-guide.md#physics-layer).

```json
{"grid": "icon-ch1",
 "bounds": {"T_2M": 0.05, "PMSL": 2.0, "PS": 2.0, "T_[0-9]+": 0.05, "FI_[0-9]+": 2.0, "TOT_PREC_.*": 1e-5},
 "clip": {"TOT_PREC_.*": [0, null]},
 "static_fields": ["FIS", "cos_latitude", "sin_latitude", "cos_longitude", "sin_longitude"]}
```

`physics expand` writes the recipe's `output` section, here for the chain, from the dataset's variables in the order
anemoi writes them: an existing store of the same variables (`--variables source.zarr`), or a JSON list of the names
(`names.json`) with `--sort`, since anemoi sorts them by name. `--forcing-bound 1e-6` gives the forcings (local time
and insolation fields) a bound of 1e-6, so that the chain codes them lossily instead of storing them exactly:

```bash
anemoi-datasets-codecs physics expand lattice.json --variables names.json --sort --forcing-bound 1e-6 --yaml
```

```yaml
output:
  format: zarr3
  chunking: {dates: 1, ensembles: 1, values: 1147980}    # the whole grid in one chunk
  compressors:
  - name: field_terms
    configuration: {variables: [...], terms: {...}, bounds: {...}, margin: 0.015625}
  - name: static_ref
    configuration: {variables: [...], static_fields: [FIS, cos_latitude, cos_longitude, sin_latitude, sin_longitude],
                   inline: false}
  - name: icon_lattice_bytes
    configuration: {version: 1, variables: [...], bounds: {...}, chain: true, ...}
```

Put it in the recipe and build the dataset; the lattice and the static fields go to `out.zarr/.codecs/`:

```bash
anemoi-datasets create recipe.yaml out.zarr
anemoi-datasets-codecs check out.zarr       # variables, chain, sidecar complete, chunks decode from the store
```

Read it as any anemoi dataset, with the package installed:

```python
from anemoi.datasets import open_dataset

ds = open_dataset("out.zarr")
```

The single codec instead of the chain is one `icon_lattice_bytes` entry with the lattice configuration and the
variables ([user guide](docs/user-guide.md#anemoi-datasets-recipe-output)); the user guide also shows how to write
stores with zarr directly ([quickstart](docs/user-guide.md#quickstart)).

## Documentation

* [User guide](docs/user-guide.md): installation, writing stores with zarr and anemoi-datasets, single codec or chain,
  configuration, choosing the bounds, the physics layer, sidecars, reading and moving stores, threads, environment
  variables, command line, troubleshooting.
* Codecs and formats: [the ICON lattice codec](docs/codecs/icon_lattice.md) (how it works, chunk format),
  [static_ref](docs/codecs/static_ref.md), [field_terms](docs/codecs/field_terms.md), and the
  [chain envelope](docs/codecs/envelope.md) that carries the stages' metadata.
* [Goals](docs/goals.md): purpose, promises, non-goals, compatibility promise, anemoi-datasets integration, known
  limitations.
* [Design](docs/design.md): architecture, what is in which codec and why, design decisions.
* [Benchmarks](docs/benchmarks.md): size, error and speed on ICON-CH1.
* [Lab](docs/lab.md): the research toolkit (`anemoi-datasets-codecs lab`).
* [Contributing](docs/contributing.md): development setup, tests, the format invariants and their checks,
  conventions.
* [Scripts](scripts/README.md): benchmarks, baseline checks, real-data scripts.
* [Changelog](CHANGELOG.md)

## Repository layout

```text
src/anemoi_datasets_codecs/
    icon_lattice/        the ICON lattice codec: engine, Zarr codecs, C kernels, the bundled ICON-CH1 lattice
    static_ref/          the static_ref stage
    field_terms/         the field_terms stage
    physics/             everything that knows variable names: conventions, terms, physics expand
    zarr3.py             generic Zarr 3 adapters, the codec pipeline that serves a store's sidecar
    envelope.py          the chain envelope
    chain.py             codec chains for the tools
    cli.py               the anemoi-datasets-codecs command
    lab/                 the research toolkit
tests/                   tests; tests/data/ holds the plan and chunk baselines
scripts/                 benchmarks, baseline checks, real-data scripts
docs/                    user guide, codec formats, goals, design, benchmarks, contributing
.github/workflows/ci.yml CI: pre-commit, tests on Python 3.11 and 3.14, the anemoi-datasets tests
```

The module-level layout is in [design.md](docs/design.md#architecture).

## Development

```bash
uv sync
uv run pytest -m "not realdata"
uv run ruff check && uv run ruff format --check
```

Everything else (real-data tests, the baseline checks, C kernels, conventions) is in
[docs/contributing.md](docs/contributing.md).

## Licence

BSD 3-Clause, see [LICENSE](LICENSE). Copyright (c) 2026, MeteoSwiss.
