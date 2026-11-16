# Changelog

Every release lists what changed for users. Changes of the bytes the encoder writes for an existing configuration
and changes of defaults are always listed, under `### Encoder changes`, so that whoever needs to reproduce bytes knows
which version to pin; a new format version or structure id is always listed under `### Formats`
([compatibility promise](docs/goals.md#compatibility-promise)).

## 0.1.0 (2026-11-16)

First public release. Zarr 3 codecs for anemoi datasets: every field with a per-variable error bound is coded lossily
within that bound (verified per cell at encode time), every other field exactly, except the forcings, which the single
codec stores by default as analytic recipes within 1e-6; decoding is bit-identical across machines, thread counts and
kernel variants. Developed and measured on ICON-CH1.

### Features

* `icon_lattice_bytes` (bytes to bytes, for `output.compressors` of anemoi-datasets) and `icon_lattice` (array to
  bytes), the ICON lattice codec for datasets on ICON limited-area grids: prediction along the ICON refinement tree on
  a triangle lattice, quantisation at the bound, cross-variable terms, 32-lane rANS; C kernels in an AVX2 and a
  portable scalar variant, chosen at load time.
* `static_ref` (static fields as references into the store's `.codecs/` sidecar) and `field_terms` (orography terms
  as residuals against a per-chunk fit), grid-agnostic stages that compose with any codec in a Zarr codec chain.
* The physics layer: name conventions `icon-ch1` (COSMO names) and `ifs` (experimental), or custom ones;
  cross-variable, below-ground (experimental) and forcing-recipe terms resolved from the variable names.
* Defaults that cost no decode time: `options.rounding: direct` (the full quantisation step the bound allows) and
  `physics.recipe_fit: true` (the forcings as recipes without a valid date, as in every zarr write).
  `"rounding": "grid"` and `"recipe_fit": false` give the other behaviour (the forcings then stored exactly).
* Stores record their complete resolved configuration in `zarr.json`, every option and physics key included, and the
  package version that created the array (`written_by`, informational); later versions keep honouring a stored
  configuration.
* Sidecars: the codecs find a local store's `<store>/.codecs/` through zarr's codec pipeline; static fields captured
  while writing; content-addressed files shared between stores through `ANEMOI_DATASETS_CODECS_PATH`.
* The ICON-CH1 lattice bundled with the package (grid `icon-ch1`).
* The `anemoi-datasets-codecs` command: `physics expand` (the three chain configurations, or the `output` section of
  an anemoi-datasets recipe, from one lattice configuration), `check`, `attach`, `info`, `plan`, `grid build`,
  `grids`, `isa`, and `lab`, an experimental research toolkit (bit accounting, oracles, reference compressors, a mini
  grid, an experiment ledger, fuzzing).

### Formats

* Versioned, each frozen under its version number (layout and decoder arithmetic): the lattice container version 3
  (`LATC`), the chain envelope version 1 (`FCHN`), the `static_ref` and `field_terms` headers version 1, the
  structures `tree`, `tree_b` and `hc2_k3`, and the sidecar files.
* Golden chunks (`tests/data/golden_chunks.json`): seven stored chunks with the sha256 of their decoded values,
  decoded by the tests with both kernel variants: the three structures, every field mode except LOSSLESS, cross and
  below-ground terms, recipes, and a chain chunk.
* The compatibility promise: a format never changes without a new version number, and a reader decodes a chunk
  bit-identically to what its writer verified or fails with a clear error. Stores written by any released version
  stay readable by later versions; a 0.x format retired before 1.0 would be announced here, with a conversion tool in
  that release. Encoder bytes and defaults may change between releases. The promise becomes unconditional with 1.0
  ([towards 1.0](docs/goals.md#towards-10)).

### Known limitations

* Not on PyPI and no prebuilt wheels: installs compile the C kernels (C compiler with OpenMP); aarch64 and macOS are
  not built or tested.
* Building datasets through anemoi-datasets needs anemoi-datasets >= 0.5.50 with zarr-python >= 3.4.1 (Python 3.12
  or 3.13), where writing Zarr format 3 is experimental.
* The decoder checks little beyond the headers on corrupt input, and the chunk formats have no payload checksum: a
  corrupted payload can decode to different values without an error.
* More in [docs/goals.md](docs/goals.md#known-limitations).
