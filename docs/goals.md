# Goals

What anemoi-datasets-codecs is for, what it deliberately does not do, what it promises, and where it stands.

## Purpose

Zarr 3 codecs for [anemoi](https://github.com/ecmwf/anemoi-datasets) datasets: codecs that know the shape of an anemoi
chunk and its variables, code the fields that get a per-variable error bound lossily and every other field exactly,
static fields included (the forcings by default as analytic recipes within 1e-6), and compose in a Zarr codec chain. The
package is for people who create anemoi datasets. It was written at MeteoSwiss for its ICON-CH1 datasets and is public
for the anemoi community. The codecs so far were developed for data on the ICON model grid; the package is meant to grow
by further codecs and stages next to them. Today it has three:

* `icon_lattice` / `icon_lattice_bytes`: the ICON lattice codec, for datasets on ICON limited-area grids: spatial
  prediction along the grid's refinement tree, quantisation at a per-variable bound, cross-variable terms between
  lossy fields, rANS entropy coding ([codecs/icon_lattice.md](codecs/icon_lattice.md));
* `static_ref`: the static fields of a chunk as references into the store's sidecar, the directory
  `<store>/.codecs/` ([codecs/static_ref.md](codecs/static_ref.md));
* `field_terms`: lossy fields as residuals against a per-chunk fit on exact fields, the orography terms
  ([codecs/field_terms.md](codecs/field_terms.md)).

They combine freely. The two stages are grid-agnostic and usable ahead of any codec (`[static_ref, blosc]`). For an
ICON dataset there are two equally supported forms, and the choice is the user's
([user guide](user-guide.md#single-codec-or-chain)): the lattice codec alone, with the static fields as references
inside it, or the chain `compressors=[field_terms, static_ref, icon_lattice_bytes]`, whose three configurations
`physics expand` writes from one lattice codec configuration.

### Scope: what applies beyond ICON

Two things tie a component to a dataset: the grid, and the variable names it needs to know.

| component | grid | variable names | applies to |
|---|---|---|---|
| `icon_lattice` / `icon_lattice_bytes` | ICON limited-area grids only (cells on a triangular lattice; a new grid needs its lattice file, `grid build`) | none in the core; the physics layer's terms come from a name convention (`icon-ch1`, `ifs`, custom) | anemoi datasets on an ICON limited-area grid, any naming |
| `static_ref` | any | none (`static_fields` is a list of names) | any anemoi dataset with static fields, ahead of any codec |
| `field_terms` | any | none in the stage (terms are `target <- sources` names); the orography terms `physics expand` gives it come from the name convention | any anemoi dataset with an exact field that other fields depend on linearly |
| chain envelope | any | none | any chain of field stages |
| physics layer, `physics expand` | the lattice configuration it expands | a name convention | datasets with a convention; the expansion itself is ICON today, since it starts from a lattice configuration |

So the stages, the envelope and the chain machinery carry over to other grids as they are; the lattice codec does not,
and a dataset on another grid needs its own codec at the end of the chain (`[field_terms, static_ref, <codec>]` works
with Blosc today, at Blosc's size). The current plan resolution and the orography terms are physics of pressure-level
and surface fields, not of ICON; they apply to any dataset whose names a convention maps.

## What it promises

* **Size.** Every variable gets a pointwise absolute error bound; variables without one are stored exactly (the
  forcings, by default, as recipes within 1e-6). On the ICON-CH1 benchmark dataset the lattice codec stores a date, as
  a zarr write does, in 27.2-35.7 MiB with its default configuration (mean 31.87 MiB over four dates) and in
  31.5-44.8 MiB with `rounding: grid` and the forcings stored exactly (mean 39.0 MiB over 20 dates), against
  298.5 MiB on average with anemoi's default Blosc lz4. With the forcings stored exactly in both, the chain and the
  single codec were within 0.4% of each other, smaller or larger by date; the chain is less measured than the single
  codec ([benchmarks.md](benchmarks.md)).
* **Decode speed for training.** Datasets are written once and read many times by training loaders. The lattice codec
  decodes a chunk in one C call; on the ICON-CH1 benchmark dataset about 0.34 s per date on one core and 41 ms on 16,
  within a small factor of Blosc lz4 while reading about 9 times fewer bytes. Through zarr the chain's stages add
  17-23% on one thread and about 100 ms on 16, most of it the `static_ref` stage's write of the whole date
  ([benchmarks.md](benchmarks.md)). With one codec thread per DataLoader worker (the default there) a loader replica
  with the single codec matched or beat Blosc lz4 in samples per second. Encoding may be slow.
* **Exactness guarantees.** Every decoded value is within its variable's bound: each codec verifies, at encode time
  and per cell, the fields it codes (the lattice codec decodes and checks every lossy field, a field that misses is
  stored losslessly; `field_terms` checks that every value the codec after it may return reconstructs within the
  bound, else it leaves the field untouched). Static and lossless fields are exact, exact zeros of lattice-coded
  fields stay exact, clips are applied on decode. The forcings (local time, insolation) are by default analytic
  recipes in the single codec: every cell within `physics.recipe_bound` (1e-6) or stored exactly as a patch;
  `physics.recipe_fit: false` stores them exactly. Decoding is deterministic and bit-identical across machines,
  thread counts and the AVX2 / scalar kernel variants.
* **Self-describing stores.** The codec configurations are small JSON in `zarr.json`; the reference data a chunk needs
  (the grid's lattice, static fields) is content addressed in a sidecar inside the store, which the codecs find through
  zarr: a local store is read with nothing but the package installed. Readers never need the ICON grid file.

## Non-goals

* **A general-purpose float compressor.** The lattice codec needs a grid whose cells are a cut-out of a regular
  triangular lattice (ICON limited-area grids); other grids need a new lattice and structure or a new codec
  subpackage. The stages are grid-agnostic, but only save what their terms and static fields are worth.
* **Encode speed as the main goal.** The optimisation targets are the compression ratio and the decode speed; a
  dataset is encoded once. Encode speed is not ignored (the encoder is Python / numpy with C passes and its time goes
  to the fits and the verification, [benchmarks.md](benchmarks.md)), but it never buys bytes or decode time.
* **Zarr format 2 and numcodecs-only users.** The package is Zarr 3 only (zarr-python >= 3.1).
* **Other error metrics.** The guarantee is a pointwise absolute bound per variable; the error shape is controlled
  through `k` (maximum error `bound / k`), not through relative, PSNR or spectral targets.
* **Partial chunk decoding.** A chunk is one or more whole dates of all variables and cells; there is no cell-subset
  or single-variable read.

## Compatibility promise

A format never changes without a new version number: a reader either decodes a chunk bit-identically to what its
writer verified, or fails with a clear error, never silently differently. Stores written by any released version stay
readable by later versions; if a 0.x format is ever retired before 1.0, the changelog will say so and a conversion tool
will ship with that release. Bit-identical decoding is tested on x86-64 (AVX2 and scalar). Encoder bytes and defaults
may change between releases. The retirement of a 0.x format is the one exception the promise allows before 1.0; from
1.0 it is unconditional ([Towards 1.0](#towards-10) lists what is still to be done for that).

* **Versioned formats.** The lattice container version 3 (magic `LATC`), the chain envelope version 1 (magic `FCHN`),
  the `static_ref` and `field_terms` headers version 1, the lattice structures `tree`, `tree_b` and `hc2_k3` (a chunk
  records its structure's sha) and the sidecar files (content addressed by sha256). A format version is *frozen*: its
  byte layout and its decoder's arithmetic never change under that version number; a change is a new version (a new
  structure gets a new id), and readers keep decoding the old one. Golden chunks with the hashes of their decoded values
  and tests that pin the header bytes guard this ([contributing.md](contributing.md#the-invariants)). The full texts are
  in the format documents ([icon_lattice](codecs/icon_lattice.md#compatibility-promise),
  [envelope](codecs/envelope.md#compatibility-promise), [static_ref](codecs/static_ref.md#compatibility-promise),
  [field_terms](codecs/field_terms.md#compatibility-promise)).
* **Every decoded value is within its variable's bound**, verified per cell at encode time
  ([Exactness guarantees](#what-it-promises)), whatever the encoder version.
* **The encoder is deterministic, not frozen.** For a given package version, configuration, numpy version and input
  the encoder writes the same bytes (for the least-squares terms, also on the same CPU family, see
  [Known limitations](#known-limitations)); whoever needs to reproduce bytes pins the package version. A later
  version may write other bytes or change a default; every such change is listed in the [changelog](../CHANGELOG.md).
* **A store records what wrote it.** The array's `zarr.json` holds each codec's complete resolved configuration
  (every option and physics key, defaults included) and the package version that created the array (`written_by`,
  for example `"anemoi-datasets-codecs 0.1.0"`; informational, no codec reads it). Later
  versions keep honouring a stored configuration: its keys keep their meaning, so new chunks written into an
  existing store are coded as the store says, not with the reading version's defaults. Keys may be added; a key
  missing from a stored configuration takes the reading version's default. Values that name a behaviour rather than
  spell it out (`tables: auto`, `prob_bits: auto`, the convention `icon-ch1`) are recorded by name: should what they
  stand for change in a release, the changelog says so.
* **Names.** The format-level names (`icon_lattice`, `icon_lattice_bytes`, `static_ref`, `field_terms`, `.codecs/`,
  `LATC`, `FCHN`) do not change. A codec used alone writes the same bytes with and without `chain: true`, and without
  the key it never reads an envelope.
* **Two differences between the chain and the single codec.** (1) A field that `field_terms` replaces by a residual is
  within its bound, but not bit-exact, even where the lattice codec stores the residual LOSSLESS, and its exact zeros
  are not kept ([codecs/field_terms.md](codecs/field_terms.md#the-error-bound)); in a single-codec store "stored
  losslessly" still means exact. (2) The per-cell verification is done by each codec for what it codes: the lattice
  codec verifies the residuals it is given against `b_L`, `field_terms` verifies the reconstruction of its targets
  against `b`; no single pass decodes the whole chain. That the configurations agree is checked from the configurations
  (`anemoi-datasets-codecs check`), not per chunk.

## Integration with anemoi-datasets

The goal is that a dataset recipe selects the codecs and every anemoi tool reads the result:

* **Writing.** anemoi-datasets >= 0.5.50 writes Zarr format 3 with `output.format: zarr3` and takes the codecs of the
  data array as `output.compressors`, a list of Zarr 3 codec specs `{name, configuration}`
  ([anemoi-datasets documentation, "Zarr format"](https://anemoi.readthedocs.io/projects/datasets/en/latest/building/zarr-format.html)).
  It accepts bytes -> bytes codecs only there, hence `icon_lattice_bytes` and stages written as bytes -> bytes codecs.
  `anemoi-datasets-codecs physics expand` writes the whole `output` section (`format: zarr3`, `chunking` with the whole
  grid per chunk, `compressors`). Each codec is built when the recipe is parsed, from JSON only, so configuration
  errors fail before any data is read ([user guide](user-guide.md#codec-chains)).
* **Reading.** `open_dataset` reads the store through zarr's `zarr.codecs` entry points, and the codecs find the
  store's sidecar through zarr's codec pipeline.
* **Versions.** anemoi-datasets uses zarr-python 3 only from 3.4.1, which needs Python >= 3.12 (on Python 3.11 it
  installs zarr-python 2), and its 0.5.50 release supports Python up to 3.13. It marks writing Zarr format 3 as
  experimental, and tools that still read with zarr-python 2 cannot open these stores. The package's anemoi tests
  (`open_dataset` on a store written with the codecs; the recipe's `output` section building the codecs) pass with
  anemoi-datasets 0.5.50, zarr-python 3.4.2 and Python 3.13. The full `anemoi-datasets create` on ICON-CH1 data in
  [benchmarks.md](benchmarks.md#anemoi-datasets-end-to-end) ran with anemoi-datasets 0.5.50.

## Status

**Alpha (0.x).** The chunk formats are versioned and stable in practice; the unconditional compatibility promise
starts with 1.0 ([Compatibility promise](#compatibility-promise)). Version 0.1.0, not on PyPI: install from the
repository, with a C compiler with OpenMP. The Python API and the command line may change. The codecs were developed
and measured on one dataset, ICON-CH1.

### Experimental

Experimental means that behaviour and configuration may change between releases, not decodability: what these write
is versioned and decodes like everything else ([Compatibility promise](#compatibility-promise)).

* `structure: hc2_k3` as the choice for size (its decode speed, below);
* the below-ground terms, `physics.below_ground`;
* the `ifs` name convention (tested on synthetic data only);
* the research toolkit, `anemoi-datasets-codecs lab` ([lab.md](lab.md));
* the Python API and the command line.

The codec chain is not experimental, but it is less measured than the single codec: its sizes come from a few dates
(the single codec's within a fraction of a percent on sample date 6, four dates of one run and the three dates of the
end-to-end run), its decode speed from one date, and it has not been run in a training loader
([benchmarks.md](benchmarks.md#codec-chains)).

### Towards 1.0

What turns the compatibility promise into an unconditional one:

* Golden chunks with the hashes of their decoded values for every format element. Seven exist, on the small test
  lattice: the structures `tree`, `tree_b` and `hc2_k3`; LATTICE, CONSTANT, STATIC_REF (sidecar and inline) and
  RECIPE fields (all three recipes); cross and below-ground mask terms, a clip, exact zeros; and a chain chunk (the
  envelope with both stage headers). Still to come: LOSSLESS fields, steps split into ranges (a grid larger than the
  test lattice), and stored chunks with the stages' other header variants (the pinned-header tests cover their
  bytes).
* Bit-identical decoding of the golden chunks on aarch64, in CI.
* A second real dataset through the lattice codec, and the stages on a dataset that is not on an ICON grid.
* A decision on a payload checksum, and a check of the rANS end state (every lane back at its initial state, every
  stream consumed exactly), which needs no format change.
* The default structure and quantiser policy decided: a decode kernel for the three-source gathers of `hc2_k3`, which
  would make it the choice for size, landed or dropped.
* An end-to-end `anemoi-datasets create` with a released anemoi-datasets, and a store written by another user.
* The chain envelope kept or replaced after discussion with the anemoi and zarr developers.

### Known limitations

* **One dataset.** Sizes and speeds are measured on ICON-CH1 only ([Towards 1.0](#towards-10)). The `ifs` name
  convention is tested on synthetic data with IFS names, not on a real IFS / ERA5 dataset. On model levels the
  inter-level and hydrostatic terms are applied as on pressure levels (the encoder drops terms that do not pay), but
  their worth there is not measured; the below-ground terms are off on model levels.
* **Builds.** No prebuilt wheels: installs compile the C kernels (C compiler with OpenMP). macOS is not supported
  (Apple clang has no OpenMP). On aarch64 only the portable scalar variant is built, and it has not been built or run
  there; on x86-64 that variant is about 9 times (1 thread) and 6 times (16 threads) slower than AVX2.
* **anemoi-datasets.** Needs anemoi-datasets >= 0.5.50 on Python 3.12 or 3.13 (above). The `output` section comes
  from a separate command, `physics expand`, run on the dataset's variables in anemoi's order; anemoi builds the codecs
  one by one, so a misordered variable list that keeps the field count surfaces only through
  `anemoi-datasets-codecs check` (or a failed fit), not when the recipe is parsed. A build from GRIB on the ICON-CH1
  grid has not been run (the end-to-end run used an `anemoi_dataset` source).
* **Forcings in a chain.** Zarr passes no valid dates to a codec. The single codec recovers the forcing recipes from
  the fields (`physics.recipe_fit`); in a chain the lattice codec lacks the latitude / longitude inputs, so the
  forcings are stored losslessly there (11-18% of an ICON-CH1 chunk) unless they get a bound
  (`physics expand --forcing-bound 1e-6`, coded at about 0.6 bits per value).
* **Decode cost of the chain.** The chain decodes 16-18% slower than the single codec on one thread, mostly the
  `static_ref` stage's write of the whole date; the loader replica that showed the single codec ahead of Blosc lz4
  has not been run with the chain.
* **`hc2_k3` and the below-ground terms.** The `hc2_k3` structure is 10-12% smaller than `tree` on ICON-CH1 but
  decodes through the generic gather kernel, about 38% slower on one thread in an indicative measurement; the
  below-ground terms (another 3-5%) cost about 25% at 16 threads through the decode critical path. Both are
  experimental (above).
* **Sidecars on local filesystems only.** The codec pipeline serves the sidecar of a store on a local filesystem. A
  store on an object store gets no sidecar this way: give a local copy of the sidecar with
  `ANEMOI_DATASETS_CODECS_PATH`. With another zarr codec pipeline configured, the sidecar must be given the same way.
  `anemoi-datasets copy` does not copy the sidecar; run `anemoi-datasets-codecs attach` on the copy (not tested).
* **numpy-dependent bytes.** The below-ground terms' coefficients and the `field_terms` fit are float64 least-squares
  fits per chunk, so those chunk bytes depend on the numpy version and on the kernels its linear algebra library
  picks for the CPU (the plans do not; decoding is unaffected, and every value is within its bound either way).
* **Encoding speed.** Python / numpy with C passes: about a minute per ICON-CH1 date on one thread, most of it the
  least-squares fits ([benchmarks.md](benchmarks.md)).
* **Corrupt input.** The decoder is not hardened against corrupt input beyond basic checks (a chunk header must
  match the array's chunk shape before anything is allocated; integer arithmetic on corrupt residual streams wraps
  instead of overflowing). The chunk formats have no payload checksum: a corrupted payload can decode to different
  values without an error ([Towards 1.0](#towards-10)).

### Planned

* A recipe stage that stores the forcings of a chain as recipes; it needs the valid dates and a bit-identical
  evaluation of the frozen recipe formulas.
* A hook in anemoi-datasets that expands a `physics` block itself, which would remove the `physics expand` step and
  the risk of a stale variable list.
* Decoding the lattice codec's fields straight into their places in the full date, which would remove the chain's
  extra write (a decoder change, not a format change).
* Sidecars read through the zarr store itself, for stores on object stores.
* Distribution wheels (manylinux with a vendored libgomp) and a NEON or SSE4.1 kernel variant.
