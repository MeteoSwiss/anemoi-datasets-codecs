# Design

How the package is put together and why. The package has three Zarr 3 codecs: the ICON lattice codec
(`icon_lattice` / `icon_lattice_bytes`, [codecs/icon_lattice.md](codecs/icon_lattice.md)) and two grid-agnostic field
stages that run ahead of it in a chain, `static_ref` ([codecs/static_ref.md](codecs/static_ref.md)) and `field_terms`
([codecs/field_terms.md](codecs/field_terms.md)), which talk through the chain envelope
([codecs/envelope.md](codecs/envelope.md)). Goals, known limitations and plans are in [goals.md](goals.md).

Contents: [Architecture](#architecture) · [Engine and Zarr adapters](#engine-and-zarr-adapters) ·
[Field stages and codec chains](#field-stages-and-codec-chains) · [Per-field modes](#per-field-modes) ·
[The plan and the physics layer](#the-plan-and-the-physics-layer) ·
[Sidecars and content addressing](#sidecars-and-content-addressing) · [Threads](#threads) ·
[Build and kernel dispatch](#build-and-kernel-dispatch) · [What is in which codec](#what-is-in-which-codec) ·
[Where modularity lives](#where-modularity-lives) · [Design decisions](#design-decisions) ·
[Findings that shaped the design](#findings-that-shaped-the-design)

## Architecture

```
zarr-python 3 / anemoi-datasets
  anemoi_datasets_codecs.zarr3            generic adapters: ChunkArrayBytesCodec, ChunkBytesBytesCodec (chunk
                                          engines), FieldStageBytesCodec and the FieldStage engine base (stages);
                                          SidecarPipeline, zarr's pipeline plus the store it serves
  anemoi_datasets_codecs.envelope         the chain envelope (frozen format, codecs/envelope.md)
  anemoi_datasets_codecs.sidecar          the sidecar search path, shared by all codecs (`.codecs/`)
  anemoi_datasets_codecs.statics          static-field references and capture policy, shared by all codecs
  anemoi_datasets_codecs.config           bounds and clip matching, shared by the codecs that take bounds
  anemoi_datasets_codecs.physics          everything that knows names: conventions, terms, forcings, and
                                          `expand` (the chain of a dataset's configuration)
  anemoi_datasets_codecs.plan             the resolved plan the lattice codec consumes, decode-DAG utilities
  anemoi_datasets_codecs.chain            codec chains of stored arrays for the tools: walk, check, decode chunks
  anemoi_datasets_codecs.cli              the anemoi-datasets-codecs command
    field_terms.zarr3                     FieldTermsCodec ("field_terms", a field stage, first in a chain)
      field_terms.FieldTerms              the engine (stage.py), its arithmetic (arith.py), frozen header (header.py)
    static_ref.zarr3                      StaticRefCodec ("static_ref", a field stage)
      static_ref.StaticRef                the engine (stage.py); its frozen header (header.py, codecs/static_ref.md)
    icon_lattice.zarr3                    IconLatticeCodec ("icon_lattice"),
                                          IconLatticeBytesCodec ("icon_lattice_bytes")
      icon_lattice.LatticeCodec           the engine (codec.py): configuration, plan, encode / decode orchestration
        config.py                         configuration schema: variables, bounds, clip, options
        fields.py                         field index arithmetic; mask-term fit, RECIPE fields
        recipes.py                        recipe ids, inputs and date coefficients (format level)
        encoder.py, rdoq.py               LATTICE field encoder (numpy, in four phases), the optional rdoq path
        container.py                      container byte format, LATTICE header, chunk inspection
        structure.py                      structure model: steps, sha, decoder hints, range split, cache
          structures/                     the frozen builders: tree.py (tree, tree_b), hc2.py (hc2_k3), common.py
        details.py                        spatial details of static cross sources
        layout.py, grids.py               the triangle lattice, grid references and lattice loading (readers)
        lattice.py, gridfile.py           lattice recovery from cell centres, ICON grid files (encoder side)
        statics.py, sidecar.py            the package-level statics and sidecar, plus the lattice codec's parts
                                          (static details cleared with the arrays; the bundled lattices)
        native.py, threads.py             ctypes binding and ISA choice, thread policy
          _native (csrc/)                 C kernels: decode.c, rans.c, encode.c (AVX2 + scalar variants), dispatch.c
        store.py                          attach / check on local stores (zarr only for chained arrays)
```

The stages and the package-level modules import nothing of `icon_lattice` (a test checks it); the physics layer
imports the lattice codec's recipe table only when it resolves a plan.

The engine is a gather-structure prediction codec: the quantiser, the closed-loop step coder, the context tables, rANS,
the container and the C decoder work on any `structure.Structure` (an ordered list of delta and gather steps over the
cells) and know neither the grid nor the variables. The ICON triangle lattice enters in four places only: the lattice
layout (`layout.py`, `grids.py`), the grid-file tools (`lattice.py`, `gridfile.py`, encoder side), the structure
builders (`structures/`) and, as an optional decoder hint, the regular-group fast path (`csrc/group.h`, used when a
structure's steps have that shape).
The variables enter only through the plan.

## Engine and Zarr adapters

`LatticeCodec` is a plain class that codes a whole chunk: `encode(x, dates=None) -> bytes`, `decode(buf, out=None)
-> float32 array`, `get_config()` (the canonical configuration, small JSON), `from_config`, plus
`validate_chunk_shape`, `names` and `fields_per_date` for the adapters. It knows nothing about zarr and can be used
on its own (tests, benchmarks, tools that handle chunks directly).

`anemoi_datasets_codecs.zarr3` turns any such engine into Zarr 3 codecs. A codec package subclasses the two adapters
with two class attributes, `codec_name` (the name in `zarr.json` and in the `zarr.codecs` entry point) and `engine`:

```python
class IconLatticeCodec(ChunkArrayBytesCodec):
    codec_name = "icon_lattice"
    engine = LatticeCodec
```

The adapters build the engine from the metadata's `configuration` (`engine(**configuration)`), write
`{"name": codec_name, "configuration": engine.get_config()}`, check the dtype and the chunk shape when an array is
created, and run encode / decode in a worker thread (`asyncio.to_thread`), so zarr's concurrent chunk reads decode in
parallel when the engine releases the GIL (the C decoder does). `ChunkArrayBytesCodec` takes the place of the
`bytes` serializer; `ChunkBytesBytesCodec` follows the default `bytes` codec (little endian) for tools that only let
one choose compressors. Both write the same chunk bytes. Neither copies the chunk: the decoded array goes to zarr as a
view (an NDBuffer, or a byte view that the `bytes` codec views back as an array), so both forms read equally fast;
on encode the engine gets a view of zarr's buffer (it can be the caller's array) and must not write into it.

The lab (`anemoi_datasets_codecs.lab`, [lab.md](lab.md)) works on the engine from outside: it imports the engine,
the engine never imports it. Its one hook is `EncodeOptions.record` (`LatticeCodec.encode(..., record=)`), a
callback that receives the pieces of every LATTICE coding: instrumentation for the lab, never configuration. It is
not serialised (not in `get_config()` or a chunk), and the bytes do not depend on it.

## Field stages and codec chains

A *field stage* is a bytes-to-bytes codec that rewrites the float32 field array of a chunk, `(dates, fields,
ensembles, cells)` with the field axis 1, before the lattice codec (or any other codec) sees it: it drops fields
or replaces some by residuals, and keeps per-chunk metadata for its decoder. anemoi-datasets selects bytes-to-bytes
codecs only (`output.compressors`), so a chain is `compressors=[stage, ..., icon_lattice_bytes]` after the default
`bytes` codec. The engine is a `zarr3.FieldStage` subclass (no zarr in it); `zarr3.FieldStageBytesCodec` is its
adapter, subclassed with `codec_name` and `engine` like the chunk adapters.

**The envelope** ([codecs/envelope.md](codecs/envelope.md)). zarr gives a bytes-to-bytes codec no channel but the
bytes, validates every codec against the array's original chunk grid (not the shape earlier codecs leave), and its
two pipelines disagree on the spec a bytes-to-bytes codec gets after another one announced a new shape. So stages
keep `resolve_metadata` the identity and wrap the chunk in an in-band envelope: magic `FCHN`, the shape of the field
array, a stack of tagged headers (stage id = codec name, payload) and the payload, 8-byte aligned. On encode a stage
unpacks a bare chunk (shape from `chunk_spec`) or an envelope, runs its engine and packs the result with its header
pushed; on decode it pops its header (checking the id), runs its engine and packs the result with the rest of the
stack, or, when the stack is empty (it was the first stage), hands zarr the bare chunk, which must have exactly the
chunk spec's shape. Each stage checks the envelope's dates, ensembles and cells against `chunk_spec` and its field
count against its configuration, so a misordered or wrongly configured chain fails at the first chunk naming the
stage; at array creation it can only check `(dates, fields, ensembles, cells)` chunks that hold every field of a
date.

**Copies.** `envelope.pack` copies the stage's output once on encode. On decode nothing is copied in the common case:
the lattice codec decodes straight into the envelope it hands on (`envelope.allocate`), a stage that works in place
re-labels that envelope with the shorter stack (`envelope.repack_in_place`), a stage that changes the shape is given
`out`, the array inside the next envelope, and the first stage hands zarr a byte view. A stage that gets a read-only
buffer (from a compressor after it) works on one copy.

**The lattice codec in a chain** (`chain: true`, [codecs/icon_lattice.md](codecs/icon_lattice.md)). The chunk grid
then has the field count of the array, not of the lattice configuration, so `validate_chunk_shape` checks only the
cells and ensembles (and the adapter whole dates); the field count is checked per chunk against `variables`. An
enveloped input is coded inside its envelope (`envelope(shape, headers, LATC container)`, the headers untouched) and
decoded back into the envelope of the field array; a bare input gives the bare container, exactly the bytes without
the key. Without `chain` nothing changes: no input is sniffed, and the parse-time field-count check stays (an
enveloped input there fails with an error that names the key). A lattice configuration without the static fields
resolves the plan of the full configuration minus every static source (a test pins it), so it needs no `plan` key.

**The stage contract.** What `FieldStage` engines must honour (the adapter checks the shapes):

* *Inputs.* `encode(x)` gets the field array of `fields_in` fields (`variables`, in chunk order, named in the
  configuration) read-only: it can be the caller's array. A stage may assume only what the codecs before it
  guarantee: exact values where they are exact, and nothing about fields it does not own. Sources a stage computes
  from must reach its decoder bit-identically (static fields, lossless fields, never a lossy or RECIPE field), so
  stages read only exact inputs and decode from exact inputs.
* *Outputs.* `encode` returns `(y, header)`: `y` with the same dates, ensembles and cells and `fields_out`
  (`variables_out`) fields, and the header as bytes (or None). The header travels in every chunk through every later
  codec and is copied once per stage: per-chunk metadata of bytes to kilobytes; large data (static arrays) belongs
  in the sidecar, with a sha in the header. The envelope allows up to 4 GiB per header.
* *Decode.* `decode(y, header, out_shape, out=None)` gets the array the stage emitted (as the codecs after it
  restored it), writable and owned by the call, so it may decode in place; `out_shape` is the shape to return, and
  when it differs from `y.shape`, `out` is an uninitialised array of that shape inside the next envelope (decoding
  into it saves a copy). The result must be within the bounds the configuration promises given the codecs after it
  (a stage ahead of a lossy codec reduces the bound it hands on by a margin written in both configurations).
* *Determinism.* Encoding is deterministic for a package version, configuration and numpy; decoding is deterministic on
  every machine (float64 arithmetic rounded to float32, no BLAS, no fused multiply-add).

**Static references** (`static_ref`, [codecs/static_ref.md](codecs/static_ref.md)). After `field_terms`: it drops the
static fields of a chunk and writes per `(date, static, member)` the sha256 of the array into its header, capturing an
array it has not seen into the sidecar exactly as the lattice codec's STATIC_REF mode does (the same
`anemoi_datasets_codecs.statics` code, rule and version limit). The sidecar machinery is package level and shared: one
directory name, `.codecs/` (format level), one search path, one file format. An array that cannot be referenced goes
inline into the header rather than on as a field, so the field count the next codec sees is fixed by configuration (the
lattice codec cannot know which field is which otherwise; with `inline: true` every static is inline). Decode is the one
full-chunk write a stage that changes the field count cannot avoid, into the `out` the adapter passes. The tools learn
chains through `anemoi_datasets_codecs.chain`: the codec classes come from zarr's registry (any installed stage is
known), a stage engine may offer `header_references(header)` (the sidecar files its header references: `attach` and
`check` copy and verify them) and `describe_header(header, configuration, shape)` (`info`), and a chunk decodes through
the engines alone (`attach` and `check` verify one from the store's sidecar). `check` compares the store's variables
with the first codec's of the chain and checks the chain's consistency (below).

### The `field_terms` stage

`anemoi_datasets_codecs/field_terms/` ([codecs/field_terms.md](codecs/field_terms.md)), grid-agnostic, first in a
chain. It replaces lossy targets by residuals against a per-chunk robust linear fit on exact sources (the six
orography terms on `FIS` that the lattice codec loses when the static fields leave it): per (date, member) slice an
IRLS fit in float64, float32 coefficients in its header, `pred = fl32(c0 + sum c_i s_i)` evaluated in float64 in a
fixed order, `r = fl32(x - pred)`; decode in place, `x' = clip(fl32(r' + pred))`. The bound in short: the codec
after it codes a target within a static reduced bound `b_L = b (1 - m)` (default `m = 2^-6`, written in both
configurations, so the lattice codec reads nothing from the stage's header); the reconstruction is monotone in `r'`,
so the encoder checks per cell that both float32 ends of `[r - b_L, r + b_L]` (rounded outwards) reconstruct within
`b`, and leaves a target's slice untouched (a bypass flag in the header) when any cell fails, which the lattice then
codes at `b_L <= b` and the stage clips on decode (it owns its targets' clips). Sources must be exact (no bound, not a
target, not a forcing recipe), checked at configuration time. A target the lattice stores LOSSLESS is within the
bound but not bit-exact, and its exact zeros are not kept.

### Writing and checking a chain

`physics.expand` (`anemoi-datasets-codecs physics expand`) writes the three configurations of a chain from the
dataset's variables (in the dataset's order: anemoi sorts them by name), the physics group and a lattice codec
configuration: `field_terms` gets the convention's exact-source terms (the orography terms, by role, for the targets
that are present and have a bound) with their bounds, clips and the margin; `static_ref` the static fields; the lattice
codec the other fields, the stage targets at `b_L` and without a clip, no static fields, `chain: true` and the physics
group as given. Nothing in the lattice configuration comes from a stage header. Without static fields to reference the
result is the single lattice codec.

zarr and anemoi build each codec of a chain on its own, so no codec can check what depends on another one.
`chain.chain_problems` checks it from the configurations (`check` runs it on a store, or on a JSON file of codec
entries): each stage emits the next codec's variables and a chunk engine behind stages has chain mode; every
`field_terms` target reaches the lattice codec with a bound of at most its `b_L` (or none) and without a clip; every
`field_terms` source is exact through the codecs after it (restored by a stage that drops it, such as `static_ref`, or
in the lattice codec without a bound and not a recipe of its resolved plan, and not a later stage's target). The tools
undo generic compressors at the end of a chain (`blosc`, `zstd`, `gzip`, `numcodecs.*`) before they read the envelope.

## Per-field modes

A chunk is coded field by field (one `(date, variable, member)` slice each) and every field gets one mode: LATTICE
(lossy: quantisation, prediction structure, cross terms, rANS), LOSSLESS (the configured numcodecs codec), CONSTANT (one
float32), STATIC_REF (a sha256 reference into the sidecar) or RECIPE (an analytic forcing from the valid date). The
modes are what makes a mixed chunk small: stored in every chunk, the static fields can be a large part of a coded date
even when well compressed (more than half on the ICON-CH1 benchmark dataset, [benchmarks.md](benchmarks.md#size)), and
as references they cost a few dozen bytes each; a recipe costs 64 bytes plus its patches. The mode is chosen per field
and per chunk by the encoder, with fallbacks (a lossy field that fails verification, or whose lossless coding is
smaller, is stored losslessly); the decoder only reads the record. In a chain the static fields never reach the lattice
codec (`static_ref` takes them), so it writes no STATIC_REF records and, without the latitude / longitude inputs, no
RECIPE records; its decoder keeps reading both for the single-codec stores.

## The plan and the physics layer

The core (structures, quantiser, entropy coder, container, statics) and the stages are variable-agnostic. Everything
that knows variable names or physics lives in `anemoi_datasets_codecs/physics/` (package level, since it configures
every codec of a chain):

* `conventions.py`: name conventions (roles -> names, level families with `{level}` patterns, ordered levels,
  vertical coordinate, pressure scale, the shallow-geopotential star level, forcing roles). Built in: `icon-ch1`,
  `ifs`; custom dicts may extend them. New naming conventions are data, not code.
* `terms.py`: the candidate cross-variable terms (surface, inter-level, hydrostatic) and below-ground mask terms,
  written against roles and level indices. New kinds of terms go here.
* `forcings.py`: which variables are recipes. The recipe ids, their inputs and the date-dependent coefficients are
  part of the format and live in the core (`recipes.py`), so a plan with recipes works without the physics layer.
* `expansion.py`: `expand`, the chain of a dataset's configuration (above), and `terms.orography_terms`, the
  exact-source terms it gives `field_terms`.
* `__init__.py`: the `physics` configuration group and `resolve(names, physics) -> Plan`.

The layer produces a `plan.Plan`: `cross` (target -> sources), `masks` (target -> mask source, threshold,
features), `recipes` (target -> recipe id, inputs) and a `report` of every candidate term with the sources applied
and dropped and why. `plan.finalize` applies the decode-depth limit (`max_depth`) and the 4-source cap; the codec
turns names into field indices, and only indices reach the chunk. A variable set with any subset of names works:
missing sources drop terms, nothing fails. Turning the layer off (`physics: off`) leaves a working codec (the core
only).

The physics layer is the default producer of the plan, not the only one: `LatticeCodec(..., plan=Plan(...))` takes a
candidate plan directly (checked against the variable names; `finalize` still applies the codec's limits). That is for
experiments and tools: the plan is not part of the configuration, `get_config()` is the same as without it, and a store
written with a given plan is read back (correctly: the decoder reads the terms from the chunk) by a codec whose encoder
would use the physics plan of the configuration. A given plan survives pickling, not `from_config`. Storing it would
be possible (a plan is a few KiB of names as JSON) but needs a new configuration key, a schema change not made.

## Sidecars and content addressing

The configuration must stay small JSON (it is in `zarr.json`, and anemoi builds the codec when it parses a recipe), but
the decoder needs reference data far larger than that (the lattice and the static fields, full arrays over the grid),
and readers must never need the ICON grid file. Everything a chunk only references is therefore content addressed by
sha256 and lives in sidecar directories: `grids/<sha256>.npz` (lattices) and `statics/<sha256>.f4.blosc` (static
fields), inside a store as `<store>/.codecs/` (not a zarr node, so zarr ignores it; it moves with the store
directory), in the user cache, on `$ANEMOI_DATASETS_CODECS_PATH`, and, for lattices of bundled grids, in the package.
Content addressing makes every source safe (every file is verified when read), lets stores share one sidecar directory,
and makes "is this already there" a file lookup. The encoder captures what it has not seen into the store target and the
cache (a static only into the store target when there is one), and references a static only once it is durably stored.
zarr tells no codec which store it serves, but it tells the codec pipeline: `zarr3.SidecarPipeline`, zarr's default
pipeline plus the store, is selected through zarr's `codec_pipeline.path` setting when the codecs are imported (unless
another pipeline is configured, or `ANEMOI_DATASETS_CODECS_PIPELINE=off`) and serves `<store>/.codecs/` to the codecs
of a local store during every read and write (`sidecar.serving`, a context variable, so arrays of different stores in
one process do not mix: each store captures its own statics). `register_store` and `ANEMOI_DATASETS_CODECS_PATH`
remain for sidecars elsewhere; stores that are not local directories get no sidecar from the pipeline
([goals.md](goals.md#known-limitations)).

## Threads

`threads.py`: `ANEMOI_DATASETS_CODECS_THREADS`, else 1 inside a torch DataLoader worker (only when torch is already
imported), else min(16, CPUs available). Read at every call, never stored in the configuration. The decoder runs one
OpenMP team per chunk (fields as tasks, ranges as taskloops); the encoder uses a thread pool over the fields of a DAG
wave. A process forked after its parent decoded with several threads decodes with 1 thread (GNU libgomp's pool does
not survive `fork()`; a multi-threaded decode there would hang).

## Build and kernel dispatch

`setup.py` compiles `csrc/variant.c` (a unity build of `rans.c`, `decode.c` and `encode.c`, all functions static) twice:
as the `avx2` variant (`-mavx2`, plain AVX2 baseline, no FMA, no `-march=native`) and the `scalar` variant
(`-DLC_SCALAR`: `csrc/simd.h` implements the AVX2 intrinsics the kernels use in portable C on GCC / clang generic
vectors), and links both with `csrc/dispatch.c` into the stable-ABI (abi3) extension `icon_lattice._native`, loaded with
ctypes. The variant is chosen at load time from cpuid (`ANEMOI_DATASETS_CODECS_ISA` overrides; `lc_select_isa` switches
at run time for tests). On non-x86 platforms only the scalar variant is built. Flags: `-O3 -fopenmp -ffp-contract=off`;
without fused multiply-add the dequantisation and the recipe / mask formulas round exactly like numpy, so the encoder
verifies the bound on exactly the values every build decodes. With gcc the directory of the compiler's libgomp is added
as an rpath.

## What is in which codec

Zarr 3 describes a chunk's coding as a codec list (array -> array codecs, one array -> bytes codec, bytes -> bytes
codecs), each codec seeing only the previous codec's output. The package splits along the line where that model fits:
work that needs only exact inputs is a stage; work that needs the decoded values of lossy fields stays inside the
lattice codec.

* **Static fields: a stage (`static_ref`).** A static field needs nothing the lattice codec produces, is exact, and is
  the same in every chunk: the stage replaces it by a reference before the lattice codec sees the chunk, and any codec
  can follow (`[static_ref, blosc]` works as well).
* **Exact-source terms: a stage (`field_terms`).** The orography parts of the surface and lowest-level geopotential
  terms (PMSL, FI_1000, T_2M, PS, SKT, TD_2M on FIS) have one exact source, so an open-loop fit ahead of the lattice
  codec loses nothing against the closed loop (measured on ICON-CH1: the open loop was even slightly better for
  FI_1000). They leave the lattice codec together with the statics.
* **Terms with lossy sources: inside the lattice codec, closed loop.** Inter-level, hydrostatic and most surface
  terms predict from other lossy fields. Inside the codec they predict from the source's *decoded* spatial detail,
  so the quantisation error of the source does not enter the target's budget. A stage ahead of the codec cannot see
  decoded values; an open-loop version must reserve the source's worst case in every cell of the target: on ICON-CH1
  the best open-loop variant cost +6.7% per date at `tree` k = 1 (+7.2% at `tree_b` k = 3) and a deterministic rule
  +8.0% (+8.2%) against the closed loop, recovering less than half of what the cross terms are worth. These terms
  stay in the lattice codec, which resolves its plan from its own variables: without the static fields there, the
  default resolution drops every static source (the FIS-free plan; a test pins it), so no plan travels in the
  configuration.
* **Forcing recipes: inside the lattice codec, deferred for chains.** A recipe needs the valid date and the static
  latitude / longitude fields. Zarr passes no dates to a codec; the single codec recovers the recipe coefficients from
  the forcing fields themselves with `physics.recipe_fit` (on by default), while a chain's lattice codec has no latitude
  / longitude and stores the forcings losslessly unless `physics expand --forcing-bound` gives them a bound. A recipe
  stage would need dates from somewhere and a bit-identical evaluation of the frozen formulas; it is not written
  ([goals.md](goals.md#planned)).
* **The lattice codec itself stays one array-to-bytes codec.** Its work is a DAG over fields, not a sequence of
  stages: a field's prediction uses the int16 details of up to 4 other fields, and the decoder runs every field as an
  OpenMP task with `depend` clauses, rANS decoding, prediction and residual application fused in L1-sized blocks, the
  finest level written straight to the float32 output. Separate codecs per stage (quantise, predict, entropy-code)
  would serialise the stages over the whole chunk and materialise intermediates as large as the chunk; the
  per-field parameters (weights, cross coefficients, contexts, tables) live interleaved in the chunk; and the layout
  is a versioned format (container version 3).

The chain has costs: a decoded chunk passes through each stage (`static_ref` writes the whole chunk once, `field_terms`
works in place on its targets), and each stage adds a frozen header format. Measured costs and sizes are in
[benchmarks.md](benchmarks.md#codec-chains).

## Where modularity lives

* **Codecs and stages.** A new codec is a new subpackage with an engine and adapter subclasses of
  `anemoi_datasets_codecs.zarr3` (a chunk engine: `encode`, `decode`, `get_config`, `validate_chunk_shape`; a field
  stage: `FieldStage`), registered under `zarr.codecs` in `pyproject.toml`. Stages compose with any codec after them;
  the envelope carries their headers through codecs that do not know them.
* **The physics layer.** All name knowledge is in `anemoi_datasets_codecs/physics/`: it resolves the lattice codec's
  plan and writes the stage configurations (`expand`). Conventions are data; new kinds of terms are entries in
  `physics/terms.py`; neither the lattice core nor the stages change for either. A plan can also be given to the
  lattice codec directly (`plan=`, above).
* **Structures as data.** A structure is a list of steps (targets, sources, default weights) built from the lattice
  and identified by its sha256 in every chunk. The engine is generic over it (`structure.py` is the model); the
  builders are the frozen ICON part (`structures/tree.py`, `structures/hc2.py`, registered by id in
  `structures.STRUCTURES`; a test pins their shas on the synthetic lattice). A new structure is a builder under a new
  id; existing ids are frozen.
* **One module per concern.** `codec.py` only orchestrates: the field index arithmetic, the mask-term fit and the
  RECIPE fields are `fields.py`, the static reference policy `statics.StaticCapture` (shared with `static_ref`), the
  static details `details.py`, the LATTICE header `container.lattice_header`, the date conversion
  `container.to_seconds`; the LATTICE encoder runs in four phases (step coding, tables and streams, serialisation,
  verification) with the optional rdoq path apart (`rdoq.py`). No module touches another module's private state: the
  process caches of resolved references register with `sidecar.isolated`, which `attach` / `check` use.
* **The lossless sub-codec.** `lossless` is any numcodecs codec configuration (default Blosc lz4); it is part of the
  codec configuration, not of the chunk.
* **The sidecar.** Reference data (lattices, static fields) is content addressed and outside the configuration, so a
  new grid is a lattice file (`grid build`), not a code change, and every codec shares one sidecar.

## Design decisions

* **Configuration** (`icon_lattice/config.py`): small JSON, complete at recipe-parse time: `variables` (chunk order),
  `bounds` (names or full-match regex; every key must match a variable, so typos fail at parse time; the matching is
  `anemoi_datasets_codecs.config`, shared with `field_terms`), `clip`, `static_fields`, `grid` (alias, uuid or `{name,
  uuid, n_cells, sha256}`), `n_ensembles`, `structure`, `lossless`, `options`, `physics`, `chain`; schema `version` 1.
  The stage configurations are small JSON as well and name their variables (anemoi reorders variables; indices would hit
  the wrong fields). `get_config()` writes the resolved canonical form (exact names, resolved bounds, full grid
  reference, every option and physics key with the defaults spelled out), so the metadata says exactly what every
  variable got and a later version, whose defaults may differ, encodes new chunks of the store the same way; a few KiB
  ([benchmarks.md](benchmarks.md#size)). The Zarr adapters add the writer stamp `written_by`, the package version that
  created the array metadata, which no engine reads and no comparison of configurations uses. The decoder uses only
  `grid`, `structure`, `lossless` and the field count; the codec is cheap to construct (nothing is loaded before the
  first encode / decode) and picklable (its state is its configuration).
* **Codec names** `icon_lattice` (array -> bytes), `icon_lattice_bytes`, `static_ref` and `field_terms` (bytes ->
  bytes) are format-level: stores depend on them, so they do not change with the package name, like the sidecar name
  `.codecs/` (shared by every codec), the container (`LATC`) and the envelope
  (`FCHN`). A stage's codec name is also the id of its header in the envelope.
* **A chain, not a fused stage codec.** `static_ref` and `field_terms` are separate codecs: each is useful alone
  (`[static_ref, blosc]`), each has one small frozen header, and `field_terms` decodes in place, so the separation
  costs one envelope re-label per stage on decode ([benchmarks.md](benchmarks.md#codec-chains)).
* **A static reduced bound, not a per-chunk handoff.** The lattice codec codes a `field_terms` target within `b_L = b
  (1 - m)`, written in both configurations; the stage verifies per cell at encode time that every value the lattice
  may return within `b_L` reconstructs within `b`, and leaves a slice untouched when it does not. The lattice codec
  reads nothing from a stage header, so its stored configuration says exactly what it did.
* **Statics**: content addressed `statics/<sha256>.f4.blosc` in sidecar directories, never in the configuration (as
  base64 every static field would add megabytes to the metadata). The encoder (`static_ref`, or the lattice codec's
  STATIC_REF mode in a single-codec store) captures a static field when it is not available, references it only when the
  file is durably stored (with a store target: in that store), else stores it inline (`static_ref`) or losslessly (the
  lattice codec) with a warning; at most 4 values per name. The decoder searches the process cache, inline bytes,
  registered stores, the path, the cache, the package (bundled lattices); missing data raises `MissingReference` naming
  the sha and the searched directories, never a silent fallback. `attach` makes a store self-contained after the fact
  and verifies it.
* **Grids and structures** (`grids.py`, `gridfile.py`, `structure.py`, `structures/`): the reader resolves the lattice
  by sha256 (`.npz` readable with numpy alone, bundled for ICON-CH1 and copied into store sidecars by the encoder) and
  builds the structure from it (in seconds, [benchmarks.md](benchmarks.md#grid-and-structures); cached on disk in the
  cache directory, `off` rebuilds per process). The grid file is an encoder-side input (`grid build`, validated against
  its neighbour table).
* **Build** (`setup.py`): setuptools with a custom `build_ext`; two variants and a dispatcher in one abi3 extension;
  `-ffp-contract=off` (see above and the `setup.py` docstring).
* **Threads**: the DataLoader-aware default and the fork clamp (above).
* **Python >= 3.11**: the stable ABI from 3.11 gives one extension per platform for every later CPython.
* **Zarr 3 only.** No numcodecs codec class and no Zarr format 2 metadata: one code path for one target, zarr-python 3,
  which anemoi-datasets uses to write Zarr format 3 (`output.format: zarr3`, from anemoi-datasets 0.5.50).
  `LatticeCodec` is the engine, not a numcodecs codec; numcodecs stays a dependency for Blosc (the default lossless
  codec and the sidecar static files). The configuration lives only in `zarr.json`, and the physics settings only in
  `physics` (an unknown option is an error that names the physics key to use).

## Findings that shaped the design

* **anemoi-datasets `output.compressors`** (anemoi-datasets >= 0.5.50): a list written as Zarr 3 codec specs
  `{name, configuration}`; format 2 resolves `numcodecs.<id>` with numcodecs and takes a single compressor, format 3
  accepts **bytes -> bytes codecs only** (an array -> bytes codec is rejected there by design). Hence the second codec
  `icon_lattice_bytes`; `icon_lattice` is the natural form for direct zarr use.
* **anemoi-datasets writes Zarr format 2 by default.** Format 3 needs `output.format: zarr3` and zarr-python >= 3.4.1,
  which anemoi-datasets installs only on Python >= 3.12; its 0.5.50 release supports Python up to 3.13.
* **libgomp and fork**: a decode with several threads in a process forked after its parent decoded with several
  threads hangs (GNU libgomp's thread pool does not survive `fork()`; found with `scripts/fork_check.py`). The
  decoder clamps to 1 thread in such a child with one `RuntimeWarning`; a test checks it.
* **DataLoader oversubscription**: in a loader replica a 16-thread decoder per worker gave no gain and oversubscribed
  the node; one codec thread per worker is the default inside DataLoader workers.
* **zarr gives the store to the pipeline, not to the codecs**: a codec sees bytes and an `ArraySpec`; the codec
  pipeline of an array is built with `CodecPipeline.from_array_metadata_and_store` (zarr >= 3.1, made for the Rust
  pipeline) and runs every read and write, so it is the one place that can tell the codecs which store they serve.
