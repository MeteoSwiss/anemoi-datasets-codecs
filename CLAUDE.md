# CLAUDE.md

See @README.md for the project overview and @docs/contributing.md for the development setup, the tests, the
invariants and their checks, and the conventions. This file adds what an AI coding agent needs on top.

Before changing anything read [docs/goals.md](docs/goals.md) (purpose, promises, known limitations),
[docs/design.md](docs/design.md) (architecture, what is in which codec), [docs/codecs/](docs/codecs/) (how each codec
works, the versioned formats) and [docs/user-guide.md](docs/user-guide.md) (configuration, usage).

## Sources of truth

* Behaviour: `src/`, then `tests/`. Never state a flag, configuration key or number you have not seen there.
* `README.md` and `docs/` are kept consistent with the code in the same change; measured numbers go in
  `docs/benchmarks.md`, with their setup, and other documents quote them from there.

## Invariants (do not break; details in docs/contributing.md)

* **Formats change only with a new version**: lattice container version 3 (`LATC`), chain envelope version 1
  (`FCHN`), the `static_ref` and `field_terms` headers version 1, the structures `tree`, `tree_b`, `hc2_k3`, the
  sidecar files. A layout or decoder change is a new version in its document, never a silent change; new structures
  get new ids. Without `chain: true` the lattice codec never reads an envelope.
* **Existing chunks decode bit-identically** on both kernel variants: the golden chunks
  (`tests/data/golden_chunks.json`) and the pinned-header tests guard it. **Golden chunks are never regenerated or
  changed.**
* **Byte baselines are a regression guard, not a promise.** After any change to the encoder, plan, physics,
  structures or C kernels run `uv run pytest -m "not realdata"` (with `ANEMOI_DATASETS_CODECS_ISA=scalar` too) and
  `scripts/plan_baseline.py check tests/data/chunk_sha256_baseline_numpy-2.5.4.json`; they must stay green for a
  change not meant to change encoder output. A deliberate encoder or default change adds baseline entries
  (`plan_baseline.py add`; existing settings pin their options) in the same change and is listed in CHANGELOG.md under
  `### Encoder changes`. **Never regenerate a baseline to hide an unintended change.** `scripts/baseline_chunks.py
  check` and `scripts/bench/bench_encode.py check` need the maintainers' ICON-CH1 sample and baselines (not in the
  repository): say so if you cannot run them.
* **Every decoded value is within its variable's bound**, verified per cell at encode time by the codec that codes it;
  decoding is deterministic across machines, thread counts and the two kernel variants. Stage targets are within the
  bound, not bit-exact.
* **The codecs know no variable names.** Everything name- or physics-specific lives in
  `src/anemoi_datasets_codecs/physics/` and reaches the lattice core as a resolved `plan.Plan`, the stages as plain
  configuration (`physics.expand`). The stages and package-level modules import nothing of `icon_lattice`.
* **Configurations stay small JSON** (stored complete in `zarr.json`, every option and physics key, plus the writer
  stamp `written_by`); a released key keeps its meaning; large data lives in the content-addressed sidecar
  `<store>/.codecs/`. Nothing in one codec's configuration is read from another's chunk headers;
  `chain.chain_problems` checks that the configurations of a chain agree.
* **Zarr 3 only.** No numcodecs codec, no Zarr format 2 metadata. Format-level names (`icon_lattice`,
  `icon_lattice_bytes`, `static_ref`, `field_terms`, `.codecs/`, `LATC`, `FCHN`) never change.

## Gotchas

* C kernels (`src/anemoi_datasets_codecs/icon_lattice/csrc/`): after editing run
  `uv sync --reinstall-package anemoi-datasets-codecs`. `-ffp-contract=off` is required (numpy-exact rounding).
* `realdata` tests need the ICON-CH1 sample store (`ANEMOI_DATASETS_CODECS_SAMPLE`) and grid file
  (`ANEMOI_DATASETS_CODECS_GRID_FILE`), which are not distributed; they skip without them.
* The anemoi-datasets tests run in `.venv-anemoi3`, which `scripts/make_envs.sh` creates; they skip elsewhere.
* `test_chunk_bytes_identical_to_baseline` runs only with numpy 2.5.4 (the baseline's version); its below-ground
  setting and `plan_baseline.py check` need OpenBLAS Haswell or Zen kernels (`OPENBLAS_CORETYPE=Haswell` elsewhere).
* A new codec is a subpackage with an engine and adapter subclasses (`zarr3.py`), registered under `zarr.codecs` in
  `pyproject.toml`.
* Docs and comments state facts about this package; nothing site-specific. Markdown wrapped at 120 columns.
