# The lab: a research toolkit for the codecs

`anemoi-datasets-codecs lab ...` (package `anemoi_datasets_codecs.lab`) holds the tools for working *on* the codecs
rather than with them: where the bits of a chunk go, what an idea would buy before it is built, how the codec stands
against reference compressors at the same error bound, and the plumbing that keeps experiments comparable and the
decoder honest. Nothing in the lab is on the encode or decode path of a store: the lab imports the engine, the engine
never imports the lab (its one concession is an optional recording hook, `EncodeOptions.record`, that leaves the bytes
unchanged), and no tool changes a format. The measurements are what decide whether a format version 4 or a new
codec is worth building. The lab is experimental: its commands and their output may change between releases.

Every tool takes the lattice codec configuration (`CONFIG`: a JSON / YAML file, a codec entry, a chain, or a store)
and an anemoi store (`STORE`: the `data` array `(dates, variables, ensembles, cells)` and `dates`), and reports per
field in bytes or in bits per cell; `--json` gives the same as JSON. The tools know no variable names: fields are
indices, names only label the output.

## Where the bits go: `account`, `explain`, `diff`, `residuals`

* `lab account CHUNK [--config CONFIG] [--top N] [--steps]`: the chunk file alone. Every field with its mode,
  fallback and bytes; every LATTICE payload split into header, mask block, steps section, tables, rANS streams,
  escapes and padding (they sum exactly to the chunk), with n_tables, prob_bits, escape type and recon_bits; chunk
  totals with the share of each part. With the configuration (needed for the structure; found automatically when the
  chunk sits under `STORE/data/c/...`) `--steps` adds the per-step section: contexts, table map, thresholds, weights,
  cross coefficients, rANS bytes and escapes per range. A chained chunk's lattice container is found inside its
  envelope.
* `lab account --encode CONFIG STORE --date I [--fields A,B] [--fine 32] [--steps]`: re-encodes the date with the
  recording hook and compares what was coded with what the symbols would cost: actual rANS bits, the cost under the
  coded tables, the order-0 entropy per step, the conditional entropy given the encoder's context (the **context
  gain**), the conditional entropy given `--fine` quantile buckets of the spread, per step and with shared thresholds
  (the **headroom** of finer contexts, net of their table bytes), escape bits, stream flushes, bits per target per
  step. Per field: bits per cell, escape share, coder loss, context gain, headroom.
* `lab explain CHUNK --config CONFIG [--fields A,B | --all] [-v]`: why a field ended up as it did: lossless fallback
  (1: the encoder failed, 2: lossless was smaller), the zero rule (plain quantisation for fields with exact zeros),
  forcing recipes stored losslessly, planned cross sources that are absent from the payload or have a zero
  coefficient in every step, the mask term, escape type, recon bits and table layout.
* `lab diff CHUNK_A CHUNK_B [--config CONFIG] [--top N]`: per-field byte and mode changes between two chunks of the
  same variables (two configurations, two versions), the rANS / escape / table deltas and totals per part.
* `lab residuals dump CONFIG STORE --date I --fields A,B OUT.npz [--threads N]`, then
  `lab residuals summary OUT.npz [--json]`: per field in native cell order the quantised integers `n`, the decoded
  integers `v`, the prediction, the residual `q`, the step and context of every cell, the spread and the mask, with
  offset, grid and bound (keys `FIELD/name`, the lattice under `grid/`), for notebooks; `summary` prints the Laplace
  fit against the order-0 cost, percentiles of `|q|`, escape rate, zero fraction, the correlation of `q` and `|q|`
  between lattice neighbours (what a better predictor or context could still use) and the mean `|q|` per step.

## What an idea would buy: `oracles`

`lab oracles {baseline,linear,cross,temporal,contexts,levels} CONFIG STORE [--date I] [--fields A,B|REGEX] [--json]`
estimates offline, in bits per cell, what a change of the codec would gain, without building it. The
estimates quantise as the codec does (plain rule, `recon_bits` 0; `encoder.quantise`), predict open loop along the
configuration's structure with the codec's choice between the default and the fitted sum-to-one weights (the
default weights alone with `options.fit` `fixed`), and cost the residuals with the codec's context (the spread of the
first four sources in 8 buckets) and 16 bits per escape (`lab.entropy.residual_cost_bits`).
`baseline --calibrate matched` encodes the same fields with the codec set to the same coding and prints the ratio of
the estimate to the coded bytes (0.97-0.98 on ICON-CH1; escape-heavy fields are overcounted a few percent, near-empty
fields are dominated by headers).

* `linear`: the Wiener bound: the unconstrained least-squares predictor over the sources (and the sources' own first
  source) against the sum-to-one weights, best of the three per step.
* `cross [--sample N] [--top N]`: for every pair of lossy fields, the gain of regressing one field's residual on the
  other's at the same cells, and of conditioning on the other's residual bucket (a context rather than a predictor);
  the matrix, the top pairs and whether the plan already uses each (`cross_plan`). Static sources are not included.
* `temporal [--prev J]`: the gain of the previous date (the difference predicted spatially, and the best linear mix of
  the spatial prediction and the previous value). Only meaningful between consecutive hours; the tool prints the
  dates it compares.
* `contexts [--no-adaptive]`: the conditional entropy given nothing, the step, the codec's spread buckets, 32 spread
  buckets, the nearest source's residual magnitude, their product, and the symbol stream through `xz -9e` and
  `zstd -19` as the adaptive reference. Table bytes are not charged; the output says what a context would need in
  tables.
* `levels`: cells, data and residual variance and bits per target per step (refinement level): the multiresolution
  profile.

## The ceiling: `refs`

`lab refs CONFIG STORE --date I [--fields A,B] [--methods M,...] [--layouts 1d,2d] [--out OUT.jsonl]` encodes the
date with the lattice codec once (cached next to `--out`) and compresses every lossy variable at the **same absolute
bound** with reference compressors, every result verified against the bound (a lossy method that misses it is retried
with a halved tolerance): `zfp` (`zfpy`, fixed accuracy) and `sz3` (`pysz`, absolute bound), and lossless coders of
the codec's own quantised integers, which measure entropy coding alone at exactly the codec's error: `zstd` (19), `xz`
(9 extreme), `blosc-lz4`, `blosc-zstd`; plus Blosc lz4 on the raw float32 field (anemoi's default). Two layouts: `1d`,
the cells in native order as a generic codec sees an anemoi chunk, and `2d`, the cells on the lattice image with the
holes filled from the nearest cell (the bytes are the whole image). A missing package prints "n/i" and the tool
carries on (without scipy, which fills the image, the whole `2d` layout). The optional packages are the `lab` extra
(`uv sync --extra lab`; `zfpy` must be 1.0.1, the `.post0` release breaks numcodecs' version parsing).

## The fast loop: `minigrid` and `ledger`

`lab minigrid build CONFIG STORE OUT_DIR [--fraction 1/16 | --window r0:r1,c0:c1] [--dates all]` cuts a rectangular
window of the grid's lattice image and writes a self-contained anemoi-style store with the real data of those cells
(`OUT_DIR/mini.zarr`, one date per chunk, its sidecar holding the mini lattice and the cropped static fields), the
configuration on the mini grid (`mini.json`, `grid` replaced), the native indices of the mini cells (`cells.npy`), a
manifest and a README. The window corner is aligned to `max(32, 2^(L+1))` (L the tree levels of the full grid) so
that the frozen structures built on the cut-out have the same prediction kinds as on the full grid and every fine
level is the full grid's level inside the window. `tree` loses its lateral-boundary step (`L0hg`: complete groups
outside the regular native order, absent inside the domain) and about `log4(1/fraction)` coarse levels. `lab minigrid
check OUT_DIR [--date I] [--full STORE]` validates the structures, round-trips a date within the bounds and prints
bytes per cell, next to the full grid. On ICON-CH1 a 1/16 window (73728 cells) encodes in 2-4 s with 16 threads on an
idle machine and decodes in milliseconds; its bytes per cell are not the full
grid's (fewer coarse cells, the window's weather), relative comparisons between variants transfer. Other tools
decode on it with `ANEMOI_DATASETS_CODECS_PATH=OUT_DIR/mini.zarr` or `minigrid.use(OUT_DIR)`.

`lab ledger` keeps experiments comparable: `run CONFIG STORE --out LEDGER.jsonl [--dates 6|0,6,13|2:5|all] [--tag T]
[--set options.k=3 --set structure=tree_b ...] [--dated] [--keep-chunks DIR]` encodes and decodes the dates with the
configuration plus dotted-key overrides and appends one self-describing JSON line per (run, date, field): tag, run id,
time, environment (package, numpy, kernel variant, threads, host, git commit), configuration sha, overrides, mode,
fallback, bytes, max error and RMSE over the bound, encode and decode seconds, chunk sha256. `sweep CONFIG STORE
SPEC.json --out LEDGER.jsonl` expands a grid or a list of settings (skipping those the ledger holds for the same
configuration, store, dates and package version), `compare LEDGER TAG_A TAG_B` prints the per-field byte deltas,
totals, top movers, mode changes and error statistics, `summary` lists the runs and `pareto LEDGER OUT.svg` plots
size against error or time.

## Keeping the decoder honest: `fuzz`

`scripts/lab/build_asan.sh OUT.so` builds the C kernels with AddressSanitizer and UndefinedBehaviorSanitizer for
`ANEMOI_DATASETS_CODECS_LIBRARY=OUT.so` (the script prints the `LD_PRELOAD` and `ASAN_OPTIONS` to use).
`lab fuzz (--synthetic [--chain] [--reuse] | CHUNK --config CONFIG) --iterations N --seed S --out DIR [--jobs J]
[--timeout T]` mutates a valid chunk (bit flips, byte overwrites, insertions, truncations, targeted edits of the
container, LATTICE and stage headers) and decodes every mutant as a reader would, in `--jobs` worker subprocesses with
a timeout per decode, classifying the outcome: a clean error, silent acceptance (recorded with whether the output
differs), a crash or a hang; reproducers of every non-clean case are saved under DIR (`--reuse` keeps the synthetic
base chunk of an earlier run in DIR). `lab fuzz --replay DIR/OUTCOME/case-N.bin ... --out DIR` decodes saved
reproducers again in fresh workers. `lab fuzz --parsers --iterations N --seed S --out DIR` is a separate mode: it
fuzzes the header parsers (envelope, stage headers, container) in-process, N mutants per parser.

## Findings on ICON-CH1

What the tools showed on the ICON-CH1 sample, with the configuration of `tests/data/icon_ch1_sample.json` (`tree`, k =
1, `recon_bits` 3, `rounding: grid`, physics on with `recipe_fit: false`), reproduced with the commands above; the
numbers are the tools' output, rounded. Most are from sample date 6 (43.4 MiB through zarr), a typical date: the
sample's dates range from 31.5 to 44.8 MiB. The configuration changes were confirmed on four dates (0, 6, 13, 19), whose
figures are in [benchmarks.md](benchmarks.md#configuration-choices).

* **Chunk parts** (`account`): rANS streams 82%, the three forcing fields stored losslessly 15% (through zarr the
  codec never sees the dates: 16 bits per cell for `cos_local_time`, `sin_local_time`, `insolation`; 11-18% of a chunk
  over the four dates), escapes 2.1%, tables 0.4%. The coder is 0.9% above the conditional entropy of its own contexts
  (12-bit tables and stream flushes); the contexts gain 8.6% over order 0, and finer spread contexts do not pay net
  of their tables (3.2% gross, -4.2% net per step). `FI_925` spends 40% of its bytes on escapes and `FI_850` 24%
  (bimodal residuals, every cross coefficient zero: below-ground cells with `below_ground` off); `OMEGA_850`'s 16%
  are a heavy tail. Residuals stay correlated between lattice neighbours (0.15-0.28 for `q`, 0.33-0.55 for `|q|`).
* **Configuration changes** (`ledger`, bytes of the chunk, four dates): forcings with a bound of 1e-6 (the recipes'
  own tolerance) -11.5 to -18.2%, at about 0.6 bits per cell instead of 16 and in the single codec and the chain
  alike (less at night, when the insolation is constant); `physics.recipe_fit` -11.7 to -18.4%; `hc2_k3` -10.1 to
  -12.0%; `tree_b` -6.9 to -8.3% (the gain in `OMEGA_*`; `FI_500` larger); `below_ground: true` -3.1 to -5.4%
  (on date 6 `FI_925` about -334 KiB, then `OMEGA_1000`, `T_1000`, `FI_850`, `FI_1000`; the 1000 hPa winds switch to
  the 10 m winds as source); `rounding: direct` -2.3 to -2.8%; `recon_bits` 5 and 8 -1.7% and -2.2% on average at a
  mean RMSE / bound of 0.51 instead of 0.48 (the residual-domain rule shrinks the step by `1 + 2^-m`, 11% at m = 3);
  `cross: on` without a depth limit -1.2 to -2.1% (`FI_500` gets a source, `FI_600` loses); `k = 3` +39% at a third
  of the error (date 6). The defaults `physics.recipe_fit: true` and `options.rounding: direct` follow from these
  figures; every version 3 decoder reads what they write. A geopotential plan tuned by hand reached -1.8%,
  0.2% beyond `cross: on`.
* **Oracles**: the baseline estimate is within 3% of the bytes of a codec coded the same way (plain rule, no cross
  terms); the configured codec is 10% smaller than that, so the oracles bound ideas, they do not reproduce the
  configuration. Unconstrained linear prediction gains 0.4%; the finest step carries 63% of the bits at 2.8 bits per
  target and the coarse levels 9% together (little upside for a lifting scheme); the contexts have about 2% headroom
  before model cost, and `xz` / `zstd` on the symbols are worse than the static tables. The top cross pairs not in
  the plan are `PS <- FI_1000` (3.94 bits per cell, 45% of `PS`) and `FI_1000 <- PS` (3.52): the shared orography
  signal, which the plan takes from the static `FIS` (the cross oracle has no static sources, so this is an upper
  bound the plan already captures); then `FI_1000 <- FI_925` (1.76) and the adjacent-level `FI` pairs (1.1-1.4,
  against 0.6-1.0 from the plan's `FI_500` star), `FI_925 <- FI_850` (1.04) and `SKT <- T_2M` (0.33), dropped by the
  depth limit, `OMEGA_850 <- OMEGA_925` (0.57), `QV_1000 <- TD_2M` (0.52), `T_2M <- PS` (0.73 against 0.56 from
  `PMSL`). Pair gains are measured one at a time on the no-cross baseline and are not additive; a term is judged by a
  ledger run with it in the plan. Another field's residual bucket as a context gains 0.1-0.2 bits, but an arbitrary
  field already gives 0.02-0.07, so most of that is a weather-activity signal, not the pair. The temporal oracle's
  4.4% on the sample is orography seen through a date years apart (96% of it in `FI_1000`, `PS`, `FI_925`, `T_2M`):
  the sample's dates are not consecutive hours, so it says nothing about time.
* **References** (`refs`, 97 lossy variables, same bound): zfp 3.1x (image) to 4.1x (native order) the lattice
  codec's size, Blosc lz4 on float32 7.2x, zstd / xz on the codec's own integers 1.4-1.7x, SZ3 1.42x in native order
  and 1.21x on the lattice image, where it is smaller than this configuration on 31 variables (`T_500` 0.97, `U_500`
  0.94, `V_500` 0.89, most `OMEGA_*`). That image charges SZ3 for 31% filled holes; on a hole-free window SZ3 is
  0.84-0.92 of `tree` at `recon_bits` 3 on `U_500`, `V_500`, `T_500`, and 0.99-1.11 of `tree_b` at `recon_bits` 8
  (0.95 on `OMEGA_850`). The gap is the structure and the step penalty, not the predictor. SZ3's RMSE at an absolute
  bound is about 0.58 of the bound against the codec's 0.48-0.52. `xz` is marginally smaller on the sparse
  `TOT_PREC_1H`.
* **Fuzzing**: 140k mutants (synthetic plain and chain, a real chunk) under AddressSanitizer and
  UndefinedBehaviorSanitizer: no crash, hang or memory error. The decoder guards against the two things it did find
  (signed integer overflow on corrupt residual streams, an output allocation sized from an unchecked header). 9-21% of
  corrupted chunks decode to different values without an error (the formats have no payload checksum,
  [goals.md](goals.md#known-limitations)); a rANS end-state check (every lane back at its initial state, every slice
  consumed exactly) would catch most of them without a format change.
* **The mini grid** gives per-field ratios, not chunk totals: its window is the Alps, where fixed per-field costs
  weigh 3-8%, and it misleads for the lowest geopotential levels (below-ground values): there `FI_850` / `FI_1000` as
  sources made `FI_925` 29% larger, on the full grid they change nothing.

These findings do not settle the decode speed of `tree_b` and `hc2_k3` (indicative figures are in
[benchmarks.md](benchmarks.md#decode-speed); the structure choice is a throughput decision), whether the bound may
trade RMSE for size (`recon_bits`, `rounding`), or the bytes of `below_ground` across numpy versions (its fit is
float64 least squares). The ledger records encode and decode times, but they are only comparable within one run on
an otherwise idle machine.
