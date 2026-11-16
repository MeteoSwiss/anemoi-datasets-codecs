"""Encoder-side real-data baseline: sha256 and time of the chunk ``LatticeCodec.encode`` makes for sample dates.

    python scripts/bench/bench_encode.py write OUT.json [--dates 0,6,13,19] [--threads N] [--no-valid-dates]
    python scripts/bench/bench_encode.py check BASELINE.json [--dates ...] [--threads N] [--no-valid-dates]

Each sample date (shape ``(1, variables, 1, cells)``) is encoded with ``LatticeCodec(**sample_config())`` twice: without
dates (as any zarr write: the forcings are stored lossless) and with the date the sample gives it (``src["dates"][i]``);
``--no-valid-dates`` skips the dated encodes. The dated encode tries the forcing recipes, but the dates of the
ICON-CH1 sample are relabelled (not the valid dates of its fields), so there the recipes fall back to lossless storage
and the dated chunks differ from the undated ones only by the stored date: on the sample this script does not guard
the recipe encoder (``plan_baseline.py`` does, on synthetic data); on a store with true valid dates it does.
``write`` records per date and mode the seconds, the bytes and the sha256 of the chunk, and once the package and
numpy versions, the kernel variant, the thread count, the sha256 of the sample configuration (JSON, pretty, stable key
order) and the codec's complete resolved configuration (``codec``: every option and physics key, defaults included).
``check`` re-encodes (by default the dates and modes of the baseline) with the baseline's encoder settings, not the
running version's defaults: the ``codec`` configuration of the baseline, or, for a baseline written before it was
recorded (before release 0.1.0 changed the defaults), the sample configuration with the defaults of that time pinned
(``LEGACY``: ``rounding: grid``, ``recipe_fit: false``). It compares every sha256, prints a table (date, mode,
baseline s, now s, bytes, identical / DIFFERENT) and the total encode time of the run, and exits non-zero on any
difference or missing entry: run it after any change of the encoder to prove the chunks did not change, and as the
encode benchmark. An encoder change that is meant to change the bytes needs a new baseline (``write`` to a new file),
never a rewritten one.
``--threads`` sets ``ANEMOI_DATASETS_CODECS_THREADS`` for the process (default: the codec's). Timings on a shared
machine are noisy; the sha256s are not (they do not depend on the thread count).
Inputs as in ``realdata_store.py`` (``$ANEMOI_DATASETS_CODECS_SAMPLE``, ``tests/data/icon_ch1_sample.json``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realdata_store import CONFIG, SAMPLE, sample_config  # noqa: E402

MODES = ("nodates", "dated")
# the encoder settings of the defaults before release 0.1.0, for baselines written without a ``codec`` entry
LEGACY = {"options": {"rounding": "grid"}, "physics": {"recipe_fit": False}}


def _config_sha256(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def _baseline_config(meta: dict) -> dict:
    """The configuration a baseline's chunks were encoded with (module docstring)."""
    if "codec" in meta:
        return dict(meta["codec"])
    cfg = sample_config()
    if meta.get("config_sha256") != _config_sha256(cfg):
        print(f"warning: {CONFIG.name} differs from the configuration the baseline was written with", flush=True)
    return {
        **cfg,
        "options": {**cfg.get("options", {}), **LEGACY["options"]},
        "physics": {**cfg.get("physics", {}), **LEGACY["physics"]},
    }


def _encode_all(dates: list[int], modes: tuple[str, ...], cfg: dict | None = None) -> tuple[dict, dict]:
    import zarr

    from anemoi_datasets_codecs import __version__
    from anemoi_datasets_codecs.icon_lattice import LatticeCodec, native

    sample = sample_config()
    src = zarr.open_group(str(SAMPLE), mode="r")
    codec = LatticeCodec(**(sample if cfg is None else cfg))
    _ = codec.structure  # built before the timings
    meta = {
        "package_version": __version__,
        "numpy_version": np.__version__,
        "kernel": native.isa(),
        "threads": os.environ.get("ANEMOI_DATASETS_CODECS_THREADS", "default"),
        "config": str(CONFIG.name),
        "config_sha256": _config_sha256(sample),
        "codec": codec.get_config(),
    }
    chunks: dict[str, dict] = {}
    for d in dates:
        x = np.ascontiguousarray(src["data"][d : d + 1])
        valid = np.asarray(src["dates"][d])
        for mode in modes:
            t0 = time.perf_counter()
            out = codec.encode(x, dates=valid if mode == "dated" else None)
            dt = time.perf_counter() - t0
            chunks[f"{d}/{mode}"] = {
                "seconds": round(dt, 3),
                "bytes": len(out),
                "sha256": hashlib.sha256(out).hexdigest(),
            }
            print(f"date {d} {mode}: {dt:.1f} s, {len(out) / 2**20:.2f} MiB", flush=True)
    return meta, chunks


def _dates(s: str | None, default: list[int]) -> list[int]:
    return [int(v) for v in s.split(",")] if s else default


def write(a: argparse.Namespace) -> int:
    if Path(a.file).exists():
        raise SystemExit(f"{a.file} exists: a baseline is never rewritten (write a new file)")
    modes = MODES[:1] if a.no_valid_dates else MODES
    meta, chunks = _encode_all(_dates(a.dates, [0, 6, 13, 19]), modes)
    Path(a.file).write_text(json.dumps({"meta": meta, "chunks": chunks}, indent=2, sort_keys=True) + "\n")
    print(f"wrote {a.file}: {len(chunks)} chunks, total {sum(c['seconds'] for c in chunks.values()):.1f} s")
    return 0


def check(a: argparse.Namespace) -> int:
    base = json.loads(Path(a.file).read_text())
    keys = list(base["chunks"])
    dates = _dates(a.dates, sorted({int(k.split("/")[0]) for k in keys}))
    modes = MODES[:1] if a.no_valid_dates else MODES
    meta, now = _encode_all(dates, modes, _baseline_config(base["meta"]))
    was = {k: v for k, v in base["meta"].items() if k != "codec"}
    show = {k: v for k, v in meta.items() if k != "codec"}
    print(f"\nbaseline: {was}\nnow:      {show}\n")
    print(f"{'date':>4} {'mode':<8} {'baseline s':>10} {'now s':>8} {'bytes':>10}  result")
    bad = 0
    for k, c in now.items():
        b = base["chunks"].get(k)
        same = b is not None and b["sha256"] == c["sha256"] and b["bytes"] == c["bytes"]
        bad += not same
        d, mode = k.split("/")
        bs = f"{b['seconds']:.1f}" if b else "-"
        verdict = "identical" if same else "DIFFERENT"
        print(f"{d:>4} {mode:<8} {bs:>10} {c['seconds']:>8.1f} {c['bytes']:>10}  {verdict}")
    print(f"total encode time {sum(c['seconds'] for c in now.values()):.1f} s")
    print(f"{len(now) - bad} of {len(now)} chunks identical" + ("" if not bad else f", {bad} DIFFERENT or missing"))
    return 1 if bad else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=["write", "check"])
    p.add_argument("file", help="baseline JSON to write / check against")
    p.add_argument("--dates", help="sample date indices, e.g. 0,6 (default: write 0,6,13,19; check the baseline's)")
    p.add_argument("--threads", type=int, help="ANEMOI_DATASETS_CODECS_THREADS for the process")
    p.add_argument("--no-valid-dates", action="store_true", help="skip the encodes with the valid date")
    a = p.parse_args()
    if a.threads:
        os.environ["ANEMOI_DATASETS_CODECS_THREADS"] = str(a.threads)
    sys.exit({"write": write, "check": check}[a.command](a))


if __name__ == "__main__":
    main()
