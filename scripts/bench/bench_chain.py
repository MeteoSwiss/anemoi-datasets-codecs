"""Size, write time and zarr decode time of one ICON-CH1 date: the single lattice codec, the codec chain of ``physics
expand`` (``[field_terms, static_ref, icon_lattice_bytes]``) and Blosc lz4, all through zarr (no valid dates, as in any
zarr write: the single codec stores the forcings as recipes, ``physics.recipe_fit`` being the default, the chain
losslessly).

    python scripts/bench/bench_chain.py write DIR DATE STRUCTURE K OUT.jsonl [--forcing-bound B]
    python scripts/bench/bench_chain.py read DIR OUT.jsonl [--repeats 7]

``write`` writes sample date DATE into ``DIR/single-<STRUCTURE>-k<K>.zarr``, ``DIR/chain-<STRUCTURE>-k<K>.zarr`` and
(once per date: kept when it holds DATE, rewritten otherwise) ``DIR/blosc.zarr``, each store with the attribute
``date_index`` and registered so that its sidecar holds what its chunks reference; it times the zarr write
(``a[:] = x``, encode included), reads the date back and checks every variable against its bound (the single codec's
recipes against ``recipe_bound``, the others for equality), and appends one JSON line per store: bytes of the chunk,
sidecar bytes, write seconds, worst error / bound. ``--forcing-bound B`` also writes
``DIR/chain-fb-<STRUCTURE>-k<K>.zarr``, the chain of ``physics expand --forcing-bound B`` (the forcings lossy within
B, checked against B).
``read`` times ``a[...]`` on every store in DIR at the thread count of ``ANEMOI_DATASETS_CODECS_THREADS`` (Blosc with as
many threads), after two warm-up reads (sidecar files, static fields, pages), and appends best / median ms and the
sha256 of the decoded bytes (equal for every thread count).
Inputs as in ``realdata_store.py`` (``$ANEMOI_DATASETS_CODECS_SAMPLE``, ``tests/data/icon_ch1_sample.json``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realdata_store import SAMPLE, sample_config  # noqa: E402


def _row(out: Path, row: dict) -> None:
    print(json.dumps(row), flush=True)
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _size(path: Path, sub: str) -> int:
    d = path / sub
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.is_dir() else 0


def write(a: argparse.Namespace) -> None:
    import zarr

    from anemoi_datasets_codecs import chain, physics
    from anemoi_datasets_codecs.cli import read_variables
    from anemoi_datasets_codecs.icon_lattice import LatticeCodec, register_store, statics, unregister_store

    src = zarr.open_group(str(SAMPLE), mode="r")
    x = np.ascontiguousarray(src["data"][a.date : a.date + 1])  # (1, variables, 1, cells)
    names, ranges = read_variables(SAMPLE)
    cfg = {**sample_config(), "variables": names, "structure": a.structure, "options": {"k": a.k}}
    single = LatticeCodec(**cfg)
    bounds = {v.name: v.max_error for v in single.variables if v.lossy}
    recipe_tol = dict.fromkeys(single.plan.recipes, float(single.physics["recipe_bound"]))
    stores = {
        f"single-{a.structure}-k{a.k:g}": [{"name": "icon_lattice_bytes", "configuration": single.get_config()}],
        f"chain-{a.structure}-k{a.k:g}": physics.expand(names, None, cfg, ranges=ranges),
        "blosc": None,
    }
    if a.forcing_bound is not None:
        stores[f"chain-fb-{a.structure}-k{a.k:g}"] = physics.expand(
            names, None, cfg, ranges=ranges, forcing_bound=a.forcing_bound
        )
    a.dir.mkdir(parents=True, exist_ok=True)
    for label, entries in stores.items():
        path = a.dir / f"{label}.zarr"
        if entries is None and path.exists():  # Blosc does not depend on STRUCTURE and K: once per date
            try:
                same = zarr.open_group(str(path), mode="r").attrs.get("date_index") == a.date
            except Exception:  # not a readable store: rewrite it
                same = False
            if same:
                continue
        if entries is None:
            compressors = [zarr.codecs.BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")]
        else:
            compressors = [chain.codec_class(e["name"])(**e["configuration"]) for e in entries]
        g = zarr.open_group(str(path), mode="w")
        register_store(path)
        arr = g.create_array("data", shape=x.shape, chunks=x.shape, dtype="float32", compressors=compressors)
        t0 = time.perf_counter()
        arr[:] = x
        seconds = time.perf_counter() - t0
        g.attrs["variables"] = names
        g.attrs["date_index"] = a.date
        unregister_store(path)
        statics.clear_caches()
        os.environ["ANEMOI_DATASETS_CODECS_PATH"] = str(path)
        back = zarr.open_group(str(path), mode="r")["data"][...]
        worst = 0.0
        for i, v in enumerate(names):
            err = float(np.abs(back[0, i].astype(np.float64) - x[0, i]).max())
            if v in bounds:
                worst = max(worst, err / bounds[v])
                if err > bounds[v]:
                    raise SystemExit(f"{label}: {v} error {err} > bound {bounds[v]}")
            elif v in recipe_tol and label.startswith("single"):  # a recipe, or exact when stored losslessly
                if err > recipe_tol[v]:
                    raise SystemExit(f"{label}: {v} recipe error {err} > recipe_bound {recipe_tol[v]}")
            elif v in recipe_tol and label.startswith("chain-fb"):  # a forcing under the forcing bound
                if err > a.forcing_bound:
                    raise SystemExit(f"{label}: {v} error {err} > forcing bound {a.forcing_bound}")
            elif not np.array_equal(back[0, i].view(np.uint32), x[0, i].view(np.uint32)):
                raise SystemExit(f"{label}: {v} is not exact")
        _row(
            a.out,
            {
                "store": label,
                "date_index": a.date,
                "codecs": [type(c).__name__ for c in compressors],
                "bytes": _size(path, "data/c"),
                "MiB": round(_size(path, "data/c") / 2**20, 3),
                "sidecar_bytes": _size(path, ".codecs"),
                "write_s": round(seconds, 1),
                "worst_err_over_bound": round(worst, 4),
            },
        )


def read(a: argparse.Namespace) -> None:
    import numcodecs.blosc
    import zarr

    from anemoi_datasets_codecs.icon_lattice import native, statics
    from anemoi_datasets_codecs.icon_lattice.threads import default_threads

    threads = default_threads()
    numcodecs.blosc.use_threads = threads > 1
    numcodecs.blosc.set_nthreads(threads)
    for path in sorted(a.dir.glob("*.zarr")):
        os.environ["ANEMOI_DATASETS_CODECS_PATH"] = str(path)
        statics.clear_caches()
        arr = zarr.open_array(str(path / "data"), mode="r")
        for _ in range(2):
            y = arr[...]
        ts = []
        for _ in range(a.repeats):
            t0 = time.perf_counter()
            y = arr[...]
            ts.append(time.perf_counter() - t0)
        _row(
            a.out,
            {
                "store": path.stem,
                "threads": threads,
                "isa": native.isa(),
                "best_ms": round(min(ts) * 1e3, 1),
                "median_ms": round(statistics.median(ts) * 1e3, 1),
                "repeats": a.repeats,
                "sha256": hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),
            },
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("dir", type=Path)
    w.add_argument("date", type=int)
    w.add_argument("structure")
    w.add_argument("k", type=float)
    w.add_argument("out", type=Path)
    w.add_argument("--forcing-bound", type=float, help="also write the chain of physics expand --forcing-bound")
    w.set_defaults(fn=write)
    r = sub.add_parser("read")
    r.add_argument("dir", type=Path)
    r.add_argument("out", type=Path)
    r.add_argument("--repeats", type=int, default=7)
    r.set_defaults(fn=read)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
