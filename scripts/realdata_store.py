"""Real-data store check (ICON-CH1): write sample dates with the codec into a Zarr format 3 store and read them back.

    python scripts/realdata_store.py write OUT.zarr N_DATES    # recompress the first N sample dates
    python scripts/realdata_store.py read OUT.zarr N_DATES     # read back, compare with the sample, print sizes

Inputs: the ICON-CH1 sample store (``$ANEMOI_DATASETS_CODECS_SAMPLE``, default ``.local/data/icon-ch1-sample.zarr``,
Zarr format 3, 20 dates) and the codec configuration ``tests/data/icon_ch1_sample.json`` (bounds, clip, static
fields), lossless fields Blosc lz4. ``write`` lays the store out as anemoi does: ``data`` (one date per chunk, codec
``icon_lattice`` as the serializer), the coordinates, statistics and other arrays of the sample (``dates`` cut to N)
and its group attributes (``end_date`` adjusted); the store is registered after it is created, so that the lattice
and the captured static fields land in its sidecar. ``read`` (run it in a fresh process with
``ANEMOI_DATASETS_CODECS_CACHE=off`` and ``ANEMOI_DATASETS_CODECS_PATH=OUT``) opens the store with anemoi-datasets
when importable, else with zarr, checks every lossy variable against its bound and every other one for equality
(the forcing recipes within the physics ``recipe_bound``: with ``recipe_fit``, the default, they are recipes even
though zarr passes no dates), and prints the size per date, the sidecar size and the codec configuration size. Exits
non-zero on any violation.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SAMPLE = Path(os.environ.get("ANEMOI_DATASETS_CODECS_SAMPLE", REPO / ".local/data/icon-ch1-sample.zarr"))
CONFIG = REPO / "tests/data/icon_ch1_sample.json"
LZ4 = {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1}


def sample_config() -> dict:
    """Codec configuration of the sample (``tests/data/icon_ch1_sample.json`` with lossless Blosc lz4)."""
    return {**json.loads(CONFIG.read_text()), "lossless": dict(LZ4)}


def write(out: Path, n: int) -> None:
    import zarr

    from anemoi_datasets_codecs.icon_lattice import LatticeCodec, register_store
    from anemoi_datasets_codecs.icon_lattice.zarr3 import IconLatticeCodec

    src = zarr.open_group(str(SAMPLE), mode="r")
    cfg = sample_config()
    codec = LatticeCodec(**cfg)
    if list(codec.names) != list(src.attrs["variables"]):
        raise SystemExit("variables of the configuration differ from the sample's")
    g = zarr.open_group(str(out), mode="w", zarr_format=3)
    register_store(out)  # after the group exists: a store root is recognised by its zarr.json
    shape = (n, *src["data"].shape[1:])
    a = g.create_array(
        "data",
        shape=shape,
        chunks=(1, *shape[1:]),
        dtype="float32",
        serializer=IconLatticeCodec(**cfg),
        compressors=None,
    )
    a.attrs.update(dict(src["data"].attrs))
    for name in sorted(src.array_keys()):
        if name == "data":
            continue
        x = src[name][:n] if name == "dates" else src[name][...]
        b = g.create_array(name, shape=x.shape, dtype=x.dtype, chunks=x.shape)
        b[...] = x
        b.attrs.update(dict(src[name].attrs))
    attrs = dict(src.attrs)
    attrs["end_date"] = str(src["dates"][n - 1].astype("datetime64[s]"))
    g.attrs.update(attrs)
    for i in range(n):
        t = time.perf_counter()
        a[i] = src["data"][i]
        print(f"date {i}: encoded in {time.perf_counter() - t:.1f} s", flush=True)


def _tree_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def read(out: Path, n: int) -> None:
    import zarr

    from anemoi_datasets_codecs.icon_lattice import LatticeCodec, sidecar, store

    print(f"cache {os.environ.get(sidecar.CACHE_ENV)}, path {os.environ.get(sidecar.PATH_ENV)}")
    src = zarr.open_group(str(SAMPLE), mode="r")
    names = list(src.attrs["variables"])
    try:
        from anemoi.datasets import open_dataset
    except ImportError:
        z = zarr.open_group(str(out), mode="r")["data"]
        get, how = (lambda i: np.asarray(z[i])), "zarr"
    else:
        ds = open_dataset(str(out))
        get, how = (lambda i: np.asarray(ds[i])), "anemoi.datasets.open_dataset"
        if list(ds.variables) != names:
            raise SystemExit("anemoi variables differ from the sample's")
    meta = store.array_meta(out)
    cfg = {k: v for k, v in meta["codec"].items() if k != "id"}
    codec = LatticeCodec.from_config(cfg)  # the store's configuration (the stamp is ignored)
    worst, bad = 0.0, []
    recipes, rb = set(codec.plan.recipes), float(codec.physics["recipe_bound"])
    for i in range(n):
        t = time.perf_counter()
        y = get(i)[:, 0]
        dt = time.perf_counter() - t
        x = src["data"][i, :, 0]
        for k, v in enumerate(codec.variables):
            if v.lossy:
                err = float(np.abs(y[k].astype(np.float64) - x[k]).max())
                worst = max(worst, err / v.max_error)
                if err > v.max_error:
                    bad.append(f"date {i} {v.name}: error {err:.6g} > bound {v.max_error:.6g}")
            elif v.name in recipes:  # a recipe, or exact when stored losslessly
                err = float(np.abs(y[k].astype(np.float64) - x[k]).max())
                if err > rb:
                    bad.append(f"date {i} {v.name}: recipe error {err:.6g} > recipe_bound {rb:.6g}")
            elif not np.array_equal(y[k], x[k]):
                bad.append(f"date {i} {v.name}: not exact")
        print(f"date {i}: read with {how} in {dt:.2f} s", flush=True)
    for b in bad:
        print("VIOLATION", b)
    size = _tree_size(out / "data") - (out / "data" / "zarr.json").stat().st_size
    side = _tree_size(out / sidecar.SIDECAR)
    print(f"data {size / n / 2**20:.2f} MiB per date, sidecar {side / 2**20:.1f} MiB, worst error / bound {worst:.4f}")
    print(f"codec configuration {len(json.dumps(cfg))} B (in data/zarr.json)")
    if bad:
        raise SystemExit(1)
    print(f"{n} dates: every lossy variable within its bound, every recipe within recipe_bound, every other one exact")


if __name__ == "__main__":
    cmd, out, n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
    {"write": write, "read": read}[cmd](out, n)
