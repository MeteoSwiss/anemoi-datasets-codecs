"""Copy a plain anemoi store (Blosc arrays, group attributes; Zarr format 2 or 3) into a Zarr format 3 store with
zarr-python 3, and check the copy: a Zarr 3 copy of an existing anemoi dataset (or of some of its dates) to try the
codecs on.

    python scripts/sample_store.py convert SRC DST [--batch 4]     # DST must not exist
    python scripts/sample_store.py check SRC DST [--dates 0,9,19]  # attributes, metadata, small arrays, some dates

``data`` is chunked ``(1, n_vars, 1, n_cells)`` and copied ``--batch`` dates at a time (bounded memory); every
other array is one chunk. Every array gets Blosc lz4, level 5, byte shuffle (``zarr.codecs.BloscCodec``) and keeps
its dtype, fill value and attributes; the group attributes are copied as they are. ``check`` compares group and
array attributes, shapes and dtypes, the small arrays in full and the given dates of ``data`` bit for bit; it exits
non-zero on any difference. Needs zarr-python 3 only (not this package).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def _blosc():
    from zarr.codecs import BloscCodec

    return BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")


def _fill(a) -> object:
    f = a.fill_value
    return None if f is None else np.asarray(f, dtype=a.dtype)[()]


def convert(src: Path, dst: Path, batch: int) -> None:
    import zarr

    if dst.exists():
        raise SystemExit(f"{dst} exists")
    s = zarr.open_group(str(src), mode="r")
    g = zarr.open_group(str(dst), mode="w", zarr_format=3)
    t0 = time.perf_counter()
    for name in sorted(s.array_keys()):
        a = s[name]
        if name == "data":
            chunks = (1, *a.shape[1:])
        else:
            chunks = a.shape
        b = g.create_array(name, shape=a.shape, chunks=chunks, dtype=a.dtype, fill_value=_fill(a), compressors=_blosc())
        b.attrs.update(dict(a.attrs))
        if name == "data":
            for i in range(0, a.shape[0], batch):
                t = time.perf_counter()
                j = min(i + batch, a.shape[0])
                b[i:j] = a[i:j]
                print(f"data[{i}:{j}] copied in {time.perf_counter() - t:.1f} s", flush=True)
        else:
            b[...] = a[...]
            print(f"{name} {a.shape} {a.dtype} copied", flush=True)
    g.attrs.update(dict(s.attrs))
    print(f"{src} -> {dst} in {time.perf_counter() - t0:.0f} s")


def _canon(attrs) -> str:
    return json.dumps(dict(attrs), sort_keys=True, default=str)


def _sha(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def check(src: Path, dst: Path, dates: list[int]) -> None:
    import zarr

    s, d = zarr.open_group(str(src), mode="r"), zarr.open_group(str(dst), mode="r")
    bad: list[str] = []
    if d.metadata.zarr_format != 3:
        bad.append(f"{dst} is Zarr format {d.metadata.zarr_format}")
    if _canon(s.attrs) != _canon(d.attrs):
        bad.append("group attributes differ")
    if sorted(s.array_keys()) != sorted(d.array_keys()):
        bad.append(f"arrays differ: {sorted(s.array_keys())} vs {sorted(d.array_keys())}")
    for name in sorted(set(s.array_keys()) & set(d.array_keys())):
        a, b = s[name], d[name]
        if (a.shape, a.dtype) != (b.shape, b.dtype):
            bad.append(f"{name}: {a.shape} {a.dtype} vs {b.shape} {b.dtype}")
            continue
        if _canon(a.attrs) != _canon(b.attrs):
            bad.append(f"{name}: attributes differ")
        if name == "data":
            for i in dates:
                same = _sha(a[i]) == _sha(b[i])
                print(f"data[{i}]: {'bit-identical' if same else 'DIFFERENT'}", flush=True)
                if not same:
                    bad.append(f"data[{i}] differs")
        elif _sha(a[...]) != _sha(b[...]):
            bad.append(f"{name} differs")
    print(f"group attributes: {len(s.attrs)} keys; arrays: {sorted(d.array_keys())}")
    for b in bad:
        print("MISMATCH", b)
    if bad:
        raise SystemExit(1)
    print("check OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("src", type=Path)
    c.add_argument("dst", type=Path)
    c.add_argument("--batch", type=int, default=4, help="dates of data copied per write")
    k = sub.add_parser("check")
    k.add_argument("src", type=Path)
    k.add_argument("dst", type=Path)
    k.add_argument("--dates", default="0,9,19", help="comma-separated date indices of data to compare")
    args = ap.parse_args()
    if args.cmd == "convert":
        convert(args.src, args.dst, args.batch)
    else:
        check(args.src, args.dst, [int(v) for v in args.dates.split(",")])


if __name__ == "__main__":
    sys.exit(main())
