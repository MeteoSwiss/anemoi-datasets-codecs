"""Chunk decode time of this package on a Zarr format 3 store written by the codec.

    python scripts/bench/bench_decode.py STORE OUT.jsonl [--dates 1] [--threads 1,16] [--repeats 9] [--isa auto]

Reads the first ``--dates`` chunk files of ``STORE/data`` (array ``--array``) straight from the store directory, builds
the codec from the array's metadata and decodes each chunk into a preallocated buffer: two warm-up decodes, then
``--repeats`` timed ones per thread count (``ANEMOI_DATASETS_CODECS_THREADS``). ``--isa`` sets
``ANEMOI_DATASETS_CODECS_ISA`` before the kernels load. One JSON line per (chunk, thread count) is printed and appended
to OUT.jsonl: kernel variant, threads, best / median ms, chunk size and the sha256 of the decoded float32 bytes (equal
for every variant and thread count; ``summary.py`` checks it). The store's sidecar is registered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("store", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--array", default="data")
    ap.add_argument("--dates", type=int, default=1, help="number of chunks (the first ones in chunk order)")
    ap.add_argument("--threads", default="1,16", help="comma-separated thread counts")
    ap.add_argument("--repeats", type=int, default=9)
    ap.add_argument("--isa", choices=("auto", "avx2", "scalar"), default=None)
    args = ap.parse_args()
    if args.isa:
        os.environ["ANEMOI_DATASETS_CODECS_ISA"] = args.isa

    import numpy as np

    from anemoi_datasets_codecs.icon_lattice import LatticeCodec, native, register_store, store

    register_store(args.store)
    meta = store.array_meta(args.store, args.array)
    codec = LatticeCodec.from_config({k: v for k, v in meta["codec"].items() if k != "id"})
    paths = list(store.chunk_paths(args.store, args.array))[: args.dates]
    if not paths:
        raise SystemExit(f"no chunk files in {args.store / args.array}")
    buf = np.empty(int(np.prod(meta["chunks"])), np.float32)
    host, isa = os.uname().nodename, native.isa()
    for path in paths:
        enc = path.read_bytes()
        for t in [int(v) for v in args.threads.split(",")]:
            os.environ["ANEMOI_DATASETS_CODECS_THREADS"] = str(t)
            for _ in range(2):
                codec.decode(enc, out=buf)
            times = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                codec.decode(enc, out=buf)
                times.append(time.perf_counter() - t0)
            row = {
                "isa": isa,
                "threads": t,
                "best_ms": round(1e3 * min(times), 2),
                "median_ms": round(1e3 * statistics.median(times), 2),
                "n": len(times),
                "chunk": str(path.relative_to(args.store)),
                "chunk_MiB": round(len(enc) / 2**20, 3),
                "decoded_sha256": hashlib.sha256(buf.tobytes()).hexdigest(),
                "store": str(args.store),
                "host": host,
            }
            print(json.dumps(row), flush=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
