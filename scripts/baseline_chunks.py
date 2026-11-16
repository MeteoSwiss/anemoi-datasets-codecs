"""Byte and decode baseline of the chunks of a Zarr format 3 store written by the codec: the guard that existing
chunks keep decoding bit-identically (docs/codecs/icon_lattice.md, compatibility promise 1).

    python scripts/baseline_chunks.py write STORE OUT_DIR [--array data]   # writes OUT_DIR/<store name>.chunks.json
    python scripts/baseline_chunks.py check STORE BASELINE.json

``write`` records, for every chunk file of the array, its size, the sha256 of the file and the sha256 of the decoded
float32 bytes (native cell order, as ``LatticeCodec.decode`` returns them), with the codec configuration and the
package version / kernel variant that produced them. ``check`` re-decodes every chunk of the baseline with the current
package and compares: a changed chunk file, a missing chunk or a different decode is a failure (exit 1); chunks not in
the baseline are listed. A note that the store's codec configuration differs from the baseline's is not a failure
(the metadata of a store can be rewritten, for instance to current option names or with keys a later version writes
out, without touching a chunk; the writer stamp ``written_by`` is not compared): the chunk and decode comparison
decides. Decoding does not depend on the encoder options, so a baseline written before the defaults changed
(``rounding``, ``recipe_fit``) still checks the same chunks. The store's sidecar is registered; set
``ANEMOI_DATASETS_CODECS_ISA=scalar`` to check the scalar kernels and ``ANEMOI_DATASETS_CODECS_THREADS`` for the
thread count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def _codec(store: Path, array: str):
    from anemoi_datasets_codecs.icon_lattice import LatticeCodec, register_store
    from anemoi_datasets_codecs.icon_lattice import store as st

    register_store(store)
    meta = st.array_meta(store, array)
    cfg = {k: v for k, v in meta["codec"].items() if k != "id"}
    return LatticeCodec.from_config(cfg), cfg, [p.relative_to(store) for p in st.chunk_paths(store, array)]


def _entry(codec, path: Path) -> dict:
    import numpy as np

    enc = path.read_bytes()
    dec = np.ascontiguousarray(codec.decode(enc), dtype=np.float32)
    return {
        "bytes": len(enc),
        "sha256": hashlib.sha256(enc).hexdigest(),
        "decoded_sha256": hashlib.sha256(dec.tobytes()).hexdigest(),
    }


def _provenance() -> dict:
    import numpy as np

    from anemoi_datasets_codecs.icon_lattice import __version__, native

    return {"package": __version__, "isa": native.isa(), "numpy": np.__version__}


def write(store: Path, out_dir: Path, array: str) -> None:
    codec, cfg, paths = _codec(store, array)
    if not paths:
        raise SystemExit(f"no chunk files in {store / array}")
    chunks = {}
    for rel in paths:
        t = time.perf_counter()
        chunks[str(rel)] = e = _entry(codec, store / rel)
        print(
            f"{rel}: {e['bytes']} B, sha256 {e['sha256'][:16]}, decoded {e['decoded_sha256'][:16]} "
            f"({time.perf_counter() - t:.2f} s)",
            flush=True,
        )
    doc = {"store": str(store), "array": array, "written_by": _provenance(), "codec": cfg, "chunks": chunks}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{store.resolve().name.removesuffix('.zarr')}.chunks.json"
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path}: {len(chunks)} chunks")


def check(store: Path, baseline: Path) -> int:
    doc = json.loads(baseline.read_text())
    array = doc.get("array", "data")
    codec, cfg, paths = _codec(store, array)
    print(f"baseline {baseline} (written by {doc.get('written_by')}); checking with {_provenance()}")
    bad = []

    def unstamped(c: dict) -> dict:
        return {k: v for k, v in c.items() if k != "written_by"}

    if unstamped(cfg) != unstamped(doc["codec"]):
        print("note: the store's codec configuration differs from the baseline's")
    for rel, want in doc["chunks"].items():
        p = store / rel
        if not p.is_file():
            bad.append(f"{rel}: missing")
            continue
        got = _entry(codec, p)
        if got["sha256"] != want["sha256"]:
            bad.append(f"{rel}: chunk file changed (sha256 {got['sha256'][:16]}, baseline {want['sha256'][:16]})")
        elif got["decoded_sha256"] != want["decoded_sha256"]:
            bad.append(
                f"{rel}: decodes DIFFERENTLY (sha256 {got['decoded_sha256'][:16]}, "
                f"baseline {want['decoded_sha256'][:16]})"
            )
        else:
            print(f"{rel}: bit-identical decode", flush=True)
    extra = sorted(set(map(str, paths)) - set(doc["chunks"]))
    if extra:
        print(f"not in the baseline: {extra}")
    for b in bad:
        print("FAIL", b)
    print(f"{len(doc['chunks']) - len(bad)} / {len(doc['chunks'])} chunks OK")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("store", type=Path)
    w.add_argument("out_dir", type=Path)
    w.add_argument("--array", default="data")
    c = sub.add_parser("check")
    c.add_argument("store", type=Path)
    c.add_argument("baseline", type=Path)
    args = ap.parse_args()
    if args.cmd == "write":
        write(args.store, args.out_dir, args.array)
        return 0
    return check(args.store, args.baseline)


if __name__ == "__main__":
    sys.exit(main())
