"""Golden chunks: small synthetic chunks written once and kept in tests/data, the decoder-side guard of every frozen
element (``tests/test_codec.py::test_golden_chunk_decodes_bit_identically`` decodes the STORED bytes with every
kernel variant and compares the sha256 of the decoded float32; it never regenerates them).

    python scripts/golden_chunks.py add      # write the chunks of GOLDEN that tests/data/golden_chunks.json lacks
    python scripts/golden_chunks.py sizes    # encode every chunk of GOLDEN (nothing written) and print its size

``add`` never rewrites an existing entry or file: a chunk once written is a fixture of the format, whatever the
encoder that wrote it (golden chunks are decoder fixtures: any valid chunk will do, so a later encoder change is no
reason to touch them). The first entry, ``tree`` (``golden_chunk_tree.latc``), predates this script; it was written
with the defaults of that time (``rounding: grid``, ``recipe_fit: false``), the entries below with those of release
0.1.0 (``rounding: direct``, ``recipe_fit: true``), each with its complete configuration.

Every chunk is built from the synthetic fixtures of ``tests/conftest.py`` (the test lattice, 16361 cells, and fields
derived from ``make_chunk`` / ``smooth``), one date, one member. The static fields a chunk references through the
sidecar go to ``tests/data/golden_sidecar/statics/`` (the test puts that directory alone on the search path).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
import conftest  # noqa: E402

DATA = REPO / "tests" / "data"
INDEX = DATA / "golden_chunks.json"
SIDECAR = DATA / "golden_sidecar"
LZ4 = {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1}


def _fields(lattice) -> dict[str, np.ndarray]:
    """The fields of the golden chunks: those of ``conftest.make_chunk`` plus a few derived ones (float32)."""
    layout, rr, cc, lat, lon = lattice
    base = dict(zip(conftest.NAMES, conftest.make_chunk(lattice), strict=True))
    fis = np.where(base["FIS"] > 3000, 2500.0, 500.0)  # two values: a small static file and inline payload
    shared = conftest.smooth(rr, cc, 11, 4)
    t2m = 285 - 0.0065 * fis / 9.81 + 3 * conftest.smooth(rr, cc, 3) + 0.5 * shared
    ps = 100000 + 900 * conftest.smooth(rr, cc, 12, 25)
    below = ps < 100000  # the 1000 hPa level is below ground there
    t1000 = np.where(below, t2m + 0.0065 * (100000 - ps) / 12.0, t2m - 1 + 2 * conftest.smooth(rr, cc, 13, 30))
    return {
        **base,
        "FIS": fis,
        "T_2M": t2m,
        "PS": ps,
        "T_1000": t1000,
    }


def _bounds(names: list[str]) -> dict[str, float]:
    known = {n: b for n, b, _ in conftest.VARS if b is not None}
    known.update({"PS": 2.0, "T_1000": 0.05})
    return {n: known[n] for n in names if n in known}


def _lattice_config(layout, names: list[str], **kw) -> dict:
    clip = {n: list(c) for n, _, c in conftest.VARS if c is not None and n in names}
    return {"variables": names, "grid": layout, "bounds": _bounds(names), "clip": clip, "lossless": dict(LZ4), **kw}


# name -> what it covers, and the codec configuration (lattice codec alone) or "chain"
GOLDEN = {
    "tree_b": (
        "structure tree_b; a clip; exact zeros of a precipitation field",
        lambda lay: _lattice_config(lay, ["PMSL", "CLCT", "TOT_PREC_1H"], structure="tree_b"),
    ),
    "hc2_k3": (
        "structure hc2_k3; a cross term on a lattice source (T_2M <- PMSL)",
        lambda lay: _lattice_config(lay, ["T_2M", "PMSL"], structure="hc2_k3"),
    ),
    "below_ground": (
        "a below-ground mask term (T_1000 where PS < 1000 hPa, from T_2M and PS); cross terms",
        lambda lay: _lattice_config(lay, ["PS", "T_2M", "T_1000"], physics={"below_ground": True}),
    ),
    "recipes_static_ref": (
        "RECIPE fields (coefficients fitted without a date, recipe_fit) whose inputs are STATIC_REF fields resolved "
        "through the sidecar (tests/data/golden_sidecar)",
        lambda lay: _lattice_config(
            lay,
            [
                "sin_latitude",
                "cos_latitude",
                "sin_longitude",
                "cos_longitude",
                "cos_local_time",
                "sin_local_time",
                "insolation",
            ],
            static_fields=["sin_latitude", "cos_latitude", "sin_longitude", "cos_longitude"],
        ),
    ),
    "static_inline": (
        "a STATIC_REF field stored inline (options.statics: inline) and used as a cross source (static details)",
        lambda lay: _lattice_config(lay, ["T_2M", "FIS"], static_fields=["FIS"], options={"statics": "inline"}),
    ),
    "chain": (
        "the chain field_terms (T_2M <- FIS) + static_ref (FIS through the sidecar) + icon_lattice_bytes (chain: "
        "true): the FCHN envelope with both stage headers, decoded through the chain",
        None,
    ),
}
CHAIN_VARS = ["T_2M", "PMSL", "FIS", "CLCT"]


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _lattice_chunk(name: str, lattice, fields) -> tuple[dict, bytes, bytes, dict[str, bytes]]:
    """(index entry without file, chunk bytes, decoded bytes, the sidecar files of the statics it references)."""
    from anemoi_datasets_codecs import statics
    from anemoi_datasets_codecs.icon_lattice import LatticeCodec, container

    cfg = GOLDEN[name][1](lattice[0])
    codec = LatticeCodec(**cfg)
    x = np.stack([fields[n] for n in codec.names]).astype(np.float32)
    files = {}
    if codec.options["statics"] == "ref":
        for n in codec.static_fields:
            v = x[codec.names.index(n)]
            files[statics.sha256_of(v)] = statics.static_file_bytes(v)
            statics.add_static(v)  # durable: referenced, not captured
    enc = codec.encode(x)
    modes = [int(m) for m in container.parse(enc).modes]
    dec = np.ascontiguousarray(codec.decode(enc), np.float32)
    entry = {"covers": GOLDEN[name][0], "modes": modes, "configuration": codec.get_config()}
    return entry, enc, dec.tobytes(), files


def _chain_chunk(lattice, fields) -> tuple[dict, bytes, bytes, dict[str, bytes]]:
    """Written through zarr (the real pipeline) into a temporary store: the chunk file and the codec entries of its
    ``zarr.json`` (stamps included)."""
    import zarr

    from anemoi_datasets_codecs import chain, envelope, physics
    from anemoi_datasets_codecs.icon_lattice import grids

    layout = lattice[0]
    ref = grids.register_lattice(layout)
    lat = _lattice_config(ref, CHAIN_VARS, static_fields=["FIS"])
    entries = physics.expand(CHAIN_VARS, None, lat)
    assert [e["name"] for e in entries] == ["field_terms", "static_ref", "icon_lattice_bytes"], entries
    x = np.stack([fields[n] for n in CHAIN_VARS]).astype(np.float32)[None, :, None, :]
    with tempfile.TemporaryDirectory() as tmp:
        g = zarr.open_group(f"{tmp}/s.zarr", mode="w", zarr_format=3)
        a = g.create_array(
            "data",
            shape=x.shape,
            chunks=x.shape,
            dtype="float32",
            compressors=[chain.codec_class(e["name"])(**e["configuration"]) for e in entries],
        )
        a[:] = x
        meta = json.loads(Path(f"{tmp}/s.zarr/data/zarr.json").read_text())
        stored = [c for c in meta["codecs"] if c["name"] != "bytes"]
        enc = Path(f"{tmp}/s.zarr/data/c/0/0/0/0").read_bytes()
        env = envelope.parse(enc)
        assert [h[0] for h in env.headers] == ["field_terms", "static_ref"], env.headers
        files = {p.name.split(".")[0]: p.read_bytes() for p in Path(f"{tmp}/s.zarr/.codecs/statics").iterdir()}
        dec = np.ascontiguousarray(chain.decode_chunk(stored, enc, x.shape), np.float32)
    np.testing.assert_array_equal(dec[0, 2, 0], x[0, 2, 0])  # FIS exact
    entry = {"covers": GOLDEN["chain"][0], "shape": list(x.shape), "chain": stored}
    return entry, enc, dec.tobytes(), files


def build(name: str, lattice, fields) -> tuple[dict, bytes, bytes, dict[str, bytes]]:
    return _chain_chunk(lattice, fields) if name == "chain" else _lattice_chunk(name, lattice, fields)


def main() -> None:
    from anemoi_datasets_codecs import __version__, sidecar

    mode = sys.argv[1]
    lattice = conftest.make_lattice()
    fields = {k: np.asarray(v, np.float32) for k, v in _fields(lattice).items()}
    with tempfile.TemporaryDirectory() as cache:
        os.environ[sidecar.CACHE_ENV] = cache  # captures land here, never in the user's cache or tests/data
        os.environ.pop(sidecar.PATH_ENV, None)
        index = json.loads(INDEX.read_text())
        for name in GOLDEN:
            if mode == "add" and name in index:
                print(f"{name}: kept")
                continue
            entry, enc, dec, files = build(name, lattice, fields)
            print(f"{name}: {len(enc)} bytes, statics in the sidecar {[s[:12] for s in files]}")
            if mode != "add":
                continue
            path = DATA / f"golden_chunk_{name}.{'fchn' if name == 'chain' else 'latc'}"
            if path.exists():
                raise SystemExit(f"{path} exists: golden chunks are never rewritten")
            path.write_bytes(enc)
            for s, data in files.items():  # the static files the chunk references, in the golden sidecar
                sidecar.write(SIDECAR, "statics", s, data)
            index[name] = {
                "file": path.name,
                "chunk_sha256": _sha(enc),
                "decoded_sha256": _sha(dec),
                "sidecar_statics": sorted(files),
                "written_by": f"anemoi-datasets-codecs {__version__}",
                **entry,
            }
            INDEX.write_text(json.dumps(index, indent=1) + "\n")


if __name__ == "__main__":
    main()
