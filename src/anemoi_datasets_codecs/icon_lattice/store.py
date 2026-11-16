"""Store-level helpers: read an array's codec configuration from its ``zarr.json``, enumerate its chunk files, read
the static references of a chunk from its headers, and make a store self-contained (:func:`attach`) or check it
(:func:`check`). Local directory stores only. Plain lattice stores need no zarr; for a chained array (field stages
such as ``static_ref`` ahead of ``icon_lattice_bytes`` with ``chain: true``) the codec classes come from zarr's
registry and chunks are walked and decoded through ``anemoi_datasets_codecs.chain``.
"""

from __future__ import annotations

import itertools
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .. import chain, envelope
from ..config import STAMP_KEY
from . import container, grids, sidecar, statics

CODEC_NAMES = ("icon_lattice", "icon_lattice_bytes")


def array_meta(store: str | os.PathLike[str], array: str = "data") -> dict[str, Any]:
    """``{format, shape, chunks, codec, chain, codecs, separator, prefix}`` of a local array. ``codec``: the lattice
    codec's configuration (None in a chain without it); ``codecs``: the codec entries ``{name, configuration}`` after
    the ``bytes`` codec (all of them without one), in encode order; ``chain``: the same for an array with field
    stages, else None."""
    root = Path(store) / array
    if not (root / "zarr.json").is_file():
        raise FileNotFoundError(f"{root} is not a Zarr 3 array (no zarr.json)")
    m = json.loads((root / "zarr.json").read_text())
    if m.get("zarr_format") not in (None, 3) or m.get("node_type", "array") != "array":
        raise ValueError(f"{root} is not a Zarr 3 array")
    entries = []
    for c in m.get("codecs", []):
        c_cfg = dict(c.get("configuration") or {})
        c_cfg.pop("id", None)
        entries.append({"name": c.get("name"), "configuration": c_cfg})
    names = [e["name"] for e in entries]
    after = entries[names.index("bytes") + 1 :] if "bytes" in names else entries
    staged = any(e["name"] not in CODEC_NAMES and chain.is_stage(e["name"]) for e in after)
    codec = next((c for c in m.get("codecs", []) if c.get("name") in CODEC_NAMES), None)
    if codec is None and not staged:
        raise ValueError(f"{root}: no {' / '.join(CODEC_NAMES)} codec and no field stage in {names}")
    enc = m.get("chunk_key_encoding", {"name": "default", "configuration": {"separator": "/"}})
    sep = (enc.get("configuration") or {}).get("separator", "/")
    prefix = "c" + sep if enc.get("name", "default") == "default" else ""
    chunks = m["chunk_grid"]["configuration"]["chunk_shape"]
    cfg = None
    if codec is not None:
        cfg = dict(codec.get("configuration") or {})
        cfg.pop("id", None)
    return {
        "format": 3,
        "shape": m["shape"],
        "chunks": chunks,
        "codec": cfg,
        "chain": after if staged else None,
        "codecs": after,
        "separator": sep,
        "prefix": prefix,
    }


def chunk_paths(store: str | os.PathLike[str], array: str = "data") -> Iterator[Path]:
    """Existing chunk files of the array, in C order of the chunk grid."""
    m = array_meta(store, array)
    root = Path(store) / array
    grid = [-(-s // c) for s, c in zip(m["shape"], m["chunks"])]
    for idx in itertools.product(*(range(n) for n in grid)):
        key = m["separator"].join(map(str, idx))
        p = root / (m["prefix"] + key)
        if p.is_file():
            yield p


def _container_static_refs(buf: np.ndarray) -> set[str]:
    """sha256 of every STATIC_REF field of a lattice container (reads the header and 32 bytes per reference)."""
    hb = container.header_size(buf[: container.PREFIX])
    ch = container.parse(buf[:hb], payloads=False)
    return {bytes(buf[int(o) : int(o) + 32]).hex() for o in ch.offsets[:-1][ch.modes == container.MODE_STATIC_REF]}


def static_refs(path: Path, entries: list[dict[str, Any]] | None = None) -> set[str]:
    """sha256 of every static field a chunk file references in the sidecar: the STATIC_REF fields of a lattice
    container and, for a chunk in a chain envelope, the references in the stage headers (``static_ref``) plus those
    of the container inside. Reads the headers only, unless ``entries`` (the array's chain) end in a generic
    compressor, which is undone first."""
    buf = chain.read_chunk(path)
    if entries is not None:
        _, buf = chain.strip_compressors(entries, buf)
    env = envelope.parse(buf)
    try:
        if env is None:
            return _container_static_refs(buf)
        out = set(chain.header_references(env.headers).get("statics", set()))
        if env.kind == envelope.KIND_BYTES and bytes(env.payload[:4]) == container.MAGIC:
            out |= _container_static_refs(env.payload)
        return out
    except ValueError as e:
        what = "lattice codec" if env is None else "chain"
        raise ValueError(f"{path} is not a {what} chunk: {e}") from None


def decode_chunk_file(store: str | os.PathLike[str], path: Path, array: str = "data") -> np.ndarray:
    from .codec import LatticeCodec

    m = array_meta(store, array)
    if m["chain"] is not None:
        return chain.decode_chunk(m["chain"], path.read_bytes(), m["chunks"])
    codec = LatticeCodec.from_config(m["codec"])
    _, buf = chain.strip_compressors(m["codecs"], path.read_bytes())  # a generic compressor after the codec
    return codec.decode(buf, out=np.empty(m["chunks"], np.float32))


def attach(store: str | os.PathLike[str], array: str = "data", sample: int | None = None, verify: bool = True) -> dict:
    """Copy the lattice and the static fields the array's chunks reference into ``<store>/.codecs`` (from the
    sidecar search path: cache, ``$ANEMOI_DATASETS_CODECS_PATH``, ...), so the store is self-contained. ``sample``: scan
    only that many chunks (default, or 0: all). With ``verify``, decode the first chunk from the store's sidecar
    alone. A chained array: the statics of the stage headers too; the lattice only when the chain has the lattice
    codec."""
    store = Path(store)
    side = store / sidecar.SIDECAR
    m = array_meta(store, array)
    report: dict[str, Any] = {"store": str(store), "sidecar": str(side), "grid": None}
    if m["chain"] is not None:
        report["chain"] = [e["name"] for e in m["chain"]]
    if m["codec"] is not None:
        ref = grids.grid_ref(m["codec"]["grid"])
        report["grid"] = ref.get("name") or ref["sha256"][:12]
        data = grids.lattice_bytes(ref["sha256"])
        if data is None:
            raise sidecar.MissingReference(f"lattice {ref['sha256'][:12]} not found on {sidecar.describe_search()}")
        sidecar.write(side, "grids", ref["sha256"], data)
    shas: set[str] = set()
    paths = list(itertools.islice(chunk_paths(store, array), sample or None))
    for p in paths:
        shas |= static_refs(p, m["codecs"])
    missing = []
    for sha in sorted(shas):
        if (side / "statics" / sidecar.file_name("statics", sha)).is_file():
            continue
        b = statics.static_bytes(sha)
        if b is None:
            missing.append(sha)
        else:
            sidecar.write(side, "statics", sha, b)
    report.update(chunks_scanned=len(paths), statics=len(shas), missing=missing)
    if missing:
        raise sidecar.MissingReference(
            f"{len(missing)} static fields referenced by the chunks are not available: {[s[:12] for s in missing]}"
        )
    if verify and paths:
        with sidecar.isolated(store):
            decode_chunk_file(store, paths[0], array)
        report["verified"] = str(paths[0].relative_to(store))
    return report


def check(store: str | os.PathLike[str], array: str = "data", sample: int | None = 4) -> dict:
    """Check that the codec's variables match the store's ``variables`` attribute (anemoi; the ``attributes`` of
    ``<store>/zarr.json``), that the sidecar holds what the sampled chunks reference (the first ``sample``, all when
    None or 0), and that the first chunk decodes from the store's sidecar alone. A chained array: the variables of
    its first codec (the array's), and the consistency of the chain (each stage emits the next codec's variables).
    Problems are reported, not raised (a codec after this package's codecs the tools cannot undo is one). The report
    also gives the codecs' writer stamps (``written_by``), when they have one."""
    store = Path(store)
    m = array_meta(store, array)
    problems = []
    group = store / "zarr.json"
    attrs = (json.loads(group.read_text()).get("attributes") or {}) if group.is_file() else {}
    if m["chain"] is not None:
        names = chain.input_variables(m["chain"])
        problems += chain.chain_problems(m["chain"])
    else:
        names = m["codec"]["variables"]
    if "variables" in attrs and names is not None and list(attrs["variables"]) != list(names):
        problems.append(f"codec variables differ from the store's variables attribute: {names} vs {attrs['variables']}")
    side = store / sidecar.SIDECAR
    if m["codec"] is not None:
        sha = grids.grid_ref(m["codec"]["grid"])["sha256"]
        if not (side / "grids" / sidecar.file_name("grids", sha)).is_file():
            problems.append(f"lattice {sha[:12]} missing from {side}")
    paths = list(itertools.islice(chunk_paths(store, array), sample or None))
    root = store / array

    def key(p: Path) -> str:  # the chunk key relative to the array, e.g. c/1/0/0/0
        return p.relative_to(root).as_posix()

    missing: dict[str, list[str]] = {}  # sha -> chunk keys that reference it
    for p in paths:
        try:
            refs = static_refs(p, m["codecs"])
        except ValueError as e:  # an unreadable chunk or a codec the tools cannot undo: report, do not raise
            problems.append(f"reading the headers of {key(p)} failed: {e}")
            break
        for s in sorted(refs):
            if not (side / "statics" / sidecar.file_name("statics", s)).is_file():
                missing.setdefault(s, []).append(key(p))
    for s, keys in missing.items():
        by = ", ".join(keys[:3]) + (f" and {len(keys) - 3} more" if len(keys) > 3 else "")
        problems.append(f"static {s[:12]} (referenced by {by}) missing from {side}")
    if paths and not problems:
        try:
            with sidecar.isolated(store):
                decode_chunk_file(store, paths[0], array)
        except Exception as e:  # report, do not raise
            problems.append(f"decoding {key(paths[0])} from the store alone failed: {e}")
    out = {"store": str(store), "format": m["format"], "chunks_checked": len(paths), "problems": problems}
    if m["chain"] is not None:
        out["chain"] = [e["name"] for e in m["chain"]]
    stamps = written_by(m)
    if stamps:
        out["written_by"] = stamps
    return out


def written_by(meta: dict[str, Any]) -> dict[str, Any]:
    """The writer stamps of the array's codecs (``config.STAMP_KEY``), by codec name; codecs without one (stores
    written before the stamp existed, other packages' codecs) are left out. Informational only."""
    return {e["name"]: e["configuration"][STAMP_KEY] for e in meta["codecs"] if STAMP_KEY in e["configuration"]}
