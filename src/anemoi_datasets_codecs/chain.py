"""Codec chains of stored arrays, for the tools (``attach``, ``check``, ``info``; no zarr pipeline involved).

A chained array has field stages (``zarr3.FieldStageBytesCodec``) among the codecs of its ``zarr.json``, and its
chunk files start with the chain envelope (``envelope.py``) when the chain ends in a codec that keeps it (a
chain-mode chunk engine such as ``icon_lattice_bytes`` with ``chain: true``, or the last stage). This module reads
the codec list, finds each codec's class through zarr's codec registry (the ``zarr.codecs`` entry points, so a stage
of any installed package is known), walks a chunk's headers and decodes a chunk through the engines alone.

Generic compressors at the end of a chain (``[static_ref, blosc]``: zarr's ``blosc``, ``zstd``, ``gzip`` or a
``numcodecs.<id>`` codec) are undone with numcodecs (:func:`strip_compressors`) before the envelope is read.

A stage engine may offer, as static methods, ``header_references(header) -> {kind: {sha256, ...}}`` (the sidecar
files a chunk with this header needs, by sidecar kind: ``statics``, ``grids``) and ``describe_header(header,
configuration=None, shape=None) -> dict`` (a summary for ``info``). A stage without them references nothing.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import envelope

try:
    from zarr.errors import UnknownCodecError as _UnknownCodec
except ImportError:  # zarr releases without it raise KeyError or ValueError
    _UnknownCodec = ValueError  # type: ignore[assignment,misc]

# lossless codecs (numcodecs ids, a zarr ``numcodecs.`` prefix dropped) that return their input exactly: a field the
# lattice codec stores losslessly with one of them is exact
EXACT_LOSSLESS = frozenset({"blosc", "zstd", "lz4", "zlib", "gzip", "bz2", "lzma", "shuffle"})

__all__ = [
    "EXACT_LOSSLESS",
    "build",
    "chain_problems",
    "codec_class",
    "decode_chunk",
    "describe_header",
    "header_references",
    "input_variables",
    "is_stage",
    "read_chunk",
    "strip_compressors",
]


def codec_class(name: str, missing_ok: bool = False) -> type | None:
    """The Zarr codec class registered under ``name``. When zarr knows no such codec: ``ValueError``, or None with
    ``missing_ok``."""
    from zarr.registry import get_codec_class

    try:
        return get_codec_class(name)
    except (KeyError, _UnknownCodec):
        if missing_ok:
            return None
        raise ValueError(
            f"codec {name!r} is not known to zarr (a typo, or the package that provides it is not installed)"
        ) from None


def is_stage(name: str) -> bool:
    """Is ``name`` a field stage (a ``FieldStageBytesCodec`` subclass)?"""
    from .zarr3 import FieldStageBytesCodec

    cls = codec_class(name, missing_ok=True)
    return isinstance(cls, type) and issubclass(cls, FieldStageBytesCodec)


def _engine_class(name: str) -> type:
    cls = codec_class(name, missing_ok=True)
    engine = getattr(cls, "engine", None)
    if engine is None:
        raise ValueError(f"codec {name!r} is not a codec of this package (or the package that provides it is missing)")
    return engine


def build(entry: dict[str, Any]) -> Any:
    """The engine of a codec entry ``{name, configuration}`` (as the adapter builds it from ``zarr.json``)."""
    cls = codec_class(entry["name"], missing_ok=True)
    if cls is None or getattr(cls, "engine", None) is None:
        raise ValueError(f"codec {entry['name']!r} is not a codec of this package (or its package is missing)")
    return cls.from_dict({"name": entry["name"], "configuration": dict(entry.get("configuration") or {})}).codec


def read_chunk(path: str | os.PathLike[str]) -> np.ndarray:
    """A chunk file as a read-only uint8 array, memory-mapped: parsing headers reads only the pages they are on."""
    if Path(path).stat().st_size == 0:
        return np.empty(0, np.uint8)
    return np.memmap(path, np.uint8, mode="r")


def _numcodecs_decoder(entry: dict[str, Any]) -> Any:
    from numcodecs import Blosc, GZip, Zstd, get_codec

    name, cfg = entry["name"], dict(entry.get("configuration") or {})
    if name == "blosc":
        return Blosc()  # the frame is self-describing
    if name == "zstd":
        return Zstd()
    if name == "gzip":
        return GZip()
    if name.startswith("numcodecs."):
        return get_codec({"id": name.split(".", 1)[1], **cfg})
    raise ValueError(
        f"codec {name!r} after the codecs of this package is not handled by the tools (blosc, zstd, gzip, numcodecs.*)"
    )


def strip_compressors(entries: Sequence[dict[str, Any]], buf: Any) -> tuple[list[dict[str, Any]], Any]:
    """Undo the generic compressors at the end of a chain (the codecs after its last codec of this package): the
    remaining entries and the chunk as they left it."""
    entries = list(entries)
    while entries and getattr(codec_class(entries[-1]["name"], missing_ok=True), "engine", None) is None:
        buf = _numcodecs_decoder(entries.pop()).decode(buf)
    return entries, buf


def header_references(headers: Sequence[envelope.Header]) -> dict[str, set[str]]:
    """The sidecar files the stage headers of an envelope reference, by sidecar kind."""
    out: dict[str, set[str]] = {}
    for sid, payload in headers:
        fn = getattr(_engine_class(sid), "header_references", None)
        if fn is None:
            continue
        for kind, shas in fn(payload).items():
            out.setdefault(kind, set()).update(shas)
    return out


def describe_header(
    stage_id: str, header: bytes, configuration: dict[str, Any] | None = None, shape: Sequence[int] | None = None
) -> dict[str, Any]:
    """The stage's summary of its header (``{"bytes": n}`` when the stage offers none)."""
    fn = getattr(_engine_class(stage_id), "describe_header", None)
    out: dict[str, Any] = {"bytes": len(header)}
    if fn is not None:
        out.update(fn(header, configuration, None if shape is None else tuple(shape)))
    return out


def input_variables(entries: Sequence[dict[str, Any]]) -> list[str] | None:
    """The variables the first codec of the chain that names them receives: the array's variables."""
    for e in entries:
        names = (e.get("configuration") or {}).get("variables")
        if names is not None:
            return [n if isinstance(n, str) else n["name"] for n in names]
    return None


def chain_problems(entries: Sequence[dict[str, Any]]) -> list[str]:
    """Inconsistencies between the codecs of a chain, from the configurations alone (the rules no stage can see on
    its own; ``physics expand`` writes chains that satisfy them):

    * each stage emits the next codec's variables (``static_ref``'s output is the lattice codec's ``variables``), and
      a chunk engine behind stages has chain mode and is a bytes-to-bytes codec (``icon_lattice_bytes``, not the
      array-to-bytes ``icon_lattice``);
    * a terms stage (an engine with ``lattice_bounds``, i.e. ``field_terms``): every target reaches the chunk engine
      with a bound of at most its ``b_L`` (or none) and without a clip there; every source is exact through the
      codecs after the stage: not a target of a later terms stage, and either restored by a later stage that drops
      it (``static_ref``) or, in the chunk engine, without a bound, not a recipe of its resolved plan and stored
      losslessly by an exact codec (``EXACT_LOSSLESS``).

    A codec zarr does not know is a problem; a codec that cannot be built is one problem, and the comparisons with
    its neighbours are skipped.
    """
    from zarr.abc.codec import ArrayBytesCodec

    problems = []
    engines: list[tuple[str, Any, bool]] = []  # (name, engine or None when it failed to build, is a stage)
    for e in entries:
        cls = codec_class(e["name"], missing_ok=True)
        if cls is None:
            problems.append(
                f"codec {e['name']!r} is not known to zarr (a typo, or the package that provides it is not installed)"
            )
            continue
        if engines and any(st for _, _, st in engines) and issubclass(cls, ArrayBytesCodec):
            problems.append(
                f"codec {e['name']!r} is an array-to-bytes codec and cannot follow the field stages (in a chain the "
                "chunk engine is the bytes-to-bytes variant, e.g. icon_lattice_bytes)"
            )
        if getattr(cls, "engine", None) is None:
            continue  # a generic compressor: it has no variables
        try:
            engines.append((e["name"], build(e), is_stage(e["name"])))
        except Exception as err:  # report, do not raise
            problems.append(f"codec {e['name']!r} cannot be built from its configuration: {err}")
            engines.append((e["name"], None, is_stage(e["name"])))
    for (n0, a, stage0), (n1, b, stage1) in zip(engines, engines[1:]):
        if a is None or b is None:
            continue  # reported above
        if not stage0:
            problems.append(f"codec {n1!r} follows {n0!r}, which is not a field stage")
            continue
        want, got = list(a.variables_out), list(b.names)
        if want != got:
            problems.append(f"{n0!r} emits the variables {want}, the next codec {n1!r} is configured for {got}")
        if not stage1 and not getattr(b, "chain", False):
            problems.append(f"{n1!r} follows field stages and needs chain: true in its configuration")
    for i, (name, eng, stage) in enumerate(engines):
        if eng is not None and stage and getattr(eng, "lattice_bounds", None) is not None:
            problems += _terms_problems(name, eng, engines[i + 1 :])
    return problems


def _terms_problems(name: str, stage: Any, later: Sequence[tuple[str, Any, bool]]) -> list[str]:
    """The rules of a terms stage against the codecs after it (``chain_problems``; an engine None failed to build:
    nothing is compared with it or beyond it)."""
    problems = []
    chunk = next(((n, e) for n, e, st in later if not st or e is None), None)
    if chunk is not None and chunk[1] is None:
        chunk = None
    lat_vars = {v.name: v for v in getattr(chunk[1], "variables", ())} if chunk is not None else {}
    plan = getattr(chunk[1], "plan", None) if chunk is not None else None
    recipes = set(getattr(plan, "recipes", None) or ())
    lossless = getattr(chunk[1], "lossless_config", None) if chunk is not None else None
    lossless_id = None if lossless is None else str(lossless.get("id", "")).removeprefix("numcodecs.")
    if chunk is not None:
        for t, b_l in stage.lattice_bounds.items():
            v = lat_vars.get(t)
            if v is None:
                continue  # dropped on the way (reported by the variable checks when the chain disagrees)
            if v.max_error is not None and v.max_error > b_l:
                problems.append(
                    f"{name!r} target {t!r}: {chunk[0]!r} codes it within {v.max_error:g}, more than b_L = {b_l:g} "
                    "(the stage's bound b (1 - margin))"
                )
            if v.clip is not None:
                problems.append(f"{name!r} target {t!r}: {chunk[0]!r} clips it (it codes the stage's residual)")
    for src in sorted({s for tm in stage.terms for s in tm.sources}):
        for n, e, st in later:
            if e is None:
                break  # failed to build: reported
            if st and src in getattr(e, "targets", ()):
                problems.append(f"{name!r} source {src!r} is a target of the later stage {n!r}: not exact")
                break
            if st and src in e.variables and src not in e.variables_out:
                break  # dropped and restored exactly by that stage (static_ref)
            if not st:
                v = lat_vars.get(src)
                if v is not None and v.max_error is not None:
                    problems.append(f"{name!r} source {src!r} has a bound in {n!r} (lossy there): not exact")
                if src in recipes:
                    problems.append(f"{name!r} source {src!r} is a forcing recipe of {n!r}: not exact")
                if v is not None and v.max_error is None and lossless_id is not None:
                    if lossless_id not in EXACT_LOSSLESS:
                        problems.append(
                            f"{name!r} source {src!r} is stored by {n!r} with its lossless codec {lossless_id!r}, "
                            f"which is not known to be exact (exact: {', '.join(sorted(EXACT_LOSSLESS))})"
                        )
                break
    return problems


def decode_chunk(entries: Sequence[dict[str, Any]], buf: Any, chunk_shape: Sequence[int]) -> np.ndarray:
    """Decode a stored chunk of a chained array through the engines of its codecs (``entries``: the codecs after the
    ``bytes`` codec, encode order). The chunk is an envelope once the generic compressors at the end are undone
    (:func:`strip_compressors`): the chain ends in a chain-mode chunk engine (the payload is its bytes) or in a stage
    (a float32 array)."""
    entries, buf = strip_compressors(entries, buf)
    env = envelope.parse(buf)
    if env is None:
        raise ValueError("the chunk is not a chain envelope; read it with zarr instead")
    stages = [e for e in entries if is_stage(e["name"])]
    if env.kind == envelope.KIND_BYTES:
        last = entries[-1]
        if is_stage(last["name"]):
            raise ValueError("the envelope holds the bytes of a later codec, but the chain ends in a field stage")
        if env.shape[0] != chunk_shape[0] or env.shape[2:] != tuple(chunk_shape[2:]):
            raise ValueError(f"the envelope holds an array of shape {env.shape}, the array has chunks {chunk_shape}")
        engine = build(last)
        per = getattr(engine, "fields_per_date", None)
        if per is not None and env.shape[1] * env.shape[2] != per:
            raise ValueError(f"the envelope holds {env.shape[1]} fields, {last['name']} codes {per} per date")
        from .zarr3 import _decode_into

        x = _decode_into(engine, env.payload, np.empty(env.shape, np.float32))  # header checked against the shape
    else:
        x = env.array().copy()
    headers = list(env.headers)
    if [h[0] for h in headers] != [e["name"] for e in stages]:
        raise ValueError(f"the chunk's stage headers {[h[0] for h in headers]} are not the chain's stages")
    for e in reversed(stages):
        engine = build(e)
        _, header = headers.pop()
        out_shape = (x.shape[0], engine.fields_in, *x.shape[2:])
        out = None if out_shape == x.shape else np.empty(out_shape, np.float32)
        x = engine.decode(x, header, out_shape, out=out)
    if tuple(x.shape) != tuple(chunk_shape):
        raise ValueError(f"the chain decoded a chunk of shape {tuple(x.shape)}, the array has chunks {chunk_shape}")
    return x
