"""lab fuzz: a mutation fuzzer for the decode path, and an in-process fuzzer of the header parsers.

A valid chunk (the synthetic test chunk, or a chunk file of a store) is mutated: random bit flips, byte overwrites,
insertions, deletions, truncations, duplicated / zeroed blocks, and *targeted* mutations of the structured fields
(container header, field records, LATTICE header counts and sizes, steps blob, tables, rANS stream headers; with a
chain also the ``FCHN`` envelope and the stage headers) set to boundary values. For a chain the inner buffers (the
container in the envelope payload, a stage header) are mutated and re-wrapped in a consistent envelope, so that the
mutation reaches the codec behind the envelope check.

Every mutant is decoded the way a reader would (``LatticeCodec.decode`` for a plain container, ``chain.decode_chunk``
for an enveloped chunk) in a pool of worker subprocesses, each with a timeout, so that a crash or a hang of the C code
is caught and classified. A crash or hang is re-run once in a fresh worker (a worker that decoded earlier mutants may
carry corrupted state). Outcomes:

* ``clean_error``: a ``ValueError`` or another exception class of this package (the codec's own errors, including
  the native ``ERRORS``);
* ``error_other``: another Python exception (``MemoryError``, ``IndexError``, ``struct.error`` ...): the input was
  rejected, but not by a check of the codec;
* ``accepted_same`` / ``accepted_diff``: decoded without error, to the original's values or to different ones (a
  weakness: the format has no integrity check, so a corrupted chunk can decode to wrong values);
* ``sanitizer``: decoded or failed, but ASan / UBSan printed a report (build the kernels with
  ``scripts/lab/build_asan.sh``);
* ``crash``: the worker died (signal, abort, ASan report); ``crash_unconfirmed`` when the fresh re-run did not die;
* ``hang`` / ``hang_unconfirmed``: no answer within the timeout.

Non-clean cases are saved under the output directory as ``<outcome>/<case>.bin`` (the mutant) and ``.json`` (mutation,
outcome, exception, stderr). Seeds make every case reproducible: case ``i`` of seed ``s`` is always the same mutant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import selectors
import signal
import struct
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import envelope
from ..icon_lattice import container

OUTCOMES = (
    "unchanged",
    "clean_error",
    "error_other",
    "accepted_same",
    "accepted_diff",
    "sanitizer",
    "crash",
    "crash_unconfirmed",
    "hang",
    "hang_unconfirmed",
)
SAVED = {"error_other", "accepted_diff", "sanitizer", "crash", "crash_unconfirmed", "hang", "hang_unconfirmed"}
SAVE_CAP = {"error_other": 5, "accepted_diff": 20, "accepted_same": 0, "sanitizer": 200, "crash": 200, "hang": 200}
SANITIZER_MARKERS = (b"AddressSanitizer", b"runtime error:", b"UndefinedBehaviorSanitizer", b"LeakSanitizer")

# a "region" is a field of a buffer: (group, start, end, width, is_float). A scalar has end - start == width; a longer
# region is a blob in which an aligned-or-not integer of ``width`` bytes is chosen at random.
Region = tuple[str, int, int, int, bool]


# ============================================================================================== mutations
def _interesting(rng: random.Random, width: int, cur: int, is_float: bool) -> int:
    top = (1 << (8 * width)) - 1
    if is_float and width == 4:
        special = [0x7FC00000, 0x7F800000, 0xFF800000, 0x80000000, 0x00000001, 0x7F7FFFFF, 0x00800000, 0x3F800000]
        if rng.random() < 0.6:
            return rng.choice(special)
    choices = [
        0,
        1,
        2,
        top,
        top - 1,
        (top >> 1),
        (top >> 1) + 1,
        (cur + 1) & top,
        (cur - 1) & top,
        cur ^ (1 << rng.randrange(8 * width)),
        rng.randrange(top + 1),
        rng.randrange(256) if width > 1 else 0,
        (cur << 1) & top,
        cur >> 1,
        255,
        256 & top,
        32,
        33,
        1 << 30,
        (1 << 31) & top,
        (1 << 32) & top if width > 4 else 0,
    ]
    return rng.choice(choices) & top


def _set_int(b: bytearray, off: int, width: int, value: int) -> None:
    b[off : off + width] = int(value).to_bytes(width, "little")


def mutate_region(b: bytearray, regions: Sequence[Region], group: str, rng: random.Random) -> str:
    """Set an integer of one region of ``group`` to an interesting value; returns the description."""
    cands = [r for r in regions if r[0] == group]
    name, start, end, width, is_float = rng.choice(cands)
    if end - start > width:  # a blob: a width-byte integer at a random offset
        width = min(rng.choice((1, 2, 4, 4, 8)), end - start)
        off = start + rng.randrange(end - start - width + 1)
    else:
        off = start
    cur = int.from_bytes(b[off : off + width], "little")
    new = _interesting(rng, width, cur, is_float)
    _set_int(b, off, width, new)
    return f"{name} @{off} w{width}: {cur:#x} -> {new:#x}"


def generic_mutation(data: bytes, rng: random.Random, op: str | None = None) -> tuple[bytes, str, str]:
    """A random unstructured mutation of ``data``: ``(mutant, op, description)``."""
    n = len(data)
    op = op or rng.choice(("bitflip", "bitflip", "byteset", "insert", "delete", "truncate", "extend", "dup", "zero"))
    b = bytearray(data)
    if n == 0:
        return bytes(b + bytes(rng.randrange(256) for _ in range(8))), "extend", "extend an empty buffer"
    if op == "bitflip":
        k = rng.choice((1, 1, 1, 2, 3, 8))
        pos = [(rng.randrange(n), rng.randrange(8)) for _ in range(k)]
        for p, bit in pos:
            b[p] ^= 1 << bit
        return bytes(b), op, "flip " + ", ".join(f"{p}.{bit}" for p, bit in pos)
    if op == "byteset":
        k = rng.choice((1, 1, 2, 4))
        pos = [(rng.randrange(n), rng.choice((0, 255, 127, 128, 1, rng.randrange(256)))) for _ in range(k)]
        for p, v in pos:
            b[p] = v
        return bytes(b), op, "set " + ", ".join(f"{p}={v}" for p, v in pos)
    if op == "insert":
        p, k = rng.randrange(n + 1), rng.choice((1, 2, 4, 8, 16, 64, rng.randrange(1, 4096)))
        fill = rng.choice((b"\0", b"\xff", None))
        ins = fill * k if fill else bytes(rng.randrange(256) for _ in range(k))
        return bytes(b[:p] + ins + b[p:]), op, f"insert {k} bytes at {p}"
    if op == "delete":
        p, k = rng.randrange(n), rng.choice((1, 2, 4, 8, 16, 64, rng.randrange(1, 4096)))
        del b[p : p + k]
        return bytes(b), op, f"delete {k} bytes at {p}"
    if op == "truncate":
        keep = rng.choice((0, 1, 4, 39, 40, 56, rng.randrange(n), rng.randrange(n), n - 1, n - 4, max(0, n - 32)))
        keep = max(0, min(n - 1, keep)) if n > 1 else 0
        return bytes(b[:keep]), op, f"truncate to {keep} of {n}"
    if op == "extend":
        k = rng.choice((1, 4, 8, 32, 1024))
        fill = rng.choice((b"\0", b"\xff", None))
        return bytes(b + (fill * k if fill else bytes(rng.randrange(256) for _ in range(k)))), op, f"append {k} bytes"
    if op == "dup":
        k = min(rng.choice((4, 16, 256, 4096)), n)
        s, d = rng.randrange(n - k + 1), rng.randrange(n - k + 1)
        b[d : d + k] = data[s : s + k]
        return bytes(b), op, f"copy {k} bytes from {s} to {d}"
    k = min(rng.choice((4, 16, 256, 4096)), n)
    s = rng.randrange(n - k + 1)
    b[s : s + k] = bytes(k)
    return bytes(b), "zero", f"zero {k} bytes at {s}"


# ----------------------------------------------------------------------------------- regions of the formats
def container_regions(buf: bytes) -> list[Region]:
    """Fields of a lattice container (docs/codecs/icon_lattice.md): header, records, and every LATTICE payload's
    header, sections and rANS stream header."""
    R: list[Region] = [
        ("container.magic", 0, 4, 4, False),
        ("container.version", 4, 6, 2, False),
        ("container.flags", 6, 8, 2, False),
        ("container.n_fields", 8, 12, 4, False),
        ("container.n_dates", 12, 16, 4, False),
        ("container.header_bytes", 16, 20, 4, False),
        ("container.reserved", 20, 24, 4, False),
        ("container.sha", 24, 40, 1, False),
    ]
    ch = container.parse(buf)
    n, nd = len(ch.modes), len(ch.dates)
    R.append(("container.dates", 40, 40 + 8 * nd, 8, False))
    rec = 40 + 8 * nd
    for i in range(n):
        o = rec + 16 * i
        R += [
            ("record.mode", o, o + 1, 1, False),
            ("record.fallback", o + 1, o + 2, 1, False),
            ("record.reserved", o + 2, o + 8, 2, False),
            ("record.nbytes", o + 8, o + 16, 8, False),
        ]
    for i in range(n):
        p0, p1 = int(ch.offsets[i]), int(ch.offsets[i + 1])
        mode = int(ch.modes[i])
        if mode == container.MODE_LATTICE and p1 - p0 >= container.LATTICE_HEADER:
            R += _lattice_regions(buf, p0, p1)
        elif mode == container.MODE_RECIPE:
            R += [("recipe.header", p0, min(p1, p0 + 64), 4, False), ("recipe.patch", min(p1, p0 + 64), p1, 4, True)]
        elif mode == container.MODE_STATIC_REF:
            R.append(("staticref.sha", p0, p1, 1, False))
        elif mode == container.MODE_CONSTANT:
            R.append(("constant.value", p0, p1, 4, True))
        elif mode == container.MODE_LOSSLESS:
            R.append(("lossless.blob", p0, p1, 1, False))
    return [r for r in R if r[2] > r[1]]


def _lattice_regions(buf: bytes, p0: int, p1: int) -> list[Region]:
    h = container.lattice_header(memoryview(buf)[p0:p1])
    R: list[Region] = []
    for name, o, w, f in (
        ("vt", 0, 1, False),
        ("et", 1, 1, False),
        ("n_steps", 2, 1, False),
        ("n_tables", 3, 1, False),
        ("n_cross", 4, 1, False),
        ("flags", 5, 1, False),
        ("pbits", 6, 1, False),
        ("residual_shift", 7, 1, False),
        ("offset", 8, 4, True),
        ("step", 12, 4, True),
        ("clip_lo", 16, 4, True),
        ("clip_hi", 20, 4, True),
        ("n_esc", 24, 4, False),
        ("rans_bytes", 28, 4, False),
        ("tables_bytes", 32, 4, False),
        ("steps_bytes", 36, 4, False),
        ("cross_src", 40, 2, False),
        ("cross_src", 42, 2, False),
        ("cross_src", 44, 2, False),
        ("cross_src", 46, 2, False),
        ("cross_shift", 48, 1, False),
        ("cross_shift", 49, 1, False),
        ("cross_shift", 50, 1, False),
        ("cross_shift", 51, 1, False),
        ("header_zero", 52, 4, False),
    ):
        R.append((f"lattice.{name}", p0 + o, p0 + o + w, w, f))
    pos = p0 + container.LATTICE_HEADER
    if h.flags & 4:
        nfeat = buf[pos + 2]
        size = container.MASK_BLOCK + 16 * min(nfeat, 8)
        R += [
            ("mask.src", pos, pos + 2, 2, False),
            ("mask.n_feat", pos + 2, pos + 3, 1, False),
            ("mask.table", pos + 3, pos + 4, 1, False),
            ("mask.threshold", pos + 4, pos + 8, 4, True),
            ("mask.coefs", pos + 8, pos + size, 4, True),
        ]
        pos += size
    steps_end = min(p1, pos + h.steps_bytes)
    R.append(("steps.blob", pos, steps_end, 4, False))
    R.append(("steps.tail", max(pos, steps_end - 16), steps_end, 4, False))
    pos = steps_end
    t_end = min(p1, pos + h.tables_bytes)
    R.append(("tables.blob", pos, t_end, 1, False))
    pos = t_end
    r_end = min(p1, pos + h.rans_bytes)
    if r_end - pos >= 8:
        R += [("rans.lanes", pos, pos + 4, 4, False), ("rans.state", pos + 4, min(r_end, pos + 132), 4, False)]
        R.append(("rans.words", min(r_end, pos + 132), r_end, 2, False))
    pos = r_end
    R.append(("escapes.blob", pos, p1, 1, False))
    return [r for r in R if r[2] > r[1]]


def envelope_regions(buf: bytes) -> list[Region]:
    """Fields of the ``FCHN`` envelope prefix (docs/codecs/envelope.md)."""
    R: list[Region] = [
        ("env.magic", 0, 4, 4, False),
        ("env.version", 4, 5, 1, False),
        ("env.ndim", 5, 6, 1, False),
        ("env.kind", 6, 7, 1, False),
        ("env.reserved0", 7, 8, 1, False),
        ("env.n_headers", 8, 10, 2, False),
        ("env.reserved1", 10, 12, 2, False),
        ("env.header_bytes", 12, 16, 4, False),
        ("env.payload_bytes", 16, 24, 8, False),
    ]
    ndim = buf[5]
    R.append(("env.shape", 24, 24 + 8 * ndim, 8, False))
    off = 24 + 8 * ndim
    hb = struct.unpack_from("<I", buf, 12)[0]
    for _ in range(struct.unpack_from("<H", buf, 8)[0]):
        if off + 8 > hb:
            break
        n_id, _, m = struct.unpack_from("<HHI", buf, off)
        R += [
            ("env.record.id_bytes", off, off + 2, 2, False),
            ("env.record.zero", off + 2, off + 4, 2, False),
            ("env.record.header_bytes", off + 4, off + 8, 4, False),
            ("env.record.id", off + 8, off + 8 + n_id, 1, False),
            ("env.record.payload", off + 8 + n_id, off + 8 + n_id + m, 4, False),
        ]
        end = off + 8 + n_id + m
        off = end + (-end % 8)
    return [r for r in R if r[2] > r[1]]


def static_ref_regions(buf: bytes) -> list[Region]:
    """Fields of a ``static_ref`` stage header (docs/codecs/static_ref.md)."""
    R: list[Region] = [
        ("static_ref.version", 0, 1, 1, False),
        ("static_ref.reserved", 1, 4, 1, False),
        ("static_ref.n_static", 4, 8, 4, False),
        ("static_ref.n_slots", 8, 12, 4, False),
        ("static_ref.reserved2", 12, 16, 4, False),
    ]
    off, n = 16, struct.unpack_from("<I", buf, 8)[0]
    for _ in range(n):
        if off + 40 > len(buf):
            break
        arg = struct.unpack_from("<I", buf, off + 4)[0]
        R += [
            ("static_ref.slot.kind", off, off + 1, 1, False),
            ("static_ref.slot.reserved", off + 1, off + 4, 1, False),
            ("static_ref.slot.arg", off + 4, off + 8, 4, False),
            ("static_ref.slot.sha", off + 8, off + 40, 1, False),
        ]
        off += 40
        if buf[off - 40] == 1:
            R.append(("static_ref.slot.data", off, min(len(buf), off + arg), 1, False))
            off += arg
    return [r for r in R if r[2] > r[1]]


def field_terms_regions(buf: bytes) -> list[Region]:
    """Fields of a ``field_terms`` stage header (docs/codecs/field_terms.md)."""
    R: list[Region] = [
        ("field_terms.version", 0, 1, 1, False),
        ("field_terms.reserved", 1, 2, 1, False),
        ("field_terms.n_terms", 2, 4, 2, False),
        ("field_terms.n_slices", 4, 8, 4, False),
    ]
    off, nt = 8, struct.unpack_from("<H", buf, 2)[0]
    for _ in range(nt):
        if off + 4 > len(buf):
            break
        k = struct.unpack_from("<H", buf, off + 2)[0]
        R += [
            ("field_terms.term.target", off, off + 2, 2, False),
            ("field_terms.term.n_sources", off + 2, off + 4, 2, False),
            ("field_terms.term.sources", off + 4, off + 4 + 2 * k, 2, False),
        ]
        off += 4 + 2 * k
    off += -off % 4
    R.append(("field_terms.records", off, len(buf), 4, True))
    R.append(("field_terms.record.flags", off, min(len(buf), off + 4), 1, False))
    return [r for r in R if r[2] > r[1]]


STAGE_REGIONS: dict[str, Callable[[bytes], list[Region]]] = {
    "static_ref": static_ref_regions,
    "field_terms": field_terms_regions,
}


# ---------------------------------------------------------------------------------------------- scopes
@dataclass
class Scope:
    """A buffer to mutate: its bytes, its regions and how to put a mutated copy back into the whole chunk."""

    name: str
    buf: bytes
    regions: list[Region]
    wrap: Callable[[bytes], bytes] = lambda b: b
    groups: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.groups = sorted({r[0] for r in self.regions})


def scopes_of(chunk: bytes, chain: bool) -> list[Scope]:
    """The scopes of a valid chunk: the container alone, or for an enveloped chunk the whole buffer, the payload and
    every stage header (the last two re-wrapped in a consistent envelope)."""
    if not chain:
        return [Scope("container", chunk, container_regions(chunk))]
    env = envelope.parse(chunk)
    if env is None:
        raise ValueError("the chunk is not a chain envelope")
    out = [Scope("envelope", chunk, envelope_regions(chunk))]

    def rewrap(headers: list, payload: bytes) -> bytes:
        if env.kind == envelope.KIND_BYTES:
            return bytes(envelope.pack_bytes(env.shape, headers, payload))
        return bytes(envelope.pack(np.frombuffer(payload, "<f4").reshape(env.shape), headers))

    payload = bytes(env.payload)
    if env.kind == envelope.KIND_BYTES:
        out.append(Scope("payload", payload, container_regions(payload), lambda b: rewrap(env.headers, b)))
    for k, (sid, h) in enumerate(env.headers):
        fn = STAGE_REGIONS.get(sid)

        def wrap(b: bytes, k: int = k) -> bytes:
            hs = list(env.headers)
            hs[k] = (hs[k][0], bytes(b))
            return rewrap(hs, payload)

        out.append(Scope(f"hdr:{sid}", bytes(h), fn(bytes(h)) if fn else [], wrap))
    return out


def make_mutant(scopes: Sequence[Scope], seed: int, i: int) -> tuple[bytes, str, str]:
    """Case ``i`` of ``seed``: ``(mutant chunk, kind, description)``; ``kind`` is ``<scope>/<operation or field>``."""
    rng = random.Random(seed * 1_000_003 + i)
    weights = [3 if s.name in ("container", "payload") else 1 for s in scopes]
    sc = rng.choices(list(scopes), weights)[0]
    if sc.groups and rng.random() < 0.6:
        group = rng.choice(sc.groups)
        b = bytearray(sc.buf)
        desc = mutate_region(b, sc.regions, group, rng)
        return sc.wrap(bytes(b)), f"{sc.name}/{group}", desc
    mutant, op, desc = generic_mutation(sc.buf, rng)
    return sc.wrap(mutant), f"{sc.name}/{op}", desc


# ============================================================================================== targets
@dataclass
class Target:
    """A decode target: a spec (JSON-able) and, once loaded, the decode function and the original chunk."""

    spec: dict[str, Any]


def _conftest() -> Any:
    """The tests' synthetic lattice and chunk (``tests/conftest.py`` of a source checkout)."""
    try:
        import conftest
    except ImportError:
        tests = Path(__file__).resolve().parents[3] / "tests"
        if not (tests / "conftest.py").is_file():
            raise SystemExit("--synthetic needs a source checkout (it reuses tests/conftest.py)") from None
        sys.path.insert(0, str(tests))
        import conftest
    return conftest


def make_synthetic(work: str | os.PathLike[str], chain: bool = False) -> dict[str, Any]:
    """Write the synthetic test chunk with the lattice codec alone (``chain`` False) or the chain ``[field_terms,
    static_ref, icon_lattice_bytes]`` into a zarr store under ``work``; returns the target spec. The encode caches its
    lattice and structures in ``work/cache`` (the spec's ``cache``, which the workers use); the calling process keeps
    its own cache setting."""
    from ..icon_lattice import sidecar as sc

    work = Path(work)
    prev = os.environ.get(sc.CACHE_ENV)
    os.environ[sc.CACHE_ENV] = str(work / "cache")
    try:
        return _write_synthetic(work, chain)
    finally:
        if prev is None:
            os.environ.pop(sc.CACHE_ENV, None)
        else:
            os.environ[sc.CACHE_ENV] = prev


def _write_synthetic(work: Path, chain: bool) -> dict[str, Any]:
    import zarr

    from .. import chain as chain_mod
    from ..icon_lattice import register_store

    cf = _conftest()
    work.mkdir(parents=True, exist_ok=True)
    lat = cf.make_lattice()
    layout, chunk = lat[0], cf.make_chunk(lat)
    store = work / ("chain.zarr" if chain else "plain.zarr")
    if store.exists():
        import shutil

        shutil.rmtree(store)
    g = zarr.open_group(str(store), mode="w")
    register_store(store)
    data = chunk[None, :, None, :]
    if chain:
        from ..physics import expand

        entries = expand(cf.NAMES, None, cf.make_config(layout))
        codecs = [chain_mod.codec_class(e["name"])(**e["configuration"]) for e in entries]
        kw = {"compressors": codecs}
    else:
        from ..icon_lattice.zarr3 import IconLatticeCodec

        kw = {"serializer": IconLatticeCodec(**cf.make_config(layout)), "compressors": None}
    a = g.create_array("data", shape=data.shape, chunks=data.shape, dtype="float32", **kw)
    a[:] = data
    return {
        "store": str(store),
        "config": None,
        "chunk": str(store / "data" / "c" / "0" / "0" / "0" / "0"),
        "cache": str(work / "cache"),
    }


def load_decoder(spec: dict[str, Any]) -> tuple[Callable[[bytes], np.ndarray], bytes, bool]:
    """``(decode(chunk bytes) -> array, original chunk, is_chain)`` of a spec. A spec with ``selftest`` set decodes
    toy chunks (used to test the pool)."""
    if spec.get("selftest"):
        return _selftest_decode, b"\x00" * 64, False
    from ..cli import load_config
    from ..icon_lattice import LatticeCodec, container, register_store
    from ..icon_lattice import sidecar as sc

    if spec.get("cache"):
        os.environ[sc.CACHE_ENV] = spec["cache"]
    store = spec.get("store")
    if store:
        register_store(store)
    cfg = load_config(Path(spec["config"] or store))
    original = Path(spec["chunk"]).read_bytes()
    if isinstance(cfg, list):
        from .. import chain
        from ..icon_lattice import store as store_mod

        if not store:
            raise SystemExit("a chain needs --config STORE (the array's chunk shape comes from its metadata)")
        shape = tuple(store_mod.array_meta(store)["chunks"])

        def decode_chain(buf: bytes) -> np.ndarray:
            return chain.decode_chunk(cfg, np.frombuffer(buf, np.uint8), shape)

        return decode_chain, original, True
    from ..cli import lattice_config

    codec = LatticeCodec.from_config(lattice_config(cfg))
    expected = len(container.parse(original).modes) * codec.n_cells  # what zarr's chunk spec would hold

    def decode_plain(buf: bytes) -> np.ndarray:
        # decoding into a buffer of the expected size: the adapter allocates n_fields * n_cells from the header
        out = np.empty(expected, np.float32)
        codec.decode(np.frombuffer(buf, np.uint8), out=out)
        return out

    return decode_plain, original, False


def _selftest_decode(buf: bytes) -> np.ndarray:
    op = buf[0] if buf else 0
    if op == 1:
        os.abort()
    if op == 2:
        time.sleep(3600)
    if op == 3:
        raise ValueError("toy error")
    if op == 4:
        raise IndexError("toy error")
    if op == 5:
        sys.stderr.write("==1==ERROR: AddressSanitizer: toy report\n")
        sys.stderr.flush()
    return np.frombuffer(bytes(buf[:64]).ljust(64, b"\0"), np.uint8).astype(np.float32) * (2 if op == 6 else 1)


# ============================================================================================== workers
def worker_main(spec_path: str | None = None) -> None:
    """Entry of a worker subprocess: loads the spec, then answers length-prefixed chunk frames on stdin with one
    ``R <json>`` line each on the real stdout (fd 1 is redirected to stderr so that stray output cannot corrupt the
    protocol)."""
    try:  # a crash is the finding; its core dump in the working directory is not
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, ValueError, OSError):
        pass
    spec = json.loads(Path(spec_path or sys.argv[1]).read_text())
    proto = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    decode, original, _ = load_decoder(spec)
    ref = hashlib.sha256(np.asarray(decode(original)).tobytes()).hexdigest()
    proto.write(b"READY " + ref.encode() + b"\n")
    stdin = sys.stdin.buffer
    while True:
        head = stdin.read(8)
        if len(head) < 8:
            return
        n = struct.unpack("<Q", head)[0]
        data = stdin.read(n)
        t0 = time.perf_counter()
        try:
            out = decode(data)
            res = {"ok": True, "same": hashlib.sha256(np.asarray(out).tobytes()).hexdigest() == ref}
        except BaseException as e:  # noqa: BLE001 - every outcome of a decode is reported
            res = {
                "ok": False,
                "exc": type(e).__name__,
                "msg": str(e)[:300],
                "own": isinstance(e, ValueError) or type(e).__module__.startswith("anemoi_datasets_codecs"),
            }
        res["ms"] = round(1000 * (time.perf_counter() - t0), 2)
        proto.write(b"R " + json.dumps(res).encode() + b"\n")


class Worker:
    """A worker subprocess with its stderr in a file; ``submit`` sends a chunk, ``poll`` collects the answer."""

    def __init__(self, spec_path: Path, errfile: Path, startup_timeout: float):
        self.errfile = errfile
        self._err = open(errfile, "wb")
        code = "from anemoi_datasets_codecs.lab.fuzz import worker_main; worker_main()"
        self.proc = subprocess.Popen(
            [sys.executable, "-c", code, str(spec_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._err,
            env=os.environ.copy(),
        )
        self.fd = self.proc.stdout.fileno()
        os.set_blocking(self.fd, False)
        self.buf = b""
        self.ready = False
        self.reference: str | None = None
        self.task: dict[str, Any] | None = None
        self.deadline = time.monotonic() + startup_timeout
        self.err_pos = 0

    def submit(self, task: dict[str, Any], timeout: float) -> bool:
        self.task = task
        self.err_pos = self.errfile.stat().st_size
        self.deadline = time.monotonic() + timeout
        try:
            self.proc.stdin.write(struct.pack("<Q", len(task["data"])) + task["data"])
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass  # the worker is dead; the next poll sees it
        return True

    def lines(self) -> list[bytes]:
        try:
            chunk = os.read(self.fd, 1 << 16)
        except BlockingIOError:
            return []
        if not chunk:
            self.buf += b"\0EOF"
            return []
        self.buf += chunk
        *done, self.buf = self.buf.split(b"\n")
        return done

    def stderr_since(self, limit: int = 20000) -> bytes:
        self._err.flush()
        with open(self.errfile, "rb") as f:
            f.seek(self.err_pos)
            data = f.read()
        return data if len(data) <= limit else data[: limit // 2] + b"\n[...]\n" + data[-limit // 2 :]

    def kill(self) -> int | None:
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            rc = self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            rc = None
        for f in (self.proc.stdin, self.proc.stdout, self._err):
            try:
                f.close()
            except OSError:
                pass
        return rc


def _describe_exit(rc: int | None) -> str:
    if rc is None:
        return "unknown exit"
    if rc < 0:
        try:
            return f"killed by signal {-rc} ({signal.Signals(-rc).name})"
        except ValueError:
            return f"killed by signal {-rc}"
    return f"exit code {rc}"


@dataclass
class Campaign:
    """Result of a campaign: counts by kind and outcome, the saved reproducers and the exceptions seen."""

    iterations: int = 0
    counts: Counter = field(default_factory=Counter)  # (kind, outcome) -> n
    saved: list[str] = field(default_factory=list)
    exceptions: Counter = field(default_factory=Counter)  # (outcome, exception type, message prefix) -> n
    seconds: float = 0.0

    def totals(self) -> Counter:
        t: Counter = Counter()
        for (_, o), n in self.counts.items():
            t[o] += n
        return t


def run_campaign(
    spec: dict[str, Any],
    mutants: Callable[[int], tuple[bytes, str, str]],
    iterations: int,
    out: str | os.PathLike[str],
    timeout: float = 10.0,
    jobs: int = 4,
    startup_timeout: float = 300.0,
    base: bytes | None = None,
    log: Callable[[str], None] = lambda s: None,
) -> Campaign:
    """Decode ``mutants(0 .. iterations-1)`` in ``jobs`` worker subprocesses; classify and save the non-clean cases
    under ``out``. A mutant equal to ``base`` (a mutation that wrote the value already there) is counted as
    ``unchanged`` and not decoded."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    wdir = out / "workers"
    wdir.mkdir(exist_ok=True)
    spec_path = wdir / "spec.json"
    spec_path.write_text(json.dumps(spec))
    camp = Campaign(iterations=iterations)
    saved_n: Counter = Counter()
    tasks: list[dict[str, Any]] = []  # retries (fresh worker) first
    next_i = 0
    done = 0
    t0 = time.monotonic()
    sel = selectors.DefaultSelector()
    workers: list[Worker | None] = [None] * jobs
    reference: list[str | None] = [None]
    serial = [0]

    def spawn(k: int) -> None:
        serial[0] += 1
        w = Worker(spec_path, wdir / f"w{k}-{serial[0]}.err", startup_timeout)
        workers[k] = w
        sel.register(w.fd, selectors.EVENT_READ, k)

    def retire(k: int) -> int | None:
        w = workers[k]
        try:
            sel.unregister(w.fd)
        except (KeyError, ValueError):
            pass
        rc = w.kill()
        workers[k] = None
        return rc

    def record(task: dict[str, Any], outcome: str, extra: dict[str, Any]) -> None:
        nonlocal done
        done += 1
        camp.counts[(task["kind"], outcome)] += 1
        if "exc" in extra:
            camp.exceptions[(outcome, extra["exc"], extra.get("msg", "")[:80])] += 1
        if outcome in SAVED:
            key = outcome if outcome in SAVE_CAP else outcome.split("_")[0]
            ckey = (outcome, extra.get("exc", "")) if outcome == "error_other" else outcome
            if saved_n[ckey] < SAVE_CAP.get(key, 50):
                saved_n[ckey] += 1
                d = out / outcome
                d.mkdir(exist_ok=True)
                stem = d / f"case-{task['i']:06d}"
                stem.with_suffix(".bin").write_bytes(task["data"])
                meta = {"case": task["i"], "seed": spec.get("seed"), "kind": task["kind"], "mutation": task["desc"]}
                meta.update(outcome=outcome, **extra)
                stem.with_suffix(".json").write_text(json.dumps(meta, indent=1))
                camp.saved.append(str(stem.with_suffix(".bin")))
        if outcome in ("crash", "hang", "sanitizer") or done % max(1, iterations // 20) == 0:
            log(f"[{done}/{iterations}] case {task['i']} {task['kind']}: {outcome} {extra.get('detail', '')}")

    def finish_task(k: int, task: dict[str, Any], res: dict[str, Any] | None, died: bool, hung: bool) -> None:
        w = workers[k]
        err = w.stderr_since() if w else b""
        if died or hung:
            rc = retire(k)
            detail = "timeout" if hung else _describe_exit(rc)
            kind = "hang" if hung else "crash"
            if not task["retry"]:  # confirm in a fresh worker
                tasks.insert(0, {**task, "retry": True, "first": (kind, detail, err.decode("utf8", "replace"))})
                return
            record(task, kind, {"detail": detail, "first": task["first"][1], "stderr": err.decode("utf8", "replace")})
            return
        extra: dict[str, Any] = {"ms": res["ms"]}
        if task["retry"]:
            # the fresh worker survived: the crash / hang depended on the state left by earlier mutants
            kind, detail, first_err = task["first"]
            extra.update(detail=detail, stderr=first_err, fresh_result=res)
            record(task, f"{kind}_unconfirmed", extra)
        elif any(m in err for m in SANITIZER_MARKERS):
            extra["stderr"] = err.decode("utf8", "replace")
            extra.update({k2: res[k2] for k2 in ("exc", "msg") if k2 in res})
            record(task, "sanitizer", extra)
        elif not res["ok"]:
            extra.update(exc=res["exc"], msg=res["msg"])
            record(task, "clean_error" if res["own"] else "error_other", extra)
        else:
            record(task, "accepted_same" if res["same"] else "accepted_diff", extra)

    try:
        for k in range(jobs):
            spawn(k)
        while done < iterations:
            for k in range(jobs):
                w = workers[k]
                if w is None:
                    spawn(k)
                    w = workers[k]
                if w.ready and w.task is None:
                    if tasks:
                        task = tasks.pop(0)
                    else:
                        task = None
                        while task is None and next_i < iterations:
                            data, kind, desc = mutants(next_i)
                            task = {"i": next_i, "data": data, "kind": kind, "desc": desc, "retry": False}
                            next_i += 1
                            if base is not None and data == base:
                                record(task, "unchanged", {})
                                task = None
                        if task is None:
                            continue
                    w.submit(task, timeout)
            for key, _ in sel.select(timeout=0.05):
                k = key.data
                w = workers[k]
                if w is None:
                    continue
                for line in w.lines():
                    if line.startswith(b"READY "):
                        w.ready = True
                        w.reference = line[6:].decode()
                        if reference[0] is None:
                            reference[0] = w.reference
                    elif line.startswith(b"R ") and w.task is not None:
                        task, w.task = w.task, None
                        finish_task(k, task, json.loads(line[2:]), False, False)
                if w.buf.endswith(b"\0EOF") and workers[k] is w:
                    task, w.task = w.task, None
                    if task is None:
                        if not w.ready:
                            raise RuntimeError(f"worker died at startup:\n{w.stderr_since().decode('utf8', 'replace')}")
                        retire(k)
                    else:
                        finish_task(k, task, None, True, False)
            now = time.monotonic()
            for k, w in enumerate(workers):
                if w is None or now <= w.deadline or (w.ready and w.task is None):
                    continue
                if not w.ready:
                    raise RuntimeError("worker did not start in time:\n" + w.stderr_since().decode("utf8", "replace"))
                task, w.task = w.task, None
                finish_task(k, task, None, False, True)
    finally:
        for k in range(jobs):
            if workers[k] is not None:
                retire(k)
    camp.seconds = time.monotonic() - t0
    return camp


def summary_table(camp: Campaign) -> str:
    """Counts by mutation kind (rows) and outcome (columns), and the totals."""
    present = [o for o in OUTCOMES if any(o == k[1] for k in camp.counts)]
    kinds = sorted({k[0] for k in camp.counts})
    w = max([len(k) for k in kinds] + [10])
    lines = ["kind".ljust(w) + "".join(f"{o:>{max(len(o), 7) + 1}}" for o in present) + "   total"]
    for kind in kinds:
        row = [camp.counts.get((kind, o), 0) for o in present]
        lines.append(
            kind.ljust(w) + "".join(f"{n:>{max(len(o), 7) + 1}}" for n, o in zip(row, present)) + f"{sum(row):>8}"
        )
    tot = camp.totals()
    lines.append(
        "TOTAL".ljust(w) + "".join(f"{tot[o]:>{max(len(o), 7) + 1}}" for o in present) + f"{sum(tot.values()):>8}"
    )
    return "\n".join(lines)


# ============================================================================================== parsers
def fuzz_parsers(iterations: int, seed: int = 0, per_case_seconds: int = 5) -> dict[str, Any]:
    """Fuzz the header parsers in-process: ``envelope.parse``, the ``static_ref`` and ``field_terms`` header parsers
    and ``container.parse`` / ``lattice_header`` / ``inspect_chunk``, on mutants of valid headers. A parser must
    return or raise ``ValueError``; any other exception, an inconsistent result or a hang is a finding. Returns
    ``{parser: {"cases", "accepted", "rejected", "findings": [{...}]}}``."""
    from ..field_terms import header as ft
    from ..static_ref import header as sr

    rng = random.Random(seed)
    sha = bytes(range(32)).hex()
    slots = [
        sr.Slot(sr.REFERENCE, sha),
        sr.Slot(sr.INLINE, "ab" * 32, data=bytes(range(40))),
        sr.Slot(sr.REPEAT, sha, source=0),
        sr.Slot(sr.REFERENCE, "cd" * 32),
    ]
    sr_hdr = sr.pack(2, slots)
    terms = [(3, (0, 1)), (5, (2,))]
    ft_hdr = ft.pack(terms, [[ft.Record(False, np.array([1, 2, 3], "<f4")), ft.Record(True, np.zeros(2, "<f4"))]] * 3)
    shape = (1, 3, 1, 5)
    env = bytes(envelope.pack_bytes(shape, [("static_ref", sr_hdr), ("field_terms", ft_hdr)], bytes(range(64))))
    env_arr = bytes(envelope.pack(np.arange(15, dtype="<f4").reshape(shape), [("a", b"xyz")]))
    n = 5
    payloads = [bytes(range(i, i + 8 * (i + 1))) for i in range(n)]
    ch = container.pack([0, 1, 2, 3, 4], [0] * n, payloads, np.array([1, 2], "<i8"), bytes(16))
    lattice = bytes([2, 2, 3, 2, 1, 4, 12, 0]) + bytes(48) + bytes(64)

    def check_env(b: bytes) -> None:
        e = envelope.parse(b)
        if e is not None:
            if e.header_bytes + e.payload.nbytes != len(b):
                raise AssertionError("envelope.parse: inconsistent lengths")
            if e.kind == envelope.KIND_ARRAY:
                e.array()
            envelope.unpack(b, shape)

    def check_chunk(b: bytes) -> None:
        c = container.parse(b)
        container.header_size(b[: container.PREFIX])
        if sum(len(p) for p in c.payloads) + c.header_bytes != len(b):
            raise AssertionError("container.parse: inconsistent lengths")
        container.inspect_chunk(b)

    cases = {
        "envelope.parse": (env, check_env),
        "envelope.parse (array payload)": (env_arr, check_env),
        "static_ref.header.parse": (sr_hdr, sr.parse),
        "field_terms.header.parse": (ft_hdr, lambda b: ft.parse(b, terms, 3)),
        "container.parse": (ch, check_chunk),
        "container.lattice_header": (lattice, container.lattice_header),
    }
    results: dict[str, Any] = {}

    def on_alarm(*_: Any) -> None:
        raise TimeoutError("parser hang")

    old = signal.signal(signal.SIGALRM, on_alarm)
    try:
        for name, (base, fn) in cases.items():
            r = {"cases": 0, "accepted": 0, "rejected": 0, "findings": []}
            reg = []
            if name.startswith("envelope"):
                reg = envelope_regions(base)
            elif name.startswith("static_ref"):
                reg = static_ref_regions(base)
            elif name.startswith("field_terms"):
                reg = field_terms_regions(base)
            elif name == "container.parse":
                reg = [
                    ("container.n_fields", 8, 12, 4, False),
                    ("container.n_dates", 12, 16, 4, False),
                    ("container.header_bytes", 16, 20, 4, False),
                    ("container.version", 4, 6, 2, False),
                    ("record.nbytes", 40 + 16, 40 + 24, 8, False),
                    ("record.mode", 56, 57, 1, False),
                ]
            groups = sorted({x[0] for x in reg})
            for _ in range(iterations):
                if groups and rng.random() < 0.5:
                    b = bytearray(base)
                    mutate_region(b, reg, rng.choice(groups), rng)
                    if rng.random() < 0.3:
                        b = bytearray(generic_mutation(bytes(b), rng)[0])
                    data = bytes(b)
                else:
                    data = generic_mutation(base, rng)[0]
                r["cases"] += 1
                signal.alarm(per_case_seconds)
                try:
                    fn(data)
                    r["accepted"] += 1
                except ValueError:
                    r["rejected"] += 1
                except BaseException as e:  # noqa: BLE001
                    if len(r["findings"]) < 10:
                        r["findings"].append({"exc": f"{type(e).__name__}: {e}"[:200], "data": data.hex()[:4000]})
                finally:
                    signal.alarm(0)
            results[name] = r
    finally:
        signal.signal(signal.SIGALRM, old)
    return results


# ============================================================================================== command line
def _fuzz(a: argparse.Namespace) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.parsers:
        res = fuzz_parsers(a.iterations, a.seed)
        bad = 0
        for name, r in res.items():
            print(
                f"{name:34s} cases {r['cases']:8d}  accepted {r['accepted']:8d}  rejected {r['rejected']:8d}  "
                f"findings {len(r['findings'])}"
            )
            bad += len(r["findings"])
        (out / "parsers.json").write_text(json.dumps(res, indent=1))
        return 1 if bad else 0
    if a.synthetic:
        spec = make_synthetic(out / "base", chain=a.chain) if not (a.reuse and (out / "base").exists()) else None
        if spec is None:
            store = out / "base" / ("chain.zarr" if a.chain else "plain.zarr")
            spec = {
                "store": str(store),
                "config": None,
                "chunk": str(store / "data/c/0/0/0/0"),
                "cache": str(out / "base" / "cache"),
            }
    else:
        if not a.chunk or not a.config:
            raise SystemExit("give --synthetic, or CHUNK and --config (a store or a configuration file)")
        store = Path(a.config) if Path(a.config).is_dir() else None
        spec = {"store": str(store) if store else None, "config": None if store else a.config, "chunk": a.chunk}
    spec["seed"] = a.seed
    base = Path(spec["chunk"]).read_bytes()
    chain = envelope.parse(base) is not None
    scopes = scopes_of(base, chain)
    print(f"base chunk {len(base)} bytes, {'chain' if chain else 'plain'}, scopes {[s.name for s in scopes]}")
    if a.iterations <= 0:
        return 0
    os.environ.setdefault("ANEMOI_DATASETS_CODECS_THREADS", "1")
    camp = run_campaign(
        spec,
        lambda i: make_mutant(scopes, a.seed, i),
        a.iterations,
        out,
        base=base,
        timeout=a.timeout,
        jobs=a.jobs,
        log=lambda s: print(s, flush=True),
    )
    table = summary_table(camp)
    print(table)
    tot = camp.totals()
    print("\nexceptions:")
    for (o, exc, msg), n in camp.exceptions.most_common(25):
        print(f"  {n:7d} {o:12s} {exc}: {msg}")
    print(f"\n{camp.iterations} iterations in {camp.seconds:.1f} s; reproducers saved: {len(camp.saved)} in {out}")
    (out / "summary.txt").write_text(table + "\n")
    (out / "summary.json").write_text(
        json.dumps(
            {
                "iterations": camp.iterations,
                "seed": a.seed,
                "totals": dict(tot),
                "counts": {f"{k}|{o}": n for (k, o), n in sorted(camp.counts.items())},
                "exceptions": [[o, e, m, n] for (o, e, m), n in camp.exceptions.most_common()],
                "saved": camp.saved,
            },
            indent=1,
        )
    )
    return 1 if any(tot[o] for o in ("crash", "hang", "sanitizer", "crash_unconfirmed", "hang_unconfirmed")) else 0


def _replay(a: argparse.Namespace) -> int:
    """Decode saved mutants (``--replay FILE...``) in one fresh worker each and print the outcome."""
    spec_file = Path(a.replay[0]).resolve().parents[1] / "workers" / "spec.json"
    if not spec_file.is_file():
        raise SystemExit(f"{spec_file} not found: replay a reproducer saved under the output directory of a run")
    rc = 0
    for f in a.replay:
        w = Worker(spec_file, Path(f).with_suffix(".replay.err"), 300)
        while not w.ready:
            for line in w.lines():
                w.ready = w.ready or line.startswith(b"READY ")
            if w.buf.endswith(b"\0EOF"):
                print(f"{f}: worker died at startup")
                rc = 1
                break
            time.sleep(0.01)
        if not w.ready:
            continue
        w.submit({"data": Path(f).read_bytes()}, a.timeout)
        res, t0 = None, time.monotonic()
        while res is None and time.monotonic() - t0 < a.timeout and not w.buf.endswith(b"\0EOF"):
            for line in w.lines():
                if line.startswith(b"R "):
                    res = json.loads(line[2:])
            time.sleep(0.01)
        err = w.stderr_since(2000).decode("utf8", "replace")
        code = w.kill()
        print(f"{f}: {res if res else 'no answer (' + _describe_exit(code) + ')'}")
        if err.strip():
            print(err)
        rc |= res is None
    return int(rc)


def add_parsers(sub: argparse._SubParsersAction) -> None:
    """Register ``lab fuzz`` under ``lab``."""
    p = sub.add_parser("fuzz", help="mutation fuzzer of the decode path and of the header parsers")
    p.add_argument("chunk", nargs="?", help="a valid chunk file (omit with --synthetic)")
    p.add_argument("--config", help="a store, or a codec configuration file, that decodes CHUNK")
    p.add_argument("--synthetic", action="store_true", help="the synthetic test chunk (tests/conftest.py)")
    p.add_argument(
        "--chain", action="store_true", help="with --synthetic: the chain [field_terms, static_ref, lattice]"
    )
    p.add_argument("--iterations", "-n", type=int, default=1000, help="mutants to decode (with --parsers: per parser)")
    p.add_argument("--seed", type=int, default=0, help="seed of the mutations; case i of a seed is reproducible")
    p.add_argument("--out", required=True, help="output directory (base chunk, workers, reproducers, summary)")
    p.add_argument("--timeout", type=float, default=10.0, help="seconds per decode before it counts as a hang")
    p.add_argument("--jobs", "-j", type=int, default=4, help="worker subprocesses")
    p.add_argument("--reuse", action="store_true", help="with --synthetic: reuse OUT/base when it exists")
    p.add_argument("--parsers", action="store_true", help="fuzz the header parsers in-process instead")
    p.add_argument("--replay", nargs="+", help="decode saved reproducers (OUT/<outcome>/case-N.bin) and print outcomes")
    p.set_defaults(fn=lambda a: _replay(a) if a.replay else _fuzz(a))
