"""Chunk container of the lattice codec, format version 3 (little endian; frozen, see docs/codecs/icon_lattice.md).

::

    0   4s  magic b"LATC"
    4   u16 format version (3)
    6   u16 flags (0)
    8   u32 n_fields                  (dates x variables x ensembles, C order)
    12  u32 n_dates
    16  u32 header_bytes              (offset of the first payload, multiple of 8)
    20  u32 reserved
    24  16s structure sha (first 16 bytes of the structure's sha256)
    40  i64 dates[n_dates]            (valid date, seconds since 1970-01-01; INT64_MIN = unknown)
    ..  n_fields x 16 B records: u8 mode, u8 fallback, u16 0, u32 0, u64 payload bytes
    ..  zero padding to header_bytes, then the payloads (each a multiple of 4 bytes)

Modes: 0 LATTICE (lossy, see encoder.py / csrc/decode.c), 1 LOSSLESS (the configuration's ``lossless`` codec on the
native order), 2 CONSTANT (f32), 3 STATIC_REF (sha256 of the field's float32 bytes [+ the lossless-compressed field
when statics are stored inline]), 4 RECIPE (analytic forcing, see csrc/decode.c). ``fallback``: why a lossy field was
stored losslessly (0 it was not, 1 the lossy coding failed or missed the bound, 2 lossless was smaller).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from numcodecs.compat import ensure_contiguous_ndarray

MAGIC = b"LATC"
FORMAT_VERSION = 3  # the first published version (1 and 2 were development formats); the only one decoded
MODE_LATTICE = 0
MODE_LOSSLESS = 1
MODE_CONSTANT = 2
MODE_STATIC_REF = 3
MODE_RECIPE = 4
MODE_LABELS = {
    MODE_LATTICE: "lattice",
    MODE_LOSSLESS: "lossless",
    MODE_CONSTANT: "constant",
    MODE_STATIC_REF: "static_ref",
    MODE_RECIPE: "recipe",
}
EXACT_MODES = frozenset({MODE_LOSSLESS, MODE_CONSTANT, MODE_STATIC_REF})  # decoded bit-identical to the data
FALLBACK_NONE = 0
FALLBACK_FAILED = 1  # the lossy coding failed or missed the bound in verification
FALLBACK_SMALLER = 2  # the lossless payload was smaller
NO_DATE = -(2**63)
LATTICE_HEADER = 56  # bytes of the fixed header of a LATTICE payload
MASK_BLOCK = 16  # bytes of a mask-term block before its features


class FieldRecord(NamedTuple):
    """Record of one field: storage mode, payload size, fallback reason (``FALLBACK_*``)."""

    mode: int
    nbytes: int
    fallback: int = FALLBACK_NONE


class LatticeHeader(NamedTuple):
    """The fixed header of a LATTICE payload (docs/codecs/icon_lattice.md "LATTICE payload"; written by
    ``encoder.py``). ``cross``: the field indices of the cross-term sources (``n_cross`` of them); ``mask_src``: the
    field index of the mask source when the payload has a mask term (flag 4), else None."""

    vt: int
    et: int
    n_steps: int
    n_tables: int
    n_cross: int
    flags: int
    prob_bits: int
    recon_bits: int
    offset: float
    step: float
    lo: float
    hi: float
    n_esc: int
    rans_bytes: int
    tables_bytes: int
    steps_bytes: int
    cross: tuple[int, ...]
    shifts: tuple[int, ...]
    mask_src: int | None


_LATTICE = struct.Struct("<8B4f4I4H4B4x")


def lattice_header(payload: Any) -> LatticeHeader:
    """Parse the header of a LATTICE payload (and the mask source of its mask block). Raises ``ValueError`` when the
    payload is shorter than the header."""
    p = memoryview(payload).cast("B")
    if len(p) < LATTICE_HEADER:
        raise ValueError(f"LATTICE payload of {len(p)} bytes is shorter than its header")
    v = _LATTICE.unpack_from(p)
    n = v[4]
    has_mask = v[5] & 4 and len(p) >= LATTICE_HEADER + MASK_BLOCK
    mask = int.from_bytes(p[LATTICE_HEADER : LATTICE_HEADER + 2], "little") if has_mask else None
    return LatticeHeader(*v[:16], tuple(v[16 : 16 + n]), tuple(v[20 : 20 + n]), mask)


def to_seconds(dates: Any, n: int) -> np.ndarray:
    """The container's dates: int64 seconds since 1970 for ``n`` dates (numpy datetime64 or ISO strings; one date for
    all, or one per date; None -> ``NO_DATE``, unknown)."""
    if dates is None:
        return np.full(n, NO_DATE, np.int64)
    d = np.asarray(dates)
    if np.issubdtype(d.dtype, np.datetime64):
        s = d.astype("datetime64[s]").astype(np.int64)
    else:
        s = np.array([np.datetime64(str(x), "s").astype(np.int64) for x in np.atleast_1d(d)], np.int64)
    s = np.atleast_1d(s)
    if len(s) == 1 and n > 1:
        s = np.repeat(s, n)
    if len(s) != n:
        raise ValueError(f"{len(s)} dates for a chunk of {n} dates")
    return s


_HEAD = struct.Struct("<4sHHIIII16s")
_RECORD = np.dtype([("mode", "u1"), ("fallback", "u1"), ("r0", "<u2"), ("r1", "<u4"), ("nbytes", "<u8")])


PREFIX = 40  # bytes of the chunk header before the dates: enough for ``header_size``


@dataclass
class Chunk:
    """Parsed container: modes, fallbacks, payload views (empty when parsed without payloads), dates (int64 s),
    structure sha, payload offsets (``n_fields + 1``, from the chunk start)."""

    modes: np.ndarray
    fallbacks: np.ndarray
    payloads: list[memoryview]
    dates: np.ndarray
    structure_sha: bytes
    header_bytes: int
    buffer: memoryview
    offsets: np.ndarray


def pack(modes: list[int], fallbacks: list[int], payloads: list[bytes], dates: np.ndarray, sha16: bytes) -> bytes:
    """Assemble a chunk (payloads padded to a multiple of 4 bytes)."""
    n = len(modes)
    dates = np.asarray(dates, "<i8")
    payloads = [p + b"\0" * (-len(p) % 4) for p in payloads]
    rec = np.zeros(n, _RECORD)
    rec["mode"] = modes
    rec["fallback"] = fallbacks
    rec["nbytes"] = [len(p) for p in payloads]
    hb = _HEAD.size + 8 * len(dates) + rec.nbytes
    hb += -hb % 8
    head = _HEAD.pack(MAGIC, FORMAT_VERSION, 0, n, len(dates), hb, 0, sha16) + dates.tobytes() + rec.tobytes()
    head += b"\0" * (hb - len(head))
    return b"".join([head, *payloads])


def _head(mv: Any) -> tuple:
    if len(mv) < _HEAD.size:
        raise ValueError(f"lattice codec chunk too short ({len(mv)} bytes)")
    head = _HEAD.unpack_from(mv)
    magic, version = head[:2]
    if magic != MAGIC:
        raise ValueError(f"not a lattice codec chunk (magic {magic!r})")
    if version != FORMAT_VERSION:
        raise ValueError(f"lattice codec container version {version} is not supported ({FORMAT_VERSION})")
    return head


def header_size(prefix: Any) -> int:
    """``header_bytes`` of a chunk from its first ``PREFIX`` bytes (the header, record table included, to read for
    ``parse(..., payloads=False)``). Raises ``ValueError`` when it is not a lattice codec chunk."""
    return int(_head(memoryview(ensure_contiguous_ndarray(prefix)).cast("B"))[5])


def parse(buf: Any, payloads: bool = True) -> Chunk:
    """Parse the header and record table (no payload is decoded). Raises ``ValueError`` on a malformed chunk. With
    ``payloads=False``, ``buf`` may be the header alone (its first ``header_size`` bytes): ``Chunk.payloads`` is
    then empty and ``Chunk.offsets`` says where each payload is."""
    mv = memoryview(ensure_contiguous_ndarray(buf)).cast("B")
    _, _, _, n, n_dates, hb, _, sha = _head(mv)
    rec_off = _HEAD.size + 8 * n_dates
    if rec_off + n * _RECORD.itemsize > hb or hb > len(mv):
        raise ValueError("lattice codec chunk truncated in the header")
    dates = np.frombuffer(mv, "<i8", count=n_dates, offset=_HEAD.size).copy()
    rec = np.frombuffer(mv, _RECORD, count=n, offset=rec_off)
    sizes = rec["nbytes"].astype(np.int64)
    if payloads and hb + int(sizes.sum()) != len(mv):
        raise ValueError("lattice codec chunk size does not match its field table")
    if np.any(rec["mode"] > MODE_RECIPE):
        raise ValueError("lattice codec chunk has an unknown field mode")
    bounds = np.concatenate([[0], np.cumsum(sizes)]) + hb
    views = [mv[bounds[i] : bounds[i + 1]] for i in range(n)] if payloads else []
    return Chunk(rec["mode"].copy(), rec["fallback"].copy(), views, dates, bytes(sha), hb, mv, bounds)


def inspect_chunk(buf: Any) -> list[FieldRecord]:
    """One ``FieldRecord`` per field (C order; ``mode`` one of ``MODE_*``). ``nbytes`` includes the field's share of
    the chunk header, split pro rata to payload size, so that the records sum to the chunk size."""
    ch = parse(buf)
    sizes = np.array([len(p) for p in ch.payloads], np.int64)
    total = int(sizes.sum())
    share = (sizes * ch.header_bytes) // max(total, 1) if total else np.zeros_like(sizes)
    rest = ch.header_bytes - int(share.sum())
    if len(share):
        share[int(np.argmax(sizes))] += rest
    out = []
    for i, m in enumerate(ch.modes):
        out.append(FieldRecord(int(m), int(sizes[i] + share[i]), int(ch.fallbacks[i])))
    return out


def field_references(buf: Any) -> list[dict[str, Any]]:
    """Per field, what it references by field index: ``cross`` (the cross-term sources a LATTICE field uses, after
    the encoder dropped the terms that did not pay), ``mask`` (the mask source of a below-ground term), ``recipe``
    (recipe id and inputs of a RECIPE field). Reads the payload headers only."""
    ch = parse(buf)
    out: list[dict[str, Any]] = []
    for m, p in zip(ch.modes, ch.payloads, strict=True):
        d: dict[str, Any] = {}
        if m == MODE_LATTICE and len(p) >= LATTICE_HEADER:
            h = lattice_header(p)
            d["cross"] = list(h.cross)
            if h.mask_src is not None:
                d["mask"] = h.mask_src
        elif m == MODE_RECIPE and len(p) >= 64:
            src = [int.from_bytes(p[4 + 2 * j : 6 + 2 * j], "little") for j in range(4)]
            d["recipe"] = {"id": int(p[0]), "inputs": [x for x in src if x != 0xFFFF]}
        out.append(d)
    return out
