"""The per-chunk header of the ``field_terms`` stage, format version 1 (little endian; frozen, see
docs/codecs/field_terms.md). It travels in the chain envelope under the stage id ``field_terms``::

    0   u8   header version (1)
    1   u8   0
    2   u16  n_terms T                    (the configured terms, in the order of their targets in ``variables``)
    4   u32  n_slices S                   (dates x ensemble members of the chunk)
    8   T term entries: u16 target index, u16 n_sources k, k x u16 source index (indices into ``variables``)
    ..  zero padding to a multiple of 4
    ..  S x T records, slice-major (date, then member, then term in entry order); each: u8 flags (bit 0: bypass),
        3 zero bytes, (1 + k) float32 coefficients c0, c1 .. ck (all zero when bypassed)

The term entries repeat the configuration so that a chunk read with another configuration fails clearly instead of
decoding wrongly.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

VERSION = 1
FLAG_BYPASS = 1

_HEAD = struct.Struct("<BBHI")
_U16 = np.dtype("<u2")
_F4 = np.dtype("<f4")

TermIndex = tuple[int, tuple[int, ...]]  # (target index, source indices)


@dataclass(frozen=True)
class Record:
    """One target in one (date, member) slice: bypassed (the target was left untouched) or the float32 coefficients
    ``c0, c1 .. ck`` of ``pred = c0 + c1 s1 + .. + ck sk``."""

    bypass: bool
    coefs: np.ndarray  # float32, 1 + k


def _entries_size(terms: Sequence[TermIndex]) -> int:
    n = sum(4 + 2 * len(s) for _, s in terms)
    return n + (-n % 4)


def size(terms: Sequence[TermIndex], n_slices: int) -> int:
    """Bytes of a header for ``terms`` and ``n_slices`` slices."""
    return _HEAD.size + _entries_size(terms) + n_slices * sum(4 + 4 * (1 + len(s)) for _, s in terms)


def pack(terms: Sequence[TermIndex], records: Sequence[Sequence[Record]]) -> bytes:
    """The header of a chunk: ``records[slice][term]``."""
    out = bytearray(_HEAD.pack(VERSION, 0, len(terms), len(records)))
    for t, srcs in terms:
        out += struct.pack(f"<HH{len(srcs)}H", t, len(srcs), *srcs)
    out += b"\0" * (-len(out) % 4)
    for row in records:
        if len(row) != len(terms):
            raise ValueError(f"field_terms header: {len(row)} records for {len(terms)} terms")
        for (_, srcs), rec in zip(terms, row, strict=True):
            c = np.asarray(rec.coefs, _F4)
            if c.shape != (1 + len(srcs),):
                raise ValueError(f"field_terms header: {c.shape} coefficients for {len(srcs)} sources")
            if rec.bypass:
                c = np.zeros_like(c)
            out += struct.pack("<B3x", FLAG_BYPASS if rec.bypass else 0) + c.tobytes()
    return bytes(out)


def _table(b: bytes) -> tuple[list[TermIndex], int, int]:
    """The stored term table, the slice count and the offset of the records."""
    if len(b) < _HEAD.size:
        raise ValueError(f"field_terms header: {len(b)} bytes is too short")
    version, zero, n_terms, n_sl = _HEAD.unpack_from(b, 0)
    if version != VERSION:
        raise ValueError(f"field_terms header version {version} is not supported (this reader knows {VERSION})")
    if zero:
        raise ValueError("field_terms header: nonzero reserved byte")
    pos = _HEAD.size
    stored: list[TermIndex] = []
    for _ in range(n_terms):
        if pos + 4 > len(b):
            raise ValueError("field_terms header: truncated term table")
        t, k = struct.unpack_from("<HH", b, pos)
        if pos + 4 + 2 * k > len(b):
            raise ValueError("field_terms header: truncated term table")
        stored.append((t, tuple(int(i) for i in np.frombuffer(b, _U16, k, pos + 4))))
        pos += 4 + 2 * k
    pad = -pos % 4
    if b[pos : pos + pad] != b"\0" * pad:
        raise ValueError("field_terms header: nonzero padding")
    return stored, n_sl, pos + pad


def parse(buf: bytes, terms: Sequence[TermIndex], n_slices: int) -> list[list[Record]]:
    """``records[slice][term]`` of a header written for ``terms`` and ``n_slices`` slices (raises ``ValueError`` on
    another version, another term table or slice count, a wrong length or a nonzero reserved field)."""
    b = bytes(buf)
    stored, n_sl, pos = _table(b)
    want = [(int(t), tuple(int(i) for i in s)) for t, s in terms]
    if stored != want:
        raise ValueError(
            f"field_terms header: the chunk was written with the terms {stored} (target, sources as variable "
            f"indices), the configuration has {want}"
        )
    if n_sl != n_slices:
        raise ValueError(f"field_terms header: {n_sl} slices, the chunk has {n_slices} (dates x members)")
    if len(b) != size(terms, n_slices):
        raise ValueError(f"field_terms header: {len(b)} bytes, expected {size(terms, n_slices)}")
    out: list[list[Record]] = []
    for _ in range(n_slices):
        row = []
        for _, srcs in want:
            flags, r0, r1, r2 = struct.unpack_from("<BBBB", b, pos)
            if flags & ~FLAG_BYPASS or r0 or r1 or r2:
                raise ValueError(f"field_terms header: unknown flags {flags:#x} or nonzero reserved bytes")
            c = np.frombuffer(b, _F4, 1 + len(srcs), pos + 4).copy()
            row.append(Record(bool(flags & FLAG_BYPASS), c))
            pos += 4 + 4 * (1 + len(srcs))
        out.append(row)
    return out


def read(buf: bytes) -> tuple[list[TermIndex], list[list[Record]]]:
    """The term table and ``records[slice][term]`` of a header as stored, without a configuration (for tools; the
    same checks as :func:`parse` but the comparison with a configuration)."""
    b = bytes(buf)
    stored, n_sl, _ = _table(b)
    return stored, parse(b, stored, n_sl)
