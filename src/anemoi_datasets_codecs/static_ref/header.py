"""The ``static_ref`` stage header, format version 1 (little endian; frozen, see docs/codecs/static_ref.md).

The header the stage pushes on the chain envelope (stage id ``static_ref``) says, for every static field it dropped
from the chunk, where its values are. One *slot* per ``(date, static field, member)`` of the chunk, in C order of
``x[:, static_fields, :, :]`` (the static fields in chunk order)::

    0   u8  header version (1)
    1   u8  0
    2   u16 0
    4   u32 n_static                (static fields per date and member)
    8   u32 n_slots                 (dates x n_static x members)
    12  u32 0
    16  n_slots records, each: u8 kind, u8 0, u16 0, u32 arg, 32 B sha256 of the slot's float32 little-endian bytes,
        then, for kind INLINE, ``arg`` bytes of data

Kinds: 0 REFERENCE (the sidecar file ``statics/<sha256>.f4.blosc``; ``arg`` 0), 1 INLINE (``arg`` bytes follow: the
numcodecs Blosc frame of the float32 values, the codec of the sidecar files), 2 REPEAT (the values of the earlier slot
``arg``, which has the same sha256).
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from typing import NamedTuple

__all__ = ["INLINE", "KIND_LABELS", "REFERENCE", "REPEAT", "VERSION", "Slot", "pack", "parse"]

VERSION = 1
REFERENCE = 0
INLINE = 1
REPEAT = 2
KIND_LABELS = {REFERENCE: "reference", INLINE: "inline", REPEAT: "repeat"}

_HEAD = struct.Struct("<BBHIII")  # version, 0, 0, n_static, n_slots, 0
_SLOT = struct.Struct("<BBHI32s")  # kind, 0, 0, arg, sha256


class Slot(NamedTuple):
    """One ``(date, static field, member)`` of the chunk: ``kind``, ``sha`` (hex), ``data`` (the Blosc frame, INLINE
    only, else empty) and ``source`` (the earlier slot, REPEAT only, else -1)."""

    kind: int
    sha: str
    data: bytes = b""
    source: int = -1


def pack(n_static: int, slots: Sequence[Slot]) -> bytes:
    """The header bytes of these slots."""
    parts = [_HEAD.pack(VERSION, 0, 0, n_static, len(slots), 0)]
    for k, s in enumerate(slots):
        sha = bytes.fromhex(s.sha)
        if len(sha) != 32:
            raise ValueError(f"slot {k}: sha256 of {len(sha)} bytes")
        if s.kind == REFERENCE:
            parts.append(_SLOT.pack(REFERENCE, 0, 0, 0, sha))
        elif s.kind == INLINE:
            if not s.data:
                raise ValueError(f"slot {k}: inline slot without data")
            parts.append(_SLOT.pack(INLINE, 0, 0, len(s.data), sha) + bytes(s.data))
        elif s.kind == REPEAT:
            if not 0 <= s.source < k or slots[s.source].sha != s.sha:
                raise ValueError(f"slot {k}: repeats slot {s.source}, which is not an earlier slot with its sha256")
            parts.append(_SLOT.pack(REPEAT, 0, 0, s.source, sha))
        else:
            raise ValueError(f"slot {k}: unknown kind {s.kind}")
    return b"".join(parts)


def parse(buf: bytes | memoryview) -> tuple[int, list[Slot]]:
    """``(n_static, slots)`` of a header. Raises ``ValueError`` when it is malformed or of another version."""
    b = memoryview(buf).cast("B")
    if len(b) < _HEAD.size:
        raise ValueError(f"static_ref header of {len(b)} bytes is too short")
    version, r0, r1, n_static, n_slots, r2 = _HEAD.unpack_from(b)
    if version != VERSION:
        raise ValueError(f"static_ref header version {version} is not supported (this is {VERSION})")
    if r0 or r1 or r2:
        raise ValueError("static_ref header: reserved fields are not zero")
    off = _HEAD.size
    slots: list[Slot] = []
    for k in range(n_slots):
        if off + _SLOT.size > len(b):
            raise ValueError(f"static_ref header truncated at slot {k}")
        kind, r3, r4, arg, sha_b = _SLOT.unpack_from(b, off)
        off += _SLOT.size
        sha = sha_b.hex()
        if r3 or r4:
            raise ValueError(f"static_ref header: reserved fields of slot {k} are not zero")
        if kind == REFERENCE:
            if arg:
                raise ValueError(f"static_ref header: reference slot {k} with an argument")
            slots.append(Slot(REFERENCE, sha))
        elif kind == INLINE:
            if not arg or off + arg > len(b):
                raise ValueError(f"static_ref header: inline slot {k} truncated")
            slots.append(Slot(INLINE, sha, bytes(b[off : off + arg])))
            off += arg
        elif kind == REPEAT:
            if arg >= k or slots[arg].sha != sha:
                raise ValueError(f"static_ref header: slot {k} repeats slot {arg}, not an earlier one with its sha256")
            slots.append(Slot(REPEAT, sha, source=arg))
        else:
            raise ValueError(f"static_ref header: slot {k} has unknown kind {kind}")
    if off != len(b):
        raise ValueError(f"static_ref header: {len(b) - off} bytes after the last slot")
    return n_static, slots
