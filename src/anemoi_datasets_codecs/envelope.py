"""The in-band envelope of a codec chain, format version 1 (little endian; frozen, see docs/codecs/envelope.md).

Field stages (``zarr3.FieldStage``) are bytes-to-bytes codecs that change the chunk (drop fields, rewrite them) and
keep per-chunk metadata (a fitted coefficient, a sha). zarr gives a bytes-to-bytes codec no channel but the bytes, so
the stages wrap the chunk in an envelope: the shape of the field array it holds, a stack of tagged headers (one per
stage, pushed on encode and popped on decode) and the payload. A codec that does not own a header passes the others
through unread::

    0   4s  magic b"FCHN"
    4   u8  envelope version (1)
    5   u8  ndim (1..32)
    6   u8  payload kind: 0 float32 array, 1 opaque bytes
    7   u8  0
    8   u16 n_headers
    10  u16 0
    12  u32 header_bytes                (offset of the payload, a multiple of 8)
    16  u64 payload_bytes
    24  i64 shape[ndim]                 (the field array, C order; (dates, fields, ensembles, cells) for field stages)
    ..  n_headers records, bottom of the stack first; each: u16 id bytes n, u16 0, u32 header payload bytes m,
        the stage id (n bytes ASCII), the header payload (m bytes), zero padding to a multiple of 8
    ..  the payload at header_bytes: kind 0, the little-endian float32 array of ``shape`` (prod(shape) x 4 bytes);
        kind 1, the bytes of a later codec that coded the array (an ``icon_lattice`` container), opaque here

A buffer is an envelope only when every field is consistent (magic, version, kind, zero reserved fields, the
records ending exactly at ``header_bytes``, ``header_bytes + payload_bytes`` = its length, and for kind 0
``payload_bytes`` = prod(shape) x 4); anything else is a bare buffer. A bare buffer is the plain little-endian float32
chunk that the ``bytes`` codec writes, and unpacks as an array of the shape the caller supplies, with no headers.
The payload is 8-byte aligned relative to the start of the buffer, so the array is a view, never a copy.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "KIND_ARRAY",
    "KIND_BYTES",
    "MAGIC",
    "VERSION",
    "Envelope",
    "Header",
    "allocate",
    "header_size",
    "pack",
    "pack_bytes",
    "parse",
    "repack_in_place",
    "unpack",
]

MAGIC = b"FCHN"
VERSION = 1
KIND_ARRAY = 0  # little-endian float32 array of the envelope's shape
KIND_BYTES = 1  # opaque bytes written by a later codec
MAX_NDIM = 32
F4 = np.dtype("<f4")

Header = tuple[str, bytes]  # (stage id, header payload)

_HEAD = struct.Struct("<4sBBBBHHIQ")  # magic, version, ndim, kind, 0, n_headers, 0, header_bytes, payload_bytes
_RECORD = struct.Struct("<HHI")  # id bytes, 0, header payload bytes


@dataclass(frozen=True)
class Envelope:
    """A parsed envelope: ``shape`` of the field array, the header stack (bottom first), the payload ``kind``, the
    payload (a uint8 view into ``buffer``) and ``header_bytes`` (the payload offset)."""

    shape: tuple[int, ...]
    headers: list[Header]
    kind: int
    payload: np.ndarray
    buffer: np.ndarray
    header_bytes: int

    def array(self) -> np.ndarray:
        """The float32 field array (a view of the payload; kind 0 only)."""
        if self.kind != KIND_ARRAY:
            raise ValueError("the envelope payload is coded by a later codec, not a float32 array")
        return self.payload.view(F4).reshape(self.shape)


def _check_headers(headers: Sequence[Header]) -> list[tuple[bytes, bytes]]:
    out = []
    for h in headers:
        stage_id, payload = h
        if not isinstance(stage_id, str) or not stage_id or not stage_id.isascii() or not stage_id.isprintable():
            raise ValueError(f"a stage id is a non-empty printable ASCII string, got {stage_id!r}")
        sid = stage_id.encode("ascii")
        if len(sid) > 0xFFFF:
            raise ValueError(f"stage id of {len(sid)} bytes is too long")
        payload = bytes(payload)
        if len(payload) > 0xFFFFFFFF:
            raise ValueError(f"header of {stage_id!r} is {len(payload)} bytes, more than 4 GiB")
        out.append((sid, payload))
    if len(out) > 0xFFFF:
        raise ValueError(f"{len(out)} headers, at most 65535")
    return out


def _prefix(shape: Sequence[int], headers: Sequence[Header], kind: int, payload_bytes: int) -> bytes:
    shape = tuple(int(s) for s in shape)
    if not 1 <= len(shape) <= MAX_NDIM or any(s < 0 for s in shape):
        raise ValueError(f"envelope shape must have 1..{MAX_NDIM} non-negative dimensions, got {shape}")
    parts = [b"", struct.pack(f"<{len(shape)}q", *shape)]
    for sid, payload in _check_headers(headers):
        rec = _RECORD.pack(len(sid), 0, len(payload)) + sid + payload
        parts.append(rec + b"\0" * (-len(rec) % 8))
    body = b"".join(parts)
    hb = _HEAD.size + len(body)  # _HEAD.size and every part are multiples of 8
    parts[0] = _HEAD.pack(MAGIC, VERSION, len(shape), kind, 0, len(headers), 0, hb, payload_bytes)
    return b"".join(parts)


def header_size(shape: Sequence[int], headers: Sequence[Header]) -> int:
    """Bytes before the payload of an envelope with this shape and header stack."""
    return len(_prefix(shape, headers, KIND_ARRAY, 0))


def allocate(shape: Sequence[int], headers: Sequence[Header]) -> tuple[np.ndarray, np.ndarray]:
    """An envelope with an uninitialised float32 payload: (the whole buffer, uint8; the array view to fill). Lets a
    codec decode straight into the envelope it hands on."""
    n = int(np.prod(shape, dtype=np.int64)) * 4
    head = _prefix(shape, headers, KIND_ARRAY, n)
    buf = np.empty(len(head) + n, np.uint8)
    buf[: len(head)] = np.frombuffer(head, np.uint8)
    return buf, buf[len(head) :].view(F4).reshape(tuple(int(s) for s in shape))


def pack(array: np.ndarray, headers: Sequence[Header]) -> np.ndarray:
    """The envelope of a float32 ``array`` (any dtype convertible to little-endian float32) and the header stack
    (bottom first), as a uint8 array: the array is copied once into it."""
    buf, view = allocate(np.shape(array), headers)
    view[...] = array
    return buf


def pack_bytes(shape: Sequence[int], headers: Sequence[Header], payload: Any) -> np.ndarray:
    """The envelope of an opaque payload (kind 1: bytes a later codec wrote for the array of ``shape``)."""
    p = np.frombuffer(memoryview(payload).cast("B"), np.uint8)
    head = _prefix(shape, headers, KIND_BYTES, p.nbytes)
    buf = np.empty(len(head) + p.nbytes, np.uint8)
    buf[: len(head)] = np.frombuffer(head, np.uint8)
    buf[len(head) :] = p
    return buf


def _as_uint8(buf: Any) -> np.ndarray:
    if isinstance(buf, np.ndarray):
        if buf.dtype == np.uint8 and buf.ndim == 1:
            return buf
        if buf.flags.c_contiguous:
            return buf.reshape(-1).view(np.uint8)
    return np.frombuffer(memoryview(buf).cast("B"), np.uint8)


def parse(buf: Any) -> Envelope | None:
    """The envelope in ``buf`` (bytes-like or a contiguous numpy array), or None when ``buf`` is not an envelope
    (a bare buffer, or one that starts with the magic but is not consistent; see the module docstring)."""
    b = _as_uint8(buf)
    n = b.nbytes
    if n < _HEAD.size or bytes(b[:4]) != MAGIC:
        return None
    magic, version, ndim, kind, r0, n_headers, r1, hb, pb = _HEAD.unpack_from(b)
    if version != VERSION or kind not in (KIND_ARRAY, KIND_BYTES) or r0 or r1 or not 1 <= ndim <= MAX_NDIM:
        return None
    if hb % 8 or hb + pb != n or hb < _HEAD.size + 8 * ndim:
        return None
    shape = struct.unpack_from(f"<{ndim}q", b, _HEAD.size)
    if any(s < 0 for s in shape):
        return None
    if kind == KIND_ARRAY and pb != int(np.prod(shape, dtype=np.int64)) * 4:
        return None
    off = _HEAD.size + 8 * ndim
    headers: list[Header] = []
    for _ in range(n_headers):
        if off + _RECORD.size > hb:
            return None
        n_id, r2, m = _RECORD.unpack_from(b, off)
        end = off + _RECORD.size + n_id + m
        if r2 or not n_id or end > hb:
            return None
        sid = bytes(b[off + _RECORD.size : off + _RECORD.size + n_id])
        if not (sid.isascii() and sid.decode("ascii").isprintable()):
            return None
        headers.append((sid.decode("ascii"), bytes(b[end - m : end])))
        off = end + (-end % 8)
    if off != hb:
        return None
    return Envelope(tuple(int(s) for s in shape), headers, int(kind), b[hb:], b, int(hb))


def unpack(buf: Any, shape: Sequence[int] | None = None) -> tuple[np.ndarray, list[Header], tuple[int, ...]]:
    """(the float32 array, a view; the header stack, bottom first; the shape) of an envelope, or of a bare buffer
    (then ``shape`` is required, the buffer must hold exactly prod(shape) float32 values, and the stack is empty).
    Raises ``ValueError`` when the envelope's payload is not a float32 array."""
    env = parse(buf)
    if env is not None:
        return env.array(), env.headers, env.shape
    if shape is None:
        raise ValueError("not an envelope, and no shape given for a bare buffer")
    shape = tuple(int(s) for s in shape)
    b = _as_uint8(buf)
    if b.nbytes != int(np.prod(shape, dtype=np.int64)) * 4:
        raise ValueError(f"bare buffer of {b.nbytes} bytes does not hold a float32 array of shape {shape}")
    return b.view(F4).reshape(shape), [], shape


def repack_in_place(env: Envelope, headers: Sequence[Header]) -> np.ndarray | None:
    """The envelope of ``env``'s float32 payload (unchanged position, possibly rewritten in place) with a shorter or
    equal header stack, written into ``env.buffer`` just before the payload: no copy of the array. None when the
    buffer is read-only or the new prefix does not fit."""
    if env.kind != KIND_ARRAY or not env.buffer.flags.writeable:
        return None
    head = _prefix(env.shape, headers, KIND_ARRAY, env.payload.nbytes)
    start = env.header_bytes - len(head)
    if start < 0:
        return None
    env.buffer[start : env.header_bytes] = np.frombuffer(head, np.uint8)
    return env.buffer[start:]
