"""The in-band envelope of a codec chain (``anemoi_datasets_codecs.envelope``, frozen format version 1)."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from anemoi_datasets_codecs import envelope as E

# a fixed envelope of each payload kind, as written by format version 1 (docs/codecs/envelope.md); never change
FROZEN_ARRAY = bytes.fromhex(
    "4643484e010400000200000060000000180000000000000001000000000000000200000000000000010000000000000003000000"
    "00000000040000000300000064726f70010203000b000000000000006669656c645f7465726d730000000000000000000000003f"
    "0000803f0000c03f0000004000002040"
)
FROZEN_BYTES = bytes.fromhex(
    "4643484e010401000100000050000000080000000000000002000000000000000300000000000000010000000000000005000000"
    "000000000100000008000000736162636465666768000000000000004c41544330313233"
)


def _fixed_array():
    return np.arange(6, dtype=np.float32).reshape(1, 2, 1, 3) * np.float32(0.5)


def test_envelope_frozen_bytes():
    headers = [("drop", b"\x01\x02\x03"), ("field_terms", b"")]
    assert E.pack(_fixed_array(), headers).tobytes() == FROZEN_ARRAY
    assert E.pack_bytes((2, 3, 1, 5), [("s", b"abcdefgh")], b"LATC0123").tobytes() == FROZEN_BYTES
    # the layout, field by field (docs/codecs/envelope.md)
    magic, version, ndim, kind, r0, n_headers, r1, hb, pb = struct.unpack_from("<4sBBBBHHIQ", FROZEN_ARRAY)
    assert (magic, version, ndim, kind, r0, n_headers, r1, hb, pb) == (b"FCHN", 1, 4, 0, 0, 2, 0, 96, 24)
    assert struct.unpack_from("<4q", FROZEN_ARRAY, 24) == (1, 2, 1, 3)
    assert struct.unpack_from("<HHI4s", FROZEN_ARRAY, 56) == (4, 0, 3, b"drop")  # record padded to 16 bytes
    assert struct.unpack_from("<HHI11s", FROZEN_ARRAY, 72) == (11, 0, 0, b"field_terms")  # padded to 24 bytes
    assert FROZEN_ARRAY[hb:] == _fixed_array().tobytes()
    # and both read back
    x, h, shape = E.unpack(FROZEN_ARRAY)
    assert np.array_equal(x, _fixed_array()) and h == headers and shape == (1, 2, 1, 3)
    env = E.parse(FROZEN_BYTES)
    assert env.kind == E.KIND_BYTES and env.shape == (2, 3, 1, 5) and env.headers == [("s", b"abcdefgh")]
    assert env.payload.tobytes() == b"LATC0123"
    with pytest.raises(ValueError, match="later codec"):
        E.unpack(FROZEN_BYTES)


@pytest.mark.parametrize("n_headers", [0, 1, 3])
def test_envelope_round_trip(n_headers):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 5, 3, 7)).astype(np.float32)
    headers = [(f"stage_{i}", rng.bytes(int(rng.integers(0, 40)))) for i in range(n_headers)]
    buf = E.pack(x, headers)
    assert buf.dtype == np.uint8 and E.header_size(x.shape, headers) % 8 == 0
    assert buf.nbytes == E.header_size(x.shape, headers) + x.nbytes
    y, h, shape = E.unpack(buf)
    assert np.array_equal(y.view(np.uint32), x.view(np.uint32)) and h == headers and shape == x.shape
    assert np.shares_memory(y, buf)  # a view, no copy
    assert E.unpack(bytes(buf))[1] == headers  # any bytes-like
    # allocate: the same bytes once filled
    out, view = E.allocate(x.shape, headers)
    view[...] = x
    assert out.tobytes() == buf.tobytes()


def test_envelope_bare_and_malformed():
    x = np.arange(12, dtype=np.float32)
    bare = x.tobytes()
    y, h, shape = E.unpack(bare, (1, 3, 1, 4))
    assert h == [] and shape == (1, 3, 1, 4) and np.array_equal(y.reshape(-1), x)
    with pytest.raises(ValueError, match="no shape"):
        E.unpack(bare)
    with pytest.raises(ValueError, match="does not hold"):
        E.unpack(bare, (1, 3, 1, 5))
    # a float32 chunk whose first value has the bits of the magic is still a bare chunk
    z = np.zeros(64, np.float32)
    z[0] = np.frombuffer(b"FCHN", np.float32)[0]
    assert E.parse(z) is None
    good = E.pack(_fixed_array(), [("a", b"xyz")]).tobytes()
    assert E.parse(good) is not None
    assert E.parse(good[:-4]) is None  # truncated
    assert E.parse(good + b"\0" * 8) is None  # trailing bytes
    assert E.parse(good[:4] + b"\x02" + good[5:]) is None  # another version
    assert E.parse(good[:6] + b"\x07" + good[7:]) is None  # unknown payload kind
    with pytest.raises(ValueError, match="stage id"):
        E.pack(_fixed_array(), [("", b"")])
    with pytest.raises(ValueError, match="dimensions"):
        E.pack(np.float32(1), [])


def test_envelope_repack_in_place():
    x = _fixed_array()
    buf = E.pack(x, [("a", b"1" * 20), ("b", b"2")])
    env = E.parse(buf)
    env.array()[0, 1] += 1  # a stage decoding in place
    out = E.repack_in_place(env, env.headers[:1])
    assert out is not None and np.shares_memory(out, buf)
    y, h, _ = E.unpack(out)
    assert h == [("a", b"1" * 20)] and np.shares_memory(y, buf)
    assert out.tobytes() == E.pack(env.array(), h).tobytes()
    ro = np.frombuffer(E.pack(x, [("a", b"")]).tobytes(), np.uint8)
    assert E.repack_in_place(E.parse(ro), []) is None  # read-only: the caller packs
