"""Generic Zarr 3 adapters (zarr-python >= 3.0.8) for chunk codec engines and field stages.

Two kinds of engine know nothing about zarr; this module turns an engine class into Zarr 3 codecs.

* A *chunk engine* (:class:`ChunkCodec`) codes a whole chunk: an array in, bytes out (``icon_lattice``).
  * :class:`ChunkArrayBytesCodec` (array to bytes): the chunk array goes straight to the engine, so the codec takes
    the place of the ``bytes`` codec (``serializer=...``, ``compressors=None``);
  * :class:`ChunkBytesBytesCodec` (bytes to bytes): the engine after the default ``bytes`` codec (little endian), for
    tools that only let one choose ``compressors``. The stored chunks are identical.
* A *field stage* (:class:`FieldStage`) rewrites the float32 field array of a chunk (dates, fields, ensembles,
  cells) into another float32 field array plus a small per-chunk header: it drops fields, or replaces some by
  residuals. :class:`FieldStageBytesCodec` runs it as a bytes-to-bytes codec after the ``bytes`` codec, so stages
  chain in ``compressors=[stage, ..., icon_lattice_bytes]`` ahead of a chunk engine.

Stages talk through the in-band envelope (``envelope.py``): the shape of the field array and a stack of headers,
one per stage. On encode a stage unpacks a bare chunk (the ``bytes`` codec's output, of the chunk spec's shape) or an
envelope, runs its engine and packs the result with its header pushed on the stack; on decode it pops its header,
runs its engine and packs the result again, or hands zarr the bare float32 chunk when the stack is empty (it was the
first stage). A chunk engine in chain mode (``chain`` true, e.g. ``icon_lattice`` with ``chain: true``) codes the
array inside an envelope and passes the headers through unread: encode gives ``envelope(shape, headers, engine
bytes)``, decode gives the envelope of the decoded array back. A bare input always gives the bare engine bytes, so a
chunk engine used alone writes exactly the chunks it writes without the envelope.

zarr validates every codec against the array's chunk grid, never against the shape the codecs before it leave
(``ArrayV3Metadata`` passes the original grid to ``validate``), and the two zarr pipelines disagree on the spec a
bytes-to-bytes codec gets when an earlier one announces a new shape through ``resolve_metadata``. So
``resolve_metadata`` stays the identity, the shape travels in the envelope, every stage checks the envelope's dates,
ensembles and cells against ``chunk_spec`` and its field count against its configuration when it codes a chunk, and
codecs in a chain check at array creation only what the original grid tells: ``(dates, fields, ensembles, cells)``
chunks holding every field of a date.

A codec package subclasses an adapter with two class attributes, ``codec_name`` (the name in ``zarr.json``, in the
``zarr.codecs`` entry point and, for a stage, the id of its header in the envelope) and ``engine`` (the engine class,
built as ``engine(**configuration)``). The metadata is ``{"name": codec_name, "configuration": engine.get_config()}``
plus the writer stamp ``written_by`` (``config.STAMP_KEY``: the package version that created the array metadata;
informational, never given to the engine, ignored by every comparison; see :class:`_ChunkCodecBase`).
Encoding and decoding run in a worker thread (``asyncio.to_thread``), so zarr's concurrent chunk reads and writes run
in parallel when the engine releases the GIL.

Copies: the adapters do not copy the chunk. The decoded array goes to zarr as a view (an NDBuffer, or a byte view for
the ``bytes`` codec, which views it back as an array); the input of ``encode`` reaches the engine as a view (it can be
the caller's array: a stage gets it read-only). A stage's output is copied once into the envelope on encode; on
decode a stage that works in place re-labels the envelope it was given (no copy), one that changes the shape decodes
into the next envelope (``out``), and a chain-mode chunk engine decodes straight into the envelope it hands on.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
from zarr.abc.codec import ArrayBytesCodec, BytesBytesCodec
from zarr.core.codec_pipeline import BatchedCodecPipeline
from zarr.core.config import config as zarr_config
from zarr.registry import register_pipeline
from zarr.storage import LocalStore

from . import envelope, sidecar
from .config import STAMP_KEY, writer_stamp

__all__ = ["ChunkArrayBytesCodec", "ChunkBytesBytesCodec", "ChunkCodec", "FieldStage", "FieldStageBytesCodec"]

F4 = np.dtype("<f4")


@runtime_checkable
class ChunkCodec(Protocol):
    """What a chunk engine offers the adapters. A new codec builds it as ``engine(**configuration)``; stored metadata
    is read with ``engine.from_config(configuration)`` when the engine has that classmethod, else the same way. An
    engine with a true ``chain`` attribute takes part in a chain of field stages (module docstring): it then gets
    enveloped inputs and must accept a chunk grid whose field count differs from its own in ``validate_chunk_shape``."""

    @property
    def names(self) -> Sequence[str]:
        """Names of the chunk's fields (variables), for messages."""
        ...

    @property
    def fields_per_date(self) -> int: ...

    def encode(self, x: np.ndarray) -> bytes:
        """The bytes of one chunk (any shape the engine accepts; C order)."""
        ...

    def decode(self, buf: Any, out: Any = None) -> np.ndarray:
        """Decode a chunk into ``out`` or a new flat array."""
        ...

    def get_config(self) -> dict[str, Any]:
        """The canonical configuration (small JSON): ``type(self)(**get_config())`` is an equal engine."""
        ...

    def validate_chunk_shape(self, chunks: tuple[int, ...]) -> None:
        """Raise ``ValueError`` when the engine cannot code chunks of this shape."""
        ...


class FieldStage:
    """Base of a field-stage engine: rewrites the float32 field array ``(dates, fields, ensembles, cells)`` of a
    chunk, the field axis 1. Subclasses implement ``encode`` and ``decode`` and extend ``get_config`` (and
    ``variables_out`` when they change the fields). The contract (docs/design.md, "Field stages"):

    * ``variables``: the names of the fields the stage receives, in chunk order (the configuration names them;
      indices are resolved inside the engine). ``variables_out`` / ``fields_out``: what it emits.
    * ``encode(x) -> (y, header)``: ``x`` has ``fields_in`` fields and is read-only (it can be the caller's array);
      ``y`` has the same dates, ensembles and cells and ``fields_out`` fields (a new array or a view of ``x``);
      ``header`` is the per-chunk metadata the decoder needs (bytes, or None for none), kept small: it travels
      through every later codec of the chain and is copied once per stage.
    * ``decode(y, header, out_shape, out=None) -> x``: ``y`` is the array the stage emitted (bit-identical when the
      codecs after it are exact; a lossy codec after it gives it within that codec's bounds), writable and owned by
      the call, so the stage may work in place on it; ``header`` is the bytes ``encode`` returned (empty for None);
      ``out_shape`` is the shape to return. When ``out_shape`` differs from ``y.shape`` the adapter passes ``out``,
      an uninitialised float32 array of that shape inside the envelope it hands on: decoding into it saves a copy.
    * Deterministic: the same input and configuration give the same bytes, on every machine.
    """

    def __init__(self, variables: Sequence[str]) -> None:
        if isinstance(variables, str) or not isinstance(variables, Sequence) or not variables:
            raise ValueError("variables must be a non-empty list of names")
        names = tuple(variables)
        if not all(isinstance(n, str) and n for n in names):
            raise ValueError(f"variables must be names, got {list(names)!r}")
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate variables: {sorted({n for n in names if names.count(n) > 1})}")
        self._variables = names

    @property
    def variables(self) -> tuple[str, ...]:
        """Names of the fields the stage receives (chunk order)."""
        return self._variables

    @property
    def names(self) -> tuple[str, ...]:
        return self._variables

    @property
    def variables_out(self) -> tuple[str, ...]:
        """Names of the fields the stage emits (what the next codec of the chain is configured with)."""
        return self._variables

    @property
    def fields_in(self) -> int:
        return len(self.variables)

    @property
    def fields_out(self) -> int:
        return len(self.variables_out)

    def encode(self, x: np.ndarray) -> tuple[np.ndarray, bytes | None]:
        """``(y, header)``: the ``fields_out`` fields to hand on (same dates, ensembles and cells; a new array or a view
        of ``x``, which is read-only) and the per-chunk header (bytes, or None). Deterministic."""
        raise NotImplementedError

    def decode(
        self, y: np.ndarray, header: bytes, out_shape: tuple[int, ...], out: np.ndarray | None = None
    ) -> np.ndarray:
        """The ``fields_in`` fields of ``out_shape`` from ``y`` (the array :meth:`encode` emitted, as the codecs after
        the stage returned it; writable and owned by the call) and ``header`` (empty for None). ``out``, when given,
        is an uninitialised float32 array of ``out_shape`` to decode into."""
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        """The canonical configuration (small JSON); subclasses add their keys."""
        return {"variables": list(self.variables)}

    def validate_chunk_shape(self, chunks: tuple[int, ...]) -> None:
        """Raise ``ValueError`` when the stage cannot code chunks of this grid shape (the adapter has already checked
        ``(dates, fields, ensembles, cells)`` chunks holding every field of a date)."""

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and other.get_config() == self.get_config()  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash((type(self).__qualname__, json.dumps(self.get_config(), sort_keys=True, default=str)))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(variables={len(self.variables)}, fields_out={self.fields_out})"


def _native_dtype(dtype: Any) -> np.dtype:
    """numpy dtype of an ArraySpec / array dtype (a ZDType in zarr >= 3.1, a numpy dtype before)."""
    to_native = getattr(dtype, "to_native_dtype", None)
    return np.dtype(to_native() if to_native is not None else dtype)


def _chunk_shape(chunk_grid: Any) -> tuple[int, ...] | None:
    """The chunk shape of a regular chunk grid (None for other grids)."""
    shape = getattr(chunk_grid, "chunk_shape", None)
    return None if shape is None else tuple(int(s) for s in shape)


def _check_chain_grid(name: str, shape: Sequence[int], chunks: tuple[int, ...]) -> None:
    """What a codec in a chain can check against the array's original chunk grid."""
    if len(chunks) != 4 or len(shape) != 4:
        raise ValueError(f"{name}: a codec chain needs (dates, fields, ensembles, cells) chunks, got {chunks}")
    if chunks[1] != shape[1]:
        raise ValueError(f"{name}: chunks {chunks} must hold every field of a date ({shape[1]} fields)")


def _check_like_spec(name: str, shape: tuple[int, ...], spec: tuple[int, ...]) -> None:
    """The envelope's dates, ensembles and cells must be the chunk's (only the field count may differ)."""
    if len(shape) != 4 or len(spec) != 4 or shape[0] != spec[0] or shape[2:] != spec[2:]:
        raise ValueError(
            f"{name}: the envelope holds a field array of shape {shape}, which does not fit chunks of shape {spec} "
            "(only the field count may differ)"
        )


def _same_array(a: np.ndarray, b: np.ndarray) -> bool:
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and a.strides == b.strides
        and a.__array_interface__["data"][0] == b.__array_interface__["data"][0]
    )


def _bare(x: np.ndarray) -> Any:
    """The little-endian float32 bytes of ``x`` as a byte view (no copy when ``x`` is already such an array)."""
    x = np.ascontiguousarray(x, F4)
    return memoryview(x.reshape(-1).view(np.uint8))


def _decode_into(codec: Any, buf: Any, out: np.ndarray, copy: bool = True) -> np.ndarray:
    """Decode a chunk into ``out``, allocated by the adapter from the chunk shape, so that the engine checks the
    chunk's header against it before it allocates anything. An engine that ignores ``out`` returns its own array:
    copied into ``out`` (``copy``), else returned in ``out``'s shape."""
    res = np.asarray(codec.decode(buf, out=out))
    if res.__array_interface__["data"][0] == out.__array_interface__["data"][0]:
        return out
    if not copy:
        return res.reshape(out.shape)
    out[...] = res.reshape(out.shape)
    return out


class _ChunkCodecBase:
    """Configuration handling and validation common to the adapters (zarr metadata ``{name, configuration}``).

    The ``configuration`` is the engine's ``get_config()`` plus the writer stamp ``written_by`` (``config.STAMP_KEY``),
    which the engine never sees. A codec built from a configuration with a stamp keeps it, whether zarr reads it from
    ``zarr.json`` (``from_dict``) or it is passed as a keyword: a store's metadata that a later version reads and
    writes again (zarr rewrites ``zarr.json`` when an array is resized, e.g. to append dates) keeps naming the version
    that created the array. A codec built without one is stamped with the running version (a new array, a recipe's
    codec entry, ``physics expand``'s output). Equality ignores the stamp."""

    codec_name: ClassVar[str]
    engine: ClassVar[type]
    dtype: ClassVar[np.dtype] = np.dtype("float32")  # the data type the engine codes
    is_fixed_size = False

    def __init__(self, **configuration: Any) -> None:
        configuration.pop("id", None)  # a numcodecs-style identifier is not part of a Zarr 3 configuration
        stamp = configuration.pop(STAMP_KEY, None)
        self._attach(self.engine(**configuration), stamp)

    def _attach(self, codec: Any, stamp: Any = None) -> None:
        cfg = dict(codec.get_config())
        cfg.pop("id", None)
        cfg[STAMP_KEY] = writer_stamp() if stamp is None else stamp
        object.__setattr__(self, "_codec", codec)  # zarr codecs are frozen dataclasses
        object.__setattr__(self, "configuration", cfg)
        object.__setattr__(self, "_chain", bool(getattr(codec, "chain", False)))

    @property
    def codec(self) -> Any:
        """The engine doing the work."""
        return self._codec

    @property
    def written_by(self) -> Any:
        """The writer stamp (``anemoi-datasets-codecs <version>``) of the configuration: the version that created
        the array metadata, or the running version for a new codec."""
        return self.configuration[STAMP_KEY]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Any:
        if not isinstance(data, dict) or data.get("name") != cls.codec_name:
            raise ValueError(f"expected a {cls.codec_name!r} codec, got {data!r}")
        cfg = dict(data.get("configuration") or {})
        cfg.pop("id", None)
        stamp = cfg.pop(STAMP_KEY, None)
        from_config = getattr(cls.engine, "from_config", None)
        self = cls.__new__(cls)
        self._attach(cls.engine(**cfg) if from_config is None else from_config(cfg), stamp)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.codec_name, "configuration": dict(self.configuration)}

    def _unstamped(self) -> dict[str, Any]:
        return {k: v for k, v in self.configuration.items() if k != STAMP_KEY}

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: Any) -> int:
        raise NotImplementedError  # variable size

    def validate(self, *, shape: tuple[int, ...], dtype: Any, chunk_grid: Any) -> None:
        if _native_dtype(dtype) != self.dtype:
            raise ValueError(f"{self.codec_name} needs {self.dtype} data, got {dtype}")
        chunks = _chunk_shape(chunk_grid)
        if chunks is not None:
            if self._chain:
                _check_chain_grid(self.codec_name, shape, chunks)
            try:
                self._codec.validate_chunk_shape(chunks)
            except ValueError as e:
                raise ValueError(f"{self.codec_name}: {e}") from None

    def _check_fields(self, shape: tuple[int, ...]) -> None:
        """Chain mode: a chunk's field array must have the engine's fields (variables x ensembles per date)."""
        names = list(self._codec.names)
        if len(shape) != 4 or shape[1] != len(names) or shape[1] * shape[2] != self._codec.fields_per_date:
            raise ValueError(
                f"{self.codec_name}: a field array of shape {shape} reached the codec, which is configured for "
                f"{len(names)} variables and {self._codec.fields_per_date} fields per date"
            )

    def _encode_array(self, x: np.ndarray) -> bytes:
        return self._codec.encode(np.ascontiguousarray(x, dtype=self.dtype))

    def __eq__(self, other: object) -> bool:  # the stamp is informational: not compared
        return type(other) is type(self) and other._unstamped() == self._unstamped()  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash((self.codec_name, hash(self._codec)))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._codec!r})"


class ChunkArrayBytesCodec(_ChunkCodecBase, ArrayBytesCodec):
    """Array-to-bytes adapter (takes the place of the ``bytes`` codec). Subclasses set ``codec_name`` and
    ``engine``."""

    def _decode_sync(self, chunk_bytes: Any, chunk_spec: Any) -> Any:
        out = np.empty(tuple(chunk_spec.shape), self.dtype)  # the engine checks its header against it
        out = _decode_into(self._codec, chunk_bytes.as_numpy_array(), out, copy=False)
        dt = _native_dtype(chunk_spec.dtype)
        if out.dtype != dt:
            out = out.astype(dt)
        return chunk_spec.prototype.nd_buffer.from_numpy_array(out)

    def _encode_sync(self, chunk_array: Any, chunk_spec: Any) -> Any:
        if self._chain:  # the spec is the shape the array codecs before it resolved
            self._check_fields(tuple(chunk_spec.shape))
        return chunk_spec.prototype.buffer.from_bytes(self._encode_array(chunk_array.as_numpy_array()))

    async def _decode_single(self, chunk_bytes: Any, chunk_spec: Any) -> Any:
        return await asyncio.to_thread(self._decode_sync, chunk_bytes, chunk_spec)

    async def _encode_single(self, chunk_array: Any, chunk_spec: Any) -> Any:
        return await asyncio.to_thread(self._encode_sync, chunk_array, chunk_spec)


class ChunkBytesBytesCodec(_ChunkCodecBase, BytesBytesCodec):
    """Bytes-to-bytes adapter: follows the ``bytes`` codec, which must write little-endian values (its default).
    Subclasses set ``codec_name`` and ``engine``. In chain mode it also takes enveloped inputs (module docstring)."""

    def _decode_sync(self, chunk_bytes: Any, chunk_spec: Any) -> Any:
        buf = chunk_bytes.as_numpy_array()
        if self._chain:
            env = envelope.parse(buf)
            if env is not None:
                return self._decode_envelope(env, chunk_spec)
        out = _decode_into(self._codec, buf, np.empty(tuple(chunk_spec.shape), self.dtype.newbyteorder("<")), False)
        out = np.ascontiguousarray(out, self.dtype.newbyteorder("<"))
        return chunk_spec.prototype.buffer.from_bytes(memoryview(out.reshape(-1).view(np.uint8)))  # a view, no copy

    def _encode_sync(self, chunk_bytes: Any, chunk_spec: Any) -> Any:
        buf = chunk_bytes.as_numpy_array()
        if self._chain:
            env = envelope.parse(buf)
            if env is not None:
                return self._encode_envelope(env, chunk_spec)
            self._check_fields(tuple(chunk_spec.shape))
        x = np.frombuffer(buf, self.dtype.newbyteorder("<"))
        try:
            enc = self._encode_array(x)
        except ValueError as e:
            if not self._chain and envelope.parse(buf) is not None:
                raise ValueError(
                    f"{self.codec_name}: the chunk comes enveloped from a field stage before this codec, which needs "
                    "chain: true in its configuration"
                ) from e
            raise
        return chunk_spec.prototype.buffer.from_bytes(enc)

    def _encode_envelope(self, env: envelope.Envelope, chunk_spec: Any) -> Any:
        if env.kind != envelope.KIND_ARRAY:
            raise ValueError(f"{self.codec_name}: the chunk was already coded by another codec of the chain")
        _check_like_spec(self.codec_name, env.shape, tuple(chunk_spec.shape))
        self._check_fields(env.shape)
        enc = self._encode_array(env.array())
        return chunk_spec.prototype.buffer.from_bytes(envelope.pack_bytes(env.shape, env.headers, enc))

    def _decode_envelope(self, env: envelope.Envelope, chunk_spec: Any) -> Any:
        if env.kind != envelope.KIND_BYTES:
            raise ValueError(f"{self.codec_name}: the envelope holds a float32 array, not this codec's bytes")
        spec = tuple(chunk_spec.shape)
        _check_like_spec(self.codec_name, env.shape, spec)
        self._check_fields(env.shape)
        if env.headers:  # decode straight into the envelope handed to the stages before it
            out, view = envelope.allocate(env.shape, env.headers)
            self._decode_into(env.payload, view)
            return chunk_spec.prototype.buffer.from_bytes(out)
        if env.shape != spec:
            raise ValueError(f"{self.codec_name}: a bare chunk of shape {env.shape} does not fit chunks of {spec}")
        x = np.empty(env.shape, F4)
        self._decode_into(env.payload, x)
        return chunk_spec.prototype.buffer.from_bytes(_bare(x))

    def _decode_into(self, payload: np.ndarray, view: np.ndarray) -> None:
        _decode_into(self._codec, payload, view)

    async def _decode_single(self, chunk_bytes: Any, chunk_spec: Any) -> Any:
        return await asyncio.to_thread(self._decode_sync, chunk_bytes, chunk_spec)

    async def _encode_single(self, chunk_bytes: Any, chunk_spec: Any) -> Any:
        return await asyncio.to_thread(self._encode_sync, chunk_bytes, chunk_spec)


class FieldStageBytesCodec(_ChunkCodecBase, BytesBytesCodec):
    """Bytes-to-bytes adapter of a :class:`FieldStage` engine (module docstring). Subclasses set ``codec_name`` (also
    the id of the stage's header in the envelope) and ``engine``. Its output is always an envelope (with its header
    pushed) on encode; on decode it is an envelope while headers of earlier stages remain, else the bare chunk."""

    def validate(self, *, shape: tuple[int, ...], dtype: Any, chunk_grid: Any) -> None:
        if _native_dtype(dtype) != self.dtype:
            raise ValueError(f"{self.codec_name} needs {self.dtype} data, got {dtype}")
        chunks = _chunk_shape(chunk_grid)
        if chunks is not None:
            _check_chain_grid(self.codec_name, shape, chunks)
            try:
                self._codec.validate_chunk_shape(chunks)
            except ValueError as e:
                raise ValueError(f"{self.codec_name}: {e}") from None

    def _check_count(self, shape: tuple[int, ...], n: int, what: str) -> None:
        if shape[1] != n:
            raise ValueError(
                f"{self.codec_name}: the chunk {what} has {shape[1]} fields per date, the stage is configured for "
                f"{n} (variables {list(self._codec.variables)})"
            )

    def _encode_sync(self, chunk_bytes: Any, chunk_spec: Any) -> Any:
        spec = tuple(chunk_spec.shape)
        buf = chunk_bytes.as_numpy_array()
        env = envelope.parse(buf)
        if env is None:
            x, headers, shape = envelope.unpack(buf, spec)
            if len(shape) != 4:
                raise ValueError(f"{self.codec_name}: needs (dates, fields, ensembles, cells) chunks, got {shape}")
        else:
            if env.kind != envelope.KIND_ARRAY:
                raise ValueError(f"{self.codec_name}: the chunk was already coded by a codec that must come later")
            x, headers, shape = env.array(), env.headers, env.shape
            _check_like_spec(self.codec_name, shape, spec)
        self._check_count(shape, self._codec.fields_in, "it receives")
        x = x.view()
        x.flags.writeable = False  # it can be the caller's array
        y, header = self._codec.encode(x)
        want = (shape[0], self._codec.fields_out, *shape[2:])
        if tuple(y.shape) != want:
            raise ValueError(f"{self.codec_name}: the engine returned shape {tuple(y.shape)}, expected {want}")
        out = envelope.pack(y, [*headers, (self.codec_name, b"" if header is None else bytes(header))])
        return chunk_spec.prototype.buffer.from_bytes(out)

    def _decode_sync(self, chunk_bytes: Any, chunk_spec: Any) -> Any:
        spec = tuple(chunk_spec.shape)
        env = envelope.parse(chunk_bytes.as_numpy_array())
        if env is None or env.kind != envelope.KIND_ARRAY:
            raise ValueError(
                f"{self.codec_name}: expected an envelope holding a float32 array (the codecs after this stage in "
                "the chain must pass it through)"
            )
        if not env.headers or env.headers[-1][0] != self.codec_name:
            top = env.headers[-1][0] if env.headers else None
            raise ValueError(f"{self.codec_name}: the envelope's top header is {top!r}, not this stage's")
        header, rest = env.headers[-1][1], env.headers[:-1]
        _check_like_spec(self.codec_name, env.shape, spec)
        self._check_count(env.shape, self._codec.fields_out, "it decodes")
        out_shape = (env.shape[0], self._codec.fields_in, *env.shape[2:])
        last = not rest  # the first stage of the chain: hand zarr the bare chunk
        if last and out_shape != spec:
            raise ValueError(f"{self.codec_name}: decodes a chunk of shape {out_shape}, zarr expects {spec}")
        y = env.array()
        target = None  # (buffer, array) of the envelope handed on, when allocated here
        out = None
        if out_shape == env.shape:
            if not y.flags.writeable:  # the stage may work in place: give it a copy where the result goes
                if last:
                    y = y.copy()
                else:
                    target = envelope.allocate(out_shape, rest)
                    target[1][...] = y
                    y = target[1]
        elif last:
            out = np.empty(out_shape, F4)
        else:
            target = envelope.allocate(out_shape, rest)
            out = target[1]
        x = self._codec.decode(y, header, out_shape, out=out)
        if tuple(x.shape) != out_shape:
            raise ValueError(f"{self.codec_name}: the engine decoded shape {tuple(x.shape)}, expected {out_shape}")
        if last:
            return chunk_spec.prototype.buffer.from_bytes(_bare(x))
        if target is not None and _same_array(x, target[1]):
            res = target[0]
        else:
            res = envelope.repack_in_place(env, rest) if _same_array(x, env.array()) else None
            if res is None:
                res = envelope.pack(x, rest)
        return chunk_spec.prototype.buffer.from_bytes(res)

    async def _decode_single(self, chunk_bytes: Any, chunk_spec: Any) -> Any:
        return await asyncio.to_thread(self._decode_sync, chunk_bytes, chunk_spec)

    async def _encode_single(self, chunk_bytes: Any, chunk_spec: Any) -> Any:
        return await asyncio.to_thread(self._encode_sync, chunk_bytes, chunk_spec)


# ---------------------------------------------------------------------------------- the pipeline that knows the store
DEFAULT_PIPELINE = "zarr.core.codec_pipeline.BatchedCodecPipeline"


@dataclass(frozen=True)
class SidecarPipeline(BatchedCodecPipeline):
    """zarr's default codec pipeline, plus the store it serves. zarr hands the store to the pipeline of every array
    (``from_array_metadata_and_store``), never to a codec; when the array's codecs include one of this package's and
    the store is a local directory, the pipeline serves ``<store>/.codecs`` (``sidecar.serving``) for the duration of
    each read and write, so the codecs find the store's lattice and static fields and the encoder captures into the
    store without ``register_store`` or ``ANEMOI_DATASETS_CODECS_PATH``. Selected through zarr's
    ``codec_pipeline.path`` setting when this module is imported, unless another pipeline is configured (then the
    codecs behave as without it: the sidecar must be registered or on the path) or ``ANEMOI_DATASETS_CODECS_PIPELINE``
    is ``off``. Other store types serve no sidecar."""

    sidecar: Path | None = None

    @classmethod
    def from_array_metadata_and_store(cls, array_metadata: Any, store: Any) -> Any:
        from zarr.core.array import create_codec_pipeline

        pipeline = create_codec_pipeline(array_metadata)  # the regular construction with the configured class
        if isinstance(pipeline, SidecarPipeline):
            pipeline = replace(pipeline, sidecar=cls.sidecar_of(array_metadata, store))
        return pipeline

    @staticmethod
    def sidecar_of(array_metadata: Any, store: Any) -> Path | None:
        """``<store>/.codecs`` for a local store whose array uses a codec of this package, else None."""
        codecs = getattr(array_metadata, "codecs", ())
        ours = (_ChunkCodecBase, FieldStageBytesCodec)
        if isinstance(store, LocalStore) and any(isinstance(c, ours) for c in codecs):
            return Path(store.root) / sidecar.SIDECAR
        return None

    def evolve_from_array_spec(self, array_spec: Any) -> Any:
        return replace(super().evolve_from_array_spec(array_spec), sidecar=self.sidecar)

    async def read(self, *args: Any, **kwargs: Any) -> Any:
        with sidecar.serving(self.sidecar):
            return await super().read(*args, **kwargs)

    async def write(self, *args: Any, **kwargs: Any) -> Any:
        with sidecar.serving(self.sidecar):
            return await super().write(*args, **kwargs)

    async def encode(self, *args: Any, **kwargs: Any) -> Any:
        with sidecar.serving(self.sidecar):
            return await super().encode(*args, **kwargs)

    async def decode(self, *args: Any, **kwargs: Any) -> Any:
        with sidecar.serving(self.sidecar):
            return await super().decode(*args, **kwargs)


SIDECAR_PIPELINE = f"{SidecarPipeline.__module__}.{SidecarPipeline.__qualname__}"
PIPELINE_ENV = "ANEMOI_DATASETS_CODECS_PIPELINE"
register_pipeline(SidecarPipeline)
if (
    os.environ.get(PIPELINE_ENV, "").strip().lower() not in ("off", "0", "false", "none")
    and zarr_config.get("codec_pipeline.path") == DEFAULT_PIPELINE
):
    zarr_config.set({"codec_pipeline.path": SIDECAR_PIPELINE})
