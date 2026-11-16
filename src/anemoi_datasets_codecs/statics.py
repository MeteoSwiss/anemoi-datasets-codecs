"""Static fields (identical in every chunk), shared by the codecs that reference them (``icon_lattice``'s STATIC_REF
mode, the ``static_ref`` stage).

A static field is stored per chunk as a reference: the sha256 of the field's float32 bytes (in ``icon_lattice``,
with option ``statics: inline``, the sha is followed by the blosc-lz4 field, so the chunk is self-contained). The
arrays are content addressed: never in the codec configuration, but in sidecar files ``statics/<sha256>.f4.blosc``
found on the sidecar search path (``sidecar.py``: registered stores, ``$ANEMOI_DATASETS_CODECS_PATH``, the cache
directory).

Encoder: a static field whose sha256 is not available yet is *captured* from the chunk: written to the capture target
(the registered / path store's sidecar when writable; the cache directory only without one; logged at level INFO on
the ``anemoi_datasets_codecs.statics`` logger). Only a field that is
durably available (found, or persisted) is coded as a reference; otherwise the codec stores it in that chunk with a
warning (``icon_lattice`` losslessly, ``static_ref`` inline), so a chunk never references data that may not exist. At
most ``MAX_VERSIONS`` different arrays are captured per variable name and codec (a "static" that changes every date is
stored in the chunk instead).

Decoder: process cache, then inline bytes, then the search path; a missing static raises
``sidecar.MissingReference`` naming the sha and the searched directories (the chunk cannot be decoded without it:
the field itself and every field with a cross term on it or a recipe that reads it).

Decoded arrays live in a process-wide cache keyed by sha256 (``_arrays``). Codecs that derive data from them and
cache it (``icon_lattice.details``: the spatial details of a static cross source) register a hook with
:func:`register_clear_hook`, so that :func:`clear_caches` empties those caches too.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numcodecs import Blosc

from . import sidecar

STATIC_CODEC = {"cname": "zstd", "clevel": 9, "shuffle": Blosc.BITSHUFFLE}
INLINE_CODEC = {"cname": "lz4", "clevel": 5, "shuffle": Blosc.SHUFFLE}
log = logging.getLogger(__name__)
MAX_VERSIONS = 4

_arrays: dict[str, np.ndarray] = {}
_durable: set[str] = set()  # shas available from a persistent source (found on the search path or persisted)
_logged: set[str] = set()
_placed: set[tuple[str, Path]] = set()  # (sha, sidecar directory) known to hold the file
_lock = threading.Lock()
_clear_hooks: list[Callable[[], None]] = []


def sha256_of(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, "<f4").tobytes()).hexdigest()


def static_file_bytes(x: np.ndarray) -> bytes:
    """Content of a sidecar static file (blosc zstd9 bit shuffle of the float32 bytes)."""
    return bytes(Blosc(**STATIC_CODEC).encode(np.ascontiguousarray(x, "<f4")))


def _decompress(data: bytes) -> np.ndarray | None:
    """float32 values of a Blosc frame, or None when the frame is damaged."""
    try:
        return np.frombuffer(Blosc().decode(data), "<f4").astype(np.float32, copy=False)
    except (RuntimeError, ValueError, TypeError):  # numcodecs' Blosc raises RuntimeError on a damaged frame
        return None


def read_static_file(path: Path, sha: str) -> np.ndarray:
    """The static array of a sidecar file; ``ValueError`` naming the file when its content is not ``sha``."""
    x = _decompress(path.read_bytes())
    if x is None or sha256_of(x) != sha:
        raise ValueError(f"static file {path} is corrupt (sha256 mismatch)")
    return x


def _cache(sha: str, x: np.ndarray) -> np.ndarray:
    y = np.array(x, np.float32) if x.flags.writeable or x.dtype != np.float32 else x
    y.setflags(write=False)
    return _arrays.setdefault(sha, y)


def find(sha: str) -> np.ndarray | None:
    """The static array with this sha256 from the process cache or the search path, or None."""
    x = _arrays.get(sha)
    if x is not None and sha in _durable:
        return x
    path = sidecar.find("statics", sha)
    if path is None:
        return x
    with _lock:
        y = _arrays.get(sha)
        if y is None:
            y = _cache(sha, read_static_file(path, sha))
        _durable.add(sha)
        return y


def lookup(sha: str, inline: bytes | memoryview | None = None) -> np.ndarray:
    """The static array (process cache, then inline bytes, then the search path). Raises
    ``sidecar.MissingReference`` (a ``LookupError`` and ``ValueError``)."""
    x = _arrays.get(sha)
    if x is not None:
        return x
    if inline is not None and len(inline):
        x = _decompress(bytes(inline))
        if x is None or sha256_of(x) != sha:
            raise ValueError("inline static field does not match its sha256")
        with _lock:
            return _cache(sha, x)
    x = find(sha)
    if x is None:
        raise sidecar.MissingReference(
            f"static field {sha[:12]} is not available; searched: {sidecar.describe_search()}. Make the store's "
            f"sidecar visible (ANEMOI_DATASETS_CODECS_PATH=<store>, or register_store(<store>)), or run "
            f"`anemoi-datasets-codecs attach <store>` where the encoder's cache is"
        )
    return x


def ensure_durable(name: str, sha: str, x: np.ndarray) -> bool:
    """Encoder: True when the static ``sha`` is (or has just been made) durably available. With a store target
    (``sidecar.store_targets``: a registered store or a writable ``$ANEMOI_DATASETS_CODECS_PATH`` entry) the file
    must be in that store's sidecar (written when missing), so that the store is self-contained; without one, it must
    be on the search path or is captured into the cache. False when it could not be persisted anywhere."""
    targets = sidecar.store_targets()
    if targets:
        if (sha, targets[0]) in _placed:
            return True
        dirs = targets
    else:
        if sha not in _durable:
            find(sha)  # marks it durable when it is on the search path
        if sha in _durable:
            return True
        dirs = sidecar.capture_dirs()
    data = None
    written = []
    for d in dirs:
        path = d / "statics" / sidecar.file_name("statics", sha)
        try:
            if not path.is_file():
                data = data or static_file_bytes(x)
                sidecar.write(d, "statics", sha, data)
                written.append(d)
        except OSError as e:
            warnings.warn(f"could not write static field {name!r} into {d}: {e}", stacklevel=3)
            continue
        with _lock:
            _placed.add((sha, d))
    if not any((sha, d) in _placed for d in dirs):
        return False
    with _lock:
        _cache(sha, x)
        _durable.add(sha)
    if written and sha not in _logged:
        _logged.add(sha)
        log.info("captured static field %r (sha256 %s) into %s", name, sha[:12], ", ".join(map(str, written)))
    return True


class StaticCapture:
    """The encoder's reference policy for one codec: a static field is coded as a reference only when it is durably
    available (:func:`ensure_durable`), and at most ``MAX_VERSIONS`` different arrays per variable name are
    captured; otherwise the codec stores it in the chunk (``fallback`` says how, for the warning)."""

    def __init__(self, fallback: str = "stored losslessly") -> None:
        self._versions: dict[str, set[str]] = {}
        self._fallback = fallback

    def ok(self, name: str, sha: str, x: np.ndarray) -> bool:
        """May the static field ``name`` (array ``x``, sha256 ``sha``) be coded as a reference?"""
        seen = self._versions.setdefault(name, set())
        if sha not in seen and len(seen) >= MAX_VERSIONS:
            many = f"static field {name!r} has more than {MAX_VERSIONS} different values across chunks"
            warnings.warn(f"{many}: {self._fallback}", stacklevel=3)
            return False
        if not ensure_durable(name, sha, x):
            warnings.warn(
                f"static field {name!r} (sha256 {sha[:12]}) is not in any sidecar and could not be written to one "
                f"(no writable store / cache): {self._fallback}",
                stacklevel=3,
            )
            return False
        seen.add(sha)
        return True


def add_static(x: np.ndarray, directory: str | Path | None = None) -> str:
    """Register a static array in this process and, when ``directory`` is given, write its sidecar file there.
    Returns its sha256."""
    sha = sha256_of(x)
    if directory is not None:
        sidecar.write(Path(directory), "statics", sha, static_file_bytes(x))
        with _lock:
            _durable.add(sha)
    with _lock:
        _cache(sha, np.asarray(x, np.float32))
    return sha


def static_bytes(sha: str) -> bytes | None:
    """The sidecar file content of a static field (from any source), or None."""
    path = sidecar.find("statics", sha)
    if path is not None:
        return path.read_bytes()
    x = _arrays.get(sha)
    return None if x is None else static_file_bytes(x)


def register_clear_hook(fn: Callable[[], None]) -> None:
    """Call ``fn`` from :func:`clear_caches` (a codec's cache of data derived from static arrays)."""
    with _lock:
        if fn not in _clear_hooks:
            _clear_hooks.append(fn)


def clear_caches() -> None:
    """Forget the static arrays of this process and what codecs derived from them (:func:`register_clear_hook`;
    tests: emulate a fresh reader)."""
    with _lock:
        _arrays.clear()
        _durable.clear()
        _logged.clear()
        _placed.clear()
        hooks = list(_clear_hooks)
    for fn in hooks:
        fn()


def _isolate():
    """``sidecar.isolated``: forget the static arrays (before and after: they were resolved on another path)."""
    clear_caches()
    return clear_caches


sidecar.register_cache(_isolate)


def inline_payload(x: np.ndarray) -> bytes:
    return bytes(Blosc(**INLINE_CODEC).encode(np.ascontiguousarray(x)))
