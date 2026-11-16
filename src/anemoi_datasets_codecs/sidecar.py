"""Where the codecs find what a chunk only references: lattices (the grid's cell positions, by sha256) and static
fields (by the sha256 of their float32 bytes). Shared by every codec of the package (``icon_lattice``,
``static_ref``): one sidecar directory, one search path, one set of file formats. See docs/design.md "Sidecars and
content addressing" for the design.

A *sidecar directory* holds content-addressed files::

    grids/<sha256>.npz          lattice file (layout.TriangleLatticeLayout.to_bytes)
    statics/<sha256>.f4.blosc   float32 static field, numcodecs Blosc (zstd 9, bit shuffle)

and lives inside a store as ``<store>/.codecs/`` (zarr ignores it: no ``zarr.json``; it is copied with the store
directory), in the user cache, or anywhere on ``$ANEMOI_DATASETS_CODECS_PATH``. The directory name and the file
formats are format-level and frozen; every codec uses the same ones.

The store a codec serves is known while zarr runs it: the codec pipeline (``zarr3.SidecarPipeline``) learns the store
from zarr and sets it for the duration of every read and write (:func:`serving`, a context variable, so concurrent
arrays of different stores do not mix). That store's sidecar comes first in the search order and is the capture
target, so a local store needs no registration to be written self-contained or to be read.

Search order (:func:`search_dirs`): the sidecar of the store being served, stores registered in this process
(:func:`register_store`), then the entries of ``$ANEMOI_DATASETS_CODECS_PATH`` (``os.pathsep`` separated; a store
root or a sidecar directory), then the cache directory (``$ANEMOI_DATASETS_CODECS_CACHE``, default
``~/.cache/anemoi_datasets_codecs``; ``off`` disables it), then the directories bundled with codec packages
(:func:`register_bundled`: the lattices of ``icon_lattice``). Every file is verified against its sha256 when read, so
any source is safe.

Capture targets (:func:`capture_dirs`, where the encoder persists a static field or lattice it had to capture from
the data): the store being served, else the first registered store or ``$ANEMOI_DATASETS_CODECS_PATH`` entry whose
directory exists and is writable, plus the cache directory.

:func:`isolated` resolves references from one store's sidecar alone (``attach`` / ``check`` verify a store with it);
the modules that keep resolved references in process caches (lattices, static arrays) register a hook with
:func:`register_cache` so that it can empty and restore them.
"""

from __future__ import annotations

import contextvars
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

SIDECAR = ".codecs"
PATH_ENV = "ANEMOI_DATASETS_CODECS_PATH"
CACHE_ENV = "ANEMOI_DATASETS_CODECS_CACHE"
KINDS = {"grids": ".npz", "statics": ".f4.blosc"}


class MissingReference(LookupError, ValueError):
    """A lattice or static field a chunk references is not on the sidecar search path."""


_registered: list[Path] = []
_bundled: list[Path] = []
_serving: contextvars.ContextVar[Path | None] = contextvars.ContextVar("anemoi_datasets_codecs_serving", default=None)
_lock = threading.Lock()
_caches: list[Callable[[], Callable[[], None]]] = []


def cache_dir() -> Path | None:
    """``$ANEMOI_DATASETS_CODECS_CACHE`` or ``~/.cache/anemoi_datasets_codecs`` (None when set to off / none / 0)."""
    value = os.environ.get(CACHE_ENV, "").strip()
    if value.lower() in ("off", "none", "0", "false"):
        return None
    if value:
        return Path(value).expanduser()
    return Path.home() / ".cache" / "anemoi_datasets_codecs"


def register_bundled(directory: str | os.PathLike[str]) -> None:
    """Add a sidecar directory shipped with a codec package (read-only, last in the search order)."""
    d = Path(directory)
    with _lock:
        if d not in _bundled:
            _bundled.append(d)


def bundled_dirs() -> list[Path]:
    """The sidecar directories shipped with codec packages (:func:`register_bundled`), in registration order."""
    with _lock:
        return list(_bundled)


def is_store(path: Path) -> bool:
    """A zarr store root (``zarr.json``) or a directory that already holds a sidecar."""
    return any((path / name).exists() for name in ("zarr.json", SIDECAR))


def sidecar_of(path: str | os.PathLike[str]) -> Path:
    """The sidecar directory for ``path``: ``<path>/.codecs`` for a store root, else ``path`` itself."""
    p = Path(path).expanduser()
    return p / SIDECAR if is_store(p) else p


def serving_dir() -> Path | None:
    """The sidecar of the store the current encode / decode serves (set by the zarr pipeline), or None."""
    return _serving.get()


@contextmanager
def serving(sidecar_dir: Path | None) -> Iterator[None]:
    """Serve the store whose sidecar is ``sidecar_dir`` (first in the search order, the capture target) in this
    context: tasks and threads started inside inherit it (a context variable). None serves no store."""
    token = _serving.set(sidecar_dir)
    try:
        yield
    finally:
        _serving.reset(token)


def register_store(path: str | os.PathLike[str]) -> Path:
    """Search (and capture into) the sidecar of the store at ``path`` (or a sidecar directory) in this process.
    Returns the sidecar directory. Registered stores come first in the search order, most recent first."""
    sc = sidecar_of(path).resolve()
    with _lock:
        if sc in _registered:
            _registered.remove(sc)
        _registered.insert(0, sc)
    return sc


def unregister_store(path: str | os.PathLike[str]) -> None:
    sc = sidecar_of(path).resolve()
    with _lock:
        if sc in _registered:
            _registered.remove(sc)


def env_dirs() -> list[Path]:
    value = os.environ.get(PATH_ENV, "")
    return [sidecar_of(p) for p in value.split(os.pathsep) if p.strip()]


def search_dirs() -> list[Path]:
    """Sidecar directories in search order (see the module docstring)."""
    served = _serving.get()
    out = [served] if served is not None else []
    with _lock:
        out += _registered
    out += env_dirs()
    c = cache_dir()
    if c is not None:
        out.append(c)
    out += bundled_dirs()
    seen: set[Path] = set()
    return [d for d in out if not (d in seen or seen.add(d))]


def _writable(sc: Path) -> bool:
    if sc.is_dir():
        return os.access(sc, os.W_OK)
    parent = sc.parent
    return sc.name == SIDECAR and parent.is_dir() and os.access(parent, os.W_OK)


def store_targets() -> list[Path]:
    """The store sidecar the encoder writes captured data into: the store being served, else the first writable
    registered store or ``$ANEMOI_DATASETS_CODECS_PATH`` entry (at most one)."""
    served = _serving.get()
    candidates = [served] if served is not None else []
    with _lock:
        candidates += _registered
    candidates += env_dirs()
    return [sc for sc in candidates if _writable(sc)][:1]


def capture_dirs() -> list[Path]:
    """Where the encoder persists what it captures: the store target (if any), then the cache directory (when
    enabled)."""
    out = store_targets()
    c = cache_dir()
    if c is not None and c not in out:
        out.append(c)
    return out


def file_name(kind: str, sha: str) -> str:
    if kind not in KINDS or not sha or any(ch not in "0123456789abcdef" for ch in sha):
        raise ValueError(f"bad sidecar reference {kind}/{sha!r}")
    return f"{sha}{KINDS[kind]}"


def find(kind: str, sha: str) -> Path | None:
    """First sidecar file ``kind`` / ``sha`` on the search path, or None."""
    name = file_name(kind, sha)
    for d in search_dirs():
        p = d / kind / name
        if p.is_file():
            return p
    return None


def write(directory: Path, kind: str, sha: str, data: bytes) -> Path:
    """Write a sidecar file atomically (no-op when it exists). Returns its path; raises ``OSError``."""
    path = directory / kind / file_name(kind, sha)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}.{threading.get_ident()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def describe_search() -> str:
    """The search path, for error messages."""
    return ", ".join(str(d) for d in search_dirs()) or "(empty)"


def register_cache(isolate: Callable[[], Callable[[], None]]) -> None:
    """Register a process cache of resolved references for :func:`isolated`: ``isolate()`` empties the cache and
    returns the function that restores it."""
    with _lock:
        _caches.append(isolate)


@contextmanager
def isolated(store: str | os.PathLike[str]) -> Iterator[None]:
    """Resolve lattices and statics from the sidecar of ``store`` alone: no cache directory, no other path entries,
    no registered or served stores, and empty process caches (restored afterwards). Not thread-safe: it changes the
    environment of the process."""
    saved = {k: os.environ.get(k) for k in (CACHE_ENV, PATH_ENV)}
    token = _serving.set(None)
    with _lock:
        registered = list(_registered)
        _registered.clear()
        hooks = list(_caches)
    os.environ[CACHE_ENV] = "off"
    os.environ[PATH_ENV] = str(store)
    restore = [isolate() for isolate in hooks]
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        with _lock:
            _registered[:] = registered
        _serving.reset(token)
        for r in restore:
            r()
