"""The sidecar search path of the lattice codec. The search path is the package-level ``anemoi_datasets_codecs.sidecar``
(shared by all codecs), whose public names this module re-exports. What it adds: the lattices bundled with
``icon_lattice`` (:func:`bundled_dir`), registered last in the search order when the lattice codec is imported."""

from __future__ import annotations

from pathlib import Path

from ..sidecar import (
    CACHE_ENV,
    KINDS,
    PATH_ENV,
    SIDECAR,
    MissingReference,
    bundled_dirs,
    cache_dir,
    capture_dirs,
    describe_search,
    env_dirs,
    file_name,
    find,
    is_store,
    isolated,
    register_bundled,
    register_cache,
    register_store,
    search_dirs,
    serving,
    serving_dir,
    sidecar_of,
    store_targets,
    unregister_store,
    write,
)

__all__ = [
    "CACHE_ENV",
    "KINDS",
    "PATH_ENV",
    "SIDECAR",
    "MissingReference",
    "bundled_dir",
    "bundled_dirs",
    "cache_dir",
    "capture_dirs",
    "describe_search",
    "env_dirs",
    "file_name",
    "find",
    "is_store",
    "isolated",
    "register_bundled",
    "register_cache",
    "register_store",
    "search_dirs",
    "serving",
    "serving_dir",
    "sidecar_of",
    "store_targets",
    "unregister_store",
    "write",
]


def bundled_dir() -> Path:
    """The lattice codec's data directory (lattices of known grids, ``grids/index.json``)."""
    return Path(__file__).resolve().parent / "data"


register_bundled(bundled_dir())
