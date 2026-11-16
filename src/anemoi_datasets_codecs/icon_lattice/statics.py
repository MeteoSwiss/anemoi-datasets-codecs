"""Static fields of the lattice codec. The references, the process cache and the capture are the package-level
``anemoi_datasets_codecs.statics`` (shared with the ``static_ref`` stage), whose public names this module re-exports.
What it adds: the spatial details of static cross sources (``details.py``) are cleared with the static arrays
(:func:`clear_caches`). The lattice codec stores a static field as STATIC_REF (``container.py``), or losslessly when it
cannot be referenced.

Forcing recipes (mode RECIPE) are ``recipes.py`` (format) and ``physics/forcings.py`` (which variables).
"""

from __future__ import annotations

from ..statics import (
    INLINE_CODEC,
    MAX_VERSIONS,
    STATIC_CODEC,
    StaticCapture,
    add_static,
    clear_caches,
    ensure_durable,
    find,
    inline_payload,
    lookup,
    read_static_file,
    register_clear_hook,
    sha256_of,
    static_bytes,
    static_file_bytes,
)
from . import details

__all__ = [
    "INLINE_CODEC",
    "MAX_VERSIONS",
    "STATIC_CODEC",
    "StaticCapture",
    "add_static",
    "clear_caches",
    "ensure_durable",
    "find",
    "inline_payload",
    "lookup",
    "read_static_file",
    "register_clear_hook",
    "sha256_of",
    "static_bytes",
    "static_file_bytes",
]

register_clear_hook(details.clear_cache)
