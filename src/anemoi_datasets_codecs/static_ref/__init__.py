"""``anemoi_datasets_codecs.static_ref``: the ``static_ref`` field stage, grid-agnostic and exact. It takes the static
fields out of a chunk and references them by sha256 in the shared sidecar (``anemoi_datasets_codecs.sidecar``), so
the codec after it (``icon_lattice_bytes`` with ``chain: true``, or any compressor) codes only the other fields.

``StaticRef`` is the engine (``zarr3.FieldStage``; no zarr); ``anemoi_datasets_codecs.static_ref.zarr3`` has the
Zarr 3 codec ``static_ref`` around it, registered through the ``zarr.codecs`` entry point. Header format:
``header.py``, docs/codecs/static_ref.md.
"""

from __future__ import annotations

from . import header
from .stage import StaticRef

__all__ = ["StaticRef", "header"]
