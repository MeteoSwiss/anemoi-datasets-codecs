"""The Zarr 3 codec of the ``static_ref`` stage, registered through the ``zarr.codecs`` entry point.

``static_ref`` (:class:`StaticRefCodec`, bytes to bytes): the generic field-stage adapter
(``anemoi_datasets_codecs.zarr3.FieldStageBytesCodec``) around :class:`~anemoi_datasets_codecs.static_ref.StaticRef`.
It follows the ``bytes`` codec, ahead of the codec that codes the remaining fields::

    zarr.create_array(store, shape=..., chunks=(1, n_vars, n_ens, n_cells), dtype="float32",
                      compressors=[StaticRefCodec(variables=..., static_fields=...),
                                   IconLatticeBytesCodec(**lattice_config, chain=True)])

The ``configuration`` in ``zarr.json`` is ``StaticRef.get_config()``; the stage's header in the chain envelope is
docs/codecs/static_ref.md.
"""

from __future__ import annotations

from ..zarr3 import FieldStageBytesCodec
from .stage import StaticRef

__all__ = ["StaticRefCodec"]


class StaticRefCodec(FieldStageBytesCodec):
    """Zarr 3 bytes-to-bytes codec ``static_ref``: a field stage that drops the static fields of the chunk and
    references them in the sidecar (or stores them inline in its header)."""

    codec_name = "static_ref"
    engine = StaticRef
