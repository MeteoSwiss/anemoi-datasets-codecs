"""The Zarr 3 codec of the ``field_terms`` stage, registered through the ``zarr.codecs`` entry point.

``field_terms`` (:class:`FieldTermsCodec`, bytes to bytes) follows the default ``bytes`` codec and comes first in a
chain, ahead of the codec that codes the residuals::

    zarr.create_array(store, shape=..., chunks=(1, n_vars, n_ens, n_cells), dtype="float32",
                      compressors=[FieldTermsCodec(**stage_config), IconLatticeBytesCodec(**lattice_config)])

with ``chain: true`` in the lattice configuration and the stage targets' bounds there at most
``FieldTerms.lattice_bounds`` (docs/codecs/field_terms.md). It is the generic field-stage adapter of
``anemoi_datasets_codecs.zarr3`` around the engine :class:`~anemoi_datasets_codecs.field_terms.FieldTerms`; its
header travels in the chain envelope under the id ``field_terms``.
"""

from __future__ import annotations

from ..zarr3 import FieldStageBytesCodec
from .stage import FieldTerms

__all__ = ["FieldTermsCodec"]


class FieldTermsCodec(FieldStageBytesCodec):
    """Zarr 3 bytes-to-bytes codec ``field_terms`` (a field stage; decodes in place)."""

    codec_name = "field_terms"
    engine = FieldTerms
