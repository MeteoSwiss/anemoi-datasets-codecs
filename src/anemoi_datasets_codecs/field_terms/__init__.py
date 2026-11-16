"""The ``field_terms`` field stage: lossy targets replaced by residuals against a per-chunk robust linear fit on exact
source fields, ahead of an error-bounded codec in a chain (docs/codecs/field_terms.md).

Grid-agnostic: it needs no lattice and no grid, only the field array of a chunk ``(dates, fields, ensembles, cells)``.

* :class:`FieldTerms`: the engine (``encode(x) -> (residual array, header)``, ``decode`` in place, ``get_config``);
* :class:`FieldTermsCodec`: the Zarr 3 codec ``field_terms`` (bytes to bytes, a ``zarr3.FieldStageBytesCodec``);
* ``header``: the frozen per-chunk header format; ``arith``: the fit, prediction and verification arithmetic.
"""

from __future__ import annotations

from . import arith, header
from .stage import CODEC_ID, DEFAULT_MARGIN, FieldTerms, Term, reduced_bound
from .zarr3 import FieldTermsCodec

__all__ = [
    "CODEC_ID",
    "DEFAULT_MARGIN",
    "FieldTerms",
    "FieldTermsCodec",
    "Term",
    "arith",
    "header",
    "reduced_bound",
]
