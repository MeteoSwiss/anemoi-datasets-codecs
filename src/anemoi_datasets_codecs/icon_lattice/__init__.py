"""``anemoi_datasets_codecs.icon_lattice``: the ICON lattice codec, error-bounded, for anemoi datasets on ICON
triangle-lattice grids.

``LatticeCodec`` is the engine (chunk array in, container bytes out; no zarr);
``anemoi_datasets_codecs.icon_lattice.zarr3`` has the Zarr 3 codecs around it (``icon_lattice``, array to bytes, and
``icon_lattice_bytes``, bytes to bytes), registered through the ``zarr.codecs`` entry point. This is the public API
of the codec.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .codec import LatticeCodec
from .config import CODEC_ID, Variable
from .container import MODE_LABELS, inspect_chunk
from .grids import known_grids, load_lattice, register_lattice
from .layout import TriangleLatticeLayout
from .sidecar import register_store, unregister_store
from .threads import default_threads

try:
    __version__ = version("anemoi-datasets-codecs")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

__all__ = [
    "CODEC_ID",
    "MODE_LABELS",
    "LatticeCodec",
    "TriangleLatticeLayout",
    "Variable",
    "__version__",
    "default_threads",
    "inspect_chunk",
    "known_grids",
    "load_lattice",
    "register_lattice",
    "register_store",
    "unregister_store",
]
