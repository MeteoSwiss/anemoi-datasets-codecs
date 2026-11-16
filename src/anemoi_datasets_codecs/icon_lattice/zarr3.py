"""The Zarr 3 codecs of the ICON lattice codec, registered through the ``zarr.codecs`` entry point.

* ``icon_lattice`` (:class:`IconLatticeCodec`, array to bytes): the natural form. The codec maps the float32 chunk
  array straight to the container bytes, so it *replaces* the ``bytes`` codec::

      zarr.create_array(store, shape=..., chunks=(1, n_vars, n_ens, n_cells), dtype="float32",
                        serializer=IconLatticeCodec(**config), compressors=None)

* ``icon_lattice_bytes`` (:class:`IconLatticeBytesCodec`, bytes to bytes): the same codec after the default
  ``bytes`` codec (little endian). It exists because anemoi-datasets' ``output.compressors`` accepts bytes-to-bytes
  codecs only. The stored chunks are identical.

Both are the generic adapters of ``anemoi_datasets_codecs.zarr3`` around the engine
:class:`~anemoi_datasets_codecs.icon_lattice.LatticeCodec`: the ``configuration`` in ``zarr.json`` is
``LatticeCodec.get_config()`` (small JSON), the chunk bytes are the container (format v3,
docs/codecs/icon_lattice.md). Encoding and decoding run in a worker thread (the C decoder releases the GIL), so
zarr's concurrent chunk reads decode in parallel; each decode also uses ``threads.default_threads()`` OpenMP
threads. With ``chain: true`` in the configuration, ``icon_lattice_bytes`` follows field stages in a codec chain: it
codes the field array inside the stages' envelope and passes their headers through (``anemoi_datasets_codecs.zarr3``,
docs/codecs/envelope.md); a bare input still gives the bare container.
"""

from __future__ import annotations

from ..zarr3 import ChunkArrayBytesCodec, ChunkBytesBytesCodec
from .codec import LatticeCodec

__all__ = ["IconLatticeBytesCodec", "IconLatticeCodec"]


class IconLatticeCodec(ChunkArrayBytesCodec):
    """Zarr 3 array-to-bytes codec ``icon_lattice`` (takes the place of the ``bytes`` codec)."""

    codec_name = "icon_lattice"
    engine = LatticeCodec


class IconLatticeBytesCodec(ChunkBytesBytesCodec):
    """Zarr 3 bytes-to-bytes codec ``icon_lattice_bytes``: follows the ``bytes`` codec, which must write
    little-endian float32 (its default)."""

    codec_name = "icon_lattice_bytes"
    engine = LatticeCodec
