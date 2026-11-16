"""anemoi-datasets-codecs: Zarr 3 codecs for anemoi datasets, lossy with a per-variable error bound or exact.

Codecs (one subpackage each; the codec names are format-level and do not change with the package name):

* ``anemoi_datasets_codecs.icon_lattice``: the ICON lattice codec, Zarr 3 codecs ``icon_lattice`` (array to bytes)
  and ``icon_lattice_bytes`` (bytes to bytes) around the engine ``LatticeCodec``.
* ``anemoi_datasets_codecs.static_ref``: the field stage ``static_ref`` (bytes to bytes, ahead of a codec in a chain)
  around the engine ``StaticRef``: static fields out of the chunk, referenced in the shared sidecar.
* ``anemoi_datasets_codecs.field_terms``: the field stage ``field_terms`` (bytes to bytes, first in a chain) around the
  engine ``FieldTerms``: lossy fields replaced by residuals against a per-chunk fit on exact fields.

A store is written with the lattice codec alone or with the chain ``[field_terms, static_ref, icon_lattice_bytes]``;
``anemoi_datasets_codecs.physics`` (``physics.expand``, ``anemoi-datasets-codecs physics expand``) writes the chain's
three configurations from a dataset's variables and a lattice codec configuration. The physics layer is the only part of
the package that knows variable names; ``anemoi_datasets_codecs.config`` holds the bounds and clip matching the codecs
share.

A codec subpackage provides an engine (chunk array in, bytes out; no zarr) and subclasses the generic adapters of
``anemoi_datasets_codecs.zarr3``; the codecs are registered through the ``zarr.codecs`` entry point. Field stages
(``zarr3.FieldStage``) run ahead of a codec in a chain of bytes-to-bytes codecs and pass their per-chunk metadata on in
the in-band envelope (``anemoi_datasets_codecs.envelope``).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("anemoi-datasets-codecs")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

CODECS = ("icon_lattice", "static_ref", "field_terms")

__all__ = ["CODECS", "__version__"]
