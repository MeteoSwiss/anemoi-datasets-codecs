"""``TriangleLatticeLayout``: the position of every ICON cell on the triangle lattice (a cut-out of a regular
triangular lattice, as ICON's limited-area grids are), as a pixel index ``row * W + col`` of an ``(H, W)`` image.

The codec's prediction structures are a pure function of this index (``structure.py``). Readers get it from a
lattice file (``grids.py``: bundled with the package, in a store's sidecar, or in the cache), never from the grid
file; :meth:`TriangleLatticeLayout.from_latlon` (scipy) and ``gridfile.lattice_from_grid_file`` build it on the
encoder side.

File format (``.npz``, numpy only): ``pixel_delta`` (int32, ``diff`` of the pixel index with a leading 0, so that
the zlib of ``np.savez_compressed`` sees small numbers), ``shape`` (int64 ``(2,)``), ``meta`` (JSON string: name,
uuid, n_cells, sha256, orientation, source).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from .lattice import LayoutError, recover_lattice

MIN_SIDE = 32
MAX_SIDE = 2047


def check_shape(shape: tuple[int, int]) -> tuple[int, int]:
    """Return ``shape`` as ints; raise ``LayoutError`` unless both sides are in [32, 2047]."""
    h, w = int(shape[0]), int(shape[1])
    if not (MIN_SIDE <= h <= MAX_SIDE and MIN_SIDE <= w <= MAX_SIDE):
        raise LayoutError(f"image {h} x {w}: both sides must be in [{MIN_SIDE}, {MAX_SIDE}] (tiling not supported)")
    return h, w


def pixel_index_sha256(pixel_index: np.ndarray) -> str:
    """sha256 of the raw int32 little-endian pixel index (the lattice's content address)."""
    return hashlib.sha256(np.ascontiguousarray(pixel_index, "<i4").tobytes()).hexdigest()


class TriangleLatticeLayout:
    """Triangle-strip image of a lattice cut-out.

    Stores the pixel index (``row * W + col``) of every cell, in the native cell order of the dataset. Images
    smaller than 32 in a side are padded up to 32 (only relevant for tiny test grids).
    """

    kind = "triangle_lattice"

    def __init__(
        self,
        pixel_index: np.ndarray,
        shape: tuple[int, int],
        orientation: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            pixel_index: int32 ``(n_cells,)`` flat pixel index of each cell (unique, within ``H*W``).
            shape: ``(H, W)`` of the image.
            orientation: The lattice alpha used to build it (informative only).
            meta: Informative metadata (name, uuid, source), kept in lattice files.
        """
        self.shape = check_shape(shape)
        self.pixel_index = np.ascontiguousarray(pixel_index, np.int32)
        self.n_cells = len(self.pixel_index)
        self.orientation = orientation
        self.meta = dict(meta or {})
        size = self.shape[0] * self.shape[1]
        if self.pixel_index.ndim != 1 or self.n_cells == 0:
            raise LayoutError("pixel_index must be a non-empty 1D array")
        if self.pixel_index.min() < 0 or self.pixel_index.max() >= size:
            raise LayoutError(f"pixel_index out of range for image {self.shape}")
        if len(np.unique(self.pixel_index)) != self.n_cells:
            raise LayoutError("pixel_index is not unique")
        self.pixel_index.setflags(write=False)
        self._sha256: str | None = None

    @classmethod
    def from_latlon(cls, lat: np.ndarray, lon: np.ndarray, orientation: str | int = "auto") -> TriangleLatticeLayout:
        """Recover the lattice from the cell centres (degrees) with ``lattice.recover_lattice`` (scipy). Raises
        ``LayoutError`` when the cells are not a triangular-lattice cut-out."""
        res = recover_lattice(lat, lon, orientation)
        h, w = max(res.shape[0], MIN_SIDE), max(res.shape[1], MIN_SIDE)
        return cls((res.row * w + res.col).astype(np.int32), (h, w), res.orientation)

    @property
    def sha256(self) -> str:
        """sha256 of the raw int32 little-endian pixel index."""
        if self._sha256 is None:
            self._sha256 = pixel_index_sha256(self.pixel_index)
        return self._sha256

    def rows_cols(self) -> tuple[np.ndarray, np.ndarray]:
        """Lattice row and column of every cell."""
        pix = self.pixel_index.astype(np.int64)
        return pix // self.shape[1], pix % self.shape[1]

    # ------------------------------------------------------------------------------------------- lattice files
    def ref(self) -> dict[str, Any]:
        """The grid reference of the codec configuration (small JSON)."""
        out = {k: self.meta[k] for k in ("name", "uuid") if self.meta.get(k)}
        return {**out, "n_cells": self.n_cells, "sha256": self.sha256}

    def to_bytes(self) -> bytes:
        """The lattice file (``.npz``, see the module docstring)."""
        meta = {**self.meta, "n_cells": self.n_cells, "sha256": self.sha256, "orientation": self.orientation}
        buf = io.BytesIO()
        delta = np.diff(self.pixel_index.astype(np.int64), prepend=0).astype("<i4")
        np.savez_compressed(buf, pixel_delta=delta, shape=np.array(self.shape, "<i8"), meta=np.array(json.dumps(meta)))
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes, sha256: str | None = None) -> TriangleLatticeLayout:
        """Inverse of :meth:`to_bytes`; checks the content against its sha256 (and ``sha256`` when given). Raises
        ``ValueError`` (``sha256 mismatch``) on a damaged file."""
        try:
            with np.load(io.BytesIO(data), allow_pickle=False) as z:
                pix = np.cumsum(z["pixel_delta"].astype(np.int64)).astype(np.int32)
                shape = tuple(int(v) for v in z["shape"])
                meta = json.loads(str(z["meta"]))
            if not isinstance(meta, dict):
                raise ValueError("meta is not a mapping")
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, zlib.error) as e:
            raise ValueError(f"lattice file is corrupt (sha256 mismatch: not a readable lattice file, {e})") from None
        got = pixel_index_sha256(pix)
        for want in (meta.get("sha256"), sha256):
            if want is not None and got != want:
                raise ValueError(
                    f"lattice file is corrupt or not the requested one (sha256 mismatch: {got[:12]} != {want[:12]})"
                )
        orientation = meta.pop("orientation", None)
        layout = cls(pix, shape, orientation, {k: v for k, v in meta.items() if k not in ("n_cells", "sha256")})
        layout._sha256 = got
        return layout

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the lattice file atomically. Returns its path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
        tmp.write_bytes(self.to_bytes())
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: str | os.PathLike[str], sha256: str | None = None) -> TriangleLatticeLayout:
        try:
            return cls.from_bytes(Path(path).read_bytes(), sha256)
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from None

    def __repr__(self) -> str:
        name = self.meta.get("name") or self.meta.get("uuid") or self.sha256[:12]
        return f"TriangleLatticeLayout({name}, n_cells={self.n_cells}, shape={self.shape})"
