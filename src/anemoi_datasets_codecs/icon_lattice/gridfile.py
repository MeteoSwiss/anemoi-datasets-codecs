"""Encoder side: the lattice of an ICON grid file (netCDF4 or h5py, and scipy; extra ``grid``).

:func:`lattice_from_grid_file` recovers the lattice from the cell centres
(``layout.TriangleLatticeLayout.from_latlon``) and :func:`check_against_grid_file` validates a lattice against the
file's neighbour table (every edge neighbour is a lattice neighbour). ``anemoi-datasets-codecs grid build`` writes the
lattice file of a new grid (``grids.save_lattice``); readers never need the grid file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .layout import TriangleLatticeLayout


def read_grid_file(path: str | Path) -> dict[str, Any]:
    """Cell centres (degrees), neighbour table (0-based, -1 = none) and attributes of an ICON grid file."""
    path = str(path)
    try:
        import netCDF4

        with netCDF4.Dataset(path) as ds:
            clat = np.degrees(np.asarray(ds.variables["clat"][:], np.float64))
            clon = np.degrees(np.asarray(ds.variables["clon"][:], np.float64))
            nb = np.asarray(ds.variables["neighbor_cell_index"][:], np.int64)
            attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
    except ImportError:
        try:
            import h5py
        except ImportError:
            raise ImportError(
                f"reading the ICON grid file {path} needs netCDF4 (or h5py): install the package's [grid] extra"
            ) from None

        with h5py.File(path, "r") as ds:
            clat = np.degrees(np.asarray(ds["clat"][:], np.float64))
            clon = np.degrees(np.asarray(ds["clon"][:], np.float64))
            nb = np.asarray(ds["neighbor_cell_index"][:], np.int64)
            attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in ds.attrs.items()}
    nb = nb.T if nb.shape[0] == 3 else nb
    return {"lat": clat, "lon": clon, "neighbours": np.where(nb > 0, nb - 1, -1), "attrs": attrs}


def lattice_from_grid_file(path: str | Path, name: str | None = None, check: bool = True) -> TriangleLatticeLayout:
    """Recover the lattice of an ICON grid file (cells in file order) and, with ``check``, validate it against the
    file's neighbour table. Encoder-side tool (needs scipy and netCDF4 or h5py: the package's [grid] extra)."""
    g = read_grid_file(path)
    try:
        import scipy  # noqa: F401
    except ImportError:
        raise ImportError("building a lattice needs scipy: install the package's [grid] extra") from None
    lay = TriangleLatticeLayout.from_latlon(g["lat"], g["lon"])
    uuid = str(g["attrs"].get("uuidOfHGrid", "")) or None
    lay.meta.update({k: v for k, v in {"name": name, "uuid": uuid, "source": Path(path).name}.items() if v})
    if check:
        check_against_grid_file(lay, g)
    return lay


def lattice_neighbours(layout: TriangleLatticeLayout) -> np.ndarray:
    """``(n_cells, 3)`` edge neighbours of every cell on the lattice (-1 outside the domain): left, right and the
    vertical neighbour (row - 1 for (row + col) even, row + 1 otherwise)."""
    h, w = layout.shape
    r, c = layout.rows_cols()
    img = np.full(h * w, -1, np.int64)
    img[layout.pixel_index] = np.arange(layout.n_cells)
    img = img.reshape(h, w)
    vert = np.where((r + c) % 2 == 0, r - 1, r + 1)
    out = []
    for rr, cc in ((r, c - 1), (r, c + 1), (vert, c)):
        ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        x = np.full(len(r), -1, np.int64)
        x[ok] = img[rr[ok], cc[ok]]
        out.append(x)
    return np.stack(out, 1)


def check_against_grid_file(layout: TriangleLatticeLayout, grid: str | Path | dict[str, Any]) -> None:
    """Raise ``ValueError`` unless the lattice's edge neighbours are exactly the grid file's neighbour table."""
    g = read_grid_file(grid) if not isinstance(grid, dict) else grid
    nb = g["neighbours"]
    if len(nb) != layout.n_cells:
        raise ValueError(f"grid file has {len(nb)} cells, the lattice {layout.n_cells}")
    a = np.sort(lattice_neighbours(layout), 1)
    b = np.sort(np.where(nb >= 0, nb, -1), 1)
    bad = int(np.any(a != b, 1).sum())
    if bad:
        raise ValueError(f"lattice disagrees with the grid file's neighbour table on {bad} cells")
