"""Grids: the codec configuration names a grid; readers resolve it to a lattice without the ICON grid file.

A grid reference (``LatticeCodec(grid=...)``, stored in the codec configuration) is small JSON::

    {"name": "icon-ch1", "uuid": "17643da2-...", "n_cells": 1147980, "sha256": "<sha256 of the pixel index>"}

``sha256`` is the content address of the lattice (``layout.TriangleLatticeLayout``): the reader loads the lattice
file ``grids/<sha256>.npz`` from the sidecar search path (``sidecar.py``: registered stores,
``$ANEMOI_DATASETS_CODECS_PATH``, the cache, and the lattices bundled with this package), verifies it, and builds the
prediction structure from it (``structure.py``, cached on disk). Writers may pass an alias of a bundled grid
(``"icon-ch1"``) or its uuid instead of the full reference.

This module is the reader side (numpy only). Building the lattice of a new grid from its ICON grid file is the
encoder-side ``gridfile.py`` (``anemoi-datasets-codecs grid build``).
"""

from __future__ import annotations

import functools
import json
import threading
import warnings
from pathlib import Path
from typing import Any

from . import sidecar
from .layout import TriangleLatticeLayout

_lattices: dict[str, TriangleLatticeLayout] = {}
_lock = threading.Lock()


def _isolate():
    """``sidecar.isolated``: empty the lattice cache; afterwards add back what it held before."""
    with _lock:
        saved = dict(_lattices)
        _lattices.clear()

    def restore() -> None:
        with _lock:
            _lattices.update(saved)

    return restore


sidecar.register_cache(_isolate)


@functools.lru_cache(maxsize=1)
def known_grids() -> dict[str, dict[str, Any]]:
    """Grids bundled with the package: alias -> reference (``data/grids/index.json``)."""
    path = sidecar.bundled_dir() / "grids" / "index.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def grid_ref(grid: Any) -> dict[str, Any]:
    """Normalise ``grid`` (alias, uuid, reference dict or ``TriangleLatticeLayout``) to a reference dict. A layout
    is registered in this process (and its reference returned); nothing is loaded otherwise."""
    if isinstance(grid, TriangleLatticeLayout):
        register_lattice(grid)
        return grid.ref()
    if isinstance(grid, str):
        known = known_grids()
        if grid in known:
            return dict(known[grid])
        for ref in known.values():
            if ref.get("uuid") == grid:
                return dict(ref)
        raise ValueError(f"unknown grid {grid!r}: known grids are {sorted(known)} (or give a reference dict)")
    if isinstance(grid, dict):
        ref = dict(grid)
        if "sha256" not in ref:
            base = None
            if ref.get("name") in known_grids():
                base = known_grids()[ref["name"]]
            else:
                base = next(
                    (r for r in known_grids().values() if ref.get("uuid") and r.get("uuid") == ref["uuid"]), None
                )
            if base is None:
                raise ValueError(f"grid reference {ref} has no sha256 and is not a known grid")
            ref = {**base, **ref}
        if not isinstance(ref.get("sha256"), str) or len(ref["sha256"]) != 64:
            raise ValueError(f"grid reference needs the 64-hex sha256 of its lattice, got {ref.get('sha256')!r}")
        try:
            ref["n_cells"] = int(ref["n_cells"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"grid reference needs n_cells, got {grid!r}") from None
        return ref
    raise TypeError(f"grid must be a name, a reference dict or a TriangleLatticeLayout, got {type(grid).__name__}")


def register_lattice(layout: TriangleLatticeLayout) -> dict[str, Any]:
    """Make a lattice available to this process (e.g. one built from a grid file). Returns its reference."""
    with _lock:
        _lattices.setdefault(layout.sha256, layout)
    return layout.ref()


def save_lattice(layout: TriangleLatticeLayout, directory: str | Path | None = None) -> Path:
    """Write the lattice file into a sidecar directory (default: the cache directory). Returns the path."""
    d = Path(directory) if directory is not None else sidecar.cache_dir()
    if d is None:
        raise ValueError("no directory given and the cache is disabled")
    return sidecar.write(Path(d), "grids", layout.sha256, layout.to_bytes())


def lattice_bytes(sha: str) -> bytes | None:
    """The lattice file content for ``sha`` (process cache, then the search path), or None."""
    lay = _lattices.get(sha)
    if lay is not None:
        return lay.to_bytes()
    path = sidecar.find("grids", sha)
    return None if path is None else path.read_bytes()


def load_lattice(ref: dict[str, Any] | str | TriangleLatticeLayout) -> TriangleLatticeLayout:
    """The lattice of a grid reference (process cache, then the sidecar search path). Raises
    ``sidecar.MissingReference`` with the searched directories when it is not found."""
    ref = grid_ref(ref)
    sha = ref["sha256"]
    lay = _lattices.get(sha)
    if lay is not None:
        return lay
    with _lock:
        lay = _lattices.get(sha)
        if lay is not None:
            return lay
        path = sidecar.find("grids", sha)
        if path is None:
            name = ref.get("name") or ref.get("uuid") or sha[:12]
            raise sidecar.MissingReference(
                f"lattice of grid {name!r} (sha256 {sha[:12]}) not found; searched: {sidecar.describe_search()}. "
                f"Add the store's sidecar (ANEMOI_DATASETS_CODECS_PATH=<store> or register_store), "
                f"or build the lattice from the grid file: anemoi-datasets-codecs grid build <grid.nc>"
            )
        lay = TriangleLatticeLayout.load(path, sha)
        if lay.n_cells != ref["n_cells"]:
            raise ValueError(f"lattice {sha[:12]} has {lay.n_cells} cells, the grid reference says {ref['n_cells']}")
        _lattices[sha] = lay
        return lay


def persist_lattice(layout: TriangleLatticeLayout) -> None:
    """Encoder side: make sure the lattice is in the capture targets (the store's sidecar when registered), so
    that the written data is readable without this process. Best effort (warns)."""
    data = None
    for d in sidecar.capture_dirs():
        try:
            if not (d / "grids" / sidecar.file_name("grids", layout.sha256)).is_file():
                data = data or layout.to_bytes()
                sidecar.write(d, "grids", layout.sha256, data)
        except OSError as e:
            warnings.warn(f"could not write the lattice file into {d}: {e}", stacklevel=3)
