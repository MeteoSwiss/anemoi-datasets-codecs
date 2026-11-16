"""Structure ``hc2_k3``: the honeycomb two-orientation hierarchy with every step capped at the 3 nearest decoded cells
(frozen; see docs/codecs/icon_lattice.md "Structures").

Orientation A (triangular point lattice) as a 4:1 hierarchy of E (row midpoints) and D (diagonal) steps, then
orientation B as its own hierarchy from all of A and decoded B. 3 gathers per cell.
"""

from __future__ import annotations

import numpy as np

from ..lattice import LayoutError
from ..structure import Structure
from .common import finish, gather, rc, snake, valid_offsets


def _lsub(r: np.ndarray, c: np.ndarray, m: int) -> np.ndarray:
    u = 2**m
    return (r % u == 0) & ((c - r) % (2 * u) == 0)


def build_hc2_k3(cimg: np.ndarray, levels: int = 4, k: int = 3) -> Structure:
    """Honeycomb two-orientation hierarchy, k nearest decoded cells per step (see module docstring). Small
    lattices get fewer levels."""
    for lv in range(levels, 0, -1):
        try:
            return _hc2(cimg, lv, k)
        except LayoutError:
            if lv == 1:
                raise
    raise AssertionError


def _hc2(cimg: np.ndarray, levels: int, k: int) -> Structure:
    n_cells = int((cimg >= 0).sum())
    r, c = rc(cimg.shape)
    a_pat = (r + c) % 2 == 0
    w0 = [1.0 / k] * k
    parts = []
    # orientation A: coarsest sub-lattice delta-coded, then E (row midpoints) and D (diagonals) per level
    coarse_a = _lsub(r, c, levels)
    for m in range(levels - 1, -1, -1):
        u = 2**m
        lc = _lsub(r, c, m + 1)
        e = (r % (2 * u) == 0) & ((c - r) % (4 * u) == 2 * u)
        d = (r % (2 * u) == u) & ((c - r) % (2 * u) == 0)
        rm, cm = min(40, 8 * u), min(80, 16 * u)
        parts.append((f"A{m}_E", *gather(cimg, e, lc, valid_offsets(e, lc, 0, k, rm, cm)), w0))
        parts.append((f"A{m}_D", *gather(cimg, d, lc | e, valid_offsets(d, lc | e, 0, k, rm, cm)), w0))

    # orientation B: the A pattern shifted by (0, 1), predicted from all of A plus decoded B
    def lb(m: int) -> np.ndarray:
        return _lsub(r, c - 1, m)

    ub = lb(levels)
    parts.append(("B_coarse", *gather(cimg, ub, a_pat, valid_offsets(ub, a_pat, 1, k, 40, 80)), w0))
    for m in range(levels - 1, -1, -1):
        u = 2**m
        lc = lb(m + 1)
        e = (r % (2 * u) == 0) & ((c - 1 - r) % (4 * u) == 2 * u)
        d = (r % (2 * u) == u) & ((c - 1 - r) % (2 * u) == 0)
        rm, cm = min(40, 8 * u + 4), min(80, 16 * u + 8)
        de = a_pat | lc
        parts.append((f"B{m}_E", *gather(cimg, e, de, valid_offsets(e, de, 1, k, rm, cm)), w0))
        dd = a_pat | lc | e
        parts.append((f"B{m}_D", *gather(cimg, d, dd, valid_offsets(d, dd, 1, k, rm, cm)), w0))
    return finish("hc2_k3", n_cells, snake(cimg, coarse_a), parts)
