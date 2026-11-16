"""Recover the triangular-lattice (row, col) index of every cell from its centre.

Algorithm (see ``prototypes/lattice.py``, reference only):

1. Gnomonic projection about the domain centre.
2. Three nearest neighbours (KD-tree); an edge neighbour has distance < ``1.3 * median(3rd-neighbour distance)``.
3. Global orientation ``theta = angle(mean(exp(6i*phi))) / 6 - 30deg + alpha``, alpha in {0, 60, 120} deg.
4. Per cell, the neighbour with the largest ``|sin(phi - theta)|`` (> 0.75) is vertical (dr = sign), the others are
   left/right (dc = sign of cos).
5. BFS over the neighbour graph propagates integer (r, c).
6. Check invariants: connected, (r, c) unique, every edge agrees with the coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_SIDE = 2047
ORIENTATIONS = (0, 60, 120)
NEIGHBOUR_FACTOR = 1.3  # edge neighbours at 0.74-1.06 h, next shell from 1.41 h on ICON-CH1
VERTICAL_MIN_SIN = 0.75


class LayoutError(ValueError):
    """The cell centres do not form a valid triangular-lattice cut-out (or the image exceeds the image size limits)."""


@dataclass(frozen=True)
class LatticeResult:
    """Output of :func:`recover_lattice`.

    Attributes:
        row: int ``(n_cells,)`` row of each cell, shifted to start at 0.
        col: int ``(n_cells,)`` column of each cell, shifted to start at 0.
        shape: ``(H, W)`` of the strip image.
        orientation: The alpha (0, 60 or 120) that was used.
    """

    row: np.ndarray
    col: np.ndarray
    shape: tuple[int, int]
    orientation: int


def gnomonic(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Gnomonic projection (degrees in) about the mean direction. Returns ``(n, 2)`` plane coordinates."""
    la, lo = np.radians(np.asarray(lat, np.float64)), np.radians(np.asarray(lon, np.float64))
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    c = xyz.mean(0)
    c /= np.linalg.norm(c)
    e = np.cross([0.0, 0.0, 1.0], c)
    if np.linalg.norm(e) < 1e-12:  # centre at a pole: any east vector will do
        e = np.cross([1.0, 0.0, 0.0], c)
    e /= np.linalg.norm(e)
    n = np.cross(c, e)
    d = xyz @ c
    if np.any(d <= 0):
        raise LayoutError("domain spans more than a hemisphere; gnomonic projection undefined")
    return np.stack([(xyz @ e) / d, (xyz @ n) / d], -1)


def find_neighbours(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """3 nearest neighbours per point.

    Returns:
        ``(idx, valid, h)``: ``idx`` ``(n, 3)`` neighbour indices, ``valid`` ``(n, 3)`` bool (distance below
        ``1.3 * h``), ``h`` the median third-neighbour distance.
    """
    from scipy.spatial import cKDTree

    if len(points) < 4:
        raise LayoutError(f"need at least 4 cells, got {len(points)}")
    dist, idx = cKDTree(points).query(points, k=4)
    dist, idx = dist[:, 1:], idx[:, 1:]
    h = float(np.median(dist[:, 2]))
    return idx, dist < NEIGHBOUR_FACTOR * h, h


def _steps(points: np.ndarray, idx: np.ndarray, valid: np.ndarray, alpha: int) -> tuple[np.ndarray, np.ndarray]:
    """Integer (dr, dc) step from each cell to each of its neighbours for orientation ``alpha`` (steps 3-4)."""
    d = points[idx] - points[:, None, :]
    phi = np.arctan2(d[..., 1], d[..., 0])
    # Neighbour directions are theta+30 mod 60 for up and down triangles, so 6*phi is the same for all edges.
    theta = np.angle(np.exp(6j * phi)[valid].mean()) / 6 - np.pi / 6 + np.radians(alpha)
    rot = phi - theta
    vert = np.abs(np.sin(rot))
    n = len(points)
    is_vert = np.zeros_like(valid)
    is_vert[np.arange(n), np.argmax(np.where(valid, vert, -1.0), axis=1)] = True
    is_vert &= valid & (vert > VERTICAL_MIN_SIN)
    dr = np.where(is_vert, np.sign(np.sin(rot)), 0).astype(np.int64)
    dc = np.where(valid & ~is_vert, np.sign(np.cos(rot)), 0).astype(np.int64)
    return dr, dc


def _propagate(idx: np.ndarray, valid: np.ndarray, dr: np.ndarray, dc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """BFS tree over the neighbour graph, then coordinates by pointer jumping along the tree (step 5)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order

    n = len(idx)
    src = np.repeat(np.arange(n), 3)[valid.ravel()]
    dst = idx.ravel()[valid.ravel()]
    graph = coo_matrix((np.ones(len(src), np.int8), (src, dst)), shape=(n, n)).tocsr()
    order, pred = breadth_first_order(graph, 0, directed=False, return_predecessors=True)
    if len(order) != n:
        raise LayoutError(f"neighbour graph is not connected: {len(order)} of {n} cells reachable from cell 0")
    # Offset of each child relative to its parent. The BFS graph is undirected, so the tree edge may only be listed
    # by the child (parent -> child = -(child -> parent)).
    child = order[1:]
    parent = pred[child]
    fwd = (idx[parent] == child[:, None]) & valid[parent]
    bwd = (idx[child] == parent[:, None]) & valid[child]
    has_fwd = fwd.any(1)
    sf, sb = np.argmax(fwd, 1), np.argmax(bwd, 1)
    off_r = np.zeros(n, np.int64)
    off_c = np.zeros(n, np.int64)
    off_r[child] = np.where(has_fwd, dr[parent, sf], -dr[child, sb])
    off_c[child] = np.where(has_fwd, dc[parent, sf], -dc[child, sb])
    up = np.arange(n)
    up[child] = parent
    # Pointer jumping: coord(i) = off(i) + coord(up(i)); log2(tree depth) vectorised passes instead of a BFS loop.
    while np.any(up[up] != up):
        off_r, off_c, up = off_r + off_r[up], off_c + off_c[up], up[up]
    return off_r, off_c


def check_invariants(
    row: np.ndarray,
    col: np.ndarray,
    idx: np.ndarray,
    valid: np.ndarray,
    dr: np.ndarray | None = None,
    dc: np.ndarray | None = None,
) -> int:
    """Step 6. Return the number of bad edges; raise ``LayoutError`` if (row, col) are not unique.

    A grid edge (cell, neighbour) is good if the coordinates differ by the step assigned to it (``dr``, ``dc``) when
    given, else if the two pixels are 4-adjacent (left/right or vertical).
    """
    i, j = np.nonzero(valid)
    nb = idx[i, j]
    ddr, ddc = row[nb] - row[i], col[nb] - col[i]
    if dr is not None and dc is not None:
        ok = (ddr == dr[i, j]) & (ddc == dc[i, j])
    else:
        ok = np.abs(ddr) + np.abs(ddc) == 1
    n_bad = int((~ok).sum())
    key = (row - row.min()) * (col.max() - col.min() + 1) + (col - col.min())
    n_dup = len(row) - len(np.unique(key))
    if n_dup:
        raise LayoutError(f"lattice coordinates are not unique: {n_dup} duplicate pixels ({n_bad} bad edges)")
    return n_bad


def _coords(points: np.ndarray, idx: np.ndarray, valid: np.ndarray, alpha: int) -> tuple[np.ndarray, np.ndarray, int]:
    dr, dc = _steps(points, idx, valid, alpha)
    row, col = _propagate(idx, valid, dr, dc)
    return row, col, check_invariants(row, col, idx, valid, dr, dc)


def recover_lattice(lat: np.ndarray, lon: np.ndarray, orientation: str | int = "auto") -> LatticeResult:
    """Recover the lattice coordinates of all cells.

    Args:
        lat: ``(n,)`` latitudes in degrees.
        lon: ``(n,)`` longitudes in degrees.
        orientation: ``"auto"`` tries alpha in {0, 60, 120} and keeps the valid one with the smallest ``H * W``;
            or an explicit alpha.

    Raises:
        LayoutError: no alpha yields a valid lattice (random points, disconnected, bad edges), or the image exceeds
            2047 in a dimension.
    """
    lat, lon = np.asarray(lat, np.float64).ravel(), np.asarray(lon, np.float64).ravel()
    if lat.shape != lon.shape:
        raise ValueError(f"lat and lon differ in size: {lat.shape} vs {lon.shape}")
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise LayoutError("non-finite latitudes/longitudes")
    if orientation != "auto" and orientation not in ORIENTATIONS:
        raise ValueError(f"orientation must be 'auto' or one of {ORIENTATIONS}, got {orientation!r}")
    alphas = ORIENTATIONS if orientation == "auto" else (int(orientation),)
    # Neighbours and the projection do not depend on alpha: compute once.
    points = gnomonic(lat, lon)
    idx, valid, h = find_neighbours(points)
    n_nb = valid.sum(1)
    header = (
        f"{len(lat)} cells, median 3rd-neighbour distance {h:.4g}, cells with 0/1/2/3 edge neighbours: "
        f"{[int((n_nb == k).sum()) for k in range(4)]}"
    )
    results: list[LatticeResult] = []
    problems: list[str] = []
    for alpha in alphas:
        try:
            row, col, bad = _coords(points, idx, valid, alpha)
        except LayoutError as e:
            problems.append(f"alpha={alpha}: {e}")
            continue
        row, col = row - row.min(), col - col.min()
        shape = (int(row.max()) + 1, int(col.max()) + 1)
        if bad:
            problems.append(f"alpha={alpha}: {bad} of {int(valid.sum())} edges disagree with the coordinates")
        elif max(shape) > MAX_SIDE:
            problems.append(f"alpha={alpha}: image {shape[0]} x {shape[1]} exceeds {MAX_SIDE} (tiling not supported)")
        else:
            results.append(LatticeResult(row, col, shape, alpha))
    if not results:
        raise LayoutError("not a triangular-lattice cut-out (" + header + "):\n  " + "\n  ".join(problems))
    return min(results, key=lambda r: r.shape[0] * r.shape[1])
