"""Helpers shared by the structure builders: lattice geometry and the assembly of steps into a ``Structure``.

Frozen with the builders: the steps (and so every structure sha and every chunk) depend on them bit for bit.
"""

from __future__ import annotations

import numpy as np

from ..lattice import LayoutError
from ..structure import Step, Structure

HH = np.sqrt(3) / 2


def rc(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Row and column index images of ``shape``."""
    r, c = np.indices(shape)
    return r, c


def centroid_disp(dr: int, dc: int, tgt_par: int) -> tuple[float, float]:
    """Centroid displacement (x, y) in units of the same-orientation spacing for a lattice offset (dr, dc) from a
    cell of parity ``tgt_par`` ((r + c) % 2; 0 = orientation A, vertical edge neighbour at r - 1)."""
    src_par = (tgt_par + dr + dc) % 2
    yoff = {0: -HH / 3, 1: -2 * HH / 3}
    return 0.5 * dc, -dr * HH + yoff[src_par] - yoff[tgt_par]


def valid_offsets(
    tpos: np.ndarray, dpos: np.ndarray, tgt_par: int, k: int, rmax: int = 40, cmax: int = 80
) -> list[tuple[int, int]]:
    """The k offsets nearest in physical distance whose positions are decoded (``dpos``) for every interior target
    of the pattern ``tpos``."""
    h, w = tpos.shape
    rmax, cmax = min(rmax, h // 3), min(cmax, w // 3)
    tr, tc = np.nonzero(tpos[rmax : h - rmax, cmax : w - cmax])
    if len(tr) == 0:
        raise LayoutError("lattice too small for the structure")
    tr, tc = tr + rmax, tc + cmax
    sel = np.linspace(0, len(tr) - 1, min(len(tr), 4000)).astype(int)
    tr, tc = tr[sel], tc[sel]
    cands = []
    for dr in range(-rmax, rmax + 1):
        for dc in range(-cmax, cmax + 1):
            if (dr, dc) != (0, 0):
                dx, dy = centroid_disp(dr, dc, tgt_par)
                cands.append((round(dx * dx + dy * dy, 9), abs(dr), dr, dc))
    cands.sort()
    out: list[tuple[int, int]] = []
    for _, _, dr, dc in cands:
        if np.all(dpos[tr + dr, tc + dc]):
            out.append((dr, dc))
            if len(out) == k:
                return out
    raise LayoutError(f"no {k} decoded neighbours within the search window")


def gather(cimg: np.ndarray, tpos: np.ndarray, dpos: np.ndarray, offsets: list[tuple[int, int]]):
    """Targets (cells at ``tpos``) and their sources at ``offsets`` (-1 where outside / padding / not ``dpos``)."""
    h, w = cimg.shape
    tr, tc = np.nonzero(tpos & (cimg >= 0))
    src = np.full((len(offsets), len(tr)), -1, np.int64)
    for j, (dr, dc) in enumerate(offsets):
        rr, cc = tr + dr, tc + dc
        ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        ok[ok] = dpos[rr[ok], cc[ok]] & (cimg[rr[ok], cc[ok]] >= 0)
        src[j, ok] = cimg[rr[ok], cc[ok]]
    return cimg[tr, tc], src


def snake(cimg: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Cells at ``pos`` in boustrophedon order (rows top to bottom, alternating direction)."""
    r, c = np.nonzero(pos & (cimg >= 0))
    key = r * (cimg.shape[1] + 1) + np.where(r % 2 == 0, c, cimg.shape[1] - c)
    o = np.argsort(key, kind="stable")
    return cimg[r[o], c[o]]


def finish(sid: str, n_cells: int, delta: np.ndarray, parts: list[tuple[str, np.ndarray, np.ndarray, list]]):
    """Assemble steps: fill missing sources with the nearest present one, move source-less targets to the delta
    step, sort targets by native index."""
    extra: list[np.ndarray] = []
    steps: list[Step] = []
    for name, t, src, w0 in parts:
        pres = src >= 0
        none = ~pres.any(0)
        if none.any():
            extra.append(t[none])
            t, src, pres = t[~none], src[:, ~none], pres[:, ~none]
        if len(t) == 0:
            continue
        first = src[np.argmax(pres, axis=0), np.arange(src.shape[1])]
        src = np.where(pres, src, first[None, :])
        o = np.argsort(t, kind="stable")
        steps.append(
            Step(
                name,
                "gather",
                t[o].astype(np.int32),
                np.ascontiguousarray(src[:, o], np.int32),
                np.asarray(w0, np.float64),
            )
        )
    d = np.concatenate([delta, *extra]).astype(np.int32)
    return Structure(sid, n_cells, [Step("delta", "delta", d), *steps])
