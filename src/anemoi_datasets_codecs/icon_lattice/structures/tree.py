"""Structures ``tree`` and ``tree_b``: the ICON refinement tree in native order (frozen; see docs/codecs/icon_lattice.md
"Structures").

Level-k nodes are the central children of level k-1 (the value of a parent is its central child's value); level-(k+1)
coordinates of a central (r, c) are (r // 2, (c + 1) // 2) (central iff (r % 4, c % 4) in CENTRAL, validated against
the grid file). Corners of level k are predicted from the 3 nearest level-(k+1) nodes (own parent + the two
neighbouring parents); coarsest level (<= ~2000 nodes) delta-coded. 0.75 gathers per cell. ``tree_b``: at every
level the corners of down-parents first, then those of up-parents from their 3 edge neighbours.

The corner steps are reordered as parent triples (the decoder's group hint, ``Structure.step_groups``) and the last
step is split into its suffix of regular native groups (``Structure.regular_groups``).
"""

from __future__ import annotations

import numpy as np

from ..lattice import LayoutError
from ..structure import Step, Structure
from .common import finish, gather, rc, snake, valid_offsets

CENTRAL = frozenset({(3, 3), (1, 1), (2, 1), (0, 3)})  # (r % 4, c % 4) of central children (grid-file validated)
COARSEST = 2000


def _central_mask(shape: tuple[int, int]) -> np.ndarray:
    r, c = rc(shape)
    m = np.zeros(shape, bool)
    for a, b in CENTRAL:
        m |= (r % 4 == a) & (c % 4 == b)
    return m


def _check_central_rule() -> None:
    """The CENTRAL pattern must be consistent with the lattice convention (A: (r + c) even, vertical edge
    neighbour at r - 1): every central's 3 edge neighbours are corners and every corner touches one central."""
    cen = _central_mask((12, 12))
    for r in range(4, 8):
        for c in range(4, 8):
            nb = [(r, c - 1), (r, c + 1), (r - 1 if (r + c) % 2 == 0 else r + 1, c)]
            hits = sum(cen[x, y] for x, y in nb)
            if cen[r, c] != (hits == 0) or (not cen[r, c] and hits != 1):
                raise LayoutError("lattice coordinates do not follow the ICON refinement pattern")


def level_images(cimg: np.ndarray, k: int) -> list[np.ndarray]:
    """Refinement-tree level images ``[level 0 (= cimg), level 1 (central children), ...]``."""
    imgs = [cimg]
    for _ in range(k):
        ci = imgs[-1]
        cen = _central_mask(ci.shape) & (ci >= 0)
        nxt = np.full((ci.shape[0] // 2 + 1, (ci.shape[1] + 1) // 2 + 1), -1, np.int64)
        rr, cc = np.nonzero(cen)
        nxt[rr // 2, (cc + 1) // 2] = ci[rr, cc]
        imgs.append(nxt)
    return imgs


# ---------------------------------------------------------------------------------------------- builders
def build_tree(cimg: np.ndarray, bpass: bool = False) -> Structure:
    """Native-order refinement tree (see module docstring). ``bpass``: at every level the corners of parity-0
    ("down") parents first, from the 3 nearest centrals, then the corners of up-parents from their 3 edge neighbours
    (the central sibling and 2 corners of neighbouring down-parents, decoded just before)."""
    sid = "tree_b" if bpass else "tree"
    _check_central_rule()
    n_cells = int((cimg >= 0).sum())
    imgs = [cimg]
    while (imgs[-1] >= 0).sum() > COARSEST:
        imgs = level_images(cimg, len(imgs))
    levels = len(imgs) - 1
    parts = []
    for k in range(levels - 1, -1, -1):
        ci = imgs[k]
        r, c = rc(ci.shape)
        cenpat = _central_mask(ci.shape)
        ts, ss = [], []
        for a in range(4):
            for b in range(4):
                if (a, b) in CENTRAL:
                    continue
                tpat = (r % 4 == a) & (c % 4 == b)
                offs = valid_offsets(tpat, cenpat, (a + b) % 2, 3, rmax=8, cmax=16)
                t, src = gather(ci, tpat, cenpat, offs)
                ts.append(t)
                ss.append(src)
        T, SRC = np.concatenate(ts), np.concatenate(ss, axis=1)
        if not bpass:
            parts.append((f"L{k}", T, SRC, [0.5, 0.25, 0.25]))
            continue
        pr, pc = np.nonzero(ci >= 0)
        R = np.full(int(cimg.max()) + 1, -1, np.int64)
        C = np.full_like(R, -1)
        R[ci[pr, pc]], C[ci[pr, pc]] = pr, pc
        cen = SRC[0]
        up = np.zeros(len(T), bool)
        ok = cen >= 0
        up[ok] = (R[cen[ok]] // 2 + (C[cen[ok]] + 1) // 2) % 2 == 1
        parts.append((f"L{k}d", T[~up], SRC[:, ~up], [0.5, 0.25, 0.25]))
        tu, sib = T[up], cen[up]
        r0, c0 = R[tu], C[tu]
        h, w = ci.shape
        nbs = []
        for rr, cc in ((r0, c0 - 1), (r0, c0 + 1), (np.where((r0 + c0) % 2 == 0, r0 - 1, r0 + 1), c0)):
            inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            x = np.full(len(tu), -1, np.int64)
            x[inside] = ci[rr[inside], cc[inside]]
            nbs.append(x)
        nb = np.stack(nbs)
        order = np.argsort(nb == sib[None, :], axis=0, kind="stable")  # the two non-sibling neighbours first
        others = np.take_along_axis(nb, order, axis=0)[:2]
        parts.append((f"L{k}u", tu, np.concatenate([sib[None, :], others]), [1 / 3] * 3))
    s = finish(sid, n_cells, snake(imgs[levels], imgs[levels] >= 0), parts)
    head, tail = _split_regular(s.steps[-1], n_cells)
    steps = [s.steps[0]]
    for st in [*s.steps[1:-1], *([head] if head is not None else [] if tail is not None else [s.steps[-1]])]:
        steps += _split_triples(st)
    if tail is not None:
        steps.append(tail)
    return Structure(sid, n_cells, steps)


def _split_triples(st: Step) -> list[Step]:
    """Reorder a corner step as parent triples (3 targets sharing source 0 whose sources 1-2 come from 3 distinct
    neighbours; ordered by central) followed, in a separate step, by the targets that do not form such triples."""
    s0 = st.src[0].astype(np.int64)
    order = np.lexsort((st.t, s0))
    t, src = st.t[order], st.src[:, order]
    key = src[0].astype(np.int64)
    start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    size = np.diff(np.r_[start, len(key)])
    good = np.zeros(len(t), bool)
    for b in start[size == 3]:
        nbs = set(src[1, b : b + 3].tolist()) | set(src[2, b : b + 3].tolist())
        if len(nbs) == 3 and src[1, b] != src[2, b] and all(src[1, b + j] != src[2, b + j] for j in range(3)):
            good[b : b + 3] = True
    out = []
    if good.any():
        out.append(Step(st.name + "g", "gather", t[good].astype(np.int32), np.ascontiguousarray(src[:, good]), st.w0))
    if (~good).any():
        rest = np.flatnonzero(~good)
        rest = rest[np.argsort(t[rest], kind="stable")]
        out.append(Step(st.name + "r", "gather", t[rest].astype(np.int32), np.ascontiguousarray(src[:, rest]), st.w0))
    return out


def _split_regular(st: Step, n_cells: int) -> tuple[Step | None, Step | None]:
    """Split a step into a head and its longest suffix of regular groups (see ``Structure.regular_groups``)."""
    t, s0 = st.t.astype(np.int64), st.src[0].astype(np.int64)
    ng = 0
    while 3 * (ng + 1) <= len(t):  # grow the suffix from the end, block-wise for speed
        m = min(len(t) // 3, max(2 * ng, 1024))
        tt, ss = t[len(t) - 3 * m :].reshape(m, 3), s0[len(t) - 3 * m :].reshape(m, 3)
        base = n_cells - 4 * m + 4 * np.arange(m)
        good = (tt == base[:, None] + np.array([0, 1, 3])).all(1) & (ss == base[:, None] + 2).all(1)
        bad = np.flatnonzero(~good)
        ng = m - (int(bad[-1]) + 1 if len(bad) else 0)
        if len(bad) or m == len(t) // 3:
            break
    if ng < 8:
        return None, None
    cut = len(t) - 3 * ng
    tail = Step(st.name + "g", "gather", st.t[cut:].copy(), np.ascontiguousarray(st.src[:, cut:]), st.w0)
    head = (
        Step(st.name + "h", "gather", st.t[:cut].copy(), np.ascontiguousarray(st.src[:, :cut]), st.w0) if cut else None
    )
    return head, tail
