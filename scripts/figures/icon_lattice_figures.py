"""The figures of docs/codecs/icon_lattice.md, computed from the code and the bundled data.

    python scripts/figures/icon_lattice_figures.py [--out DIR] [--only NAME ...]

Writes plain SVG (real text, generic fonts, an opaque light background) into docs/codecs/figures/icon_lattice/:

* ``lattice.svg``: the ICON-CH1 domain on its lattice image (bundled lattice) and a close-up of the two triangle
  orientations and the edge-neighbour rule.
* ``tree.svg``: the refinement tree on a cut-out of ICON-CH1 (a parent and its 4 children in native order, one gather
  step with its sources, the decode step of every cell) and the 12 steps of ``tree`` on ICON-CH1.
* ``structures.svg``: ``tree``, ``tree_b`` and ``hc2_k3`` on the same cut-out, with one target and its sources each.
* ``quantisation.svg``: the plain and the residual-domain rule on a number line, and the closed loop.
* ``pipeline.svg``: the stages of one LATTICE field, encoder and decoder.
* ``dag.svg``: cross terms of the default ``icon-ch1`` plan of tests/data/icon_ch1_sample.json as a decode DAG.
* ``container.svg``: the chunk layout (container version 3) with the field records and the LATTICE payload of the
  synthetic test chunk (tests/conftest.py) as ``container.py`` and ``encoder.py`` write them.

Every number shown is computed here (structures are built from the lattice, the plan is resolved, the test chunk is
encoded), so the figures follow the code. The structures and the encoder run in a temporary cache directory.
"""

from __future__ import annotations

import argparse
import html
import math
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "codecs" / "figures" / "icon_lattice"
HH = math.sqrt(3) / 2

# ------------------------------------------------------------------------------------------------ palette
INK = "#1f2328"
MUTED = "#57606a"
LINE = "#8c959f"
LIGHT = "#d0d7de"
FAINT = "#f0f2f4"
BG = "#ffffff"
DECODED = "#2f6db5"  # already decoded (sources, decoded values, LATTICE fields)
TARGET = "#d9711c"  # being predicted (targets)
STATIC = "#2e8b57"  # static / exact (static fields, lossless, constants)
RECIPE = "#7d8590"  # analytic (recipes)


def mix(a: str, b: str, t: float) -> str:
    """Colour ``a`` mixed with ``b`` (t = 0: a, t = 1: b)."""
    pa = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    pb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(pa, pb, strict=True))


DECODED_LIGHT = mix(DECODED, "#ffffff", 0.78)
TARGET_LIGHT = mix(TARGET, "#ffffff", 0.75)
STATIC_LIGHT = mix(STATIC, "#ffffff", 0.78)


def ramp(n: int) -> list[str]:
    """``n`` greys for the decode order of the steps, dark (decoded first) to light (decoded last)."""
    if n == 1:
        return [LINE]
    return [mix("#3d444d", "#eceff2", (i / (n - 1)) ** 1.8) for i in range(n)]


# ------------------------------------------------------------------------------------------------ SVG writer
def _f(v: float) -> str:
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


class Svg:
    """A minimal SVG writer: shapes and real text on an opaque background."""

    def __init__(self, width: int, height: int, title: str) -> None:
        self.w, self.h, self.title = width, height, title
        self.parts: list[str] = []
        self.markers: dict[str, str] = {}

    def rect(self, x, y, w, h, fill="none", stroke="none", sw=1.0, rx=0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        r = f' rx="{_f(rx)}"' if rx else ""
        self.parts.append(
            f'<rect x="{_f(x)}" y="{_f(y)}" width="{_f(w)}" height="{_f(h)}"{r} fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}"{d}/>'
        )

    def poly(self, pts, fill="none", stroke="none", sw=1.0, join="round"):
        p = " ".join(f"{_f(x)},{_f(y)}" for x, y in pts)
        self.parts.append(
            f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="{join}"/>'
        )

    def path(self, d, fill="none", stroke="none", sw=1.0, dash=None, marker=None, extra=""):
        a = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{self._marker(marker)})"' if marker else ""
        self.parts.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{a}{m}{extra}/>')

    def line(self, x1, y1, x2, y2, stroke=LINE, sw=1.0, dash=None, marker=None):
        self.path(f"M{_f(x1)},{_f(y1)} L{_f(x2)},{_f(y2)}", stroke=stroke, sw=sw, dash=dash, marker=marker)

    def circle(self, x, y, r, fill="none", stroke="none", sw=1.0):
        self.parts.append(
            f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(r)}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def text(self, x, y, s, size=13, anchor="start", weight="normal", fill=INK, mono=False, italic=False, rotate=0):
        a = [f'x="{_f(x)}"', f'y="{_f(y)}"', f'font-size="{size}"', f'fill="{fill}"']
        if anchor != "start":
            a.append(f'text-anchor="{anchor}"')
        if weight != "normal":
            a.append(f'font-weight="{weight}"')
        if mono:
            a.append('font-family="monospace"')
        if italic:
            a.append('font-style="italic"')
        if rotate:
            a.append(f'transform="rotate({rotate} {_f(x)} {_f(y)})"')
        self.parts.append(f"<text {' '.join(a)}>{html.escape(s, quote=False)}</text>")

    def lines(self, x, y, rows, size=13, gap=None, **kw):
        """Several lines of text from baseline ``y``; returns the baseline after the last line."""
        gap = gap or round(size * 1.35)
        for i, s in enumerate(rows):
            self.text(x, y + i * gap, s, size=size, **kw)
        return y + len(rows) * gap

    def arrow(self, x1, y1, x2, y2, color=INK, sw=1.4, shrink0=0.0, shrink1=0.0, dash=None, curve=0.0):
        """Arrow from (x1, y1) to (x2, y2), shortened at both ends, optionally bent sideways by ``curve`` px."""
        dx, dy = x2 - x1, y2 - y1
        n = math.hypot(dx, dy) or 1.0
        ux, uy = dx / n, dy / n
        ax, ay, bx, by = x1 + ux * shrink0, y1 + uy * shrink0, x2 - ux * shrink1, y2 - uy * shrink1
        if curve:
            mx, my = (ax + bx) / 2 - uy * curve, (ay + by) / 2 + ux * curve
            d = f"M{_f(ax)},{_f(ay)} Q{_f(mx)},{_f(my)} {_f(bx)},{_f(by)}"
        else:
            d = f"M{_f(ax)},{_f(ay)} L{_f(bx)},{_f(by)}"
        self.path(d, stroke=color, sw=sw, dash=dash, marker=color)

    def _marker(self, color: str) -> str:
        mid = "a" + color.lstrip("#")
        self.markers[mid] = (
            f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,1 L10,5 L0,9 z" fill="{color}"/></marker>'
        )
        return mid

    def swatch(self, x, y, color, label, size=13, stroke="none", w=14, h=14):
        self.rect(x, y - h + 3, w, h, fill=color, stroke=stroke, sw=1)
        self.text(x + w + 6, y, label, size=size)
        return x + w + 6 + text_width(label, size) + 16

    def save(self, path: Path) -> Path:
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}" font-family="sans-serif">\n'
            f"<title>{html.escape(self.title)}</title>\n"
        )
        defs = "<defs>" + "".join(self.markers.values()) + "</defs>\n" if self.markers else ""
        bg = f'<rect x="0" y="0" width="{self.w}" height="{self.h}" fill="{BG}"/>\n'
        path.write_text(head + defs + bg + "\n".join(self.parts) + "\n</svg>\n")
        return path


def text_width(s: str, size: float, mono: bool = False) -> float:
    """A conservative width estimate (DejaVu-like metrics)."""
    return len(s) * size * (0.61 if mono else 0.58)


def panel_label(svg: Svg, x: float, y: float, label: str, title: str) -> None:
    svg.text(x, y, f"{label}  {title}", size=14, weight="bold")


# ------------------------------------------------------------------------------------------------ lattice geometry
def tri(r: int, c: int, s: float = 1.0, x0: float = 0.0) -> list[tuple[float, float]]:
    """Vertices (x, y; y down, unit = side of a finest cell) of lattice cell (r, c) on a lattice of side ``s`` and
    column offset ``x0``. (r + c) even: down triangle, edge neighbour at r - 1 (``structures/common.py``)."""
    xc, top = x0 + s * 0.5 * c, s * r * HH
    if (r + c) % 2 == 0:
        return [(xc - s / 2, top), (xc + s / 2, top), (xc, top + s * HH)]
    return [(xc - s / 2, top + s * HH), (xc + s / 2, top + s * HH), (xc, top)]


def level_x0(k: int) -> float:
    """Column offset of the level-k lattice: a central (r, c) of level k has its parent at (r // 2, (c + 1) // 2)."""
    return -0.5 * (2**k - 1)


def parent_of(r: int, c: int) -> tuple[int, int]:
    return r // 2, (c + 1) // 2


def centroid(pts):
    return sum(p[0] for p in pts) / 3, sum(p[1] for p in pts) / 3


class View:
    """Maps lattice units to pixels for a window of fine cells."""

    def __init__(self, r0: int, c0: int, ox: float, oy: float, scale: float) -> None:
        self.xmin, self.ymin = 0.5 * c0 - 0.5, r0 * HH
        self.ox, self.oy, self.s = ox, oy, scale

    def p(self, pt):
        return self.ox + (pt[0] - self.xmin) * self.s, self.oy + (pt[1] - self.ymin) * self.s

    def pts(self, pts):
        return [self.p(q) for q in pts]


# ------------------------------------------------------------------------------------------------ data
class Data:
    """Lazily computed inputs shared by the figures."""

    def __init__(self) -> None:
        self._layout = self._cimg = None
        self._structures: dict[str, object] = {}

    @property
    def layout(self):
        if self._layout is None:
            from anemoi_datasets_codecs.icon_lattice import grids

            self._layout = grids.load_lattice("icon-ch1")
        return self._layout

    @property
    def cimg(self) -> np.ndarray:
        if self._cimg is None:
            from anemoi_datasets_codecs.icon_lattice.structure import cell_image

            self._cimg = cell_image(self.layout.pixel_index, self.layout.shape)
        return self._cimg

    def structure(self, sid: str):
        """(structure, step index of every native cell, position of every cell within its step)."""
        if sid not in self._structures:
            from anemoi_datasets_codecs.icon_lattice.structures import STRUCTURES

            s = STRUCTURES[sid](self.cimg)
            s.validate()
            stepof = np.full(s.n_cells, -1, np.int32)
            within = np.full(s.n_cells, -1, np.int64)
            for i, st in enumerate(s.steps):
                stepof[st.t] = i
                within[st.t] = np.arange(st.n)
            assert (stepof >= 0).all()
            self._structures[sid] = (s, stepof, within)
        return self._structures[sid]

    def sources(self, sid: str, cell: int) -> list[int]:
        s, stepof, within = self.structure(sid)
        st = s.steps[stepof[cell]]
        return [int(x) for x in st.src[:, within[cell]]]

    def rc(self, cell: int) -> tuple[int, int]:
        p = int(self.layout.pixel_index[cell])
        return p // self.layout.shape[1], p % self.layout.shape[1]


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


# ================================================================================================ figure 1
def fig_lattice(data: Data, out: Path) -> Path:
    lay = data.layout
    h, w = lay.shape
    svg = Svg(900, 500, "ICON-CH1 on the triangle lattice")

    # (a) the domain on the lattice image, in true proportions (columns 1/2 side, rows sqrt(3)/2 side apart)
    panel_label(svg, 20, 28, "(a)", "ICON-CH1 on its lattice image")
    occ = np.zeros(h * w, bool)
    occ[lay.pixel_index] = True
    occ = occ.reshape(h, w)
    scale = 380 / (h * HH)
    fx, fy = 92, 66
    fw, fh = (0.5 * w) * scale, h * HH * scale
    svg.rect(fx, fy, fw, fh, fill=FAINT, stroke=LINE, sw=1)
    # outline: every row of the image holds one run of cells (checked), so the boundary is the run ends per row
    left, right = [], []
    for r in range(h):
        cols = np.flatnonzero(occ[r])
        if len(cols) == 0:
            continue
        assert cols[-1] - cols[0] + 1 == len(cols), "domain row with a gap"
        y0, y1 = fy + r * HH * scale, fy + (r + 1) * HH * scale
        xl, xr = fx + 0.5 * cols[0] * scale, fx + 0.5 * (cols[-1] + 1) * scale
        left += [(xl, y0), (xl, y1)]
        right += [(xr, y0), (xr, y1)]
    svg.poly(_simplify(left + right[::-1], 0.35), fill=LIGHT, stroke=INK, sw=1.2)
    svg.text(fx, fy - 8, "col 0", size=12, fill=MUTED)
    svg.text(fx + fw, fy - 8, f"col {w - 1}", size=12, fill=MUTED, anchor="end")
    svg.text(fx - 6, fy + 10, "row 0", size=12, fill=MUTED, anchor="end")
    svg.text(fx - 6, fy + fh, f"row {h - 1}", size=12, fill=MUTED, anchor="end")
    svg.text(fx + fw / 2, fy + fh + 20, f"W = {w} columns, ½ side apart", size=13, anchor="middle")
    svg.text(fx - 26, fy + fh / 2, f"H = {h} rows, √3/2 side apart", size=13, anchor="middle", rotate=-90)
    n = lay.n_cells
    cx, cy = fx + fw * 0.5, fy + fh * 0.66
    svg.text(cx, cy, f"{fmt_int(n)} cells", size=14, anchor="middle", weight="bold")
    svg.text(cx, cy + 18, f"{100 * n / (h * w):.0f}% of the {h} × {w} image", size=13, anchor="middle")
    svg.text(cx, cy + 38, "pixel = row · W + col", size=13, anchor="middle", mono=True)
    r0, c0, nr, nc = 512, 800, 4, 10  # the close-up
    px, py = fx + 0.5 * (c0 + nc / 2) * scale, fy + (r0 + nr / 2) * HH * scale
    svg.circle(px, py, 4.5, fill=TARGET, stroke=INK, sw=1)
    svg.text(px + 9, py + 5, "(b)", size=13, weight="bold")
    svg.text(fx, fy + fh + 42, "image rows and columns, not a map projection", size=12, fill=MUTED)

    # (b) close-up: the two orientations and the edge neighbours
    bx = 470
    panel_label(svg, bx - 20, 28, "(b)", f"close-up: rows {r0}-{r0 + nr - 1}, cols {c0}-{c0 + nc - 1}")
    S = 74.0
    v = View(r0, c0, bx + 34, 80, S)
    nb = {
        (513, 802): [(513, 801), (513, 803), (514, 802)],  # r + c odd: points up
        (513, 807): [(513, 806), (513, 808), (512, 807)],  # r + c even: points down
    }
    labels = {(513, 801): "c−1", (513, 803): "c+1", (514, 802): "r+1"}
    labels.update({(513, 806): "c−1", (513, 808): "c+1", (512, 807): "r−1"})
    for r in range(r0, r0 + nr):
        for c in range(c0, c0 + nc):
            assert data.cimg[r, c] >= 0
            fill = BG if (r + c) % 2 == 0 else FAINT
            if (r, c) in nb:
                fill = TARGET_LIGHT
            elif any((r, c) in x for x in nb.values()):
                fill = DECODED_LIGHT
            svg.poly(v.pts(tri(r, c)), fill=fill, stroke=LINE, sw=1)
    for cell, nbs in nb.items():
        svg.poly(v.pts(tri(*cell)), stroke=TARGET, sw=2.2)
        x, y = v.p(centroid(tri(*cell)))
        svg.text(x, y + 5, "(r, c)", size=12, anchor="middle", weight="bold")
        for q in nbs:
            svg.poly(v.pts(tri(*q)), stroke=DECODED, sw=2.2)
            qx, qy = v.p(centroid(tri(*q)))
            svg.text(qx, qy + 5, labels[q], size=12, anchor="middle")
    for r in range(r0, r0 + nr):
        x, y = v.p((0.5 * c0 - 0.5, (r + 0.5) * HH))
        svg.text(x - 8, y + 4, str(r), size=12, anchor="end", fill=MUTED)
    for c in range(c0, c0 + nc):
        x, y = v.p((0.5 * c, r0 * HH))
        svg.text(x, y - 8, str(c), size=12, anchor="middle", fill=MUTED)
    svg.text(v.ox - 8, v.oy - 8, "row \\ col", size=12, anchor="end", fill=MUTED)
    y = v.oy + nr * HH * S + 30
    svg.text(bx - 20, y, "edge neighbours of a cell (r, c):", size=13, weight="bold")
    svg.swatch(bx - 20, y + 22, BG, "r + c even, pointing down: (r, c±1) and (r−1, c)", stroke=LINE)
    svg.swatch(bx - 20, y + 44, FAINT, "r + c odd, pointing up: (r, c±1) and (r+1, c)", stroke=LINE)
    svg.lines(
        bx - 20,
        y + 74,
        [
            f"(513, 802) is pixel 513 · {w} + 802 = {513 * w + 802}, native cell {int(data.cimg[513, 802])}.",
            "grid build checks that every edge neighbour in the grid",
            "file is a lattice neighbour.",
        ],
        size=13,
    )
    return svg.save(out / "lattice.svg")


def _simplify(pts, tol):
    """Ramer-Douglas-Peucker on a closed polygon (the two halves separately)."""
    a = np.asarray(pts, float)

    def rdp(seg):
        if len(seg) < 3:
            return seg
        p, q = seg[0], seg[-1]
        d = q - p
        nrm = math.hypot(*d) or 1.0
        dist = np.abs(d[0] * (seg[:, 1] - p[1]) - d[1] * (seg[:, 0] - p[0])) / nrm
        i = int(np.argmax(dist))
        if dist[i] <= tol:
            return np.array([p, q])
        return np.vstack([rdp(seg[: i + 1])[:-1], rdp(seg[i:])])

    half = len(a) // 2
    return [tuple(x) for x in np.vstack([rdp(a[: half + 1])[:-1], rdp(np.vstack([a[half:], a[:1]]))[:-1]])]


# ================================================================================================ figure 2
def _tree_group(name: str) -> str:
    """Level of a tree / tree_b step: "delta" or "L<k>"."""
    return "delta" if name == "delta" else name[:2]


def _regular_share(s) -> float:
    """Share of the finest-level parents whose 4 children are consecutive in native order with the central child
    at slot 2 (the level-0 corners grouped by their source 0, the own parent's central)."""
    lv0 = [st for st in s.steps if st.name.startswith("L0")]
    t = np.concatenate([st.t for st in lv0]).astype(np.int64)
    c = np.concatenate([st.src[0] for st in lv0]).astype(np.int64)
    o = np.lexsort((t, c))
    t, c = t[o], c[o]
    cen, start, cnt = np.unique(c, return_index=True, return_counts=True)
    full = cnt == 3
    t3 = t[start[full, None] + np.arange(3)]
    lo = np.minimum(t3.min(1), cen[full])
    ok = (np.maximum(t3.max(1), cen[full]) - lo == 3) & (cen[full] == lo + 2)
    return int(ok.sum()) / len(cen)


def fig_tree(data: Data, out: Path) -> Path:
    s, stepof, _ = data.structure("tree")
    cimg = data.cimg
    svg = Svg(900, 836, "The refinement tree and the tree structure")

    # (a) a parent and its 4 children (native order: 4q + 0..3, central at slot 2)
    central = (512, 803)
    pr, pc = parent_of(*central)
    kids = [central, (512, 802), (512, 804), (513, 803)]
    idx = {k: int(cimg[k]) for k in kids}
    base = min(idx.values())
    assert sorted(idx.values()) == list(range(base, base + 4)) and idx[central] == base + 2
    ptri = tri(pr, pc, 2, level_x0(1))
    assert np.allclose(centroid(ptri), centroid(tri(*central)))  # the central child shares the parent's centroid
    panel_label(svg, 20, 28, "(a)", "a parent and its 4 children")
    S = 96.0
    v = View(512, 802, 40, 54, S)
    for k in kids:
        is_c = k == central
        svg.poly(v.pts(tri(*k)), fill=DECODED_LIGHT if is_c else TARGET_LIGHT, stroke=LINE, sw=1)
        x, y = v.p(centroid(tri(*k)))
        svg.text(x, y - 2, fmt_int(idx[k]), size=12, anchor="middle", weight="bold")
        svg.text(x, y + 13, f"slot {idx[k] - base}", size=12, anchor="middle")
    svg.poly(v.pts(ptri), stroke=INK, sw=2.4)
    y = v.oy + 2 * HH * S + 26
    svg.lines(
        24,
        y,
        [
            f"parent ({pr}, {pc}) on level 1 = (r // 2, (c + 1) // 2)",
            f"of its central child {central}; the central child",
            "carries the parent's value (decoded at a coarser",
            "level), the 3 corners are the targets of this level",
            "central iff (r % 4, c % 4) in (1,1) (3,3) (2,1) (0,3)",
            "native order: children 4q + 0..3, central at slot 2",
            f"({_regular_share(s):.1%} of the finest-level parents of ICON-CH1)",
        ],
        size=13,
    )

    # (b) one gather step: a level-0 corner from the 3 nearest centrals
    target = idx[(512, 802)]
    src = data.sources("tree", target)
    st = s.steps[stepof[target]]
    trc, srcs_rc = data.rc(target), [data.rc(x) for x in src]
    assert srcs_rc[0] == central and np.allclose(st.w0, [0.5, 0.25, 0.25])
    panel_label(svg, 360, 28, "(b)", f"one gather step: step {stepof[target]} ({st.name}), k = 3")
    wr0, wc0, wnr, wnc = 510, 799, 6, 10
    S2 = 47.0
    v2 = View(wr0, wc0, 372, 48, S2)
    inwin = [(r, c) for r in range(wr0, wr0 + wnr) for c in range(wc0, wc0 + wnc)]
    for rc_ in inwin:
        fill = TARGET_LIGHT if rc_ == trc else DECODED_LIGHT if rc_ in srcs_rc else BG
        svg.poly(v2.pts(tri(*rc_)), fill=fill, stroke=LIGHT, sw=0.8)
    xmax, ymax = 0.5 * (wc0 + wnc), (wr0 + wnr) * HH
    for pr_, pc_ in sorted({parent_of(r, c) for r, c in inwin}):
        pts = tri(pr_, pc_, 2, level_x0(1))
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        if (
            min(xs) >= v2.xmin - 1e-9
            and max(xs) <= xmax + 1e-9
            and min(ys) >= v2.ymin - 1e-9
            and max(ys) <= ymax + 1e-9
        ):
            svg.poly(v2.pts(pts), stroke=MUTED, sw=1.5)
    svg.poly(v2.pts(tri(*trc)), stroke=TARGET, sw=2.4)
    tx, ty = v2.p(centroid(tri(*trc)))
    svg.text(tx, ty + 5, "t", size=14, anchor="middle", weight="bold")
    for j, q in enumerate(srcs_rc):
        svg.poly(v2.pts(tri(*q)), stroke=DECODED, sw=2.4)
        qx, qy = v2.p(centroid(tri(*q)))
        svg.text(qx, qy + 5, f"s{j}", size=14, anchor="middle", weight="bold", fill=DECODED)
        if j:
            svg.arrow(qx, qy, tx, ty, color=DECODED, sw=1.8, shrink0=13, shrink1=9)
    gx = 372 + (wnc * 0.5 + 0.5) * S2 + 22
    yb = svg.lines(
        gx,
        64,
        [
            "s0: own parent's central (edge",
            "neighbour of t), weight 0.5",
            "s1, s2: centrals of the two",
            "neighbouring parents, 0.25 each",
            "(default weights; the encoder",
            "fits W per step, 8 fractional bits)",
        ],
        size=13,
    )
    svg.lines(
        gx,
        yb + 8,
        ["pred_sp = v[s0] +", "  ((W1 (v[s1]-v[s0])", "  + W2 (v[s2]-v[s0])", "  + 128) >> 8)"],
        size=12.5,
        mono=True,
    )
    svg.text(372, v2.oy + wnr * HH * S2 + 22, "dark outlines: level-1 parents", size=12, fill=MUTED)

    # (c) the decode step of every cell of a 32 x 64 cut-out, by level
    names = [x.name for x in s.steps]
    groups = list(dict.fromkeys(_tree_group(n) for n in names))
    colours = dict(zip(groups, ramp(len(groups)), strict=True))
    r0, c0, nr, nc = 512, 768, 32, 64
    S3 = 13.0
    y0 = 418
    panel_label(svg, 20, y0 - 24, "(c)", f"level of each cell (rows {r0}-{r0 + nr - 1}, cols {c0}-{c0 + nc - 1})")
    v3 = View(r0, c0, 24, y0, S3)
    counts: dict[str, int] = {}
    for r in range(r0, r0 + nr):
        for c in range(c0, c0 + nc):
            g = _tree_group(names[stepof[cimg[r, c]]])
            counts[g] = counts.get(g, 0) + 1
            svg.poly(v3.pts(tri(r, c)), fill=colours[g], stroke=BG, sw=0.5)
    svg.lines(
        24,
        y0 + nr * HH * S3 + 22,
        [
            "cells: " + ", ".join(f"{g} {counts.get(g, 0)}" for g in groups) + " (greys as in (d))",
            "the nodes of level k + 1 are the central children of level k",
        ],
        size=12.5,
        fill=MUTED,
    )

    # (d) the steps of tree on ICON-CH1
    tx0, ty0 = 474, y0
    panel_label(svg, tx0, ty0 - 24, "(d)", f"the {len(s.steps)} steps of tree on ICON-CH1")
    svg.text(tx0, ty0 + 8, "#", size=12, fill=MUTED)
    svg.text(tx0 + 22, ty0 + 8, "step", size=12, fill=MUTED)
    svg.text(tx0 + 140, ty0 + 8, "targets", size=12, fill=MUTED, anchor="end")
    svg.text(tx0 + 150, ty0 + 8, "log scale", size=12, fill=MUTED)
    svg.text(tx0 + 256, ty0 + 8, "decoder", size=12, fill=MUTED)
    total = 0
    for i, st in enumerate(s.steps):
        y = ty0 + 30 + 22.5 * i
        total += st.n
        svg.text(tx0, y, str(i), size=13)
        svg.text(tx0 + 22, y, st.name, size=13, mono=True)
        svg.text(tx0 + 140, y, fmt_int(st.n), size=13, anchor="end")
        svg.rect(
            tx0 + 150,
            y - 11,
            96 * math.log10(max(st.n, 1)) / 6,
            13,
            fill=colours[_tree_group(st.name)],
            stroke=LINE,
            sw=0.5,
        )
        if st.kind == "delta":
            note = "1D delta, snake order"
        elif i == len(s.steps) - 1 and s.regular_groups():
            note = "regular groups"
        elif s.step_groups(i) is not None:
            note = "parent triples"
        else:
            note = "the rest (domain edge)"
        if len(s.step_ranges(i)) > 1:
            note += f", {len(s.step_ranges(i))} ranges"
        svg.text(tx0 + 256, y, note, size=13)
    assert total == s.n_cells
    svg.lines(
        tx0,
        ty0 + 30 + 22.5 * len(s.steps) + 4,
        [
            f"{fmt_int(total)} cells; the coarsest {fmt_int(s.steps[0].n)} nodes are delta-coded",
            "L<k>: corners of level k (0 = finest); g: parent triples",
            "(group kernel), r: the rest, h: the finest level before",
            "its suffix of regular native groups, which the decoder",
            "writes straight to the float32 output",
        ],
        size=12.5,
        fill=MUTED,
    )
    return svg.save(out / "tree.svg")


# ================================================================================================ figure 3
def fig_structures(data: Data, out: Path) -> Path:
    cimg = data.cimg
    svg = Svg(900, 470, "The three structures")
    r0, c0, nr, nc = 512, 796, 10, 20
    S = 26.6

    def coarse(level: int, rest: str) -> str:
        return "≤L2" if level >= 2 else rest

    specs = {
        "tree": (
            lambda n: "≤L2" if n == "delta" else coarse(int(n[1]), n[:2]),
            lambda n: n.startswith("L0") and not n.endswith("r"),
            [
                "corners of each level from the 3",
                "nearest centrals (0.5, 0.25, 0.25);",
                "group kernel: 3 parent values per",
                "4 cells (0.75 gathers per cell)",
            ],
        ),
        "tree_b": (
            lambda n: "≤L2" if n == "delta" else coarse(int(n[1]), n[:3]),
            lambda n: n == "L0ur",
            [
                "per level: corners of down-parents",
                "from the 3 nearest centrals, then",
                "corners of up-parents from their 3",
                "edge neighbours (generic kernel)",
            ],
        ),
        "hc2_k3": (
            lambda n: (
                "A≥1"
                if n == "delta" or (n[0] == "A" and int(n[1]) >= 1)
                else "B≥1"
                if n == "B_coarse" or (n[0] == "B" and int(n[1]) >= 1)
                else n
            ),
            lambda n: n == "A0_D",
            [
                "down cells (A) first, as a 4:1 point",
                "hierarchy from the nearest decoded",
                "A cells, then up cells (B) from all",
                "of A and decoded B (3 gathers per cell)",
            ],
        ),
    }
    for p, (sid, (grp, is_example, desc)) in enumerate(specs.items()):
        s, stepof, _ = data.structure(sid)
        names = [x.name for x in s.steps]
        x0 = 20 + p * 298
        panel_label(svg, x0, 28, f"({'abc'[p]})", sid)
        svg.text(x0 + 30 + text_width(sid, 14), 28, f"{len(s.steps)} steps on ICON-CH1", size=13, fill=MUTED)
        v = View(r0, c0, x0, 44, S)
        cells = [(r, c) for r in range(r0, r0 + nr) for c in range(c0, c0 + nc)]
        order = list(dict.fromkeys(grp(n) for n in names))
        present = [g for g in order if any(grp(names[stepof[cimg[q]]]) == g for q in cells)]
        cols = dict(zip(present, ramp(len(present)), strict=True))
        for q in cells:
            svg.poly(v.pts(tri(*q)), fill=cols[grp(names[stepof[cimg[q]]])], stroke=BG, sw=1)
        # example: a target near the middle of the cut-out whose sources are inside it
        mid = (r0 + nr // 2, c0 + nc // 2)
        best = None
        for q in cells:
            cell = int(cimg[q])
            if not is_example(names[stepof[cell]]):
                continue
            srcs = [data.rc(x) for x in data.sources(sid, cell)]
            if not all(r0 <= a < r0 + nr and c0 + 1 <= b < c0 + nc - 1 for a, b in srcs):
                continue
            d = (q[0] - mid[0]) ** 2 + 0.3 * (q[1] - mid[1]) ** 2
            if best is None or d < best[0]:
                best = (d, q, srcs, names[stepof[cell]])
        assert best is not None
        _, q, srcs, ex_step = best
        svg.poly(v.pts(tri(*q)), fill=TARGET, stroke=INK, sw=1.2)
        tx, ty = v.p(centroid(tri(*q)))
        svg.text(tx, ty + 4.5, "t", size=13, anchor="middle", weight="bold", fill=BG)
        for j, a in enumerate(srcs):
            svg.poly(v.pts(tri(*a)), fill=DECODED, stroke=INK, sw=1.2)
            ax, ay = v.p(centroid(tri(*a)))
            svg.text(ax, ay + 4.5, f"s{j}", size=12, anchor="middle", weight="bold", fill=BG)
        y = v.oy + nr * HH * S + 24
        lx = x0
        for g in present:
            wdt = 14 + 6 + text_width(g, 12.5, mono=True) + 12
            if lx + wdt > x0 + 284:
                lx, y = x0, y + 20
            svg.rect(lx, y - 11, 14, 14, fill=cols[g], stroke=LINE, sw=0.5)
            svg.text(lx + 20, y, g, size=12.5, mono=True)
            lx += wdt
        svg.text(x0, y + 24, f"t: a target of step {ex_step}; s0-s2: its sources", size=12.5, fill=MUTED)
        svg.lines(x0, y + 48, desc, size=13)
    y = 452
    lx = svg.swatch(20, y, TARGET, "target (being predicted)", stroke=INK)
    lx = svg.swatch(lx, y, DECODED, "its sources (decoded earlier)", stroke=INK)
    svg.text(lx, y, "greys: decode order of the steps, darkest first", size=13, fill=MUTED)
    return svg.save(out / "structures.svg")


# ================================================================================================ figure 4
def fig_quantisation(out: Path) -> Path:
    svg = Svg(900, 604, "Quantisation and the bound")
    b, k, m = 1.0, 1, 3
    x = 1.40  # the true value, in units of the bound; offset 0 (the grid is anchored at zero)
    umin, umax = -2.45, 2.75
    X0, SC = 46, 152

    def px(u):
        return X0 + (u - umin) * SC

    def axis(y, ticks):
        svg.line(px(umin), y, px(umax), y, stroke=LINE, sw=1)
        for u in ticks:
            svg.line(px(u), y - 4, px(u), y + 4, stroke=LINE, sw=1)
        for u in (-2, -1, 0, 1, 2):
            svg.text(px(u), y + 20, f"{u:g} b", size=12, anchor="middle", fill=MUTED)

    def band(y):
        svg.rect(px(x - b / k), y - 30, (2 * b / k) * SC, 60, fill=FAINT, stroke=LIGHT, sw=1)
        svg.text(px(x + b / k) - 4, y - 34, "x ± bound / k", size=12, anchor="end", fill=MUTED)

    def mark_x(y):
        svg.line(px(x), y - 26, px(x), y + 26, stroke=INK, sw=2)
        svg.text(px(x), y + 44, "x (true)", size=13, anchor="middle", weight="bold")

    def recon(u, y):
        svg.circle(px(u), y, 6, fill=BG, stroke=DECODED, sw=1.8)

    def decoded(u, y, label=True):
        svg.circle(px(u), y, 6, fill=DECODED, stroke=DECODED, sw=1.8)
        if label:
            svg.text(px(u), y - 14, "v", size=14, anchor="middle", weight="bold", fill=DECODED)

    def diamond(u, y):
        svg.path(f"M{_f(px(u))},{_f(y - 9)} l7,9 l-7,9 l-7,-9 z", fill=TARGET, stroke=TARGET)

    # (a) plain rule: v on the fixed grid offset + step Z
    step = 2 * b / k
    ya = 104
    panel_label(svg, 20, 28, "(a)", "plain rule (recon_bits = 0): step = 2 (bound − e_fp) / k, here k = 1")
    band(ya)
    axis(ya, [])
    va = round(x / step) * step
    for j in range(-1, 2):
        recon(j * step, ya)
    mark_x(ya)
    decoded(va, ya)
    svg.text(px(-1.0), ya - 38, "v = offset + step · n, n = rint((x − offset) / step):", size=13, anchor="middle")
    svg.text(px(-1.0), ya - 20, "the prediction only changes the coded residual n − pred", size=13, anchor="middle")
    svg.text(px(umin), ya + 44, f"|v − x| = {abs(va - x):.2f} b ≤ step / 2 = (bound − e_fp) / k", size=13)

    # (b) residual-domain reconstruction: fine grid g, reconstruction points pred + 2^m q
    step = 2 * b / k / (1 + 2.0**-m)
    g = step / 2**m
    n = round(x / g)
    pred = -5  # a fine-grid integer, computed from the decoded sources
    d = n - pred
    q = (d + (1 << m) // 2) >> m
    if (d & ((1 << m) - 1)) == (1 << m) // 2 and q & 1:
        q -= 1  # ties to even (encoder.py)
    v = pred + (q << m)
    assert abs(v * g - x) <= step / 2 + g / 2 + 1e-12 and abs(step / 2 + g / 2 - b / k) < 1e-12
    yb = 270
    panel_label(svg, 20, 186, "(b)", f"residual-domain rule (recon_bits = m = {m}, the default)")
    svg.text(20, 206, "fine grid g = step / 2^m, step = 2 (bound − e_fp) / k / (1 + 2^−m)", size=13)
    band(yb)
    axis(yb, [j * g for j in range(math.ceil(umin / g), math.floor(umax / g) + 1)])
    for j in range(-1, 3):
        u = (pred + (j << m)) * g
        if umin < u < umax:
            recon(u, yb)
    mark_x(yb)
    svg.line(px(n * g), yb - 12, px(n * g), yb + 4, stroke=INK, sw=1.6)
    svg.text(
        px(n * g) - 4 if x > n * g else px(n * g) + 4,
        yb - 14,
        "n",
        size=14,
        weight="bold",
        anchor="end" if x > n * g else "start",
    )
    diamond(pred * g, yb)
    svg.text(px(pred * g), yb - 14, "pred", size=14, anchor="middle", weight="bold", fill=TARGET)
    decoded(v * g, yb)
    ya2 = yb + 56
    svg.arrow(px(pred * g), ya2, px(v * g), ya2, color=MUTED, sw=1.4)
    svg.text((px(pred * g) + px(v * g)) / 2, ya2 + 18, f"2^m · q = {1 << m} · {q} fine steps", size=13, anchor="middle")
    yt = ya2 + 52
    svg.lines(
        px(umin),
        yt,
        [
            f"n = rint((x − offset) / g) = {n}",
            f"pred = {pred} (from the decoded sources)",
            f"q = round((n − pred) / 2^m) = round({d} / {1 << m}) = {q}",
        ],
        size=13,
    )
    svg.lines(
        470,
        yt,
        [
            f"v = pred + 2^m q = {v}   (all in units of g)",
            "|v − x| ≤ |v − n| + |n − x|",
            "  ≤ step / 2 + g / 2 = (bound − e_fp) / k",
        ],
        size=13,
    )
    # symbol legend
    yl = yt + 70
    recon(umin + 0.05, yl - 4)
    svg.text(px(umin + 0.05) + 12, yl, "values v can take", size=12.5, fill=MUTED)
    decoded(umin + 1.45, yl - 4, label=False)
    svg.text(px(umin + 1.45) + 12, yl, "decoded v", size=12.5, fill=MUTED)
    diamond(umin + 2.35, yl - 4)
    svg.text(px(umin + 2.35) + 12, yl, "prediction (fine grid)", size=12.5, fill=MUTED)
    svg.text(px(umin + 3.7), yl, "small ticks: the fine grid g; q rounds ties to even", size=12.5, fill=MUTED)

    # (c) closed loop
    yc = 482
    panel_label(svg, 20, yc, "(c)", "closed loop: the prediction uses decoded values, so encoder and decoder agree")
    boxes = [
        ("v[s0], v[s1], v[s2]", "decoded sources", DECODED_LIGHT, DECODED),
        ("pred", "gather + cross terms", TARGET_LIGHT, TARGET),
        ("q", "coded residual (rANS)", FAINT, LINE),
        ("v[t] = pred + 2^m q", "decoded target", DECODED_LIGHT, DECODED),
    ]
    bw, gap = 178, 44
    xs = [20 + i * (bw + gap) for i in range(4)]
    for (t1, t2, fill, stroke), bx in zip(boxes, xs, strict=True):
        svg.rect(bx, yc + 14, bw, 50, fill=fill, stroke=stroke, sw=1.4, rx=6)
        svg.text(bx + bw / 2, yc + 35, t1, size=13, anchor="middle", mono=True, weight="bold")
        svg.text(bx + bw / 2, yc + 54, t2, size=12.5, anchor="middle")
    for i in range(3):
        svg.arrow(xs[i] + bw, yc + 39, xs[i + 1], yc + 39, color=INK, sw=1.4, shrink0=3, shrink1=3)
    fx0, fx1 = xs[3] + bw / 2, xs[0] + bw / 2
    svg.path(
        f"M{_f(fx0)},{_f(yc + 64)} L{_f(fx0)},{_f(yc + 88)} L{_f(fx1)},{_f(yc + 88)} L{_f(fx1)},{_f(yc + 68)}",
        stroke=DECODED,
        sw=1.4,
        marker=DECODED,
    )
    svg.text(
        (fx0 + fx1) / 2,
        yc + 106,
        "v[t] is a source of later steps; the encoder never predicts from x or n",
        size=13,
        anchor="middle",
    )
    return svg.save(out / "quantisation.svg")


# ================================================================================================ figure 5
def fig_pipeline(out: Path) -> Path:
    svg = Svg(900, 492, "One LATTICE field, encode and decode")
    bw, bh, gap, x0 = 128, 92, 17, 20
    xs = [x0 + i * (bw + gap) for i in range(6)]
    ye, yd = 84, 322

    def box(x, y, title, rows, fill, stroke):
        svg.rect(x, y, bw, bh, fill=fill, stroke=stroke, sw=1.4, rx=6)
        svg.text(x + bw / 2, y + 20, title, size=13.5, anchor="middle", weight="bold")
        for i, s in enumerate(rows):
            svg.text(x + bw / 2, y + 40 + 17 * i, s, size=12.5, anchor="middle")

    svg.text(20, 26, "encoder (numpy, C rANS encoder), per field", size=14, weight="bold")
    enc = [
        ("field x", ["float32", "native cell order"], FAINT, LINE),
        ("quantise", ["n = rint((x − o) / g)", "zero-anchored offset", "int16 or int32"], FAINT, LINE),
        ("predict", ["per step, from the", "decoded sources", "+ cross terms"], TARGET_LIGHT, TARGET),
        ("residual", ["q = round((n − pred)", "/ 2^m), ties to even", "v = pred + 2^m q"], FAINT, LINE),
        ("entropy code", ["zigzag symbols", "contexts → tables", "rANS, 32 lanes"], FAINT, LINE),
        ("payload", ["LATTICE record", "of the chunk"], DECODED_LIGHT, DECODED),
    ]
    for (t, rows, fill, stroke), x in zip(enc, xs, strict=True):
        box(x, ye, t, rows, fill, stroke)
    for i in range(5):
        svg.arrow(xs[i] + bw, ye + bh / 2, xs[i + 1], ye + bh / 2, color=INK, shrink0=2, shrink1=2)
    fb_y = ye + bh + 20
    svg.path(
        f"M{_f(xs[3] + bw / 2)},{_f(ye + bh)} L{_f(xs[3] + bw / 2)},{_f(fb_y)} L{_f(xs[2] + bw / 2)},{_f(fb_y)} "
        f"L{_f(xs[2] + bw / 2)},{_f(ye + bh + 4)}",
        stroke=DECODED,
        sw=1.4,
        marker=DECODED,
    )
    svg.text((xs[2] + xs[3] + bw) / 2, fb_y + 16, "v of decoded targets (closed loop)", size=12.5, anchor="middle")
    svg.text(xs[2] + bw / 2, ye - 30, "W, C fitted per step (IRLS)", size=12.5, anchor="middle", fill=MUTED)
    svg.text(xs[2] + bw / 2, ye - 14, "details d_c of decoded fields", size=12.5, anchor="middle", fill=MUTED)
    svg.arrow(xs[2] + bw / 2, ye - 10, xs[2] + bw / 2, ye, color=MUTED, sw=1.2)
    vx = xs[5] + bw / 2
    svg.arrow(vx, ye + bh, vx, yd, color=MUTED, sw=1.2, dash="4 3")
    svg.text(vx - 8, ye + bh + 40, "verify:", size=12.5, anchor="end", weight="bold")
    svg.lines(
        vx - 8,
        ye + bh + 58,
        ["the C decoder", "reproduces v, every", "cell within the bound;", "else LOSSLESS"],
        size=12.5,
        anchor="end",
    )

    svg.text(
        20, yd - 22, "decoder: lc_decode_chunk, one C call per chunk, one OpenMP task per field", size=14, weight="bold"
    )
    dec = [
        ("output", ["float32", "native cell order"], FAINT, LINE),
        ("dequantise", ["clip(offset +", "g · (float32) v)", "2 float32 roundings"], FAINT, LINE),
        ("predict", ["same arithmetic;", "W, C from payload,", "details d_c of sources"], TARGET_LIGHT, TARGET),
        ("apply", ["v = pred + 2^m r", "d = sat16(v−pred_sp)", "(detail for later fields)"], FAINT, LINE),
        ("entropy decode", ["rANS → symbols,", "table by source", "spread; r or escape"], FAINT, LINE),
        ("payload", ["header, steps,", "tables, rANS,", "escapes"], DECODED_LIGHT, DECODED),
    ]
    for (t, rows, fill, stroke), x in zip(dec, xs, strict=True):
        box(x, yd, t, rows, fill, stroke)
    for i in (4, 3):
        svg.arrow(xs[i + 1], yd + bh / 2, xs[i] + bw, yd + bh / 2, color=INK, shrink0=2, shrink1=2)
    svg.arrow(xs[2] + bw, yd + bh / 2, xs[3], yd + bh / 2, color=INK, shrink0=2, shrink1=2)
    svg.arrow(xs[1], yd + bh / 2, xs[0] + bw, yd + bh / 2, color=INK, shrink0=2, shrink1=2)
    fb = yd - 12
    svg.path(
        f"M{_f(xs[3] + bw / 2)},{_f(yd)} L{_f(xs[3] + bw / 2)},{_f(fb)} L{_f(xs[2] + bw / 2)},{_f(fb)} "
        f"L{_f(xs[2] + bw / 2)},{_f(yd - 4)}",
        stroke=DECODED,
        sw=1.4,
        marker=DECODED,
    )
    lb = yd + bh + 22
    svg.path(
        f"M{_f(xs[3] + bw / 2)},{_f(yd + bh)} L{_f(xs[3] + bw / 2)},{_f(lb)} L{_f(xs[1] + bw / 2)},{_f(lb)} "
        f"L{_f(xs[1] + bw / 2)},{_f(yd + bh + 4)}",
        stroke=INK,
        sw=1.4,
        marker=INK,
    )
    svg.text((xs[1] + xs[3] + bw) / 2, lb + 17, "after the last step: all values v", size=12.5, anchor="middle")
    svg.lines(
        xs[3] + bw + 10,
        yd + bh + 30,
        [
            "steps in order; the ranges of a step in",
            "parallel (taskloop); predict, rANS and",
            "apply fused in L1-sized blocks",
        ],
        size=12.5,
        fill=MUTED,
    )
    return svg.save(out / "pipeline.svg")


# ================================================================================================ figure 6
def fig_dag(out: Path) -> Path:
    from anemoi_datasets_codecs import plan as P
    from anemoi_datasets_codecs.cli import lattice_config, load_config
    from anemoi_datasets_codecs.icon_lattice.codec import LatticeCodec

    cfg = lattice_config(load_config(REPO / "tests" / "data" / "icon_ch1_sample.json"))
    codec = LatticeCodec.from_config(cfg)
    cross = codec.cross_plan
    report = {t["target"]: t for t in codec.plan_report()}
    statics = set(cfg["static_fields"])
    max_depth = int(codec.options["max_depth"])
    lattice_plan = {t: [x for x in s if x not in statics] for t, s in cross.items()}
    depth = P.depth([n for n in cfg["variables"] if n not in statics], lattice_plan)
    svg = Svg(900, 576, "Cross-variable terms as a decode DAG")
    col_x = {"static": 24, 0: 262, 1: 520}
    bw, bh = 132, 30
    layout = {  # name: (column, y)
        "T_2M": (1, 96),
        "FIS": ("static", 152),
        "PMSL": (0, 152),
        "PS": (1, 208),
        "FI_500": (0, 290),
        "T_600": (0, 336),
        "T_850": (0, 382),
        "QV_600": (0, 428),
        "QV_850": (0, 474),
        "FI_700": (1, 313),
        "T_700": (1, 359),
        "QV_700": (1, 451),
    }
    pos = {nme: (col_x[col], y) for nme, (col, y) in layout.items()}
    for nme, (col, _) in layout.items():  # the columns are the plan's decode depths
        assert col == ("static" if nme in statics else depth[nme]), (nme, depth.get(nme))
    assert max(depth.values()) == max_depth
    svg.text(col_x["static"], 32, "static (exact, free)", size=13.5, weight="bold")
    svg.text(col_x[0], 32, "LATTICE, depth 0", size=13.5, weight="bold")
    svg.text(col_x[1], 32, f"LATTICE, depth {max_depth} (max_depth)", size=13.5, weight="bold")
    svg.text(col_x["static"], 50, "detail from the static array", size=12.5, fill=MUTED)
    svg.text(col_x[0], 50, "no LATTICE source", size=12.5, fill=MUTED)
    svg.text(col_x[1], 50, "decoded after their sources", size=12.5, fill=MUTED)
    svg.text(24, 82, "surface", size=12.5, fill=MUTED, italic=True)
    svg.line(20, 256, 690, 256, stroke=LIGHT, sw=1, dash="3 3")
    svg.text(24, 278, "levels", size=12.5, fill=MUTED, italic=True)

    def anchor(nme, side):
        x, y = pos[nme]
        return (x + bw if side == "r" else x), y + bh / 2

    for t in pos:  # applied terms (a source listed twice in the plan is drawn once)
        for src in dict.fromkeys(cross.get(t, [])):
            if src not in pos:
                continue
            (x1, y1), (x2, y2) = anchor(src, "r"), anchor(t, "l")
            colr = STATIC if src in statics else DECODED
            if layout[src][0] == "static" and layout[t][0] == 1:  # around the depth-0 column
                svg.path(
                    f"M{_f(x1)},{_f(y1)} C{_f(x1 + 50)},{_f(y1)} {_f(x1 + 40)},{_f(y2)} {_f(x1 + 90)},{_f(y2)} "
                    f"L{_f(x2 - 2)},{_f(y2)}",
                    stroke=colr,
                    sw=1.5,
                    marker=colr,
                )
            else:
                svg.arrow(x1, y1, x2, y2, color=colr, sw=1.5, shrink1=2)
    for t in pos:  # terms dropped by the depth limit
        for src, why in report.get(t, {}).get("dropped", []):
            if src not in pos:
                continue
            assert why.startswith("decode depth limit")
            (x1, y1), (x2, y2) = anchor(src, "r"), anchor(t, "r")
            bend = 48
            svg.path(
                f"M{_f(x1)},{_f(y1)} C{_f(x1 + bend)},{_f(y1)} {_f(x2 + bend)},{_f(y2)} {_f(x2 + 3)},{_f(y2)}",
                stroke=LINE,
                sw=1.3,
                dash="5 4",
                marker=LINE,
            )
            svg.text(x1 + 0.75 * bend + 1, (y1 + y2) / 2 + 5, "×", size=16, fill=INK, anchor="middle")
    for nme, (x, y) in pos.items():
        st = nme in statics
        svg.rect(
            x, y, bw, bh, fill=STATIC_LIGHT if st else DECODED_LIGHT, stroke=STATIC if st else DECODED, sw=1.4, rx=5
        )
        svg.text(x + bw / 2, y + 20, nme, size=13, anchor="middle", mono=True)
    nx = 712
    svg.text(nx, 96, "the decoder", size=13.5, weight="bold")
    svg.lines(
        nx,
        116,
        [
            "before the C call: LOSSLESS",
            "fields, static arrays and",
            "their details (Python)",
            "",
            "one OpenMP task per field",
            "with depend(in: sources),",
            "created longest chain first",
            "",
            "depth-0 fields run in",
            "parallel; the ranges of a",
            "step run as a taskloop",
            "",
            f"max_depth {max_depth}: at most {max_depth + 1} LATTICE",
            "fields in a chain; deeper",
            "sources are dropped (×)",
        ],
        size=12.5,
    )
    y = 538
    svg.arrow(24, y - 4, 60, y - 4, color=DECODED, sw=1.5)
    svg.text(68, y, "term on a LATTICE field (its decoded detail)", size=13)
    svg.arrow(372, y - 4, 408, y - 4, color=STATIC, sw=1.5)
    svg.text(416, y, "on a static field", size=13)
    svg.line(548, y - 4, 584, y - 4, stroke=LINE, sw=1.3, dash="5 4")
    svg.text(592, y, "dropped by the depth limit", size=13)
    svg.text(
        24,
        y + 24,
        "default icon-ch1 plan of the ICON-CH1 sample configuration; 12 of its 115 variables, edges "
        "to other fields omitted",
        size=12.5,
        fill=MUTED,
    )
    return svg.save(out / "dag.svg")


# ================================================================================================ figure 7
MODE_COLOUR = {0: DECODED, 1: STATIC, 2: mix(STATIC, "#ffffff", 0.35), 3: mix(STATIC, "#ffffff", 0.6), 4: RECIPE}


def fig_container(out: Path) -> Path:
    sys.path.insert(0, str(REPO / "tests"))
    import conftest  # noqa: E402

    from anemoi_datasets_codecs.icon_lattice import container

    lat = conftest.make_lattice()
    chunk = conftest.make_chunk(lat)[None, :, None, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enc = conftest.make_codec(lat[0]).encode(chunk, dates=[conftest.DATE])
    ch = container.parse(enc)
    names = conftest.NAMES
    sizes = [len(p) for p in ch.payloads]
    n_fields, n_dates = len(ch.modes), len(ch.dates)
    hb = 40 + 8 * n_dates + 16 * n_fields
    assert ch.header_bytes == hb + (-hb % 8) and len(enc) == ch.header_bytes + sum(sizes)
    svg = Svg(900, 836, "The chunk format, container version 3")

    # (a) the chunk header, to scale
    panel_label(svg, 20, 28, "(a)", "chunk header (integers little endian)")
    B = 21.0
    x0, y0 = 24, 58
    fields = [
        (0, 4, "LATC", "magic", "4 bytes"),
        (4, 2, "3", "version", "u16"),
        (6, 2, "0", "flags", "u16"),
        (8, 4, str(n_fields), "n_fields", "u32"),
        (12, 4, str(n_dates), "n_dates", "u32"),
        (16, 4, str(ch.header_bytes), "header_bytes", "u32"),
        (20, 4, "0", "reserved", "u32"),
        (24, 16, ch.structure_sha.hex()[:16] + "…", "structure sha256, first 16 bytes", "16 bytes"),
    ]
    for off, nb, val, lbl, typ in fields:
        svg.rect(x0 + off * B, y0, nb * B, 34, fill=FAINT, stroke=INK, sw=1)
        svg.text(x0 + (off + nb / 2) * B, y0 + 22, val, size=12.5, anchor="middle", mono=True)
        svg.text(x0 + off * B, y0 - 6, str(off), size=12, fill=MUTED)
        svg.text(x0 + (off + nb / 2) * B, y0 + 52, lbl, size=12, anchor="middle")
        svg.text(x0 + (off + nb / 2) * B, y0 + 67, typ, size=12, anchor="middle", fill=MUTED)
    svg.text(x0 + 40 * B, y0 - 6, "40", size=12, fill=MUTED, anchor="end")
    y1 = y0 + 106
    segs = [
        (f"dates: i64 × {n_dates}", 140, FAINT),
        (f"records: 16 B × {n_fields}", 210, FAINT),
        ("0-pad to 8", 96, BG),
        ("payloads in field order, see (b)", 406, mix(DECODED, "#ffffff", 0.88)),
    ]
    xx = x0
    for lbl, wdt, fill in segs:
        svg.rect(xx, y1, wdt, 30, fill=fill, stroke=INK, sw=1)
        svg.text(xx + wdt / 2, y1 + 20, lbl, size=12.5, anchor="middle")
        xx += wdt
    svg.text(x0, y1 - 6, "40", size=12, fill=MUTED)
    svg.text(x0 + 140, y1 - 6, str(40 + 8 * n_dates), size=12, fill=MUTED)
    svg.text(x0 + 446, y1 - 6, f"header_bytes = {ch.header_bytes} (a multiple of 8)", size=12, fill=MUTED)
    svg.text(
        x0,
        y1 + 50,
        "record: u8 mode, u8 fallback, u16 0, u32 0, u64 payload bytes (payloads padded to 4 bytes); "
        "dates in s since 1970",
        size=12.5,
    )

    # (b) the records and payloads of the synthetic test chunk
    yb = y1 + 96
    panel_label(
        svg, 20, yb, "(b)", f"the {n_fields} fields of the test chunk (tests/conftest.py): {fmt_int(len(enc))} bytes"
    )
    total = sum(sizes)
    W = 852
    xx, yb2 = x0, yb + 16
    for i, sz in enumerate(sizes):
        wdt = W * sz / total
        svg.rect(xx, yb2, max(wdt, 0.6), 26, fill=MODE_COLOUR[int(ch.modes[i])], stroke=BG, sw=0.6)
        if wdt > 54:
            svg.text(xx + wdt / 2, yb2 + 18, names[i], size=12, anchor="middle", fill=BG, mono=True)
        xx += wdt
    small = [nm for nm, sz in zip(names, sizes, strict=True) if sz <= 64]
    svg.text(
        x0 + W,
        yb2 + 44,
        f"payload bytes to scale; the {len(small)} payloads of at most 64 bytes are the hairlines before EXTRA",
        size=12,
        anchor="end",
        fill=MUTED,
    )
    yt = yb2 + 74
    per_col = 6
    for i, nme in enumerate(names):
        col, row = divmod(i, per_col)
        x, y = x0 + col * 290, yt + row * 21
        m = int(ch.modes[i])
        svg.rect(x, y - 11, 12, 12, fill=MODE_COLOUR[m], stroke=LINE, sw=0.5)
        svg.text(x + 18, y, nme, size=12.5, mono=True)
        svg.text(x + 150, y, container.MODE_LABELS[m].upper(), size=12)
        svg.text(x + 272, y, fmt_int(sizes[i]), size=12.5, anchor="end")
    yl = yt + per_col * 21 + 8
    svg.text(x0, yl, "mode:", size=12.5, fill=MUTED)
    lx = x0 + 50
    for m, lbl in ((0, "0 LATTICE"), (1, "1 LOSSLESS"), (2, "2 CONSTANT"), (3, "3 STATIC_REF"), (4, "4 RECIPE")):
        lx = svg.swatch(lx, yl, MODE_COLOUR[m], lbl, size=12.5)
    svg.text(lx, yl, "(green: exact)", size=12.5, fill=MUTED)

    # (c) one LATTICE payload
    i = names.index("T_2M")
    p = ch.payloads[i]
    h = container.lattice_header(p)
    yc = yl + 46
    panel_label(svg, 20, yc, "(c)", f"the LATTICE payload of T_2M: {fmt_int(len(p))} bytes")
    head_fields = [
        (8, "8 × u8", "vt et steps tables", "n_cross flags pbits m"),
        (16, "4 × f32", "offset, step g,", "clip low, clip high"),
        (16, "4 × u32", "escape units, rANS,", "table, step-section bytes"),
        (8, "4 × u16", "cross source", "field indices"),
        (4, "4 × u8", "shifts", ""),
        (4, "", "0", ""),
    ]
    Bc = 15.2
    x, off, yc2 = x0, 0, yc + 34
    for nb, typ, l1, l2 in head_fields:
        svg.rect(x, yc2, nb * Bc, 34, fill=FAINT, stroke=INK, sw=1)
        svg.text(x + nb * Bc / 2, yc2 + 22, typ or "zero", size=12, anchor="middle", mono=bool(typ))
        svg.text(x, yc2 - 6, str(off), size=12, fill=MUTED)
        if nb >= 8:
            svg.text(x + nb * Bc / 2, yc2 + 52, l1, size=12, anchor="middle")
            svg.text(x + nb * Bc / 2, yc2 + 67, l2, size=12, anchor="middle")
        x, off = x + nb * Bc, off + nb
    assert off == container.LATTICE_HEADER
    svg.text(x, yc2 - 6, str(off), size=12, fill=MUTED, anchor="end")
    svg.text(x - 4 * Bc, yc2 + 52, "shifts", size=12, anchor="middle")
    # the sections after the header, with this payload's sizes
    has_mask = bool(h.flags & 4)
    mask_b = 0  # the test chunk has no mask term
    assert not has_mask
    pad1 = -(container.LATTICE_HEADER + mask_b + h.steps_bytes + h.tables_bytes) % 4
    esc_b = h.n_esc * (h.et if h.et in (2, 4) else 1)
    pad2 = len(p) - (container.LATTICE_HEADER + mask_b + h.steps_bytes + h.tables_bytes + pad1 + h.rans_bytes + esc_b)
    assert 0 <= pad2 < 4
    secs = [
        ("header", container.LATTICE_HEADER),
        ("mask term", mask_b),
        ("steps", h.steps_bytes),
        ("tables", h.tables_bytes),
        ("pad4", pad1),
        ("rANS streams", h.rans_bytes),
        ("escapes", esc_b),
        ("pad4", pad2),
    ]
    ys = yc2 + 96
    widths = [max(text_width(nme, 12), text_width(f"{nb} B", 12, True)) + 18 for nme, nb in secs]
    k = (876 - x0) / sum(widths)
    sx = x0
    for (nme, nb), wdt in zip(secs, widths, strict=True):
        w_ = wdt * k
        fill = DECODED_LIGHT if nme == "rANS streams" else FAINT if nb else BG
        svg.rect(sx, ys, w_, 36, fill=fill, stroke=INK, sw=1)
        svg.text(sx + w_ / 2, ys + 15, nme, size=12, anchor="middle")
        svg.text(sx + w_ / 2, ys + 30, f"{nb} B", size=12, anchor="middle", mono=True)
        sx += w_
    svg.text(x0, ys - 8, "then, in order (not to scale):", size=12, fill=MUTED)
    svg.lines(
        x0,
        ys + 56,
        [
            f"this payload: {h.n_steps} structure steps, {h.n_tables} tables, rANS precision {h.prob_bits} bits, "
            f"m = {h.recon_bits}, {h.n_cross} cross terms (sources {' and '.join(names[c] for c in h.cross)}), "
            "no mask term",
            "steps: per step u8 n_ctx, table map, thresholds, W (i16), C (i32), per range u32 rANS bytes and escape "
            "units",
        ],
        size=12.5,
    )
    yf = ys + 106
    svg.rect(20, yf, 860, 50, fill=FAINT, stroke=INK, sw=1.2, rx=6)
    svg.text(
        32,
        yf + 20,
        "frozen (container version 3): every offset, type and order above, the mode codes and the decode rule.",
        size=12.5,
        weight="bold",
    )
    svg.text(
        32,
        yf + 38,
        "not frozen: what the encoder writes into them (weights, tables, contexts, which terms, sizes).",
        size=12.5,
    )
    return svg.save(out / "container.svg")


# ================================================================================================ main
FIGURES = ("lattice", "tree", "structures", "quantisation", "pipeline", "dag", "container")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--only", nargs="*", choices=FIGURES)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["ANEMOI_DATASETS_CODECS_CACHE"] = tmp  # structures, captured statics: nothing outside tmp
        os.environ.pop("ANEMOI_DATASETS_CODECS_PATH", None)
        data = Data()
        for name in a.only or FIGURES:
            t0 = time.perf_counter()
            if name == "lattice":
                path = fig_lattice(data, a.out)
            elif name == "tree":
                path = fig_tree(data, a.out)
            elif name == "structures":
                path = fig_structures(data, a.out)
            elif name == "quantisation":
                path = fig_quantisation(a.out)
            elif name == "pipeline":
                path = fig_pipeline(a.out)
            elif name == "dag":
                path = fig_dag(a.out)
            else:
                path = fig_container(a.out)
            print(f"{path.relative_to(REPO) if path.is_relative_to(REPO) else path}: {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
