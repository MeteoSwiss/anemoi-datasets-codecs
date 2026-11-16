"""The figures of docs/codecs/field_terms.md, computed with the stage's own code.

    python scripts/figures/field_terms_figures.py [--out DIR] [--only NAME ...]

Writes plain SVG (real text, generic fonts, an opaque light background) into docs/codecs/figures/field_terms/, with
the helpers and the palette of ``icon_lattice_figures.py`` (same colour, same meaning):

* ``overview.svg``: the stage first in the chain ``[field_terms, static_ref, icon_lattice_bytes]``, and one
  (date, member) slice of a synthetic T_2M against FIS: the fit, the prediction and the residual.
* ``bound.svg``: the error bound of one cell of that slice on a number line in units of the bound: the interval of
  residuals the codec after the stage may return, its float32 ends, their reconstruction and the float32 grid of the
  decoded values near ``x + b_L`` and ``x + b`` (why the margin is needed).
* ``bypass.svg``: the per-slice decision (keep the residual or bypass), the decode of both, and the clip of a bypassed
  slice on a number line.
* ``header.svg``: header version 1, byte by byte, of the fixed input of ``tests/test_field_terms.py``
  (``test_header_pinned_bytes``: two dates, a bypassed record, a two-source term without intercept).
* ``flow.svg``: encode and decode of one target slice, step by step.

Every number shown is computed here with ``field_terms.FieldTerms`` and ``field_terms.arith`` on synthetic data, so
the figures follow the code. The output is deterministic.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from icon_lattice_figures import (
    BG,
    DECODED,
    DECODED_LIGHT,
    FAINT,
    INK,
    LIGHT,
    LINE,
    MUTED,
    REPO,
    STATIC,
    STATIC_LIGHT,
    TARGET,
    TARGET_LIGHT,
    Svg,
    _f,
    mix,
    panel_label,
    text_width,
)

from anemoi_datasets_codecs import envelope
from anemoi_datasets_codecs.field_terms import DEFAULT_MARGIN, FieldTerms, arith, header

OUT = REPO / "docs" / "codecs" / "figures" / "field_terms"
F4, F8 = np.float32, np.float64
GRAVITY = 9.80665
VARIABLES = ["FIS", "PMSL", "T_2M", "U_10M"]  # chunk order (anemoi's: sorted by name)
BOUNDS = {"PMSL": 2.0, "T_2M": 0.05, "U_10M": 0.1}
CELL = 10  # the cell of the T_2M slice that the bound figure follows (x below 256 K)


def f32(v) -> str:
    """The shortest decimal that rounds to the float32 ``v``."""
    return np.format_float_positional(F4(v), unique=True, trim="-")


def f32e(v) -> str:
    return np.format_float_scientific(F4(v), unique=True, exp_digits=1).replace("-", "−")


def minus(s: str) -> str:
    return s.replace("-", "−")


# ------------------------------------------------------------------------------------------------ data
class Example:
    """One date, one member, 16 cells of a synthetic winter slice: FIS of terrain from 200 to 3000 m, T_2M falling
    with height (268 K - 6.5 K/km + noise, so it crosses 256 K, where the float32 spacing halves), PMSL and U_10M
    noise. Encoded with the stage ``PMSL, T_2M <- FIS``."""

    def __init__(self) -> None:
        n = 16
        rng = np.random.default_rng(7)
        h = np.sort(rng.uniform(200, 3000, n)).round(0)
        self.s = (GRAVITY * h).astype(F4)
        self.x = (268.0 - 0.0065 * h + rng.normal(0, 0.4, n)).astype(F4)
        pmsl = (101325 + rng.normal(0, 60, n)).astype(F4)
        u = rng.normal(0, 3, n).astype(F4)
        self.chunk = np.stack([self.s, pmsl, self.x, u])[None, :, None, :]
        self.stage = FieldTerms(
            variables=VARIABLES, terms={"PMSL": ["FIS"], "T_2M": ["FIS"]}, bounds=BOUNDS, margin=DEFAULT_MARGIN
        )
        self.y, self.header = self.stage.encode(self.chunk)
        rec = self.stage.records(self.header, self.chunk.shape)[0][1]  # terms in variable order: PMSL, T_2M
        assert not rec.bypass
        self.coefs = rec.coefs
        self.pred = arith.predict(rec.coefs[0], rec.coefs[1:], [self.s])
        self.r = arith.residual(self.x, self.pred)
        assert np.array_equal(self.r, self.y[0, 2, 0])
        self.b = BOUNDS["T_2M"]
        self.b_l = self.stage.lattice_bounds["T_2M"]


# ================================================================================================ figure 1
def fig_overview(ex: Example, out: Path) -> Path:
    svg = Svg(900, 712, "What the field_terms stage does")

    # (a) the chain
    panel_label(svg, 20, 28, "(a)", "first in the chain: compressors = [field_terms, static_ref, icon_lattice_bytes]")
    cw, gap, x0, y0, fh = 104, 146, 16, 74, 26
    xs = [x0 + i * (cw + gap) for i in range(4)]
    lw = 132  # the stored chunk
    colour = {
        "FIS": (STATIC_LIGHT, STATIC),
        "PMSL": (TARGET_LIGHT, TARGET),
        "T_2M": (TARGET_LIGHT, TARGET),
        "U_10M": (DECODED_LIGHT, DECODED),
    }

    def field(x, y, name, residual=False):
        fill, stroke = colour[name]
        if residual:
            svg.rect(x, y, cw, fh - 4, fill=BG, stroke=stroke, sw=1.4, dash="4 2")
            svg.text(x + cw / 2, y + 16, f"r of {name}", size=12.5, anchor="middle", mono=True)
        else:
            svg.rect(x, y, cw, fh - 4, fill=fill, stroke=stroke, sw=1.2)
            svg.text(x + cw / 2, y + 16, name, size=12.5, anchor="middle", mono=True)

    def head(x, y, label):
        svg.rect(x, y, cw, fh - 4, fill=FAINT, stroke=LINE, sw=1)
        svg.text(x + cw / 2, y + 16, label, size=12, anchor="middle")

    titles = ["chunk (one date)", "after field_terms", "after static_ref", "stored chunk"]
    for x, t, w in zip(xs, titles, (cw, cw, cw, lw), strict=True):
        svg.text(x + w / 2, y0 - 12, t, size=12.5, anchor="middle", fill=MUTED)
    for i, v in enumerate(VARIABLES):
        field(xs[0], y0 + i * fh, v)
        field(xs[1], y0 + i * fh, v, residual=v in ("PMSL", "T_2M"))
    for i, v in enumerate(VARIABLES[1:]):
        field(xs[2], y0 + i * fh, v, residual=v in ("PMSL", "T_2M"))
    yh = y0 + 4 * fh + 12
    svg.text(xs[1], yh - 2, "headers:", size=12, fill=MUTED)
    head(xs[1], yh + 4, f"field_terms, {len(ex.header)} B")
    svg.text(xs[2], yh - 2, "headers:", size=12, fill=MUTED)
    head(xs[2], yh + 4, f"field_terms, {len(ex.header)} B")
    head(xs[2], yh + 4 + fh, "static_ref: FIS sha")
    # the stored chunk
    sh = 4 * fh + 2 * fh + 12
    svg.rect(xs[3], y0, lw, sh, fill=FAINT, stroke=INK, sw=1.2, rx=4)
    svg.lines(
        xs[3] + 8,
        y0 + 18,
        [
            "envelope FCHN:",
            "the shape, the",
            "two headers,",
            "then the LATC",
            "container of",
            "r of PMSL,",
            "r of T_2M, U_10M",
        ],
        size=12,
        gap=17,
    )
    # the codecs between the columns
    codecs = [
        ("field_terms", ["targets → residuals", "r = fl32(x − pred);", "coefficients in", "its header"]),
        ("static_ref", ["drops FIS (exact,", "restored from the", "sidecar by sha256)"]),
        ("icon_lattice_bytes", ["codes the rest,", "the residuals at", "b_L = b (1 − m)"]),
    ]
    for i, (name, rows) in enumerate(codecs):
        ax0, ax1 = xs[i] + cw, xs[i + 1]
        my = y0 + 2 * fh - 2
        svg.arrow(ax0, my, ax1, my, color=INK, shrink0=4, shrink1=4)
        svg.text((ax0 + ax1) / 2, my - 8, name, size=12, anchor="middle", mono=True, weight="bold")
        svg.lines((ax0 + ax1) / 2, my + 20, rows, size=12, anchor="middle", gap=16, fill=MUTED)
    yl = y0 + sh + 26
    lx = svg.swatch(20, yl, STATIC_LIGHT, "exact source", size=12.5, stroke=STATIC)
    lx = svg.swatch(lx, yl, TARGET_LIGHT, "target x", size=12.5, stroke=TARGET)
    svg.rect(lx, yl - 11, 14, 14, fill=BG, stroke=TARGET, sw=1.4, dash="3 2")
    lx += 20
    svg.text(lx, yl, "its residual r", size=12.5)
    lx += text_width("its residual r", 12.5) + 16
    lx = svg.swatch(lx, yl, DECODED_LIGHT, "other lossy field (unchanged)", size=12.5, stroke=DECODED)
    svg.text(880, yl, "decode: right to left", size=12.5, anchor="end", fill=MUTED)

    # (b) one slice: the fit, the prediction and the residual
    yb = yl + 46
    panel_label(svg, 20, yb, "(b)", "one (date, member) slice of T_2M against its source FIS, 16 cells")
    px0, px1 = 84, 490
    smax = 30000.0

    def sx(s):
        return px0 + (float(s) / smax) * (px1 - px0)

    # scatter x against s
    t0, t1 = 248.0, 268.0
    ty0, ty1 = yb + 26, yb + 236

    def ty(v):
        return ty1 - (float(v) - t0) / (t1 - t0) * (ty1 - ty0)

    svg.line(px0, ty1, px1, ty1, stroke=LINE)
    svg.line(px0, ty0, px0, ty1, stroke=LINE)
    for v in (250, 255, 260, 265):
        svg.line(px0 - 4, ty(v), px0, ty(v), stroke=LINE)
        svg.text(px0 - 8, ty(v) + 4, f"{v}", size=12, anchor="end", fill=MUTED)
    svg.line(px0, ty(256), px1, ty(256), stroke=LINE, dash="3 3")
    svg.lines(px0 + 6, ty(256) + 16, ["256 K: float32 spacing", "2^−15 K above, 2^−16 K below"], size=12, fill=MUTED)
    svg.text(px0 - 46, (ty0 + ty1) / 2, "x = T_2M (K)", size=12.5, anchor="middle", rotate=-90)
    c0, c1 = ex.coefs
    line_s = np.array([0, smax], F4)
    line_p = arith.predict(c0, [c1], [line_s])
    svg.line(sx(0), ty(line_p[0]), sx(smax), ty(line_p[1]), stroke=TARGET, sw=2)
    svg.text(sx(14000), ty(265), "pred = fl32(c0 + c1 s)", size=12.5, fill=TARGET, weight="bold")
    for s, x, p in zip(ex.s, ex.x, ex.pred, strict=True):
        svg.line(sx(s), ty(x), sx(s), ty(p), stroke=MUTED, sw=1.2)
    for i, (s, x) in enumerate(zip(ex.s, ex.x, strict=True)):
        svg.circle(sx(s), ty(x), 3.6, fill=INK if i != CELL else BG, stroke=INK, sw=1.4)
    svg.text(sx(ex.s[CELL]) - 8, ty(ex.x[CELL]) + 16, f"cell {CELL}", size=12, fill=MUTED, anchor="end")

    # the residual against s
    r0, r1 = -1.0, 1.0
    ry0, ry1 = ty1 + 22, ty1 + 122

    def ry(v):
        return ry1 - (float(v) - r0) / (r1 - r0) * (ry1 - ry0)

    svg.line(px0, ry1, px1, ry1, stroke=LINE)
    svg.line(px0, ry0, px0, ry1, stroke=LINE)
    svg.line(px0, ry(0), px1, ry(0), stroke=TARGET, sw=1.2)
    for v in (-1, 0, 1):
        svg.line(px0 - 4, ry(v), px0, ry(v), stroke=LINE)
        svg.text(px0 - 8, ry(v) + 4, minus(f"{v:g}"), size=12, anchor="end", fill=MUTED)
    for i, (s, r) in enumerate(zip(ex.s, ex.r, strict=True)):
        svg.line(sx(s), ry(0), sx(s), ry(r), stroke=MUTED, sw=1.2)
        svg.circle(sx(s), ry(r), 3.6, fill=DECODED if i != CELL else BG, stroke=DECODED, sw=1.4)
    svg.text(px0 - 46, (ry0 + ry1) / 2, "r (K)", size=12.5, anchor="middle", rotate=-90)
    for v in (0, 10000, 20000, 30000):
        svg.line(sx(v), ry1, sx(v), ry1 + 4, stroke=LINE)
        svg.text(sx(v), ry1 + 18, f"{v:,}".replace(",", " "), size=12, anchor="middle", fill=MUTED)
    svg.text((px0 + px1) / 2, ry1 + 36, "s = FIS (m²/s²), the exact source", size=12.5, anchor="middle")

    # the numbers
    tx = 532
    k = CELL
    y = svg.lines(
        tx,
        yb + 40,
        [
            "fit: robust (IRLS) in float64 on the 16 cells,",
            "then rounded to float32:",
        ],
        size=12.5,
    )
    for a, b in (("c1", f32e(c1)), ("c0", f32(c0))):
        svg.text(tx + 12, y, f"{a} = {b}", size=12.5, mono=True)
        y += 20
    svg.text(tx + 12, y, "(c0: the mean of x − c1 s, not fitted)", size=12.5, fill=MUTED)
    y += 30
    svg.text(tx, y, minus(f"x: {f32(ex.x.min())} to {f32(ex.x.max())} K"), size=12.5)
    svg.text(tx, y + 20, minus(f"r: {f32(ex.r.min())} to {f32(ex.r.max())} K"), size=12.5)
    y += 50
    svg.text(tx, y, f"cell {k} (hollow):", size=12.5, weight="bold")
    y = svg.lines(
        tx + 12,
        y + 20,
        [
            f"s = {f32(ex.s[k])}, x = {f32(ex.x[k])}",
            f"pred = {f32(ex.pred[k])}",
            minus(f"r = fl32(x − pred) = {f32(ex.r[k])}"),
        ],
        size=12.5,
        mono=True,
        gap=19,
    )
    svg.lines(
        tx,
        y + 14,
        [
            "Header record of the slice: flags 0, c0, c1",
            "(12 bytes). r replaces x in the chunk and the",
            "lattice codec codes it within b_L = b (1 − 2^−6)",
            f"≈ {ex.b_l:.4f} K. PMSL gets its own record;",
            "FIS and U_10M pass unchanged.",
        ],
        size=12.5,
    )
    return svg.save(out / "overview.svg")


# ================================================================================================ figure 2
def fig_bound(ex: Example, out: Path) -> Path:
    k, b, b_l = CELL, ex.b, ex.b_l
    x, pred, r = ex.x[k : k + 1], ex.pred[k : k + 1], ex.r[k : k + 1]
    lo, hi = arith.endpoints(r, b_l)
    f_lo, f_hi = arith.reconstruct(lo, pred, None), arith.reconstruct(hi, pred, None)
    assert arith.verify(x, r, pred, None, b, b_l)
    _, hi0 = arith.endpoints(r, b)  # without the margin
    f_hi0 = arith.reconstruct(hi0, pred, None)
    fails = [not arith.verify(ex.x[i : i + 1], ex.r[i : i + 1], ex.pred[i : i + 1], None, b, b) for i in range(16)]
    assert fails == list(ex.x < 256)  # without the margin exactly the cells below 256 K fail
    fails0 = sum(fails)
    assert not arith.verify(x, r, pred, None, b, b)
    x64 = float(x[0])

    def ub(v):  # distance from x, in units of b
        return (float(v[0]) - x64) / b

    def rb(v):  # distance from r, in units of b
        return (float(v[0]) - float(r[0])) / b

    svg = Svg(900, 624, "The error bound of field_terms")
    umin, umax, X0, SC = -1.3, 1.3, 60, 300

    def px(u):
        return X0 + (u - umin) * SC

    def axis(y, name, dy):
        svg.line(px(umin), y, px(umax), y, stroke=LINE)
        for u in (-1, 1):
            svg.line(px(u), y - 4, px(u), y + 4, stroke=LINE)
            svg.text(
                px(u) + 4 * u,
                y + dy,
                f"{u:+g} b".replace("-", "−"),
                size=12,
                anchor="end" if u < 0 else "start",
                fill=MUTED,
            )
        svg.text(px(umax), y + 20, name, size=12.5, anchor="end", fill=MUTED)

    panel_label(svg, 20, 28, "(a)", f"cell {k} of the T_2M slice, in units of b = {b:g} K (b_L = b (1 − m), m = 2^−6)")
    # the residual domain
    ya = 104
    svg.text(20, ya - 46, "what the codec after the stage may return: any float32 r' with |r' − r| ≤ b_L", size=13)
    svg.rect(px(rb(lo)), ya - 13, (rb(hi) - rb(lo)) * SC, 26, fill=DECODED_LIGHT, stroke=DECODED, sw=1.2)
    axis(ya, "r' − r", 20)
    for v, name in ((lo, "lo"), (hi, "hi")):
        svg.line(px(rb(v)), ya - 18, px(rb(v)), ya + 18, stroke=DECODED, sw=2)
        svg.text(px(rb(v)), ya - 22, name, size=13, anchor="middle", weight="bold", fill=DECODED)
    svg.circle(px(0), ya, 5, fill=INK)
    svg.text(px(0), ya - 20, minus(f"r = fl32(x − pred) = {f32(r[0])} K"), size=12.5, anchor="middle")
    # the map
    yb = 262
    for v, fv in ((lo, f_lo), (hi, f_hi)):
        svg.arrow(px(rb(v)), ya + 20, px(ub(fv)), yb - 20, color=MUTED, sw=1.2, dash="4 3")
    svg.lines(
        px(0),
        ya + 46,
        [
            "lo, hi: r − b_L and r + b_L widened by 2^−50 (|r| + b_L), rounded outwards to float32;",
            "x' = f(r') = clip(fl32(r' + pred)) is monotone in r',",
            "so f(lo) ≤ f(r') ≤ f(hi) for every r' the codec may return",
        ],
        size=12.5,
        anchor="middle",
    )
    # the decoded domain
    svg.rect(px(-1), yb - 16, 2 * SC, 32, fill=FAINT, stroke=LIGHT, sw=1)
    for a0, a1 in ((-1, -b_l / b), (b_l / b, 1)):
        svg.rect(px(a0), yb - 16, (a1 - a0) * SC, 32, fill=LIGHT, stroke="none")
    svg.rect(px(ub(f_lo)), yb - 11, (ub(f_hi) - ub(f_lo)) * SC, 22, fill=DECODED_LIGHT, stroke=DECODED, sw=1.2)
    axis(yb, "x' − x", 26)
    svg.line(px(0), yb - 22, px(0), yb + 22, stroke=INK, sw=2)
    svg.text(px(0), yb + 40, f"x = {f32(x[0])} K", size=12.5, anchor="middle", weight="bold")
    for v, name, anc, dx in ((f_lo, "f(lo)", "start", 6), (f_hi, "f(hi)", "end", -6)):
        svg.text(px(ub(v)) + dx, yb + 5, name, size=12.5, anchor=anc, weight="bold", fill=DECODED)
    svg.text(px(1) + 8, yb - 6, "margin m b", size=12.5, fill=MUTED)
    svg.text(px(-0.5), yb - 24, "allowed: |x' − x| ≤ b", size=12.5, anchor="middle", fill=MUTED)
    # the zoom window
    z0, z1 = ub(f_hi) - 0.006, 1.006
    svg.rect(px(z0), yb - 26, (z1 - z0) * SC, 52, stroke=INK, sw=1, dash="2 2")

    # (b) zoom on the upper end, in float32 steps of x'
    ulp = float(np.spacing(x[0]))
    yz = 408
    panel_label(
        svg,
        20,
        yz - 64,
        "(b)",
        f"the dashed box: float32 values of x' near x + b (spacing 2^{minus(str(int(np.log2(ulp))))} K "
        f"= {ulp / b:.6f} b here)",
    )
    W, n = 360, 5  # window width, float32 steps shown per window
    step = W / n
    q_hi = round((float(f_hi[0]) - x64) / ulp)
    q_b = b / ulp
    windows = [(40, q_hi - n / 2), (500, np.floor(q_b) + 0.5 - n / 2)]

    def qx(win, q):
        wx, qa = windows[win]
        return wx + (q - qa) * step

    xb = qx(1, q_b)
    svg.rect(xb, yz - 26, 500 + W - xb, 52, fill=mix(TARGET, "#ffffff", 0.88))
    for win, (wx, qa) in enumerate(windows):
        svg.line(wx, yz, wx + W, yz, stroke=LINE)
        for q in range(int(np.ceil(qa)), int(np.floor(qa + n)) + 1):
            svg.line(qx(win, q), yz - 6, qx(win, q), yz + 6, stroke=INK, sw=1.2)
    gx = (40 + W + 500) / 2
    svg.text(gx, yz + 5, "…", size=16, anchor="middle")
    svg.text(gx, yz + 28, f"{(q_b - q_hi):.1f} steps", size=12, anchor="middle", fill=MUTED)
    # window 1: x + b_L, hi + pred, f(hi)
    qbl = b_l / ulp
    svg.line(qx(0, qbl), yz - 34, qx(0, qbl), yz + 34, stroke=MUTED, sw=1.4, dash="4 3")
    svg.text(qx(0, qbl), yz - 40, "x + b_L", size=12.5, anchor="middle", fill=MUTED)
    s_hi = (float(hi[0]) + float(pred[0]) - x64) / ulp  # the exact sum hi + pred
    svg.path(f"M{_f(qx(0, s_hi) - 5)},{_f(yz + 18)} l10,10 m0,-10 l-10,10", stroke=INK, sw=1.4)
    svg.text(qx(0, s_hi), yz + 48, "hi + pred (exact sum)", size=12.5, anchor="middle")
    svg.circle(qx(0, q_hi), yz, 6, fill=DECODED, stroke=DECODED)
    svg.text(qx(0, q_hi) + 4, yz - 14, "f(hi)", size=12.5, anchor="start", weight="bold", fill=DECODED)
    # window 2: x + b and the case m = 0
    svg.line(xb, yz - 34, xb, yz + 34, stroke=INK, sw=2)
    svg.text(xb, yz - 40, "x + b", size=12.5, anchor="middle", weight="bold")
    q0 = (float(f_hi0[0]) - x64) / ulp
    s0 = (float(hi0[0]) + float(pred[0]) - x64) / ulp
    svg.path(f"M{_f(qx(1, s0) - 5)},{_f(yz + 18)} l10,10 m0,-10 l-10,10", stroke=MUTED, sw=1.4)
    svg.circle(qx(1, q0), yz, 6, fill=BG, stroke=INK, sw=1.6)
    svg.text(qx(1, q0), yz - 14, "f(hi) with m = 0", size=12.5, anchor="start", weight="bold")
    svg.text(qx(1, s0), yz + 48, "hi + pred with m = 0", size=12.5, anchor="middle", fill=MUTED)
    svg.text(500 + W - 6, yz + 22, "beyond the bound", size=12.5, anchor="end", fill=TARGET)
    svg.lines(
        40,
        yz + 86,
        [
            f"With the margin: f(hi) − x = {ub(f_hi):.7f} b and x − f(lo) = {-ub(f_lo):.7f} b, both ≤ b: the cell is "
            "accepted",
            "(arith.verify checks both ends in every cell). By monotonicity |f(r') − x| ≤ b for every r' the codec "
            "may return.",
        ],
        size=12.5,
    )
    svg.lines(
        40,
        yz + 146,
        [
            f"Without it (b_L = b): x + b lies {q_b - np.floor(q_b):.1f} steps above a float32 value, so r + b + pred "
            "rounds up to the next one:",
            f"f(hi) − x = {ub(f_hi0):.7f} b > b. That fails here in the {fails0} cells below 256 K, and one failing "
            "cell bypasses the slice.",
            "The margin m b pays for these roundings: half a float32 step of x' and of r, and the widening "
            "2^−50 (|r| + b_L)",
            f"of lo and hi (the float32 spacing of r is {abs(float(np.spacing(r[0]))) / b:.1e} b here).",
        ],
        size=12.5,
    )
    return svg.save(out / "bound.svg")


# ================================================================================================ figure 3
def box(svg: Svg, x, y, w, h, title, rows, fill=FAINT, stroke=LINE, size=12.5, mono_rows=False):
    """A rounded box with a bold title and centred rows."""
    svg.rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.4, rx=6)
    svg.text(x + w / 2, y + 19, title, size=13, anchor="middle", weight="bold")
    for i, s in enumerate(rows):
        svg.text(x + w / 2, y + 37 + 16 * i, s, size=size, anchor="middle", mono=mono_rows)


def within(y: np.ndarray, bound: float, up: bool) -> np.ndarray:
    """The float32 values farthest from ``y`` (below, or above with ``up``) that are within ``bound`` of it (the
    float64 comparison): the ends of what the codec after the stage may return."""
    target = y.astype(F8) + (bound if up else -bound)
    v = target.astype(F4)
    off = v.astype(F8) > target if up else v.astype(F8) < target
    return np.where(off, np.nextafter(v, F4(-np.inf if up else np.inf)), v)


def fig_bypass(ex: Example, out: Path) -> Path:
    svg = Svg(900, 716, "Bypass and clip in field_terms")
    panel_label(svg, 20, 28, "(a)", "encode, per (date, member) slice and target")
    bh = 72
    y1, y2 = 46, 168
    box(svg, 20, y1, 170, bh, "fit and predict", ["float32 c0 .. ck,", "pred, r = fl32(x − pred)"])
    box(
        svg,
        240,
        y1,
        250,
        bh,
        "every cell verified?",
        ["c0 .. ck finite, and f(lo), f(hi)", "within b of x in every cell"],
        fill=BG,
        stroke=INK,
    )
    box(
        svg,
        560,
        y1,
        320,
        bh,
        "keep the residual",
        ["emits r in place of x;", "record: flags 0, c0 .. ck"],
        fill=TARGET_LIGHT,
        stroke=TARGET,
    )
    svg.arrow(190, y1 + bh / 2, 240, y1 + bh / 2, shrink0=2, shrink1=2)
    svg.arrow(490, y1 + bh / 2, 560, y1 + bh / 2, shrink0=2, shrink1=2)
    svg.text(525, y1 + bh / 2 - 8, "yes", size=12.5, anchor="middle", weight="bold")
    svg.text(373, y1 + bh + 22, "no (any cell fails)", size=12.5, weight="bold")
    box(
        svg,
        20,
        y2,
        250,
        bh,
        "bypass the slice",
        ["emits x untouched;", "record: flags 1 (bypass), zeros"],
        fill=DECODED_LIGHT,
        stroke=DECODED,
    )
    box(
        svg,
        330,
        y2,
        200,
        bh,
        "target clipped and",
        ["|clip(x) − x| > b", "in some cell?"],
        fill=BG,
        stroke=INK,
    )
    box(
        svg,
        600,
        y2,
        280,
        bh,
        "ValueError",
        ["data outside the clip by more", "than b: widen the clip"],
        fill=FAINT,
        stroke=INK,
    )
    svg.path(
        f"M365,{y1 + bh + 2} L365,{y1 + bh + 28} L145,{y1 + bh + 28} L145,{y2 - 2}", stroke=INK, sw=1.4, marker=INK
    )
    svg.arrow(270, y2 + bh / 2, 330, y2 + bh / 2, shrink0=2, shrink1=2)
    svg.arrow(530, y2 + bh / 2, 600, y2 + bh / 2, shrink0=2, shrink1=2)
    svg.text(565, y2 + bh / 2 - 8, "yes", size=12.5, anchor="middle", weight="bold")
    svg.arrow(430, y2 + bh, 430, y2 + bh + 30, shrink0=2)
    svg.text(438, y2 + bh + 22, "no", size=12.5, weight="bold")
    svg.text(
        430,
        y2 + bh + 48,
        "the codec after the stage codes x itself, within b_L ≤ b; other slices keep their residuals",
        size=12.5,
        anchor="middle",
    )

    # (b) decode
    yd = y2 + bh + 96
    panel_label(svg, 20, yd, "(b)", "decode, per record (the stage owns its targets' clips)")
    box(
        svg,
        20,
        yd + 14,
        420,
        58,
        "flags 0",
        ["x' = clip(fl32(r' + pred)), pred from c0 .. ck and the sources"],
        fill=TARGET_LIGHT,
        stroke=TARGET,
    )
    box(
        svg,
        460,
        yd + 14,
        420,
        58,
        "flags 1 (bypass)",
        ["x' = clip(v), v as the codec after the stage returned it"],
        fill=DECODED_LIGHT,
        stroke=DECODED,
    )

    # (c) the clip of a bypassed cell on a number line
    b = 0.05
    b_l = ex.stage.reduced_bound(b)
    lo = 0.0
    clip = (lo, None)
    yc = yd + 116
    panel_label(svg, 20, yc, "(c)", f"a bypassed cell clipped at lo, in units of b = {b:g} (b_L = b (1 − 2^−6))")
    umin, X0, SC = -2.6, 60, 180

    def px(u):
        return X0 + (u - umin) * SC

    cases = []
    for d in (0.6, 1.4):  # x below lo by d b
        x = np.array([lo - d * b], F4)
        ends = [within(x, b_l, False), within(x, b_l, True)]
        clipped = [arith.apply_clip(e.copy(), clip) for e in ends]
        err = max(abs(float(c[0]) - float(x[0])) for c in clipped)
        cases.append((x, ends, clipped, err, arith.clip_within(x, clip, b)))
    assert cases[0][4] and not cases[1][4]
    for i, (x, ends, clipped, err, ok) in enumerate(cases):
        y = yc + 76 + i * 112
        ux = (float(x[0]) - lo) / b
        svg.rect(px(umin), y - 18, (0 - umin) * SC, 36, fill=FAINT)  # clipped away
        svg.rect(px(ux - 1), y - 18, 2 * SC, 36, stroke=INK, sw=1, dash="3 3")
        svg.line(px(umin), y, px(0.45), y, stroke=LINE)
        svg.line(px(0), y - 26, px(0), y + 26, stroke=INK, sw=2)
        svg.text(px(0) + 6, y - 26, "lo", size=12.5, weight="bold")
        e0, e1 = ((float(e[0]) - lo) / b for e in ends)
        svg.rect(px(e0), y - 8, (e1 - e0) * SC, 16, fill=DECODED_LIGHT, stroke=DECODED, sw=1.2)
        c0, c1 = ((float(c[0]) - lo) / b for c in clipped)
        if c1 > c0:
            svg.rect(px(c0), y - 4, (c1 - c0) * SC, 8, fill=DECODED, stroke=DECODED)
        else:
            svg.circle(px(c0), y, 5, fill=DECODED, stroke=DECODED)
        svg.line(px(ux), y - 22, px(ux), y + 22, stroke=INK, sw=1.6)
        svg.text(px(ux), y + 38, "x", size=13, anchor="middle", weight="bold")
        svg.text(px(ux - 1) + 4, y - 24, "x ± b", size=12, fill=MUTED)
        word = "accepted" if ok else "refused at encode (ValueError)"
        svg.text(20, y - 44, minus(f"x = lo − {(lo - float(x[0])) / b:.1f} b: {word}"), size=12.5, weight="bold")
        rows = (
            [f"max |clip(v) − x| = {err / b:.3f} b", "= max(b_L, lo − x) ≤ b"]
            if ok
            else ["clip(v) = lo for every v:", f"|lo − x| = {err / b:.3f} b > b"]
        )
        svg.lines(px(0.5), y - 4, rows, size=12.5, gap=17)
    yl = yc + 76 + 112 + 66
    lx = svg.swatch(20, yl, DECODED_LIGHT, "v the codec may return (within b_L of x)", size=12.5, stroke=DECODED)
    lx = svg.swatch(lx, yl, DECODED, "clip(v)", size=12.5, stroke=DECODED)
    lx = svg.swatch(lx, yl, FAINT, "below lo: clipped", size=12.5, stroke=LIGHT)
    return svg.save(out / "bypass.svg")


# ================================================================================================ figure 4
def pinned_header() -> tuple[FieldTerms, np.ndarray, bytes]:
    """The fixed input of ``test_header_pinned_bytes`` (tests/test_field_terms.py): variables A, S, B, C over 16
    cells and two dates; ``A <- S`` with intercept (A = 3 + 2 S on date 0, 1e7 + 2 S on date 1: bypassed, the
    float32 spacing 1 exceeds the bound 0.6) and ``B <- S, C`` without intercept (B = 0.5 S + 4 C)."""
    n = 16
    s = np.arange(n, dtype=F4)
    c = (np.arange(n) * 7 % 5).astype(F4)
    x = np.zeros((2, 4, 1, n), F4)
    x[:, 1, 0], x[:, 3, 0] = s, c
    x[0, 0, 0] = 3 + 2 * s
    x[1, 0, 0] = 1e7 + 2 * s
    x[:, 2, 0] = 0.5 * s + 4 * c
    st = FieldTerms(
        variables=["A", "S", "B", "C"],
        terms={"B": {"sources": ["S", "C"], "intercept": False}, "A": ["S"]},
        bounds={"A": 0.6, "B": 0.01},
    )
    y, h = st.encode(x)
    return st, y, h


def fig_header(ex: Example, out: Path) -> Path:
    st, y, h = pinned_header()
    names = st.variables
    recs = st.records(h, y.shape)
    index = [(tm.t, tm.s) for tm in st.terms]
    assert len(h) == header.size(index, 2) == 80
    svg = Svg(900, 652, "The field_terms header, version 1")
    panel_label(svg, 20, 28, "(a)", f"the header of a two-date chunk, {len(h)} bytes, little endian")
    svg.text(
        20,
        50,
        "variables A, S, B, C; terms A ← S (intercept) and B ← S, C (no intercept); A is bypassed on date 1",
        size=12.5,
        fill=MUTED,
    )
    B, x0 = 26, 136
    coef = TARGET_LIGHT

    def row(y, label, start, groups, captions):
        """groups: (offset, bytes, name, value, fill); captions: (offset, text)."""
        svg.text(20, y + 20, label, size=12.5, weight="bold")
        for off, text in captions:
            svg.text(x0 + (off - start) * B, y - 24, text, size=12.5)
        for off, nb, name, value, fill in groups:
            gx = x0 + (off - start) * B
            for j in range(nb):
                svg.rect(gx + j * B, y, B, 30, fill=fill, stroke=LIGHT, sw=0.8)
                svg.text(gx + j * B + B / 2, y + 20, f"{h[off + j]:02x}", size=12, anchor="middle", mono=True)
            svg.rect(gx, y, nb * B, 30, stroke=INK, sw=1.3)
            svg.text(gx + 1, y - 6, str(off), size=12, fill=MUTED)
            if name:
                svg.text(gx + nb * B / 2, y + 47, name, size=12, anchor="middle")
            if value:
                svg.text(gx + nb * B / 2, y + 62, value, size=12, anchor="middle", fill=MUTED)

    # head and term table
    pos = 8
    table = []
    for t, srcs in index:
        table += [(pos, 2, "target", f"{t} ({names[t]})", FAINT), (pos + 2, 2, "k", str(len(srcs)), FAINT)]
        pos += 4
        for i in srcs:
            table.append((pos, 2, "source", f"{i} ({names[i]})", FAINT))
            pos += 2
    pad = -pos % 4
    table.append((pos, pad, "pad", "", BG))
    head = [
        (0, 1, "version", "1", FAINT),
        (1, 1, "", "", BG),
        (2, 2, "T (terms)", str(len(index)), FAINT),
        (4, 4, "S (slices)", "2", FAINT),
    ]
    yr = 104
    row(
        yr,
        "head, terms",
        0,
        head + table,
        [(0, "u8, u8 0, u16, u32"), (8, "term table: u16 target, u16 k, k × u16 source")],
    )
    rstart = pos + pad
    assert rstart == 24
    # the records, one row per slice
    for sl in range(2):
        groups, captions = [], []
        off = rstart + sl * 28
        for (t, srcs), rec in zip(index, recs[sl], strict=True):
            captions.append(
                (off, f"record of {names[t]}: {4 + 4 * (1 + len(srcs))} B" + (" (bypassed)" if rec.bypass else ""))
            )
            flag_fill = DECODED_LIGHT if rec.bypass else FAINT
            groups += [(off, 1, "flags", str(int(rec.bypass)), flag_fill), (off + 1, 3, "", "", BG)]
            off += 4
            for j, cf in enumerate(rec.coefs):
                groups.append((off, 4, f"c{j} (f32)", minus(f32(cf)), coef))
                off += 4
        row(yr + 108 * (sl + 1), f"slice {sl} (date {sl})", rstart + sl * 28, groups, captions)
    assert off == len(h)
    yl = yr + 3 * 108 - 18
    lx = svg.swatch(20, yl, FAINT, "fields", size=12.5, stroke=LINE)
    lx = svg.swatch(lx, yl, BG, "zero (reserved, padding)", size=12.5, stroke=LINE)
    lx = svg.swatch(lx, yl, coef, "coefficients: pred = c0 + c1 s1 + .. + ck sk", size=12.5, stroke=LINE)
    svg.swatch(lx, yl, DECODED_LIGHT, "flags bit 0: bypass", size=12.5, stroke=LINE)

    # (b) in the chunk: the envelope's header record
    env = bytes(envelope.pack(y, [("field_terms", h)]))
    rec0 = 24 + 8 * y.ndim
    n_id, zero, m = np.frombuffer(env, "<u2", 2, rec0).tolist() + [int(np.frombuffer(env, "<u4", 1, rec0 + 4)[0])]
    assert (n_id, zero, m) == (11, 0, len(h)) and env[rec0 + 8 : rec0 + 19] == b"field_terms"
    rec_len = 8 + n_id + m
    padb = -rec_len % 8
    yb = yl + 50
    panel_label(svg, 20, yb, "(b)", "in the chunk: one header record of the chain envelope (envelope.md)")
    Bb = 8.0
    segs = [
        (2, "n", FAINT),
        (2, "0", FAINT),
        (4, "m", FAINT),
        (n_id, "field_terms", FAINT),
        (m, f"the header above, {m} B", coef),
        (padb, "", BG),
    ]
    xx, ys = 20, yb + 40
    for nb, val, fill in segs:
        w = nb * Bb
        svg.rect(xx, ys, w, 30, fill=fill, stroke=INK, sw=1)
        if w > 50:
            svg.text(xx + w / 2, ys + 20, val, size=12, anchor="middle", mono=val == "field_terms")
        xx += w
    svg.text(20, ys - 8, f"u16 n = {n_id}, u16 0, u32 m = {m}", size=12, fill=MUTED)
    svg.text(xx, ys - 8, f"pad, {padb} B", size=12, fill=MUTED, anchor="end")
    svg.text(
        20,
        ys + 50,
        "to scale: n bytes of stage id, m bytes of header, then zero padding to a multiple of 8; the envelope keeps "
        "one such record per stage",
        size=12.5,
    )

    yf = ys + 76
    svg.rect(20, yf, 860, 50, fill=FAINT, stroke=INK, sw=1.2, rx=6)
    svg.text(
        32,
        yf + 20,
        "frozen (header version 1): every offset, type and order above, the flag bit, and the decoder arithmetic.",
        size=12.5,
        weight="bold",
    )
    svg.text(
        32,
        yf + 38,
        "not frozen: the coefficients the encoder writes (its fit) and which slices it bypasses.",
        size=12.5,
    )
    return svg.save(out / "header.svg")


# ================================================================================================ figure 5
def fig_flow(ex: Example, out: Path) -> Path:
    svg = Svg(900, 530, "field_terms, encode and decode")
    bw, bh, gap, x0 = 128, 118, 17, 20
    xs = [x0 + i * (bw + gap) for i in range(6)]
    ye, yd = 52, 340

    def card(x, y, title, rows, fill, stroke):
        box(svg, x, y, bw, bh, title, rows, fill=fill, stroke=stroke, size=12)

    svg.text(
        20,
        28,
        "encoder, per (date, member) slice and target x with bound b and sources s_1 .. s_k",
        size=14,
        weight="bold",
    )
    enc = [
        ("inputs", ["x, s_1 .. s_k:", "float32 cells,", "finite (else", "ValueError)"], FAINT, LINE),
        (
            "fit",
            [
                "float64, all cells:",
                "least squares, then",
                f"{arith.IRLS_ITERATIONS} reweighted steps,",
                "weights 1 / max(",
                "|residual|, b / 2)",
            ],
            FAINT,
            LINE,
        ),
        ("coefficients", ["slopes → float32;", "c0 = fl32(mean of", "x − Σ c_i s_i)", "(or 0)"], TARGET_LIGHT, TARGET),
        ("predict", ["p = c0, then", "p = p + c_i s_i", "float64, in order", "pred = fl32(p)"], TARGET_LIGHT, TARGET),
        ("residual", ["r = fl32(x − pred)", "float32", "subtraction"], FAINT, LINE),
        ("verify", ["lo, hi: ends of", "r ± b_L, outwards;", "|f(lo) − x| ≤ b and", "|f(hi) − x| ≤ b"], BG, INK),
    ]
    for (t, rows, fill, stroke), x in zip(enc, xs, strict=True):
        card(x, ye, t, rows, fill, stroke)
    for i in range(5):
        svg.arrow(xs[i] + bw, ye + bh / 2, xs[i + 1], ye + bh / 2, shrink0=2, shrink1=2)
    # the outcome of the verification
    vx = xs[5] + bw / 2
    yo = ye + bh + 52
    svg.arrow(vx, ye + bh, vx, yo - 14, shrink0=2)
    svg.text(vx + 6, ye + bh + 22, "every cell", size=12, fill=MUTED)
    svg.lines(
        xs[5] + bw,
        yo + 2,
        ["pass: emit r, record flags 0, c0 .. ck", "fail: emit x, record flags 1 (bypass)"],
        size=12.5,
        anchor="end",
    )
    svg.lines(
        20,
        yo + 2,
        [
            "Each slice is fitted on its own; the records of all slices (date-major, then",
            "member, then term) form the header, which travels in the chain envelope.",
        ],
        size=12.5,
        fill=MUTED,
    )
    # the shared prediction
    svg.line(xs[3] + bw / 2, ye + bh, xs[3] + bw / 2, yd, stroke=TARGET, sw=1.2, dash="4 3")
    svg.text(xs[3] + bw / 2 + 6, yd - 64, "the same code", size=12, fill=TARGET)
    svg.text(xs[3] + bw / 2 + 6, yd - 49, "(arith.predict)", size=12, fill=TARGET)

    svg.text(
        20,
        yd - 22,
        "decoder, in place on the array the codecs after the stage returned: no fit",
        size=14,
        weight="bold",
    )
    dec = [
        ("record", ["from the header:", "flags, c0 .. ck", "(float32)"], TARGET_LIGHT, TARGET),
        (
            "sources",
            ["s_1 .. s_k from the", "same array: exact,", "bit-identical to", "the encoder's"],
            STATIC_LIGHT,
            STATIC,
        ),
        ("predict", ["as in the encoder:", "pred identical,", "bit for bit"], TARGET_LIGHT, TARGET),
        ("reconstruct", ["x' = fl32(r' + pred)", "float32 addition;", "r' within b_L of r"], FAINT, LINE),
        ("clip", ["max(x', fl32(lo)),", "then min(x',", "fl32(hi)), when", "configured"], FAINT, LINE),
        ("output", ["x' in place of r',", "|x' − x| ≤ b"], DECODED_LIGHT, DECODED),
    ]
    for (t, rows, fill, stroke), x in zip(dec, xs, strict=True):
        card(x, yd, t, rows, fill, stroke)
    for i in (1, 2, 3, 4):
        svg.arrow(xs[i] + bw, yd + bh / 2, xs[i + 1], yd + bh / 2, shrink0=2, shrink1=2)
    svg.path(
        f"M{_f(xs[0] + bw / 2)},{_f(yd)} L{_f(xs[0] + bw / 2)},{_f(yd - 10)} L{_f(xs[2] + 30)},{_f(yd - 10)} "
        f"L{_f(xs[2] + 30)},{_f(yd - 2)}",
        stroke=TARGET,
        sw=1.4,
        marker=TARGET,
    )
    # the bypass path
    yb = yd + bh + 26
    svg.path(
        f"M{_f(xs[0] + bw / 2)},{_f(yd + bh + 2)} L{_f(xs[0] + bw / 2)},{_f(yb)} L{_f(xs[4] + bw / 2)},{_f(yb)} "
        f"L{_f(xs[4] + bw / 2)},{_f(yd + bh + 2)}",
        stroke=DECODED,
        sw=1.4,
        marker=DECODED,
    )
    svg.text(
        (xs[0] + xs[4] + bw) / 2,
        yb + 18,
        "flags 1 (bypass): the value the codec after the stage returned is only clipped",
        size=12.5,
        anchor="middle",
    )
    return svg.save(out / "flow.svg")


# ================================================================================================ main
FIGURES = ("overview", "bound", "bypass", "header", "flow")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--only", nargs="*", choices=FIGURES)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    ex = Example()
    for name in a.only or FIGURES:
        t0 = time.perf_counter()
        fn = globals()[f"fig_{name}"]
        path = fn(ex, a.out)
        print(f"{path.relative_to(REPO) if path.is_relative_to(REPO) else path}: {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
