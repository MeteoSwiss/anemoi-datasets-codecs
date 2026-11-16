"""The figures of docs/benchmarks.md, drawn from the numbers transcribed in benchmarks_data.json.

    python scripts/figures/benchmarks_figures.py [--out DIR] [--only NAME ...]

Writes plain SVG (real text, generic fonts, an opaque light background) into docs/benchmarks/figures/, with the
helpers and the palette of ``icon_lattice_figures.py``:

* ``size.svg``: bytes per ICON-CH1 date, raw float32 and Blosc lz4 against the single codec and the chain (the
  ratio to Blosc at every bar), and the structure / ``k`` trade-off (size against the maximum error).
* ``decode.svg``: decode time per date, the C decoder alone (AVX2 and scalar, 1 and 16 threads), the zarr read of a
  date (single codec, chain, Blosc lz4; 1 and 16 threads) and where the chain's time goes.
* ``encode.svg``: encode time per date (``LatticeCodec.encode`` at 1 and 16 threads, range over runs) and
  ``anemoi-datasets create`` of three dates.
* ``physics.svg``: what each part of the physics layer is worth: the growth of the chunk when it is turned off.

The data file ``benchmarks_data.json`` repeats the tables of docs/benchmarks.md with the section, the machine and
the conditions of every series; the document is the record, the figures only highlight. Timings are comparable only
within one run, so every panel draws one run; sizes are deterministic (for a package version, configuration and
numpy) and may be compared across runs. Deterministic output: rerun after an edit of the data file and commit the diff.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from icon_lattice_figures import (  # noqa: E402
    BG,
    DECODED,
    DECODED_LIGHT,
    FAINT,
    LINE,
    MUTED,
    RECIPE,
    STATIC,
    TARGET,
    Svg,
    _f,
    mix,
    panel_label,
    text_width,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "benchmarks" / "figures"
DATA = Path(__file__).resolve().parent / "benchmarks_data.json"
W = 900
BAR = 18  # bar thickness
ROW = 26  # row pitch of single bars
DECODED_MID = mix(DECODED, "#ffffff", 0.45)  # second series next to DECODED (lighter, same hue)
RECIPE_LIGHT = mix(RECIPE, "#ffffff", 0.7)


def load() -> dict:
    return json.loads(DATA.read_text())


# ------------------------------------------------------------------------------------------------ chart helpers
def bar(svg: Svg, x: float, y: float, w: float, h: float, fill: str, r: float = 4.0, round_end: bool = True) -> None:
    """A horizontal bar from ``x`` (square at the baseline) with a rounded data end."""
    if w <= 0:
        return
    if w <= r or not round_end:
        svg.rect(x, y, w, h, fill=fill)
        return
    x1 = x + w
    d = (
        f"M{_f(x)},{_f(y)} H{_f(x1 - r)} Q{_f(x1)},{_f(y)} {_f(x1)},{_f(y + r)} V{_f(y + h - r)} "
        f"Q{_f(x1)},{_f(y + h)} {_f(x1 - r)},{_f(y + h)} H{_f(x)} Z"
    )
    svg.path(d, fill=fill)


class Axis:
    """A linear horizontal axis from zero: pixel mapping, recessive gridlines and tick labels."""

    def __init__(self, x0: float, x1: float, vmax: float, step: float, unit: str) -> None:
        self.x0, self.x1, self.vmax, self.step, self.unit = x0, x1, vmax, step, unit

    def px(self, v: float) -> float:
        return self.x0 + (self.x1 - self.x0) * v / self.vmax

    def grid(self, svg: Svg, ytop: float, ybot: float) -> None:
        v = 0.0
        while v <= self.vmax + 1e-9:
            x = self.px(v)
            svg.line(x, ytop, x, ybot, stroke=LINE if v == 0 else FAINT, sw=1)
            v += self.step

    def ticks(self, svg: Svg, y: float) -> None:
        v = 0.0
        while v <= self.vmax + 1e-9:
            svg.text(self.px(v), y, f"{v:g}", size=12, anchor="middle", fill=MUTED)
            last = f"{v:g}"
            v += self.step
        svg.text(self.x1 + text_width(last, 12) / 2 + 6, y, self.unit, size=12, fill=MUTED)


def header(svg: Svg, title: str, subtitle: str) -> float:
    svg.text(20, 26, title, size=16, weight="bold")
    svg.text(20, 46, subtitle, size=12.5, fill=MUTED)
    return 46


def wrap(text: str, width: float, size: float) -> list[str]:
    """Greedy word wrap of ``text`` to ``width`` px at ``size`` (text_width's estimate)."""
    rows, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}".strip()
        if cur and text_width(cand, size) > width:
            rows.append(cur)
            cur = word
        else:
            cur = cand
    return rows + [cur] if cur else rows


def footnote(svg: Svg, y: float, paragraphs: list[str]) -> None:
    """Muted 12 px paragraphs, wrapped to the figure width, closing the figure (its height ends after them)."""
    rows = [r for p in paragraphs for r in wrap(p, W - 40, 12)]
    svg.h = svg.lines(20, y, rows, size=12, gap=16, fill=MUTED) + 4


def range_bar(svg: Svg, ax: Axis, y: float, best: float, median: float, fill: str, light: str, label: str) -> None:
    """Solid bar to ``best``, a lighter extension to ``median`` (2 px surface gap), the label after the end."""
    xb, xm = ax.px(best), ax.px(median)
    bar(svg, ax.x0, y, xb - ax.x0, BAR, fill, round_end=False)
    bar(svg, xb + 2, y, xm - xb - 2, BAR, light)
    svg.text(max(xm, xb) + 8, y + BAR - 5, label, size=12.5)


def legend(svg: Svg, x: float, y: float, items: list[tuple[str, str]], size: float = 12.5) -> float:
    """Swatches in a row from ``x``, wrapping to a new row at the right margin; returns the last baseline."""
    x0 = x
    for color, label in items:
        if x > x0 and x + 18 + text_width(label, size) > W - 20:
            x, y = x0, y + 20
        x = svg.swatch(x, y, color, label, size=size, w=12, h=12)
    return y


# ================================================================================================ size
def fig_size(d: dict, out: Path) -> Path:
    s, st = d["sizes_date6"], d["structures"]
    blosc = next(r["mib"] for r in s["rows"] if r["key"] == "blosc")
    svg = Svg(W, 0, "Size per ICON-CH1 date: the codecs against Blosc lz4 and raw float32")
    header(
        svg,
        "Size per date, ICON-CH1 (115 variables, 1,147,980 cells), sample date 6",
        "Bytes of one chunk (one date). Sizes are deterministic and comparable across runs; one store per bar.",
    )
    # (a) bytes per date
    panel_label(svg, 20, 78, "(a)", "MiB per date, with the ratio to Blosc lz4")
    rows = [
        ("raw", "raw float32 (no codec)", LINE),
        ("blosc", "Blosc lz4, clevel 5, byte shuffle", RECIPE),
        ("single_tree_k1", "single codec · tree, k = 1", DECODED),
        ("chain_tree_k1", "chain · tree, k = 1", DECODED),
        ("single_tree_b_k3", "single codec · tree_b, k = 3", DECODED),
        ("chain_tree_b_k3", "chain · tree_b, k = 3", DECODED),
        ("defaults_tree_k1", "single · the defaults, tree, k = 1", DECODED_MID),
        ("recipes_tree_k1", "single, with dates · tree, k = 1", DECODED_MID),
        ("recipes_tree_b_k3", "single, with dates · tree_b, k = 3", DECODED_MID),
    ]
    by = {r["key"]: r for r in s["rows"]}
    ax = Axis(262, 690, 520, 100, "MiB")
    ytop = 92
    ybot = ytop + len(rows) * ROW + 4
    ax.grid(svg, ytop, ybot)
    for i, (key, label, color) in enumerate(rows):
        y = ytop + 6 + i * ROW
        mib = by[key]["mib"]
        svg.text(ax.x0 - 10, y + BAR - 5, label, size=12.5, anchor="end")
        bar(svg, ax.x0, y, ax.px(mib) - ax.x0, BAR, color)
        if key == "raw":
            tip = f"{mib} MiB"
        elif key == "blosc":
            tip = f"{mib} MiB (the reference, 1x)"
        else:
            tip = f"{mib} MiB ({blosc / mib:.1f}x smaller than Blosc)"
        svg.text(ax.px(mib) + 8, y + BAR - 5, tip, size=12.5)
    ax.ticks(svg, ybot + 16)
    yl = legend(
        svg,
        ax.x0,
        ybot + 36,
        [
            (DECODED, "rounding: grid, recipe_fit: false, no valid dates (zarr): forcings lossless"),
            (DECODED_MID, "forcings as recipes: the defaults, or rounding: grid with dates"),
        ],
    )
    # (b) structure and k
    y0 = yl + 34
    panel_label(svg, 20, y0, "(b)", "Structure and k: size against the maximum error, mean over the 20 sample dates")
    px0, px1, py0, py1 = 70, 700, y0 + 20, y0 + 210
    sx = Axis(px0, px1, 60, 10, "MiB per date")
    ymax = 1.2

    def py(v: float) -> float:
        return py1 - (py1 - py0) * v / ymax

    for v in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        svg.line(px0, py(v), px1, py(v), stroke=LINE if v == 0 else FAINT, sw=1)
        svg.text(px0 - 8, py(v) + 4, f"{v:g}", size=12, anchor="end", fill=MUTED)
    sx.grid(svg, py0, py1)
    sx.ticks(svg, py1 + 18)
    svg.text(px0 + 10, py0 + 16, "worst error / bound", size=12, fill=MUTED)
    # label placement: (dx, dy, anchor) per point, chosen so that no two labels touch
    place = {
        ("tree", 1): (12, 6, "start"),
        ("tree_b", 1): (0, -16, "middle"),
        ("hc2_k3", 1): (-10, 24, "end"),
        ("tree_b", 3): (-10, -10, "end"),
        ("tree_b", 5): (-10, 18, "end"),
    }
    for r in st["rows"]:
        x, y = sx.px(r["mib"]), py(r["worst_error_over_bound"])
        svg.circle(x, y, 7, fill=BG)
        svg.circle(x, y, 5, fill=DECODED)
        dx, dy, anchor = place[(r["structure"], r["k"])]
        label = f"{r['structure']}, k = {r['k']}: {r['mib']:.2f} MiB"
        svg.text(x + dx, y + dy, label, size=12.5, anchor=anchor)
    svg.lines(
        px0 + 12,
        py1 - 104,
        ["k divides the maximum error (bound / k);", "the bytes it costs are the trade-off."],
        size=12.5,
        gap=16,
        fill=MUTED,
    )
    footnote(
        svg,
        py1 + 50,
        [
            "(a) Zarr writes of the codec-chains run (rounding: grid, forcings lossless); the defaults (rounding: "
            "direct, physics.recipe_fit: true) from the configuration-choices run through the engine without dates "
            "(the bytes a zarr write gives); the two 'with dates' bars from the physics-layer run (rounding: grid, "
            "scripts/physics_size.py); raw float32 from the shape. The ratio is to the 298.95 MiB of the Blosc store "
            "of date 6.",
            "(b) Structure table: rounding: grid, forcings as recipes, statics referenced; the worst error / bound "
            "is 1 / k in every row. The RMSE / bound of each configuration is in the Configuration choices and "
            "Physics layer sections.",
        ],
    )
    return svg.save(out / "size.svg")


# ================================================================================================ decode
def fig_decode(d: dict, out: Path) -> Path:
    bd, zr, cs = d["bench_decode"], d["zarr_read"], d["chain_stages"]
    svg = Svg(W, 0, "Decode time per ICON-CH1 date")
    header(
        svg,
        "Decode time per date, ICON-CH1 sample (AMD EPYC 7713)",
        "Solid bar: best; lighter extension: median. Each panel is one run; timings from different runs move by 5-10%.",
    )
    # (a) the C decoder alone
    panel_label(svg, 20, 78, "(a)", "The C decoder alone, bench_decode (sample date 0)")
    ax = Axis(150, 690, 3600, 600, "ms")
    ytop = 92
    ybot = ytop + 4 * ROW + 4
    ax.grid(svg, ytop, ybot)
    for i, r in enumerate(bd["rows"]):
        y = ytop + 6 + i * ROW
        best, med = r["best_ms"][0], r["median_ms"][0]
        label = f"{r['variant']}, {r['threads']} thread{'s' if r['threads'] > 1 else ''}"
        svg.text(ax.x0 - 10, y + BAR - 5, label, size=12.5, anchor="end")
        range_bar(svg, ax, y, best, med, DECODED, DECODED_LIGHT, f"{best} / {med} ms")
    ax.ticks(svg, ybot + 16)
    # (b) zarr read
    y0 = 258
    panel_label(svg, 20, y0, "(b)", "Zarr read of sample date 6, every stage included (bench_chain)")
    ax = Axis(190, 690, 800, 100, "ms")
    ytop = y0 + 14
    groups = [("1 thread", "t1"), ("16 threads", "t16")]
    n = len(zr["rows"])
    ybot = ytop + 2 * (n * ROW + 22) + 4
    ax.grid(svg, ytop, ybot)
    y = ytop + 6
    for gname, gkey in groups:
        svg.text(ax.x0 - 10, y + 12, gname, size=12.5, anchor="end", weight="bold", fill=MUTED)
        y += 22
        for r in zr["rows"]:
            ref = r["store"].startswith("Blosc")
            v = r[gkey]
            svg.text(ax.x0 - 10, y + BAR - 5, r["store"], size=12.5, anchor="end")
            range_bar(
                svg,
                ax,
                y,
                v["best_ms"],
                v["median_ms"],
                RECIPE if ref else DECODED,
                RECIPE_LIGHT if ref else DECODED_LIGHT,
                f"{v['best_ms']} / {v['median_ms']} ms",
            )
            y += ROW
    ax.ticks(svg, ybot + 16)
    legend(
        svg,
        ax.x0,
        ybot + 42,
        [
            (DECODED, "best"),
            (DECODED_LIGHT, "median (extension)"),
            (RECIPE, "Blosc lz4, the reference (reads 298.95 MiB instead of 36-55)"),
        ],
    )
    # (c) chain stages
    y0 = ybot + 76
    panel_label(
        svg,
        20,
        y0,
        "(c)",
        "Where the chain's time goes, tree k = 1, one profiled read",
    )
    ax = Axis(190, 690, 600, 100, "ms")
    ytop = y0 + 14
    ybot = ytop + 2 * ROW + 4
    ax.grid(svg, ytop, ybot)
    colors = [DECODED, STATIC, TARGET]
    for i, (gname, key) in enumerate([("1 thread", "t1_ms"), ("16 threads", "t16_ms")]):
        y = ytop + 6 + i * ROW
        svg.text(ax.x0 - 10, y + BAR - 5, gname, size=12.5, anchor="end")
        x = ax.x0
        parts = []
        for j, stg in enumerate(cs["stages"]):
            v = stg[key]
            w = ax.px(v) - ax.x0
            last = j == len(cs["stages"]) - 1
            bar(svg, x, y, w - (0 if last else 2), BAR, colors[j], round_end=last)
            label = str(v)
            if text_width(label, 12) + 10 < w:
                svg.text(x + w / 2 - (0 if last else 1), y + BAR - 5, label, size=12, anchor="middle", fill=BG)
            parts.append(label)
            x += w
        svg.text(x + 8, y + BAR - 5, f"{' + '.join(parts)} = {sum(s[key] for s in cs['stages'])} ms", size=12.5)
    ax.ticks(svg, ybot + 16)
    yl = legend(
        svg,
        ax.x0,
        ybot + 42,
        [
            (DECODED, "icon_lattice_bytes (102 fields)"),
            (STATIC, "static_ref (one write of the 115-field date)"),
            (TARGET, "field_terms (six targets, in place)"),
        ],
    )
    footnote(
        svg,
        yl + 30,
        [
            "(a) One real chunk written at the defaults (31.2 MiB, 115 fields) decoded into a preallocated "
            "buffer, 3 interleaved rounds in separate processes. The Blosc library decodes a date in about 90 ms "
            "on one core and 13-26 ms on 16-32 threads.",
            "(b) Stores written at the defaults, the chains without a forcing bound; a[...] after two warm-up "
            "reads, best / median of 3 rounds of 7, each round and thread count in its own process; Blosc with "
            "as many threads. Every store decodes to the same bits at every thread count.",
            "(c) The stages of one read of an earlier run (rounding: grid); the envelope is well under 1 ms. "
            "static_ref is single-threaded numpy, so at 16 threads it is most of the chain's extra time over "
            "the single codec.",
        ],
    )
    return svg.save(out / "decode.svg")


# ================================================================================================ encode
def fig_encode(d: dict, out: Path) -> Path:
    en, ac = d["encode"], d["anemoi_create"]
    svg = Svg(W, 0, "Encode time per ICON-CH1 date")
    header(
        svg,
        "Encode time per date, ICON-CH1",
        "Python / numpy encoder with C passes, fitting and verification included.",
    )
    # (a) LatticeCodec.encode
    panel_label(svg, 20, 78, "(a)", "LatticeCodec.encode of one sample date: range over its encodes")
    ax = Axis(190, 690, 60, 10, "s")
    ytop = 92
    groups = [("1 thread", "t1_s"), ("16 threads", "t16_s")]
    n = len(en["rows"])
    ybot = ytop + 2 * (n * ROW + 22) + 4
    ax.grid(svg, ytop, ybot)
    y = ytop + 6
    for gname, gkey in groups:
        svg.text(ax.x0 - 10, y + 12, gname, size=12.5, anchor="end", weight="bold", fill=MUTED)
        y += 22
        for r in en["rows"]:
            lo, hi = r[gkey]
            svg.text(ax.x0 - 10, y + BAR - 5, f"date {r['date']}", size=12.5, anchor="end")
            label = f"{lo:g} s" if lo == hi else f"{lo:g}-{hi:g} s"
            range_bar(svg, ax, y, lo, hi, DECODED, DECODED_LIGHT, label)
            y += ROW
    ax.ticks(svg, ybot + 16)
    yl = legend(svg, ax.x0, ybot + 42, [(DECODED, "fastest encode"), (DECODED_LIGHT, "slowest encode (extension)")])
    # (b) anemoi create
    y0 = yl + 34
    panel_label(svg, 20, y0, "(b)", "anemoi-datasets create of sample dates 0-2, tree k = 1, 16 codec threads")
    ax = Axis(190, 690, 600, 100, "s")
    ytop = y0 + 14
    ybot = ytop + 2 * ROW + 4
    ax.grid(svg, ytop, ybot)
    for i, r in enumerate(ac["rows"]):
        y = ytop + 6 + i * ROW
        svg.text(ax.x0 - 10, y + BAR - 5, r["store"], size=12.5, anchor="end")
        bar(svg, ax.x0, y, ax.px(r["create_s"]) - ax.x0, BAR, DECODED)
        label = f"{r['create_s']} s for 3 dates; open_dataset + read {r['read_s']} s"
        svg.text(ax.px(r["create_s"]) + 8, y + BAR - 5, label, size=12.5)
    ax.ticks(svg, ybot + 16)
    footnote(
        svg,
        ybot + 44,
        [
            "(a) A direct call (no zarr) at the defaults, one run on 32 reserved cores of a shared node. At 16 "
            "threads every date is encoded without and with its date; at one thread once (date 0 is the first "
            "encode of the process).",
            "(b) Wall-clock time of the command (init, load, finalise, patch, cleanup, verify); anemoi's own "
            "report of the create is 82 s (chain) and 106 s (single codec), most of it loading the source field "
            "by field. The read is open_dataset in a fresh process with the cache off (cold); both stores "
            "within their bounds (worst error / bound 0.99998).",
        ],
    )
    return svg.save(out / "encode.svg")


# ================================================================================================ physics
def fig_physics(d: dict, out: Path) -> Path:
    ph = d["physics"]
    svg = Svg(W, 0, "The physics layer: growth of the chunk with each part turned off")
    c0, c1 = ph["columns"]
    m0, m1 = ph["default_mib"]
    header(
        svg,
        "The physics layer, ICON-CH1, sample date 6 with dates: what each part is worth",
        f"Growth of the chunk when a part is turned off. With everything on: {m0} MiB ({c0}), {m1} MiB ({c1}).",
    )
    ax = Axis(250, 690, 14, 2, "MiB more per date")
    order = [
        ("one kind of cross term off", None),
        ("surface_terms: false", 0),
        ("level_terms: false", 1),
        ("hydrostatic: false", 2),
        ("all cross terms off", None),
        ("cross: off", 3),
        ("the forcings stored losslessly", None),
        ("forcings: false", 4),
        ("everything off", None),
        ("physics: off", 5),
    ]
    ytop = 70
    h = 15
    pitch = 2 * h + 2 + 8
    y = ytop + 4
    ys = []
    for label, idx in order:
        if idx is None:
            ys.append((label, None, y))
            y += 20
        else:
            ys.append((label, idx, y))
            y += pitch
    ybot = y
    ax.grid(svg, ytop, ybot)
    for label, idx, y in ys:
        if idx is None:
            svg.text(ax.x0 - 10, y + 12, label, size=12, anchor="end", fill=MUTED, italic=True)
            continue
        r = ph["rows"][idx]
        svg.text(ax.x0 - 10, y + h + 6, label, size=12.5, anchor="end", mono=True)
        for j, (dv, mib, color) in enumerate(zip(r["delta"], r["mib"], (DECODED, DECODED_MID), strict=True)):
            yy = y + j * (h + 2)
            bar(svg, ax.x0, yy, ax.px(dv) - ax.x0, h, color)
            base = ph["default_mib"][j]
            svg.text(ax.px(dv) + 8, yy + h - 3, f"+{dv} MiB ({dv / base * 100:.0f}% of the chunk, {mib} MiB)", size=12)
    ax.ticks(svg, ybot + 16)
    legend(svg, ax.x0, ybot + 44, [(DECODED, c0), (DECODED_MID, c1)])
    footnote(
        svg,
        ybot + 72,
        [
            "Physics layer table (scripts/physics_size.py); every chunk decoded and checked against the bounds.",
            "The settings nest: cross: off removes the three kinds above it, physics: off everything. Without "
            "valid dates, recipe_fit or a bound for the forcings, the forcings are lossless anyway, so there the "
            "layer is worth the cross terms, 11-12% of a chunk. The worst error / bound is the same in every "
            "setting.",
        ],
    )
    return svg.save(out / "physics.svg")


# ================================================================================================ main
FIGURES = ("size", "decode", "encode", "physics")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--only", nargs="*", choices=FIGURES)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    d = load()
    for name in a.only or FIGURES:
        path = {"size": fig_size, "decode": fig_decode, "encode": fig_encode, "physics": fig_physics}[name](d, a.out)
        print(path.relative_to(REPO) if path.is_relative_to(REPO) else path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
