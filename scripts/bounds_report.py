"""Are the bounds of a configuration reasonable? Every lossy variable's bound against the scales of its own data.

    python scripts/bounds_report.py stats STORE CONFIG STATS.json [--dates SPEC] [--no-tendency]
    python scripts/bounds_report.py encode STORE CONFIG ENCODE.json [--dates SPEC] [--scales 0.5,1,2,4]
    python scripts/bounds_report.py draw STATS.json OUT.svg [--encode ENCODE.json] [--png OUT.png] [--subtitle S]

``STORE`` is an anemoi dataset (Zarr, ``data`` of shape ``(dates, variables, ensembles, cells)``, its ``dates``,
optionally its ``stdev`` statistics); ``CONFIG`` a lattice codec configuration (a JSON file, a codec entry or a store
written with the codec, as ``anemoi-datasets-codecs plan`` takes it). ``--dates``: indices into the store, ``i,j,k``,
``a:b`` or ``a:b:step`` (default: 12 evenly spaced dates for ``stats``, 2 for ``encode``).

``stats`` (cheap: a few seconds per date with 16 threads) computes per lossy variable, over the given dates:

* ``ulp``: the float32 spacing at the variable's largest magnitude, and ``quantum``: the smallest step between
  two distinct values (data packed on a grid of values, as GRIB packing does, show their packing step here). The
  larger of the two is the data's resolution: a bound below it buys nothing.
* ``edge``: the median non-zero |difference| between two cells sharing an edge of the triangle lattice (what the
  spatial prediction has to beat, and the structure a bound of that size wipes out), with the share of zeros.
* ``tendency``: the median non-zero |change| over one time step, from each date and the date before it when they
  are one time step (the store's ``frequency``) apart; ``--no-tendency`` when the dates are not real (a sample of
  dates relabelled as consecutive). A bound of that size is as large as what a forecast model learns per step.
* ``std``: the store's ``stdev`` statistics, else the standard deviation over the dates.

``encode`` (expensive: one full encode per date and scale; run it on a machine with many cores) encodes every date
with all bounds multiplied by each scale and records per variable the bytes (the field's payload with its share of the
chunk header), the storage mode, the worst error and the RMSE relative to the scaled bound.

``draw`` writes the figure: one row per variable without a level, one band per family of levels (names
``<family>_<level>`` with at least 3 levels, ``--family-pattern``), each scale divided by the bound on a log axis
(the bound is the line at 1), and, with ``--encode``, the bits per value at the bound and at the other scales and
each row's share of the lossy bytes, under a panel that explains one row. Deterministic: the same JSON files give
the same SVG. ``--png`` also renders it with cairosvg (an optional extra, not a dependency of the package).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "figures"))
from icon_lattice_figures import (  # noqa: E402
    DECODED,
    DECODED_LIGHT,
    FAINT,
    INK,
    LIGHT,
    MUTED,
    TARGET,
    Svg,
    mix,
    text_width,
)

from anemoi_datasets_codecs.cli import lattice_config, load_config  # noqa: E402
from anemoi_datasets_codecs.icon_lattice import LatticeCodec  # noqa: E402

STD = "#3d444d"  # standard deviation marks (dark grey)
PANEL = "#f6f8fa"  # background of the "how to read one row" panel
FAMILY = r"(?P<family>.+)_(?P<level>\d+)"


# ------------------------------------------------------------------------------------------------ inputs
def parse_dates(spec: str | None, n: int, default: int) -> list[int]:
    """Date indices of ``i,j,k``, ``a:b[:step]`` or, without a spec, ``default`` evenly spaced ones."""
    if not spec:
        k = min(default, n)
        return sorted({round(i * (n - 1) / max(k - 1, 1)) for i in range(k)})
    out: list[int] = []
    for part in spec.split(","):
        if ":" in part:
            out.extend(range(*[int(p) for p in part.split(":")]))
        else:
            out.append(int(part))
    if not out or min(out) < 0 or max(out) >= n:
        raise SystemExit(f"--dates {spec}: indices must be in [0, {n})")
    return sorted(set(out))


def open_inputs(store: Path, config: Path) -> tuple[object, LatticeCodec, dict]:
    """The store's group, the codec of the configuration and its canonical configuration."""
    import zarr

    cfg = lattice_config(load_config(config))
    if cfg.get("chain"):
        raise SystemExit("a chained configuration: give the configuration of the single lattice codec")
    codec = LatticeCodec(**cfg)
    g = zarr.open_group(str(store), mode="r")
    names = list(g.attrs.get("variables", []))
    if names and names != list(codec.names):
        raise SystemExit("the variables of the configuration differ from the store's")
    if g["data"].shape[-1] != codec.n_cells:
        raise SystemExit(f"the store has {g['data'].shape[-1]} cells, the configuration's grid {codec.n_cells}")
    return g, codec, codec.get_config()


def edge_pairs(layout) -> tuple[np.ndarray, np.ndarray]:
    """Native indices of the two cells of every edge of the triangle lattice: horizontal neighbours on a row of the
    lattice image, and (r, c) with (r - 1, c) where r + c is even (a down triangle, see layout.py)."""
    h, w = layout.shape
    img = np.full(h * w, -1, np.int64)
    img[layout.pixel_index] = np.arange(layout.n_cells)
    img = img.reshape(h, w)
    r, c = np.meshgrid(np.arange(1, h), np.arange(w), indexing="ij")
    even = (r + c) % 2 == 0
    a = np.concatenate([img[:, :-1].ravel(), img[1:, :][even]])
    b = np.concatenate([img[:, 1:].ravel(), img[:-1, :][even]])
    keep = (a >= 0) & (b >= 0)
    return a[keep], b[keep]


def nonzero_median(d: np.ndarray) -> tuple[float | None, float]:
    """Median of the non-zero values of ``d`` (None when all are zero) and the share of zeros."""
    nz = d[d != 0]
    return (float(np.median(nz)) if len(nz) else None), 1.0 - len(nz) / max(len(d), 1)


def time_step(g) -> np.timedelta64 | None:
    """The store's time step: its ``frequency`` attribute (``1h``, ``6h``, ``1d``), else the median date spacing."""
    freq = str(g.attrs.get("frequency", ""))
    m = re.fullmatch(r"(\d+)\s*([hdm])", freq)
    if m:
        return np.timedelta64(int(m.group(1)), {"h": "h", "d": "D", "m": "m"}[m.group(2)]).astype("timedelta64[s]")
    dates = g["dates"][:]
    return np.median(np.diff(dates)).astype("timedelta64[s]") if len(dates) > 1 else None


def _duration(step: np.timedelta64) -> str:
    """``3600 s`` -> ``1 h``; days, hours, minutes or seconds, whichever divides."""
    sec = int(step.astype("timedelta64[s]").astype(np.int64))
    for unit, n in (("d", 86400), ("h", 3600), ("min", 60)):
        if sec % n == 0:
            return f"{sec // n} {unit}"
    return f"{sec} s"


# ------------------------------------------------------------------------------------------------ stats
def stats(a: argparse.Namespace) -> None:
    g, codec, cfg = open_inputs(a.store, a.config)
    data, dates = g["data"], g["dates"][:].astype("datetime64[s]")
    picked = parse_dates(a.dates, data.shape[0], 12)
    step = None if a.no_tendency else time_step(g)
    ea, eb = edge_pairs(codec.layout)
    lossy = [(i, v) for i, v in enumerate(codec.variables) if v.lossy]
    per: dict[str, dict[str, list]] = {v.name: {} for _, v in lossy}

    def one(i: int, x: np.ndarray, prev: np.ndarray | None) -> dict:
        f = x[i]
        u = np.unique(f)
        r = {
            "maxabs": float(np.abs(f).max()),
            "quantum": float(np.diff(u.astype(np.float64)).min()) if len(u) > 1 else None,
        }
        f64 = f.astype(np.float64)
        r["edge"], r["edge_zero"] = nonzero_median(np.abs(f64[ea] - f64[eb]))
        r["mean"], r["sq"] = float(f64.mean()), float((f64**2).mean())
        if prev is not None:
            r["tendency"], r["tendency_zero"] = nonzero_median(np.abs(f64 - prev[i]))
        return r

    t0 = time.perf_counter()
    used_pairs = 0
    with ThreadPoolExecutor(a.threads) as pool:
        for d in picked:
            x = np.asarray(data[d, :, 0, :])
            prev = None
            if step is not None and d > 0 and dates[d] - dates[d - 1] == step:
                prev = np.asarray(data[d - 1, :, 0, :])
                used_pairs += 1
            for (_, v), r in zip(lossy, pool.map(lambda iv, x=x, prev=prev: one(iv[0], x, prev), lossy), strict=True):
                for k, val in r.items():
                    per[v.name].setdefault(k, []).append(val)
            print(f"date {d} ({dates[d]}): {time.perf_counter() - t0:.0f} s", flush=True)

    stdev = np.asarray(g["stdev"][:], np.float64) if "stdev" in g.array_keys() else None

    def med(xs: list | None) -> float | None:
        xs = [x for x in (xs or []) if x is not None]
        return float(np.median(xs)) if xs else None

    rows = []
    for i, v in lossy:
        p = per[v.name]
        maxabs = max(p["maxabs"])
        if stdev is not None:
            std = float(stdev[i])
        else:
            m, sq = float(np.mean(p["mean"])), float(np.mean(p["sq"]))
            std = math.sqrt(max(sq - m * m, 0.0))
        rows.append(
            {
                "name": v.name,
                "bound": v.max_error,
                "ulp": float(np.spacing(np.float32(maxabs))),
                "quantum": med(p["quantum"]),
                "edge": med(p["edge"]),
                "edge_zero": med(p["edge_zero"]),
                "tendency": med(p.get("tendency")),
                "tendency_zero": med(p.get("tendency_zero")),
                "std": std,
                "std_source": "statistics" if stdev is not None else "dates",
            }
        )
    out = {
        "store": a.store.name,
        "dates": [str(dates[d]) for d in picked],
        "date_indices": picked,
        "time_step": None if step is None else _duration(step),
        "tendency_pairs": used_pairs,
        "n_variables": len(codec.variables),
        "lossless": [v.name for v in codec.variables if not v.lossy],
        "options": cfg["options"],
        "structure": cfg["structure"],
        "variables": rows,
    }
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(rows)} lossy variables, {len(picked)} dates, {used_pairs} tendency pairs -> {a.out}")


# ------------------------------------------------------------------------------------------------ encode
def encode(a: argparse.Namespace) -> None:
    from anemoi_datasets_codecs.icon_lattice import container

    g, codec, cfg = open_inputs(a.store, a.config)
    data, dates = g["data"], g["dates"][:].astype("datetime64[s]")
    picked = parse_dates(a.dates, data.shape[0], 2)
    scales = [float(s) for s in a.scales.split(",")]
    out = {
        "store": a.store.name,
        "n_cells": codec.n_cells,
        "scales": scales,
        "dates": [str(dates[d]) for d in picked],
        "date_indices": picked,
        "options": cfg["options"],
        "structure": cfg["structure"],
        "runs": [],
    }
    for d in picked:
        x = np.asarray(data[d])  # (variables, ensembles, cells)
        for s in scales:
            c = LatticeCodec.from_config({**cfg, "bounds": {k: b * s for k, b in cfg["bounds"].items()}})
            t = time.perf_counter()
            enc = c.encode(x, dates=[dates[d]])
            t_enc = time.perf_counter() - t
            dec = np.asarray(c.decode(enc)).reshape(x.shape)
            recs = container.inspect_chunk(enc)
            per = {}
            for i, v in enumerate(c.variables):
                if not v.lossy:
                    continue
                e = dec[i].astype(np.float64) - x[i]
                rec = recs[i * x.shape[1]]  # first member (fields are variables x ensembles in C order)
                nbytes = sum(r.nbytes for r in recs[i * x.shape[1] : (i + 1) * x.shape[1]])
                per[v.name] = {
                    "bytes": nbytes,
                    "mode": container.MODE_LABELS[rec.mode],
                    "fallback": rec.fallback,
                    "max_err_over_bound": float(np.abs(e).max()) / v.max_error,
                    "rmse_over_bound": float(np.sqrt(np.mean(e**2))) / v.max_error,
                }
            out["runs"].append(
                {"date_index": d, "scale": s, "bytes": len(enc), "encode_s": round(t_enc, 1), "variables": per}
            )
            a.out.write_text(json.dumps(out, indent=1) + "\n")  # after every encode: a killed job keeps its results
            print(f"date {d} scale {s}: {len(enc) / 2**20:.2f} MiB in {t_enc:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------------ figure
class Row:
    """One variable (a level of a family or a variable without level) with its scales relative to its bound."""

    def __init__(self, v: dict, level: int | None = None) -> None:
        self.name, self.level, self.bound = v["name"], level, v["bound"]
        b = v["bound"]
        self.floor = max(v["ulp"], v["quantum"] or 0.0) / b
        self.edge = None if v["edge"] is None else v["edge"] / b
        self.tendency = None if v.get("tendency") is None else v["tendency"] / b
        self.std = v["std"] / b if v["std"] > 0 else None
        self.tolerance: float | None = None  # a tolerance given with --tolerances, divided by the bound
        self.bits: dict[float, float] = {}  # scale -> bits per value (mean over the encoded dates)
        self.bytes = 0.0  # mean bytes per date at scale 1
        self.lossless = False

    def signal(self) -> float:
        s = [x for x in (self.edge, self.tendency) if x]
        return min(s) if s else math.inf


def tolerance_of(name: str, tolerances: dict[str, float]) -> float | None:
    """The tolerance of ``name``: an exact key first, then the keys as regular expressions (full match) in order."""
    if name in tolerances:
        return tolerances[name]
    for pat, t in tolerances.items():
        if re.fullmatch(pat, name):
            return t
    return None


def rows_of(
    st: dict, enc: dict | None, pattern: str, tolerances: dict[str, float] | None = None
) -> tuple[list[tuple[str, list[Row]]], list[Row]]:
    """Families (name, levels ascending) and the variables without level."""
    groups: dict[str, list[tuple[int, dict]]] = {}
    for v in st["variables"]:
        m = re.fullmatch(pattern, v["name"])
        if m:
            groups.setdefault(m.group("family"), []).append((int(m.group("level")), v))
    fams, singles = [], []
    in_family = set()
    for fam in sorted(groups):
        if len(groups[fam]) >= 3:
            fams.append((fam, [Row(v, lev) for lev, v in sorted(groups[fam], key=lambda t: t[0])]))
            in_family.update(v["name"] for _, v in groups[fam])
    singles = [Row(v) for v in st["variables"] if v["name"] not in in_family]
    for r in singles + [r for _, rs in fams for r in rs]:
        t = tolerance_of(r.name, tolerances or {})
        r.tolerance = None if t is None else t / r.bound
    if enc:
        n_dates: dict[float, int] = {}  # dates encoded per scale (a partial run has fewer at the last scales)
        for run in enc["runs"]:
            n_dates[run["scale"]] = n_dates.get(run["scale"], 0) + 1
        allrows = {r.name: r for r in singles + [r for _, rs in fams for r in rs]}
        for run in enc["runs"]:
            s, n = run["scale"], n_dates[run["scale"]]
            for name, rec in run["variables"].items():
                r = allrows.get(name)
                if r is None:
                    continue
                r.bits[s] = r.bits.get(s, 0.0) + 8 * rec["bytes"] / enc["n_cells"] / n
                if s == 1.0:
                    r.bytes += rec["bytes"] / n
                    r.lossless |= rec["mode"] != "lattice"
    singles.sort(key=lambda r: (r.signal(), r.name))
    return fams, singles


def fmt_bound(b: float) -> str:
    s = f"{b:.2g}"
    if "e" in s:
        m, e = s.split("e")
        s = f"{m}e{int(e)}"
    return s


def decade_label(k: int) -> str:
    """Tick label of 10**k: plain decimals (superscript glyphs are missing from some renderers' fonts)."""
    if k < 0:
        return f"{10.0**k:.{-k}f}"
    return f"{10**k:,}".replace(",", " ")


def draw(a: argparse.Namespace) -> None:
    st = json.loads(a.stats.read_text())
    enc = json.loads(a.encode.read_text()) if a.encode else None
    tol = json.loads(a.tolerances.read_text()) if a.tolerances else None
    fams, singles = rows_of(st, enc, a.family_pattern, tol)
    all_rows = singles + [r for _, rs in fams for r in rs]
    has_t = any(r.tendency for r in all_rows)
    has_tol = any(r.tolerance for r in all_rows)
    scales = sorted(enc["scales"]) if enc else []

    W, M = 900, 16
    lab_x, bound_x = M, 172  # name (left) and bound (right-aligned) columns
    lx0 = 186
    lx1 = 556 if enc else W - M - 24
    bx0, bx1 = 612, 806  # bits per value panel
    sx = W - M  # share column (right-aligned)
    lo, hi = a.decades
    span = hi - lo

    def X(v: float) -> float:
        return lx0 + (min(max(math.log10(v), lo), hi) - lo) / span * (lx1 - lx0)

    bmax = a.bits_max or 2 * math.ceil(max([b for r in all_rows for b in r.bits.values()] + [1.0]) / 2)
    total = sum(r.bytes for r in all_rows) or 1.0

    def BX(v: float) -> float:
        return bx0 + min(v, bmax) / bmax * (bx1 - bx0)

    unit = f" {a.level_unit}" if a.level_unit else ""
    step = st.get("time_step") or "time step"
    per = "per hour" if step == "1 h" else f"per {step}"

    # vertical layout
    pitch, fam_pad, row_h = a.level_pitch, 12, 22
    title_y, sub_y = 30, 52
    inset_y = sub_y + 34
    inset_h = 168 + (18 if has_tol else 0)
    head_y = inset_y + inset_h + 30
    arrows_y = head_y + 20
    axis_top = arrows_y + 26
    y = axis_top + 20
    layout = []
    if fams:
        levels = {tuple(r.level for r in rs) for _, rs in fams}
        if len(levels) == 1:
            lv = next(iter(levels))
            sec = f"With levels: one band per family, {len(lv)} levels, {lv[0]}{unit} at the top, "
            sec += f"{lv[-1]}{unit} at the bottom"
        else:
            sec = "With levels: one band per family, levels in ascending order from the top"
        layout.append(("section", y, sec))
        y += 12
        for fam, rs in fams:
            h = max(len(rs) * pitch, 36) + fam_pad
            layout.append(("family", y, (fam, rs, h)))
            y += h
        y += 22
    if singles:
        layout.append(("section", y, "Without level: sorted by the closest of edge difference and change"))
        y += 10
        for r in singles:
            layout.append(("single", y, r))
            y += row_h
    axis_bot = y + 4
    H = int(axis_bot + 70)
    svg = Svg(W, H, a.title)

    # title, subtitle
    svg.text(M, title_y, a.title, size=17, weight="bold")
    svg.text(M, sub_y, a.subtitle or f"{st['store']}: {len(st['dates'])} dates", size=13, fill=MUTED)
    svg.text(
        M,
        sub_y + 18,
        f"{len(st['variables'])} variables with an error bound; the {len(st['lossless'])} stored exactly are not "
        "shown.",
        size=13,
        fill=MUTED,
    )
    _inset(svg, M, inset_y, W - 2 * M, inset_h, per, enc is not None, has_t, has_tol, scales)

    # column headers, the reading direction, the axes
    svg.text(lx0, head_y, "Each scale divided by the variable's bound", size=13, weight="bold")
    xb = X(1.0)
    svg.text(xb - 6, arrows_y, "← smaller than the bound: lost", size=12, anchor="end", fill=MUTED)
    svg.text(xb + 6, arrows_y, "larger: kept →", size=12, fill=MUTED)
    for k in range(lo, hi + 1):
        xx = X(10.0**k)
        svg.line(xx, axis_top, xx, axis_bot, stroke=FAINT if k else INK, sw=1 if k else 1.5)
        for ty in (axis_top - 6, axis_bot + 16):
            svg.text(
                xx,
                ty,
                decade_label(k),
                size=12,
                anchor="middle",
                fill=INK if k == 0 else MUTED,
                weight="bold" if k == 0 else "normal",
            )
    if enc:
        svg.text(bx0, head_y, "Bits per value", size=13, weight="bold")
        svg.text(sx, head_y, "Share", size=13, weight="bold", anchor="end")
        stride = 2 if bmax <= 16 else 4
        for k in range(0, bmax + 1, stride):
            xx = BX(k)
            svg.line(xx, axis_top, xx, axis_bot, stroke=FAINT, sw=1)
            for ty in (axis_top - 6, axis_bot + 16):
                svg.text(xx, ty, str(k), size=12, anchor="middle", fill=MUTED)
    for i, line in enumerate(_footnote(st, enc)):
        svg.text(M, axis_bot + 40 + 17 * i, line, size=12, fill=MUTED)

    # rows
    for kind, yy, item in layout:
        if kind == "section":  # on an opaque strip: the grid lines stop at the text
            svg.rect(lab_x - 2, yy - 13, text_width(item, 12) + 8, 18, fill="#ffffff")
            svg.text(lab_x, yy, item, size=12, fill=MUTED, italic=True)
        elif kind == "single":
            _single(svg, item, yy + row_h / 2, X, BX, bmax, enc, lab_x, bound_x, sx, total, row_h, scales)
        else:
            fam, rs, h = item
            _family(svg, fam, rs, yy, h, pitch, X, BX, bmax, enc, lab_x, sx, total, scales)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    svg.save(a.out)
    print(f"wrote {a.out}")
    if a.png:
        write_png(a.out, a.png, a.png_width)
        print(f"wrote {a.png}")
    if a.table:
        write_table(a.table, [(fam, r) for fam, rs in fams for r in rs] + [("", r) for r in singles], scales, total)
        print(f"wrote {a.table}")


def write_png(svg_path: Path, png_path: Path, width: int) -> None:
    """Render the SVG with cairosvg (optional: the module when importable, else its command line)."""
    try:
        import cairosvg
    except ImportError:
        exe = shutil.which("cairosvg")
        if exe is None:
            raise SystemExit(
                "--png needs cairosvg (not a dependency of the package): pip install cairosvg, or render with "
                f"uvx --from cairosvg cairosvg {svg_path} -o {png_path} -W {width}"
            ) from None
        subprocess.run([exe, str(svg_path), "-o", str(png_path), "-W", str(width)], check=True)
    else:
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=width)


def write_table(path: Path, rows: list[tuple[str, Row]], scales: list[float], total: float) -> None:
    """The figure's numbers as CSV (every scale divided by the bound; bits per value per scale)."""
    head = ["family", "name", "bound", "resolution", "edge", "tendency", "std", "tolerance"]
    head += [f"bits_x{s:g}" for s in scales] + ["share"]
    lines = [",".join(head)]
    for fam, r in rows:
        vals = [r.bound, r.floor, r.edge, r.tendency, r.std, r.tolerance] + [r.bits.get(s) for s in scales]
        vals.append(r.bytes / total if scales else None)
        lines.append(",".join([fam, r.name] + ["" if v is None else f"{v:.4g}" for v in vals]))
    path.write_text("\n".join(lines) + "\n")


def _inset(
    svg: Svg,
    x0: float,
    y0: float,
    w: float,
    h: float,
    per: str,
    enc: bool,
    has_t: bool,
    has_tol: bool,
    scales: list[float],
) -> None:
    """How to read one row: a schematic row with a short label on every mark, and the two verdict rules."""
    svg.rect(x0, y0, w, h, fill=PANEL, rx=6)
    svg.text(x0 + 14, y0 + 22, "How to read one row", size=13, weight="bold")
    y = y0 + 70  # the schematic row
    above, below = y - 16, y + 25
    gx0, gx1, xb = x0 + 24, x0 + 104, x0 + 160
    svg.rect(gx0, y - 6, gx1 - gx0, 12, fill=LIGHT)
    svg.text(gx0, below, "data cannot resolve finer", size=12)
    svg.line(xb, y - 13, xb, y + 13, stroke=INK, sw=1.5)
    svg.text(xb, above, "the bound", size=12, anchor="middle", weight="bold")
    xr, xe, xt = x0 + 520, x0 + 300, x0 + 420
    _dot(svg, xr, y, "std")
    svg.text(xr, below, "spread of the field", size=12, anchor="middle")
    _dot(svg, xe, y, "edge")
    svg.text(xe, below, "neighbouring cells differ by", size=12, anchor="middle")
    if has_t:
        _dot(svg, xt, y, "tendency")
        svg.text(xt, above, f"changes {per} by", size=12, anchor="middle")
    if enc:
        bb0, bd, bb1 = x0 + 640, x0 + 700, x0 + 780
        svg.line(bb0, y, bb1, y, stroke=DECODED_LIGHT, sw=7)
        svg.circle(bd, y, 4.5, fill=DECODED, stroke="#ffffff", sw=1.5)
        svg.path(f"M{bd + 6:.1f},{y - 9:.1f} V{y - 12:.1f} H{bb1:.1f} V{y - 9:.1f}", stroke=INK, sw=1)
        svg.text((bd + bb1) / 2, above - 4, "what halving the bound costs", size=12, anchor="middle")
        svg.text(bb0, below, "bits per value at the bound;", size=12)
        svg.text(bb0, below + 16, f"bar: bound {_times(scales[-1])} to {_times(scales[0])}", size=12, fill=MUTED)
    ry = y0 + h - (54 if has_tol else 36)
    svg.text(x0 + 14, ry, "Too tight:", size=12, weight="bold")
    svg.text(x0 + 90, ry, "the grey bar touches the line; the bound buys bytes and no information.", size=12)
    svg.text(x0 + 14, ry + 18, "Too loose:", size=12, weight="bold")
    loose = (
        f"the diamond within a few bounds; the signal {per} drowns in the error."
        if has_t
        else ("the disc left of the line where the grid-scale detail matters.")
    )
    svg.text(x0 + 90, ry + 18, loose, size=12)
    if has_tol:
        _dot(svg, x0 + 20, ry + 32, "tolerance")
        svg.text(x0 + 32, ry + 36, "the tolerance given for the variable", size=12)


def _legend_text(svg: Svg, x: float, y: float, s: str) -> float:
    svg.text(x, y, s, size=12)
    return x + text_width(s, 12) + 18


def _dot(svg: Svg, x: float, y: float, kind: str, r: float = 4.5) -> None:
    """The mark of a scale: edge difference a blue disc, change an orange diamond, std a larger dark ring (drawn
    first, so that a disc or diamond at the same place shows inside it)."""
    if kind == "edge":
        svg.circle(x, y, r, fill=DECODED, stroke="#ffffff", sw=1.2)
    elif kind == "tendency":
        d = r * 1.25
        svg.poly([(x, y - d), (x + d, y), (x, y + d), (x - d, y)], fill=TARGET, stroke="#ffffff", sw=1.2)
    elif kind == "tolerance":
        svg.rect(x - 1.5, y - r * 1.6, 3, r * 3.2, fill=INK)
    else:
        svg.circle(x, y, r * 1.55, fill="none", stroke=STD, sw=1.3)


def _times(scale: float) -> str:
    """``4`` -> ``× 4``, ``0.5`` -> ``÷ 2``."""
    return f"× {scale:g}" if scale >= 1 else f"÷ {1 / scale:g}"


def _footnote(st: dict, enc: dict | None) -> list[str]:
    s = f"Statistics: {len(st['dates'])} dates"
    if st.get("tendency_pairs"):
        s += f", the change from {st['tendency_pairs']} pairs with the date before"
    s += "; medians of the non-zero differences."
    out = [s]
    if enc:
        out.append(
            f"Encoded: {len(enc['date_indices'])} dates, structure {enc['structure']}, k = {enc['options']['k']:g}. "
            "Share: the row's part of the bytes of all bounded variables."
        )
    return out


MARKS = ("tolerance", "std", "edge", "tendency")  # drawing order: the ring under the disc and the diamond


def _bits(svg: Svg, r: Row, y: float, BX, bmax: int, scales: list[float], thick: float = 7) -> None:
    if not r.bits:
        return
    xs = [BX(r.bits[s]) for s in scales if s in r.bits]
    svg.line(min(xs), y, max(xs), y, stroke=DECODED_LIGHT, sw=thick)
    if max(r.bits.values()) > bmax:
        _clipped(svg, max(xs), y)
    if 1.0 in r.bits:
        svg.circle(BX(r.bits[1.0]), y, 4.5, fill=DECODED, stroke="#ffffff", sw=1.5)


def _clipped(svg: Svg, x: float, y: float) -> None:
    """A small arrow head at the end of the bits axis: the value is beyond it."""
    svg.poly([(x + 1, y - 5), (x + 7, y), (x + 1, y + 5)], fill=DECODED)


def _single(svg, r: Row, y, X, BX, bmax, enc, lab_x, bound_x, sx, total, row_h, scales) -> None:
    svg.text(lab_x, y + 4.5, r.name, size=13)
    svg.text(bound_x, y + 4.5, fmt_bound(r.bound), size=12, anchor="end", fill=MUTED)
    lx0 = X(1e-300)
    if r.floor > 0 and X(r.floor) > lx0:
        svg.rect(lx0, y - 6, X(r.floor) - lx0, 12, fill=LIGHT)
    svg.line(X(1.0), y - row_h / 2, X(1.0), y + row_h / 2, stroke=INK, sw=1.5)
    for kind in MARKS:
        v = getattr(r, kind)
        if v:
            _dot(svg, X(v), y, kind)
    if enc:
        _bits(svg, r, y, BX, bmax, scales)
        if r.bytes:
            svg.text(sx, y + 4.5, _pct(r.bytes / total), size=12, anchor="end", fill=MUTED)
        if r.lossless:
            svg.text(BX(0) - 6, y + 4.5, "exact", size=12, anchor="end", fill=MUTED)


def _family(svg, fam, rs, y0, h, pitch, X, BX, bmax, enc, lab_x, sx, total, scales) -> None:
    n = len(rs)
    top = y0 + (h - n * pitch) / 2

    def Y(i: int) -> float:
        return top + (i + 0.5) * pitch

    svg.line(lab_x, y0, sx, y0, stroke=LIGHT, sw=1)
    bs = [r.bound for r in rs]
    lo, hi = min(bs), max(bs)
    svg.text(lab_x, y0 + 20, fam, size=13, weight="bold")
    svg.text(
        lab_x,
        y0 + 37,
        "bound " + (fmt_bound(lo) if lo == hi else f"{fmt_bound(lo)} to {fmt_bound(hi)}"),
        size=12,
        fill=MUTED,
    )
    lx0 = X(1e-300)
    for i, r in enumerate(rs):
        if r.floor > 0 and X(r.floor) > lx0:
            svg.rect(lx0, Y(i) - pitch / 2, X(r.floor) - lx0, pitch, fill=LIGHT)
    svg.line(X(1.0), y0 + 2, X(1.0), y0 + h - 2, stroke=INK, sw=1.5)
    colours = {"tolerance": INK, "std": STD, "edge": DECODED, "tendency": TARGET}
    pts = {k: [(X(getattr(r, k)), Y(i)) for i, r in enumerate(rs) if getattr(r, k)] for k in MARKS}
    for k in MARKS:  # the profiles first, thin and light, then every mark on top
        if len(pts[k]) > 1:
            svg.path(_polyline(pts[k]), stroke=mix(colours[k], "#ffffff", 0.45), sw=1.2)
    for k in MARKS:
        for px, py in pts[k]:
            _dot(svg, px, py, k, r=3)
    if enc and any(r.bits for r in rs):
        lo_s, hi_s = min(scales), max(scales)
        left = [(BX(r.bits[hi_s]), Y(i)) for i, r in enumerate(rs) if hi_s in r.bits]
        right = [(BX(r.bits[lo_s]), Y(i)) for i, r in enumerate(rs) if lo_s in r.bits]
        if left and right:
            svg.poly(left + right[::-1], fill=DECODED_LIGHT)
        for i, r in enumerate(rs):
            if r.bits and max(r.bits.values()) > bmax:
                _clipped(svg, BX(bmax), Y(i))
        mid = [(BX(r.bits[1.0]), Y(i)) for i, r in enumerate(rs) if 1.0 in r.bits]
        if len(mid) > 1:
            svg.path(_polyline(mid), stroke=DECODED, sw=1.2)
        for px, py in mid:
            svg.circle(px, py, 2.5, fill=DECODED)
        b = sum(r.bytes for r in rs)
        if b:
            svg.text(sx, y0 + 20, _pct(b / total), size=12, anchor="end", fill=MUTED)
        for i, r in enumerate(rs):
            if r.lossless:
                svg.text(BX(0) - 6, Y(i) + 4, "exact", size=12, anchor="end", fill=MUTED)


def _polyline(pts: list[tuple[float, float]]) -> str:
    return "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in pts)


def _pct(f: float) -> str:
    return f"{100 * f:.1f} %" if f >= 0.001 else "< 0.1 %"


# ------------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("stats", "encode"):
        q = sub.add_parser(name)
        q.add_argument("store", type=Path)
        q.add_argument("config", type=Path)
        q.add_argument("out", type=Path)
        q.add_argument("--dates", help="date indices: i,j,k or a:b[:step]")
        if name == "stats":
            q.add_argument("--no-tendency", action="store_true", help="the store's dates are not real time steps")
            q.add_argument("--threads", type=int, default=16)
        else:
            q.add_argument("--scales", default="0.5,1,2,4", help="bound multipliers (must include 1)")
        q.set_defaults(fn=stats if name == "stats" else encode)
    q = sub.add_parser("draw")
    q.add_argument("stats", type=Path)
    q.add_argument("out", type=Path)
    q.add_argument("--encode", type=Path, help="the JSON of the encode step (adds bits per value and shares)")
    q.add_argument("--table", type=Path, help="also write the figure's numbers as CSV")
    q.add_argument("--tolerances", type=Path, help="JSON {name or regex: tolerance}: drawn as a mark when given")
    q.add_argument("--title", default="Error bounds against the scales of the data")
    q.add_argument("--subtitle", help="the dataset (default: the store's name and the number of dates)")
    q.add_argument("--family-pattern", default=FAMILY, help="regex with groups family and level")
    q.add_argument("--decades", type=int, nargs=2, default=(-3, 4), metavar=("LO", "HI"), help="ladder axis")
    q.add_argument("--level-pitch", type=float, default=7.0, help="px per level in a family band")
    q.add_argument("--level-unit", default="", help="unit of the levels in the section header (e.g. hPa)")
    q.add_argument("--bits-max", type=int, help="end of the bits axis (default: the largest value, rounded up)")
    q.add_argument("--png", type=Path, help="also render a PNG (needs cairosvg, an optional extra, not a dependency)")
    q.add_argument("--png-width", type=int, default=2700, help="PNG width in px (default 2700, 3x the SVG)")
    q.set_defaults(fn=draw)
    a = p.parse_args(argv)
    if a.cmd == "encode" and 1.0 not in [float(s) for s in a.scales.split(",")]:
        p.error("--scales must include 1")
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
