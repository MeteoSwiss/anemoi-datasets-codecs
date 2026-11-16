"""lab ledger: an append-only experiment ledger, so that every experiment is comparable with every other one.

``run`` encodes dates of a store with the codec of a configuration plus overrides, decodes every chunk back and
appends one JSON line per (run, date, field) to a ``.jsonl`` ledger. ``sweep`` expands a specification into runs and
skips settings the ledger already holds. ``compare`` sets two tags side by side, ``summary`` lists the runs, ``pareto``
plots them (plain SVG).

Row schema (every row is self-describing; there is no run table):

* run: ``run`` (id, time-based), ``tag``, ``time`` (UTC ISO), ``env`` (``lab.data.environment()``), ``config_sha``
  (sha256 of the resolved ``codec.get_config()``), ``overrides`` (the ``--set`` settings as a dict of dotted key to
  value), ``store``, ``dated`` (whether the valid date was passed to the encoder)
* date: ``date_index``, ``encode_s``, ``decode_s``, ``total_bytes`` (chunk size), ``chunk_sha256``
* field: ``field`` (label), ``mode`` (lattice, lossless, constant, static_ref, recipe), ``fallback`` (code of
  ``container.FALLBACK_*``), ``bytes`` (``inspect_chunk``, header share included, sums to ``total_bytes``), ``lossy``
  (the variable has a bound), ``bound`` (null for exact variables), ``max_err`` and ``rmse`` (absolute, against the
  input), ``max_err_over_bound`` and ``rmse_over_bound`` (null without a bound), ``exact`` (decoded bit-identical)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import re
import sys
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..icon_lattice import container
from ..icon_lattice.codec import LatticeCodec
from . import data as labdata

# ------------------------------------------------------------------ settings


def parse_value(text: str) -> Any:
    """A ``--set`` value: JSON when it parses, else the string (``tree``, ``true``, ``3``, ``[0, null]``)."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def parse_settings(items: list[str] | None) -> dict[str, Any]:
    """``KEY=VALUE`` strings to a dict of dotted key to value."""
    out: dict[str, Any] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set expects KEY=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = parse_value(v)
    return out


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``config`` with dotted-key overrides set (``options.k`` sets ``config['options']['k']``); the
    dictionaries along a path are copied, other values (a grid layout) are shared."""
    cfg = dict(config)
    for key, value in overrides.items():
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            child = node.get(p)
            node[p] = dict(child) if isinstance(child, dict) else {}
            node = node[p]
        node[parts[-1]] = copy.deepcopy(value)
    return cfg


def build_codec(config: str | os.PathLike[str] | dict[str, Any], overrides: dict[str, Any]) -> LatticeCodec:
    """The codec of a configuration (dict or path) with dotted-key overrides."""
    cfg = dict(config) if isinstance(config, dict) else labdata.load_lattice_config(config)
    return labdata.codec_for(apply_overrides(cfg, overrides))


def parse_dates(text: str | None, n_dates: int) -> list[int]:
    """Date indices from ``6``, ``0,6,13``, ``2:5`` (range) or ``all``; default the first date."""
    if text is None:
        return [0]
    if text == "all":
        return list(range(n_dates))
    out: list[int] = []
    for part in text.split(","):
        if ":" in part:
            a, b = part.split(":")
            out.extend(range(int(a), int(b)))
        else:
            out.append(int(part))
    bad = [i for i in out if not 0 <= i < n_dates]
    if bad:
        raise SystemExit(f"date indices {bad} outside the store's {n_dates} dates")
    return out


def default_tag(overrides: dict[str, Any]) -> str:
    return ",".join(f"{k}={json.dumps(v) if not isinstance(v, str) else v}" for k, v in sorted(overrides.items())) or (
        "default"
    )


# ------------------------------------------------------------------ ledger io


def read_ledger(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """All rows of a ledger (empty when the file is missing)."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_rows(path: str | os.PathLike[str], rows: list[dict[str, Any]]) -> None:
    """Append rows to a ledger (one write, so a crash leaves whole lines)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    blob = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    with open(path, "a") as fh:
        fh.write(blob)


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


# ------------------------------------------------------------------ run


def field_rows(codec: LatticeCodec, x: np.ndarray, dec: np.ndarray, buf: bytes) -> list[dict[str, Any]]:
    """Per-field records of one chunk: the container's mode / fallback / bytes and the error against the input."""
    recs = container.inspect_chunk(buf)
    labels = labdata.field_labels(codec, 1)
    idx = labdata.field_index(codec)
    rows = []
    for i, rec in enumerate(recs):
        var = codec.variables[idx.variable(i)]
        a, b = x[i].astype(np.float64), dec[i].astype(np.float64)
        d = np.abs(a - b)
        exact = bool(np.array_equal(x[i].view(np.uint32), dec[i].view(np.uint32)))
        bound = float(var.max_error) if var.lossy else None
        mx, rmse = float(d.max()), float(np.sqrt(np.mean(d * d)))
        rows.append(
            {
                "field": labels[i],
                "mode": container.MODE_LABELS[rec.mode],
                "fallback": int(rec.fallback),
                "bytes": int(rec.nbytes),
                "lossy": bound is not None,
                "bound": bound,
                "max_err": mx,
                "rmse": rmse,
                "max_err_over_bound": mx / bound if bound else None,
                "rmse_over_bound": rmse / bound if bound else None,
                "exact": exact,
            }
        )
    return rows


def run_codec(
    codec: LatticeCodec,
    store: str | os.PathLike[str],
    dates: list[int] | None = None,
    *,
    tag: str = "",
    overrides: dict[str, Any] | None = None,
    dated: bool = False,
    keep_chunks: str | os.PathLike[str] | None = None,
    run_id: str | None = None,
    log=None,
) -> list[dict[str, Any]]:
    """Encode and decode dates of ``store`` with ``codec``; the rows (not yet written) of the run."""
    data, valid = labdata.open_data(store)
    dates = [0] if dates is None else dates
    run_id = run_id or new_run_id()
    cfg_sha = labdata.config_sha256(codec.get_config())
    env = labdata.environment()
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []
    if keep_chunks:
        Path(keep_chunks).mkdir(parents=True, exist_ok=True)
    for di in dates:
        x = labdata.read_date(data, di)
        valid_date = None
        if dated:
            if valid is None:
                raise SystemExit("--dated needs a `dates` array in the store")
            valid_date = valid[di]
        t0 = time.perf_counter()
        buf = bytes(codec.encode(x, valid_date))
        t1 = time.perf_counter()
        dec = np.asarray(codec.decode(buf)).reshape(x.shape)
        t2 = time.perf_counter()
        if keep_chunks:
            safe = re.sub(r"[^A-Za-z0-9._=,-]+", "_", tag or "run")
            (Path(keep_chunks) / f"{safe}.{run_id}.d{di}.bin").write_bytes(buf)
        base = {
            "run": run_id,
            "tag": tag,
            "time": stamp,
            "env": env,
            "config_sha": cfg_sha,
            "overrides": overrides or {},
            "store": str(store),
            "dated": dated,
            "date_index": int(di),
            "encode_s": t1 - t0,
            "decode_s": t2 - t1,
            "total_bytes": len(buf),
            "chunk_sha256": hashlib.sha256(buf).hexdigest(),
        }
        rows.extend({**base, **f} for f in field_rows(codec, x, dec, buf))
        if log:
            log(f"{tag or run_id} date {di}: {len(buf)} bytes, encode {t1 - t0:.1f} s, decode {t2 - t1:.2f} s")
    return rows


def run(
    config: str | os.PathLike[str] | dict[str, Any],
    store: str | os.PathLike[str],
    ledger: str | os.PathLike[str],
    dates: str | None = None,
    *,
    tag: str | None = None,
    overrides: dict[str, Any] | None = None,
    dated: bool = False,
    keep_chunks: str | os.PathLike[str] | None = None,
    codec: LatticeCodec | None = None,
    log=None,
) -> list[dict[str, Any]]:
    """Build the codec, run it on ``dates`` of ``store`` and append the rows to ``ledger``."""
    overrides = overrides or {}
    codec = codec or build_codec(config, overrides)
    data, _ = labdata.open_data(store)
    idx = parse_dates(dates, data.shape[0])
    rows = run_codec(
        codec,
        store,
        idx,
        tag=tag if tag is not None else default_tag(overrides),
        overrides=overrides,
        dated=dated,
        keep_chunks=keep_chunks,
        log=log,
    )
    append_rows(ledger, rows)
    return rows


# ------------------------------------------------------------------ sweep


def expand_spec(spec: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """``(tag, overrides)`` of every setting of a sweep specification: ``settings`` (a list of override dicts) and /
    or ``grid`` (a dict of dotted key to list of values, cartesian product); ``base`` overrides apply to all; ``tag``
    is the prefix of the derived tags (``prefix/key=value,...``)."""
    base = dict(spec.get("base", {}))
    settings = [dict(s) for s in spec.get("settings", [])]
    grid = spec.get("grid")
    if grid:
        keys = list(grid)
        settings += [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(grid[k] for k in keys))]
    if not settings:
        raise SystemExit("the sweep specification has neither `settings` nor `grid`")
    prefix = spec.get("tag")
    out = []
    for s in settings:
        ov = {**base, **s}
        t = default_tag(s)
        out.append((f"{prefix}/{t}" if prefix else t, ov))
    return out


def already_done(rows: list[dict[str, Any]], config_sha: str, store: str, dates: list[int], dated: bool) -> bool:
    """Whether the ledger holds every date of a setting with the same configuration, store and package version."""
    from .. import __version__

    have = {
        r["date_index"]
        for r in rows
        if r["config_sha"] == config_sha
        and r["store"] == store
        and r["env"].get("package") == __version__
        and r.get("dated", False) == dated
    }
    return set(dates) <= have


def sweep(
    config: str | os.PathLike[str] | dict[str, Any],
    store: str | os.PathLike[str],
    spec: dict[str, Any],
    ledger: str | os.PathLike[str],
    *,
    force: bool = False,
    dated: bool | None = None,
    keep_chunks: str | os.PathLike[str] | None = None,
    log=print,
) -> dict[str, list[str]]:
    """Run every setting of ``spec`` not already in the ledger; the tags that ran and that were skipped."""
    base_cfg = dict(config) if isinstance(config, dict) else labdata.load_lattice_config(config)
    data, _ = labdata.open_data(store)
    dates_text = spec.get("dates")
    if isinstance(dates_text, list):
        dates_text = ",".join(str(d) for d in dates_text)
    elif dates_text is not None:
        dates_text = str(dates_text)
    idx = parse_dates(dates_text, data.shape[0])
    dated = bool(spec.get("dated", False)) if dated is None else dated
    ran, skipped = [], []
    for tag, ov in expand_spec(spec):
        codec = build_codec(base_cfg, ov)
        sha = labdata.config_sha256(codec.get_config())
        if not force and already_done(read_ledger(ledger), sha, str(store), idx, dated):
            log(f"skip {tag} (in the ledger)")
            skipped.append(tag)
            continue
        run(
            base_cfg,
            store,
            ledger,
            ",".join(map(str, idx)),
            tag=tag,
            overrides=ov,
            dated=dated,
            keep_chunks=keep_chunks,
            codec=codec,
            log=log,
        )
        ran.append(tag)
    return {"ran": ran, "skipped": skipped}


# ------------------------------------------------------------------ select, summary, compare


def runs_of(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Rows grouped by run id, in ledger order."""
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[r["run"]].append(r)
    return dict(out)


def select_run(rows: list[dict[str, Any]], spec: str) -> list[dict[str, Any]]:
    """The rows of ``tag`` (its latest run) or ``tag@runid`` (run ids may be abbreviated to a prefix)."""
    tag, _, rid = spec.partition("@")
    runs = [(rs[0]["time"], run_id, rs) for run_id, rs in runs_of(rows).items() if rs[0]["tag"] == tag]
    if rid:
        runs = [r for r in runs if r[1].startswith(rid)]
    if not runs:
        raise SystemExit(f"no run for {spec!r} in the ledger")
    if rid and len(runs) > 1:
        raise SystemExit(f"{spec!r} matches several runs: {[r[1] for r in runs]}")
    return runs[-1][2]  # the latest appended


def per_date(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """One record per date of a run (the date-level values repeat on every row)."""
    out: dict[int, dict[str, Any]] = {}
    for r in rows:
        out.setdefault(
            r["date_index"],
            {k: r[k] for k in ("encode_s", "decode_s", "total_bytes", "chunk_sha256")},
        )
    return out


def summary_lines(rows: list[dict[str, Any]]) -> list[str]:
    """One line per run: tag, run id, time, store, dates, settings, bytes per date, worst error / bound, seconds."""
    lines = []
    for rid, rs in runs_of(rows).items():
        pd = per_date(rs)
        r0 = rs[0]
        nd = len(pd)
        total = sum(d["total_bytes"] for d in pd.values())
        worst = max((r["max_err_over_bound"] for r in rs if r["max_err_over_bound"] is not None), default=0.0)
        enc = sum(d["encode_s"] for d in pd.values())
        dec = sum(d["decode_s"] for d in pd.values())
        ds = ",".join(map(str, sorted(pd)))
        ov = json.dumps(r0["overrides"], sort_keys=True)
        lines.append(
            f"{r0['tag']}  {rid}  {r0['time']}  {Path(r0['store']).name}  dates [{ds}]  {ov}  "
            f"bytes {total} ({total // nd}/date)  worst err/bound {worst:.4f}  encode {enc:.1f} s  decode {dec:.2f} s"
        )
    return lines


def _fmt_pct(a: float, b: float) -> str:
    return f"{100 * (b - a) / a:+.2f}%" if a else "n/a"


def compare(
    rows: list[dict[str, Any]], spec_a: str, spec_b: str, dates: list[int] | None = None, top: int = 15
) -> list[str]:
    """The comparison of two runs as text lines (B - A, summed over the common dates)."""
    ra, rb = select_run(rows, spec_a), select_run(rows, spec_b)
    if ra[0]["store"] != rb[0]["store"]:
        raise SystemExit(f"different stores: {ra[0]['store']} vs {rb[0]['store']}")
    if ra[0].get("dated", False) != rb[0].get("dated", False):
        raise SystemExit("one run passed the valid dates to the encoder and the other did not")
    common = sorted(set(per_date(ra)) & set(per_date(rb)))
    if dates is not None:
        common = [d for d in common if d in dates]
    if not common:
        raise SystemExit(
            f"{spec_a!r} (dates {sorted(per_date(ra))}) and {spec_b!r} (dates {sorted(per_date(rb))}) "
            "have no common date"
        )
    cs = set(common)

    def by_field(rs):
        out: dict[str, dict[str, Any]] = {}
        for r in rs:
            if r["date_index"] in cs:
                f = out.setdefault(r["field"], {"bytes": 0, "modes": set(), "fallbacks": set(), "rows": []})
                f["bytes"] += r["bytes"]
                f["modes"].add(r["mode"])
                f["fallbacks"].add(r["fallback"])
                f["rows"].append(r)
        return out

    fa, fb = by_field(ra), by_field(rb)
    if set(fa) != set(fb):
        raise SystemExit("the runs have different fields")
    lines = [
        f"A {spec_a}  run {ra[0]['run']}  {json.dumps(ra[0]['overrides'], sort_keys=True)}",
        f"B {spec_b}  run {rb[0]['run']}  {json.dumps(rb[0]['overrides'], sort_keys=True)}",
        f"store {ra[0]['store']}  common dates {common}",
    ]
    if ra[0]["config_sha"] == rb[0]["config_sha"]:
        lines.append("note: same configuration sha")
    ta = sum(f["bytes"] for f in fa.values())
    tb = sum(f["bytes"] for f in fb.values())
    lines.append(f"total bytes  A {ta}  B {tb}  B-A {tb - ta:+d} ({_fmt_pct(ta, tb)})")
    deltas = sorted(((fb[k]["bytes"] - fa[k]["bytes"], k) for k in fa), key=lambda t: t[0])

    def line(d, k):
        ba, bb = fa[k]["bytes"], fb[k]["bytes"]
        return f"  {k:<28} {ba:>12} -> {bb:>12}  {d:+d} ({_fmt_pct(ba, bb)})"

    lines.append(f"top {top} reductions (B smaller):")
    lines += [line(d, k) for d, k in deltas[:top] if d < 0] or ["  none"]
    lines.append(f"top {top} increases (B larger):")
    lines += [line(d, k) for d, k in reversed(deltas[-top:]) if d > 0] or ["  none"]
    changes = [
        f"  {k}: mode {sorted(fa[k]['modes'])} -> {sorted(fb[k]['modes'])}, "
        f"fallback {sorted(fa[k]['fallbacks'])} -> {sorted(fb[k]['fallbacks'])}"
        for k in fa
        if fa[k]["modes"] != fb[k]["modes"] or fa[k]["fallbacks"] != fb[k]["fallbacks"]
    ]
    lines.append("mode / fallback changes:" + ("" if changes else " none"))
    lines += changes

    def errstats(f):
        rs = [r for v in f.values() for r in v["rows"] if r["max_err_over_bound"] is not None]
        if not rs:
            return "no lossy fields"
        w = max(rs, key=lambda r: r["max_err_over_bound"])
        mean_rmse = float(np.mean([r["rmse_over_bound"] for r in rs]))
        return f"worst max/bound {w['max_err_over_bound']:.4f} ({w['field']}), mean rmse/bound {mean_rmse:.4f}"

    lines.append(f"error A: {errstats(fa)}")
    lines.append(f"error B: {errstats(fb)}")
    pa, pb = per_date(ra), per_date(rb)
    for key in ("encode_s", "decode_s"):
        sa, sb = sum(pa[d][key] for d in common), sum(pb[d][key] for d in common)
        lines.append(f"{key}  A {sa:.2f}  B {sb:.2f}  B/A {sb / sa:.3f}" if sa else f"{key}  A 0  B {sb:.2f}")
    return lines


# ------------------------------------------------------------------ pareto (plain SVG)

INK, MUTED, LIGHT, BG, POINT = "#1f2328", "#57606a", "#d0d7de", "#ffffff", "#2f6db5"
YKEYS = ("rmse", "maxerr", "encode_s", "decode_s")


def run_point(rs: list[dict[str, Any]], x: str, y: str) -> tuple[float, float]:
    """(x, y) of a run: bytes per date, and the mean over its lossy fields (rmse / maxerr, as a fraction of the bound)
    or the seconds per date."""
    pd = per_date(rs)
    n = len(pd)
    xv = sum(d["total_bytes"] for d in pd.values()) / n
    if y == "encode_s":
        yv = sum(d["encode_s"] for d in pd.values()) / n
    elif y == "decode_s":
        yv = sum(d["decode_s"] for d in pd.values()) / n
    else:
        key = "rmse_over_bound" if y == "rmse" else "max_err_over_bound"
        vals = [r[key] for r in rs if r[key] is not None]
        yv = float(np.mean(vals)) if y == "rmse" else float(np.max(vals))
    return xv, yv


def pareto_svg(rows: list[dict[str, Any]], y: str = "rmse") -> str:
    """A scatter of runs, one point per run (bytes per date against ``y``), labelled by tag."""
    pts = [(r[0]["tag"], *run_point(r, "bytes", y)) for r in runs_of(rows).values()]
    if not pts:
        raise SystemExit("the ledger is empty")
    W, H, L, R, T, B = 900, 560, 80, 30, 30, 60
    xs, ys = [p[1] for p in pts], [p[2] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    px = (x1 - x0) or max(x1, 1) * 0.1
    py = (y1 - y0) or max(y1, 1e-9) * 0.1
    x0, x1, y0, y1 = x0 - 0.08 * px, x1 + 0.3 * px, y0 - 0.08 * py, y1 + 0.08 * py

    def sx(v):
        return L + (v - x0) / (x1 - x0) * (W - L - R)

    def sy(v):
        return H - B - (v - y0) / (y1 - y0) * (H - B - T)

    ylabel = {"rmse": "mean rmse / bound", "maxerr": "worst max error / bound"}.get(
        y, y.replace("_s", " seconds per date")
    )
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="sans-serif" font-size="12">',
        f'<rect width="{W}" height="{H}" fill="{BG}"/>',
    ]
    for i in range(6):
        gx, gy = x0 + (x1 - x0) * i / 5, y0 + (y1 - y0) * i / 5
        o.append(f'<line x1="{sx(gx):.1f}" y1="{T}" x2="{sx(gx):.1f}" y2="{H - B}" stroke="{LIGHT}"/>')
        o.append(f'<line x1="{L}" y1="{sy(gy):.1f}" x2="{W - R}" y2="{sy(gy):.1f}" stroke="{LIGHT}"/>')
        o.append(f'<text x="{sx(gx):.1f}" y="{H - B + 18}" text-anchor="middle" fill="{MUTED}">{gx / 1e6:.3f}</text>')
        o.append(f'<text x="{L - 8}" y="{sy(gy) + 4:.1f}" text-anchor="end" fill="{MUTED}">{gy:.4g}</text>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{H - 14}" text-anchor="middle" fill="{INK}">MB per date</text>')
    o.append(
        f'<text transform="translate(18 {(T + H - B) / 2}) rotate(-90)" text-anchor="middle" fill="{INK}">'
        f"{ylabel}</text>"
    )
    for tag, xv, yv in pts:
        o.append(f'<circle cx="{sx(xv):.1f}" cy="{sy(yv):.1f}" r="4.5" fill="{POINT}"/>')
        label = tag.replace("&", "&amp;").replace("<", "&lt;")
        o.append(f'<text x="{sx(xv) + 7:.1f}" y="{sy(yv) - 6:.1f}" fill="{INK}">{label}</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


# ------------------------------------------------------------------ command line


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _cmd_run(a: argparse.Namespace) -> int:
    rows = run(
        a.config,
        a.store,
        a.out,
        a.dates,
        tag=a.tag,
        overrides=parse_settings(a.set),
        dated=a.dated,
        keep_chunks=a.keep_chunks,
        log=_log,
    )
    print(f"{len(rows)} rows appended to {a.out} (run {rows[0]['run']})")
    return 0


def _cmd_sweep(a: argparse.Namespace) -> int:
    spec = json.loads(Path(a.spec).read_text())
    res = sweep(a.config, a.store, spec, a.out, force=a.force, keep_chunks=a.keep_chunks, log=_log)
    print(f"ran {len(res['ran'])}, skipped {len(res['skipped'])}")
    return 0


def _cmd_compare(a: argparse.Namespace) -> int:
    dates = [int(d) for d in a.dates.split(",")] if a.dates else None
    print("\n".join(compare(read_ledger(a.ledger), a.tag_a, a.tag_b, dates, a.top)))
    return 0


def _cmd_summary(a: argparse.Namespace) -> int:
    print("\n".join(summary_lines(read_ledger(a.ledger))))
    return 0


def _cmd_pareto(a: argparse.Namespace) -> int:
    Path(a.out).write_text(pareto_svg(read_ledger(a.ledger), a.y))
    print(f"wrote {a.out}")
    return 0


def add_parsers(sub: argparse._SubParsersAction) -> None:
    """Register this tool's subcommands under ``lab``."""
    led = sub.add_parser("ledger", help="experiment ledger: run, sweep, compare, summary, pareto")
    ls = led.add_subparsers(dest="ledger_cmd", required=True)

    p = ls.add_parser("run", help="encode dates with a configuration plus overrides, append rows to a ledger")
    p.add_argument("config")
    p.add_argument("store")
    p.add_argument("--out", required=True, help="ledger (.jsonl, appended to)")
    p.add_argument("--dates", help="date indices: 6, 0,6,13, 2:5 or all (default 0)")
    p.add_argument("--tag")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override (dotted key, JSON value)")
    p.add_argument("--dated", action="store_true", help="pass the valid dates to the encoder")
    p.add_argument("--keep-chunks", metavar="DIR")
    p.set_defaults(fn=_cmd_run)

    p = ls.add_parser("sweep", help="run the settings of a specification not yet in the ledger")
    p.add_argument("config")
    p.add_argument("store")
    p.add_argument("spec", help="JSON: settings and / or grid, dates, tag, base, dated")
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true", help="run settings already in the ledger")
    p.add_argument("--keep-chunks", metavar="DIR")
    p.set_defaults(fn=_cmd_sweep)

    p = ls.add_parser("compare", help="per-field byte deltas, errors and times of two tags (tag or tag@runid)")
    p.add_argument("ledger")
    p.add_argument("tag_a")
    p.add_argument("tag_b")
    p.add_argument("--dates", help="restrict to these date indices (comma separated)")
    p.add_argument("--top", type=int, default=15)
    p.set_defaults(fn=_cmd_compare)

    p = ls.add_parser("summary", help="one line per run")
    p.add_argument("ledger")
    p.set_defaults(fn=_cmd_summary)

    p = ls.add_parser("pareto", help="scatter of runs (bytes per date against error or time) as SVG")
    p.add_argument("ledger")
    p.add_argument("out")
    p.add_argument("--x", default="bytes", choices=["bytes"])
    p.add_argument("--y", default="rmse", choices=list(YKEYS))
    p.set_defaults(fn=_cmd_pareto)
