"""lab residuals: the lattice codec's residuals per cell, for analysis in notebooks.

``lab residuals dump CONFIG STORE --date I --fields A,B,... OUT.npz`` encodes date ``I`` of ``STORE`` with ``CONFIG``
(the whole date: cross terms need their sources) recording the chosen fields (``lab account``'s recorder) and writes
per field, in native cell order: ``n`` (the quantised integers on the fine grid ``g``), ``v`` (the decoded integers,
the closed-loop reconstruction), ``pred`` (the integer prediction on the fine grid), ``q`` (the coded residual: ``v =
pred + 2^m q``), ``step`` (the structure step that codes the cell), ``ctx`` (its entropy table), ``spread`` (the
context spread, -1 in the delta step), ``masked`` (predicted by the mask term), and the scalars ``offset``, ``g``,
``m``, ``bound``; the decoded value of a cell is ``clip(offset + g * v)``. Keys are ``FIELD/name``; ``meta`` (JSON) has
the run, ``grid/pixel_index`` and ``grid/shape`` the lattice.

``lab residuals summary OUT.npz`` prints per field: the Laplace fit of ``q`` (and the order-0 bits per cell of the
residuals against those of the fitted discrete Laplace), percentiles of ``|q|``, the escape rate (``|q| > 127``), the
zero fraction, the correlation of ``q`` and of ``|q|`` between edge neighbours of the triangle lattice, and the mean
``|q|`` per step.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..icon_lattice.container import MODE_LABELS
from ..icon_lattice.structure import cell_image
from . import account, data
from .entropy import entropy_bits, laplace_fit, symbols

PERCENTILES = (50, 90, 99, 99.9)


def _narrow(a: np.ndarray) -> np.ndarray:
    """int32 when the values fit, else the array as it is."""
    if a.dtype.kind == "i" and len(a) and -(2**31) <= int(a.min()) and int(a.max()) < 2**31:
        return a.astype(np.int32)
    return a


def cell_arrays(pieces: dict[str, Any], structure: Any) -> dict[str, np.ndarray]:
    """Per cell (native order) arrays of one recorded LATTICE coding (see the module docstring)."""
    st, m = pieces["streams"], int(pieces["m"])
    offset, g, n = pieces["quant"]
    v = np.asarray(pieces["values"], np.int64)
    nc = len(v)
    q = np.zeros(nc, np.int64)
    step = np.zeros(nc, np.uint8)
    ctx = np.zeros(nc, np.uint8)
    spread = np.full(nc, -1, np.int64)
    masked = np.zeros(nc, bool)
    for s, (stp, sc) in enumerate(zip(structure.steps, pieces["steps"], strict=True)):
        t = stp.t_idx
        q[t] = account.residuals_of(sc)
        step[t] = s
        ctx[t] = account.table_ids(sc, st, s)
        if sc.spread is not None:
            spread[t] = sc.spread
        if sc.masked is not None:
            masked[t] = sc.masked
    pred = v - (q << m)
    return {
        "n": _narrow(np.asarray(n, np.int64)),
        "v": _narrow(v),
        "pred": _narrow(pred),
        "q": _narrow(q),
        "step": step,
        "ctx": ctx,
        "spread": _narrow(spread),
        "masked": masked,
        "offset": np.float32(offset),
        "g": np.float32(g),
        "m": np.int64(m),
    }


def dump(codec: Any, store: Any, date: int, fields: set[int], out: Path, valid_date: bool = True) -> dict[str, Any]:
    """Encode the date with recording and write the npz of the recorded fields; returns the meta."""
    chunk, final, run = account.encode_recorded(codec, store, date, fields, keep=True, valid_date=valid_date)
    labels = data.field_labels(codec)
    bounds = {i: b for i, _, b, _ in data.lossy_fields(codec)}
    arrays: dict[str, np.ndarray] = {}
    meta_fields = []
    for i in sorted(final):
        summary, pieces, mode, fb = final[i]
        lab = labels[i]
        for k, a in cell_arrays(pieces, codec.structure).items():
            arrays[f"{lab}/{k}"] = np.asarray(a)
        arrays[f"{lab}/bound"] = np.float64(bounds.get(i, np.nan))
        meta_fields.append({"index": i, "field": lab, "mode": MODE_LABELS[mode], "fallback": fb,
                            "bytes": summary["bytes"]})  # fmt: skip
    layout = codec.layout
    arrays["grid/pixel_index"] = layout.pixel_index
    arrays["grid/shape"] = np.asarray(layout.shape, np.int64)
    meta = {"run": run, "chunk_bytes": len(chunk), "fields": meta_fields, "structure": codec.structure.id,
            "steps": [s.name for s in codec.structure.steps], "environment": data.environment()}  # fmt: skip
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(out, **arrays)
    return meta


def neighbour_pairs(pixel_index: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Every pair of edge neighbours of the triangle lattice once: ``(a, b)`` native cell indices. Neighbours of
    ``(r, c)``: ``(r, c - 1)``, ``(r, c + 1)`` and ``(r - 1, c)`` when ``r + c`` is even, else ``(r + 1, c)``."""
    img = cell_image(pixel_index, tuple(int(s) for s in shape))
    h_a, h_b = img[:, :-1].ravel(), img[:, 1:].ravel()
    r, c = np.indices(img.shape)
    even = ((r + c) % 2 == 0)[1:, :]  # (r, c) with r >= 1 and r + c even: neighbour (r - 1, c)
    v_a, v_b = img[1:, :][even], img[:-1, :][even]
    a = np.concatenate([h_a, v_a])
    b = np.concatenate([h_b, v_b])
    ok = (a >= 0) & (b >= 0)
    return a[ok], b[ok]


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x.astype(np.float64), y.astype(np.float64)
    sx, sy = x.std(), y.std()
    if not sx or not sy:
        return 0.0
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def laplace_bits(q: np.ndarray, scale: float) -> float:
    """Bits per value of ``q`` under the discrete Laplace ``p(k) = (1 - t) / (1 + t) t^|k|``, ``t = exp(-1 /
    scale)`` (the shape the entropy coder would see if the residuals were Laplacian)."""
    if scale <= 0:
        return 0.0
    t = np.exp(-1.0 / scale)
    a = np.abs(q.astype(np.float64))
    return float(-(np.log2((1 - t) / (1 + t)) + a.mean() * np.log2(t)))


def summarise_field(z: Any, lab: str, pairs: tuple[np.ndarray, np.ndarray] | None) -> dict[str, Any]:
    """The summary of one field of a residuals npz (see the module docstring)."""
    q = np.asarray(z[f"{lab}/q"], np.int64)
    step = np.asarray(z[f"{lab}/step"])
    aq = np.abs(q)
    loc, scale = laplace_fit(q)
    sym = symbols(q)
    d: dict[str, Any] = {
        "field": lab,
        "n": len(q),
        "m": int(z[f"{lab}/m"]),
        "g": float(z[f"{lab}/g"]),
        "bound": float(z[f"{lab}/bound"]),
        "laplace_loc": loc,
        "laplace_scale": scale,
        "h0_bits": entropy_bits(sym) / len(q),
        "laplace_bits": laplace_bits(q - int(round(loc)), scale),
        "percentiles": {str(p): float(np.percentile(aq, p)) for p in PERCENTILES},
        "max_abs": int(aq.max()),
        "escape_rate": float(np.count_nonzero(sym == 255) / len(q)),
        "zero_frac": float(np.count_nonzero(q == 0) / len(q)),
        "mean_abs_per_step": [
            float(aq[step == s].mean()) if np.any(step == s) else 0.0 for s in range(int(step.max()) + 1)
        ],
    }
    if pairs is not None:
        a, b = pairs
        d["neighbour_corr"] = _corr(q[a], q[b])
        d["neighbour_corr_abs"] = _corr(aq[a], aq[b])
    return d


def summary(path: Path) -> dict[str, Any]:
    """Per field summaries of a residuals npz."""
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        pairs = neighbour_pairs(z["grid/pixel_index"], tuple(z["grid/shape"])) if "grid/shape" in z.files else None
        fields = [summarise_field(z, f["field"], pairs) for f in meta["fields"]]
    for f, m in zip(fields, meta["fields"], strict=True):
        f["mode"], f["bytes"] = m["mode"], m["bytes"]
    return {"meta": meta, "fields": fields}


def print_summary(s: dict[str, Any]) -> None:
    run = s["meta"]["run"]
    print(f"date {run['date']} (valid date {run['valid_date']}), structure {s['meta']['structure']}")
    pct = " ".join(f"{'p' + str(p):>6}" for p in PERCENTILES)
    print(
        f"{'field':16s} {'mode':8s} {'m':>1} {'scale':>7} {'H0 b':>6} {'Lap b':>6} {pct} {'max':>7} {'esc':>8}"
        f" {'zero':>6} {'corr q':>7} {'corr|q|':>7}"
    )
    for f in s["fields"]:
        p = " ".join(f"{v:6.1f}" for v in f["percentiles"].values())
        print(
            f"{f['field']:16s} {f['mode']:8s} {f['m']:1d} {f['laplace_scale']:7.3f} {f['h0_bits']:6.3f}"
            f" {f['laplace_bits']:6.3f} {p} {f['max_abs']:7d} {f['escape_rate']:8.5f} {f['zero_frac']:6.3f}"
            f" {f.get('neighbour_corr', float('nan')):7.3f} {f.get('neighbour_corr_abs', float('nan')):7.3f}"
        )
        print("    mean |q| per step: " + " ".join(f"{v:.2f}" for v in f["mean_abs_per_step"]))


def _dump(a: argparse.Namespace) -> int:
    if a.threads:
        os.environ["ANEMOI_DATASETS_CODECS_THREADS"] = str(a.threads)
    codec = data.codec_for(a.config)
    fields = account.select_fields(codec, a.fields)
    meta = dump(codec, a.store, a.date, fields or set(), a.out, not a.no_valid_date)
    names = ", ".join(f["field"] for f in meta["fields"])
    print(f"{a.out}: {len(meta['fields'])} fields of date {a.date} ({names})")
    return 0


def _summary(a: argparse.Namespace) -> int:
    s = summary(a.npz)
    account.dump_json(s) if a.json else print_summary(s)
    return 0


def add_parsers(sub: argparse._SubParsersAction) -> None:
    """Register ``residuals dump`` and ``residuals summary`` under ``lab``."""
    p = sub.add_parser("residuals", help="per-cell residuals of chosen fields to an npz (dump), and their summary")
    rsub = p.add_subparsers(dest="residuals_cmd", required=True)
    d = rsub.add_parser("dump", help="encode a date recording the chosen fields, write their per-cell arrays")
    d.add_argument("config", type=Path, help="lattice codec configuration (store, JSON / YAML file)")
    d.add_argument("store", type=Path, help="anemoi store (Zarr) with the data array")
    d.add_argument("--date", type=int, required=True, help="date index")
    d.add_argument("--fields", required=True, help="comma-separated fields")
    d.add_argument("out", type=Path, metavar="OUT.npz", help="output file")
    d.add_argument("--no-valid-date", action="store_true", help="encode without the valid date")
    d.add_argument("--threads", type=int, help="encoder threads (ANEMOI_DATASETS_CODECS_THREADS)")
    d.set_defaults(fn=_dump)
    q = rsub.add_parser("summary", help="per field: Laplace fit, |q| percentiles, escapes, neighbour correlation")
    q.add_argument("npz", type=Path, metavar="OUT.npz", help="a file written by dump")
    q.add_argument("--json", action="store_true", help="JSON output")
    q.set_defaults(fn=_summary)
