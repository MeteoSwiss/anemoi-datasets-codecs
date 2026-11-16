"""``anemoi-datasets-codecs lab oracles ...``: the subcommands, their runs over the fields of one date, and the text
and JSON output."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import numpy as np

from ...icon_lattice.threads import blas_serial, map_threads
from .. import data as D
from . import core as C
from . import cross as X
from . import estimates as M

NOTE_COMMON = (
    "fields quantised by the plain rule (recon_bits 0; the configuration's k), predicted open loop from the quantised "
    "values (exact for the plain rule), no cross or mask terms unless stated; costs are symbol entropies with per-step "
    "spread contexts plus 16 bits per escape, table bytes ignored"
)
NOTE_DATES = (
    "temporal results need consecutive dates: check that the store's dates are what they claim (relabelled sample "
    "stores have consecutive indices that are not consecutive hours)"
)


# ----------------------------------------------------------------------------------------------- helpers
def _table(rows: list[list[Any]], head: list[str], out=None) -> None:
    out = out or sys.stdout
    cells = [[_fmt(v) for v in r] for r in rows]
    w = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(head)]
    print("  ".join(h.ljust(w[0]) if i == 0 else h.rjust(w[i]) for i, h in enumerate(head)), file=out)
    for r in cells:
        print("  ".join(v.ljust(w[0]) if i == 0 else v.rjust(w[i]) for i, v in enumerate(r)), file=out)


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


def _pct(a: float, b: float) -> float:
    return 100.0 * a / b if b else 0.0


def _per_field(inp: C.Inputs, fn) -> list[Any]:
    """``fn(i, n)`` for every selected field (``n`` its plain-rule integers), on the package's thread pool."""

    C.drop_uncodable(inp)

    def one(i: int):
        return fn(i, C.quantised(inp.x[i], inp.fields[i], inp.k)[2])

    with blas_serial():
        return map_threads(one, len(inp.fields))


def _header(inp: C.Inputs, oracle: str, notes: list[str]) -> dict[str, Any]:
    return {
        "oracle": oracle,
        "store": inp.store,
        "config": inp.config,
        "date": inp.date,
        "date_value": inp.date_value,
        "structure": inp.codec.structure.id,
        "k": inp.k,
        "contexts": inp.nb,
        "n_cells": inp.codec.n_cells,
        "notes": notes,
        "skipped": inp.meta.get("skipped", {}),
        "environment": D.environment(),
    }


def _emit(a: argparse.Namespace, doc: dict[str, Any], text) -> int:
    if a.json:
        print(json.dumps(doc, indent=1, default=float))
    else:
        print(
            f"lab oracles {doc['oracle']}: {doc['store']} date {doc['date']} ({doc['date_value']}), "
            f"structure {doc['structure']}, k {doc['k']:g}, {len(doc.get('fields', doc.get('labels', [])))} fields, "
            f"{doc['elapsed_s']:.1f} s"
        )
        for n in doc["notes"]:
            print(f"note: {n}")
        if doc["skipped"]:
            print("skipped: " + ", ".join(f"{k} ({v})" for k, v in doc["skipped"].items()))
        text(doc)
    return 0


def _inputs(a: argparse.Namespace) -> tuple[C.Inputs, Any, Any]:
    return C.load_inputs(a.store, a.config, a.date, a.fields)


# ------------------------------------------------------------------------------------------------ baseline
def run_baseline(inp: C.Inputs, calibrate: str = "matched", full_x: np.ndarray | None = None) -> dict[str, Any]:
    """The baseline over the selected fields, with the calibration against coded bytes (``none``, ``matched``: the
    selected fields alone with recon_bits 0 and the physics layer off, ``full``: also the configuration's codec on
    the whole date ``full_x``, which then must be given)."""
    S, nc = inp.codec.structure, inp.codec.n_cells
    res = _per_field(inp, lambda i, n: M.baseline_field(n, S, inp.nb, inp.fit))
    rows = []
    for f, r in zip(inp.fields, res, strict=True):
        rows.append(
            {"field": f.label, "index": f.index, **{k: (v / nc if k.startswith("bits") else v) for k, v in r.items()}}
        )
    if calibrate in ("matched", "full"):
        cod = M.calibration_codec(inp.codec, inp.fields, matched=True)
        with blas_serial():
            cb = M.coded_bytes(cod, inp.x)
        for row, c in zip(rows, cb, strict=True):
            row["coded_matched"] = 8.0 * c["bytes"] / nc
            row["mode_matched"] = c["mode"]
            row["ratio"] = row["bits"] / row["coded_matched"] if c["mode"] == "lattice" else None
            row["ratio_esc"] = row["bits_esc"] / row["coded_matched"] if c["mode"] == "lattice" else None
    if calibrate == "full":
        if full_x is None:
            raise ValueError("calibrate full needs the whole date")
        cod = D.codec_for(inp.codec.get_config(), options={"statics": "inline"})
        cb = M.coded_bytes(cod, full_x, [f.index for f in inp.fields])
        for row, c in zip(rows, cb, strict=True):
            row["coded_full"] = 8.0 * c["bytes"] / nc
            row["mode_full"] = c["mode"]
    tot = {k: float(np.mean([r[k] for r in rows])) for k in ("bits_w0", "bits_fit", "bits", "bits_esc")}
    lat = [r for r in rows if r.get("ratio") is not None]
    if lat:
        tot["coded_matched"] = float(np.mean([r["coded_matched"] for r in lat]))
        tot["ratio"] = float(sum(r["bits"] for r in lat) / sum(r["coded_matched"] for r in lat))
        tot["ratio_esc"] = float(sum(r["bits_esc"] for r in lat) / sum(r["coded_matched"] for r in lat))
        tot["ratio_fields_min"] = float(min(r["ratio"] for r in lat))
        tot["ratio_fields_max"] = float(max(r["ratio"] for r in lat))
    if calibrate == "full":
        tot["coded_full"] = float(np.mean([r["coded_full"] for r in rows]))
    return {"fields": rows, "total": tot, "calibrate": calibrate}


def _text_baseline(doc: dict[str, Any]) -> None:
    cal = doc["calibrate"]
    head = ["field", "w0", "fit", "codec", "esc", "+esc"]
    if cal != "none":
        head += ["coded", "ratio", "ratio+esc", "mode"]
    if cal == "full":
        head += ["config"]
    rows = []
    for r in doc["fields"]:
        row = [r["field"], r["bits_w0"], r["bits_fit"], r["bits"], r["escapes"], r["bits_esc"]]
        if cal != "none":
            row += [r["coded_matched"], r["ratio"], r["ratio_esc"], r["mode_matched"]]
        if cal == "full":
            row += [r["coded_full"]]
        rows.append(row)
    print(
        "bits per cell; w0: default weights, fit: fitted sum-to-one, codec: the encoder's per-step choice, esc: "
        "escapes, +esc: codec with the escapes at their side-stream size instead of 16 bits"
        + (
            "; coded: the matched codec's bytes (recon_bits 0, physics off), ratio = codec / coded, ratio+esc = "
            "+esc / coded"
            if cal != "none"
            else ""
        )
        + ("; config: the configuration's codec on the whole date" if cal == "full" else "")
    )
    _table(rows, head)
    t = doc["total"]
    line = f"mean: w0 {t['bits_w0']:.4f}  fit {t['bits_fit']:.4f}  codec {t['bits']:.4f}  +esc {t['bits_esc']:.4f}"
    if "ratio" in t:
        line += f"  coded {t['coded_matched']:.4f}  ratio {t['ratio']:.4f}  ratio+esc {t['ratio_esc']:.4f}"
        line += f" (fields {t['ratio_fields_min']:.3f}..{t['ratio_fields_max']:.3f})"
    if "coded_full" in t:
        line += f"  config {t['coded_full']:.4f}"
    print(line)


# -------------------------------------------------------------------------------------------------- linear
def run_linear(inp: C.Inputs) -> dict[str, Any]:
    S, nc = inp.codec.structure, inp.codec.n_cells
    first = C.first_source(S)
    res = _per_field(inp, lambda i, n: M.linear_field(n, S, inp.nb, first, inp.fit))
    rows = []
    for f, r in zip(inp.fields, res, strict=True):
        d = {k: v / nc for k, v in r.items()}
        d.update(field=f.label, gain=d["bits"] - d["bits_best"])
        d["pct"] = _pct(d["gain"], d["bits"])
        rows.append(d)
    tot = {
        k: float(np.mean([r[k] for r in rows])) for k in ("bits", "bits_fit", "bits_lin", "bits_lin_gp", "bits_best")
    }
    tot["gain"] = tot["bits"] - tot["bits_best"]
    tot["pct"] = _pct(tot["gain"], tot["bits"])
    return {"fields": rows, "total": tot}


def _text_linear(doc: dict[str, Any]) -> None:
    print(
        "bits per cell; codec: the encoder's choice of sum-to-one weights; fit: the fitted sum-to-one weights; lin: "
        "unconstrained linear + intercept; lin+gp: also the nearest source's nearest source; best: per-step minimum "
        "of codec, lin, lin+gp; gain = codec - best"
    )
    keys = ("bits", "bits_fit", "bits_lin", "bits_lin_gp", "bits_best", "gain", "pct")
    _table(
        [[r["field"], *[r[k] for k in keys]] for r in doc["fields"]],
        ["field", "codec", "fit", "lin", "lin+gp", "best", "gain", "%"],
    )
    t = doc["total"]
    print("mean: " + "  ".join(f"{k} {t[k]:.4f}" for k in keys[:-1]) + f"  ({t['pct']:.2f}%)")


# --------------------------------------------------------------------------------------------------- cross
def _text_cross(doc: dict[str, Any]) -> None:
    print(
        f"bits per cell on a sample of {doc['sample']} of {doc['n_cells']} cells; reg: one-coefficient regression "
        "of the target's residual on the source's per step; cond: conditional entropy given the source's residual "
        "bucket; plan: the codec's plan has the term (reverse: the plan has source <- target)"
    )
    head = ["target", "source", "base", "reg", "%", "cond", "plan"]

    def rows(ps):
        return [
            [p["target"], p["source"], p["base"], p["gain_reg"], p["pct_reg"], p["gain_cond"], p["plan"]] for p in ps
        ]

    print("\ntop pairs")
    _table(rows(doc["top"]), head)
    print("\ntop pairs not in the plan (reverses of plan terms excluded)")
    _table(rows(doc["top_missing"]), head)
    if doc["in_plan"]:
        print("\npairs of the plan (both fields selected)")
        _table(rows(doc["in_plan"]), head)
    print("\nper target: best single source")
    _table(
        [
            [
                t["target"],
                t["base"],
                t["best_source"],
                t["gain_reg"],
                _pct(t["gain_reg"], t["base"]),
                t["gain_cond_max"],
                ",".join(t["plan_sources"]),
            ]
            for t in doc["per_target"]
        ],
        ["target", "base", "best", "reg", "%", "cond max", "plan sources"],
    )
    if len(doc["labels"]) <= 16:
        print("\nregression gain matrix (rows: target, columns: source)")
        _table([[lab, *row] for lab, row in zip(doc["labels"], doc["gain_reg"], strict=True)], ["", *doc["labels"]])


# ------------------------------------------------------------------------------------------------ temporal
def run_temporal(inp: C.Inputs, xprev: np.ndarray) -> dict[str, Any]:
    """The temporal oracle; ``xprev``: the previous date's rows of ``inp.fields`` (as given, before any is dropped)."""
    S, nc = inp.codec.structure, inp.codec.n_cells
    labels = [f.label for f in inp.fields]
    C.drop_uncodable(inp)
    xprev = xprev[[labels.index(f.label) for f in inp.fields]]

    def one(i: int, n: np.ndarray):
        off, step, _ = C.quantised(inp.x[i], inp.fields[i], inp.k)
        p = C.requantised(xprev[i], inp.fields[i], inp.k, off, step)
        return M.temporal_field(n, p, S, inp.nb, inp.fit)

    res = _per_field(inp, one)
    rows = []
    for f, r in zip(inp.fields, res, strict=True):
        d = {k: (v / nc if k.startswith("bits") else v) for k, v in r.items()}
        d.update(field=f.label, gain=d["bits"] - d["bits_best"])
        d["pct"] = _pct(d["gain"], d["bits"])
        rows.append(d)
    keys = ("bits", "bits_diff", "bits_mix", "bits_best", "gain")
    tot = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    tot["pct"] = _pct(tot["gain"], tot["bits"])
    return {"fields": rows, "total": tot}


def _text_temporal(doc: dict[str, Any]) -> None:
    print(
        f"previous date {doc['prev']} ({doc['prev_value']}); bits per cell; spatial: no temporal term; diff: spatial "
        "prediction of n - prev; mix: robust per-step mix of s(n), prev, s(prev); best: per-step minimum"
    )
    _table(
        [
            [r["field"], r["bits"], r["bits_diff"], r["bits_mix"], r["bits_best"], r["gain"], r["pct"]]
            for r in doc["fields"]
        ],
        ["field", "spatial", "diff", "mix", "best", "gain", "%"],
    )
    t = doc["total"]
    print(
        f"mean: spatial {t['bits']:.4f}  diff {t['bits_diff']:.4f}  mix {t['bits_mix']:.4f}  "
        f"best {t['bits_best']:.4f}  gain {t['gain']:.4f} ({t['pct']:.2f}%)"
    )


# ------------------------------------------------------------------------------------------------ contexts
def run_contexts(inp: C.Inputs, adaptive: bool = True) -> dict[str, Any]:
    S, nc = inp.codec.structure, inp.codec.n_cells
    res = _per_field(inp, lambda i, n: M.contexts_field(n, S, inp.nb, adaptive, inp.fit))
    rows = []
    for f, r in zip(inp.fields, res, strict=True):
        row = {"field": f.label, "tables": r["tables"], "table_bytes": r["table_bytes"]}
        for what in ("bits", "charged"):
            row[what] = {k: (None if v is None else v / nc) for k, v in r[what].items()}
        rows.append(row)
    out: dict[str, Any] = {"fields": rows, "variants": M.CONTEXT_HELP}
    for what in ("bits", "charged"):
        tot = {}
        for k in list(rows[0][what]) if rows else []:
            vals = [r[what][k] for r in rows]
            tot[k] = None if any(v is None for v in vals) else float(np.mean(vals))
        base = tot.get("spread8") or 0.0
        out["total" if what == "bits" else "total_charged"] = tot
        out["headroom_pct" if what == "bits" else "headroom_charged_pct"] = {
            k: (None if v is None else _pct(base - v, base)) for k, v in tot.items()
        }
    return out


def _text_contexts(doc: dict[str, Any]) -> None:
    names = list(doc["total"])
    print(
        "bits per cell under each context (escapes included, tables not charged); "
        + "; ".join(f"{k}: {v}" for k, v in doc["variants"].items())
        + "; zstd19 / xz9e: the symbol stream compressed adaptively"
    )
    _table([[r["field"], *[r["bits"][k] for k in names]] for r in doc["fields"]], ["field", *names])
    print("mean: " + "  ".join(f"{k} {_fmt(v)}" for k, v in doc["total"].items()))
    print("headroom vs spread8 (%): " + "  ".join(f"{k} {_fmt(v)}" for k, v in doc["headroom_pct"].items()))
    print("with approximate table bytes charged:")
    print("mean: " + "  ".join(f"{k} {_fmt(v)}" for k, v in doc["total_charged"].items()))
    print("headroom vs spread8 (%): " + "  ".join(f"{k} {_fmt(v)}" for k, v in doc["headroom_charged_pct"].items()))
    if doc["fields"]:
        tb = doc["fields"][0]
        print(
            f"tables (first field {tb['field']}): "
            + "  ".join(f"{k} {tb['tables'][k]} (~{tb['table_bytes'][k]} B)" for k in tb["tables"])
        )


# -------------------------------------------------------------------------------------------------- levels
def run_levels(inp: C.Inputs) -> dict[str, Any]:
    S, nc = inp.codec.structure, inp.codec.n_cells
    res = _per_field(inp, lambda i, n: M.levels_field(n, S, inp.nb, inp.fit))
    rows = [
        {"field": f.label, "steps": r, "bits": sum(s["bits"] for s in r) / nc}
        for f, r in zip(inp.fields, res, strict=True)
    ]
    agg = []
    total = sum(sum(s["bits"] for s in r) for r in res) or 1.0
    for j, st in enumerate(S.steps):
        per = [r[j] for r in res]
        b = sum(p["bits"] for p in per)
        agg.append(
            {
                "step": st.name,
                "kind": st.kind,
                "n": st.n,
                "frac_cells": st.n / nc,
                "bits_per_target": b / (len(res) * st.n) if st.n else 0.0,
                "share": b / total,
                "median_var_ratio": float(
                    np.median([p["var_res"] / p["var_data"] if p["var_data"] > 0 else 0.0 for p in per])
                ),
                "fit_chosen": sum(p["choice"] == "fit" for p in per),
            }
        )
    return {"fields": rows, "steps": agg}


def _text_levels(doc: dict[str, Any]) -> None:
    print(
        "per step (coarse to fine), over the selected fields: bits per target, share of all bits, median "
        "var(residual) / var(data), fields where the fitted weights won"
    )
    _table(
        [
            [
                s["step"],
                s["n"],
                s["frac_cells"],
                s["bits_per_target"],
                s["share"],
                s["median_var_ratio"],
                s["fit_chosen"],
            ]
            for s in doc["steps"]
        ],
        ["step", "targets", "cells", "bits/target", "share", "var ratio", "fit"],
    )
    if len(doc["fields"]) <= 20:
        print("\nbits per target per field and step")
        names = [s["step"] for s in doc["steps"]]
        _table(
            [[r["field"], *[s["bits"] / s["n"] if s["n"] else 0.0 for s in r["steps"]]] for r in doc["fields"]],
            ["field", *names],
        )


def _source_dates(store: str) -> list[str] | None:
    """The dates the fields of a sample store came from (group attribute ``sample.source_dates``), if recorded."""
    import zarr

    try:
        s = zarr.open_group(str(store), mode="r").attrs.get("sample")
    except Exception:  # noqa: BLE001 - any store without the attribute
        return None
    src = s.get("source_dates") if isinstance(s, dict) else None
    return [str(d) for d in src] if isinstance(src, list) else None


# --------------------------------------------------------------------------------------------- commands
def _cmd(oracle: str):
    def fn(a: argparse.Namespace) -> int:
        t0 = time.time()
        inp, arr, dates = _inputs(a)
        notes = [NOTE_COMMON]
        extra: dict[str, Any] = {}
        if oracle == "baseline":
            full = D.read_date(arr, a.date) if a.calibrate == "full" else None
            body = run_baseline(inp, a.calibrate, full)
            notes.append(
                "the oracle is expected a few percent below the coded size (table bytes, rANS precision, "
                "range flushes and the escapes' actual byte cost are not modelled)"
            )
            text = _text_baseline
        elif oracle == "linear":
            body, text = run_linear(inp), _text_linear
        elif oracle == "cross":
            body = X.cross(inp.x, inp.fields, inp.codec, a.sample, a.top)
            notes.append("lossy sources only; pairwise gains are not additive; coefficient bytes ignored")
            text = _text_cross
        elif oracle == "temporal":
            prev = a.date - 1 if a.prev is None else a.prev
            if not 0 <= prev < arr.shape[0] or prev == a.date:
                raise SystemExit(f"--prev {prev}: needs another date of the store")
            xprev = C.read_rows(arr, prev, [f.index for f in inp.fields])
            body = run_temporal(inp, xprev)
            extra = {"prev": prev, "prev_value": str(dates[prev]) if dates is not None else None}
            notes.append(NOTE_DATES)
            src = _source_dates(a.store)
            if src is not None:  # a sample store records the dates its fields came from
                extra.update(source_date=src[a.date], prev_source_date=src[prev])
                notes.append(
                    f"the store records source dates: date {a.date} is {src[a.date]}, date {prev} is {src[prev]}"
                )
            text = _text_temporal
        elif oracle == "contexts":
            body, text = run_contexts(inp, not a.no_adaptive), _text_contexts
            notes.append("table bytes of the richer contexts not charged (see the table counts)")
        else:
            body, text = run_levels(inp), _text_levels
        doc = {**_header(inp, oracle, notes), **extra, **body, "elapsed_s": time.time() - t0}
        return _emit(a, doc, text)

    return fn


HELP = {
    "baseline": "the common baseline: the codec's spatial prediction, calibrated against coded bytes",
    "linear": "headroom of unconstrained linear predictors over the step's sources (Wiener-type bound)",
    "cross": "gain of conditioning every lossy field's residual on every other's (pair matrix, plan terms marked)",
    "temporal": "gain of the previous date's field as a predictor (difference, linear mix)",
    "contexts": "context headroom: conditional entropies under richer contexts, zstd / xz on the symbols",
    "levels": "multiresolution profile: per step targets, variances, bits",
}


def add_parsers(sub: argparse._SubParsersAction) -> None:
    """Register ``lab oracles`` and its subcommands."""
    p = sub.add_parser("oracles", help="offline estimates (bits per cell) of what an idea would buy")
    osub = p.add_subparsers(dest="oracle", required=True)
    for name, h in HELP.items():
        q = osub.add_parser(name, help=h, description=h)
        q.add_argument("config", help="lattice codec configuration (file, codec entry, chain, or a store: its own)")
        q.add_argument("store", help="an anemoi store (its data array, dates)")
        q.add_argument("--date", type=int, default=0, help="date index in the store (default 0)")
        q.add_argument("--fields", help="lossy fields: labels, field indices or regular expressions, comma separated")
        q.add_argument("--json", action="store_true", help="JSON output")
        if name == "baseline":
            q.add_argument(
                "--calibrate",
                choices=("none", "matched", "full"),
                default="matched",
                help="compare with coded bytes: matched codec of the selected fields (default), also the "
                "configuration's codec on the whole date (full), or none",
            )
        if name == "cross":
            q.add_argument("--sample", type=int, default=X.DEFAULT_SAMPLE, help="cells sampled (0: all)")
            q.add_argument("--top", type=int, default=25, help="pairs listed")
        if name == "temporal":
            q.add_argument("--prev", type=int, help="the previous date index (default date - 1)")
        if name == "contexts":
            q.add_argument("--no-adaptive", action="store_true", help="skip zstd / xz")
        q.set_defaults(fn=_cmd(name))
