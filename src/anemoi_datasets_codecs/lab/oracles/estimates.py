"""The per-field oracles: baseline (and its calibration against coded bytes), linear, temporal, contexts, levels.

Every function takes one field's integers (or floats and its :class:`core.FieldSpec`) and the structure, and returns a
JSON-able dict with costs in bits (totals; the callers divide by the number of cells).
"""

from __future__ import annotations

import lzma
import shutil
import subprocess
from typing import Any

import numpy as np

from ...icon_lattice import container
from ...icon_lattice.codec import LatticeCodec
from ...icon_lattice.encoder import fit_robust
from ...icon_lattice.structure import Structure
from .. import entropy as E
from . import core as C


# ------------------------------------------------------------------------------------------------ baseline
def baseline_field(n: np.ndarray, structure: Structure, nb: int, fit: bool = True) -> dict[str, Any]:
    """Cost of one field with the codec's spatial prediction (:func:`core.spatial`): default weights, fitted
    weights, and the encoder's per-step choice between them, each with the codec's spread context and 16 bits per
    escape (``lab.entropy``); ``bits_esc``: the choice with the escapes at the size of the codec's side stream."""
    sp = C.spatial(n, structure, nb, fit)
    r = np.concatenate([s.r for s in sp.steps])
    esc = r[np.abs(r) > 127]
    return {
        "bits_w0": sp.bits_w0,
        "bits_fit": sp.bits_fit,
        "bits": sp.bits,
        "bits_esc": sp.bits - 16.0 * len(esc) + escape_stream_bits(esc),
        "escapes": len(esc),
        "choice": [s.choice for s in sp.steps],
    }


def escape_stream_bits(esc: np.ndarray) -> float:
    """The size of the codec's escape side stream of the escaped residuals ``esc`` (``|r| > 127``): int16 when they
    all fit, else int32, or LEB128 varints of ``zigzag(r) - 255`` when smaller (docs/codecs/icon_lattice.md
    "Entropy coding")."""
    esc = np.asarray(esc, np.int64)
    if len(esc) == 0:
        return 0.0
    et = 2 if esc.min() >= -(2**15) and esc.max() < 2**15 else 4
    zz = E.zigzag(esc) - 255
    nvar = len(zz) + sum(int(np.count_nonzero(zz >= (1 << k))) for k in (7, 14, 21, 28))
    return 8.0 * min(nvar, et * len(esc))


def calibration_codec(codec: LatticeCodec, fields: list[C.FieldSpec], matched: bool) -> LatticeCodec:
    """A codec of the selected fields alone: the configuration's options and structure, no static fields. With
    ``matched`` the plain rule (``recon_bits`` 0) and no physics layer (no cross or mask terms): the coding the
    baseline oracle models."""
    cfg = codec.get_config()
    if int(cfg.get("n_ensembles", 1)) != 1:
        raise ValueError("calibration needs n_ensembles 1")
    opts = dict(cfg["options"])
    if matched:
        opts["recon_bits"] = 0
    return LatticeCodec(
        variables=[f.label for f in fields],
        grid=codec.layout,
        bounds={f.label: f.bound for f in fields},
        clip={f.label: list(f.clip) for f in fields if f.clip is not None},
        structure=cfg["structure"],
        lossless=cfg["lossless"],
        options=opts,
        physics=False if matched else cfg["physics"],
    )


def coded_bytes(codec: LatticeCodec, x: np.ndarray, rows: list[int] | None = None) -> list[dict[str, Any]]:
    """Encode one date ``x`` (fields, cells) with ``codec``; per field (or per ``rows``) the bytes of its record in
    ``container.inspect_chunk`` (payload plus its share of the chunk header) and its mode."""
    buf = codec.encode(np.ascontiguousarray(x, np.float32).reshape(1, len(x), 1, -1))
    rec = container.inspect_chunk(buf)
    rows = list(range(len(x))) if rows is None else rows
    return [{"bytes": rec[i].nbytes, "mode": container.MODE_LABELS.get(rec[i].mode, str(rec[i].mode))} for i in rows]


# -------------------------------------------------------------------------------------------------- linear
def linear_field(n: np.ndarray, structure: Structure, nb: int, first: np.ndarray, fit: bool = True) -> dict[str, Any]:
    """The Wiener-type bound of a linear predictor over the step's sources.

    Per gather step, the unconstrained predictor ``a + b_0 v_0 + sum_j b_j (v_j - v_0)`` (intercept, no sum-to-one,
    float coefficients) fitted with ``encoder.fit_robust`` (IRLS ~ L1, the criterion the encoder uses; not the L2
    Wiener filter) on the encoder's subsample, rounded to the integer grid (``bits_lin``); the same with one extra
    regressor, the nearest source of the nearest source (``first[src_0]``, ``bits_lin_gp``); and per step the
    cheapest of the codec's choice and these two (``bits_best``: an encoder carrying coefficients would keep the old
    predictor where it is cheaper; without sum-to-one a flat neighbourhood is no longer predicted exactly, which
    costs on clipped or plateau fields). Delta steps keep their delta residuals. Costs with the codec's spread
    context, against the codec's choice (``bits``) and the fitted sum-to-one weights (``bits_fit``) of
    :func:`core.spatial`. Ignores the cost of carrying the coefficients (a few bytes per step)."""
    n = np.asarray(n, np.int64)
    sp = C.spatial(n, structure, nb, fit)
    lin = lin_gp = best = 0.0
    for st, sr in zip(structure.steps, sp.steps, strict=True):
        if st.kind == "delta" or st.n <= 6 * (st.k + 2):
            lin += sr.bits
            lin_gp += sr.bits
            best += sr.bits
            continue
        V = n[st.src_idx]
        y0 = (n[st.t_idx] - V[0]).astype(np.float64)
        v0c = V[0] - np.median(V[0][:: max(1, st.n // 4096)])
        cols = [np.ones(st.n), v0c.astype(np.float64), *[(V[j] - V[0]).astype(np.float64) for j in range(1, st.k)]]
        gp = (n[first[st.src_idx[0]]] - V[0]).astype(np.float64)
        s = C.fit_sample(st.n)
        b = []
        for extra in (False, True):
            X = np.stack(cols + ([gp] if extra else []), 1)
            w = fit_robust(X[s], y0[s])
            b.append(C.cost_bits((y0 - np.rint(X @ w)).astype(np.int64), sr.ctx))
        lin += b[0]
        lin_gp += b[1]
        best += min(sr.bits, *b)
    return {"bits": sp.bits, "bits_fit": sp.bits_fit, "bits_lin": lin, "bits_lin_gp": lin_gp, "bits_best": best}


# ------------------------------------------------------------------------------------------------ temporal
def temporal_field(n: np.ndarray, p: np.ndarray, structure: Structure, nb: int, fit: bool = True) -> dict[str, Any]:
    """Cost given the previous date ``p`` (on the same grid as ``n``, :func:`core.requantised`).

    ``bits``: spatial only (:func:`core.spatial` on ``n``). ``bits_diff``: (a), the spatial prediction of the
    difference ``n - p``. ``bits_mix``: (b), per gather step the robust fit of ``n`` on the spatial prediction
    ``s(n)``, the previous value ``p`` and its spatial prediction ``s(p)`` (``n - s(n) ~ a + b_1 (p - s(n)) + b_2
    (s(p) - s(n))``: spatial only and (a) are special cases), rounded to the grid. ``bits_best``: per step the
    cheapest of the three (an encoder with a per-step mode). Costs with the codec's spread context of the coded
    quantity; coefficient bytes ignored; open loop."""
    n = np.asarray(n, np.int64)
    p = np.asarray(p, np.int64)
    sp = C.spatial(n, structure, nb, fit, keep_pred=True)
    sd = C.spatial(n - p, structure, nb, fit)
    spp = C.spatial(p, structure, nb, fit, keep_pred=True)
    mix = best = 0.0
    for st, a, b, c in zip(structure.steps, sp.steps, sd.steps, spp.steps, strict=True):
        if st.kind == "delta" or st.n <= 24:
            m = a.bits
        else:
            y = n[st.t_idx]
            ps = a.pred
            X = np.stack([np.ones(st.n), (p[st.t_idx] - ps).astype(np.float64), (c.pred - ps).astype(np.float64)], 1)
            s = C.fit_sample(st.n)
            w = fit_robust(X[s], (y - ps)[s].astype(np.float64))
            r = y - ps - np.rint(X @ w).astype(np.int64)
            m = C.cost_bits(r, a.ctx)
        mix += m
        best += min(a.bits, b.bits, m)
    return {
        "bits": sp.bits,
        "bits_diff": sd.bits,
        "bits_mix": mix,
        "bits_best": best,
        "identical": bool((n == p).all()),
    }


# ------------------------------------------------------------------------------------------------ contexts
CONTEXTS = ("order0", "step", "spread8", "spread32", "parent", "spread8xparent", "srcsum")
CONTEXT_HELP = {
    "order0": "one table for the field",
    "step": "a table per step",
    "spread8": "per step x 8 spread buckets (the codec's context: the baseline)",
    "spread32": "per step x 32 spread buckets",
    "parent": "per step x 8 buckets of |r| at the nearest source",
    "spread8xparent": "per step x 8 spread x 8 parent-|r| buckets",
    "srcsum": "per step x 8 buckets of sum |r| over all sources",
}


def _approx_table_bytes(counts: np.ndarray) -> int:
    """Rough serialised size of the non-empty tables of a histogram ``(tables, 256)``: frequencies normalised to
    4096, LEB128 (1 byte below 128, else 2) up to the last non-zero symbol, plus one byte per table."""
    out = 0
    for c in counts:
        tot = c.sum()
        if tot == 0:
            continue
        f = np.where(c > 0, np.maximum(1, np.round(c * 4096 / tot)), 0)
        nz = np.flatnonzero(f[:255])
        m = int(nz[-1]) + 1 if len(nz) else 0
        out += 1 + int((1 + (f[:m] >= 128)).sum()) + 1 + int(f[255] >= 128)
    return out


def _compressed_bits(sym: bytes) -> dict[str, float | None]:
    """The symbol stream (one byte per symbol) through ``zstd -19`` (the executable on PATH, if any) and
    ``xz -9e`` (stdlib lzma), in bits."""
    out: dict[str, float | None] = {"zstd19": None, "xz9e": None}
    exe = shutil.which("zstd")
    if exe:
        res = subprocess.run([exe, "-19", "-q", "-c", "-"], input=sym, capture_output=True, check=False)
        if res.returncode == 0:
            out["zstd19"] = 8.0 * len(res.stdout)
    out["xz9e"] = 8.0 * len(lzma.compress(sym, preset=9 | lzma.PRESET_EXTREME))
    return out


def contexts_field(
    n: np.ndarray, structure: Structure, nb: int, adaptive: bool = True, fit: bool = True
) -> dict[str, Any]:
    """Context headroom of one field: the cost of the codec's residuals (:func:`core.spatial`, the codec's choice of
    weights) under the contexts of :data:`CONTEXTS` (conditional entropy, plus 16 bits per escape in every
    variant), with the number of non-empty tables, a rough table size (the model cost the entropies ignore) and the
    cost with those table bytes charged (``charged``).
    Parent residuals are those the decoder has at the source cells (open loop). With ``adaptive`` also the symbol
    stream in decode order through ``zstd -19`` and ``xz -9e`` plus the same escape bits: what cheap adaptive
    modelling gets without any context."""
    sp = C.spatial(n, structure, nb, fit)
    rc = sp.cells(structure)
    r = np.concatenate([s.r for s in sp.steps])
    sym = E.symbols(r)
    esc = E.escape_bits(r)
    nst = len(sp.steps)
    step = np.concatenate([np.full(len(s.r), i, np.int64) for i, s in enumerate(sp.steps)])
    spread8, n8 = sp.ctx_key()
    first = C.first_source(structure)
    s32, par, ssum = [], [], []
    for st in structure.steps:
        if st.kind == "delta":
            z = np.zeros(st.n, np.int64)
            s32.append(z)
            par.append(C.log_bucket(rc[first[st.t_idx]]))
            ssum.append(z)
            continue
        V = n[st.src_idx]
        kk = min(st.k, 4)
        s32.append(C.quantile_buckets(V[:kk].max(0) - V[:kk].min(0), 32)[0])
        par.append(C.log_bucket(rc[st.src_idx[0]]))
        ssum.append(C.log_bucket(np.abs(rc[st.src_idx]).sum(0)))
    s32 = np.concatenate(s32)
    par = np.concatenate(par)
    ssum = np.concatenate(ssum)
    keys = {
        "order0": (None, 1),
        "step": (step, nst),
        "spread8": (spread8, n8),
        "spread32": (step * 32 + s32, nst * 32),
        "parent": (step * 8 + par, nst * 8),
        "spread8xparent": (spread8 * 8 + par, n8 * 8),
        "srcsum": (step * 8 + ssum, nst * 8),
    }
    out: dict[str, Any] = {"bits": {}, "charged": {}, "tables": {}, "table_bytes": {}}
    for name, (key, nk) in keys.items():
        h = C.keyed_counts(sym, key, nk)
        out["bits"][name] = float(C.hist_bits(h).sum()) + esc
        out["tables"][name] = int((h.sum(1) > 0).sum())
        out["table_bytes"][name] = _approx_table_bytes(h)
        out["charged"][name] = out["bits"][name] + 8.0 * out["table_bytes"][name]
    if adaptive:
        for k, v in _compressed_bits(sym.astype(np.uint8).tobytes()).items():
            out["bits"][k] = out["charged"][k] = None if v is None else v + esc
    return out


# -------------------------------------------------------------------------------------------------- levels
def levels_field(n: np.ndarray, structure: Structure, nb: int, fit: bool = True) -> list[dict[str, Any]]:
    """The multiresolution profile of one field: per step (refinement level) the number of targets, the variance
    of the targets' values and of the codec's residuals (quantisation units, :func:`core.spatial`), the mean
    ``|r|`` and the cost in bits (with the codec's context)."""
    n = np.asarray(n, np.int64)
    sp = C.spatial(n, structure, nb, fit)
    out = []
    for st, sr in zip(structure.steps, sp.steps, strict=True):
        y = n[st.t_idx]
        out.append(
            {
                "step": st.name,
                "kind": st.kind,
                "n": st.n,
                "var_data": float(np.var(y)) if st.n else 0.0,
                "var_res": float(np.var(sr.r)) if st.n else 0.0,
                "mean_abs_res": float(np.mean(np.abs(sr.r))) if st.n else 0.0,
                "bits": sr.bits,
                "choice": sr.choice,
            }
        )
    return out
