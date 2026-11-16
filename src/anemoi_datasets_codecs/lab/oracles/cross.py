"""The cross oracle: what conditioning one field's residual on another's would buy, for every ordered pair.

For every lossy field the residuals of the codec's spatial prediction (:func:`core.spatial`, open loop, plain rule)
are taken on a deterministic sample of cells (evenly spaced in decode order, so every step is represented in
proportion to its size). For a target ``a`` and a source ``b`` at the same cells:

* regression: per step a robust (IRLS ~ L1, 3 iterations) scalar coefficient ``beta`` of ``r_a`` on ``r_b``, the
  remainder ``r_a - rint(beta r_b)`` costed with ``a``'s spread context; the gain is the drop in cost, per step
  where it is positive (a term that does not pay is dropped per step, as the encoder does). The source residual
  ``r_b`` is the codec's detail of ``b`` (fine-grid units at ``recon_bits`` 0), so this is a one-source cross term.
* conditioning: ``H(s_a | step, ctx_a, bucket(r_b))`` with 7 signed magnitude buckets of ``r_b`` (0, +-1, +-2..3,
  +-4 and beyond), against ``H(s_a | step, ctx_a)``: what a context on the source's residual would buy (an upper
  view that includes non-linear dependence; table bytes ignored).

Gains are in bits per cell of the sample; every entropy here (base, remainder, conditional) carries the Miller-Madow
bias correction, since the conditioning variant splits the sample into 7 times more contexts. Ignored: static and
lossless sources (orography and the like, which the plan also uses), several sources at once (gains of pairs with the
same target are not additive), the coefficient bytes, and the decode-order constraint (``a <- b`` and ``b <- a`` cannot
both be used; the plan's ``max_depth``). The plan's terms are marked: ``plan`` when ``b`` is a cross source of ``a`` in
``codec.cross_plan``, ``reverse`` when the plan has ``a`` as a cross source of ``b`` (listed in the top pairs, not
in the top pairs missing from the plan), ``dropped`` when the physics layer proposed it and the codec's limits dropped
it (``plan_report``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...icon_lattice.structure import Structure
from ...icon_lattice.threads import blas_serial, map_threads
from .. import data as D
from .. import entropy as E
from . import core as C

DEFAULT_SAMPLE = 1 << 18


def sample_positions(n_cells: int, size: int) -> np.ndarray:
    """``size`` decode-order positions evenly spaced over ``n_cells`` (all when ``size`` is 0 or larger)."""
    if size <= 0 or size >= n_cells:
        return np.arange(n_cells)
    return np.linspace(0, n_cells - 1, size).astype(np.int64)


def _field_sample(
    n: np.ndarray, structure: Structure, nb: int, pos: np.ndarray, fit: bool = True
) -> tuple[np.ndarray, np.ndarray, float]:
    sp = C.spatial(n, structure, nb, fit)
    r = np.concatenate([s.r for s in sp.steps])[pos]
    ctx = np.concatenate([s.ctx if s.ctx is not None else np.zeros(len(s.r), np.intp) for s in sp.steps])[pos]
    return np.clip(r, -(2**30), 2**30).astype(np.int32), ctx.astype(np.uint8), sp.bits


def _per_step_bits(sym: np.ndarray, key: np.ndarray, n_keys: int, n_steps: int, esc_step: np.ndarray) -> np.ndarray:
    h = C.hist_bits(C.keyed_counts(sym, key, n_keys), miller_madow=True).reshape(n_steps, -1).sum(1)
    return h + esc_step


def pair_gains(
    R: np.ndarray, CTX: np.ndarray, G: np.ndarray, nb: int, n_steps: int, a: int, iters: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For target ``a``: its baseline cost per step and the regression and conditioning gains (bits, totals over the
    sample, summed over steps where positive) from every source row of ``R``."""
    ra_i = R[a].astype(np.int64)
    ra = ra_i.astype(np.float64)
    key = G * nb + CTX[a]
    sym = E.symbols(ra_i)
    esc = 16.0 * np.bincount(G, weights=(np.abs(ra_i) > 127), minlength=n_steps)
    base = _per_step_bits(sym, key, n_steps * nb, n_steps, esc)
    nf = len(R)
    reg = np.zeros(nf)
    cond = np.zeros(nf)
    for b in range(nf):
        if b == a:
            continue
        rb_i = R[b].astype(np.int64)
        rb = rb_i.astype(np.float64)
        rab, rbb = ra * rb, rb * rb
        num = np.bincount(G, weights=rab, minlength=n_steps)
        den = np.bincount(G, weights=rbb, minlength=n_steps)
        beta = np.divide(num, den, out=np.zeros(n_steps), where=den > 0)
        for _ in range(iters):
            e = ra - beta[G] * rb
            np.abs(e, out=e)
            np.maximum(e, 0.5, out=e)
            np.divide(1.0, e, out=e)
            num = np.bincount(G, weights=e * rab, minlength=n_steps)
            den = np.bincount(G, weights=e * rbb, minlength=n_steps)
            beta = np.divide(num, den, out=np.zeros(n_steps), where=den > 0)
        rem = ra_i - np.rint(beta[G] * rb).astype(np.int64)
        esc_r = 16.0 * np.bincount(G, weights=(np.abs(rem) > 127), minlength=n_steps)
        reg[b] = np.maximum(base - _per_step_bits(E.symbols(rem), key, n_steps * nb, n_steps, esc_r), 0).sum()
        key2 = key * 7 + C.signed_bucket(rb_i)
        cond[b] = np.maximum(base - _per_step_bits(sym, key2, n_steps * nb * 7, n_steps, esc), 0).sum()
    return base, reg, cond


def plan_status(codec: Any) -> dict[tuple[str, str], str]:
    """``(target, source) -> "plan" | "reverse" | "dropped (reason)"`` of the codec's plan (names); ``reverse``: the
    plan has ``source <- target``, so this pair cannot be used as well."""
    out: dict[tuple[str, str], str] = {}
    for t in codec.plan_report():
        for s, why in t.get("dropped", []):
            out[(t["target"], s)] = f"dropped ({why})"
    for tgt, srcs in codec.cross_plan.items():
        for s in srcs:
            out[(tgt, s)] = "plan"
    for tgt, srcs in codec.cross_plan.items():
        for s in srcs:
            if out.get((s, tgt)) != "plan":
                out[(s, tgt)] = "reverse"
    return out


def cross(
    xs: np.ndarray, fields: list[C.FieldSpec], codec: Any, sample: int = DEFAULT_SAMPLE, top: int = 25
) -> dict[str, Any]:
    """The cross oracle on the rows ``xs`` of the lossy ``fields`` of one date (see the module docstring)."""
    S = codec.structure
    k = float(codec.options["k"])
    nb = max(1, min(8, int(codec.options["contexts"])))
    fit = C.fits_weights(codec)
    pos = sample_positions(S.n_cells, sample)
    G = np.concatenate([np.full(st.n, i, np.int64) for i, st in enumerate(S.steps)])[pos]
    ns, nst, nf = len(pos), len(S.steps), len(fields)

    def one(i: int):
        return _field_sample(C.quantised(xs[i], fields[i], k)[2], S, nb, pos, fit)

    with blas_serial():
        res = map_threads(one, nf)
    R = np.stack([r[0] for r in res])
    CTX = np.stack([r[1] for r in res])
    full_bits = [r[2] for r in res]
    del res
    with blas_serial():
        pg = map_threads(lambda a: pair_gains(R, CTX, G, nb, nst, a), nf)
    base = np.array([p[0].sum() for p in pg]) / ns
    reg = np.stack([p[1] for p in pg]) / ns
    cond = np.stack([p[2] for p in pg]) / ns
    idx = D.field_index(codec)
    names = [idx.name(f.index) for f in fields]
    status = plan_status(codec)
    pairs = []
    for a in range(nf):
        for b in range(nf):
            if a == b:
                continue
            pairs.append(
                {
                    "target": fields[a].label,
                    "source": fields[b].label,
                    "gain_reg": float(reg[a, b]),
                    "gain_cond": float(cond[a, b]),
                    "base": float(base[a]),
                    "pct_reg": float(100 * reg[a, b] / base[a]) if base[a] > 0 else 0.0,
                    "plan": status.get((names[a], names[b]), ""),
                }
            )
    pairs.sort(key=lambda p: -p["gain_reg"])
    missing = [p for p in pairs if p["plan"] not in ("plan", "reverse")]
    in_plan = [p for p in pairs if p["plan"] == "plan"]
    best = []
    for a in range(nf):
        b = int(np.argmax(reg[a])) if nf > 1 else a
        best.append(
            {
                "target": fields[a].label,
                "base": float(base[a]),
                "best_source": fields[b].label if nf > 1 else None,
                "gain_reg": float(reg[a, b]) if nf > 1 else 0.0,
                "gain_cond_max": float(cond[a].max()) if nf > 1 else 0.0,
                "plan_sources": list(codec.cross_plan.get(names[a], [])),
            }
        )
    return {
        "labels": [f.label for f in fields],
        "sample": int(ns),
        "n_cells": int(S.n_cells),
        "base": base.tolist(),
        "full_bits_per_cell": [b / S.n_cells for b in full_bits],
        "gain_reg": np.round(reg, 5).tolist(),
        "gain_cond": np.round(cond, 5).tolist(),
        "top": pairs[:top],
        "top_missing": missing[:top],
        "in_plan": in_plan,
        "per_target": best,
    }
