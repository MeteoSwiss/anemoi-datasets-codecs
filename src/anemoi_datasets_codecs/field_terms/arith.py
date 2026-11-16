"""The arithmetic of the ``field_terms`` stage: the robust fit (encoder only), the prediction, the residual, the
reconstruction and the per-cell verification. The decoder's half (``predict``, ``reconstruct``) is part of the
format (docs/codecs/field_terms.md) and must round exactly as written here.

* ``predict``: ``p = c0``, then ``p = p + c_i * s_i`` for the sources in configured order, every operation a float64
  ufunc (``dtype=float64`` forces the float64 loop on every numpy; no BLAS, no fused multiply-add), then rounded to
  float32: ``pred = fl32(p)``.
* residual ``r = fl32(x - pred)`` and reconstruction ``x' = clip(fl32(r' + pred))``: float32 ufuncs, which round the
  exact result to nearest, the same as rounding the float64 result (float64 has at least 2 x 24 + 2 bits, so the
  double rounding is innocuous for a sum of two float32 values).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

F4, F8 = np.float32, np.float64
IRLS_ITERATIONS = 8  # reweighting steps after the least-squares start (as measured, docs/codecs/field_terms.md)
IRLS_FLOOR = 0.5  # floor of |residual| in the weights, in units of the target's bound
_DELTA = 2.0**-50  # relative widening of the verified interval: covers the float64 roundings (field_terms.md)

Clip = tuple[float | None, float | None] | None


def fit_robust(
    y: np.ndarray, sources: Sequence[np.ndarray], intercept: bool, floor: float, iterations: int = IRLS_ITERATIONS
) -> tuple[float, np.ndarray]:
    """IRLS ~ L1 fit ``y ~ c0 + sum c_i s_i`` (``c0 = 0`` without intercept) in float64 on all cells: least squares,
    then ``iterations`` weighted least-squares steps with weights ``1 / max(|residual|, floor)``. The small normal
    equations are solved centred (with intercept); a degenerate system gives its minimum-norm solution. Returns
    ``(c0, slopes)`` in float64 (the caller rounds them)."""
    y = np.asarray(y, F8)
    s = [np.asarray(a, F8) for a in sources]
    k = len(s)
    w = np.ones_like(y)
    c0, c = 0.0, np.zeros(k)
    for _ in range(iterations + 1):
        if intercept:
            sw = w.sum()
            my = float((w * y).sum() / sw)
            ms = [float((w * a).sum() / sw) for a in s]
            yc, sc = y - my, [a - m for a, m in zip(s, ms, strict=True)]
        else:
            yc, sc = y, s
        wsc = [w * a for a in sc]
        G = np.array([[float((wa * b).sum()) for b in sc] for wa in wsc])
        g = np.array([float((wa * yc).sum()) for wa in wsc])
        c = np.linalg.lstsq(G, g, rcond=None)[0] if np.isfinite(G).all() and np.isfinite(g).all() else np.zeros(k)
        c0 = float(my - np.dot(ms, c)) if intercept else 0.0
        res = y - c0
        for ci, a in zip(c, s, strict=True):
            res -= ci * a
        w = 1.0 / np.maximum(np.abs(res), floor)
    return c0, np.asarray(c, F8)


def predict(c0: np.float32, slopes: Sequence[np.float32], sources: Sequence[np.ndarray]) -> np.ndarray:
    """``fl32(c0 + c_1 s_1 + .. + c_k s_k)``, evaluated left to right in float64 (the format's prediction)."""
    n = sources[0].shape[-1]
    p = np.full(n, F8(c0), F8)
    tmp = np.empty(n, F8)
    for c, s in zip(slopes, sources, strict=True):
        np.multiply(s, F8(F4(c)), out=tmp, dtype=F8)
        np.add(p, tmp, out=p, dtype=F8)
    return p.astype(F4)


def residual(x: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """``fl32(x - pred)``."""
    return np.subtract(x, pred, dtype=F4)


def reconstruct(r: np.ndarray, pred: np.ndarray, clip: Clip, out: np.ndarray | None = None) -> np.ndarray:
    """``clip(fl32(r + pred))`` (into ``out`` when given; ``out`` may be ``r``), clipped like the lattice codec:
    ``max(., fl32(lo))`` then ``min(., fl32(hi))``."""
    return apply_clip(np.add(r, pred, out=out, dtype=F4), clip)


def apply_clip(y: np.ndarray, clip: Clip) -> np.ndarray:
    """``y`` clipped in place like the lattice codec clips: ``max(., fl32(lo))`` then ``min(., fl32(hi))``."""
    if clip is not None:
        lo, hi = clip
        if lo is not None:
            np.maximum(y, F4(lo), out=y)
        if hi is not None:
            np.minimum(y, F4(hi), out=y)
    return y


def clip_within(x: np.ndarray, clip: Clip, bound: float) -> bool:
    """True when ``|clip(x) - x| <= bound`` in every cell (the float64 comparison): then ``clip(x')`` is within
    ``bound`` of ``x`` for every ``x'`` within ``bound`` of ``x`` (clipping moves ``x'`` towards the clip range, and
    a value it moves past ``x`` stops at ``clip(x)``). The condition for clipping a bypassed slice on decode."""
    if clip is None:
        return True
    c = apply_clip(np.array(x, dtype=F4), clip)
    return bool((np.abs(c.astype(F8) - x.astype(F8)) <= bound).all())


def _down32(a: np.ndarray) -> np.ndarray:
    """The largest float32 value <= ``a`` (float64)."""
    v = a.astype(F4)
    return np.where(v.astype(F8) > a, np.nextafter(v, F4(-np.inf)), v)


def _up32(a: np.ndarray) -> np.ndarray:
    """The smallest float32 value >= ``a`` (float64)."""
    v = a.astype(F4)
    return np.where(v.astype(F8) < a, np.nextafter(v, F4(np.inf)), v)


def endpoints(r: np.ndarray, bound_next: float) -> tuple[np.ndarray, np.ndarray]:
    """Float32 values ``lo <= r' <= hi`` for every float32 ``r'`` within ``bound_next`` of ``r`` (in real arithmetic,
    or as a float64 comparison of the float32 values as the lattice codec checks it): ``r -+ bound_next``, widened by
    ``2^-50 (|r| + bound_next)`` and rounded outwards to float32."""
    r64 = r.astype(F8)
    delta = _DELTA * (np.abs(r64) + bound_next)
    return _down32(r64 - bound_next - delta), _up32(r64 + bound_next + delta)


def verify(x: np.ndarray, r: np.ndarray, pred: np.ndarray, clip: Clip, bound: float, bound_next: float) -> bool:
    """True when every cell's reconstruction ``clip(fl32(r' + pred))`` is within ``bound`` of ``x`` for every float32
    ``r'`` within ``bound_next`` of ``r``. The reconstruction is monotone non-decreasing in ``r'`` (float32 rounding
    and clip are) and ``|x' - x|`` is largest at an end of a range of ``x'``, so it suffices to check both ends of
    ``endpoints(r, bound_next)``; ``within`` is the float64 comparison the lattice codec uses."""
    x64 = x.astype(F8)
    lo, hi = endpoints(r, bound_next)
    for v in (lo, hi):
        xv = reconstruct(v, pred, clip)
        if not bool((np.abs(xv.astype(F8) - x64) <= bound).all()):
            return False
    return True
