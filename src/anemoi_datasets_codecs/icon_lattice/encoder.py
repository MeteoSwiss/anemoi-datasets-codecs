"""Encoder of one lossy field (mode LATTICE): numpy, plus the C rANS encoder and the C decoder for verification.

Speed matters less here than in the decoder; the fits (most of the time) run numpy's LAPACK least squares without the
GIL and the field threads run in parallel (``native.Lstsq``). Per field:

1. quantise: ``n = rint((x - offset) / step)``, ``step = 2 (bound - e_fp) / k`` with ``e_fp`` the float32 rounding
   of the reconstruction ``offset + step * n`` (verified on every cell, shrunk further if needed); int16 values when
   they fit, else int32.
2. per structure step: robust (IRLS ~ L1) sum-to-one least-squares weights in 8-bit fixed point (or the structure's
   default weights when smaller), then the cross-variable terms (details of already decoded fields; fixed-point
   coefficient per step and a shift per source, dropped where they do not pay).
3. contexts: bucket of the spread of the first <= 4 sources; the table layout (no contexts / per-step thresholds
   with per-(step, bucket) tables / global thresholds with tables shared by all steps) is the cheapest by exact
   coded size plus table bytes plus a per-table penalty (decode cost).
4. rANS (32 lanes, lanes restart per step), escapes for |residual| > 127 in an int16 (or int32) side stream.
5. the payload is decoded by the C decoder and checked: values equal, every cell within the bound (after clip).

Steps 2-5 are the four phases of :func:`encode_quantised`. The optional rdoq path (``options.rdoq``) is ``rdoq.py``.
"""

from __future__ import annotations

import functools
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import native
from .container import LATTICE_HEADER
from .structure import Structure

FIT_SAMPLE = 400_000  # targets used to fit weights / coefficients (deterministic subsample)


@dataclass
class EncodeOptions:
    """Per-codec options relevant to one lossy field (see ``LatticeCodec`` for the full list)."""

    k: float = 1.0
    contexts: int = 8
    fit: str = "ls"  # "ls" | "fixed"
    tables: str = "auto"  # "auto" | "shared" | "per_step" | "none"
    table_penalty: int = 256  # bytes charged per table (decode cost) when choosing the table layout
    prob_bits: int | str = "auto"  # rANS precision: 10 (4 KB tables, L1-resident), 12, or auto (10 unless it costs
    prob_tolerance: float = 0.003  # more than this fraction of the field's coded size)
    recon_bits: int = 3  # residual-domain reconstruction on the grid step / 2^m (0: the plain rule)
    rounding: str = "grid"  # m > 0: "grid" rounds the value to the fine grid, then the residual (step / (1 + 2^-m));
    # "direct" rounds the residual from the exact value (the full step); the decoder is the same. LatticeCodec always
    # passes the configured value (options.rounding, default direct): this default only serves direct callers
    rdoq: bool = False  # also try k = 2 with encoder-chosen values inside the bound; keep the smaller
    rdoq_lambda: float = 1.0
    rdoq_max_bits: float = 2.5  # only fields whose plain coding costs less than this (bits per value)
    # for the lab (``anemoi_datasets_codecs.lab``): called at the end of every successful encode_quantised with the
    # pieces of the payload (see there); it only reads them, the bytes do not depend on it. Instrumentation, never
    # configuration: not serialised (not in get_config, not in a chunk)
    record: Callable[[dict[str, Any]], None] | None = None


@dataclass
class EncodedField:
    payload: bytes
    values: np.ndarray  # int32 decoded values (= n)
    detail: np.ndarray | None  # int32 detail (n - spatial prediction), when requested
    info: dict[str, Any] = field(default_factory=dict)
    decoded: np.ndarray | None = None  # float32 decoded field (for mask terms of later fields)


class EncodeError(ValueError):
    """The field cannot be coded in mode LATTICE (the caller falls back to lossless)."""


# ----------------------------------------------------------------------------------------------- helpers
def _ulp32(x: float) -> float:
    x = abs(float(x))
    if x == 0:
        return 2.0**-149
    return float(np.spacing(np.float32(x)))


def reconstruct(offset: np.float32, step: np.float32, n: np.ndarray, clip: tuple | None) -> np.ndarray:
    """Exactly the decoder's dequantisation: float32 ``offset + step * float32(n)`` (two roundings), then clip."""
    y = np.float32(offset) + np.float32(step) * n.astype(np.float32)
    if clip is not None:
        lo, hi = clip
        if lo is not None:
            y = np.maximum(y, np.float32(lo))
        if hi is not None:
            y = np.minimum(y, np.float32(hi))
    return y


def quantise(
    x: np.ndarray,
    bound: float,
    k: float,
    clip: tuple | None,
    rbits: int = 0,
    shrink: float = 1.0,
    direct: bool = False,
) -> tuple[np.float32, np.float32, np.ndarray]:
    """Offset, step and int64 ``n`` with every reconstructed cell within ``bound`` (raises ``EncodeError``).

    The grid is anchored at zero: ``offset = float32(step * m)`` with ``m`` the integer nearest the mid-range, so
    ``offset + step * (-m)`` is exactly 0 (float32 rounding is symmetric) and exact zeros (precipitation, clouds,
    CAPE, LPI) decode exactly. With ``rbits`` > 0 the returned step is the fine grid ``g``; ``direct``: the residuals
    will be rounded from the exact value (error <= step / 2, ``rounding: direct``), not from ``n`` (error <= step / 2
    + g / 2), so the step is not reduced by ``1 + 2^-rbits``."""
    lo, hi = float(x.min()), float(x.max())
    e_fp = _ulp32(max(abs(lo), abs(hi))) + _ulp32(hi - lo)
    if not bound > 2 * e_fp:
        raise EncodeError(f"bound {bound:.3g} is below the float32 resolution of the field")
    xd = x.astype(np.float64)
    if rbits > 0:  # residual-domain reconstruction: fine grid g = step / 2^rbits, error <= step / 2 + g / 2
        step = np.float32(2.0 * (bound - e_fp) / k / (1.0 if direct else 1.0 + 2.0**-rbits) * shrink)
        g = np.float32(step / np.float32(2**rbits))
        m0 = np.float32(np.rint(0.5 * (lo + hi) / float(g)))
        offset = np.float32(g) * m0 if abs(float(m0)) < 2**24 else np.float32(0.5 * (lo + hi))
        n = np.rint((xd - float(offset)) / float(g))
        if np.abs(n).max() >= 2**29:
            raise EncodeError("quantised values exceed int32")
        return offset, g, n.astype(np.int64)
    step = np.float32(2.0 * (bound - e_fp) / k)
    for _ in range(12):
        if not step > 0:
            break
        m = np.float32(np.rint(0.5 * (lo + hi) / float(step)))
        if abs(float(m)) < 2**24:
            offset = np.float32(step) * m  # float32 product: offset + step * (-m) == 0 exactly
        else:  # zero far outside the field's range: plain mid-range offset
            offset = np.float32(0.5 * (lo + hi))
        n = np.rint((xd - float(offset)) / float(step))
        if np.abs(n).max() >= 2**30:
            raise EncodeError("quantised values exceed int32")
        n = n.astype(np.int64)
        err = np.abs(reconstruct(offset, step, n, clip).astype(np.float64) - xd)
        if err.max() <= bound:
            return offset, step, n
        step = np.float32(float(step) * (1 - 2.0**-12))
    raise EncodeError("could not meet the bound after float32 rounding")


def fit_robust(X: np.ndarray, y: np.ndarray, it: int = 6) -> np.ndarray:
    """IRLS ~ L1 least squares (the residual entropy follows |r|, plain L2 chases the tails).

    Each solve is ``np.linalg.lstsq(X * s[:, None], y * s, rcond=None)`` (``s = 1`` first, then
    ``1 / sqrt(max(|y - X @ w|, 0.5))``), bit for bit: the weighted system is written straight into the buffers of
    :class:`native.Lstsq` (in one pass by ``native.irls_scale``), which calls numpy's LAPACK routine without
    holding the GIL."""
    if X.shape[1] == 0:
        return np.zeros(0)
    m = len(y)
    ls = native.Lstsq(m, X.shape[1])
    b = ls.b[:m]
    ls.a[...] = X
    b[...] = y
    w = ls.solve()
    s = None
    for _ in range(it):
        xw = X @ w
        if not native.irls_scale(y, xw, X, ls.a, b):  # the same expressions in numpy
            s = np.empty(m, np.float64) if s is None else s
            np.subtract(y, xw, out=s)
            np.abs(s, out=s)
            np.maximum(s, 0.5, out=s)
            np.sqrt(s, out=s)
            np.divide(1.0, s, out=s)
            np.multiply(X, s[:, None], out=ls.a)
            np.multiply(y, s, out=b)
        w = ls.solve()
    return w


@functools.lru_cache(maxsize=64)
def _sample(m: int) -> np.ndarray:
    """The fit subsample of ``m`` targets (read-only, cached per size)."""
    s = np.arange(m) if m <= FIT_SAMPLE else np.linspace(0, m - 1, FIT_SAMPLE).astype(np.int64)
    s.flags.writeable = False
    return s


def _varints(v: np.ndarray) -> bytes:
    """LEB128 of non-negative integers (the varint escapes, the table frequencies): 7-bit groups, low first, the
    high bit set on every byte but the last of a value."""
    v = np.asarray(v)
    if len(v) == 0:
        return b""
    if v.min() < 0:
        raise ValueError("LEB128 of a negative integer")
    v = v.astype(np.uint64)
    groups = max(1, (int(v.max()).bit_length() + 6) // 7)  # 7-bit groups of the largest value (<= 10)
    g = v[:, None] >> (7 * np.arange(groups, dtype=np.uint64))  # (n, groups): the value shifted by 7 j
    keep = g != 0  # group j is written when j == 0 or the value has bits at or above 7 j
    keep[:, 0] = True
    byte = (g & 127).astype(np.uint8)
    byte[:, :-1] |= keep[:, 1:].astype(np.uint8) << 7  # continuation bit: another group follows
    return byte[keep].tobytes()


def _zigzag(r: np.ndarray) -> np.ndarray:
    """``2 r`` for ``r >= 0``, ``-2 r - 1`` for ``r < 0`` (``np.where(r >= 0, 2 * r, -2 * r - 1)``, bit for bit in
    the dtype of a signed ``r``: ``-2 r - 1 = ~(2 r)`` in two's complement, wrapping alike)."""
    if r.dtype.kind == "i":
        return (r << 1) ^ (r >> (8 * r.dtype.itemsize - 1))
    return np.where(r >= 0, 2 * r, -2 * r - 1)


def _signed(r: np.ndarray) -> bool:
    return r.dtype.kind == "i" and r.dtype.itemsize >= 2


# index in the histogram of clip(r, -128, 128) + 128 of the residual of each zigzag symbol 0..254 (0, -1, 1, -2, ...)
_ZIG_SRC = np.array([128 + z // 2 if z % 2 == 0 else 127 - z // 2 for z in range(255)], np.intp)


def _symbols(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """uint8 symbols (zigzag, 255 = escape) and the escaped residuals (``|r| < 2^62``: ``min(zigzag(r), 255)`` is
    the zigzag of ``clip(r, -128, 128)``, capped)."""
    if not _signed(r):
        z = _zigzag(r)
        return np.minimum(z, 255).astype(np.uint8), r[z >= 255]
    c = np.empty(r.shape, np.int16)
    np.clip(r, -128, 128, out=c, casting="unsafe")  # clipped in the dtype of r, then stored (exactly) as int16
    z = _zigzag(c)
    np.minimum(z, 255, out=z)
    sym = z.astype(np.uint8)
    return sym, r[sym == 255]


def _entropy_bits(r: np.ndarray) -> float:
    """Order-0 cost estimate of residuals (symbols + 16 bits per escape); ``|r| < 2^62``."""
    if len(r) == 0:
        return 0.0
    if _signed(r):  # the symbol histogram from the histogram of clip(r, -128, 128)
        c = np.clip(r, -128, 128)
        c += 128
        h = np.bincount(c, minlength=257)
        hs = np.empty(256, h.dtype)
        hs[:255] = h[_ZIG_SRC]
        hs[255] = h[0] + h[256]  # the escapes: r <= -128 or r >= 128
    else:
        hs = np.bincount(np.minimum(_zigzag(r), 255), minlength=256)
    cnt = hs.astype(np.float64)
    nz = cnt[cnt > 0]
    p = nz / len(r)
    return float(-(nz * np.log2(p)).sum() + 16 * int(hs[255]))


def _table_bytes(freqs: np.ndarray) -> bytes:
    """A serialised table: the number m of symbols before the last non-zero one, then LEB128 of their frequencies
    and of the escape's."""
    nz = np.flatnonzero(freqs[:255])
    m = int(nz[-1]) + 1 if len(nz) else 0
    return bytes([m]) + _varints(np.concatenate([freqs[:m], freqs[255:256]]))


def _tables_len(freqs: np.ndarray) -> int:
    """``sum(len(_table_bytes(f)) for f in freqs)`` of uint32 tables ``(n_tables, 256)``, without serialising."""
    nv = np.ones(freqs.shape, np.int64)  # LEB128 bytes of every frequency
    for k in (7, 14, 21, 28):
        nv += freqs >= (1 << k)
    nzm = freqs[:, :255] != 0
    m = np.where(nzm.any(1), 255 - np.argmax(nzm[:, ::-1], 1), 0)  # symbols up to the last non-zero one
    upto = np.arange(255) < m[:, None]
    return int(len(freqs) + (nv[:, :255] * upto).sum() + nv[:, 255].sum())


def _coded_bits(counts: np.ndarray, bits: int = 12) -> tuple[float, np.ndarray]:
    """Exact rANS cost (bits, up to the lane flush) of a histogram with its normalised frequencies."""
    f = native.normalize_freqs(counts, 1 << bits)
    nz = counts > 0
    return float((counts[nz] * (bits - np.log2(f[nz]))).sum()), f


def _spread_counts(spread: np.ndarray) -> np.ndarray | None:
    """Histogram of the (non-negative integer) spreads when their range is small enough to count them, else None."""
    if len(spread) == 0 or spread.dtype.kind != "i":
        return None
    hi = int(spread.max())
    if hi > 4 * len(spread) + 65536 or int(spread.min()) < 0:
        return None
    return np.bincount(spread)


def _thresholds(spread: np.ndarray, nb: int, counts: np.ndarray | None = None) -> np.ndarray:
    """The distinct ``np.quantile(spread, i / nb, method="lower")``, ``0 < i < nb``. ``counts``: the histogram of
    ``spread`` (:func:`_spread_counts`), when known."""
    if nb <= 1 or len(spread) == 0:
        return np.zeros(0, np.int64)
    if counts is None:
        counts = _spread_counts(spread)
    if counts is None:
        q = np.quantile(spread, np.arange(1, nb) / nb, method="lower")
        return np.unique(q.astype(np.int64))
    return _thresholds_counted(counts, len(spread), nb)


def _thresholds_counted(counts: np.ndarray, n: int, nb: int) -> np.ndarray:
    """:func:`_thresholds` of ``n`` values from their histogram (a counting sort): ``method="lower"`` takes the
    element ``floor((n - 1) q)`` of the sorted values (numpy's index rule), the smallest ``v`` with more than that
    many values ``<= v``."""
    rank = np.floor((n - 1) * (np.arange(1, nb) / nb)).astype(np.intp)
    q = np.searchsorted(np.cumsum(counts), rank, side="right")
    return np.unique(q.astype(np.int64))


def _bucket(spread: np.ndarray, thr: np.ndarray) -> np.ndarray:
    return np.searchsorted(thr, spread, side="left")  # intp: the number of thresholds < spread


# ---------------------------------------------------------------------------------------------- encoder
@dataclass
class _StepCode:
    sym: np.ndarray
    esc: np.ndarray
    spread: np.ndarray | None
    w: np.ndarray  # int (k,) with w[0] unused
    cc: np.ndarray  # int (n_cross,)
    masked: np.ndarray | None = None  # targets predicted by the mask term (own table)


MASK_SENTINEL = -(2**31)


@dataclass
class MaskTerm:
    """Optional mask term: where the decoded field ``src`` is below ``thr`` the prediction is
    ``c0 + sum coef * A (* B)`` over decoded fields, rounded to the field's grid (``lc_mask_predict``)."""

    src: int
    thr: float
    c0: float
    feats: list[tuple[int, int | None, float]]
    msrc: np.ndarray
    A: list[np.ndarray]
    B: list[np.ndarray | None]

    def arrays(self) -> list[np.ndarray | None]:
        a = [np.ascontiguousarray(x, np.float32) for x in self.A] + [None] * (4 - len(self.A))
        b = [None if x is None else np.ascontiguousarray(x, np.float32) for x in self.B] + [None] * (4 - len(self.B))
        return [np.ascontiguousarray(self.msrc, np.float32), *a, *b]

    def predict(self, offset: float, step: float, vt: int) -> np.ndarray:
        arrs = self.arrays()
        pa, pb = native.ptr_array(arrs[1:5]), native.ptr_array(arrs[5:9])
        coef = np.array([f[2] for f in self.feats] + [0.0] * (4 - len(self.feats)), np.float64)
        pm = np.empty(len(self.msrc), np.int32)
        native.lib().lc_mask_predict(
            len(pm),
            native.ptr(arrs[0]),
            self.thr,
            self.c0,
            len(self.feats),
            pa,
            pb,
            native.ptr(coef),
            offset,
            step,
            vt,
            native.ptr(pm),
        )
        return pm

    def block(self, table: int) -> bytes:
        out = struct.pack("<HBBfd", self.src, len(self.feats), table, self.thr, self.c0)
        for a, b, c in self.feats:
            out += struct.pack("<HHId", a, 0xFFFF if b is None else b, 0, c)
        return out


def encode_field(
    x: np.ndarray,
    bound: float,
    clip: tuple | None,
    structure: Structure,
    opts: EncodeOptions,
    cross: list[tuple[int, np.ndarray]] | None = None,
    want_detail: bool = False,
    mask: MaskTerm | None = None,
) -> EncodedField:
    """Encode one lossy field. ``cross``: ``[(field index in the chunk, int32 detail array), ...]`` (<= 4);
    ``mask``: optional mask term (below-ground predictor). With ``opts.rdoq`` the field is also coded at k = 2 with
    encoder-chosen values (``rdoq.py``) and the smaller payload wins. Raises ``EncodeError`` when the field cannot be
    coded (the caller stores it losslessly)."""
    m = int(opts.recon_bits)
    if m and np.count_nonzero(x == 0) > 0.01 * len(x):  # keep exact zeros (precipitation, clouds, CAPE): plain rule
        m = 0
    direct = bool(m) and opts.rounding == "direct"
    shrink, plain = 1.0, None
    for _ in range(4):  # the fine-grid bound includes float32 rounding: shrink a little if the check fails
        try:
            q1 = quantise(x, bound, opts.k, clip, m, shrink, direct)
            exact = (x.astype(np.float64) - float(q1[0])) / float(q1[1]) if direct else None
            plain = encode_quantised(x, bound, clip, structure, opts, cross, want_detail, q1, mask, m, exact)
            break
        except EncodeError as e:
            if m == 0 or "max error" not in str(e):
                raise
            shrink *= 1 - 2.0**-10
    if plain is None:
        raise EncodeError("could not meet the bound after float32 rounding")
    if not opts.rdoq:
        return plain
    from .rdoq import encode_rdoq  # the optional path (rdoq.py imports this module)

    return encode_rdoq(x, bound, clip, structure, opts, cross, want_detail, mask, plain)


def encode_quantised(
    x, bound, clip, structure, opts, cross, want_detail, quant, mask=None, m=0, exact=None
) -> EncodedField:
    """Code the integers ``n`` of ``quant = (offset, step, n)`` (on the grid ``step``) in four phases: the closed-loop
    step coding (:func:`_code_steps`), the context / table layout and the escapes (:func:`_layout_streams`), the
    payload bytes (:func:`_serialise`) and the verification through the C decoder (:func:`_verify`). ``exact``: the
    unrounded values on that grid (float64; ``rounding: direct``), from which the residuals are then rounded."""
    cross = list(cross or [])[:4]
    offset, step, n = quant
    unit = 1 << m
    lo_v, hi_v = int(n.min()) - unit, int(n.max()) + unit
    vt = 2 if lo_v >= -(2**15) and hi_v < 2**15 else 4
    pm = mask.predict(float(offset), float(step), vt).astype(np.int64) if mask is not None else None
    masked_all = pm != MASK_SENTINEL if pm is not None else None
    steps_code, n, shifts = _code_steps(n, structure, opts, cross, pm, masked_all, m, exact)
    st = _layout_streams(steps_code, structure, opts)
    body = _serialise(quant, vt, m, clip, structure, steps_code, cross, shifts, mask, st)
    out, ival, dout, err = _verify(body, x, n, bound, structure, cross, mask, want_detail)
    info = {
        "vt": vt,
        "et": st.et,
        "n_tables": len(st.freqs),
        "n_ctx": int(max(st.n_ctx)),
        "prob_bits": st.pbits,
        "recon_bits": m,
        "n_esc": st.n_esc,
        "cross": [c[0] for c in cross],
        "cross_steps": int(sum(bool(np.any(sc.cc)) for sc in steps_code)),
        "max_err": err,
        "bits": 8 * len(body) / len(n),
        "masked": int(masked_all.sum()) if masked_all is not None else 0,
    }
    if opts.record is not None:  # for the lab: quant (offset, step, n), the coded steps, the streams, the values
        opts.record(
            {"quant": quant, "m": m, "steps": steps_code, "streams": st, "values": n, "payload": body, "info": info}
        )
    return EncodedField(body, ival, dout, info, out)


def _stencil(W: np.ndarray, D: np.ndarray, dmax: list[int], v0: np.ndarray) -> np.ndarray | None:
    """The spatial prediction ``v0 + ((W @ D + 128) >> 8)``, or None when the int32 accumulator could overflow
    (``max_i sum_j |W_j D_ji| + 256 >= 2^30``; ``dmax[j] = max_i |D_ji|`` gives an upper bound that settles the
    test without the full product unless it is close)."""
    if sum(abs(int(w)) * d for w, d in zip(W.tolist(), dmax, strict=True)) + 256 >= 2**30:
        if (np.abs(W) @ np.abs(D)).max() + 256 >= 2**30:
            return None
    p = W @ D
    p += 128
    p >>= 8
    p += v0
    return p


def _code_steps(
    n, structure, opts, cross, pm, masked_all, m, exact=None
) -> tuple[list[_StepCode], np.ndarray, list[int]]:
    """Closed-loop coding with residual-domain reconstruction: each step predicts from the reconstructed values
    ``rf`` of earlier steps and codes ``q = round((n - pred) / 2^m)``; ``rf = pred + 2^m q``. With ``m = 0`` this is
    lossless on ``n`` (the plain rule). With ``exact`` (the unrounded values, ``rounding: direct``) the residual is
    ``q = rint((exact - pred) / 2^m)`` instead (ties to even), so ``|exact - rf| <= 2^m / 2``; the fits still use
    ``n``. Returns the coded steps, ``rf`` (the values the decoder reproduces) and the cross-term shifts.

    The fit inputs of a step (the sampled stencil differences, cross details and targets) are built once and shared
    by its fits; integer expressions are evaluated in place where that gives the same integers."""
    unit, half = 1 << m, (1 << m) >> 1
    steps_code: list[_StepCode] = []
    nx = len(cross)
    xd = [c[1].astype(np.int64) for c in cross]
    xmax = [max(1, int(np.abs(d).max())) for d in xd]

    def rq(d: np.ndarray) -> np.ndarray:  # round(d / 2^m), halves to even (exact integer arithmetic); consumes d
        # Halves up would bias the reconstruction by +g / 2 on average (one residue in 2^m is an exact tie that
        # always rounds up); ties to even are unbiased. The decoder only needs q, so this is encoder-only.
        if not m:
            return d
        q = d + half
        q >>= m
        odd = q & 1
        d &= unit - 1
        odd &= d == half
        q -= odd
        return q

    def cross_x(ts: np.ndarray) -> np.ndarray:  # the cross details at the sampled targets (float64, (len, nx))
        return np.stack([d[ts] for d in xd]).T.astype(np.float64)

    # cross-term shifts from an open-loop estimate of the coefficients (fixed before the closed loop)
    shifts = [0] * nx
    if nx:
        bmax = np.full(nx, 1e-12)
        for st in structure.steps:
            if st.kind == "delta" or st.n <= 3 * (nx + st.k):
                continue
            s = _sample(st.n)[:: max(1, st.n // 100_000)]
            V = n[st.src_idx[:, s]]
            psp = V[0] + ((np.round(np.asarray(st.w0[1:]) * 256).astype(np.int64) @ (V[1:] - V[0]) + 128) >> 8)
            ts = st.t_idx[s]
            bmax = np.maximum(bmax, np.abs(fit_robust(cross_x(ts), (n[ts] - psp).astype(np.float64))) * 2)
        shifts = [int(np.clip(np.floor(np.log2(2.0**30 / (bmax[c] * xmax[c] * 1.01))), 0, 24)) for c in range(nx)]
    lim = ([-(2**30 // v) for v in xmax], [2**30 // v for v in xmax])

    def cross_term(coef: np.ndarray, xt: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        cc = np.clip(np.round(coef * np.exp2(shifts)).astype(np.int64), *lim)
        term = None
        for c in range(nx):  # sum of (cc[c] * xd[c][t] + half_c) >> shift_c
            a = xt[c] * cc[c]
            a += (1 << shifts[c]) >> 1
            a >>= shifts[c]
            if term is None:
                term = a
            else:
                term += a
        return term, cc

    rf = np.zeros(len(n), np.int64)  # reconstructed values (what the decoder will have)
    for st in structure.steps:
        t = st.t_idx
        y = n[t]
        if st.kind == "delta":
            q = np.empty(len(y), np.int64)
            prev = 0
            ex = exact[t].tolist() if exact is not None else None
            for i, (ti, yi) in enumerate(zip(t.tolist(), y.tolist(), strict=True)):  # serial chain (~1-2k cells)
                if ex is not None:
                    qi = round((ex[i] - prev) / unit)  # Python round: ties to even
                else:
                    d = yi - prev
                    qi = (d + half) >> m
                    if m and (d & (unit - 1)) == half and (qi & 1):
                        qi -= 1  # ties to even, as in rq
                q[i] = qi
                prev += qi * unit
                rf[ti] = prev
            sym, esc = _symbols(q)
            steps_code.append(_StepCode(sym, esc, None, np.zeros(1, np.int64), np.zeros(nx, np.int64)))
            continue
        V = rf[st.src_idx]  # (k, m) reconstructed sources
        v0 = V[0]
        D = V[1:] - v0
        msk = masked_all[t] if masked_all is not None else None
        has_mask = msk is not None and bool(msk.any())
        free = np.flatnonzero(~msk) if has_mask else None  # None: every target is free
        n_free = len(free) if free is not None else len(t)
        if free is None:
            s = _sample(n_free)
        else:
            s = free[_sample(n_free)] if n_free else free
        fit_w = opts.fit == "ls" and st.k > 1 and n_free > 3 * st.k
        fit_x = bool(nx) and n_free > 3 * (nx + st.k)
        if fit_w or fit_x:
            ts, ys = t[s], y[s]
        if fit_w:
            Ds = D[:, s].T.astype(np.float64)
            y0s = (ys - v0[s]).astype(np.float64)
        if fit_x:
            Xs = cross_x(ts)
            xt = [d[t] for d in xd]
        Ws = [np.round(np.asarray(st.w0[1:]) * 256).astype(np.int64)]
        if fit_w:
            wf = fit_robust(Ds, y0s)
            Ws.append(np.clip(np.round(wf * 256), -(2**14), 2**14).astype(np.int64))
        dmax = [max(-int(d.min()), int(d.max())) for d in D] if st.k > 1 and len(t) else []

        options = []  # (W, cc, psp, term)
        for W in Ws:
            psp = _stencil(W, D, dmax, v0) if st.k > 1 else v0
            if psp is None:
                continue
            options.append((W, np.zeros(nx, np.int64), psp, None))
            if fit_x:
                beta = fit_robust(Xs, (ys - psp[s]).astype(np.float64))
                term, cc = cross_term(beta, xt)
                if np.any(cc):
                    options.append((W, cc, psp, term))
        if fit_x and fit_w:  # joint stencil + cross fit
            wb = fit_robust(np.concatenate([Ds, Xs], 1), y0s)
            Wj = np.clip(np.round(wb[: st.k - 1] * 256), -(2**14), 2**14).astype(np.int64)
            psp = _stencil(Wj, D, dmax, v0)
            if psp is not None:
                term, cc = cross_term(wb[st.k - 1 :], xt)
                if np.any(cc):
                    options.append((Wj, cc, psp, term))
        if not options:
            raise EncodeError("stencil accumulator would overflow int32")
        best = None
        pm_t = pm[t] if has_mask else None
        yx = exact[t] if exact is not None else None
        for W, cc, psp, term in options:
            if has_mask:
                pred = np.where(msk, pm_t, psp if term is None else psp + term)
                d = (y if yx is None else yx) - pred
            else:  # y - (psp + term), the prediction itself is only formed for the best option
                pred = None
                d = (y if yx is None else yx) - psp
                if term is not None:
                    d -= term
            if yx is None:
                q = rq(d)
            else:
                d /= unit
                q = np.rint(d).astype(np.int64)
            cost = _entropy_bits(q[free] if free is not None else q)
            if best is None or cost < best[0]:
                best = (cost, W, cc, psp, term, pred, q)
        _, W, cc, psp, term, pred, q = best
        r = q * unit
        if pred is not None:
            r += pred
        else:
            r += psp
            if term is not None:
                r += term
        rf[t] = r
        spread = None
        if opts.contexts > 1:
            kk = min(st.k, 4)
            spread = V[:kk].max(0) - V[:kk].min(0)
        sym, esc = _symbols(q)
        steps_code.append(_StepCode(sym, esc, spread, np.concatenate([[0], W]), cc, msk if has_mask else None))
    # the decoder reproduces rf (|rf - n| <= 2^m / 2, so the value width chosen from n holds)
    return steps_code, rf, shifts


@dataclass
class _Streams:
    """The entropy-coded part of a payload: context / table layout, rANS streams, escapes."""

    pbits: int  # rANS precision
    n_ctx: list[int]  # context buckets per step
    thr: list[np.ndarray]  # bucket thresholds per step
    tmap: list[list[int]]  # table id per (step, bucket)
    freqs: np.ndarray  # (n_tables, 256)
    tables_blob: bytes
    mask_table: int
    ranges: list[list[tuple[int, int]]]  # per step and range: (rANS bytes, escape units)
    rans: bytes
    et: int  # escape type: 1 varint, 2 int16, 4 int32
    escapes: bytes
    n_esc: int  # escaped residuals
    n_esc_units: int  # escape slice length in units of et


def _layout_streams(steps_code: list[_StepCode], structure: Structure, opts: EncodeOptions) -> _Streams:
    """Choose the table layout (and the rANS precision), the escape type, and code the rANS streams: one serial
    stream through all single-range steps (lanes restart per step, no flush), then one stream per range of the
    multi-range steps (decoded in parallel); escape counts per (step, range)."""
    layout = _choose_tables(steps_code, opts, 12)
    pbits = 12
    if opts.prob_bits == 10 or opts.prob_bits == "auto":
        small = _choose_tables(steps_code, opts, 10, fixed=layout)
        if opts.prob_bits == 10 or small[0] <= layout[0] * (1 + opts.prob_tolerance):
            layout, pbits = small, 10
    _, n_ctx, thr, tmap, tid_per_step, freqs, tables_blob, mask_table, _ = layout
    esc_all = np.concatenate([sc.esc for sc in steps_code]).astype(np.int64)
    et = 2 if (len(esc_all) == 0 or (esc_all.min() >= -(2**15) and esc_all.max() < 2**15)) else 4
    zz = _zigzag(esc_all) - 255  # >= 0 for every escaped residual
    nvar = np.ones(len(zz), np.int64)
    for k in (7, 14, 21, 28):
        nvar += zz >= (1 << k)
    if len(esc_all) and int(nvar.sum()) < et * len(esc_all):
        et = 1  # varints: most escapes are just beyond +-127 and take one byte
    streams, ranges, ser_sym, ser_tid, ser_seg = [], [], [], [], [0]
    for s, sc in enumerate(steps_code):
        rr = []
        rlist = structure.step_ranges(s)
        for lo, hi in rlist:
            sym = sc.sym[lo:hi]
            tid = np.asarray(tid_per_step[s][lo:hi], np.uint8)
            if len(rlist) == 1:
                ser_sym.append(sym)
                ser_tid.append(tid)
                ser_seg.append(ser_seg[-1] + len(sym))
                rr.append((0, int(np.count_nonzero(sym == 255))))
                continue
            stream = native.rans_encode(sym, tid, np.array([0, hi - lo], np.uint64), freqs, pbits)
            streams.append(stream)
            rr.append((len(stream), int(np.count_nonzero(sym == 255))))
        ranges.append(rr)
    serial = native.rans_encode(
        np.concatenate(ser_sym), np.concatenate(ser_tid), np.array(ser_seg, np.uint64), freqs, pbits
    )
    rans = serial + b"".join(streams)
    if et == 1:  # per-range escape slices in bytes
        k = 0
        for rr in ranges:
            for i, (nbytes, cnt) in enumerate(rr):
                rr[i] = (nbytes, int(nvar[k : k + cnt].sum()))
                k += cnt
        n_esc_units = int(nvar.sum())
    else:
        n_esc_units = len(esc_all)
    escapes = _varints(zz) if et == 1 else esc_all.astype("<i2" if et == 2 else "<i4").tobytes()
    return _Streams(
        pbits, n_ctx, thr, tmap, freqs, tables_blob, mask_table, ranges, rans, et, escapes, len(esc_all), n_esc_units
    )


def _serialise(quant, vt, m, clip, structure, steps_code, cross, shifts, mask, st: _Streams) -> bytes:
    """The LATTICE payload (docs/codecs/icon_lattice.md "LATTICE payload")."""
    offset, step, _ = quant
    nx = len(cross)
    steps_blob = bytearray()
    for s, (step_, sc) in enumerate(zip(structure.steps, steps_code, strict=True)):
        steps_blob.append(st.n_ctx[s])
        steps_blob += bytes(int(v) for v in st.tmap[s])
        steps_blob += np.asarray(st.thr[s], "<i4").tobytes()
        if step_.kind == "gather":
            steps_blob += np.asarray(sc.w[1:], "<i2").tobytes()
            steps_blob += np.asarray(sc.cc, "<i4").tobytes()
        for nbytes, nesc in st.ranges[s]:
            steps_blob += struct.pack("<II", nbytes, nesc)
    flags = 0
    lo = hi = 0.0
    if clip is not None:
        if clip[0] is not None:
            flags |= 1
            lo = clip[0]
        if clip[1] is not None:
            flags |= 2
            hi = clip[1]
    if mask is not None:
        flags |= 4
    src_ids = [c[0] for c in cross] + [0xFFFF] * (4 - nx)
    head = struct.pack("<BBBBBBBB", vt, st.et, len(structure.steps), len(st.freqs), nx, flags, st.pbits, m)
    head += struct.pack("<ffff", offset, step, lo, hi)
    head += struct.pack("<IIII", st.n_esc_units, len(st.rans), len(st.tables_blob), len(steps_blob))
    head += struct.pack("<4H", *src_ids) + struct.pack("<4B", *(shifts + [0] * (4 - nx))) + b"\0" * 4
    assert len(head) == LATTICE_HEADER
    body = head + (mask.block(st.mask_table) if mask is not None else b"") + bytes(steps_blob) + st.tables_blob
    body += b"\0" * (-len(body) % 4)
    body += st.rans + st.escapes
    body += b"\0" * (-len(body) % 4)
    return body


def _verify(body, x, n, bound, structure, cross, mask, want_detail):
    """Decode the payload with the C decoder: it must reproduce ``n`` and every cell must be within the bound.
    Returns the decoded floats, the decoded integers, the detail (when wanted) and the max error."""
    S = structure.c_struct()
    out = np.empty(len(n), np.float32)
    ival = np.empty(len(n), np.int32)
    dout = np.empty(len(n), np.int32) if want_detail else None
    srcs = [np.ascontiguousarray(c[1], np.int32) for c in cross]
    parr = native.ptr_array(srcs)
    buf = np.frombuffer(body, np.uint8)
    marr = mask.arrays() if mask is not None else []
    mptr = native.ptr_array(marr) if mask is not None else None
    rc = native.lib().lc_decode_field(
        native.ptr(buf), len(body), S, parr, mptr, native.ptr(out), native.ptr(ival), native.ptr(dout)
    )
    if rc != 0:
        raise EncodeError(f"verification decode failed ({native.ERRORS.get(rc, rc)})")
    if not np.array_equal(ival, n):
        raise EncodeError("verification decode does not reproduce the quantised values")
    err = float(np.abs(out.astype(np.float64) - x).max())
    if not err <= bound:
        raise EncodeError(f"verification: max error {err:.6g} > bound {bound:.6g}")
    return out, ival, dout, err


class _StepTables:
    """The symbols of one step for the table layout, given every threshold set a layout may use for it
    (``thrs``): one bincount of (rank of the spread among the union ``u`` of those thresholds, masked, symbol)
    gives the histograms per bucket of any of them (a bucket, the number of thresholds below the spread, is a
    function of that rank), the same counts as one masked bincount per bucket."""

    def __init__(self, sc: _StepCode, thrs: list[np.ndarray]):
        self.sc = sc
        self.u = np.unique(np.concatenate([np.zeros(0, np.int64), *thrs])) if thrs else np.zeros(0, np.int64)
        idx = _bucket(sc.spread, self.u) if len(self.u) else np.zeros(len(sc.sym), np.intp)  # rank: #u < spread
        self.stride = 2 if sc.masked is not None else 1
        if sc.masked is not None:
            idx <<= 1
            idx |= sc.masked
        idx <<= 8
        idx |= sc.sym
        self.idx = idx  # (rank * stride + masked) * 256 + symbol
        h = np.bincount(idx, minlength=(len(self.u) + 1) * self.stride * 256).reshape(len(self.u) + 1, -1, 256)
        self.free = h[:, 0]  # (ranks, 256)
        self.masked_hist = h[:, 1].sum(0) if sc.masked is not None else None

    def _rank_bucket(self, th: np.ndarray) -> np.ndarray:
        # rank 0: spread <= u[0] <= every threshold; rank k: u[k - 1] < spread <= u[k], no threshold in between
        return np.concatenate([[0], np.searchsorted(th, self.u, side="right")]).astype(np.intp)

    def hists(self, th: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Histograms of the free targets per bucket ``(len(th) + 1, 256)`` and of the masked targets (or None)."""
        rb = self._rank_bucket(th)
        h = np.zeros((len(th) + 1, 256), self.free.dtype)
        for k, b in enumerate(rb.tolist()):  # <= 15 ranks
            h[b] += self.free[k]
        return h, self.masked_hist

    def tids(self, th: np.ndarray, tmap: list[int], mtab: int) -> np.ndarray:
        """uint8 table id of every symbol: ``tmap`` of its bucket, ``mtab`` for the masked targets."""
        lut = np.asarray(tmap, np.uint8)[self._rank_bucket(th)]  # per rank
        if self.stride == 2:
            lut = np.stack([lut, np.full(len(lut), mtab, np.uint8)], 1).ravel()  # per rank and masked
        return np.repeat(lut, 256)[self.idx]


def _choose_tables(steps_code: list[_StepCode], opts: EncodeOptions, bits: int = 12, fixed=None):
    """Context buckets and table assignment. Returns (cost in bytes, n_ctx per step, thresholds per step, tmap per
    step, table id per symbol per step (uint8), freqs (n_tables, 256), serialised tables, mask table, histograms
    (n_tables, 256)). With ``fixed`` (a previous result) the same layout is re-costed at precision ``bits``."""
    if fixed is not None:
        return _cost_layout(steps_code, opts, bits, *fixed[1:5], fixed[7], fixed[8])
    nb = max(1, min(8, int(opts.contexts)))
    modes = {"none": ["none"], "shared": ["shared"], "per_step": ["per_step"]}.get(opts.tables)
    if modes is None:
        modes = ["none", "per_step", "shared"]
    if nb == 1:
        modes = ["none"]
    # thresholds: per step, and global over the spreads of all gather steps (from the summed histograms when every
    # step's spreads were counted, else from their concatenation)
    step_thr: list[np.ndarray | None] = [None] * len(steps_code)
    gthr = np.zeros(0, np.int64)
    if "per_step" in modes or "shared" in modes:
        counts = [_spread_counts(sc.spread) if sc.spread is not None else None for sc in steps_code]
        if "per_step" in modes:
            step_thr = [
                _thresholds(sc.spread, nb, c) if sc.spread is not None else None
                for sc, c in zip(steps_code, counts, strict=True)
            ]
        gather = [(sc.spread, c) for sc, c in zip(steps_code, counts, strict=True) if sc.spread is not None]
        if "shared" in modes and gather:
            if all(c is not None for _, c in gather):
                total = np.zeros(max(len(c) for _, c in gather), np.int64)
                for _, c in gather:
                    total[: len(c)] += c
                gthr = _thresholds_counted(total, sum(len(sp) for sp, _ in gather), nb)
            else:
                gthr = _thresholds(np.concatenate([sp for sp, _ in gather]), nb)
    layouts = []  # per mode: n_ctx, thresholds and table keys per step
    for mode in modes:
        n_ctx, thr, keys = [], [], []
        for s, sc in enumerate(steps_code):
            if sc.spread is None or mode == "none":
                th = np.zeros(0, np.int64)
            else:
                th = gthr if mode == "shared" else step_thr[s]
            n_ctx.append(len(th) + 1)
            thr.append(th)
            if mode == "shared" and sc.spread is not None:
                keys.append([("g", b) for b in range(len(th) + 1)])
            else:
                keys.append([("s", s, b) for b in range(len(th) + 1)])
        layouts.append((n_ctx, thr, keys))
    steps = [_StepTables(sc, [lay[1][s] for lay in layouts]) for s, sc in enumerate(steps_code)]
    candidates = []
    for n_ctx, thr, keys in layouts:
        hist: dict[tuple, np.ndarray] = {}
        for s, st in enumerate(steps):
            h, hm = st.hists(thr[s])
            for b, key in enumerate(keys[s]):
                hist[key] = hist[key] + h[b] if key in hist else h[b]
            if hm is not None:
                hist[("m",)] = hist[("m",)] + hm if ("m",) in hist else hm
        live = [key for key, h in hist.items() if h.sum() > 0]
        if len(live) > 255:
            continue
        ids = {key: i for i, key in enumerate(live)}
        tmap = [[ids.get(key, 0) for key in keys[s]] for s in range(len(steps_code))]
        mtab = ids.get(("m",), 0)
        coded, freqs = 0.0, []
        for key in live:
            b, f = _coded_bits(hist[key].astype(np.uint32), bits)
            coded += b
            freqs.append(f)
        freqs = np.stack(freqs)
        cost = coded / 8 + _tables_len(freqs) + opts.table_penalty * len(live) + 4 * sum(len(t) for t in thr)
        hists = np.stack([hist[key] for key in live])
        candidates.append((cost, n_ctx, thr, tmap, None, freqs, None, mtab, hists))
    if not candidates:
        raise EncodeError("too many entropy tables")
    best = min(candidates, key=lambda c: c[0])
    tid_steps = [st.tids(th, tm, best[7]) for st, th, tm in zip(steps, best[2], best[3], strict=True)]
    blob = b"".join(_table_bytes(f) for f in best[5])
    return (*best[:4], tid_steps, best[5], blob, *best[7:])


def _cost_layout(steps_code, opts, bits, n_ctx, thr, tmap, tid_steps, mtab, hists=None):
    """Tables and cost of a given context / table layout at precision ``bits``; ``hists``: the histograms of its
    tables, when known (else counted from the table ids)."""
    if hists is None:
        n_tab = 1 + max(int(t.max()) if len(t) else 0 for t in tid_steps)
        hists = np.zeros((n_tab, 256), np.int64)
        for sc, tid in zip(steps_code, tid_steps, strict=True):
            idx = tid.astype(np.intp) << 8
            idx |= sc.sym
            hists += np.bincount(idx, minlength=256 * n_tab).reshape(n_tab, 256)
    n_tab = len(hists)
    coded, freqs, blob = 0.0, [], bytearray()
    for h in hists:
        b, f = _coded_bits(h.astype(np.uint32), bits)
        coded += b
        freqs.append(f)
        blob += _table_bytes(f)
    cost = coded / 8 + len(blob) + opts.table_penalty * n_tab + 4 * sum(len(t) for t in thr)
    return (cost, n_ctx, thr, tmap, tid_steps, np.stack(freqs), bytes(blob), mtab, hists)
