"""Shared machinery of the oracles: quantisation, the open-loop spatial prediction along a structure, costs in bits,
context buckets, field selection and the input side (store, configuration, date).

Everything works on field indices and integer arrays; names only label the output. The codec's engine is imported
(``encoder.quantise``, ``encoder.fit_robust``, ``encoder.FIT_SAMPLE``), never changed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...icon_lattice.encoder import FIT_SAMPLE, EncodeError, fit_robust, quantise, reconstruct
from ...icon_lattice.structure import Structure
from .. import data as D
from .. import entropy as E

N_SYM = E.ESCAPE + 1


# ------------------------------------------------------------------------------------------------- costs
def hist_bits(counts: np.ndarray, miller_madow: bool = False) -> np.ndarray:
    """Entropy in bits (total) of every row of a histogram ``(rows, symbols)``: ``N log2 N - sum n log2 n``. With
    ``miller_madow`` plus the Miller-Madow bias correction ``(m - 1) / (2 ln 2)`` bits per row of ``m`` non-empty
    symbols (the plug-in entropy of a sample underestimates the more, the more contexts share the sample)."""
    c = np.asarray(counts, np.float64)
    c = c.reshape(-1, c.shape[-1])
    tot = c.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cl = np.where(c > 0, c * np.log2(np.where(c > 0, c, 1.0)), 0.0).sum(1)
        tl = np.where(tot > 0, tot * np.log2(np.where(tot > 0, tot, 1.0)), 0.0)
    h = np.maximum(tl - cl, 0.0)
    if miller_madow:
        h += np.maximum((c > 0).sum(1) - 1, 0) / (2 * np.log(2))
    return h


def keyed_counts(sym: np.ndarray, key: np.ndarray | None, n_keys: int | None = None) -> np.ndarray:
    """Histogram ``(n_keys, 256)`` of symbols per integer key (one row without keys)."""
    if key is None:
        return np.bincount(sym, minlength=N_SYM)[None, :]
    k = np.asarray(key, np.int64)
    n_keys = int(k.max()) + 1 if n_keys is None and len(k) else int(n_keys or 1)
    return np.bincount(k * N_SYM + sym, minlength=n_keys * N_SYM).reshape(n_keys, N_SYM)


def cost_bits(r: np.ndarray, ctx: np.ndarray | None = None) -> float:
    """``lab.entropy.residual_cost_bits(r, ctx)`` (symbol entropy conditional on the context plus 16 bits per escape)
    by one bincount instead of a sort; the same number up to float rounding."""
    r = np.asarray(r)
    if r.size == 0:
        return 0.0
    sym = E.symbols(r)
    return float(hist_bits(keyed_counts(sym, ctx)).sum()) + E.escape_bits(r)


def n_escapes(r: np.ndarray) -> int:
    return int(np.count_nonzero(np.abs(np.asarray(r, np.int64)) > 127))


# ------------------------------------------------------------------------------------------------ contexts
def quantile_buckets(v: np.ndarray, nb: int) -> tuple[np.ndarray, int]:
    """The codec's context rule on any non-negative integer feature: thresholds the distinct
    ``np.quantile(v, i / nb, method="lower")`` (``0 < i < nb``), bucket = the number of thresholds below ``v``.
    Returns ``(bucket per value (intp), number of buckets)``."""
    if nb <= 1 or len(v) == 0:
        return np.zeros(len(v), np.intp), 1
    thr = np.unique(np.quantile(v, np.arange(1, nb) / nb, method="lower").astype(np.int64))
    return np.searchsorted(thr, v, side="left"), len(thr) + 1


def log_bucket(a: np.ndarray, nb: int = 8) -> np.ndarray:
    """``min(bit_length(|a|), nb - 1)``: 0, 1, 2-3, 4-7, ... (a magnitude context)."""
    a = np.abs(np.asarray(a, np.int64))
    b = np.zeros(len(a), np.int64)
    x = a.copy()
    while True:
        nz = x > 0
        if not nz.any():
            break
        b += nz
        x >>= 1
    return np.minimum(b, nb - 1)


def signed_bucket(a: np.ndarray) -> np.ndarray:
    """7 signed magnitude buckets of a residual, 0..6: ``3 + sign(a) min(bit_length(|a|), 3)`` (0, +-1, +-2..3,
    +-4 and beyond)."""
    a = np.asarray(a, np.int64)
    return 3 + np.sign(a) * log_bucket(a, 4)


# ---------------------------------------------------------------------------------------------- prediction
def fit_sample(m: int) -> np.ndarray:
    """The encoder's deterministic fit subsample of ``m`` targets (``FIT_SAMPLE`` evenly spaced)."""
    return np.arange(m) if m <= FIT_SAMPLE else np.linspace(0, m - 1, FIT_SAMPLE).astype(np.int64)


def stencil(W: np.ndarray, V: np.ndarray) -> np.ndarray:
    """The codec's spatial prediction ``v0 + ((sum_j W_j (v_j - v0) + 128) >> 8)`` from sources ``V`` ``(k, m)``
    and int weights ``W`` ``(k - 1,)`` (int64 arithmetic: the encoder's int32 overflow check is not applied)."""
    v0 = V[0]
    if V.shape[0] == 1:
        return v0.copy()
    p = np.asarray(W, np.int64) @ (V[1:] - v0)
    p += 128
    p >>= 8
    p += v0
    return p


@dataclass
class StepResult:
    """One step of :func:`spatial`: the residuals ``r`` of the codec's choice (int64, target order), the context
    bucket ``ctx`` (None for the delta step), the choice (``delta``, ``w0`` or ``fit``), the costs in bits (with the
    context) of the default and the fitted weights and of the choice, and the prediction when kept."""

    choice: str
    r: np.ndarray
    ctx: np.ndarray | None
    nctx: int
    bits: float
    bits_w0: float
    bits_fit: float
    weights: list[int] = field(default_factory=list)
    pred: np.ndarray | None = None


@dataclass
class Spatial:
    """Open-loop spatial prediction of one integer field along a structure (:func:`spatial`)."""

    steps: list[StepResult]

    @property
    def bits(self) -> float:
        return float(sum(s.bits for s in self.steps))

    @property
    def bits_w0(self) -> float:
        return float(sum(s.bits_w0 for s in self.steps))

    @property
    def bits_fit(self) -> float:
        return float(sum(s.bits_fit for s in self.steps))

    def cells(self, structure: Structure, what: str = "r") -> np.ndarray:
        """``r`` (or ``ctx``, 0 for the delta step) of every cell in native order."""
        out = np.zeros(structure.n_cells, np.int64)
        for st, sr in zip(structure.steps, self.steps, strict=True):
            v = sr.r if what == "r" else (sr.ctx if sr.ctx is not None else 0)
            out[st.t_idx] = v
        return out

    def ctx_key(self, extra: list[np.ndarray] | None = None) -> tuple[np.ndarray, int]:
        """Decode-order key (step, spread bucket) of every target, and the number of keys (``nb`` per step)."""
        nb = max(s.nctx for s in self.steps)
        key = np.concatenate(
            [i * nb + (s.ctx if s.ctx is not None else np.zeros(len(s.r), np.intp)) for i, s in enumerate(self.steps)]
        )
        return key, len(self.steps) * nb


def fits_weights(codec: Any) -> bool:
    """Whether the codec fits sum-to-one weights per step (``options.fit`` ``ls``) or keeps the defaults (``fixed``)."""
    return codec.options["fit"] == "ls"


def spatial(n: np.ndarray, structure: Structure, nb: int = 8, fit: bool = True, keep_pred: bool = False) -> Spatial:
    """The codec's spatial prediction of the integers ``n`` (native order) along ``structure``, open loop.

    Per step as the encoder does without cross and mask terms: the delta step codes the differences of its targets
    in order (from 0); a gather step predicts each target from its sources with the default weights ``w0`` and with
    the robust sum-to-one least-squares weights fitted on the encoder's subsample (``encoder.fit_robust`` on the
    differences to the nearest source, rounded to 8 fractional bits), and keeps the one with the smaller order-0
    cost (the encoder's rule); ``fit`` False (``options.fit`` ``fixed``, :func:`fits_weights`): the default weights
    only. Context of a target: the spread ``max - min`` of its first ``min(k, 4)`` sources, bucketed by the per-step
    quantile thresholds into ``nb`` buckets (the codec's ``per_step`` table layout).

    Open loop: the sources are the quantised values themselves. With the plain rule (``recon_bits`` 0) the decoder
    reconstructs exactly these values, so the residuals are the ones the codec codes; with residual-domain
    reconstruction they are not. Not modelled: table bytes, the ``none`` / ``shared`` table layouts, range flushes,
    the rANS precision."""
    out = []
    n = np.asarray(n, np.int64)
    for st in structure.steps:
        y = n[st.t_idx]
        if st.kind == "delta":
            r = np.diff(y, prepend=0)
            b = cost_bits(r)
            out.append(StepResult("delta", r, None, 1, b, b, b, pred=(y - r) if keep_pred else None))
            continue
        V = n[st.src_idx]
        W0 = np.round(np.asarray(st.w0[1:]) * 256).astype(np.int64)
        cands = [("w0", W0)]
        if fit and st.k > 1 and st.n > 3 * st.k:
            s = fit_sample(st.n)
            wf = fit_robust((V[1:, s] - V[0, s]).T.astype(np.float64), (y[s] - V[0, s]).astype(np.float64))
            cands.append(("fit", np.clip(np.round(wf * 256), -(2**14), 2**14).astype(np.int64)))
        kk = min(st.k, 4)
        ctx, nctx = quantile_buckets(V[:kk].max(0) - V[:kk].min(0), nb)
        res = []
        for name, W in cands:
            p = stencil(W, V)
            r = y - p
            res.append((cost_bits(r), name, W, r, p))  # order-0 cost: the encoder's choice rule
        best = min(res, key=lambda t: t[0])  # first wins ties (w0, as in the encoder)
        bits = {name: cost_bits(r, ctx) for _, name, _, r, _ in res}
        out.append(
            StepResult(
                best[1],
                best[3],
                ctx,
                nctx,
                bits[best[1]],
                bits["w0"],
                bits.get("fit", bits["w0"]),
                [int(w) for w in best[2]],
                best[4] if keep_pred else None,
            )
        )
    return Spatial(out)


def first_source(structure: Structure) -> np.ndarray:
    """The nearest source of every cell (native order): ``src[0]`` of the step that decodes it; for the delta step the
    previous target (the first one is its own)."""
    out = np.empty(structure.n_cells, np.int64)
    for st in structure.steps:
        if st.kind == "delta":
            t = st.t_idx
            out[t] = np.concatenate([t[:1], t[:-1]])
        else:
            out[st.t_idx] = st.src_idx[0]
    return out


# --------------------------------------------------------------------------------------------- fields
@dataclass
class FieldSpec:
    """One lossy field of a date: index within the date, label, bound and clip."""

    index: int
    label: str
    bound: float
    clip: tuple | None


def lossy_specs(codec: Any) -> list[FieldSpec]:
    labels = D.field_labels(codec)
    return [FieldSpec(i, labels[i], b, c) for i, _, b, c in D.lossy_fields(codec)]


def select_fields(codec: Any, spec: str | None) -> list[FieldSpec]:
    """The lossy fields named in ``spec`` (comma separated: a label, a field index, or a regular expression matched
    in full against the labels), in field order; every lossy field without ``spec``."""
    fields = lossy_specs(codec)
    if not spec:
        return fields
    keep: set[int] = set()
    for tok in (t.strip() for t in spec.split(",")):
        if not tok:
            continue
        hit = [f.index for f in fields if f.label == tok]
        if not hit and tok.isdigit():
            hit = [f.index for f in fields if f.index == int(tok)]
        if not hit:
            rx = re.compile(tok)
            hit = [f.index for f in fields if rx.fullmatch(f.label)]
        if not hit:
            raise SystemExit(f"--fields: {tok!r} matches no lossy field")
        keep.update(hit)
    return [f for f in fields if f.index in keep]


def quantised(x: np.ndarray, f: FieldSpec, k: float) -> tuple[float, float, np.ndarray]:
    """``encoder.quantise`` with the plain rule (``recon_bits`` 0): ``(offset, step, int64 n)``."""
    off, step, n = quantise(np.asarray(x, np.float32), f.bound, k, f.clip, 0)
    return float(off), float(step), n


def requantised(x_other: np.ndarray, f: FieldSpec, k: float, offset: float, step: float) -> np.ndarray:
    """Another date of the same field as its decoder would see it (quantised on its own grid by the plain rule and
    reconstructed), expressed on the grid ``(offset, step)``: ``rint((xhat - offset) / step)``."""
    try:
        o2, s2, n2 = quantise(np.asarray(x_other, np.float32), f.bound, k, f.clip, 0)
        xhat = reconstruct(o2, s2, n2, f.clip).astype(np.float64)
    except EncodeError:  # the other date cannot be coded at this bound: its exact values
        xhat = np.asarray(x_other, np.float64)
    return np.rint((xhat - offset) / step).astype(np.int64)


# ---------------------------------------------------------------------------------------------- input side
@dataclass
class Inputs:
    """What an oracle runs on: the codec of the configuration, the selected lossy fields and their rows of a date."""

    codec: Any
    fields: list[FieldSpec]
    x: np.ndarray  # float32 (len(fields), cells)
    date: int
    date_value: str | None = None
    store: str | None = None
    config: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def k(self) -> float:
        return float(self.codec.options["k"])

    @property
    def nb(self) -> int:
        return max(1, min(8, int(self.codec.options["contexts"])))

    @property
    def fit(self) -> bool:
        return fits_weights(self.codec)


def drop_uncodable(inp: Inputs) -> Inputs:
    """Drop (in place) the fields that the plain rule cannot quantise at their bound (the codec stores them
    losslessly) and constant fields (mode CONSTANT); their labels go to ``inp.meta["skipped"]``. Idempotent."""
    if inp.meta.get("checked"):
        return inp
    keep, skipped = [], dict(inp.meta.get("skipped", {}))
    for i, f in enumerate(inp.fields):
        x = inp.x[i]
        if x.min() == x.max():
            skipped[f.label] = "constant"
            continue
        try:
            quantise(x, f.bound, inp.k, f.clip, 0)
        except EncodeError as e:
            skipped[f.label] = f"not codable: {e}"
            continue
        keep.append(i)
    if len(keep) < len(inp.fields):
        inp.fields = [inp.fields[i] for i in keep]
        inp.x = np.ascontiguousarray(inp.x[keep])
    inp.meta.update(checked=True, skipped=skipped)
    return inp


def read_rows(arr: Any, date: int, rows: list[int]) -> np.ndarray:
    """Rows (fields within a date) of date ``date`` of an anemoi array, float32 ``(len(rows), cells)``."""
    x = D.read_date(arr, date)
    out = np.ascontiguousarray(x[rows])
    del x
    return out


def load_inputs(store: str, config: str | None, date: int, fields: str | None) -> tuple[Inputs, Any, Any]:
    """Codec, selected fields and their rows of ``date``; also the store's array and dates (for other dates)."""
    codec = D.codec_for(config or store)
    arr, dates = D.open_data(store)
    if not 0 <= date < arr.shape[0]:
        raise SystemExit(f"--date {date}: the store has {arr.shape[0]} dates")
    specs = select_fields(codec, fields)
    x = read_rows(arr, date, [f.index for f in specs])
    dv = str(dates[date]) if dates is not None else None
    return drop_uncodable(Inputs(codec, specs, x, date, dv, str(store), str(config or store))), arr, dates
