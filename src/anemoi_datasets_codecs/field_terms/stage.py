"""The ``field_terms`` engine: a grid-agnostic field stage that replaces lossy target fields by their residuals
against a per-chunk robust linear fit on exact source fields (docs/codecs/field_terms.md).

Configuration (``FieldTerms(**config)``)::

    variables: [FI_1000, FIS, PMSL, ...]    # the fields the stage receives, in chunk order
    terms: {PMSL: [FIS], FI_1000: {sources: [FIS], intercept: true}}   # target -> exact sources
    bounds: {PMSL: 2.0, "FI_.*": 2.0}       # max abs error per variable (names or regular expressions, as icon_lattice)
    clip: {SOME_TARGET: [0, null]}          # optional clip of decoded targets (targets only)
    margin: 0.015625                        # m: the codec after the stage codes a target within b_L = b (1 - m)
    intercept: true                         # default of the terms' intercept

Bounds and clips are matched as in the lattice configuration (``anemoi_datasets_codecs.config.resolve_variables``:
exact name first, else the first regular expression that matches the whole name; every key must match a variable),
so one ``bounds`` mapping can serve both codecs. Only the targets' bounds and clips are used; a source must have no
bound (it must reach the decoder exactly).

Encode, per (date, member) slice and target ``x`` with bound ``b``: fit ``x ~ c0 + sum c_i s_i`` robustly in float64
(``arith.fit_robust``), round the coefficients to float32 (``c0`` is the mean of ``x - sum c_i s_i`` with the rounded
slopes, or 0 without intercept), ``pred = arith.predict(...)``, ``r = fl32(x - pred)``; keep ``r`` when every cell's
reconstruction ``clip(fl32(r' + pred))`` is within ``b`` of ``x`` for every float32 ``r'`` within ``b_L`` of ``r``
(``arith.verify``), else leave ``x`` untouched for this slice (bypass, flagged in the header; a bypassed slice
with a clip must have ``|clip(x) - x| <= b`` in every cell, else ``ValueError``). Decode, in place:
``x = clip(fl32(r + pred))`` for every target not bypassed, ``x = clip(x)`` for a bypassed one (the stage owns its
targets' clips; the codec after it does not clip them). Targets are never sources, so the order of the terms does
not matter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np

from ..config import IGNORED_KEYS, Clip, UnknownKeys, resolve_variables
from ..zarr3 import FieldStage
from . import arith
from . import header as H

CODEC_ID = "field_terms"
DEFAULT_MARGIN = 2.0**-6  # b_L = b (1 - 2^-6); ICON-CH1 needed b_L / b >= 0.9917 (field_terms.md)
MAX_SOURCES = 16


@cache
def recipe_names() -> frozenset[str]:
    """Variable names that the lattice codec's physics layer may evaluate as forcing recipes (mode RECIPE: within
    ``recipe_bound`` of the data, not exact) under any built-in name convention: never a source."""
    from ..physics.conventions import BUILTIN, RECIPE_ROLES

    return frozenset(c.forcings[r] for c in BUILTIN.values() for r in RECIPE_ROLES if r in c.forcings)


@dataclass(frozen=True)
class Term:
    """One resolved term: ``target = c0 + sum c_i sources_i + residual``."""

    target: str
    sources: tuple[str, ...]
    intercept: bool
    bound: float
    clip: Clip | None
    t: int  # target index in variables
    s: tuple[int, ...]  # source indices


def _term_spec(target: str, value: Any, intercept: bool) -> tuple[tuple[str, ...], bool]:
    if isinstance(value, Mapping):
        unknown = set(value) - {"sources", "intercept"}
        if unknown:
            raise ValueError(f"terms.{target}: unknown keys {sorted(unknown)} (known: sources, intercept)")
        if "intercept" in value:
            intercept = value["intercept"]
        value = value.get("sources")
    if isinstance(value, str) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"terms.{target}: the sources must be a non-empty list of variable names, got {value!r}")
    if not isinstance(intercept, bool):
        raise ValueError(f"terms.{target}: intercept must be true or false, got {intercept!r}")
    srcs = tuple(value)
    if not all(isinstance(n, str) and n for n in srcs):
        raise ValueError(f"terms.{target}: sources must be variable names, got {list(srcs)!r}")
    return srcs, intercept


class FieldTerms(FieldStage):
    """Field stage ``field_terms`` (module docstring). ``lattice_bounds`` gives the bound ``b_L`` per target that
    the codec after the stage must not exceed."""

    def __init__(
        self,
        variables: Sequence[str],
        terms: Mapping[str, Any],
        bounds: Mapping[str, Any] | None = None,
        clip: Mapping[str, Any] | None = None,
        margin: float = DEFAULT_MARGIN,
        intercept: bool = True,
        **ignored: Any,
    ) -> None:
        extra = set(ignored) - IGNORED_KEYS
        if extra:
            raise UnknownKeys(f"unknown {CODEC_ID} configuration keys: {sorted(extra)}")
        super().__init__(variables)
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not 0 <= float(margin) < 1:
            raise ValueError(f"margin must be a number in [0, 1), got {margin!r}")
        if not isinstance(intercept, bool):
            raise ValueError(f"intercept must be true or false, got {intercept!r}")
        self._margin = float(margin)
        if not isinstance(terms, Mapping) or not terms:
            raise ValueError("terms must be a non-empty mapping target -> sources")
        names = self.variables
        pos = {n: i for i, n in enumerate(names)}
        resolved = {v.name: v for v in resolve_variables(list(names), bounds, clip)}
        unknown = [t for t in terms if t not in pos]
        if unknown:
            raise ValueError(f"terms: targets {unknown} are not variables (targets are exact names)")
        specs = {t: _term_spec(t, terms[t], intercept) for t in terms}
        targets = set(specs)
        out = []
        for t in sorted(specs, key=pos.__getitem__):
            srcs, icpt = specs[t]
            v = resolved[t]
            if v.max_error is None:
                raise ValueError(f"terms.{t}: the target has no bound (a lossless field gains nothing from a term)")
            bad = [n for n in srcs if n not in pos]
            if bad:
                raise ValueError(f"terms.{t}: sources {bad} are not variables")
            if len(set(srcs)) != len(srcs) or t in srcs:
                raise ValueError(f"terms.{t}: the sources {list(srcs)} repeat a name or contain the target")
            if len(srcs) > MAX_SOURCES:
                raise ValueError(f"terms.{t}: at most {MAX_SOURCES} sources, got {len(srcs)}")
            for n in srcs:
                if n in targets:
                    raise ValueError(
                        f"terms.{t}: source {n!r} is a target of the stage (decoded within its bound, not exact); "
                        "sources must be exact"
                    )
                if resolved[n].max_error is not None:
                    raise ValueError(
                        f"terms.{t}: source {n!r} has a bound (lossy in the codec after the stage); sources must be "
                        "exact (static or lossless fields)"
                    )
                if n in recipe_names():
                    raise ValueError(
                        f"terms.{t}: source {n!r} is a forcing recipe of the lattice codec (within recipe_bound, not "
                        "exact); sources must be exact"
                    )
            out.append(Term(t, srcs, icpt, float(v.max_error), v.clip, pos[t], tuple(pos[n] for n in srcs)))
        clipped = [v.name for v in resolved.values() if v.clip is not None and v.name not in targets]
        if clipped:
            raise ValueError(f"clip applies to the stage's targets only; {clipped} are not targets")
        self._terms = tuple(out)
        self._index: tuple[H.TermIndex, ...] = tuple((tm.t, tm.s) for tm in out)

    # ------------------------------------------------------------------ configuration
    @property
    def terms(self) -> tuple[Term, ...]:
        return self._terms

    @property
    def margin(self) -> float:
        return self._margin

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(tm.target for tm in self._terms)

    @property
    def lattice_bounds(self) -> dict[str, float]:
        """``b_L = b (1 - margin)`` per target: the bound the codec after the stage must code the target with (or
        less). The lattice configuration of a chain carries exactly these values (``physics expand`` writes both)."""
        return {tm.target: self.reduced_bound(tm.bound) for tm in self._terms}

    def reduced_bound(self, bound: float) -> float:
        """``b (1 - margin)`` for a target bound ``b``."""
        return reduced_bound(bound, self._margin)

    def get_config(self) -> dict[str, Any]:
        """The configuration: ``variables``, ``terms`` (each with its sources and intercept), the targets'
        ``bounds`` and ``clip``, ``margin``."""
        cfg = super().get_config()
        cfg["terms"] = {tm.target: {"sources": list(tm.sources), "intercept": tm.intercept} for tm in self._terms}
        cfg["bounds"] = {tm.target: tm.bound for tm in self._terms}
        clips = {tm.target: list(tm.clip) for tm in self._terms if tm.clip is not None}
        if clips:
            cfg["clip"] = clips
        cfg["margin"] = self._margin
        return cfg

    def __getstate__(self) -> dict[str, Any]:  # the configuration, as StaticRef (DataLoader workers, spawn)
        return self.get_config()

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(**state)  # type: ignore[misc]

    def __repr__(self) -> str:
        t = ", ".join(f"{tm.target}<-{'+'.join(tm.sources)}" for tm in self._terms)
        return f"FieldTerms(variables={len(self.variables)}, terms=[{t}], margin={self._margin:g})"

    # ------------------------------------------------------------------ coding
    def _check_finite(self, x: np.ndarray) -> None:
        need = sorted({tm.t for tm in self._terms} | {i for tm in self._terms for i in tm.s})
        for i in need:
            ok = np.isfinite(x[:, i]).all(axis=-1)
            if not ok.all():
                d, e = (int(a[0]) for a in np.nonzero(~ok))
                raise ValueError(
                    f"{CODEC_ID}: variable {self.variables[i]!r} has non-finite values (date offset {d}, member {e}); "
                    "the stage's targets and sources must be finite"
                )

    def encode_term(self, tm: Term, x: np.ndarray, sources: Sequence[np.ndarray]) -> tuple[H.Record, np.ndarray]:
        """The record and the field to emit (the residual, or ``x`` when bypassed) for one target slice."""
        k = len(sources)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            c0, c = arith.fit_robust(x, sources, tm.intercept, arith.IRLS_FLOOR * tm.bound)
            slopes = c.astype(np.float32)
            if tm.intercept:
                p0 = arith.predict(np.float32(0), slopes, sources)
                c0 = float(np.mean(x.astype(np.float64) - p0.astype(np.float64)))
            coefs = np.concatenate([[np.float32(c0)], slopes]).astype(np.float32)
            if np.isfinite(coefs).all():
                pred = arith.predict(coefs[0], coefs[1:], sources)
                r = arith.residual(x, pred)
                if arith.verify(x, r, pred, tm.clip, tm.bound, self.reduced_bound(tm.bound)):
                    return H.Record(False, coefs), r
        if not arith.clip_within(x, tm.clip, tm.bound):
            raise ValueError(
                f"{CODEC_ID}: target {tm.target!r} lies outside its clip {list(tm.clip or ())} by more than its bound "
                f"{tm.bound:g}; decoded values are clipped, so it cannot be coded within the bound (widen the clip)"
            )
        return H.Record(True, np.zeros(1 + k, np.float32)), x

    def encode(self, x: np.ndarray) -> tuple[np.ndarray, bytes]:
        """The chunk with each target replaced by its residual per (date, member) slice (or left untouched where the
        margin does not hold: bypass), and the header with the term table and the coefficients (module docstring)."""
        if x.ndim != 4:
            raise ValueError(f"{CODEC_ID}: needs (dates, fields, ensembles, cells) arrays, got shape {x.shape}")
        self._check_finite(x)
        y = np.array(x, dtype=np.float32, order="C")
        records = []
        for d in range(x.shape[0]):
            for e in range(x.shape[2]):
                row = []
                for tm in self._terms:
                    rec, out = self.encode_term(tm, x[d, tm.t, e], [x[d, i, e] for i in tm.s])
                    if not rec.bypass:
                        y[d, tm.t, e] = out
                    row.append(rec)
                records.append(row)
        return y, H.pack(self._index, records)

    @staticmethod
    def describe_header(
        header: bytes, configuration: dict[str, Any] | None = None, shape: tuple[int, ...] | None = None
    ) -> dict[str, Any]:
        """A summary of the header for ``anemoi-datasets-codecs info``: version, terms, slices (dates x members),
        bypassed records, and per term ``{target, sources, bypassed, coefficients}`` (``coefficients``: ``c0, c1 ..
        ck`` of each slice, None where bypassed; names from the stage's ``configuration`` when given, else variable
        indices). Needs no configuration: the header holds its term table."""
        terms, records = H.read(header)
        names = None
        if configuration is not None:
            names = [v if isinstance(v, str) else v["name"] for v in configuration.get("variables") or ()]

        def nm(i: int) -> Any:
            return names[i] if names and i < len(names) else i

        out: dict[str, Any] = {"version": H.VERSION, "terms": len(terms), "slices": len(records)}
        out["bypassed"] = sum(r.bypass for row in records for r in row)
        out["per_term"] = [
            {
                "target": nm(t),
                "sources": [nm(i) for i in srcs],
                "bypassed": sum(row[k].bypass for row in records),
                "coefficients": [None if row[k].bypass else [float(c) for c in row[k].coefs] for row in records],
            }
            for k, (t, srcs) in enumerate(terms)
        ]
        return out

    def records(self, header: bytes, shape: tuple[int, ...]) -> list[list[H.Record]]:
        """The header's records ``[slice][term]`` for a chunk of ``shape`` (slices: date-major, then member)."""
        return H.parse(header, self._index, shape[0] * shape[2])

    def decode(
        self, y: np.ndarray, header: bytes, out_shape: tuple[int, ...], out: np.ndarray | None = None
    ) -> np.ndarray:
        """In place on ``y`` (the stage keeps the field count)."""
        if tuple(y.shape) != tuple(out_shape) or y.ndim != 4:
            raise ValueError(f"{CODEC_ID}: decodes in place, got shape {y.shape} for {out_shape}")
        recs = self.records(header, y.shape)
        for d in range(y.shape[0]):
            for e in range(y.shape[2]):
                for tm, rec in zip(self._terms, recs[d * y.shape[2] + e], strict=True):
                    if rec.bypass:  # x as the codec after the stage returned it, clipped
                        arith.apply_clip(y[d, tm.t, e], tm.clip)
                        continue
                    pred = arith.predict(rec.coefs[0], rec.coefs[1:], [y[d, i, e] for i in tm.s])
                    r = y[d, tm.t, e]
                    arith.reconstruct(r, pred, tm.clip, out=r)
        return y


def reduced_bound(bound: float, margin: float = DEFAULT_MARGIN) -> float:
    """``b_L = b (1 - m)`` in float64: the bound of a stage target in the configuration of the codec after the stage
    (``FieldTerms.lattice_bounds``)."""
    if not (math.isfinite(bound) and bound > 0 and 0 <= margin < 1):
        raise ValueError(f"reduced_bound: needs a finite bound > 0 and a margin in [0, 1), got {bound!r}, {margin!r}")
    return float(bound) * (1.0 - float(margin))
