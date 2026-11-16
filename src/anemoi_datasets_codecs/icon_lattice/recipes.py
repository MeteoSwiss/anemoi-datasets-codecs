"""Forcing recipes (mode RECIPE), the format-level part: recipe ids, their inputs and the date-dependent coefficients.

Recipe ids and formulas are part of the chunk format (docs/codecs/icon_lattice.md "RECIPE payload", evaluated by
``csrc/decode.c``); the encoder stores the float64 coefficients of the valid date in the payload. Inputs, in this
order: sin_latitude, cos_latitude, sin_longitude, cos_longitude.

1 cos_local_time, 2 sin_local_time: ``cos / sin(2 pi (hour + lon / 15) / 24)`` in angle-addition form;
3 insolation: earthkit's instantaneous cos solar zenith angle at the hour, clipped at 0.

Evaluated in float64 without fused operations, cast to float32; cells further than the recipe bound from the data are
patched with their exact value (``fields.encode_recipe``). Which variables are recipes is the plan's business
(``plan.Plan.recipes``, from the physics layer, ``physics/forcings.py``).

Without the valid date (zarr passes none) :func:`fit_params` recovers the coefficients from the field itself
(``physics.recipe_fit``): a least-squares fit of the formula, snapped to a valid date (a whole second of the day for
the local time, a day of the year and a whole hour for the insolation), so that the coefficients are
``recipe_params`` of that date. The sums are ``math.fsum`` (correctly rounded) over a fixed subsample and the snap
is an integer choice, so the coefficients do not depend on the numpy version. The decoder reads only the
coefficients; the payload's date is then unknown (``NO_DATE``).
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np

FIT_STRIDE = 8  # the fit uses every 8th cell (a fixed subsample; the snap then verifies all of them)
_EPOCH_2000 = 946684800  # 2000-01-01T00 (a leap year: days 0..365 of every year)

RECIPE_IDS = {"cos_local_time": 1, "sin_local_time": 2, "insolation": 3}  # recipe -> id (format)
RECIPE_INPUTS = ("sin_latitude", "cos_latitude", "sin_longitude", "cos_longitude")  # roles, in payload order


def to_datetime(seconds: int) -> dt.datetime:
    return dt.datetime(1970, 1, 1) + dt.timedelta(seconds=int(seconds))


def recipe_params(rid: int, seconds: int) -> np.ndarray:
    """float64 coefficients of recipe ``rid`` at the valid date (seconds since 1970)."""
    d = to_datetime(seconds)
    if rid in (1, 2):
        hour = d.hour + d.minute / 60 + d.second / 3600
        a = 2 * math.pi * hour / 24
        return np.array([math.cos(a), math.sin(a), 0.0, 0.0])
    if rid == 3:  # earthkit cos_solar_zenith_angle, instantaneous at d.hour
        dl = d - dt.datetime(d.year, 1, 1)
        a = (dl.days + dl.seconds / 86400.0) / 365.25 * math.pi * 2
        dec = (
            0.396372
            - 22.91327 * math.cos(a)
            + 4.025430 * math.sin(a)
            - 0.387205 * math.cos(2 * a)
            + 0.051967 * math.sin(2 * a)
            - 0.154527 * math.cos(3 * a)
            + 0.084798 * math.sin(3 * a)
        )
        tc = (
            0.004297
            + 0.107029 * math.cos(a)
            - 1.837877 * math.sin(a)
            - 0.837378 * math.cos(2 * a)
            - 2.340475 * math.sin(2 * a)
        )
        b = math.radians((d.hour - 12) * 15 + tc)
        return np.array([math.sin(math.radians(dec)), math.cos(math.radians(dec)), math.cos(b), math.sin(b)])
    raise ValueError(f"unknown recipe {rid}")


def _fsum(a: np.ndarray) -> float:
    return math.fsum(a.tolist())


def _det3(m: list[list[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _solve(m: list[list[float]], v: list[float]) -> list[float] | None:
    """``m x = v`` by Cramer's rule (2 x 2 or 3 x 3, float64); None when singular."""
    if len(v) == 2:
        d = m[0][0] * m[1][1] - m[0][1] * m[1][0]
        if not d or not math.isfinite(d):
            return None
        return [(v[0] * m[1][1] - m[0][1] * v[1]) / d, (m[0][0] * v[1] - v[0] * m[1][0]) / d]
    d = _det3(m)
    if not d or not math.isfinite(d):
        return None
    out = []
    for k in range(3):
        mk = [row[:] for row in m]
        for r in range(3):
            mk[r][k] = v[r]
        out.append(_det3(mk) / d)
    return out


def _normal(features: list[np.ndarray], y: np.ndarray) -> list[float] | None:
    m = [[_fsum(a * b) for b in features] for a in features]
    return _solve(m, [_fsum(a * y) for a in features])


def fit_params(
    rid: int, y: np.ndarray, sla: np.ndarray, cla: np.ndarray, slo: np.ndarray, clo: np.ndarray
) -> np.ndarray | None:
    """Coefficients of recipe ``rid`` recovered from the field ``y`` and its float32 inputs, snapped to a valid date
    (module docstring): ``recipe_params`` of the date that fits best, or None when the fit is degenerate (an
    insolation field with fewer than 64 sunlit cells in the subsample). Whether they fit every cell is the caller's
    check (``fields.encode_recipe``)."""
    s = slice(None, None, FIT_STRIDE)
    yy = y[s].astype(np.float64)
    if rid in (1, 2):  # 1: p0 cos_lon - p1 sin_lon, 2: p1 cos_lon + p0 sin_lon; (p0, p1) = cos, sin of 2 pi hour / 24
        c, sn = clo[s].astype(np.float64), slo[s].astype(np.float64)
        f = [c, -sn] if rid == 1 else [sn, c]
        p = _normal(f, yy)
        if p is None:
            return None
        a = math.atan2(p[1], p[0]) % (2 * math.pi)
        return recipe_params(rid, round(a / (2 * math.pi) * 86400) % 86400)
    if rid == 3:  # p0 sin_lat + p1 cos_lat (p2 cos_lon - p3 sin_lon) where it is > 0
        lit = np.flatnonzero(yy > 1e-3)
        if len(lit) < 64:
            return None
        sl, cl = sla[s].astype(np.float64)[lit], cla[s].astype(np.float64)[lit]
        c, sn = clo[s].astype(np.float64)[lit], slo[s].astype(np.float64)[lit]
        p = _normal([sl, cl * c, -cl * sn], yy[lit])
        if p is None:
            return None
        h0 = round(12 + math.degrees(math.atan2(p[2], p[1])) / 15)  # the equation of time is within +-17 minutes
        best: tuple[float, int] | None = None
        for day in range(366):
            for h in (h0 - 1, h0, h0 + 1):
                sec = _EPOCH_2000 + day * 86400 + (h % 24) * 3600
                q = recipe_params(3, sec)
                d = max(abs(q[0] - p[0]), abs(q[1] * q[2] - p[1]), abs(q[1] * q[3] - p[2]))
                if best is None or d < best[0]:
                    best = (d, sec)
        assert best is not None
        return recipe_params(3, best[1])
    raise ValueError(f"unknown recipe {rid}")
