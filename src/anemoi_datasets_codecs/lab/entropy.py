"""Entropy estimates of residuals and symbols, shared by the tools (bits, never bytes).

The codec's symbols are ``zigzag(r)`` for 0..254 and 255 for an escape (docs/codecs/icon_lattice.md "Entropy
coding"); :func:`symbols` reproduces that mapping so that estimates here are comparable with the coded size.
"""

from __future__ import annotations

import numpy as np

ESCAPE = 255


def zigzag(r: np.ndarray) -> np.ndarray:
    """``2r`` for ``r >= 0``, ``-2r - 1`` for ``r < 0`` (int64)."""
    r = np.asarray(r, np.int64)
    return np.where(r >= 0, 2 * r, -2 * r - 1)


def symbols(r: np.ndarray) -> np.ndarray:
    """The codec's symbols of residuals ``r``: ``zigzag(r)`` capped at :data:`ESCAPE`."""
    return np.minimum(zigzag(r), ESCAPE).astype(np.int64)


def entropy_bits(sym: np.ndarray, n_symbols: int | None = None) -> float:
    """Order-0 entropy of a symbol stream in bits (total, not per symbol)."""
    sym = np.asarray(sym)
    if sym.size == 0:
        return 0.0
    counts = np.bincount(sym.ravel(), minlength=n_symbols or 0).astype(np.float64)
    nz = counts[counts > 0]
    return float(-(nz * np.log2(nz / sym.size)).sum())


def conditional_entropy_bits(sym: np.ndarray, ctx: np.ndarray) -> float:
    """``H(sym | ctx)`` in bits (total): the order-0 entropy within every context, summed."""
    sym, ctx = np.asarray(sym).ravel(), np.asarray(ctx).ravel()
    if sym.size == 0:
        return 0.0
    order = np.argsort(ctx, kind="stable")
    cs, ss = ctx[order], sym[order]
    bounds = np.flatnonzero(np.diff(cs)) + 1
    return float(sum(entropy_bits(part) for part in np.split(ss, bounds)))


def escape_bits(r: np.ndarray, escape_bits_each: float = 16.0) -> float:
    """Side-stream cost of the escapes of residuals ``r`` (``|r| > 127``) at ``escape_bits_each`` bits each."""
    r = np.asarray(r, np.int64)
    return float(np.count_nonzero(np.abs(r) > 127) * escape_bits_each)


def residual_cost_bits(r: np.ndarray, ctx: np.ndarray | None = None) -> float:
    """Estimated coded size of residuals ``r`` in bits: symbol entropy (conditional on ``ctx`` when given) plus the
    escapes at 16 bits each. The quantity the codec's encoder minimises, without table bytes."""
    s = symbols(r)
    h = entropy_bits(s, ESCAPE + 1) if ctx is None else conditional_entropy_bits(s, ctx)
    return h + escape_bits(r)


def laplace_fit(r: np.ndarray) -> tuple[float, float]:
    """Maximum-likelihood Laplace fit of residuals: ``(location = median, scale = mean |r - median|)``."""
    r = np.asarray(r, np.float64)
    loc = float(np.median(r))
    return loc, float(np.mean(np.abs(r - loc)))
