"""Spatial details of integer fields under a structure, with the decoder's fixed-point arithmetic.

The detail of a field is what its spatial prediction misses: ``q - prediction`` with the steps' default weights in
8-bit fixed point (the rule of ``csrc/decode.c``). A cross-variable term on a static source (STATIC_REF) uses the
detail of the static array, computed here on both sides (encoder in native order, decoder in decode order) and cached
per process by (sha256, structure sha); the details of lattice sources come from the encoder / decoder themselves.
"""

from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .structure import Structure

_details: dict[tuple[str, str], np.ndarray] = {}
_lock = threading.Lock()


def spatial_detail(structure: Structure, q: np.ndarray, saturate: bool = True) -> np.ndarray:
    """Detail ``q - prediction`` of an integer field under the structure with its default weights (fixed point,
    the decoder's arithmetic). Used for static cross sources."""
    q = q.astype(np.int64)
    d = np.zeros(len(q), np.int64)
    for st in structure.steps:
        y = q[st.t]
        if st.kind == "delta":
            d[st.t] = y - np.concatenate([[0], y[:-1]])
            continue
        V = q[st.src]
        W = np.round(np.asarray(st.w0[1:]) * 256).astype(np.int64)
        d[st.t] = y - (V[0] + ((W @ (V[1:] - V[0]) + 128) >> 8))
    if not saturate:
        return d
    return np.clip(d, -(2**15), 2**15 - 1).astype(np.int32)  # details are int16 (saturated) on both sides


def static_detail(sha: str, x: np.ndarray, structure: Structure, decode_order: bool = False) -> np.ndarray:
    """Cached detail of a static field (quantised at the finest power-of-two step whose details fit int16), in native
    order (encoder) or in the structure's decode order (what the C chunk decoder reads)."""
    if decode_order:
        key = (sha, structure.sha + "/pos")
        d = _details.get(key)
        if d is None:
            nat = static_detail(sha, x, structure)
            d = np.empty(len(nat), np.int16)
            d[structure.positions()] = nat
            d.setflags(write=False)
            with _lock:
                _details.setdefault(key, d)
                d = _details[key]
        return d
    key = (sha, structure.sha)
    d = _details.get(key)
    if d is None:
        amax = float(np.abs(x).max()) or 1.0
        step = 2.0 ** math.floor(math.log2(amax / 2**20))
        while True:  # the finest power-of-two step whose details fit int16 (no saturation)
            d = spatial_detail(structure, np.rint(x.astype(np.float64) / step), saturate=False)
            if np.abs(d).max() < 2**15:
                break
            step *= 2.0
        d.setflags(write=False)
        with _lock:
            _details.setdefault(key, d)
            d = _details[key]
    return d


def clear_cache() -> None:
    """Forget the static details of this process."""
    with _lock:
        _details.clear()
