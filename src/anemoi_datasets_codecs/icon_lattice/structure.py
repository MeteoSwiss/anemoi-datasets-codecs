"""Prediction structures: which cells are predicted from which, in which order (the pluggable part of the codec).

A structure is an ordered list of steps over the cells in native order:

* ``delta`` (first step only): its targets are coded in that order with a 1D delta predictor (the coarsest cells,
  in snake order on their sub-lattice, plus any cell left without a decoded neighbour).
* ``gather``: targets ``t`` (sorted by native index) predicted from ``k`` already decoded sources ``src[j, i]``
  (native cell indices). Sources are ordered by distance (``src[0]`` is the nearest); the codec stores per-step
  fixed-point weights. Where a source falls outside the domain it is replaced by the nearest present one.

Every cell is the target of exactly one step and every source is decoded by an earlier step
(:meth:`Structure.validate`). Structures are a pure function of the lattice coordinates (``TriangleLatticeLayout``
pixel index, i.e. of the grid), built once per process and cached on disk in the cache directory
(``$ANEMOI_DATASETS_CODECS_CACHE``, default ``~/.cache/anemoi_datasets_codecs``; ``off`` rebuilds them in every process,
a few seconds for ICON-CH1). Chunks store the structure's sha so that a mismatch is detected.

This module holds the model (``Step``, ``Structure``: sha, decoder hints, range split, persistence, validation)
and the cache; the engine (encoder, C decoder) is generic over it. The builders, the only ICON knowledge on this
side, are frozen and live in ``structures/`` (``tree``, ``tree_b``: ``structures/tree.py``; ``hc2_k3``:
``structures/hc2.py``), registered by id in ``structures.STRUCTURES``. A new structure is a builder
``(cell image) -> Structure`` under a new id there (and in ``config.STRUCTURES``); existing ids never change.
"""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from . import native, sidecar

STRUCTURE_VERSION = 3
RANGE_T = 131072  # targets per independent range (csrc/lc.h)
RANGE_G = 43648  # groups per range of the regular-group step


@dataclass(eq=False)
class Step:
    """One step: ``kind`` "delta" or "gather", targets ``t`` (int32), sources ``src`` (int32 ``(k, n)``) and
    default weights ``w0`` (float ``(k,)``, sum 1: used when weights are not fitted)."""

    name: str
    kind: str
    t: np.ndarray
    src: np.ndarray | None = None
    w0: np.ndarray | None = None

    @property
    def k(self) -> int:
        return 0 if self.src is None else int(self.src.shape[0])

    @property
    def n(self) -> int:
        return len(self.t)

    @cached_property
    def t_idx(self) -> np.ndarray:
        """``t`` as a native-int index (the encoder's gathers: no index conversion per call)."""
        return self.t.astype(np.intp)

    @cached_property
    def src_idx(self) -> np.ndarray | None:
        """``src`` as a native-int ``(k, n)`` index (see :attr:`t_idx`)."""
        return None if self.src is None else np.ascontiguousarray(self.src, np.intp)


@dataclass(eq=False)
class Structure:
    """An ordered list of steps over ``n_cells`` cells (see module docstring)."""

    id: str
    n_cells: int
    steps: list[Step]
    sha: str = ""
    _c: Any = field(default=None, repr=False)
    _keep: list[Any] = field(default_factory=list, repr=False)
    _lock: Any = field(default_factory=threading.Lock, repr=False)
    _groups: dict[int, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.sha:
            h = hashlib.sha256(f"{self.id}/{STRUCTURE_VERSION}/{self.n_cells}".encode())
            for s in self.steps:
                h.update(f"{s.kind}/{s.k}/{s.n}".encode())
                h.update(np.ascontiguousarray(s.t, "<i4").tobytes())
                if s.src is not None:
                    h.update(np.ascontiguousarray(s.src, "<i4").tobytes())
            self.sha = h.hexdigest()

    @property
    def sha16(self) -> bytes:
        """First 16 bytes of the sha (stored in every chunk)."""
        return bytes.fromhex(self.sha)[:16]

    @property
    def max_n(self) -> int:
        return max(s.n for s in self.steps)

    def c_struct(self) -> native.Struct:
        """The ctypes ``lc_struct_t`` (built once, under a lock: the arrays stay referenced by this object)."""
        with self._lock:
            return self._build_c()

    def _build_c(self) -> native.Struct:
        if self._c is None:
            steps = (native.Step * len(self.steps))()
            pos = self.positions()
            keep: list[Any] = [pos]
            p0 = 0
            for i, s in enumerate(self.steps):
                t = np.ascontiguousarray(s.t, np.int32)
                src = None if s.src is None else np.ascontiguousarray(s.src, np.int32)
                srcp = None if src is None else np.ascontiguousarray(pos[src], np.int32)
                keep += [t, src, srcp]
                steps[i] = native.Step(
                    0 if s.kind == "delta" else 1, s.k, s.n, native.ptr(t), native.ptr(src), p0, native.ptr(srcp)
                )
                p0 += s.n
            st = native.Struct(self.n_cells, len(self.steps), self.max_n, steps, None, native.ptr(pos))
            groups = (native.Group * len(self.steps))()
            for i in range(len(self.steps)):
                g = self.step_groups(i)
                if g is None:
                    continue
                arrs = [np.ascontiguousarray(pos[g[k]], np.int32) for k in ("central", "nb0", "nb1", "nb2")]
                keep += [*arrs, g["pat"]]
                groups[i] = native.Group(i, 0, 0, g["c0"], g["ng"], *map(native.ptr, arrs), native.ptr(g["pat"]))
            if any(groups[i].ng for i in range(len(self.steps))):
                st.group = groups
                keep.append(groups)
            self._keep = [keep, steps]
            self._c = st
        return self._c

    def positions(self) -> np.ndarray:
        """Decode-order position of every native cell (the order of the steps' targets): the decoder keeps values
        and details in that order, so every step writes contiguously and coarse sources stay in a compact prefix."""
        pos = np.empty(self.n_cells, np.int32)
        pos[np.concatenate([s.t for s in self.steps])] = np.arange(self.n_cells, dtype=np.int32)
        return pos

    def step_groups(self, s: int) -> dict[str, Any] | None:
        """Parent triples of step ``s`` (decoder hint, see lc_group_t): None unless every 3 consecutive targets share
        source 0 and draw sources 1-2 from 3 distinct neighbours. Keys: ng, c0 (>= 0 when the triples are the regular
        native groups of the last step, else -1), central / nb0 / nb1 / nb2 (native cells), pat (uint8)."""
        if s in self._groups:
            return self._groups[s]
        st, out = self.steps[s], None
        if st.kind == "gather" and st.k == 3 and st.n % 3 == 0 and st.n >= 24:
            ng = st.n // 3
            s0, s1, s2 = (st.src[j].reshape(ng, 3).astype(np.int64) for j in range(3))
            if np.all(s0 == s0[:, :1]):
                n0, n1 = s1[:, 0], s2[:, 0]
                others = np.concatenate([s1[:, 1:], s2[:, 1:]], 1)
                n2 = np.where((others != n0[:, None]) & (others != n1[:, None]), others, -1).max(1)
                nb = np.stack([n0, n1, n2])
                pat = np.zeros(ng, np.int64)
                ok = (n2 >= 0) & (n0 != n1)
                for j in (1, 2):
                    for k, x in ((0, s1[:, j]), (1, s2[:, j])):
                        code = np.argmax(nb == x[None, :], axis=0)
                        ok &= nb[code, np.arange(ng)] == x
                        pat |= code << (4 * j - 4 + 2 * k)
                if ok.all():
                    reg = self.regular_groups() if s == len(self.steps) - 1 else None
                    out = {
                        "ng": ng,
                        "c0": reg[0] if reg else -1,
                        "central": s0[:, 0],
                        "nb0": n0,
                        "nb1": n1,
                        "nb2": n2,
                        "pat": pat.astype(np.uint8),
                    }
        self._groups[s] = out
        return out

    def step_ranges(self, s: int) -> list[tuple[int, int]]:
        """Target ranges [lo, hi) of step ``s``, each coded with its own rANS stream and escape slice so that the
        decoder can run them in parallel (``step_ranges`` in csrc/decode.c: RANGE_T targets, or RANGE_G groups for
        a step with parent triples)."""
        st = self.steps[s]
        if st.kind == "delta":
            return [(0, st.n)]
        unit, size, m = (3, RANGE_G, st.n // 3) if self.step_groups(s) is not None else (1, RANGE_T, st.n)
        return [(a * unit, min(a + size, m) * unit) for a in range(0, m, size)][:16] or [(0, 0)]

    def regular_groups(self) -> tuple[int, int] | None:
        """``(c0, ng)`` when the last step is entirely regular groups of 4 consecutive cells (targets
        c0 + 4 q + {0, 1, 3}, source 0 = c0 + 4 q + 2, c0 + 4 ng = n_cells): the decoder's group fast path."""
        st = self.steps[-1]
        if st.kind != "gather" or st.k != 3 or st.n % 3 or st.n < 24:
            return None
        ng = st.n // 3
        c0 = self.n_cells - 4 * ng
        base = c0 + 4 * np.arange(ng)
        ok = np.array_equal(st.t.reshape(ng, 3), base[:, None] + np.array([0, 1, 3]))
        ok = ok and np.array_equal(st.src[0].reshape(ng, 3), np.repeat(base[:, None] + 2, 3, axis=1))
        return (int(c0), int(ng)) if ok else None

    def validate(self) -> None:
        """Raise ``ValueError`` unless every cell is a target exactly once and sources are decoded earlier."""
        if not self.steps or self.steps[0].kind != "delta" or any(s.kind != "gather" for s in self.steps[1:]):
            raise ValueError(f"structure {self.id}: expected one delta step followed by gather steps")
        if len(self.steps) > 64 or any(s.k > 8 or (s.kind == "gather" and s.k < 1) for s in self.steps):
            raise ValueError(f"structure {self.id}: at most 64 steps of 1..8 sources")
        native.check(native.lib().lc_struct_check(self.c_struct()), f"structure {self.id} check")

    def to_npz(self, path: Path) -> None:
        arrays: dict[str, Any] = {"meta": np.array([self.id, self.sha, str(self.n_cells), str(STRUCTURE_VERSION)])}
        for i, s in enumerate(self.steps):
            arrays[f"t{i}"] = s.t
            arrays[f"n{i}"] = np.array([s.name, s.kind])
            if s.src is not None:
                arrays[f"s{i}"] = s.src
                arrays[f"w{i}"] = s.w0
        tmp = path.with_suffix(f".tmp{os.getpid()}.npz")
        np.savez(tmp, **arrays)
        os.replace(tmp, path)

    @classmethod
    def from_npz(cls, path: Path) -> Structure:
        z = np.load(path)
        sid, sha, n_cells, version = (str(x) for x in z["meta"])
        if int(version) != STRUCTURE_VERSION:
            raise ValueError("stale structure cache")
        steps = []
        i = 0
        while f"t{i}" in z:
            name, kind = (str(x) for x in z[f"n{i}"])
            src = z[f"s{i}"] if f"s{i}" in z else None
            steps.append(Step(name, kind, z[f"t{i}"], src, z[f"w{i}"] if src is not None else None))
            i += 1
        s = cls(sid, int(n_cells), steps)
        if s.sha != sha:
            raise ValueError("structure cache is corrupt (sha mismatch)")
        return s


def cell_image(pixel_index: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """``(H, W)`` int64 image of native cell indices, -1 on padding."""
    h, w = shape
    img = np.full(h * w, -1, np.int64)
    img[np.asarray(pixel_index, np.int64)] = np.arange(len(pixel_index))
    return img.reshape(h, w)


# ------------------------------------------------------------------------------------------------ cache
_cache: dict[tuple[str, str], Structure] = {}
_cache_lock = threading.Lock()


def cache_dir() -> Path | None:
    """Disk cache of the structures (``sidecar.cache_dir() / "structures"``; None when the cache is off)."""
    c = sidecar.cache_dir()
    return None if c is None else c / "structures"


def get_structure(sid: str, layout: Any) -> Structure:
    """The structure ``sid`` for a ``TriangleLatticeLayout`` (memory cache, then disk cache, then built)."""
    from .structures import STRUCTURES

    if sid not in STRUCTURES:
        raise ValueError(f"unknown lattice codec structure {sid!r}; known: {sorted(STRUCTURES)}")
    if getattr(layout, "kind", None) != "triangle_lattice":
        raise ValueError(f"the lattice codec needs a triangle_lattice layout, got {getattr(layout, 'kind', None)!r}")
    lsha = layout.sha256
    key = (sid, lsha)
    with _cache_lock:
        if key in _cache:
            return _cache[key]
        d = cache_dir()
        path = None if d is None else d / f"{sid}-v{STRUCTURE_VERSION}-{lsha[:24]}.npz"
        s = None
        if path is not None and path.exists():
            try:
                s = Structure.from_npz(path)
            except (ValueError, KeyError, OSError):
                s = None
        if s is None:
            s = STRUCTURES[sid](cell_image(layout.pixel_index, layout.shape))
            s.validate()
            if path is not None:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    s.to_npz(path)
                except OSError:
                    pass  # read-only cache: rebuild next time
        _cache[key] = s
        return s
