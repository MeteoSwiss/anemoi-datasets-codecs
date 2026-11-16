"""ctypes bindings of the C kernels (``csrc/``), compiled at build time into the extension ``_native``.

The library holds two variants of the kernels (AVX2 and a portable scalar emulation of the same intrinsics, see
``csrc/simd.h``) behind one set of exported functions; the variant is chosen when the library is loaded (AVX2 when
the CPU has it, overridable with ``$ANEMOI_DATASETS_CODECS_ISA`` = auto | avx2 | scalar) and can be switched with
:func:`select_isa`. Both variants decode bit-identically.

The library is loaded lazily, once per process, under a lock (a forked child re-uses the parent's handle). Every chunk
decode opens an OpenMP team; libgomp keeps its worker threads between teams, and they do not survive fork(), so a
forked child decodes with 1 thread (``threads.decoder_threads``). ctypes releases the GIL for the
duration of each call. ``$ANEMOI_DATASETS_CODECS_LIBRARY`` points at another build of the library (development).

:class:`Lstsq` binds numpy's own LAPACK least-squares routine the same way, for the encoder's fits.
"""

from __future__ import annotations

import ctypes as C
import importlib.util
import os
import threading
import warnings
from pathlib import Path
from typing import Any

import numpy as np

ERRORS = {
    -1: "malformed container or payload",
    -2: "payload does not match the structure",
    -3: "entropy stream overrun",
    -4: "escape stream overrun",
    -5: "out of memory",
    -6: "missing static array or cross-variable detail",
    -7: "field dependency cycle or invalid dependency",
}

_lib: C.CDLL | None = None
_lock = threading.Lock()

_P = C.c_void_p
_SZ = C.c_size_t


class Step(C.Structure):
    _fields_ = [
        ("kind", C.c_int32),
        ("k", C.c_int32),
        ("n", C.c_int64),
        ("t", _P),
        ("src", _P),
        ("p0", C.c_int64),
        ("srcp", _P),
    ]


class Group(C.Structure):
    _fields_ = [
        ("step", C.c_int32),
        ("pad", C.c_int32),
        ("i0", C.c_int64),
        ("c0", C.c_int64),
        ("ng", C.c_int64),
        ("gpos", _P),
        ("nb0", _P),
        ("nb1", _P),
        ("nb2", _P),
        ("pat", _P),
    ]


class Struct(C.Structure):
    _fields_ = [
        ("n_cells", C.c_int64),
        ("n_steps", C.c_int32),
        ("max_n", C.c_int32),
        ("steps", C.POINTER(Step)),
        ("group", C.POINTER(Group)),
        ("pos", _P),
    ]


def library_path() -> Path:
    """Path of the compiled kernels (``$ANEMOI_DATASETS_CODECS_LIBRARY`` or the ``_native`` extension module)."""
    env = os.environ.get("ANEMOI_DATASETS_CODECS_LIBRARY")
    if env:
        return Path(env)
    spec = importlib.util.find_spec(f"{__package__}._native")
    if spec is None or not spec.origin:
        raise ImportError(
            "the compiled kernels of anemoi_datasets_codecs.icon_lattice (_native) are missing: the package was not "
            "built (install it with pip / uv, which compiles the C extension)"
        )
    return Path(spec.origin)


def lib() -> C.CDLL:
    """The loaded library."""
    global _lib
    if _lib is None:
        with _lock:
            if _lib is None:
                handle = C.CDLL(str(library_path()))
                handle.lc_normalize_freqs.argtypes = [_P, _P, C.c_uint32]
                handle.lc_normalize_freqs.restype = C.c_int
                handle.lc_rans_encode.argtypes = [_P, _P, _P, C.c_int, _P, C.c_int, _P, _SZ, C.c_int]
                handle.lc_rans_encode.restype = _SZ
                handle.lc_rans_decode_test.argtypes = [_P, _SZ, _P, C.c_int, _P, _P, _P, C.c_int]
                handle.lc_rans_decode_test.restype = C.c_int
                handle.lc_decode_field.argtypes = [_P, _SZ, C.POINTER(Struct), _P, _P, _P, _P, _P]
                handle.lc_decode_field.restype = C.c_int
                handle.lc_mask_predict.argtypes = [
                    C.c_int64,
                    _P,
                    C.c_float,
                    C.c_double,
                    C.c_int,
                    _P,
                    _P,
                    _P,
                    C.c_float,
                    C.c_float,
                    C.c_int,
                    _P,
                ]
                handle.lc_mask_predict.restype = None
                handle.lc_decode_chunk.argtypes = [_P, _SZ, C.POINTER(Struct), _P, C.c_int, _P, _P, _P, _P]
                handle.lc_decode_chunk.restype = C.c_int
                handle.lc_recipe_eval.argtypes = [C.c_int, _P, _P, _P, _P, _P, C.c_int64, C.c_int64, _P]
                handle.lc_recipe_eval.restype = None
                handle.lc_irls_scale.argtypes = [C.c_int64, C.c_int64, _P, _P, _P, C.c_int64, C.c_int64, _P, _P]
                handle.lc_irls_scale.restype = None
                handle.lc_struct_check.argtypes = [C.POINTER(Struct)]
                handle.lc_struct_check.restype = C.c_int
                handle.lc_select_isa.argtypes = [C.c_char_p]
                handle.lc_select_isa.restype = C.c_int
                handle.lc_isa.argtypes = []
                handle.lc_isa.restype = C.c_char_p
                handle.lc_available_isas.argtypes = []
                handle.lc_available_isas.restype = C.c_char_p
                _check_isa_env(handle)
                _lib = handle
    return _lib


def _check_isa_env(handle: C.CDLL) -> None:
    """Warn when ``$ANEMOI_DATASETS_CODECS_ISA`` names a variant that is not in use (the library falls back to auto
    for an unknown variant or one this CPU cannot run)."""
    want = os.environ.get("ANEMOI_DATASETS_CODECS_ISA", "").strip()
    got = handle.lc_isa().decode()
    if want and want != "auto" and want != got:
        avail = handle.lc_available_isas().decode()
        warnings.warn(
            f"ANEMOI_DATASETS_CODECS_ISA={want!r} is not available (auto, or one of {avail} on this CPU): using {got}",
            RuntimeWarning,
            stacklevel=3,
        )


def isa() -> str:
    """The kernel variant in use: "avx2" or "scalar"."""
    return lib().lc_isa().decode()


def available_isas() -> list[str]:
    """The variants this CPU can run, best first."""
    return lib().lc_available_isas().decode().split(",")


def select_isa(name: str = "auto") -> str:
    """Switch the kernel variant ("auto", "avx2" or "scalar"; between calls, not during a decode). Returns the
    variant in use; raises ``ValueError`` when the variant is unknown or not supported by this CPU."""
    if lib().lc_select_isa(name.encode()) != 0:
        raise ValueError(f"kernel variant {name!r} is not available (available: {available_isas()})")
    return isa()


def check(rc: int, what: str) -> None:
    """Raise ``ValueError`` for a negative return code."""
    if rc != 0:
        raise ValueError(f"lattice codec {what} failed: {ERRORS.get(rc, f'error {rc}')}")


def ptr(a: np.ndarray | None) -> int | None:
    """Data pointer of a contiguous array (None for None)."""
    if a is None:
        return None
    assert a.flags.c_contiguous
    return a.ctypes.data


def ptr_array(arrays: list[np.ndarray | None]) -> Any:
    """``void *[]`` of the arrays' data pointers (NULL for None). Keep the arrays alive while it is used."""
    out = (C.c_void_p * max(1, len(arrays)))()
    for i, a in enumerate(arrays):
        out[i] = None if a is None else ptr(a)
    return out


def normalize_freqs(counts: np.ndarray, total: int = 4096) -> np.ndarray:
    """Normalise a 256-bin histogram to rANS frequencies summing to ``total`` (escape symbol always >= 1)."""
    c = np.ascontiguousarray(counts, np.uint32)
    f = np.zeros(256, np.uint32)
    check(lib().lc_normalize_freqs(ptr(c), ptr(f), total), "normalize_freqs")
    return f


def rans_encode(syms: np.ndarray, tid: np.ndarray, seg: np.ndarray, freqs: np.ndarray, bits: int = 12) -> bytes:
    """32-lane rANS stream of ``syms`` (uint8), symbol i with table ``tid[i]``; lanes restart at each segment."""
    syms = np.ascontiguousarray(syms, np.uint8)
    tid = np.ascontiguousarray(tid, np.uint8)
    seg = np.ascontiguousarray(seg, np.uint64)
    freqs = np.ascontiguousarray(freqs, np.uint32)
    cap = 4 + 4 * 32 + 2 * len(syms) + 64
    out = np.empty(cap, np.uint8)
    n = lib().lc_rans_encode(ptr(syms), ptr(tid), ptr(seg), len(seg) - 1, ptr(freqs), len(freqs), ptr(out), cap, bits)
    if n == 0:
        raise ValueError("lattice codec rANS encode failed (zero frequency for a coded symbol)")
    return out[:n].tobytes()


def irls_scale(y: np.ndarray, xw: np.ndarray, X: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    """One IRLS reweighting (``csrc/encode.c``): ``s = 1 / sqrt(max(|y - xw|, 0.5))``, ``a = X * s[:, None]``
    (``a`` Fortran-ordered), ``b = y * s``, bit-identical to those numpy expressions. Returns False (nothing written)
    when the arrays are not float64 with element strides (the caller then uses numpy)."""
    f8 = np.dtype(np.float64)
    if not (X.dtype == y.dtype == xw.dtype == f8 and X.ndim == 2 and y.flags.c_contiguous and xw.flags.c_contiguous):
        return False
    if any(st % 8 for st in X.strides) or not (a.flags.f_contiguous and b.flags.c_contiguous):
        return False
    m, n = X.shape
    sr, sc = X.strides[0] // 8, X.strides[1] // 8
    lib().lc_irls_scale(m, n, ptr(y), ptr(xw), X.ctypes.data, sr, sc, a.ctypes.data, ptr(b))
    return True


def recipe_eval(rid: int, params: np.ndarray, sla: Any, cla: Any, slo: Any, clo: Any, n: int) -> np.ndarray:
    """Evaluate recipe ``rid`` (see ``csrc/decode.c``) on float32 static fields; returns float32 ``(n,)``."""
    p = np.ascontiguousarray(params, np.float64)
    arrs = [None if a is None else np.ascontiguousarray(a, np.float32) for a in (sla, cla, slo, clo)]
    out = np.empty(n, np.float32)
    lib().lc_recipe_eval(rid, ptr(p), *(ptr(a) for a in arrs), 0, n, ptr(out))
    return out


# ----------------------------------------------------------------------------------------- least squares
_gelsd_fn: Any = None  # numpy's own LAPACK dgelsd (ctypes), False when unavailable, None before the first look
_gelsd_int: Any = C.c_int64
_gelsd_lock = threading.Lock()


def _find_gelsd() -> Any:
    """The ``dgelsd`` that ``np.linalg.lstsq`` calls, looked up through numpy's ``_umath_linalg`` extension (so the
    dynamic linker resolves it to the same library), or False. Only numpy's scipy-openblas builds name it; the
    binding is checked once against ``np.linalg.lstsq`` (bit-identical solutions) before it is used."""
    global _gelsd_int
    try:
        from numpy.linalg import _umath_linalg

        lapack = np.__config__.CONFIG["Build Dependencies"]["lapack"]
        if lapack.get("name") != "scipy-openblas":
            return False
        ilp64 = "USE64BITINT" in str(lapack.get("openblas configuration", ""))
        # the symbol name carries the integer width (``_64_``): a wrong guess finds no symbol
        fn = getattr(C.CDLL(_umath_linalg.__file__), "scipy_dgelsd_64_" if ilp64 else "scipy_dgelsd_")
    except Exception:  # another numpy build, or no symbol: np.linalg.lstsq
        return False
    _gelsd_int = C.c_int64 if ilp64 else C.c_int32
    fn.argtypes = [_P] * 14
    fn.restype = None
    rng = np.random.default_rng(20261106)
    try:
        for order, (m, k) in (("F", (5000, 3)), ("C", (2000, 5)), ("F", (40, 1)), ("C", (100_000, 6))):
            X = np.asarray(rng.integers(-3000, 3000, (m, k)) * rng.random(k), order=order)
            y = X @ rng.random(k) + rng.standard_normal(m) * 10
            solver = Lstsq(m, k, fn)
            solver.a[...] = X
            solver.b[:m] = y
            got, want = solver.solve(), np.linalg.lstsq(X, y, rcond=None)[0]
            if not np.array_equal(got.view(np.int64), want.view(np.int64)):
                return False
    except Exception:  # the call fails on this build: np.linalg.lstsq
        return False
    return fn


def gelsd() -> Any:
    """numpy's ``dgelsd`` through ctypes (which releases the GIL during the call), or None."""
    global _gelsd_fn
    if _gelsd_fn is None:
        with _gelsd_lock:
            if _gelsd_fn is None:
                _gelsd_fn = _find_gelsd()
    return _gelsd_fn or None


class Lstsq:
    """``np.linalg.lstsq(a, b, rcond=None)[0]`` for one shape ``(m, n)`` and one right-hand side, bit-identical:
    the same LAPACK routine (``dgelsd`` of numpy's OpenBLAS) with the arguments numpy's gufunc passes (``a`` in
    Fortran order, ``rcond = eps * max(m, n)``, the workspace sizes of the query), called through ctypes so that
    concurrent fits run in parallel: numpy's gufunc holds the GIL for the whole call when it is given a single
    problem (it releases it only for calls on many stacked problems). The caller fills :attr:`a` and ``b[:m]``,
    then calls :meth:`solve` (which overwrites both). Without the binding (another numpy build) :meth:`solve` calls
    ``np.linalg.lstsq``."""

    def __init__(self, m: int, n: int, fn: Any = None) -> None:
        self.m, self.n = m, n
        self.a = np.empty((m, n), np.float64, order="F")
        self.b = np.zeros(max(m, n, 1), np.float64)
        self._fn = (fn or gelsd()) if m > 0 and n > 0 else None
        if self._fn is None:
            return
        it = _gelsd_int
        self._s = np.empty(min(m, n), np.float64)
        self._rcond = np.array([np.finfo(np.float64).eps * max(n, m)])
        self._ints = [it(m), it(n), it(1), it(max(m, 1)), it(max(m, n, 1)), it(0), it(-1), it(0)]
        M, N, NRHS, LDA, LDB, self._rank, self._lwork, self._info = self._ints
        work_q, iwork_q = np.empty(1, np.float64), np.empty(1, np.int64 if it is C.c_int64 else np.int32)
        p = C.addressof
        head = [p(M), p(N), p(NRHS), self.a.ctypes.data, p(LDA), self.b.ctypes.data, p(LDB), self._s.ctypes.data]
        head += [self._rcond.ctypes.data, p(self._rank)]
        self._fn(*head, work_q.ctypes.data, p(self._lwork), iwork_q.ctypes.data, p(self._info))  # workspace query
        if self._info.value != 0:
            raise np.linalg.LinAlgError(f"dgelsd workspace query failed ({self._info.value})")
        self._lwork.value = int(work_q[0])
        self._work = np.empty(int(work_q[0]), np.float64)
        self._iwork = np.empty(int(iwork_q[0]), iwork_q.dtype)
        self._args = (*head, self._work.ctypes.data, p(self._lwork), self._iwork.ctypes.data, p(self._info))

    def solve(self) -> np.ndarray:
        """The solution (n,) of ``a x = b[:m]`` in the least-squares sense."""
        if self._fn is None:
            return np.linalg.lstsq(self.a, self.b[: self.m], rcond=None)[0]
        self._fn(*self._args)
        if self._info.value != 0:
            raise np.linalg.LinAlgError("SVD did not converge in Linear Least Squares")
        return self.b[: self.n].copy()
