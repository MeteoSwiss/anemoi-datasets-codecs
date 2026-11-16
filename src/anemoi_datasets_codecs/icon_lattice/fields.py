"""Field-level helpers of the chunk encoder: the field index arithmetic of a chunk and the encoders of the two field
kinds that exist because of the plan (mask terms of LATTICE fields, RECIPE fields).

A chunk ``(dates, variables, ensembles, cells)`` holds ``n_dates * n_variables * n_ensembles`` fields in C order;
:class:`FieldIndex` maps a field index to its variable and to the field of another variable at the same date and
member (the plan is by variable name, the chunk by field index).
"""

from __future__ import annotations

import struct
from collections.abc import Sequence

import numpy as np

from ..plan import MaskSpec, RecipeSpec
from . import native
from .container import EXACT_MODES, MODE_LATTICE, MODE_LOSSLESS, MODE_RECIPE, NO_DATE
from .encoder import MaskTerm, fit_robust
from .recipes import fit_params, recipe_params


class FieldIndex:
    """Field indices of a chunk with ``names`` (variables, chunk order) and ``n_ens`` members per date."""

    def __init__(self, names: Sequence[str], n_ens: int) -> None:
        self.names = tuple(names)
        self.n_ens = int(n_ens)
        self.per = len(self.names) * self.n_ens  # fields per date
        self._pos = {n: k for k, n in enumerate(self.names)}

    def __contains__(self, name: object) -> bool:
        return name in self._pos

    def variable(self, i: int) -> int:
        """Variable index of field ``i``."""
        return (i % self.per) // self.n_ens

    def name(self, i: int) -> str:
        return self.names[self.variable(i)]

    def sibling(self, i: int, name: str) -> int:
        """The field of variable ``name`` at the date and member of field ``i``."""
        return (i // self.per) * self.per + self._pos[name] * self.n_ens + i % self.n_ens

    def describe(self, i: int) -> str:
        """For messages: ``variable 'T_2M' (date offset 0)``."""
        return f"variable {self.name(i)!r} (date offset {i // self.per})"

    def label(self, i: int) -> str:
        """Short name of field ``i``: the variable name, with ``[member]`` when there are several members."""
        return self.name(i) + (f"[{i % self.n_ens}]" if self.n_ens > 1 else "")


def mask_term(
    fields: np.ndarray,
    i: int,
    spec: MaskSpec,
    idx: FieldIndex,
    modes: list[int],
    decoded: dict[int, np.ndarray],
) -> MaskTerm | None:
    """Fit the below-ground mask term ``spec`` of field ``i`` on its masked cells (robust LS on the decoded sources:
    the decoded floats of LATTICE fields, ``decoded``, and the exact values of the other fields). None when a source
    is not available or fewer than 64 cells are masked."""
    src, thr, feats = spec

    def floats(name: str) -> tuple[int, np.ndarray] | None:
        j = idx.sibling(i, name)
        if modes[j] == MODE_LATTICE:
            return (j, decoded[j]) if j in decoded else None
        if modes[j] in EXACT_MODES:
            return j, fields[j]  # exact
        return None

    m = floats(src)
    fa = [(floats(a), None if b is None else floats(b)) for a, b in feats]
    if m is None or any(a is None or (b is None) != (fb is None) for (a, b), (_, fb) in zip(fa, feats, strict=True)):
        return None
    sel = np.flatnonzero(m[1] < np.float32(thr))
    if len(sel) < 64:
        return None
    X = np.stack(
        [a[1][sel].astype(np.float64) * (1.0 if b is None else b[1][sel].astype(np.float64)) for a, b in fa]
        + [np.ones(len(sel))],
        1,
    )
    coef = fit_robust(X, fields[i][sel].astype(np.float64))
    return MaskTerm(
        m[0],
        float(np.float32(thr)),
        float(coef[-1]),
        [(a[0], None if b is None else b[0], float(c)) for (a, b), c in zip(fa, coef[:-1], strict=True)],
        m[1],
        [a[1] for a, _ in fa],
        [None if b is None else b[1] for _, b in fa],
    )


def encode_recipe(
    fields: np.ndarray,
    i: int,
    spec: RecipeSpec,
    seconds: int,
    idx: FieldIndex,
    modes: list[int],
    bound: float,
    fit: bool = False,
) -> tuple[int, bytes]:
    """``(MODE_RECIPE, payload)`` of field ``i`` (the recipe ``spec`` at the valid date ``seconds``, its cells
    further than ``bound`` from the data patched), or ``(MODE_LOSSLESS, b"")`` when the date is unknown (and not
    ``fit``), an input is not an exactly decoded field, or more than 1% of the cells need patching. With ``fit`` and
    no date the coefficients are recovered from the field (``recipes.fit_params``; the payload's date stays
    ``NO_DATE``)."""
    rid, inputs = spec
    if (seconds == NO_DATE and not fit) or not all(n in idx for n in inputs):
        return MODE_LOSSLESS, b""
    src = [idx.sibling(i, n) for n in inputs]
    if any(modes[j] not in EXACT_MODES for j in src):
        return MODE_LOSSLESS, b""
    if seconds == NO_DATE:
        fitted = fit_params(rid, fields[i], *(fields[j] for j in src))
        if fitted is None:
            return MODE_LOSSLESS, b""
        params = fitted
    else:
        params = recipe_params(rid, int(seconds))
    y = native.recipe_eval(rid, params, *(fields[j] for j in src), fields.shape[1])
    x = fields[i]
    bad = np.flatnonzero(np.abs(y.astype(np.float64) - x) > float(bound))
    if len(bad) > 0.01 * len(x):
        return MODE_LOSSLESS, b""
    head = struct.pack("<BBH4HI4dIIq", rid, 4, 0, *src, 0, *params, len(bad), 0, int(seconds))
    patch = np.zeros(len(bad), [("cell", "<u4"), ("value", "<f4")])
    patch["cell"], patch["value"] = bad, x[bad]
    return MODE_RECIPE, head + patch.tobytes()
