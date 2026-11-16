"""Forcing recipes: fields computed from the valid date and static inputs instead of being stored (mode RECIPE).

This module chooses which variables are recipes, by role from the name convention: the roles are the recipe names of
the lattice codec's ``icon_lattice.recipes.RECIPE_IDS`` (``cos_local_time``, ``sin_local_time``, ``insolation``;
``conventions.RECIPE_ROLES``), the inputs the roles of ``recipes.RECIPE_INPUTS``. The recipe ids, formulas and their
date-dependent coefficients are part of the lattice codec's chunk format and live in its core
(``icon_lattice/recipes.py``, imported when a plan is resolved, so that importing the physics layer does not import
the lattice codec); a field is stored losslessly instead when its inputs are not static / lossless fields, when the
valid date is unknown (zarr passes none) and ``recipe_fit`` is off, or when more than 1% of its cells would need
patching (``recipe_bound``, default 1e-6). With ``recipe_fit`` the lattice codec recovers the coefficients from the
field when it has no date (``icon_lattice.recipes.fit_params``).
"""

from __future__ import annotations

from collections.abc import Sequence

from ..plan import RecipeSpec, Term
from .conventions import Convention


def recipe_terms(
    names: Sequence[str], c: Convention, on: bool, fit: bool = False
) -> tuple[dict[str, RecipeSpec], list[Term]]:
    """Recipe fields among ``names`` (target -> (id, input names)) and the report."""
    from ..icon_lattice.recipes import RECIPE_IDS, RECIPE_INPUTS

    present = set(names)
    out: dict[str, RecipeSpec] = {}
    report: list[Term] = []
    inputs = tuple(c.forcings.get(r) or f"<{r}>" for r in RECIPE_INPUTS)
    for role, rid in RECIPE_IDS.items():
        tgt = c.forcings.get(role)
        if tgt is None or tgt not in present:
            continue
        term = Term(tgt, "forcing")
        report.append(term)
        if not on:
            term.note = "forcings off"
            continue
        missing = [x for x in inputs if x not in present]
        term.applied = [x for x in inputs if x in present]
        term.dropped = [(x, "absent") for x in missing]
        if missing:
            term.note = "inputs absent: stored losslessly"
            continue
        out[tgt] = (rid, inputs)  # type: ignore[assignment]
        if fit:
            term.note = "coefficients from the valid date, or fitted to the field without one (recipe_fit)"
        else:
            term.note = "needs the valid date (encode(..., dates=...)); zarr passes none: stored losslessly then"
    return out, report
