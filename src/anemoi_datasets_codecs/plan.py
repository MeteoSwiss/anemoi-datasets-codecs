"""The resolved plan the lattice codec consumes, and the decode-DAG utilities. No variable names are known here: the
plan is produced by the physics layer (``anemoi_datasets_codecs.physics``) from the configured name convention, or
built by hand and given to the codec (``LatticeCodec(plan=...)``, checked by :func:`check_names`). Package level so
that the physics layer needs nothing of ``icon_lattice`` to be imported (the lattice codec checks the recipe ids of a
plan against its format's ``icon_lattice.recipes.RECIPE_IDS`` with :func:`check_names`).

A plan has three parts, all keyed by variable name of the codec configuration:

* ``cross``: target -> candidate cross sources, in priority order (the encoder fits one fixed-point coefficient per
  (step, source) on the source's spatial detail and drops a term where it does not pay; at most 4 are kept);
* ``masks``: target -> ``(mask source, threshold, [(A, B or None), ...])``: where the decoded mask source is below
  the threshold, the target is predicted by ``c0 + sum coef_j A_j (B_j or 1)`` (fitted per chunk);
* ``recipes``: target -> ``(recipe id, (4 input names))``: the field is evaluated from the valid date and the inputs
  (formulas in docs/codecs/icon_lattice.md, ids and coefficients in ``icon_lattice.recipes``).

``report`` lists, per candidate term, whether it was applied or dropped and why (``Term``). The plan must be a DAG
(cross and mask sources together); its depth is the decoder's critical path.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Any

MaskSpec = tuple[str, float, list[tuple[str, str | None]]]  # (mask source, threshold, features (a, b or None))
RecipeSpec = tuple[int, tuple[str, str, str, str]]  # (recipe id, input names)


@dataclass
class Term:
    """One candidate term of the plan, for reporting: ``kind`` is the group (``surface``, ``level``,
    ``hydrostatic``, ``below_ground``, ``forcing``), ``applied`` the sources kept, ``dropped`` ``(source, reason)``."""

    target: str
    kind: str
    applied: list[str] = field(default_factory=list)
    dropped: list[tuple[str, str]] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"target": self.target, "kind": self.kind, "applied": list(self.applied)}
        if self.dropped:
            d["dropped"] = [list(x) for x in self.dropped]
        if self.note:
            d["note"] = self.note
        return d


@dataclass
class Plan:
    """Resolved plan (see the module docstring)."""

    cross: dict[str, list[str]] = field(default_factory=dict)
    masks: dict[str, MaskSpec] = field(default_factory=dict)
    recipes: dict[str, RecipeSpec] = field(default_factory=dict)
    report: list[Term] = field(default_factory=list)


def check_names(plan: Plan, names: Sequence[str], recipe_ids: Collection[int] | None = None) -> None:
    """Raise ``ValueError`` unless every name the plan's terms use is one of ``names`` (a plan built by hand; the
    physics layer only emits present names) and every recipe has 4 inputs and, given ``recipe_ids`` (the codec's
    format), a known id. Absent recipe inputs are allowed: such a recipe field is stored losslessly."""
    known = set(names)
    used = [*plan.cross, *(s for v in plan.cross.values() for s in v)]
    used += [*plan.masks, *(s for spec in plan.masks.values() for s in mask_sources(spec))]
    used += list(plan.recipes)
    unknown = sorted(set(used) - known)
    if unknown:
        raise ValueError(f"the plan uses names that are not variables: {unknown}")
    for tgt, (rid, inputs) in plan.recipes.items():
        if (recipe_ids is not None and rid not in recipe_ids) or len(inputs) != 4:
            raise ValueError(f"recipe of {tgt!r}: unknown id {rid!r} or not 4 inputs: {inputs!r}")


def mask_sources(spec: MaskSpec) -> list[str]:
    m, _, feats = spec
    return [m, *[a for a, _ in feats], *[b for _, b in feats if b is not None]]


def dependencies(plan: dict[str, list[str]], masks: dict[str, MaskSpec] | None = None) -> dict[str, list[str]]:
    """All decode-order dependencies: cross sources plus mask sources and features."""
    out = {k: list(v) for k, v in plan.items()}
    for tgt, (src, _, feats) in (masks or {}).items():
        srcs = [src] + [a for a, _ in feats] + [b for _, b in feats if b is not None]
        out[tgt] = list(dict.fromkeys(out.get(tgt, []) + [s for s in srcs if s != tgt]))
    return out


def check_acyclic(plan: dict[str, list[str]]) -> None:
    """Raise ``ValueError`` when the plan has a cycle."""
    state: dict[str, int] = {}

    def visit(u: str) -> None:
        if state.get(u) == 1:
            raise ValueError(f"cross-variable plan has a cycle through {u}")
        if state.get(u) == 2:
            return
        state[u] = 1
        for s in plan.get(u, []):
            visit(s)
        state[u] = 2

    for u in plan:
        visit(u)


def topo_order(names: Sequence[str], plan: dict[str, list[str]]) -> list[int]:
    """Variable indices in an order where every plan source comes before its users."""
    idx = {n: i for i, n in enumerate(names)}
    seen: set[str] = set()
    out: list[int] = []

    def visit(u: str) -> None:
        if u in seen:
            return
        seen.add(u)
        for s in plan.get(u, []):
            visit(s)
        out.append(idx[u])

    for u in names:
        visit(u)
    return out


def limit_depth(
    names: Sequence[str],
    plan: dict[str, list[str]],
    max_depth: int,
    free: set[str] | frozenset[str] = frozenset(),
    masks: dict[str, MaskSpec] | None = None,
) -> dict[str, list[str]]:
    """Drop cross sources so that the decode DAG has at most ``max_depth`` levels of lattice fields above any field
    (``free`` sources, e.g. the static fields, cost nothing: their details come from the caller). Targets are visited
    in topological order and keep the sources, in plan order, whose own depth is below ``max_depth``; mask sources
    are mandatory (a mask term whose sources are too deep is kept and counts)."""
    deps = dependencies(plan, masks)
    order = [names[i] for i in topo_order(names, deps)]
    d: dict[str, int] = {}
    out: dict[str, list[str]] = {}
    for u in order:
        if u in free:
            d[u] = -1
            continue
        mask_srcs = mask_sources(masks[u]) if masks and u in masks else []
        keep = [x for x in plan.get(u, []) if d.get(x, 0) < max_depth]
        if keep:
            out[u] = keep
        d[u] = 1 + max([d.get(x, 0) for x in keep + mask_srcs], default=-1)
    return out


def depth(names: Sequence[str], plan: dict[str, list[str]]) -> dict[str, int]:
    """Longest chain of plan sources below each name (0 = no source)."""
    memo: dict[str, int] = {}

    def d(u: str) -> int:
        if u not in memo:
            memo[u] = 1 + max((d(s) for s in plan.get(u, [])), default=-1)
        return memo[u]

    return {u: d(u) for u in names}


def finalize(
    plan: Plan,
    names: Sequence[str],
    max_depth: int | None,
    free: set[str] | frozenset[str] = frozenset(),
    max_sources: int = 4,
) -> Plan:
    """Apply the codec's limits to a candidate plan: the decode depth (``max_depth``, ``free`` sources cost nothing)
    and at most ``max_sources`` cross sources per target; the dropped sources are recorded in the report."""
    shallow = plan.cross
    if max_depth is not None:
        shallow = limit_depth(names, plan.cross, int(max_depth), free, plan.masks)
    cross = {k: v[:max_sources] for k, v in shallow.items()}
    for t in plan.report:
        if t.kind in ("below_ground", "forcing"):
            continue
        kept = cross.get(t.target, [])
        for s in list(t.applied):
            if s in kept:
                continue
            t.applied.remove(s)
            in_depth = s in shallow.get(t.target, [])
            t.dropped.append((s, f"more than {max_sources} sources" if in_depth else f"decode depth limit {max_depth}"))
    check_acyclic(dependencies(cross, plan.masks))
    return Plan(cross, dict(plan.masks), dict(plan.recipes), plan.report)
