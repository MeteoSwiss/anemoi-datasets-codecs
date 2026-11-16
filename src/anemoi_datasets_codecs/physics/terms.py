"""Cross-variable and below-ground terms from a name convention (roles in ``conventions.py``).

Candidate cross sources per target, in priority order (the encoder fits a coefficient per (step, source) and drops a
term where it does not pay; the codec keeps at most 4 sources after the decode-depth limit). With ``n`` levels
indexed from the top (``L[i]`` = family L at level index ``i``, ``n - 1`` = lowest):

* **surface** (``surface_terms``): PMSL <- FIS; T_2M <- FIS, PMSL; PS <- PMSL, FIS, T_2M; SKT <- T_2M, FIS, PMSL;
  TD_2M <- T_2M, QV[n-1], QV[n-2], FIS, PMSL; U/V_10M <- U/V[n-1], U/V[n-2], FIS; CAPE_MU <- CAPE_ML;
  CIN_MU <- CIN_ML; TOT_PREC_6H <- TOT_PREC_1H.
* **level** (``level_terms``), families T, U, V, QV, OMEGA, interleaved: odd indices use their two even
  neighbours (QV also T at the same level), the last odd one its upper neighbour only; with ``cross: chain`` the even
  ones use the even level two above.
* **hydrostatic** (``hydrostatic``), family FI: FI[n-1] <- FIS, T[n-1]; levels of the same parity as the lowest
  from two levels below (FI[i+2], T[i], T[i+1], T[i+2]); the others interpolate (FI[i-1], FI[i+1], T[i]). With
  ``cross: shallow`` (default) and the convention's ``star_level`` present, the levels above the fourth lowest all
  use FI[star] (and T at their level, at the star, and at their even neighbour); FI[n-3] <- FI[n-1], FI[n-4],
  T[n-3], FIS, FI[star]; FI[n-2] <- FI[n-1], FI[n-3], T[n-2], FIS, FI[star], T[n-3]: a decode DAG of depth <= 4.
* **below ground** (``below_ground``, pressure levels only): at the three lowest levels p, where the decoded PS is
  below ``p * pressure_scale``, FI[p] ~ FIS, T[p] ~ T_2M, T_2M PS, PS, U/V[p] ~ U/V_10M; OMEGA[n-1] ~ OMEGA[n-2]
  where PS < p[n-2] and OMEGA[n-2] ~ OMEGA[n-3] where PS < p[n-3]. With these on, U/V_10M no longer use the U/V
  levels and OMEGA[n-2] no longer uses OMEGA[n-1] (they would form cycles).

**Exact-source terms** (:func:`orography_terms`, for the ``field_terms`` stage of a codec chain, ``expand.py``): the
FIS parts of the surface and hydrostatic terms, ``PMSL, T_2M, PS, SKT, TD_2M <- FIS`` (``surface_terms``) and
``FI[n-1] <- FIS`` (``hydrostatic``). The surface geopotential is a static field, exact in every chunk, so these terms
need no closed loop and can leave the lattice codec together with the static fields; the lattice keeps the rest of
each term (its plan without FIS, which the default resolution gives when FIS is not among its variables).

On the ``icon-ch1`` convention this is exactly the baseline plan (same targets, same sources in the same order;
tests/data/plans_ch1_baseline.json, which guards that a change of this module does not change the plan unnoticed).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..plan import MaskSpec, Term, check_acyclic, dependencies
from .conventions import Convention

LEVEL_FAMILIES = ("T", "U", "V", "QV", "OMEGA")
GROUP_SWITCH = {"surface": "surface_terms", "level": "level_terms", "hydrostatic": "hydrostatic"}
OROGRAPHY_TARGETS = ("PMSL", "T_2M", "PS", "SKT", "TD_2M")  # surface roles with an FIS term (and FI[n-1])


class _Missing(str):
    """A role or level the convention does not map (kept in source lists for the report)."""


def _role(c: Convention, role: str) -> str:
    return c.surface.get(role) or _Missing(f"<{role}>")


def _lv(c: Convention, family: str, i: int) -> str:
    name = c.level_name(family, i)
    if name is not None:
        return name
    return _Missing(f"<{family}[{i}]>")


def candidates(c: Convention, mode: str) -> dict[str, tuple[str, list[str]]]:
    """All candidate entries ``target -> (group, sources)`` of the convention (targets may be absent / unmapped)."""
    if mode not in ("on", "chain", "shallow", "off"):
        raise ValueError(f"unknown cross mode {mode!r} (on, shallow, chain, off)")
    plan: dict[str, tuple[str, list[str]]] = {}
    if mode == "off":
        return plan
    n = c.n_levels
    S = lambda r: _role(c, r)  # noqa: E731
    L = lambda f, i: _lv(c, f, i)  # noqa: E731

    def put(target: str, group: str, srcs: list[str]) -> None:
        if not isinstance(target, _Missing):
            plan[target] = (group, srcs)

    for tgt, srcs in (
        ("PMSL", ["FIS"]),
        ("T_2M", ["FIS", "PMSL"]),
        ("PS", ["PMSL", "FIS", "T_2M"]),
        ("SKT", ["T_2M", "FIS", "PMSL"]),
        ("CAPE_MU", ["CAPE_ML"]),
        ("CIN_MU", ["CIN_ML"]),
        ("TOT_PREC_6H", ["TOT_PREC_1H"]),
    ):
        put(S(tgt), "surface", [S(s) for s in srcs])
    put(S("TD_2M"), "surface", [S("T_2M"), L("QV", n - 1), L("QV", n - 2), S("FIS"), S("PMSL")])
    put(S("U_10M"), "surface", [L("U", n - 1), L("U", n - 2), S("FIS")])
    put(S("V_10M"), "surface", [L("V", n - 1), L("V", n - 2), S("FIS")])
    for f in LEVEL_FAMILIES:
        for i in range(n):
            if i % 2 == 1 and i + 1 < n:
                put(L(f, i), "level", [L(f, i - 1), L(f, i + 1)] + ([L("T", i)] if f == "QV" else []))
            elif i % 2 == 1:
                put(L(f, i), "level", [L(f, i - 1)])
            elif mode == "chain" and i >= 2:
                put(L(f, i), "level", [L(f, i - 2)])
    if n:
        put(L("FI", n - 1), "hydrostatic", [S("FIS"), L("T", n - 1)])
    for i in range(n - 1):
        if (n - 1 - i) % 2 == 0:  # same parity as the lowest level: hydrostatic from two levels below
            put(L("FI", i), "hydrostatic", [L("FI", i + 2), L("T", i), L("T", i + 1), L("T", i + 2)])
        else:  # interpolation between the neighbouring levels
            put(L("FI", i), "hydrostatic", ([L("FI", i - 1)] if i else []) + [L("FI", i + 1), L("T", i)])
    return plan


def cross_terms(
    names: Sequence[str], c: Convention, mode: str, switches: dict[str, bool]
) -> tuple[dict[str, list[str]], list[Term]]:
    """Cross plan restricted to ``names`` and the enabled groups, and its report."""
    present = set(names)
    plan = candidates(c, mode)
    if mode == "shallow" and c.star_level is not None and c.n_levels >= 4:
        n = c.n_levels
        k = c.levels.index(c.star_level)
        star = _lv(c, "FI", k)
        if star in present:  # star around FI[star] (the checkerboard is shared between levels)
            for i in range(n):
                if i <= n - 4 and i != k:
                    tgt = _lv(c, "FI", i)
                    if not isinstance(tgt, _Missing):
                        plan[tgt] = (
                            "hydrostatic",
                            [star, _lv(c, "T", i), _lv(c, "T", k), _lv(c, "T", i - 1 if i % 2 else i)],
                        )
            plan.pop(star, None)
            fis = _role(c, "FIS")
            for tgt, srcs in (
                (_lv(c, "FI", n - 3), [_lv(c, "FI", n - 1), _lv(c, "FI", n - 4), _lv(c, "T", n - 3), fis, star]),
                (
                    _lv(c, "FI", n - 2),
                    [_lv(c, "FI", n - 1), _lv(c, "FI", n - 3), _lv(c, "T", n - 2), fis, star, _lv(c, "T", n - 3)],
                ),
            ):
                if not isinstance(tgt, _Missing):
                    plan[tgt] = ("hydrostatic", srcs)
    out: dict[str, list[str]] = {}
    report: list[Term] = []
    for tgt, (group, srcs) in plan.items():
        if tgt not in present:
            continue
        term = Term(tgt, group)
        report.append(term)
        if not switches[GROUP_SWITCH[group]]:
            term.note = f"{GROUP_SWITCH[group]} off"
            continue
        for s in srcs:
            if isinstance(s, _Missing):
                term.dropped.append((str(s), "not in the convention"))
            elif s == tgt:
                continue
            elif s not in present:
                term.dropped.append((s, "absent"))
            else:
                term.applied.append(s)
        if term.applied:
            out[tgt] = list(term.applied)
    check_acyclic(out)
    return out, report


def mask_terms(names: Sequence[str], c: Convention, on: bool) -> tuple[dict[str, MaskSpec], list[Term]]:
    """Below-ground mask terms per target name (see the module docstring); {} when off, without PS or on model
    levels."""
    present = set(names)
    ps = c.surface.get("PS")
    if not on or ps is None or ps not in present or c.vertical != "pressure" or c.n_levels < 3:
        return {}, []
    n = c.n_levels
    out: dict[str, MaskSpec] = {}
    report: list[Term] = []

    def add(tgt: str | None, thr: float, feats: list[tuple[str | None, str | None]]) -> None:
        if tgt is None or tgt not in present:
            return
        term = Term(tgt, "below_ground")
        report.append(term)
        need = [a for a, _ in feats] + [b for _, b in feats if b is not None]
        missing = [x for x in need if x is None or x not in present]
        if missing:
            term.note = "feature variables absent or not in the convention"
            term.dropped = [(str(x), "absent") for x in missing]
            return
        out[tgt] = (ps, thr, [(a, b) for a, b in feats])  # type: ignore[misc]
        term.applied = [ps, *dict.fromkeys(need)]

    sg = c.surface.get
    for i in (n - 1, n - 2, n - 3):
        thr = c.pressure_scale * c.levels[i]
        add(c.level_name("FI", i), thr, [(sg("FIS"), None)])
        add(c.level_name("T", i), thr, [(sg("T_2M"), None), (sg("T_2M"), ps), (ps, None)])
        add(c.level_name("U", i), thr, [(sg("U_10M"), None)])
        add(c.level_name("V", i), thr, [(sg("V_10M"), None)])
    for tgt, src, j in ((n - 1, n - 2, n - 2), (n - 2, n - 3, n - 3)):
        t, s = c.level_name("OMEGA", tgt), c.level_name("OMEGA", src)
        if t is not None and s is not None and t in present and s in present:
            out[t] = (ps, c.pressure_scale * c.levels[j], [(s, None)])
            report.append(Term(t, "below_ground", applied=[ps, s]))
    return out, report


def adapt_for_masks(
    plan: dict[str, list[str]], masks: dict[str, MaskSpec], c: Convention, report: list[Term]
) -> dict[str, list[str]]:
    """Remove the cross terms that would form cycles with the mask terms (U/V_10M <- U/V levels, OMEGA[n-2] <-
    OMEGA[n-1]), recording them in the report."""
    if not masks:
        return plan
    out = {k: list(v) for k, v in plan.items()}
    drops: dict[str, set[str]] = {}
    for w in ("U", "V"):
        tgt = c.surface.get(f"{w}_10M")
        if tgt in out:
            members = c.family_members(w)
            drops[tgt] = {s for s in out[tgt] if s in members}
    o1, o2 = c.level_name("OMEGA", c.n_levels - 1), c.level_name("OMEGA", c.n_levels - 2)
    if o2 in out and o1 is not None:
        drops[o2] = {o1} & set(out[o2])
    for tgt, ds in drops.items():
        out[tgt] = [s for s in out[tgt] if s not in ds]
        for t in report:
            if t.target == tgt and t.kind != "below_ground":
                t.applied = [s for s in t.applied if s not in ds]
                t.dropped += [(s, "cycle with a below-ground term") for s in sorted(ds)]
    out = {k: v for k, v in out.items() if v}
    check_acyclic(dependencies(out, masks))
    return out


def orography_terms(names: Sequence[str], c: Convention, switches: dict[str, Any]) -> dict[str, list[str]]:
    """The exact-source terms among ``names`` (module docstring): target -> ``[FIS]``, in the order PMSL, FI[n-1],
    T_2M, PS, SKT, TD_2M; {} without FIS, with the layer or the cross terms off. ``switches``: the physics group
    (``enabled``, ``cross``, ``surface_terms``, ``hydrostatic``). Whether a target qualifies (a bound, not a static
    field) is the caller's business."""
    present = set(names)
    fis = c.surface.get("FIS")
    if not switches.get("enabled", True) or switches.get("cross") == "off" or fis is None or fis not in present:
        return {}
    roles = [("surface", c.surface.get(OROGRAPHY_TARGETS[0]))]
    if c.n_levels:
        roles.append(("hydrostatic", c.level_name("FI", c.n_levels - 1)))
    roles += [("surface", c.surface.get(r)) for r in OROGRAPHY_TARGETS[1:]]
    return {t: [fis] for g, t in roles if t is not None and t in present and switches[GROUP_SWITCH[g]]}
