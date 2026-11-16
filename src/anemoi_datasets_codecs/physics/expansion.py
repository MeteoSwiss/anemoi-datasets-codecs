"""Codec chains from the physics layer: :func:`expand` (``physics.expand``) turns a dataset's variables, the physics
settings and a lattice codec configuration into the codecs of a chain ``[field_terms, static_ref,
icon_lattice_bytes]``, the ``output.compressors`` of an anemoi-datasets recipe (``anemoi-datasets-codecs physics
expand``).

What goes where:

* ``static_ref``: the configured static fields (the lattice configuration's ``static_fields`` present in the
  dataset; absent ones are dropped) leave the chunk as references into the sidecar; ``inline: true`` when
  ``options.statics`` is ``inline``.
* ``field_terms``: the exact-source terms of the name convention (``terms.orography_terms``: PMSL, FI at the lowest
  level, T_2M, PS, SKT, TD_2M <- FIS), mapped to the dataset's names, for every target that is present, has a bound
  and is not a static field, when the source is one of the static fields (exact: restored bit-identically by
  ``static_ref``). The targets' bounds and clips move to the stage; ``margin`` m (below).
* ``icon_lattice_bytes``: the other fields, with the bounds and clips of the configuration resolved per name, the
  stage targets at ``b_L = b (1 - m)`` (``FieldTerms.lattice_bounds``) and without a clip, ``static_fields: []``,
  ``chain: true`` and the physics group as given: without the static fields among its variables the default
  resolution drops every term with a static source (the FIS-free plan: the parts the stage took and the FIS parts of
  the terms it did not, e.g. U_10M, V_10M, FI_925), so no ``plan`` key is needed. One exception: with
  ``below_ground`` on and the surface pressure (the mask source of every below-ground term) a stage target, the
  lattice codec gets ``below_ground: false``: it sees the residual of PS, not PS, so the mask ``PS < level`` would hold
  on every cell and the mask predictor would replace the spatial prediction everywhere (on four ICON-CH1 dates the
  chunk was 14-20% larger than without the terms).

Nothing in the lattice configuration depends on a chunk or a stage header: both configurations carry the same static
``b_L``. Without static fields to reference (none configured or present, or ``options.statics: off``) there is no
chain: the result is the single lattice codec (no ``chain`` key) with the statics coded inside it as before, and the
physics group as given.

Bounds and clips are resolved as the lattice codec resolves them (``config.resolve_variables``): the configuration's
``bounds`` and ``clip`` mappings over the variables, which may be dicts ``{name, max_error, clip}``; a dataset variable
given by name takes the settings of the configuration's dict of that name.

Forcings (``forcing_bound``): zarr passes no valid date, and a chain's lattice codec has no latitude / longitude, so
the forcing recipes (cos / sin local time, insolation, by role from the convention) are stored losslessly in a chain,
about 15% of an ICON-CH1 chunk. ``forcing_bound`` b gives every forcing that is present and has no bound of its own
the bound b (an exact key ahead of the configuration's ``bounds``; an exact key there, even ``null``, is kept), so
that it is coded lossily; 1e-6 is the recipes' own tolerance (``physics.recipe_bound``). Where the result is the
single codec with ``physics.recipe_fit`` on (the default), the forcings its plan makes recipes (64 bytes each) do not
get the bound: a lossy field is never a recipe. Default None: as configured.

The margin: given, or from the data range when ``ranges`` (per variable ``(min, max)``, e.g. an anemoi store's
``minimum`` / ``maximum`` statistics) are given: the power of two at or above ``4 ulp(A) / b`` for every target
(``A = max(|min|, |max|) + b``, float32 ulps: the stage's float32 roundings of ``r`` and ``x'``), at least ``2^-10``;
a target that would need more than ``2^-4`` stays in the lattice codec (its float32 resolution is close to its bound;
the stage would bypass it). Without ranges: ``field_terms.DEFAULT_MARGIN`` (``2^-6``). The stage verifies every cell
at encode time either way; the margin only decides how often it bypasses.

Variable order: the configurations name the fields in chunk order, which for an anemoi-datasets build is anemoi's
order (it sorts the variables by name, ``param_level``), not the order of the recipe's sources: give ``variables`` in
the dataset's order, or pass the recipe's list through :func:`anemoi_order`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from ..config import STAMP_KEY, resolve_variables
from . import parse_convention, parse_physics, resolve, terms
from .conventions import RECIPE_ROLES

__all__ = ["MARGIN_MAX", "MARGIN_MIN", "anemoi_order", "expand", "margin_for", "output_block"]

MARGIN_MIN = 2.0**-10
MARGIN_MAX = 2.0**-4
LATTICE_NAMES = ("icon_lattice_bytes", "icon_lattice")


def anemoi_order(items: Iterable[Any]) -> list[str]:
    """Variable names in the order anemoi-datasets writes them (sorted by name, ``param_level``) from a recipe-style
    list: names, or mappings ``{param: [T, U], levelist: [500, 850]}`` (``levelist`` / ``level`` optional; a name
    per param and level, ``T_500``)."""
    names: list[str] = []
    for it in items:
        if isinstance(it, str):
            names.append(it)
            continue
        if not isinstance(it, Mapping) or "param" not in it:
            raise ValueError(f"a variable is a name or a mapping with a param, got {it!r}")
        params = [it["param"]] if isinstance(it["param"], str) else list(it["param"])
        levels = it.get("levelist", it.get("level"))
        if levels is None:
            names += params
        else:
            levels = [levels] if isinstance(levels, (str, int, float)) else list(levels)
            names += [f"{p}_{lv}" for p in params for lv in levels]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate variables: {sorted({n for n in names if names.count(n) > 1})}")
    return sorted(names)


def margin_for(bounds: Mapping[str, float], ranges: Mapping[str, Sequence[float]]) -> tuple[float, dict[str, float]]:
    """The margin for targets with ``bounds`` given the data ``ranges`` (module docstring) and, per target, the margin
    it needs; targets without a range need nothing."""
    need: dict[str, float] = {}
    for t, b in bounds.items():
        if t not in ranges:
            continue
        lo, hi = (float(v) for v in ranges[t])
        a = max(abs(lo), abs(hi)) + b
        need[t] = 4.0 * float(np.spacing(np.float32(a))) / b if math.isfinite(a) else math.inf
    worst = max([v for v in need.values() if v <= MARGIN_MAX], default=0.0)
    m = MARGIN_MIN
    while m < worst:
        m *= 2
    return m, need


def _entry(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "configuration": cfg}


def expand(
    variables: Sequence[str] | None = None,
    physics: Any = None,
    lattice: Mapping[str, Any] | None = None,
    *,
    margin: float | None = None,
    ranges: Mapping[str, Sequence[float]] | None = None,
    check: bool = True,
    forcing_bound: float | None = None,
) -> list[dict[str, Any]]:
    """The codec entries ``{name, configuration}`` of the chain (encode order, module docstring).

    Args:
        variables: The dataset's variable names in its order (anemoi's: :func:`anemoi_order`); default the
            lattice configuration's ``variables``.
        physics: The physics group (as the lattice configuration's ``physics``); default the lattice
            configuration's.
        lattice: The lattice codec configuration: ``grid``, ``bounds``, ``clip``, ``static_fields``, ``n_ensembles``,
            ``structure``, ``lossless``, ``options`` (its ``variables`` only as the default of ``variables``).
        margin: The stage margin m; default from ``ranges``, else ``field_terms.DEFAULT_MARGIN``.
        ranges: Per variable ``(min, max)`` of the data, for the margin.
        check: Verify the result with ``chain.chain_problems`` (raise ``ValueError`` on any problem).
        forcing_bound: A bound for the forcings without one (module docstring); default None.
    """
    from ..field_terms import DEFAULT_MARGIN, FieldTerms
    from ..icon_lattice import LatticeCodec
    from ..icon_lattice.config import check_options
    from ..static_ref import StaticRef

    lat = dict(lattice or {})
    for k in ("id", "name", "chain", "version", STAMP_KEY):  # the stamp belongs to a store, not to a recipe
        lat.pop(k, None)
    if variables is None:
        variables = lat.get("variables")
    if variables is None:
        raise ValueError("expand needs the dataset's variables (or a lattice configuration with variables)")
    configured = {v["name"]: v for v in lat.get("variables") or () if isinstance(v, Mapping) and "name" in v}
    variables = [configured.get(v, v) if isinstance(v, str) else v for v in variables]
    resolved = {v.name: v for v in resolve_variables(variables, lat.get("bounds"), lat.get("clip"))}
    names = list(resolved)
    lat["variables"] = variables
    if physics is None:
        physics = lat.get("physics")
    p = parse_physics(physics)
    statics = [n for n in lat.get("static_fields") or () if n in resolved]  # those absent from the dataset dropped
    opts = check_options(lat.get("options"))
    single = not statics or opts["statics"] == "off"  # nothing to reference: the single codec
    if forcing_bound is not None:
        fb = float(forcing_bound)
        if not (math.isfinite(fb) and fb > 0):
            raise ValueError(f"forcing_bound must be a number > 0, got {forcing_bound!r}")
    if forcing_bound is not None:
        conv = parse_convention(p["convention"])
        # the single codec with recipe_fit stores its plan's recipes as recipes (zarr's dateless writes included)
        recipes = resolve(names, p).recipes if single and p["recipe_fit"] else {}
        add = {
            n: fb
            for role, n in conv.forcings.items()
            if role in RECIPE_ROLES
            and n in resolved
            and n not in recipes
            and not resolved[n].lossy
            and n not in (lat.get("bounds") or {})
        }
        if add:
            lat["bounds"] = {**add, **dict(lat.get("bounds") or {})}
            resolved = {v.name: v for v in resolve_variables(variables, lat["bounds"], lat.get("clip"))}
    lat["static_fields"] = statics
    if single:
        return [_entry("icon_lattice_bytes", LatticeCodec(**{**lat, "physics": physics}).get_config())]
    lossy = [n for n in statics if resolved[n].lossy]
    if lossy:
        raise ValueError(f"static fields are stored exactly and cannot have a bound: {lossy}")

    st = set(statics)
    candidates = terms.orography_terms(names, parse_convention(p["convention"]), p)
    chosen = {
        t: srcs
        for t, srcs in candidates.items()
        if t not in st and resolved[t].lossy and all(s in st and not resolved[s].lossy for s in srcs)
    }
    bounds = {t: float(resolved[t].max_error) for t in chosen}  # type: ignore[arg-type]
    if margin is None:
        if ranges is not None:
            margin, need = margin_for(bounds, ranges)
            chosen = {t: s for t, s in chosen.items() if need.get(t, 0.0) <= MARGIN_MAX}
        else:
            margin = DEFAULT_MARGIN
    out = []
    reduced: dict[str, float] = {}
    if chosen:
        stage = FieldTerms(
            variables=names,
            terms=chosen,
            bounds={t: bounds[t] for t in chosen},
            clip={t: list(resolved[t].clip) for t in chosen if resolved[t].clip is not None},
            margin=margin,
        )
        reduced = stage.lattice_bounds
        out.append(_entry("field_terms", stage.get_config()))
    ref = StaticRef(variables=names, static_fields=statics, inline=opts["statics"] == "inline")
    out.append(_entry("static_ref", ref.get_config()))
    kept = list(ref.variables_out)
    lcfg = {k: v for k, v in lat.items() if k not in ("variables", "bounds", "clip", "static_fields", "physics")}
    ps = parse_convention(p["convention"]).surface.get("PS")
    if p["enabled"] and p["below_ground"] and ps in chosen:  # the lattice codec sees PS's residual (docstring)
        physics = {**(physics if isinstance(physics, Mapping) else {}), "below_ground": False}
    lcfg.update(
        variables=kept,
        bounds={n: reduced.get(n, resolved[n].max_error) for n in kept if resolved[n].lossy},
        clip={n: list(resolved[n].clip) for n in kept if resolved[n].clip is not None and n not in reduced},
        static_fields=[],
        physics=physics,
        chain=True,
    )
    out.append(_entry("icon_lattice_bytes", LatticeCodec(**lcfg).get_config()))
    if check:
        from ..chain import chain_problems

        problems = chain_problems(out)
        if problems:
            raise ValueError("the expanded chain is inconsistent: " + "; ".join(problems))
    return out


def output_block(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The ``output`` section of an anemoi-datasets recipe for the codecs ``entries``: Zarr format 3, one date and
    all members of a date per chunk, the grid in one chunk (``chunking: values``: without it anemoi splits a date
    larger than 32 MiB into grid chunks, which the lattice codec refuses)."""
    lat = next(e["configuration"] for e in entries if e["name"] in LATTICE_NAMES)
    n_cells = int(lat["grid"]["n_cells"])
    chunking = {"dates": 1, "ensembles": int(lat.get("n_ensembles", 1)), "values": n_cells}
    return {"format": "zarr3", "chunking": chunking, "compressors": [dict(e) for e in entries]}
