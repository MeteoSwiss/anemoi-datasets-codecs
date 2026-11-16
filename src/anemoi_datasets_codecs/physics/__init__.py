"""The physics layer: everything in the package that knows variable names or physics. The codecs are
variable-agnostic: the lattice codec's core (structures, quantiser, entropy coder, container, statics) consumes the
resolved ``plan.Plan`` this layer produces, and the field stages get their terms and static fields as plain
configuration (``expand.py``, ``anemoi-datasets-codecs physics expand``). Package level: importing it does not import
the lattice codec (the recipe ids are read from ``icon_lattice.recipes`` when a plan is resolved).

Three independent parts, switched by the ``physics`` group of the lattice codec configuration (``PHYSICS_DEFAULTS``):

* cross-variable terms (``terms.py``): ``cross`` (shallow | on | chain | off) and the groups ``surface_terms``,
  ``level_terms``, ``hydrostatic``;
* below-ground mask terms (``terms.py``): ``below_ground`` (off by default);
* forcing recipes (``forcings.py``): ``forcings``, ``recipe_bound``, ``recipe_fit`` (without the valid date,
  recover the recipe's coefficients from the field; on by default);
* the codec chain (``expansion.py``, :func:`expand`): the static fields and the exact-source terms
  (``terms.orography_terms``) as field stages ahead of the lattice codec;

all resolved against a name ``convention`` (``conventions.py``: built-in ``icon-ch1``, the default, or ``ifs``, or a
custom dict). ``physics: off`` (or ``false``, or ``{enabled: false}``) turns the whole layer off. The defaults may
change between releases (the lattice codec writes every key into its stored configuration, so a store keeps its
settings; the baselines pin theirs, ``scripts/plan_baseline.py``). Terms whose variables are absent are dropped;
``Plan.report`` (and ``anemoi-datasets-codecs plan``) says which terms were applied and which were dropped, and why.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..plan import Plan
from . import conventions, forcings, terms
from .conventions import BUILTIN, Convention, parse_convention

PHYSICS_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "convention": "icon-ch1",
    "cross": "shallow",
    "surface_terms": True,
    "level_terms": True,
    "hydrostatic": True,
    "below_ground": False,
    "forcings": True,
    "recipe_bound": 1e-6,
    "recipe_fit": True,
}
CROSS_MODES = ("shallow", "on", "chain", "off")
SWITCHES = ("enabled", "surface_terms", "level_terms", "hydrostatic", "below_ground", "forcings", "recipe_fit")

__all__ = [
    "BUILTIN",
    "CROSS_MODES",
    "PHYSICS_DEFAULTS",
    "Convention",
    "anemoi_order",
    "conventions",
    "cross_plan",
    "expand",
    "expansion",
    "forcings",
    "output_block",
    "parse_convention",
    "parse_physics",
    "resolve",
    "terms",
]


def parse_physics(value: Any = None) -> dict[str, Any]:
    """The canonical physics group (all keys) from ``value`` (None, true / "on", false / "off", or a dict).
    Validates everything, including the convention."""
    if value is None or value is True or value == "on":
        given: dict[str, Any] = {}
    elif value is False or value == "off":
        given = {"enabled": False}
    elif isinstance(value, Mapping):
        given = dict(value)
    else:
        raise ValueError(f"physics must be on / off or a mapping, got {value!r}")
    unknown = set(given) - set(PHYSICS_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown physics keys {sorted(unknown)} (known: {list(PHYSICS_DEFAULTS)})")
    p = {**PHYSICS_DEFAULTS, **given}
    for k in SWITCHES:
        if not isinstance(p[k], bool):
            raise ValueError(f"physics.{k} must be true or false, got {p[k]!r}")
    if p["cross"] not in CROSS_MODES:
        raise ValueError(f"physics.cross must be one of {CROSS_MODES}, got {p['cross']!r}")
    rb = p["recipe_bound"]
    if isinstance(rb, bool) or not isinstance(rb, (int, float)) or not rb > 0:
        raise ValueError(f"physics.recipe_bound must be a number > 0, got {rb!r}")
    p["recipe_bound"] = float(rb)
    parse_convention(p["convention"])
    return p


def resolve(names: Sequence[str], physics: Mapping[str, Any] | None = None) -> Plan:
    """The candidate plan for ``names`` (before the codec's decode-depth limit, ``plan.finalize``)."""
    p = parse_physics(physics)
    if not p["enabled"]:
        return Plan(report=[])
    c = parse_convention(p["convention"])
    masks, mask_report = terms.mask_terms(names, c, p["below_ground"])
    cross, report = terms.cross_terms(names, c, p["cross"], p)
    cross = terms.adapt_for_masks(cross, masks, c, report)
    recipes, recipe_report = forcings.recipe_terms(names, c, p["forcings"], p["recipe_fit"])
    return Plan(cross=cross, masks=masks, recipes=recipes, report=report + mask_report + recipe_report)


def cross_plan(names: Sequence[str], mode: str = "on", convention: Any = "icon-ch1") -> dict[str, list[str]]:
    """Candidate cross sources per variable name, all groups on (the original ``plan.cross_plan``)."""
    c = parse_convention(convention)
    return terms.cross_terms(names, c, mode, {"surface_terms": True, "level_terms": True, "hydrostatic": True})[0]


from . import expansion  # noqa: E402  (it uses parse_physics and parse_convention)
from .expansion import anemoi_order, expand, output_block  # noqa: E402
