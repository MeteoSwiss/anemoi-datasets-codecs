"""Name conventions: which variable plays which physical role, and how level families are named.

A convention has

* ``surface``: role -> variable name for the surface quantities the terms use (``SURFACE_ROLES``);
* ``families``: level family -> name pattern with ``{level}`` (``T``, ``U``, ``V``, ``QV``, ``OMEGA``, ``FI``);
* ``levels``: the levels, ordered **from the top of the atmosphere to the surface** (the last one is the lowest:
  1000 hPa on pressure levels, the lowest model level on model levels);
* ``vertical``: ``pressure`` or ``model`` (the below-ground terms need pressure levels);
* ``pressure_scale``: level value times this = the surface pressure's unit (100: hPa levels, PS in Pa);
* ``star_level``: the level whose geopotential every upper level uses in ``cross: shallow`` (500 hPa on CH1;
  None: no star, the ``on`` hydrostatic chain is used);
* ``forcings``: role -> variable name of the forcing recipes and their inputs (``FORCING_ROLES``: the recipes
  ``RECIPE_ROLES``, the keys of the lattice codec's ``recipes.RECIPE_IDS``, and their inputs ``RECIPE_INPUT_ROLES``,
  ``recipes.RECIPE_INPUTS``; a test keeps the two equal).

Built in: ``icon-ch1`` (ICON / COSMO names on the 13 pressure levels of the anemoi ICON-CH1 datasets; the default,
which reproduces the codec's original behaviour) and ``ifs`` (ECMWF short names as in anemoi's ERA5 / IFS datasets:
``2t``, ``msl``, ``t_850``, ``z_500``, ...). A custom convention is a dict: ``extends`` (a built-in name, optional)
plus any of the keys above; a role set to ``null`` is removed. Roles a convention does not map, and variables absent
from the dataset, only drop the terms that need them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

SURFACE_ROLES = (
    "T_2M",
    "PMSL",
    "PS",
    "FIS",
    "SKT",
    "TD_2M",
    "U_10M",
    "V_10M",
    "CAPE_ML",
    "CAPE_MU",
    "CIN_ML",
    "CIN_MU",
    "TOT_PREC_1H",
    "TOT_PREC_6H",
)
FAMILIES = ("T", "U", "V", "QV", "OMEGA", "FI")
RECIPE_ROLES = ("cos_local_time", "sin_local_time", "insolation")  # the forcing recipes (icon_lattice.recipes ids)
RECIPE_INPUT_ROLES = ("sin_latitude", "cos_latitude", "sin_longitude", "cos_longitude")  # their inputs, payload order
FORCING_ROLES = RECIPE_ROLES + RECIPE_INPUT_ROLES
VERTICAL = ("pressure", "model")
CONVENTION_KEYS = ("extends", "surface", "families", "levels", "vertical", "pressure_scale", "star_level", "forcings")
CH1_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)


@dataclass(frozen=True)
class Convention:
    """A naming convention (module docstring): the names of the surface roles (``surface``), the level families as
    format patterns over ``levels`` (``families``, e.g. ``"T_{level}"``), the vertical coordinate and its scale, the
    level of the below-ground star term, and the forcing roles (``forcings``). Built from data by
    :func:`parse_convention`; the built-in ones are ``BUILTIN``."""

    name: str
    surface: dict[str, str] = field(default_factory=dict)
    families: dict[str, str] = field(default_factory=dict)
    levels: tuple[Any, ...] = ()
    vertical: str = "pressure"
    pressure_scale: float = 100.0
    star_level: Any = None
    forcings: dict[str, str] = field(default_factory=dict)

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def level_name(self, family: str, i: int) -> str | None:
        """Variable name of ``family`` at level index ``i`` (0 = top), or None (no such family / index)."""
        pattern = self.families.get(family)
        if pattern is None or not 0 <= i < len(self.levels):
            return None
        return pattern.format(level=self.levels[i])

    def family_members(self, family: str) -> set[str]:
        return {self.level_name(family, i) for i in range(len(self.levels))} - {None}  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": dict(self.surface),
            "families": dict(self.families),
            "levels": list(self.levels),
            "vertical": self.vertical,
            "pressure_scale": self.pressure_scale,
            "star_level": self.star_level,
            "forcings": dict(self.forcings),
        }


ICON_CH1 = Convention(
    name="icon-ch1",
    surface={r: r for r in SURFACE_ROLES},
    families={f: f + "_{level}" for f in FAMILIES},
    levels=CH1_LEVELS,
    vertical="pressure",
    pressure_scale=100.0,
    star_level=500,
    forcings={r: r for r in FORCING_ROLES},
)

IFS = Convention(
    name="ifs",
    surface={
        "T_2M": "2t",
        "PMSL": "msl",
        "PS": "sp",
        "FIS": "z",
        "SKT": "skt",
        "TD_2M": "2d",
        "U_10M": "10u",
        "V_10M": "10v",
    },
    families={
        "T": "t_{level}",
        "U": "u_{level}",
        "V": "v_{level}",
        "QV": "q_{level}",
        "OMEGA": "w_{level}",
        "FI": "z_{level}",
    },
    levels=CH1_LEVELS,
    vertical="pressure",
    pressure_scale=100.0,
    star_level=500,
    forcings={r: r for r in FORCING_ROLES},
)

BUILTIN: dict[str, Convention] = {c.name: c for c in (ICON_CH1, IFS)}


def _names(what: str, value: Any, roles: Sequence[str], base: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"convention {what} must be a mapping role -> name, got {value!r}")
    out = dict(base)
    for role, name in value.items():
        if role not in roles:
            raise ValueError(f"convention {what}: unknown role {role!r} (known: {list(roles)})")
        if name is None:
            out.pop(role, None)
        elif not isinstance(name, str) or not name:
            raise ValueError(f"convention {what}: {role} must be a variable name, got {name!r}")
        else:
            out[role] = name
    taken: dict[str, str] = {}
    for role, name in out.items():
        if name in taken:
            raise ValueError(f"convention {what}: {name!r} is mapped to both {taken[name]} and {role}")
        taken[name] = role
    return out


def parse_convention(value: Any) -> Convention:
    """A built-in name or a custom dict (see the module docstring). Raises ``ValueError`` on anything invalid."""
    if isinstance(value, Convention):
        return value
    if isinstance(value, str):
        if value not in BUILTIN:
            raise ValueError(f"unknown name convention {value!r} (built in: {sorted(BUILTIN)}, or give a dict)")
        return BUILTIN[value]
    if not isinstance(value, Mapping):
        raise ValueError(f"physics.convention must be a name or a dict, got {value!r}")
    unknown = set(value) - set(CONVENTION_KEYS)
    if unknown:
        raise ValueError(f"unknown convention keys {sorted(unknown)} (known: {list(CONVENTION_KEYS)})")
    ext = value.get("extends")
    base = parse_convention(ext) if ext is not None else Convention(name="custom", star_level=None)
    c = replace(base, name="custom" if ext is None else f"{base.name}+custom")
    if "surface" in value:
        c = replace(c, surface=_names("surface", value["surface"], SURFACE_ROLES, c.surface))
    if "forcings" in value:
        c = replace(c, forcings=_names("forcings", value["forcings"], FORCING_ROLES, c.forcings))
    if "families" in value:
        fams = value["families"]
        if not isinstance(fams, Mapping):
            raise ValueError(f"convention families must be a mapping family -> pattern, got {fams!r}")
        out = dict(c.families)
        for fam, pattern in fams.items():
            if fam not in FAMILIES:
                raise ValueError(f"convention families: unknown family {fam!r} (known: {list(FAMILIES)})")
            if pattern is None:
                out.pop(fam, None)
            elif not isinstance(pattern, str) or pattern.count("{level}") != 1 or "{" in pattern.replace("{level}", ""):
                raise ValueError(f"convention families: {fam} needs a pattern with one '{{level}}', got {pattern!r}")
            else:
                out[fam] = pattern
        c = replace(c, families=out)
    if "levels" in value:
        levels = value["levels"]
        if isinstance(levels, str) or not isinstance(levels, Sequence) or not levels:
            raise ValueError(f"convention levels must be a non-empty list (top to surface), got {levels!r}")
        if any(isinstance(v, bool) or not isinstance(v, (int, float, str)) for v in levels):
            raise ValueError(f"convention levels must be numbers or strings, got {levels!r}")
        if len(set(levels)) != len(levels):
            raise ValueError(f"convention levels are not unique: {levels!r}")
        c = replace(c, levels=tuple(levels), star_level=value.get("star_level"))
    elif "star_level" in value:
        c = replace(c, star_level=value["star_level"])
    if c.star_level is not None and c.star_level not in c.levels:
        raise ValueError(f"convention star_level {c.star_level!r} is not one of the levels")
    if "vertical" in value:
        if value["vertical"] not in VERTICAL:
            raise ValueError(f"convention vertical must be one of {VERTICAL}, got {value['vertical']!r}")
        c = replace(c, vertical=value["vertical"])
    if "pressure_scale" in value:
        s = value["pressure_scale"]
        if isinstance(s, bool) or not isinstance(s, (int, float)) or not s > 0:
            raise ValueError(f"convention pressure_scale must be a number > 0, got {s!r}")
        c = replace(c, pressure_scale=float(s))
    if c.vertical == "pressure" and any(not isinstance(v, (int, float)) for v in c.levels):
        raise ValueError("pressure levels must be numbers (the below-ground thresholds are computed from them)")
    names = [c.level_name(f, i) for f in c.families for i in range(c.n_levels)]
    clash = (set(names) & set(c.surface.values())) | {n for n in names if names.count(n) > 1}
    if clash:
        raise ValueError(f"convention maps several roles to the same variables: {sorted(clash)[:5]}")
    return c
