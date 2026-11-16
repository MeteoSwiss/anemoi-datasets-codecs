"""Configuration helpers shared by the codecs of the package: per-variable bounds and clips.

The codecs that take ``bounds`` / ``clip`` mappings (``icon_lattice``, ``field_terms``) match them the same way, so
one ``bounds`` mapping can serve every codec of a chain:

* a variable gets the value of its exact key, else of the first regular-expression key (in the given order) that
  matches its whole name; variables without a bound are lossless;
* a value is a number (the bound), ``null`` / ``"lossless"``, or ``{max_error, clip}``; a clip is ``[lo, hi]``
  (``null`` for an open end);
* every key must match at least one variable (typos fail at parse time);
* variables may also be given as dicts ``{name, max_error, clip}``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "IGNORED_KEYS",
    "STAMP_KEY",
    "Clip",
    "UnknownKeys",
    "Variable",
    "match_keys",
    "parse_bound",
    "parse_clip",
    "resolve_variables",
    "writer_stamp",
]

Clip = tuple[float | None, float | None]

# The writer stamp: the Zarr adapters (``zarr3.py``) add ``"written_by": "anemoi-datasets-codecs <version>"`` to the
# configuration they serialise into ``zarr.json``. Purely informational: no engine reads it, it takes no part in any
# comparison of configurations (equality, ``chain.chain_problems``, ``check``), and a configuration without it is
# valid. A codec built from a configuration that carries a stamp keeps it (so a store's metadata read and written
# again by a later version, e.g. when it is resized to append dates, still names the version that created the
# array); one built from a configuration without a stamp gets the running version's. ``physics expand`` emits no
# stamp: its configurations are for a recipe, the stamp is added when the codec is serialised for the store.
STAMP_KEY = "written_by"
# keys every engine accepts and ignores: a codec entry's identifier passed along with its configuration, the stamp
IGNORED_KEYS = frozenset({"id", "name", STAMP_KEY})


def writer_stamp() -> str:
    """The stamp of this package version: ``anemoi-datasets-codecs <version>``."""
    from . import __version__

    return f"anemoi-datasets-codecs {__version__}"


class UnknownKeys(ValueError, TypeError):
    """A codec configuration with keys the codec does not know: a ``ValueError`` like every other configuration
    error, and a ``TypeError`` like an unexpected keyword argument."""


@dataclass(frozen=True)
class Variable:
    """Settings of one variable (applied to every (date, member) slice of it)."""

    name: str
    max_error: float | None = None  # None: lossless
    clip: Clip | None = None

    @property
    def lossy(self) -> bool:
        return self.max_error is not None


def parse_bound(name: str, value: Any) -> tuple[float | None, Clip | None, bool]:
    """(max_error, clip, clip given) from a bounds value."""
    clip, has_clip = None, False
    if isinstance(value, Mapping):
        unknown = set(value) - {"max_error", "clip", "name"}
        if unknown:
            raise ValueError(f"{name}: unknown bound settings {sorted(unknown)}")
        if "clip" in value:
            clip, has_clip = parse_clip(name, value["clip"]), True
        value = value.get("max_error")
    if value is None or value == "lossless":
        return None, clip, has_clip
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: a bound must be a number, null or 'lossless', got {value!r}")
    v = float(value)
    if not (v > 0 and math.isfinite(v)):
        raise ValueError(f"{name}: a bound must be a finite number > 0, got {value!r}")
    return v, clip, has_clip


def parse_clip(name: str, value: Any) -> Clip | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        raise ValueError(f"{name}: clip must be [lo, hi] (null for an open end), got {value!r}")
    lo, hi = (None if v is None else float(v) for v in value)
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"{name}: clip lower bound {lo} > upper bound {hi}")
    return None if lo is None and hi is None else (lo, hi)


def match_keys(keys: Mapping[str, Any], names: Sequence[str], what: str) -> dict[str, str]:
    """Variable name -> key of ``keys`` (exact name first, else the first full regex match). Every key must match."""
    out: dict[str, str] = {}
    used: set[str] = set()
    patterns = []
    for key in keys:
        if key in names:
            continue
        try:
            patterns.append((key, re.compile(key)))
        except re.error as e:
            raise ValueError(f"{what} key {key!r} is neither a variable nor a valid regular expression: {e}") from None
    for n in names:
        if n in keys:
            out[n] = n
            used.add(n)
            continue
        for key, rx in patterns:
            if rx.fullmatch(n):
                out[n] = key
                used.add(key)
                break
    unused = [k for k in keys if k not in used]
    if unused:
        raise ValueError(f"{what} keys match no variable: {unused} (variables: {list(names)})")
    return out


def resolve_variables(
    variables: Sequence[str | Mapping[str, Any]],
    bounds: Mapping[str, Any] | None = None,
    clip: Mapping[str, Any] | None = None,
) -> tuple[Variable, ...]:
    """The per-variable settings in chunk order (see the module docstring)."""
    if isinstance(variables, str) or not isinstance(variables, Sequence) or not variables:
        raise ValueError("variables must be a non-empty list of names")
    base: list[Variable] = []
    for v in variables:
        if isinstance(v, str):
            base.append(Variable(v))
        elif isinstance(v, Mapping) and isinstance(v.get("name"), str):
            err, c, _ = parse_bound(v["name"], dict(v))
            base.append(Variable(v["name"], err, c))
        else:
            raise ValueError(f"a variable is a name or a dict with a name, got {v!r}")
    names = [v.name for v in base]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate variables: {dup}")
    out = list(base)
    if bounds:
        m = match_keys(bounds, names, "bounds")
        for i, v in enumerate(out):
            if v.name in m:
                err, c, has_clip = parse_bound(v.name, bounds[m[v.name]])
                out[i] = Variable(v.name, err, c if has_clip else v.clip)
    if clip:
        m = match_keys(clip, names, "clip")
        for i, v in enumerate(out):
            if v.name in m:
                out[i] = Variable(v.name, v.max_error, parse_clip(v.name, clip[m[v.name]]))
    return tuple(out)
