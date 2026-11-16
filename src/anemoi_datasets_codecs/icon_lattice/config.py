"""Codec configuration: small JSON, complete before any data exists (anemoi-datasets builds the codec when it parses
the recipe), stored as the codec ``configuration`` in the array's ``zarr.json``.

Input (``LatticeCodec(**config)``; every key but ``variables`` and ``grid`` optional)::

    variables: [T_2M, PMSL, FIS, ...]      # the chunk's variable order (anemoi's "variables" attribute)
    bounds: {T_2M: 0.05, "T_.*": 0.1}      # max abs error per variable; keys are names or regular expressions
    clip: {CLCT: [0, 100]}                 # optional per-variable clip of the decoded values
    static_fields: [FIS, sin_latitude]     # lossless variables identical in every chunk (STATIC_REF)
    grid: icon-ch1                         # a bundled grid's alias / uuid, or {name, uuid, n_cells, sha256}
    n_ensembles: 1
    structure: tree                        # prediction structure: tree | tree_b | hc2_k3
    lossless: {id: blosc, cname: lz4, clevel: 5, shuffle: 1}   # numcodecs codec of LOSSLESS fields
    options: {...}                         # encoder options, see DEFAULT_OPTIONS
    physics: on                            # the physics layer, see anemoi_datasets_codecs/physics/__init__.py
    chain: false                           # true: follows field stages in a codec chain (zarr3.py)

Bounds and clips (``anemoi_datasets_codecs.config``, shared with the field stages): a variable gets the value of its
exact key, else of the first regular-expression key (in the given order) that matches its whole name; variables
without a bound are lossless. A value is a number (the bound), ``null`` / ``"lossless"``, or ``{max_error, clip}``.
Every key must match at least one variable (typos fail at parse time). Variables may also be given as dicts
``{name, max_error, clip}``.

``get_config`` writes the resolved, canonical form: exact names only, the full grid reference, every option and
every physics key (``clip`` only when a variable has one and ``chain`` only when true: their absence has one fixed
meaning, no clip and no chain). The defaults of the options and of the physics group may change between releases;
the encoder is deterministic for a given package version, configuration, numpy and input. A store therefore records
every encoder setting, so that a later version appending to it encodes as the store was configured. A stored
configuration that lacks a key (written before the key existed, or by hand) still loads: the key takes the default
of the version that reads it, which is why every key is written. The decoder only needs ``grid``, ``structure``,
``lossless`` and the number of fields per date; everything else drives the encoder.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..config import Clip, Variable, resolve_variables

CODEC_ID = "icon_lattice"
CONFIG_VERSION = 1
STRUCTURES = ("tree", "tree_b", "hc2_k3")
DEFAULT_STRUCTURE = "tree"
DEFAULT_LOSSLESS: dict[str, Any] = {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1}
DEFAULT_OPTIONS: dict[str, Any] = {
    "k": 1.0,
    "contexts": 8,
    "tables": "auto",
    "table_penalty": 256,
    "fit": "ls",
    "statics": "ref",
    "rdoq": False,
    "max_depth": 1,
    "prob_bits": "auto",
    "recon_bits": 3,
    "rounding": "direct",
}
# settings of the physics group that are sometimes looked for among the options: option name -> physics key
PHYSICS_OPTIONS = {
    "cross": "cross",
    "below_ground": "below_ground",
    "recipes": "forcings",
    "recipe_bound": "recipe_bound",
}


def check_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    """Effective encoder options (defaults filled in); raises ``ValueError`` on unknown or invalid ones."""
    opts = dict(options or {})
    unknown = set(opts) - set(DEFAULT_OPTIONS)
    if unknown:
        hint = ""
        physics = sorted(unknown & set(PHYSICS_OPTIONS))
        if physics:
            hint = "; " + ", ".join(f"{k!r} is physics.{PHYSICS_OPTIONS[k]}" for k in physics) + " (the physics group)"
        raise ValueError(f"unknown lattice codec options: {sorted(unknown)}{hint}")
    o = {**DEFAULT_OPTIONS, **opts}
    if o["statics"] not in ("ref", "inline", "off"):
        raise ValueError(f"options.statics must be ref, inline or off, got {o['statics']!r}")
    if o["fit"] not in ("ls", "fixed"):
        raise ValueError(f"options.fit must be ls or fixed, got {o['fit']!r}")
    if o["tables"] not in ("auto", "shared", "per_step", "none"):
        raise ValueError(f"options.tables must be auto, shared, per_step or none, got {o['tables']!r}")
    if o["prob_bits"] not in ("auto", 10, 12):
        raise ValueError(f"options.prob_bits must be auto, 10 or 12, got {o['prob_bits']!r}")
    if not (isinstance(o["recon_bits"], int) and 0 <= o["recon_bits"] <= 8):
        raise ValueError(f"options.recon_bits must be an integer in 0..8, got {o['recon_bits']!r}")
    if o["rounding"] not in ("grid", "direct"):
        raise ValueError(f"options.rounding must be grid or direct, got {o['rounding']!r}")
    if not (isinstance(o["contexts"], int) and 1 <= o["contexts"] <= 8):
        raise ValueError(f"options.contexts must be an integer in 1..8, got {o['contexts']!r}")
    if not float(o["k"]) > 0:
        raise ValueError(f"options.k must be > 0, got {o['k']!r}")
    if o["max_depth"] is not None and not (isinstance(o["max_depth"], int) and o["max_depth"] >= 0):
        raise ValueError(f"options.max_depth must be null or an integer >= 0, got {o['max_depth']!r}")
    o["k"] = float(o["k"])
    return o


__all__ = [
    "CODEC_ID",
    "CONFIG_VERSION",
    "DEFAULT_LOSSLESS",
    "DEFAULT_OPTIONS",
    "DEFAULT_STRUCTURE",
    "PHYSICS_OPTIONS",
    "STRUCTURES",
    "Clip",
    "Variable",
    "check_options",
    "resolve_variables",
]
