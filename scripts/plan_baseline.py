"""Byte-identity baseline of the cross-variable / physics plan: a synthetic chunk with the 115 ICON-CH1 variable names
(smooth fields with shared signals, so that the terms pay) on the test lattice, encoded under several plan settings.

    python scripts/plan_baseline.py write OUT.json     # sha256 of the chunk bytes per setting (a new file)
    python scripts/plan_baseline.py add OUT.json       # add the settings OUT.json lacks; existing entries kept
    python scripts/plan_baseline.py check OUT.json     # re-encode and compare

The keys of OUT.json name the settings (``SETTINGS``: encoder options, physics group, and whether the valid date is
passed). The baseline is tests/data/chunk_sha256_baseline_numpy-2.5.4.json (the below-ground settings depend on the
numpy version and on the kernels its linear algebra library picks for the CPU: the mask coefficients are a float64
least-squares fit; ``BASELINE_BLAS``, ``blas_architecture``). Every setting spells out the options whose
default has changed (``PINNED``), so that a later change of a default does not change what a setting encodes: the
settings of the first baseline pin the defaults they were written with (``rounding: grid``, ``recipe_fit: false``),
the ``direct-*`` settings the defaults of release 0.1.0. A change of the encoder, the plan, the structures or the C
kernels that is not meant to change the bytes must keep ``check`` identical; one that is meant to (an encoder
improvement) adds settings and entries, never rewrites existing ones.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import conftest  # noqa: E402

from anemoi_datasets_codecs.icon_lattice import LatticeCodec  # noqa: E402

LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
SURFACE = [
    "PMSL",
    "PS",
    "T_2M",
    "TD_2M",
    "SKT",
    "U_10M",
    "V_10M",
    "FIS",
    "CAPE_ML",
    "CAPE_MU",
    "CIN_ML",
    "CIN_MU",
    "TOT_PREC_1H",
    "TOT_PREC_6H",
]
STATIC = ["FIS", "sin_latitude", "cos_latitude", "sin_longitude", "cos_longitude"]
FORCING = ["cos_local_time", "sin_local_time", "insolation"]
# the encoder settings the first baseline was written with (the defaults before release 0.1.0), and those of the
# default configuration of release 0.1.0: (options, physics)
LEGACY = ({"rounding": "grid"}, {"recipe_fit": False})
DIRECT = ({"rounding": "direct"}, {"recipe_fit": True})


def _setting(pinned: tuple[dict, dict], options: dict, physics: dict, dated: bool = True) -> tuple[dict, dict, bool]:
    return {**pinned[0], **options}, {**pinned[1], **physics}, dated


# setting name -> (encoder options, physics group, valid date passed); the names of the first eight are those of the
# first baseline (written with the defaults of LEGACY, now pinned): never rename or change them
SETTINGS = {
    "default": _setting(LEGACY, {}, {}),
    "cross-on": _setting(LEGACY, {}, {"cross": "on"}),
    "cross-chain": _setting(LEGACY, {}, {"cross": "chain"}),
    "cross-off": _setting(LEGACY, {}, {"cross": "off"}),
    "below-ground": _setting(LEGACY, {}, {"below_ground": True}),
    "below-ground-on-depth2": _setting(LEGACY, {"max_depth": 2}, {"below_ground": True, "cross": "on"}),
    "no-depth-limit": _setting(LEGACY, {"max_depth": None}, {}),
    "no-recipes": _setting(LEGACY, {}, {"forcings": False}),
    # the default configuration of release 0.1.0 (direct rounding, recipe_fit), with and without the valid date (zarr
    # passes none: the recipes' coefficients are then fitted to the fields)
    "direct-recipe-fit": _setting(DIRECT, {}, {}),
    "direct-recipe-fit-nodate": _setting(DIRECT, {}, {}, dated=False),
    "direct-recipe-fit-below-ground": _setting(DIRECT, {}, {"below_ground": True}),
}


def names() -> list[str]:
    lv = [f"{f}_{p}" for f in ("FI", "OMEGA", "QV", "T", "U", "V") for p in LEVELS]
    return lv + SURFACE + [s for s in STATIC if s not in SURFACE] + FORCING


def chunk(lattice) -> np.ndarray:
    layout, rr, cc, lat, lon = lattice
    n = len(rr)
    rng = np.random.default_rng(7)
    shared = rng.standard_normal(n)
    out = {}
    for k, name in enumerate(names()):
        base = conftest.smooth(rr, cc, 100 + k, 12 + k % 7)
        if name.startswith("FI_"):
            out[name] = 1e4 * (1 + LEVELS.index(int(name[3:]))) + 300 * base + 3 * shared
        elif name in ("PMSL", "PS"):
            out[name] = 1e5 + 500 * base + 20 * shared
        else:
            out[name] = 10 * base + 0.3 * shared
    out.update(conftest.forcings(lat, lon, conftest.DATE))
    out["sin_latitude"], out["cos_latitude"] = np.sin(np.radians(lat)), np.cos(np.radians(lat))
    out["sin_longitude"], out["cos_longitude"] = np.sin(np.radians(lon)), np.cos(np.radians(lon))
    return np.stack([out[n] for n in names()]).astype(np.float32)


def bounds() -> dict:
    b = {}
    for n in names():
        if n in STATIC or n in FORCING:
            continue
        b[n] = 20.0 if n.startswith("FI_") or n in ("PMSL", "PS") else 0.05
    return b


def codec(setting: str, lattice) -> LatticeCodec:
    """The codec of a setting on the 115 ICON-CH1 names."""
    options, physics, _ = SETTINGS[setting]
    kw = {"variables": names(), "grid": lattice[0], "bounds": bounds(), "static_fields": STATIC}
    return LatticeCodec(**kw, options=options, physics=physics)


def encode(setting: str) -> str:
    """sha256 of the synthetic chunk encoded under ``setting``."""
    lattice = conftest.make_lattice()
    dates = [conftest.DATE] if SETTINGS[setting][2] else None
    return hashlib.sha256(codec(setting, lattice).encode(chunk(lattice), dates=dates)).hexdigest()


BASELINE_BLAS = ("Haswell", "Zen")  # OpenBLAS kernel families that reproduce the baseline's below-ground settings


def blas_architecture() -> str | None:
    """The kernel family numpy's OpenBLAS runs on this CPU (``OPENBLAS_CORETYPE`` overrides it), None if unknown.
    The least-squares fit of the below-ground terms rounds differently with other kernels, so those settings'
    bytes are comparable with the baseline only on ``BASELINE_BLAS`` (the decoded values are within the bounds
    with any)."""
    from threadpoolctl import threadpool_info

    return next((d.get("architecture") for d in threadpool_info() if d.get("internal_api") == "openblas"), None)


def blas_dependent(setting: str) -> bool:
    """True for the settings whose bytes depend on the linear algebra kernels (those with below-ground terms)."""
    return bool(SETTINGS[setting][1].get("below_ground"))


def main() -> None:
    mode, path = sys.argv[1], Path(sys.argv[2])
    if mode == "write":
        if path.exists():
            raise SystemExit(f"{path} exists: a baseline is never rewritten (use add for new settings)")
        res = {s: encode(s) for s in SETTINGS}
        path.write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return
    if mode == "add":
        res = json.loads(path.read_text())
        new = {s: encode(s) for s in SETTINGS if s not in res}
        path.write_text(json.dumps({**res, **new}, indent=1))
        print(json.dumps(new, indent=1))
        return
    ref = json.loads(path.read_text())
    ok = True
    for s in SETTINGS:
        if s not in ref:
            ok = False
            print(f"{s}: MISSING from {path}")
            continue
        got = encode(s)
        same = got == ref[s]
        ok &= same
        print(f"{s}: {'identical' if same else 'DIFFERENT'}")
    arch = blas_architecture()
    if not ok and arch not in BASELINE_BLAS:
        print(
            f"note: numpy's OpenBLAS uses {arch} kernels here; the below-ground settings are comparable only with "
            f"{' / '.join(BASELINE_BLAS)} kernels (run with OPENBLAS_CORETYPE=Haswell)"
        )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
