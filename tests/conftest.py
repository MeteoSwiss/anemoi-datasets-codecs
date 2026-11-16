"""Shared fixtures: an isolated cache / sidecar environment and a synthetic ICON-like lattice with a chunk of fields."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import numpy as np
import pytest

from anemoi_datasets_codecs import sidecar as sc
from anemoi_datasets_codecs.icon_lattice import TriangleLatticeLayout
from anemoi_datasets_codecs.icon_lattice.recipes import RECIPE_IDS

DATE = np.datetime64("2021-06-28T17:00:00")
STATIC = ("FIS", "sin_latitude", "cos_latitude", "sin_longitude", "cos_longitude")
VARS = [  # name, bound (None = lossless), clip
    ("T_2M", 0.05, None),  # before its cross sources PMSL and FIS in the field order
    ("PMSL", 2.0, None),
    ("WIDE", 1.0, None),  # int32 values
    ("CLCT", 0.5, (0.0, 100.0)),
    ("TOT_PREC_1H", 1e-5, None),
    ("NOISY", 0.01, None),  # white noise: escapes
    ("FLAT", 0.1, None),  # constant lossy field
    ("FIS", None, None),
    ("sin_latitude", None, None),
    ("cos_latitude", None, None),
    ("sin_longitude", None, None),
    ("cos_longitude", None, None),
    ("cos_local_time", None, None),
    ("sin_local_time", None, None),
    ("insolation", None, None),
    ("cos_julian_day", None, None),
    ("EXTRA", None, None),
]
NAMES = [v[0] for v in VARS]
LOSSLESS = {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1}

REPO_ROOT = Path(__file__).resolve().parents[1]
# the ICON-CH1 sample store (an anemoi dataset, 20 dates) and its codec configuration (variables, bounds, clip,
# static fields, grid); the real-data tests skip when the store is missing
SAMPLE_STORE = Path(
    os.environ.get("ANEMOI_DATASETS_CODECS_SAMPLE", str(REPO_ROOT / ".local/data/icon-ch1-sample.zarr"))
)
SAMPLE_CONFIG = Path(__file__).parent / "data" / "icon_ch1_sample.json"
# the ICON-CH1 grid file (netCDF), for the lattice-building test only
GRID_FILE = Path(os.environ.get("ANEMOI_DATASETS_CODECS_GRID_FILE", ""))


@pytest.fixture(autouse=True)
def _isolated(tmp_path_factory, monkeypatch):
    """Cache in a per-session temp dir, no search path from the environment, no registered stores."""
    monkeypatch.setenv(sc.CACHE_ENV, str(tmp_path_factory.getbasetemp() / "lattice-cache"))
    monkeypatch.delenv(sc.PATH_ENV, raising=False)
    monkeypatch.delenv("ANEMOI_DATASETS_CODECS_THREADS", raising=False)
    saved = list(sc._registered)
    sc._registered.clear()
    yield
    sc._registered[:] = saved


@pytest.fixture(scope="session")
def lattice():
    return make_lattice()


def make_lattice():
    """An irregular cut-out of a 128 x 200 lattice image (~18k cells) in shuffled native order, with lat/lon."""
    h, w = 128, 200
    r, c = np.indices((h, w))
    u, v = (r - h / 2) / (h / 2 - 2), (c - w / 2) / (w / 2 - 3)
    ang = np.arctan2(u, v)
    keep = (u**2 + v**2) < (0.93 + 0.05 * np.sin(3 * ang)) ** 2
    pix = np.flatnonzero(keep.ravel())
    rng = np.random.default_rng(0)
    pix = pix[rng.permutation(len(pix))]
    layout = TriangleLatticeLayout(pix.astype(np.int32), (h, w), meta={"name": "test-lattice"})
    rr, cc = pix // w, pix % w
    lat = 45.5 + 0.009 * rr
    lon = 5.8 + 0.005 * cc
    return layout, rr, cc, lat, lon


def smooth(rr, cc, seed, scale=20.0):
    rng = np.random.default_rng(seed)
    out = np.zeros(len(rr))
    for _ in range(5):
        k = rng.uniform(-1, 1, 2) / scale
        out += np.sin(k[0] * rr + k[1] * cc + rng.uniform(0, 6.3))
    return out


def forcings(lat, lon, date):
    d = date.astype(dt.datetime)
    hour = d.hour + d.minute / 60
    lt = 2 * np.pi * (hour + lon / 15) / 24
    jd = d - dt.datetime(d.year, 1, 1)
    a = (jd.days + jd.seconds / 86400) / 365.25 * 2 * np.pi
    dec = (
        0.396372
        - 22.91327 * np.cos(a)
        + 4.025430 * np.sin(a)
        - 0.387205 * np.cos(2 * a)
        + 0.051967 * np.sin(2 * a)
        - 0.154527 * np.cos(3 * a)
        + 0.084798 * np.sin(3 * a)
    )
    tc = 0.004297 + 0.107029 * np.cos(a) - 1.837877 * np.sin(a) - 0.837378 * np.cos(2 * a) - 2.340475 * np.sin(2 * a)
    la = np.radians(lat)
    ins = np.sin(np.radians(dec)) * np.sin(la) + np.cos(np.radians(dec)) * np.cos(la) * np.cos(
        np.radians((d.hour - 12) * 15 + lon + tc)
    )
    return {"cos_local_time": np.cos(lt), "sin_local_time": np.sin(lt), "insolation": np.maximum(ins, 0)}


@pytest.fixture(scope="session")
def chunk(lattice):
    return make_chunk(lattice)


def make_chunk(lattice):
    """The test chunk (17 variables, see VARS) on the lattice."""
    layout, rr, cc, lat, lon = lattice
    n = len(rr)
    rng = np.random.default_rng(1)
    fis = 3000 + 2500 * smooth(rr, cc, 1, 15)
    shared = rng.standard_normal(n)  # grid-scale signal common to PMSL and T_2M (what cross terms exploit)
    pmsl = 101000 + 400 * smooth(rr, cc, 2) + 10 * shared
    t2m = 285 - 0.0065 * fis / 9.81 + 3 * smooth(rr, cc, 3) + 0.03 * rng.standard_normal(n) + 0.5 * shared
    f = {
        "T_2M": t2m,
        "PMSL": pmsl,
        "WIDE": 2e5 * smooth(rr, cc, 4),
        "CLCT": np.clip(50 + 80 * smooth(rr, cc, 5, 8), 0, 100),
        "TOT_PREC_1H": np.maximum(0, smooth(rr, cc, 6, 6) - 1.5) * 1e-3,
        "NOISY": rng.standard_normal(n),
        "FLAT": np.full(n, 3.25),
        "FIS": fis,
        "sin_latitude": np.sin(np.radians(lat)),
        "cos_latitude": np.cos(np.radians(lat)),
        "sin_longitude": np.sin(np.radians(lon)),
        "cos_longitude": np.cos(np.radians(lon)),
        **forcings(lat, lon, DATE),
        "cos_julian_day": np.full(n, 0.25),
        "EXTRA": rng.standard_normal(n),
    }
    return np.stack([f[name] for name in NAMES]).astype(np.float32)


def make_config(layout, structure="tree", **options) -> dict:
    """The codec configuration of the test variables (JSON-able apart from ``grid``, a layout object)."""
    return {
        "variables": list(NAMES),
        "bounds": {n: b for n, b, _ in VARS if b is not None},
        "clip": {n: list(c) for n, _, c in VARS if c is not None},
        "static_fields": list(STATIC),
        "grid": layout,
        "structure": structure,
        "lossless": dict(LOSSLESS),
        "options": options,
    }


def make_codec(layout, structure="tree", **options):
    from anemoi_datasets_codecs.icon_lattice import LatticeCodec

    return LatticeCodec(**make_config(layout, structure, **options))


def check_bounds(dec, chunk, exact_lossless=True):
    for i, (name, bound, _) in enumerate(VARS):
        err = np.abs(dec[i].astype(np.float64) - chunk[i]).max()
        if bound is not None:
            assert err <= bound, (name, err, bound)
        elif name in RECIPE_IDS:
            assert err <= 1e-6, (name, err)
        elif exact_lossless:
            assert np.array_equal(dec[i].view(np.uint32), chunk[i].view(np.uint32)), name
