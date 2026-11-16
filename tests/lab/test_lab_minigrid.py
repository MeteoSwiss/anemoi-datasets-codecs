"""lab minigrid: window and alignment, crop, structures on the crop, roundtrip, build / check on a synthetic store, and
the ICON-CH1 sample (realdata)."""

from __future__ import annotations

import json
from fractions import Fraction

import numpy as np
import pytest
from conftest import DATE, SAMPLE_CONFIG, SAMPLE_STORE, check_bounds, make_codec, make_config

from anemoi_datasets_codecs import sidecar
from anemoi_datasets_codecs.icon_lattice import grids
from anemoi_datasets_codecs.lab import minigrid as M


@pytest.fixture(scope="module")
def full_structures(lattice):
    return M.build_structures(lattice[0])


def test_parse():
    assert M.parse_window(" 0:64, 32:96 ") == M.Window(0, 64, 32, 96)
    assert str(M.Window(0, 64, 32, 96)) == "0:64,32:96"
    with pytest.raises(ValueError):
        M.parse_window("0:64")
    assert M.parse_fraction("1/16") == Fraction(1, 16) and M.parse_fraction(0.25) == Fraction(1, 4)
    assert M.fraction_label(Fraction(1, 16)) == "1-16"
    with pytest.raises(ValueError):
        M.parse_fraction("2")


def test_alignment_and_default_window(lattice):
    layout = lattice[0]
    assert M.tree_levels(layout) == 2  # ~16k cells: 2 levels above the 2000-cell coarsest rule
    assert M.alignment(layout) == 32  # max(2 ** (2 + 1), 32 for hc2_k3)
    for f in ("1/4", "1/2"):
        w = M.default_window(layout, M.parse_fraction(f))
        assert w.r0 % 32 == 0 and w.c0 % 32 == 0
        M.check_window(layout, w, 32)
        mini, _ = M.crop(layout, w)
        assert 0.5 < mini.n_cells / (float(Fraction(f)) * layout.n_cells) < 2
    with pytest.raises(ValueError, match="multiples of 32"):
        M.check_window(layout, M.Window(16, 80, 32, 96), 32)
    with pytest.raises(ValueError, match="inside"):
        M.check_window(layout, M.Window(64, 160, 32, 96), 32)
    with pytest.raises(ValueError, match=">= 32"):
        M.check_window(layout, M.Window(32, 48, 32, 96), 32)


def test_crop(lattice):
    layout, rr, cc, *_ = lattice
    w = M.Window(32, 96, 32, 160)
    mini, cells = M.crop(layout, w, "test-mini")
    inside = (rr >= 32) & (rr < 96) & (cc >= 32) & (cc < 160)
    assert sorted(cells.tolist()) == np.flatnonzero(inside).tolist()
    assert cells.dtype == np.int32 and mini.shape == (64, 128)
    r, c = mini.rows_cols()
    assert np.array_equal(r + 32, rr[cells]) and np.array_equal(c + 32, cc[cells])
    assert mini.meta["window"] == [32, 96, 32, 160] and mini.meta["parent_sha256"] == layout.sha256
    assert mini.ref()["name"] == "test-mini"


@pytest.mark.parametrize("fraction", ["1/2", "1/4"])
def test_structures_on_crop(lattice, full_structures, fraction):
    layout = lattice[0]
    mini, cells = M.crop(layout, M.default_window(layout, M.parse_fraction(fraction)))
    ms = M.build_structures(mini)  # validated
    cmp = M.compare_structures(full_structures, ms)
    for sid, c in cmp.items():
        assert c["same_kinds"] and c["same_prediction"], (sid, c["kinds_only_full"], c["kinds_only_mini"])
        assert c["mini"]["n_cells"] == mini.n_cells
    if fraction == "1/2":  # big enough to keep every level
        assert all(c["mini"]["levels"] == c["full"]["levels"] for c in cmp.values())


def test_alignment_keeps_refinement_levels(lattice):
    """Aligned: every refinement level of the crop is the full grid's level inside the window. Shifted by 2 columns
    (not a multiple of 4), the level-1 cells of the crop are not the full grid's."""
    layout = lattice[0]
    full_levels = M.level_cells(layout, 2)
    for w, aligned in ((M.Window(32, 96, 32, 160), True), (M.Window(32, 96, 34, 162), False)):
        mini, cells = M.crop(layout, w)
        sub = [set(cells[x].tolist()) <= set(f.tolist()) for x, f in zip(M.level_cells(mini, 2), full_levels)]
        assert sub[0]
        assert all(sub) if aligned else not sub[1]


def test_roundtrip_on_crop(lattice, chunk):
    layout = lattice[0]
    mini, cells = M.crop(layout, M.default_window(layout, M.parse_fraction("1/2")))
    x = np.ascontiguousarray(chunk[:, cells])
    for sid in ("tree", "hc2_k3"):
        codec = make_codec(mini, sid)
        enc = codec.encode(x, dates=[DATE])
        check_bounds(np.asarray(codec.decode(enc)).reshape(x.shape), x)
        r = M.roundtrip(codec, x, DATE)
        assert r["bad"] == [] and r["nbytes"].sum() == len(r["enc"])


def _synthetic_store(path, lattice, chunk):
    import zarr

    layout, rr, cc, lat, lon = lattice
    g = zarr.open_group(str(path), mode="w", zarr_format=3)
    x = np.stack([chunk, chunk])[:, :, None, :]
    x[1, :7] += np.float32(0.25)  # date 1: other values of the first (lossy) variables, same statics
    d = g.create_array("data", shape=x.shape, chunks=(1, *x.shape[1:]), dtype="float32", compressors=M._blosc())
    d[...] = x
    dates = np.array([DATE, DATE + np.timedelta64(1, "h")], "datetime64[s]")
    g.create_array("dates", shape=(2,), dtype=dates.dtype)[...] = dates
    for name, v in (("latitudes", lat), ("longitudes", lon)):
        g.create_array(name, shape=v.shape, dtype="float64")[...] = v
    g.create_array("mean", shape=(x.shape[1],), dtype="float64")[...] = np.arange(x.shape[1], dtype=float)
    g.attrs.update({"field_shape": str([layout.n_cells]), "flatten_grid": "True"})
    return x


def test_build_and_check(lattice, chunk, tmp_path):
    layout = lattice[0]
    grids.register_lattice(layout)
    store = tmp_path / "full.zarr"
    x = _synthetic_store(store, lattice, chunk)
    cfg = {**make_config(layout), "grid": layout.ref()}
    out = tmp_path / "mini"
    logs: list[str] = []
    m = M.build(cfg, store, out, fraction="1/2", log=logs.append)
    assert m["align"] == 32 and m["dates"] == [0, 1]
    cells = np.load(out / "cells.npy")
    mini_cfg = json.loads((out / "mini.json").read_text())
    assert {k: v for k, v in mini_cfg.items() if k != "grid"} == {k: v for k, v in cfg.items() if k != "grid"}
    assert mini_cfg["grid"]["name"] == "test-lattice-mini-1-2" and mini_cfg["grid"]["n_cells"] == len(cells)
    assert all(s["same_kinds"] for s in m["structures"].values())

    import zarr

    g = zarr.open_group(str(out / "mini.zarr"), mode="r")
    assert g["data"].shape == (2, x.shape[1], 1, len(cells)) and g["data"].chunks == (1, x.shape[1], 1, len(cells))
    assert np.array_equal(g["data"][1], x[1][..., cells])
    assert np.array_equal(g["latitudes"][:], lattice[3][cells]) and np.array_equal(g["mean"][:], np.arange(x.shape[1]))
    assert g.attrs["field_shape"] == str([len(cells)]) and g.attrs["flatten_grid"] == "True"
    assert g.attrs["minigrid"]["window"] == m["window"]
    for sc in (out / "mini.zarr" / sidecar.SIDECAR, out / "lattice"):
        assert (sc / "grids" / f"{mini_cfg['grid']['sha256']}.npz").is_file()
        assert len(list((sc / "statics").iterdir())) == 5
    assert "ANEMOI_DATASETS_CODECS_PATH=" in logs[-1]

    with pytest.raises(FileExistsError):
        M.build(cfg, store, out, fraction="1/2", log=logs.append)

    res = M.check(out, date=1, full=store, log=logs.append)
    assert res["ok"] and res["bad"] == [], logs
    assert 0.5 < res["ratio_lossy"] < 2

    assert M.use(out) == mini_cfg  # registers the mini store: the lattice resolves from it
    assert grids.load_lattice(mini_cfg["grid"]).n_cells == len(cells)


@pytest.mark.realdata
def test_sample_minigrid(tmp_path):
    if not (SAMPLE_STORE / "zarr.json").is_file():
        pytest.skip(f"sample store {SAMPLE_STORE} missing")
    out = tmp_path / "mini"
    m = M.build(SAMPLE_CONFIG, SAMPLE_STORE, out, fraction="1/64", dates="6", log=lambda *a: None)
    assert m["align"] == 64 and all(s["same_prediction"] for s in m["structures"].values())
    assert m["structures"]["hc2_k3"]["same_kinds"]
    res = M.check(out, date=0, log=lambda *a: None)
    assert res["ok"], res["bad"]
