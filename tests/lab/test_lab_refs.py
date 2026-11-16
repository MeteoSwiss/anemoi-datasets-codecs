"""lab refs on the synthetic lattice: the lossless references, the image layout, zfp / SZ3 when installed."""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pytest
from conftest import NAMES, VARS, make_codec, make_config

from anemoi_datasets_codecs.icon_lattice import container, encoder
from anemoi_datasets_codecs.lab import refs

INSTALLED = {m: importlib.util.find_spec(mod) is not None for m, mod in (("zfp", "zfpy"), ("sz3", "pysz"))}


@pytest.fixture(scope="module")
def image(lattice):
    return refs.Image.of(lattice[0])


def test_image_places_cells_and_fills_holes(lattice, image):
    layout, rr, cc, _, _ = lattice
    v = np.arange(layout.n_cells, dtype=np.float32)
    img = image.put(v)
    assert img.shape == tuple(layout.shape)
    assert np.array_equal(image.take(img), v)  # the cells are untouched
    assert set(np.unique(img)) <= set(v)  # holes hold the value of some cell
    occupied = np.zeros(img.shape, bool)
    occupied[rr, cc] = True
    assert (~occupied).any()  # the test lattice has holes


def test_int_dtype():
    assert refs.int_dtype(np.array([-32768, 32767])) == np.int16
    assert refs.int_dtype(np.array([0, 32768])) == np.int32


@pytest.mark.parametrize("field", ["T_2M", "CLCT", "TOT_PREC_1H", "WIDE"])
def test_lossless_references(chunk, image, field):
    i = NAMES.index(field)
    _, bound, clip = VARS[i]
    x = chunk[i]
    out = refs.measure_field(x, bound, 1.0, clip, image, ["zstd", "xz", "blosc-lz4", "blosc-zstd", "blosc-lz4-f32"])
    assert len(out) == 10  # 5 methods x 2 layouts
    for r in out:
        assert "error" not in r, r
        assert r["bytes"] > 0
        assert r["max_error"] <= bound  # the codec's own error for the integers, 0 for float32
    assert {r["layout"] for r in out} == {"1d", "2d"}
    # all integer references see the same error; the float reference is exact
    assert len({r["max_error"] for r in out if r["method"] != "blosc-lz4-f32"}) == 1
    assert all(r["max_error"] == 0 for r in out if r["method"] == "blosc-lz4-f32")


def test_integers_match_quantise(chunk, image):
    i = NAMES.index("T_2M")
    x = chunk[i]
    _, step, n = encoder.quantise(x, 0.05, 1.0, None, 0)
    (r,) = refs.measure_field(x, 0.05, 1.0, None, image, ["xz"], ("1d",))
    ref, _ = refs.METHODS["xz"].run(n.astype(refs.int_dtype(n)))
    assert r["bytes"] == ref


def test_lossy_unavailable_is_reported(chunk, image, monkeypatch):
    monkeypatch.setitem(refs.METHODS, "zfp", refs.Method("zfp", "lossy", refs._zfp, ("no_such_module_zfpy",)))
    (r,) = refs.measure_field(chunk[0], 0.05, 1.0, None, image, ["zfp"], ("1d",))
    assert r["error"].startswith("not installed")


def test_run_without_scipy_gives_1d(lattice, chunk, tmp_path, monkeypatch, capsys):
    """No scipy: the 2d records carry the error, the 1d ones are measured."""
    import zarr

    store = tmp_path / "s.zarr"
    zarr.open_group(str(store), mode="w").create_array("data", data=chunk[None, :, None, :])
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.ndimage", None)
    rows = refs.run(str(store), make_config(lattice[0]), 0, ["T_2M"], ["zstd", "blosc-lz4-f32"])
    one = [r for r in rows if r["layout"] == "1d"]
    two = [r for r in rows if r["layout"] == "2d"]
    assert len(one) == 2 and all(r["bytes"] > 0 for r in one)
    assert len(two) == 2 and all(r["error"] == "not installed (scipy)" for r in two)
    assert "n/i" in refs.format_tables(rows)


def test_retry_halves_the_tolerance(chunk, image, monkeypatch):
    calls = []

    def sloppy(a, tol):  # honours the bound only from the third try on
        calls.append(tol)
        err = 3.0 * tol if len(calls) < 3 else 0.5 * tol
        return 100, a + np.float32(err)

    monkeypatch.setitem(refs.METHODS, "zfp", refs.Method("zfp", "lossy", sloppy))
    (r,) = refs.measure_field(chunk[0], 0.05, 1.0, None, image, ["zfp"], ("1d",))
    assert r["halvings"] == 2 and r["tolerance"] == pytest.approx(0.0125) and r["max_error"] <= 0.05


NO_REFERENCE = pytest.param("", marks=pytest.mark.skip(reason="no reference compressor installed (the lab extra)"))


@pytest.mark.parametrize("method", [m for m in INSTALLED if INSTALLED[m]] or [NO_REFERENCE])
@pytest.mark.parametrize("field", ["T_2M", "TOT_PREC_1H"])
def test_lossy_references_meet_the_bound(chunk, image, method, field):
    i = NAMES.index(field)
    out = refs.measure_field(chunk[i], VARS[i][1], 1.0, VARS[i][2], image, [method])
    assert len(out) == 2
    for r in out:
        assert "error" not in r, r
        assert r["max_error"] <= VARS[i][1]
        assert r["bytes"] > 0


@pytest.mark.skipif(not INSTALLED["zfp"], reason="zfpy not installed")
def test_zfp_is_smaller_than_raw(chunk, image):
    i = NAMES.index("PMSL")
    out = refs.measure_field(chunk[i], 2.0, 1.0, None, image, ["zfp"])
    assert all(r["bytes"] < chunk[i].nbytes for r in out)


def test_run_end_to_end(lattice, chunk, tmp_path):
    """``run`` on a synthetic store: lattice bytes, references, cached chunk, tables."""
    import zarr

    layout = lattice[0]
    store = tmp_path / "s.zarr"
    g = zarr.open_group(str(store), mode="w")
    g.create_array("data", data=chunk[None, :, None, :])
    cfg = make_config(layout)
    lines = []
    rows = refs.run(
        str(store), cfg, 0, ["T_2M", "CLCT"], ["zstd", "blosc-lz4-f32"], out_dir=tmp_path / "out", log=lines.append
    )
    assert {r["variable"] for r in rows} == {"T_2M", "CLCT"}
    assert sum(r["method"] == "lattice" for r in rows) == 2
    assert all(r["env"]["package"] and r["bound"] > 0 for r in rows)
    assert all(r["ratio"] > 0 for r in rows if r["method"] != "lattice")
    # lattice bytes are those of the codec's own chunk
    codec = make_codec(layout)
    recs = container.inspect_chunk(codec.encode(chunk[None, :, None, :]))
    for r in rows:
        if r["method"] == "lattice":
            assert r["bytes"] == recs[NAMES.index(r["variable"])].nbytes
    assert len(list((tmp_path / "out").glob("refs-chunk-*.bin"))) == 1
    again = refs.run(str(store), cfg, 0, ["T_2M"], ["zstd"], out_dir=tmp_path / "out")  # served from the cache
    assert [r["bytes"] for r in again if r["method"] == "lattice"] == [
        r["bytes"] for r in rows if r["method"] == "lattice" and r["variable"] == "T_2M"
    ]
    text = refs.format_tables(rows)
    assert "layout 1d" in text and "layout 2d" in text and "TOTAL" in text
    with pytest.raises(SystemExit):
        refs.run(str(store), cfg, 0, ["FIS"], ["zstd"])  # not a lossy variable
    json.dumps(rows)
