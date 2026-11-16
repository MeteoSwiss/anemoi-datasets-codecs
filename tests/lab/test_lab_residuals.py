"""lab residuals on the synthetic store: the per-cell arrays reproduce the decoded chunk; the summary."""

from __future__ import annotations

import json

import numpy as np
import pytest
import zarr
from conftest import DATE, NAMES, SAMPLE_CONFIG, SAMPLE_STORE, VARS, make_chunk, make_config

from anemoi_datasets_codecs import cli
from anemoi_datasets_codecs.icon_lattice import grids
from anemoi_datasets_codecs.icon_lattice.encoder import reconstruct
from anemoi_datasets_codecs.lab import account, data, residuals

FIELDS = ("T_2M", "CLCT", "NOISY")


@pytest.fixture
def dumped(lattice, tmp_path):
    """(codec, store, npz path, meta) of the synthetic store's date 0, fields FIELDS."""
    layout = lattice[0]
    grids.register_lattice(layout)
    cfg = {**make_config(layout), "grid": layout.ref()}
    store = tmp_path / "s.zarr"
    g = zarr.open_group(str(store), mode="w")
    g.create_array("data", data=make_chunk(lattice)[None, :, None, :])
    g.create_array("dates", data=np.array([DATE], dtype="datetime64[s]"))
    codec = data.codec_for(cfg)
    out = tmp_path / "r.npz"
    meta = residuals.dump(codec, store, 0, account.select_fields(codec, ",".join(FIELDS)), out)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return codec, store, out, meta


def test_arrays_reproduce_the_decoded_values(dumped, lattice):
    codec, _, out, meta = dumped
    assert [f["field"] for f in meta["fields"]] == list(FIELDS)
    chunk = codec.encode(make_chunk(lattice), dates=DATE)
    dec = codec.decode(chunk).reshape(len(NAMES), -1)
    S = codec.structure
    with np.load(out) as z:
        for name in FIELDS:
            i = NAMES.index(name)
            v, pred, q, n = (z[f"{name}/{k}"].astype(np.int64) for k in ("v", "pred", "q", "n"))
            m = int(z[f"{name}/m"])
            assert np.array_equal(v, pred + (q << m))
            assert np.abs(n - v).max() <= (1 << m) // 2  # the closed loop stays within half a coarse step
            clip = VARS[i][2]
            assert np.array_equal(reconstruct(z[f"{name}/offset"], z[f"{name}/g"], v, clip), dec[i])
            step = z[f"{name}/step"]
            for s, st in enumerate(S.steps):
                assert (step[st.t] == s).all()
            assert (z[f"{name}/spread"][S.steps[0].t] == -1).all()
            assert float(z[f"{name}/bound"]) == VARS[i][1]
        assert tuple(z["grid/shape"]) == lattice[0].shape


def test_neighbour_pairs(lattice):
    layout, rr, cc, _, _ = lattice
    a, b = residuals.neighbour_pairs(layout.pixel_index, layout.shape)
    dr, dc = rr[b] - rr[a], cc[b] - cc[a]
    horiz = (dr == 0) & (dc == 1)
    vert = (dc == 0) & (dr == -1) & ((rr[a] + cc[a]) % 2 == 0)
    assert (horiz | vert).all()
    assert len(set(zip(a.tolist(), b.tolist(), strict=True))) == len(a)
    # every cell has at most 3 edge neighbours, interior cells exactly 3
    deg = np.bincount(np.concatenate([a, b]), minlength=layout.n_cells)
    assert deg.max() == 3 and np.mean(deg == 3) > 0.9


def test_summary(dumped):
    _, _, out, _ = dumped
    s = residuals.summary(out)
    by = {f["field"]: f for f in s["fields"]}
    for f in by.values():
        p = list(f["percentiles"].values())
        assert p == sorted(p) and p[-1] <= f["max_abs"]
        assert 0 <= f["zero_frac"] <= 1 and 0 <= f["escape_rate"] <= 1
        assert len(f["mean_abs_per_step"]) == len(s["meta"]["steps"])
        assert f["laplace_bits"] > 0 and f["h0_bits"] > 0
    assert by["NOISY"]["escape_rate"] > 0 and abs(by["NOISY"]["neighbour_corr"]) < 0.2  # white noise
    assert by["NOISY"]["laplace_scale"] > by["T_2M"]["laplace_scale"]


def test_cli(dumped, capsys, tmp_path):
    _, store, _, _ = dumped
    out = tmp_path / "cli.npz"
    assert cli.main(["lab", "residuals", "dump", str(tmp_path / "config.json"), str(store), "--date", "0",
                     "--fields", "PMSL", str(out)]) == 0  # fmt: skip
    assert "1 fields" in capsys.readouterr().out
    assert cli.main(["lab", "residuals", "summary", str(out)]) == 0
    assert "PMSL" in capsys.readouterr().out
    assert cli.main(["lab", "residuals", "summary", str(out), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["fields"][0]["field"] == "PMSL"


@pytest.mark.realdata
@pytest.mark.skipif(not SAMPLE_STORE.exists(), reason="ICON-CH1 sample store missing")
def test_realdata_residuals(tmp_path):
    codec = data.codec_for(SAMPLE_CONFIG)
    out = tmp_path / "r.npz"
    residuals.dump(codec, SAMPLE_STORE, 6, account.select_fields(codec, "T_2M"), out)
    f = residuals.summary(out)["fields"][0]
    assert f["field"] == "T_2M" and f["neighbour_corr_abs"] > 0
