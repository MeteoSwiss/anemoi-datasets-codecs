"""The oracles run on the synthetic chunk and give sane numbers; the baseline agrees with the coded size."""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import NAMES, SAMPLE_CONFIG, SAMPLE_STORE, make_codec, make_config

from anemoi_datasets_codecs import cli
from anemoi_datasets_codecs.icon_lattice import LatticeCodec
from anemoi_datasets_codecs.lab import entropy
from anemoi_datasets_codecs.lab import oracles as O
from anemoi_datasets_codecs.lab.oracles import core as C

SMOOTH = ("T_2M", "PMSL", "WIDE", "NOISY")


@pytest.fixture
def inputs(lattice, chunk):
    codec = make_codec(lattice[0])
    fields = C.select_fields(codec, None)
    return C.Inputs(codec, fields, chunk[[f.index for f in fields]].copy(), 0)


def _rows(doc):
    return {r["field"]: r for r in doc["fields"]}


def test_cost_bits_matches_entropy_module():
    rng = np.random.default_rng(0)
    r = np.rint(rng.laplace(0, 40, 20000)).astype(np.int64)
    ctx = rng.integers(0, 6, len(r))
    assert C.cost_bits(r, ctx) == pytest.approx(entropy.residual_cost_bits(r, ctx), rel=1e-12)
    assert C.cost_bits(r) == pytest.approx(entropy.residual_cost_bits(r), rel=1e-12)
    h = C.keyed_counts(entropy.symbols(r), ctx)
    assert (C.hist_bits(h, miller_madow=True) >= C.hist_bits(h)).all()


def test_select_fields(inputs):
    codec = inputs.codec
    assert [f.label for f in C.select_fields(codec, "PMSL,T_.*")] == ["T_2M", "PMSL"]
    assert [f.label for f in C.select_fields(codec, "3")] == ["CLCT"]
    with pytest.raises(SystemExit):
        C.select_fields(codec, "FIS")  # lossless: not an oracle field


def test_spatial_residuals_reproduce_values(inputs):
    S = inputs.codec.structure
    n = C.quantised(inputs.x[0], inputs.fields[0], 1.0)[2]
    sp = C.spatial(n, S, keep_pred=True)
    for st, sr in zip(S.steps, sp.steps, strict=True):
        assert np.array_equal(sr.pred + sr.r, n[st.t_idx])
        assert sr.bits <= max(sr.bits_w0, sr.bits_fit) + 1e-9
    assert sp.bits == pytest.approx(sum(C.cost_bits(s.r, s.ctx) for s in sp.steps))


def test_baseline_calibration(inputs):
    doc = O.run_baseline(inputs, "matched")
    assert "FLAT" in inputs.meta["skipped"]  # constant: the codec writes CONSTANT
    rows = _rows(doc)
    for name in SMOOTH:
        r = rows[name]
        assert r["mode_matched"] == "lattice"
        # below the coded size (tables, header, precision are not modelled), and not by much on smooth fields
        assert 0.85 < r["ratio"] < 1.0, (name, r["ratio"])
        assert r["bits"] <= r["bits_w0"] + 0.01  # the fitted weights do not lose against the defaults
    assert 0.85 < doc["total"]["ratio"] < 1.0


def test_baseline_honours_fit_fixed(lattice, chunk):
    """``options.fit`` ``fixed``: the default weights in every step, as the encoder; ``ls`` fits some."""
    choices = {}
    for fit in ("ls", "fixed"):
        codec = make_codec(lattice[0], fit=fit)
        fields = C.select_fields(codec, ",".join(SMOOTH))
        doc = O.run_baseline(C.Inputs(codec, fields, chunk[[f.index for f in fields]].copy(), 0), "none")
        choices[fit] = {c for r in doc["fields"] for c in r["choice"]}
        if fit == "fixed":
            assert all(r["bits"] == r["bits_w0"] == r["bits_fit"] for r in doc["fields"])
    assert "fit" in choices["ls"]
    assert "fit" not in choices["fixed"]


def test_linear_not_worse_than_sum_to_one(inputs):
    rows = _rows(O.run_linear(inputs))
    for name, r in rows.items():
        assert r["bits_best"] <= r["bits"] + 1e-9, name
        if name in SMOOTH:
            assert r["bits_lin"] <= 1.01 * r["bits_fit"] + 0.01, name


def test_cross_duplicate_field_gives_its_whole_cost(lattice, chunk):
    layout = lattice[0]
    cfg = make_config(layout)
    cfg["variables"] = [*NAMES, "COPY"]
    cfg["bounds"]["COPY"] = cfg["bounds"]["T_2M"]
    codec = LatticeCodec(**cfg)
    x = np.concatenate([chunk, chunk[:1]])
    fields = C.select_fields(codec, "T_2M,PMSL,NOISY,COPY")
    out = O.cross(x[[f.index for f in fields]], fields, codec, sample=0)
    lab = out["labels"]
    a, b = lab.index("COPY"), lab.index("T_2M")
    base = out["base"][a]
    assert out["gain_reg"][a][b] > 0.97 * base
    assert out["gain_cond"][a][b] > 0.5 * base
    # independent noise gains (almost) nothing
    assert out["gain_reg"][lab.index("NOISY")][b] < 0.02 * out["base"][lab.index("NOISY")]
    # the plan's term T_2M <- PMSL is marked, and the duplicate is the top pair not in the plan
    marks = {(p["target"], p["source"]): p["plan"] for p in out["in_plan"]}
    assert marks.get(("T_2M", "PMSL")) == "plan"
    assert {out["top_missing"][0]["target"], out["top_missing"][0]["source"]} == {"COPY", "T_2M"}
    # its reverse PMSL <- T_2M is labelled in the top pairs and never among the pairs missing from the plan
    assert {(p["target"], p["source"]): p["plan"] for p in out["top"]}.get(("PMSL", "T_2M")) == "reverse"
    assert ("PMSL", "T_2M") not in {(p["target"], p["source"]) for p in out["top_missing"]}


def test_temporal(inputs):
    same = O.run_temporal(inputs, inputs.x.copy())
    for r in same["fields"]:
        assert r["identical"] and r["bits_diff"] == 0.0 and r["bits_best"] == 0.0, r["field"]
    rng = np.random.default_rng(3)
    bounds = np.array([f.bound for f in inputs.fields])[:, None]
    prev = inputs.x + (0.3 * bounds * rng.standard_normal(inputs.x.shape)).astype(np.float32)
    clip = [i for i, f in enumerate(inputs.fields) if f.clip is not None]
    for i in clip:
        prev[i] = np.clip(prev[i], *inputs.fields[i].clip)
    for r in O.run_temporal(inputs, prev)["fields"]:
        assert r["bits_best"] <= r["bits"] + 1e-9
        assert r["bits_best"] < 0.5 * r["bits"], r["field"]  # a previous date this close predicts well


def test_contexts(inputs):
    base = _rows(O.run_baseline(inputs, "none"))
    doc = O.run_contexts(inputs, adaptive=True)
    for name, r in _rows(doc).items():
        b = r["bits"]
        assert b["spread8"] == pytest.approx(base[name]["bits"])
        assert b["order0"] >= b["step"] - 1e-9 >= b["spread8"] - 2e-9
        assert b["spread8xparent"] <= min(b["spread8"], b["parent"]) + 1e-9
        assert b["xz9e"] is not None and b["xz9e"] > 0
        assert r["charged"]["spread8"] > b["spread8"]


def test_levels(inputs):
    doc = O.run_levels(inputs)
    S = inputs.codec.structure
    assert sum(s["n"] for s in doc["steps"]) == S.n_cells
    assert sum(s["share"] for s in doc["steps"]) == pytest.approx(1.0)
    base = _rows(O.run_baseline(inputs, "none"))
    for r in doc["fields"]:
        assert r["bits"] == pytest.approx(base[r["field"]]["bits"])


def test_cli(lattice, chunk, tmp_path, capsys):
    import zarr

    codec = make_codec(lattice[0])
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(codec.get_config()))
    g = zarr.open_group(str(tmp_path / "s.zarr"), mode="w", zarr_format=3)
    a = g.create_array("data", shape=(2, len(NAMES), 1, chunk.shape[1]), dtype="float32")
    a[0, :, 0] = chunk
    a[1, :, 0] = chunk + np.float32(0.01) * (chunk != 0)
    g.create_array("dates", shape=(2,), dtype="int64")[:] = [0, 3600]
    store = str(tmp_path / "s.zarr")
    common = [str(cfg), store, "--fields", "T_2M,PMSL,CLCT", "--date", "1"]
    for oracle, extra in (
        ("baseline", []),
        ("linear", []),
        ("cross", ["--sample", "0"]),
        ("temporal", []),
        ("contexts", ["--no-adaptive"]),
        ("levels", []),
    ):
        assert cli.main(["lab", "oracles", oracle, *common, *extra, "--json"]) == 0
        doc = json.loads(capsys.readouterr().out)
        assert doc["oracle"] == oracle and doc["date"] == 1
        assert cli.main(["lab", "oracles", oracle, *common, *extra]) == 0
        assert f"lab oracles {oracle}" in capsys.readouterr().out
    assert cli.main(["lab", "oracles", "temporal", *common, "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["prev"] == 0 and doc["total"]["bits_best"] < doc["total"]["bits"]


@pytest.mark.realdata
def test_realdata_baseline_calibration():
    if not SAMPLE_STORE.exists():
        pytest.skip(f"sample store {SAMPLE_STORE} missing")
    inp, _, _ = C.load_inputs(str(SAMPLE_STORE), str(SAMPLE_CONFIG), 6, "T_500,PMSL,QV_850")
    doc = O.run_baseline(inp, "matched")
    for r in doc["fields"]:
        assert 0.93 < r["ratio"] < 1.0, r
