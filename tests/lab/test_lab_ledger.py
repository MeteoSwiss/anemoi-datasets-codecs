"""The experiment ledger: run, sweep, compare, summary, pareto on a synthetic store."""

from __future__ import annotations

import argparse

import numpy as np
import pytest
import zarr
from conftest import DATE, NAMES, make_chunk, make_config

from anemoi_datasets_codecs.icon_lattice import container
from anemoi_datasets_codecs.lab import ledger

ROW_KEYS = {
    "run", "tag", "time", "env", "config_sha", "overrides", "store", "dated", "date_index", "encode_s", "decode_s",
    "total_bytes", "chunk_sha256", "field", "mode", "fallback", "bytes", "lossy", "bound", "max_err", "rmse",
    "max_err_over_bound", "rmse_over_bound", "exact",
}  # fmt: skip


@pytest.fixture(scope="module")
def setup(lattice, tmp_path_factory):
    d = tmp_path_factory.mktemp("ledger")
    chunk = make_chunk(lattice)
    store = d / "s.zarr"
    g = zarr.open_group(str(store), mode="w")
    arr = np.stack([chunk, chunk * np.float32(1.0)])[:, :, None, :]
    g.create_array("data", data=arr)
    g.create_array("dates", data=np.array([DATE, DATE + np.timedelta64(1, "h")], dtype="datetime64[s]"))
    cfg = make_config(lattice[0])
    return cfg, str(store), d


def test_overrides():
    cfg = {"options": {"k": 1}, "structure": "tree"}
    out = ledger.apply_overrides(cfg, {"options.k": 3, "physics.below_ground": False, "structure": "tree_b"})
    assert out == {"options": {"k": 3}, "structure": "tree_b", "physics": {"below_ground": False}}
    assert cfg == {"options": {"k": 1}, "structure": "tree"}
    assert ledger.parse_settings(["options.k=3", "structure=tree", "bounds.T_2M=0.1", "x=[0, null]"]) == {
        "options.k": 3, "structure": "tree", "bounds.T_2M": 0.1, "x": [0, None],
    }  # fmt: skip


def test_run_compare_summary(setup):
    cfg, store, d = setup
    led = d / "l.jsonl"
    ra = ledger.run(cfg, store, led, "0", tag="a", overrides={"options.k": 1}, keep_chunks=d / "chunks")
    rb = ledger.run(cfg, store, led, "0,1", tag="b", overrides={"options.k": 3}, dated=True)
    assert all(set(r) == ROW_KEYS for r in ra + rb)
    assert len(ra) == len(NAMES) and len(rb) == 2 * len(NAMES)
    assert ra[0]["config_sha"] != rb[0]["config_sha"]
    buf = next((d / "chunks").glob("a.*.d0.bin")).read_bytes()
    recs = container.inspect_chunk(buf)
    assert [r["bytes"] for r in ra] == [r.nbytes for r in recs]
    assert sum(r["bytes"] for r in ra) == ra[0]["total_bytes"] == len(buf)
    lossy = [r for r in ra if r["lossy"]]
    assert lossy and all(r["max_err_over_bound"] <= 1.0 for r in lossy)
    assert all(r["exact"] for r in ra if not r["lossy"] and r["mode"] != "recipe")
    assert len(ledger.read_ledger(led)) == len(ra) + len(rb)

    rows = ledger.read_ledger(led)
    with pytest.raises(SystemExit, match="valid dates"):
        ledger.compare(rows, "a", "b")  # one dated, one not
    ledger.run(cfg, store, led, "0", tag="c", overrides={"options.k": 3})
    rows = ledger.read_ledger(led)
    lines = ledger.compare(rows, "a", "c")
    tc = next(r["total_bytes"] for r in rows if r["tag"] == "c")
    assert f"A {len(buf)}  B {tc}  B-A {tc - len(buf):+d}" in "\n".join(lines)
    assert any("error B" in ln for ln in lines)
    ledger.run(cfg, store, led, "1", tag="d")
    with pytest.raises(SystemExit, match="no common date"):
        ledger.compare(ledger.read_ledger(led), "a", "d")
    assert len(ledger.summary_lines(rows)) == 3
    svg = ledger.pareto_svg(rows, "encode_s")
    assert svg.startswith("<svg") and ">a<" in svg


def test_tag_with_runid(setup):
    cfg, store, d = setup
    led = d / "t.jsonl"
    r1 = ledger.run(cfg, store, led, "0", tag="x")
    r2 = ledger.run(cfg, store, led, "0", tag="x", overrides={"options.k": 1})
    rows = ledger.read_ledger(led)
    assert ledger.select_run(rows, "x")[0]["run"] == r2[0]["run"]
    assert ledger.select_run(rows, "x@" + r1[0]["run"])[0]["run"] == r1[0]["run"]


def test_sweep_expand_and_skip(setup):
    cfg, store, d = setup
    spec = {"grid": {"options.k": [1, 3], "options.recon_bits": [0, 3]}, "dates": [0], "tag": "kr"}
    settings = ledger.expand_spec(spec)
    assert len(settings) == 4
    assert settings[0][0] == "kr/options.k=1,options.recon_bits=0"
    assert settings[3][1] == {"options.k": 3, "options.recon_bits": 3}
    small = {"settings": [{"options.k": 1}, {"options.k": 3}], "dates": [0], "tag": "s"}
    led = d / "sw.jsonl"
    msgs: list[str] = []
    res = ledger.sweep(cfg, store, small, led, log=msgs.append)
    assert res["ran"] == ["s/options.k=1", "s/options.k=3"] and not res["skipped"]
    res = ledger.sweep(cfg, store, small, led, log=msgs.append)
    assert not res["ran"] and len(res["skipped"]) == 2
    res = ledger.sweep(cfg, store, {**small, "settings": [{"options.k": 1}]}, led, force=True, log=msgs.append)
    assert res["ran"] == ["s/options.k=1"]
    assert len(ledger.runs_of(ledger.read_ledger(led))) == 3


def test_cli(setup, capsys):
    cfg, store, d = setup
    # the layout object is not JSON: the CLI is exercised on summary / compare / pareto of an existing ledger
    led = d / "cli.jsonl"
    ledger.run(cfg, store, led, "0", tag="a")
    ledger.run(cfg, store, led, "0", tag="b", overrides={"options.k": 1})
    p = argparse.ArgumentParser()
    ledger.add_parsers(p.add_subparsers(dest="lab_cmd", required=True))
    for argv in (
        ["summary", str(led)],
        ["compare", str(led), "a", "b", "--top", "3"],
        ["pareto", str(led), str(d / "p.svg"), "--y", "maxerr"],
    ):
        a = p.parse_args(["ledger", *argv])
        assert a.fn(a) == 0
    out = capsys.readouterr().out
    assert "total bytes" in out and (d / "p.svg").exists()
