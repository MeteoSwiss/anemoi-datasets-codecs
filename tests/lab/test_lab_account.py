"""lab account / explain / diff on the synthetic chunk: the parts of every payload, the recorder, the explanations."""

from __future__ import annotations

import json

import numpy as np
import pytest
import zarr
from conftest import DATE, NAMES, SAMPLE_CONFIG, SAMPLE_STORE, make_chunk, make_codec, make_config

from anemoi_datasets_codecs import cli, envelope
from anemoi_datasets_codecs.icon_lattice import container, grids, native
from anemoi_datasets_codecs.lab import account


@pytest.fixture(scope="module")
def encoded(lattice):
    """The synthetic chunk encoded with the recorder: (codec, chunk bytes, Recorder.final)."""
    codec = make_codec(lattice[0])
    rec = account.Recorder(codec.structure, None, keep=True)
    buf = codec.encode(make_chunk(lattice), dates=DATE, record=rec)
    return codec, buf, rec.final(buf)


@pytest.fixture
def files(lattice, tmp_path):
    """A JSON configuration (grid by reference, lattice registered), a store of the chunk and its encoded chunk."""
    layout = lattice[0]
    grids.register_lattice(layout)
    cfg = {**make_config(layout), "grid": layout.ref()}
    cpath = tmp_path / "config.json"
    cpath.write_text(json.dumps(cfg))
    store = tmp_path / "s.zarr"
    g = zarr.open_group(str(store), mode="w")
    g.create_array("data", data=make_chunk(lattice)[None, :, None, :])
    g.create_array("dates", data=np.array([DATE], dtype="datetime64[s]"))
    return cpath, store


def test_hook_leaves_bytes_unchanged(lattice, encoded):
    codec, buf, _ = encoded
    assert codec.encode(make_chunk(lattice), dates=DATE) == buf


def test_parts_sum_to_the_chunk(encoded):
    codec, buf, _ = encoded
    for c in (codec, None):
        acc = account.account_chunk(buf, c)
        assert sum(acc["parts"].values()) == len(buf)
        assert abs(sum(acc["share"].values()) - 1) < 1e-12
        ch = container.parse(buf)
        for f in acc["fields"]:
            assert f["bytes"] == len(ch.payloads[f["index"]])
            if f["mode"] != "lattice":
                continue
            assert sum(f["parts"].values()) == f["bytes"]
            assert f["parts"]["tables"] == sum(f["table_bytes"])
            if c is None:
                assert "steps" not in f
                continue
            assert [s["n"] for s in f["steps"]] == [s.n for s in codec.structure.steps]
            assert f["serial_rans_bytes"] + sum(s["rans_bytes"] for s in f["steps"]) == f["parts"]["rans"]
            assert sum(s["esc_units"] for s in f["steps"]) == f["n_esc_units"]
    assert acc["fields"][NAMES.index("T_2M")]["field"] == "#0"  # no codec: no names
    named = account.account_chunk(buf, codec)
    assert named["fields"][NAMES.index("T_2M")]["field"] == "T_2M"
    assert named["modes"]["lattice"]["fields"] == 6


def test_chained_chunk_finds_the_container(encoded):
    codec, buf, _ = encoded
    env = envelope.pack_bytes((1, len(NAMES), 1, codec.n_cells), [("x", b"abc")], buf)
    acc = account.account_chunk(env.tobytes(), codec)
    assert acc["stages"] == ["x"]
    assert acc["parts"]["envelope"] == len(env) - len(buf)
    assert sum(acc["parts"].values()) == len(env)


def test_recorder_reproduces_the_streams(encoded):
    codec, buf, final = encoded
    S = codec.structure
    ch = container.parse(buf)
    lattice_fields = [i for i, m in enumerate(ch.modes) if m == container.MODE_LATTICE]
    assert sorted(final) == lattice_fields
    for i, (summary, pieces, mode, fb) in final.items():
        assert pieces["payload"] == bytes(ch.payloads[i]) and mode == container.MODE_LATTICE and fb == 0
        st = pieces["streams"]
        ser_sym, ser_tid, seg, streams = [], [], [0], []
        for s, sc in enumerate(pieces["steps"]):  # the table ids re-code the streams byte for byte
            tid = account.table_ids(sc, st, s).astype(np.uint8)
            rl = S.step_ranges(s)
            for lo, hi in rl:
                if len(rl) == 1:
                    ser_sym.append(sc.sym[lo:hi])
                    ser_tid.append(tid[lo:hi])
                    seg.append(seg[-1] + hi - lo)
                else:
                    one = np.array([0, hi - lo], np.uint64)
                    streams.append(native.rans_encode(sc.sym[lo:hi], tid[lo:hi], one, st.freqs, st.pbits))
        serial = native.rans_encode(
            np.concatenate(ser_sym), np.concatenate(ser_tid), np.array(seg, np.uint64), st.freqs, st.pbits
        )
        assert serial + b"".join(streams) == st.rans
        assert summary["bytes"] == len(ch.payloads[i])
        assert summary["actual"] == 8 * summary["parts"]["rans"]
        assert summary["model"] >= summary["h_ctx"] - 1e-6  # cross-entropy under the normalised tables
        assert summary["h_ctx"] > 0 and summary["h0"] > 0 and summary["h_fine"] > 0
        assert sum(s["n"] for s in summary["steps"]) == codec.n_cells
        assert summary["n_esc"] == sum(int(np.count_nonzero(sc.sym == 255)) for sc in pieces["steps"])
        assert abs(summary["escape_share"] - summary["parts"]["escapes"] / summary["bytes"]) < 1e-12


def test_residuals_of_round_trip(encoded):
    _, _, final = encoded
    for _, pieces, _, _ in final.values():
        for sc in pieces["steps"]:
            q = account.residuals_of(sc)
            z = np.where(q >= 0, 2 * q, -2 * q - 1)
            assert np.array_equal(np.minimum(z, 255), sc.sym)


def test_explain(encoded):
    codec, buf, _ = encoded
    rows = {r["field"]: r for r in account.explain_chunk(buf, codec)}
    assert any("zero rule" in w for w in rows["TOT_PREC_1H"]["why"])
    assert rows["T_2M"]["planned_cross"] and set(rows["T_2M"]["cross"]) <= set(rows["T_2M"]["planned_cross"])
    assert rows["FLAT"]["mode"] == "constant"
    assert rows["EXTRA"]["why"] == ["no bound: lossless variable"]
    assert rows["T_2M"]["table_layout"] in ("none", "shared", "per_step")
    # a lossy field stored losslessly because that was smaller
    ch = container.parse(buf)
    i = NAMES.index("NOISY")
    modes, fbs = ch.modes.tolist(), ch.fallbacks.tolist()
    payloads = [bytes(p) for p in ch.payloads]
    modes[i], fbs[i], payloads[i] = container.MODE_LOSSLESS, container.FALLBACK_SMALLER, b"\0" * 8
    other = container.pack(modes, fbs, payloads, ch.dates, ch.structure_sha)
    rows = {r["field"]: r for r in account.explain_chunk(other, codec)}
    assert any("lossless was smaller" in w for w in rows["NOISY"]["why"])


def test_diff(lattice, encoded):
    codec, buf, _ = encoded
    same = account.diff_chunks(buf, buf, codec)
    assert same["delta"] == 0 and all(r["delta"] == 0 for r in same["fields"]) and not same["mode_changes"]
    other = make_codec(lattice[0], k=2.0).encode(make_chunk(lattice), dates=DATE)
    d = account.diff_chunks(buf, other, codec)
    assert d["delta"] == len(other) - len(buf)
    head = d["parts"]["container header"]
    assert sum(r["delta"] for r in d["fields"]) == d["delta"] - (head[1] - head[0])
    assert abs(d["fields"][0]["delta"]) >= abs(d["fields"][-1]["delta"])


def test_cli(files, encoded, tmp_path, capsys):
    cpath, store = files
    chunk = tmp_path / "chunk.bin"
    assert cli.main(["lab", "account", "--encode", str(cpath), str(store), "--date", "0", "--save-chunk",
                     str(chunk), "--json"]) == 0  # fmt: skip
    out = json.loads(capsys.readouterr().out)
    assert out["chunk_bytes"] == chunk.stat().st_size == len(encoded[1])
    by = {f["field"]: f for f in out["fields"]}
    assert set(by) == {"T_2M", "PMSL", "WIDE", "CLCT", "TOT_PREC_1H", "NOISY"}
    assert out["totals"]["bytes"] == sum(f["bytes"] for f in out["fields"])
    assert len(out["profile"]) == len(encoded[0].structure.steps)
    assert cli.main(["lab", "account", "--encode", str(cpath), str(store), "--date", "0", "--fields", "T_2M,PMSL",
                     "--steps"]) == 0  # fmt: skip
    text = capsys.readouterr().out
    assert "T_2M" in text and "coder loss" in text and "NOISY" not in text
    assert cli.main(["lab", "account", str(chunk), "--config", str(cpath), "--steps", "--top", "3"]) == 0
    text = capsys.readouterr().out
    assert "parts:" in text and "serial stream" in text
    assert cli.main(["lab", "account", str(chunk), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["bytes"] == chunk.stat().st_size
    assert cli.main(["lab", "explain", str(chunk), "--config", str(cpath), "-v"]) == 0
    assert "zero rule" in capsys.readouterr().out
    assert cli.main(["lab", "diff", str(chunk), str(chunk), "--config", str(cpath)]) == 0
    assert "delta +0" in capsys.readouterr().out


@pytest.mark.realdata
@pytest.mark.skipif(not SAMPLE_STORE.exists(), reason="ICON-CH1 sample store missing")
def test_realdata_account_encode_two_fields():
    codec = account.codec_of(SAMPLE_CONFIG)
    acc = account.account_encode(codec, SAMPLE_STORE, 6, account.select_fields(codec, "T_2M,QV_850"))
    assert {f["field"] for f in acc["fields"]} == {"T_2M", "QV_850"}
    for f in acc["fields"]:
        assert f["mode"] == "lattice" and 0 <= f["coder_loss"] < 0.05
