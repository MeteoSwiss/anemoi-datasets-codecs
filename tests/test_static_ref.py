"""The ``static_ref`` field stage: the engine alone (exact round trip, the frozen header, the inline fallback, missing
references), and the chain ``[static_ref, icon_lattice_bytes chain: true]`` through zarr on the synthetic lattice,
read back in a fresh process, with ``attach`` / ``check`` / ``info`` on the store."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import zarr
from conftest import NAMES, STATIC, check_bounds, make_config

from anemoi_datasets_codecs import chain, config, envelope, sidecar, statics
from anemoi_datasets_codecs import statics as pkg_statics
from anemoi_datasets_codecs.icon_lattice import container, register_store, store
from anemoi_datasets_codecs.icon_lattice.zarr3 import IconLatticeBytesCodec
from anemoi_datasets_codecs.static_ref import StaticRef
from anemoi_datasets_codecs.static_ref import header as H
from anemoi_datasets_codecs.static_ref.zarr3 import StaticRefCodec

KEPT = [n for n in NAMES if n not in STATIC]


def _data(chunk, n_dates=2, n_ens=1):
    """``(dates, variables, members, cells)``: the test chunk on every date and member, the non-static fields
    shifted per date and member so that only the statics repeat."""
    x = np.repeat(np.stack([chunk] * n_dates)[:, :, None, :], n_ens, axis=2)
    for d in range(n_dates):
        for e in range(n_ens):
            for i, n in enumerate(NAMES):
                if n not in STATIC:
                    x[d, i, e] += np.float32(0.01 * (d + 2 * e))
    return x


def _side(tmp_path):
    """A writable sidecar directory (the encoder's store target once registered)."""
    d = tmp_path / "side"
    d.mkdir(exist_ok=True)
    return d


def _same(a, b):
    return a.shape == b.shape and np.array_equal(np.asarray(a).view(np.uint32), np.asarray(b).view(np.uint32))


# ----------------------------------------------------------------------------------------------- engine
@pytest.mark.parametrize("n_ens", [1, 3])
def test_round_trip_exact(tmp_path, chunk, n_ens):
    """Drops the statics (the output has the other fields, in order), references them, and restores the chunk
    bit-identically into ``out`` (the decoder writes into the array it is given); repeats within a chunk are
    REPEAT slots."""
    register_store(_side(tmp_path))
    x = _data(chunk, 2, n_ens)
    x.flags.writeable = False  # the adapter hands the engine a read-only view
    st = StaticRef(NAMES, list(reversed(STATIC)))
    assert st.static_fields == STATIC and st.variables_out == tuple(KEPT) and st.fields_out == len(KEPT)
    y, h = st.encode(x)
    assert y.shape == (2, len(KEPT), n_ens, x.shape[-1])
    assert _same(y, x[:, [NAMES.index(n) for n in KEPT]])
    n_static, slots = H.parse(h)
    assert n_static == len(STATIC) and len(slots) == 2 * len(STATIC) * n_ens
    kinds = [s.kind for s in slots]
    assert kinds.count(H.REFERENCE) == len(STATIC) and kinds.count(H.INLINE) == 0  # each array once, then repeats
    assert len(h) == 16 + 40 * len(slots)
    assert StaticRef.header_references(h) == {"statics": {statics.sha256_of(chunk[NAMES.index(n)]) for n in STATIC}}
    statics.clear_caches()  # a fresh reader: from the sidecar
    out = np.full(x.shape, np.nan, np.float32)
    res = st.decode(y.copy(), h, x.shape, out=out)
    assert res is out and _same(res, x)
    assert _same(st.decode(y.copy(), h, x.shape), x)


def test_header_pinned_bytes(tmp_path):
    """The header format (version 1) is frozen: the bytes for a fixed input (with a store to reference into, and
    with ``inline: true``), and the decoding of those bytes."""
    x = (np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 1, 3) * 0.5).astype(np.float32)
    x[:, 1] = [1.0, -2.0, 0.25]  # "s": the same on both dates
    x[0, 3] = [7.0, 8.0, 9.0]  # "t": differs between the dates
    x[1, 3] = [7.0, 8.0, 9.5]
    sha_s = "92125c51c67d83217ab35ec0cb43b5f8b7cbca68d9faab504d0b160eeb196f53"
    sha_t0 = "b7605cfded9f92469402db20760841ee860433ba9ddd206ea3a44ca091def859"
    sha_t1 = "69365479bef5ce011586f2e9dd83b1c0479515c407752765b05d18e78f11550e"
    head = "01000000" + "02000000" + "04000000" + "00000000"  # version 1, 2 statics, 4 slots
    ref = bytes.fromhex(
        head
        + ("00000000" + "00000000" + sha_s)  # slot 0: s on date 0, REFERENCE
        + ("00000000" + "00000000" + sha_t0)  # slot 1: t on date 0, REFERENCE
        + ("02000000" + "00000000" + sha_s)  # slot 2: s on date 1, REPEAT of slot 0
        + ("00000000" + "00000000" + sha_t1)  # slot 3: t on date 1, REFERENCE
    )
    blob_s = "020196040c0000000c0000001c0000000000803f000000c00000803e"
    blob_t0 = "020196040c0000000c0000001c0000000000e0400000004100001041"
    blob_t1 = "020196040c0000000c0000001c0000000000e0400000004100001841"
    inline = bytes.fromhex(
        head
        + "010000001c000000"
        + sha_s
        + blob_s
        + "010000001c000000"
        + sha_t0
        + blob_t0
        + "0200000000000000"
        + sha_s
        + "010000001c000000"
        + sha_t1
        + blob_t1
    )
    names = ["a", "s", "b", "t"]
    register_store(_side(tmp_path))
    st = StaticRef(names, ["t", "s"])
    y, h = st.encode(x)
    assert h == ref and _same(y, x[:, [0, 2]])
    sidecar.unregister_store(_side(tmp_path))
    statics.clear_caches()
    y2, h2 = StaticRef(names, ["s", "t"], inline=True).encode(x)
    assert h2 == inline and _same(y2, y)
    statics.clear_caches()  # decoding the frozen bytes: inline needs nothing, the references the sidecar
    assert _same(st.decode(y.copy(), inline, x.shape), x)
    with sidecar.isolated(tmp_path / "side"):
        assert _same(st.decode(y.copy(), ref, x.shape), x)
    for bad in (b"\x02" + ref[1:], ref[:-1], ref + b"\0", ref[:16] + b"\x03" + ref[17:]):
        with pytest.raises(ValueError, match="static_ref header"):
            H.parse(bad)
    with pytest.raises(ValueError, match="slots"):
        StaticRef(["a", "s", "b", "t"], ["s"]).decode(y, ref, (2, 4, 1, 3))


def test_inline_fallback(tmp_path, chunk, monkeypatch):
    """A static that cannot be referenced is stored inline in the header, never passed on as a field: when there is
    nowhere to capture it (no store, cache off), and beyond ``MAX_VERSIONS`` arrays per name; ``inline: true`` for
    every static. Such chunks decode with no sidecar at all."""
    monkeypatch.setenv(sidecar.CACHE_ENV, "off")
    statics.clear_caches()
    x = _data(chunk, 1)
    st = StaticRef(NAMES, STATIC)
    with pytest.warns(UserWarning, match="could not be written to one.*stored inline in the static_ref header"):
        y, h = st.encode(x)
    assert y.shape[1] == len(KEPT) and {s.kind for s in H.parse(h)[1]} == {H.INLINE}
    assert StaticRef.header_references(h) == {"statics": set()}
    statics.clear_caches()
    assert _same(st.decode(y.copy(), h, x.shape), x)
    # version limit: with a store target, the fifth different FIS of one codec is stored inline
    monkeypatch.setattr(pkg_statics, "MAX_VERSIONS", 2)
    register_store(_side(tmp_path))
    st = StaticRef(NAMES, ["FIS"])
    kinds = []
    for v in range(3):
        xv = x.copy()
        xv[:, NAMES.index("FIS")] += np.float32(v)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            yv, hv = st.encode(xv)
        kinds.append(H.parse(hv)[1][0].kind)
        assert yv.shape[1] == len(NAMES) - 1
        if v == 2:
            assert any("more than 2 different values" in str(m.message) for m in w)
            statics.clear_caches()
            with sidecar.isolated(tmp_path / "nothing"):
                assert _same(st.decode(yv.copy(), hv, xv.shape), xv)
    assert kinds == [H.REFERENCE, H.REFERENCE, H.INLINE]
    y, h = StaticRef(NAMES, STATIC, inline=True).encode(x)
    assert {s.kind for s in H.parse(h)[1]} == {H.INLINE}


def test_missing_reference(tmp_path, chunk):
    """A referenced static that is not on the search path raises ``MissingReference`` naming the directories."""
    register_store(_side(tmp_path))
    x = _data(chunk, 1)
    st = StaticRef(NAMES, STATIC)
    y, h = st.encode(x)
    sidecar.unregister_store(_side(tmp_path))
    statics.clear_caches()
    with sidecar.isolated(tmp_path / "empty"):
        with pytest.raises(sidecar.MissingReference, match=r"static field [0-9a-f]{12} is not available; searched: "):
            st.decode(y.copy(), h, x.shape)


def test_configuration():
    """Configuration: canonical (statics in chunk order, ``inline`` always written), validated, through the Zarr
    adapter and the entry point."""
    st = StaticRef(["a", "s", "b", "t"], ["t", "s"])
    assert st.get_config() == {"variables": ["a", "s", "b", "t"], "static_fields": ["s", "t"], "inline": False}
    assert StaticRef(["a", "s", "b", "t"], ["t", "s"], written_by="anemoi-datasets-codecs 0.0.1") == st  # ignored
    assert StaticRef(**st.get_config()) == st and hash(StaticRef(**st.get_config())) == hash(st)
    assert StaticRef(["a", "s"], ["s"], inline=True).get_config()["inline"] is True
    for kw, msg in (
        ({"static_fields": ["x"]}, "not variables"),
        ({"static_fields": "s"}, "non-empty list"),
        ({"static_fields": []}, "non-empty list"),
        ({"static_fields": ["s", "s"]}, "duplicate"),
        ({"static_fields": ["s"], "inline": "yes"}, "inline must be"),
    ):
        with pytest.raises(ValueError, match=msg):
            StaticRef(["a", "s"], **kw)
    with pytest.raises(TypeError, match="bogus"):
        StaticRef(["a", "s"], ["s"], bogus=1)
    import pickle

    assert pickle.loads(pickle.dumps(st)) == st  # DataLoader workers, spawn
    codec = StaticRefCodec(variables=["a", "s"], static_fields=["s"])
    assert pickle.loads(pickle.dumps(codec)) == codec
    stamp = config.writer_stamp()
    want = {"variables": ["a", "s"], "static_fields": ["s"], "inline": False, "written_by": stamp}
    assert codec.to_dict() == {"name": "static_ref", "configuration": want}
    assert StaticRefCodec.from_dict({**codec.to_dict(), "configuration": {"id": "x", **codec.configuration}}) == codec
    import importlib.metadata

    eps = {e.name: e.value for e in importlib.metadata.entry_points(group="zarr.codecs")}
    assert eps["static_ref"] == "anemoi_datasets_codecs.static_ref.zarr3:StaticRefCodec"
    assert chain.is_stage("static_ref") and not chain.is_stage("icon_lattice_bytes") and not chain.is_stage("bytes")


# ----------------------------------------------------------------------------------------------- chain
def _lattice_cfg(layout):
    cfg = make_config(layout)
    cfg.update(variables=list(KEPT), static_fields=[], chain=True)
    cfg["bounds"] = {k: v for k, v in cfg["bounds"].items() if k in KEPT}
    cfg["clip"] = {k: v for k, v in cfg["clip"].items() if k in KEPT}
    return cfg


def _write_chain(path, layout, data, register):
    g = zarr.open_group(str(path), mode="w")
    if register:
        register_store(path)
    a = g.create_array(
        "data",
        shape=data.shape,
        chunks=(1, *data.shape[1:]),
        dtype="float32",
        compressors=[
            StaticRefCodec(variables=NAMES, static_fields=STATIC),
            IconLatticeBytesCodec(**_lattice_cfg(layout)),
        ],
    )
    a[:] = data
    g.attrs["variables"] = list(NAMES)


_READ = """
import sys, numpy as np, zarr
np.save(sys.argv[2], zarr.open_group(sys.argv[1], mode="r")["data"][...])
"""


def _fresh_read(path, out, env_path):
    """Read the array in a new process whose only sidecar source is ``ANEMOI_DATASETS_CODECS_PATH``."""
    import os

    env = {k: v for k, v in os.environ.items() if not k.startswith("ANEMOI_DATASETS_CODECS_")}
    env.update({sidecar.PATH_ENV: str(env_path), sidecar.CACHE_ENV: "off"})
    r = subprocess.run([sys.executable, "-c", _READ, str(path), str(out)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return np.load(out)


def _check_decoded(back, data):
    for d in range(data.shape[0]):
        check_bounds(back[d, :, 0], data[d, :, 0])


def test_chain_store_fresh_reader(tmp_path, lattice, chunk):
    """[static_ref, icon_lattice_bytes chain: true] written to a registered store: the chunks are envelopes with the
    static_ref header over a lattice container of the other fields, the store's sidecar holds the lattice and the
    statics, and a fresh process reads it with only ANEMOI_DATASETS_CODECS_PATH set (statics bit-identical, the rest
    within bounds). check passes; info shows the stages and the container."""
    path = tmp_path / "chain.zarr"
    data = _data(chunk, 2)
    _write_chain(path, lattice[0], data, register=True)
    meta = json.loads((path / "data" / "zarr.json").read_text())
    assert [c["name"] for c in meta["codecs"]] == ["bytes", "static_ref", "icon_lattice_bytes"]
    raw = (path / "data" / "c" / "1" / "0" / "0" / "0").read_bytes()
    env = envelope.parse(raw)
    assert env.kind == envelope.KIND_BYTES and env.shape == (1, len(KEPT), 1, data.shape[-1])
    assert [h[0] for h in env.headers] == ["static_ref"]
    assert {s.kind for s in H.parse(env.headers[0][1])[1]} == {H.REFERENCE}
    ch = container.parse(env.payload)
    assert len(ch.modes) == len(KEPT) and container.MODE_STATIC_REF not in ch.modes
    shas = {statics.sha256_of(chunk[NAMES.index(n)]) for n in STATIC}
    assert {p.name.split(".")[0] for p in (path / sidecar.SIDECAR / "statics").iterdir()} == shas
    assert store.static_refs(path / "data" / "c" / "1" / "0" / "0" / "0") == shas
    sidecar.unregister_store(path)
    statics.clear_caches()
    back = _fresh_read(path, tmp_path / "back.npy", path)
    _check_decoded(back, data)
    m = store.array_meta(path)
    assert m["codec"]["variables"] == KEPT and [e["name"] for e in m["chain"]] == ["static_ref", "icon_lattice_bytes"]
    r = store.check(path)
    assert r["problems"] == [] and r["chain"] == ["static_ref", "icon_lattice_bytes"]
    # info, through the command line
    chunk_file = next(store.chunk_paths(path))
    cli = [sys.executable, "-m", "anemoi_datasets_codecs.cli"]
    r = subprocess.run([*cli, "info", str(chunk_file), "--config", str(path), "-v"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "chain envelope" in out and "stage static_ref" in out and f"reference {len(STATIC)}" in out
    assert (
        "lattice container" in out and "FIS" in out and "PMSL" in out and "static_ref " not in out.split("lattice")[1]
    )
    r = subprocess.run([*cli, "info", str(chunk_file)], capture_output=True, text=True)
    assert r.returncode == 0 and "stage static_ref" in r.stdout, r.stderr
    from anemoi_datasets_codecs.cli import lattice_config, load_config

    cfgs = load_config(path)
    assert [e["name"] for e in cfgs] == ["static_ref", "icon_lattice_bytes"]
    assert lattice_config(cfgs)["variables"] == KEPT
    r = subprocess.run([*cli, "plan", str(path), "--json"], capture_output=True, text=True)
    assert r.returncode == 0 and "terms" in json.loads(r.stdout), r.stderr


def _strip_sidecar(path):
    """Move the store's sidecar files into the cache directory (the writer's side), as if the store had been copied
    without its sidecar (``anemoi-datasets copy``)."""
    cache = sidecar.cache_dir()
    for kind in ("grids", "statics"):
        d = path / sidecar.SIDECAR / kind
        for f in d.iterdir() if d.is_dir() else ():
            sidecar.write(cache, kind, f.name.split(".")[0], f.read_bytes())
    shutil.rmtree(path / sidecar.SIDECAR)


def test_chain_store_attach_and_check(tmp_path, lattice, chunk):
    """A store copied without its sidecar (the writer's cache still holds the files): check reports what the sidecar
    lacks, attach copies the stage's statics and the lattice into the store, then check passes and a fresh process
    reads the store alone. check compares the store's variables with the chain's first codec's and reports a
    chain whose codecs disagree on the variables."""
    from anemoi_datasets_codecs.icon_lattice import grids

    grids.save_lattice(lattice[0])
    path = tmp_path / "plain.zarr"
    data = _data(chunk, 2)
    _write_chain(path, lattice[0], data, register=False)
    _strip_sidecar(path)
    r = store.check(path)
    assert any("static" in p and "missing" in p for p in r["problems"])
    rep = store.attach(path)
    assert rep["statics"] == len(STATIC) and rep["missing"] == [] and rep["verified"]
    assert rep["chain"] == ["static_ref", "icon_lattice_bytes"] and rep["grid"]
    assert store.check(path)["problems"] == []
    statics.clear_caches()
    _check_decoded(_fresh_read(path, tmp_path / "back.npy", path), data)
    with sidecar.isolated(path):
        _check_decoded(store.decode_chunk_file(path, next(store.chunk_paths(path))), data[:1])
    zarr.open_group(str(path), mode="r+").attrs["variables"] = list(KEPT)  # the lattice's list is not the array's
    assert any("variables attribute" in p for p in store.check(path)["problems"])
    zarr.open_group(str(path), mode="r+").attrs["variables"] = list(NAMES)
    zj = path / "data" / "zarr.json"
    meta = json.loads(zj.read_text())
    meta["codecs"][1]["configuration"]["static_fields"] = ["FIS"]
    zj.write_text(json.dumps(meta))
    assert any("emits the variables" in p for p in store.check(path)["problems"])


def test_chain_decode_errors(tmp_path, lattice, chunk):
    """The engine walk refuses a chunk that is not an envelope and a stage stack that is not the chain's."""
    path = tmp_path / "c.zarr"
    _write_chain(path, lattice[0], _data(chunk, 1), register=True)
    m = store.array_meta(path)
    raw = next(store.chunk_paths(path)).read_bytes()
    with pytest.raises(ValueError, match="not a chain envelope"):
        chain.decode_chunk(m["chain"], b"\0" * 64, m["chunks"])
    with pytest.raises(ValueError, match="not the chain's stages"):
        chain.decode_chunk(m["chain"][1:], raw, m["chunks"])
    with pytest.raises(ValueError, match="not a codec of this package"):
        chain.build({"name": "bytes", "configuration": {}})
    assert _same(chain.decode_chunk(m["chain"], raw, m["chunks"]), store.decode_chunk_file(path, Path(raw_path(path))))


def raw_path(path):
    return next(store.chunk_paths(path))
