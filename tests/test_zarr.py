"""Zarr 3 store round trips (``icon_lattice`` as the array-to-bytes codec, ``icon_lattice_bytes`` after the
``bytes`` codec), the codec registry, and anemoi-datasets (``open_dataset`` on a store written with the codec; the
recipe-parse-time construction through ``output.compressors``).

The anemoi tests run in the environment with anemoi-datasets (``.venv-anemoi3``) and skip elsewhere.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import zarr
from conftest import DATE, NAMES, STATIC, check_bounds, make_codec, make_config

from anemoi_datasets_codecs import config, statics
from anemoi_datasets_codecs import sidecar as sc
from anemoi_datasets_codecs.icon_lattice import LatticeCodec, container, register_store
from anemoi_datasets_codecs.icon_lattice.container import MODE_STATIC_REF
from anemoi_datasets_codecs.icon_lattice.zarr3 import IconLatticeBytesCodec, IconLatticeCodec


def _data(chunk, n_dates=3):
    return np.stack([chunk] * n_dates)[:, :, None, :]


def _run_fresh(script_text, path, tmp_path, env_store=True, pipeline=True):
    """Run ``script_text`` (argv: store, output .npy) in a new process with a cold cache (and the store on
    ANEMOI_DATASETS_CODECS_PATH; ``pipeline=False``: without the pipeline that serves a local store's sidecar);
    returns (the saved array or None, the completed process)."""
    script = tmp_path / "read.py"
    script.write_text(script_text)
    env = {**os.environ, sc.CACHE_ENV: str(tmp_path / "cold"), sc.PATH_ENV: str(path) if env_store else ""}
    if not pipeline:
        env["ANEMOI_DATASETS_CODECS_PIPELINE"] = "off"
    out = tmp_path / "x.npy"
    r = subprocess.run([sys.executable, str(script), str(path), str(out)], env=env, capture_output=True, text=True)
    return (np.load(out) if r.returncode == 0 else None), r


READ_ZARR = "import sys, numpy as np, zarr\nnp.save(sys.argv[2], zarr.open_group(sys.argv[1], mode='r')['data'][1])\n"
READ_ANEMOI = (
    "import sys, numpy as np\nfrom anemoi.datasets import open_dataset\n"
    "np.save(sys.argv[2], np.asarray(open_dataset(sys.argv[1])[1]))\n"
)


def _codec_kw(kind, cfg):
    """create_array arguments of the two forms."""
    if kind == "serializer":
        return {"serializer": IconLatticeCodec(**cfg), "compressors": None}
    return {"compressors": [IconLatticeBytesCodec(**cfg)]}


@pytest.mark.parametrize("kind", ["serializer", "compressor"])
def test_zarr3_format3(tmp_path, lattice, chunk, kind):
    """``icon_lattice`` as the array-to-bytes codec, or ``icon_lattice_bytes`` after the bytes codec: same chunks."""
    path = tmp_path / "f3.zarr"
    cfg = make_config(lattice[0])
    data = _data(chunk)
    g = zarr.open_group(str(path), mode="w")
    register_store(path)
    a = g.create_array("data", shape=data.shape, chunks=(1, *data.shape[1:]), dtype="float32", **_codec_kw(kind, cfg))
    a[:] = data
    meta = json.loads((path / "data" / "zarr.json").read_text())
    names = [c["name"] for c in meta["codecs"]]
    assert names == (["icon_lattice"] if kind == "serializer" else ["bytes", "icon_lattice_bytes"])
    stamp = {"written_by": config.writer_stamp()}  # the adapter's writer stamp; no id
    assert meta["codecs"][-1]["configuration"] == {**LatticeCodec(**cfg).get_config(), **stamp}
    assert len(json.dumps(meta)) < 4096
    chunk_file = path / "data" / "c" / "1" / "0" / "0" / "0"
    enc = chunk_file.read_bytes()
    assert container.parse(enc).structure_sha == LatticeCodec(**cfg).structure.sha16
    assert enc == make_codec(lattice[0]).encode(data[1])  # the engine's bytes, whichever form
    statics.clear_caches()
    back = zarr.open_group(str(path), mode="r")["data"]
    check_bounds(back[2, :, 0], chunk)
    got, r = _run_fresh(READ_ZARR, path, tmp_path)
    assert got is not None, r.stderr
    check_bounds(got[:, 0], chunk)
    # without the pipeline and without the sidecar on the path: a clear error
    got, r = _run_fresh(READ_ZARR, path, tmp_path, env_store=False, pipeline=False)
    assert got is None and ("not found" in r.stderr or "not available" in r.stderr), r.stderr[-2000:]
    # the codec rejects arrays it cannot code (validated when the array is created)
    with pytest.raises(ValueError, match="float32|chunks"):
        g.create_array("bad", shape=(2, 7), chunks=(1, 7), dtype="float32", **_codec_kw(kind, cfg))
    with pytest.raises(ValueError, match=f"the array has {len(NAMES) + 1} variables per chunk, the configuration"):
        shape = (2, len(NAMES) + 1, 1, data.shape[-1])
        g.create_array("wrong_n", shape=shape, chunks=(1, *shape[1:]), dtype="float32", **_codec_kw(kind, cfg))
    with pytest.raises(ValueError, match="float32"):
        g.create_array("bad64", shape=data.shape, chunks=(1, *data.shape[1:]), dtype="float64", **_codec_kw(kind, cfg))


@pytest.mark.parametrize("kind", ["serializer", "compressor"])
def test_zarr3_corrupt_header_rejected(tmp_path, lattice, chunk, kind):
    """A chunk whose header asks for more fields than the chunk holds is an error, not a huge allocation: the
    adapters pass the chunk shape to the engine (decoder hardening F3)."""
    path = tmp_path / "bad.zarr"
    cfg = make_config(lattice[0])
    data = _data(chunk, 2)
    g = zarr.open_group(str(path), mode="w")
    a = g.create_array("data", shape=data.shape, chunks=(1, *data.shape[1:]), dtype="float32", **_codec_kw(kind, cfg))
    a[:] = data
    codec = LatticeCodec(**cfg)
    n = 1000 * codec.fields_per_date
    zero = np.float32(0).tobytes()
    forged = container.pack(
        [container.MODE_CONSTANT] * n, [0] * n, [zero] * n, np.zeros(1000, np.int64), codec.structure.sha16
    )
    (path / "data" / "c" / "1" / "0" / "0" / "0").write_bytes(forged)
    with pytest.raises(ValueError, match=f"header has {n} fields"):
        zarr.open_group(str(path), mode="r")["data"][1]


def test_zarr3_codec_registry(lattice):
    from zarr.registry import get_codec_class

    from anemoi_datasets_codecs.zarr3 import ChunkCodec

    assert get_codec_class("icon_lattice") is IconLatticeCodec
    assert get_codec_class("icon_lattice_bytes") is IconLatticeBytesCodec
    cfg = make_config(lattice[0])
    c = IconLatticeCodec(**cfg)
    assert isinstance(c.codec, LatticeCodec) and isinstance(c.codec, ChunkCodec)
    d = json.loads(json.dumps(c.to_dict()))
    assert IconLatticeCodec.from_dict(d) == c and d["name"] == "icon_lattice" and "id" not in d["configuration"]
    assert hash(IconLatticeCodec.from_dict(d)) == hash(c)
    # a configuration with a stray id (as written by early versions) still loads
    assert IconLatticeCodec.from_dict({**d, "configuration": {**d["configuration"], "id": "icon_lattice"}}) == c
    b = IconLatticeBytesCodec(**cfg)
    assert b.to_dict()["name"] == "icon_lattice_bytes" and b.configuration == c.configuration and b != c
    with pytest.raises(ValueError, match="expected a 'icon_lattice_bytes' codec"):
        IconLatticeBytesCodec.from_dict(d)
    assert "LatticeCodec(" in repr(c)


# --------------------------------------------------------------------------------------------- anemoi
def _anemoi_store(path, chunk, lattice, codec_kw, n_dates=3):
    """A minimal anemoi-datasets store in Zarr format 3 (data with the codec, dates, coordinates, statistics,
    attributes)."""
    _, _, _, lat, lon = lattice
    data = _data(chunk, n_dates)
    g = zarr.open_group(str(path), mode="w", zarr_format=3)
    register_store(path)
    dates = np.arange(DATE, DATE + np.timedelta64(n_dates, "h"), np.timedelta64(1, "h")).astype("datetime64[s]")

    def arr(name, x):
        a = g.create_array(name, shape=x.shape, dtype=x.dtype, chunks=x.shape if x.ndim else None)
        a[...] = x

    a = g.create_array("data", shape=data.shape, chunks=(1, *data.shape[1:]), dtype="float32", **codec_kw)
    a[:] = data
    arr("dates", dates)
    arr("latitudes", np.asarray(lat, np.float64))
    arr("longitudes", np.asarray(lon, np.float64))
    flat = data.astype(np.float64)
    for name, fn in (("mean", np.mean), ("stdev", np.std), ("minimum", np.min), ("maximum", np.max)):
        arr(name, fn(flat, axis=(0, 2, 3)))
    g.attrs.update(
        {
            "variables": list(NAMES),
            "frequency": "1h",
            "resolution": "test",
            "start_date": str(dates[0]),
            "end_date": str(dates[-1]),
            "ensemble_dimension": 1,
            "flatten_grid": True,
            "field_shape": [len(lat)],
            "version": "0.30",
            "data_request": {"grid": "test", "area": [0, 0, 0, 0], "param_level": {}, "param_step": {}},
            "constant_fields": list(STATIC),
            "variables_metadata": {n: {} for n in NAMES},
        }
    )
    return data


OLD_STAMP = "anemoi-datasets-codecs 0.0.1"


def test_writer_stamp_semantics(lattice):
    """``written_by``: a new codec gets the running version's stamp; a stored stamp is kept through from_dict /
    to_dict (and pickling); a configuration without one gets the running version's; the stamp takes part in no
    comparison and never reaches the engine; ``physics expand`` emits none."""
    import pickle

    from anemoi_datasets_codecs import chain, physics

    cfg = make_config(lattice[0])
    c = IconLatticeBytesCodec(**cfg)
    assert c.written_by == c.to_dict()["configuration"]["written_by"] == config.writer_stamp()
    assert config.writer_stamp().startswith("anemoi-datasets-codecs ")
    d = c.to_dict()
    old = {**d, "configuration": {**d["configuration"], "written_by": OLD_STAMP}}
    back = IconLatticeBytesCodec.from_dict(old)
    assert back.to_dict() == old and back == c and hash(back) == hash(c)
    assert pickle.loads(pickle.dumps(back)).to_dict() == old
    assert IconLatticeBytesCodec(**old["configuration"]).written_by == OLD_STAMP  # given as a keyword: kept too
    bare = {**d, "configuration": {k: v for k, v in d["configuration"].items() if k != "written_by"}}
    assert IconLatticeBytesCodec.from_dict(bare).written_by == config.writer_stamp()
    assert "written_by" not in back.codec.get_config()  # the engine never sees it
    assert LatticeCodec(**{**cfg, "written_by": OLD_STAMP}) == c.codec  # engines accept and ignore it
    entries = physics.expand(NAMES, None, {**cfg, "grid": c.codec.grid, "written_by": OLD_STAMP})
    assert not any("written_by" in e["configuration"] for e in entries)
    stamped = [chain.codec_class(e["name"])(**e["configuration"]).to_dict() for e in entries]
    assert all(e["configuration"]["written_by"] == config.writer_stamp() for e in stamped)
    other = [{**e, "configuration": {**e["configuration"], "written_by": OLD_STAMP}} for e in stamped[:1]]
    assert chain.chain_problems(other + stamped[1:]) == chain.chain_problems(entries) == []


def test_writer_stamp_kept_when_a_newer_version_appends(tmp_path, lattice, chunk):
    """A store created by an older version (its stamp) whose metadata lacks keys (rounding, recipe_fit), opened and
    resized by this version to append a date: the missing keys take this version's defaults, and zarr rewrites
    zarr.json with them filled in but the original stamp; the old chunks still decode, the new one is within the
    bounds. (So a store meant to keep its encoder settings must spell them out before a newer version appends.)"""
    path = tmp_path / "s.zarr"
    cfg = make_config(lattice[0])
    data = _data(chunk, 2)
    g = zarr.open_group(str(path), mode="w")
    a = g.create_array("data", shape=data.shape, chunks=(1, *data.shape[1:]), dtype="float32", **_codec_kw("x", cfg))
    a[:] = data
    meta_path = path / "data" / "zarr.json"
    meta = json.loads(meta_path.read_text())
    conf = meta["codecs"][-1]["configuration"]
    assert conf["written_by"] == config.writer_stamp()
    conf["written_by"] = OLD_STAMP
    del conf["options"]["rounding"], conf["physics"]["recipe_fit"]
    meta_path.write_text(json.dumps(meta))
    statics.clear_caches()
    a = zarr.open_group(str(path), mode="r+")["data"]
    a.resize((3, *data.shape[1:]))
    a[2] = data[0]
    conf = json.loads(meta_path.read_text())["codecs"][-1]["configuration"]
    assert conf["written_by"] == OLD_STAMP
    assert conf["options"]["rounding"] == "direct" and conf["physics"]["recipe_fit"] is True  # this version's
    back = zarr.open_group(str(path), mode="r")["data"]
    for d in range(3):
        check_bounds(back[d, :, 0], chunk)


def test_anemoi_open_dataset(tmp_path, lattice, chunk):
    """open_dataset on a format 3 store written with the array-to-bytes form, in this process and in a fresh one
    with a cold cache (as a training job would)."""
    pytest.importorskip("anemoi.datasets")
    from anemoi.datasets import open_dataset

    path = tmp_path / "anemoi.zarr"
    data = _anemoi_store(path, chunk, lattice, _codec_kw("serializer", make_config(lattice[0])))
    statics.clear_caches()
    ds = open_dataset(str(path))
    assert ds.variables == list(NAMES) and len(ds) == data.shape[0]
    check_bounds(np.asarray(ds[2])[:, 0], chunk)
    got, r = _run_fresh(READ_ANEMOI, path, tmp_path)
    assert got is not None, r.stderr[-2000:]
    check_bounds(got[:, 0], chunk)
    got, r = _run_fresh(READ_ANEMOI, path, tmp_path, env_store=False)  # nothing set: the pipeline serves the store
    assert got is not None, r.stderr[-2000:]
    check_bounds(got[:, 0], chunk)
    from anemoi_datasets_codecs.icon_lattice import store

    assert store.check(path)["problems"] == []


def test_anemoi_recipe_compressors_format3(tmp_path, lattice, chunk):
    """anemoi-datasets ``output.compressors`` in Zarr format 3 accepts bytes-to-bytes codecs only:
    ``icon_lattice_bytes`` (after the default bytes codec) is built at recipe-parse time from small JSON only (the
    grid by reference, no data; typos fail there), and open_dataset reads the store it writes."""
    pytest.importorskip("anemoi.datasets")
    try:
        from anemoi.datasets.create.recipe.output import GriddedOutput
    except ImportError:
        pytest.skip("anemoi-datasets without the recipe output model")
    if "compressors" not in GriddedOutput.model_fields:
        pytest.skip("anemoi-datasets without output.compressors")
    from anemoi.datasets import open_dataset

    from anemoi_datasets_codecs.icon_lattice import grids

    cfg = make_config(lattice[0])
    cfg["grid"] = grids.register_lattice(lattice[0])  # a reference, as written in a recipe
    spec = json.loads(json.dumps({"name": "icon_lattice_bytes", "configuration": cfg}))
    codecs = GriddedOutput(format="zarr3", compressors=[spec]).get_compressors(3)
    assert len(codecs) == 1 and isinstance(codecs[0], IconLatticeBytesCodec)
    assert codecs[0].configuration["grid"]["sha256"] == lattice[0].sha256
    with pytest.raises(Exception, match="match no variable"):
        bad = {**cfg, "bounds": {"T_2MM": 0.1}}
        GriddedOutput(format="zarr3", compressors=[{"name": "icon_lattice_bytes", "configuration": bad}])
    with pytest.raises(Exception, match="bytes-to-bytes"):  # the array-to-bytes form is not a compressor
        GriddedOutput(format="zarr3", compressors=[{"name": "icon_lattice", "configuration": cfg}])
    path = tmp_path / "f3.zarr"
    _anemoi_store(path, chunk, lattice, {"compressors": codecs})
    statics.clear_caches()
    check_bounds(np.asarray(open_dataset(str(path))[1])[:, 0], chunk)


class _ToyEngine:
    """A minimal engine for the generic adapters: float32 values with a 4-byte length prefix; it keeps the last
    decoded array so that a test can check what zarr receives."""

    names = ("a",)
    fields_per_date = 1

    def __init__(self, **cfg):
        self.last = None

    def encode(self, x):
        return np.uint32(x.size).tobytes() + x.astype("<f4").tobytes()

    def decode(self, buf, out=None):
        self.last = np.frombuffer(bytes(buf)[4:], "<f4").copy()
        return self.last

    def get_config(self):
        return {}

    def validate_chunk_shape(self, chunks):
        pass


def test_zarr3_adapters_no_copy():
    """Both adapters hand zarr the engine's decoded array without copying it, and the engine's input on encode is a
    view of zarr's chunk buffer."""
    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.dtype import parse_dtype

    from anemoi_datasets_codecs.zarr3 import ChunkArrayBytesCodec, ChunkBytesBytesCodec

    class ToyArray(ChunkArrayBytesCodec):
        codec_name, engine = "toy", _ToyEngine

    class ToyBytes(ChunkBytesBytesCodec):
        codec_name, engine = "toy_bytes", _ToyEngine

    proto = default_buffer_prototype()
    spec = ArraySpec((2, 3), parse_dtype("float32", zarr_format=3), 0.0, ArrayConfig.from_dict({}), proto)
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    a = ToyArray()
    enc = a._encode_sync(proto.nd_buffer.from_numpy_array(x), spec)
    got = a._decode_sync(enc, spec).as_ndarray_like()
    assert np.array_equal(got, x) and np.shares_memory(got, a.codec.last)
    b = ToyBytes()
    raw = proto.buffer.from_array_like(x.reshape(-1).view(np.uint8))
    enc_b = b._encode_sync(raw, spec)
    assert enc_b.to_bytes() == enc.to_bytes()
    got_b = b._decode_sync(enc_b, spec).as_array_like()
    assert got_b.dtype == np.uint8 and np.shares_memory(got_b, b.codec.last)
    assert np.array_equal(got_b.view("<f4"), x.reshape(-1))


class _ToyChainEngine(_ToyEngine):
    """The toy engine in chain mode (two fields per date), decoding into ``out`` when given."""

    names = ("a", "b")
    fields_per_date = 2
    chain = True

    def decode(self, buf, out=None):
        v = np.frombuffer(bytes(buf)[4:], "<f4")
        if out is None:
            out = v.copy()
        else:
            out.reshape(-1)[...] = v
        self.last = out
        return out


def _toy_stage():
    from anemoi_datasets_codecs.zarr3 import FieldStage

    class ToyStage(FieldStage):
        """In place: field 0 shifted by 1; records the arrays it was given."""

        def __init__(self, variables=("a", "b")):
            super().__init__(variables)
            self.seen = {}

        def encode(self, x):
            self.seen["encode"] = x
            y = x.copy()
            y[:, 0] -= 1
            return y, b"h"

        def decode(self, y, header, out_shape, out=None):
            assert header == b"h" and out is None and y.flags.writeable
            y[:, 0] += 1
            self.seen["decode"] = y
            return y

    return ToyStage


def test_zarr3_envelope_path_no_copy():
    """In a chain [stage, stage, chain-mode chunk engine] the stages get read-only views of zarr's buffer on encode,
    and on decode nothing is copied: the chunk engine decodes into the envelope it hands on, an in-place stage
    re-labels that envelope, and the first stage hands zarr a byte view of the same memory."""
    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.dtype import parse_dtype

    from anemoi_datasets_codecs import envelope
    from anemoi_datasets_codecs.zarr3 import ChunkBytesBytesCodec, FieldStageBytesCodec

    ToyStage = _toy_stage()

    class Stage1(FieldStageBytesCodec):
        codec_name, engine = "toy_stage1", ToyStage

    class Stage2(FieldStageBytesCodec):
        codec_name, engine = "toy_stage2", ToyStage

    class ToyChainBytes(ChunkBytesBytesCodec):
        codec_name, engine = "toy_chain_bytes", _ToyChainEngine

    proto = default_buffer_prototype()
    spec = ArraySpec((1, 2, 1, 3), parse_dtype("float32", zarr_format=3), 0.0, ArrayConfig.from_dict({}), proto)
    x = np.arange(6, dtype=np.float32).reshape(1, 2, 1, 3)
    s1, s2, last = Stage1(), Stage2(), ToyChainBytes()
    raw = proto.buffer.from_array_like(x.reshape(-1).view(np.uint8))
    e1 = s1._encode_sync(raw, spec)
    seen = s1.codec.seen["encode"]
    assert np.shares_memory(seen, x) and not seen.flags.writeable
    e2 = s2._encode_sync(e1, spec)
    assert np.shares_memory(s2.codec.seen["encode"], e1.as_numpy_array())
    stored = last._encode_sync(e2, spec)
    env = envelope.parse(stored.as_numpy_array())
    assert env.kind == envelope.KIND_BYTES and [h[0] for h in env.headers] == ["toy_stage1", "toy_stage2"]
    d2 = last._decode_sync(stored, spec)
    assert np.shares_memory(d2.as_numpy_array(), last.codec.last)  # decoded into the envelope
    d2_ro = proto.buffer.from_bytes(d2.to_bytes())  # read-only: the stage gets a copy to work in place on
    assert not d2_ro.as_numpy_array().flags.writeable
    back = s1._decode_sync(s2._decode_sync(d2_ro, spec), spec).as_array_like()
    assert np.array_equal(back.view("<f4"), x.reshape(-1))
    d1 = s2._decode_sync(d2, spec)
    assert np.shares_memory(d1.as_numpy_array(), last.codec.last)  # re-labelled in place
    got = s1._decode_sync(d1, spec).as_array_like()
    assert got.dtype == np.uint8 and np.shares_memory(got, last.codec.last)  # the bare chunk, a view
    assert np.array_equal(got.view("<f4"), x.reshape(-1))
    # a bare input to the chain-mode engine gives the bare engine bytes, as without chain mode
    assert last._encode_sync(raw, spec).to_bytes() == _ToyEngine().encode(x)


# -------------------------------------------------------------------------------- the pipeline that serves the store
def test_pipeline_serves_local_store_without_registration(tmp_path, lattice, chunk, monkeypatch):
    """No ``register_store``, no ``ANEMOI_DATASETS_CODECS_PATH``, cache off: writing a local store captures the lattice
    and the statics into ``<store>/.codecs`` (zarr tells the pipeline the store), and a fresh process with nothing
    set reads it back. With the pipeline off the same read fails with a clear error."""
    from anemoi_datasets_codecs.zarr3 import SIDECAR_PIPELINE, SidecarPipeline

    assert zarr.config.get("codec_pipeline.path") == SIDECAR_PIPELINE
    monkeypatch.setenv(sc.CACHE_ENV, "off")
    statics.clear_caches()
    path = tmp_path / "served.zarr"
    data = _data(chunk)
    g = zarr.open_group(str(path), mode="w")
    a = g.create_array(
        "data",
        shape=data.shape,
        chunks=(1, *data.shape[1:]),
        dtype="float32",
        **_codec_kw("compressor", make_config(lattice[0])),
    )
    pipeline = a._async_array.codec_pipeline
    assert isinstance(pipeline, SidecarPipeline) and pipeline.sidecar == path / sc.SIDECAR
    assert sc.serving_dir() is None  # only while the pipeline runs
    a[:] = data
    side = path / sc.SIDECAR
    assert (side / "grids" / f"{lattice[0].sha256}.npz").is_file()
    assert len(list((side / "statics").glob("*.f4.blosc"))) == len(STATIC)
    assert (
        container.inspect_chunk((path / "data" / "c" / "0" / "0" / "0" / "0").read_bytes())[NAMES.index("FIS")].mode
        == MODE_STATIC_REF
    )
    env = {k: v for k, v in os.environ.items() if k != sc.PATH_ENV}
    env[sc.CACHE_ENV] = "off"
    script = tmp_path / "read.py"
    script.write_text(READ_ZARR)
    out = tmp_path / "x.npy"
    subprocess.run([sys.executable, str(script), str(path), str(out)], check=True, env=env)
    check_bounds(np.load(out)[:, 0], chunk)
    r = subprocess.run(
        [sys.executable, str(script), str(path), str(out)],
        env={**env, "ANEMOI_DATASETS_CODECS_PIPELINE": "off"},
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0 and ("not found" in r.stderr or "not available" in r.stderr), r.stderr[-500:]


def test_pipeline_serves_each_store_its_own_sidecar(tmp_path, lattice, chunk, monkeypatch):
    """Two stores written in one process without registration: each gets the statics its chunks reference in its own
    sidecar (the served store is a context variable, not a process-global registration)."""
    monkeypatch.setenv(sc.CACHE_ENV, "off")
    statics.clear_caches()
    data = _data(chunk, 1)
    cfg = make_config(lattice[0])
    for name in ("a.zarr", "b.zarr"):
        g = zarr.open_group(str(tmp_path / name), mode="w")
        arr = g.create_array(
            "data", shape=data.shape, chunks=data.shape, dtype="float32", **_codec_kw("compressor", cfg)
        )
        arr[:] = data
    for name in ("a.zarr", "b.zarr"):
        assert len(list((tmp_path / name / sc.SIDECAR / "statics").glob("*.f4.blosc"))) == len(STATIC), name
    assert not sc._registered  # nothing was registered globally


def test_pipeline_leaves_other_stores_alone(lattice, chunk):
    """A store that is not a local directory gets no sidecar from the pipeline: the codecs work as before (here the
    cache of the test environment captures the statics)."""
    from zarr.storage import MemoryStore

    from anemoi_datasets_codecs.zarr3 import SidecarPipeline

    data = _data(chunk, 1)
    g = zarr.open_group(MemoryStore(), mode="w")
    a = g.create_array(
        "data", shape=data.shape, chunks=data.shape, dtype="float32", **_codec_kw("compressor", make_config(lattice[0]))
    )
    pipeline = a._async_array.codec_pipeline
    assert isinstance(pipeline, SidecarPipeline) and pipeline.sidecar is None
    a[:] = data
    check_bounds(a[0, :, 0], chunk)
