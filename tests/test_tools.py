"""Kernel variants, threads, the store tools (attach / check) and the CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest
import zarr
from conftest import DATE, NAMES, STATIC, check_bounds, make_codec, make_config

from anemoi_datasets_codecs import config, statics
from anemoi_datasets_codecs import sidecar as sc
from anemoi_datasets_codecs.icon_lattice import native, store, threads
from anemoi_datasets_codecs.icon_lattice.zarr3 import IconLatticeCodec


# ----------------------------------------------------------------------------------------------- kernels
def test_scalar_and_avx2_decode_bit_identical(lattice, chunk):
    """The scalar variant (portable emulation of the intrinsics) decodes exactly what the AVX2 variant decodes, for
    every structure (group kernels, contexts, cross terms, recipes, escapes) and for the encoder's verification."""
    if "avx2" not in native.available_isas():
        pytest.skip("CPU without AVX2")
    try:
        for sid in ("tree", "tree_b", "hc2_k3"):
            codec = make_codec(lattice[0], sid)
            native.select_isa("avx2")
            enc_a = codec.encode(chunk, dates=[DATE])
            dec_a = codec.decode(enc_a)
            native.select_isa("scalar")
            assert native.isa() == "scalar"
            enc_s = codec.encode(chunk, dates=[DATE])
            dec_s = codec.decode(enc_a)
            assert enc_s == enc_a, sid
            np.testing.assert_array_equal(dec_s.view(np.uint32), dec_a.view(np.uint32))
            check_bounds(dec_s.reshape(chunk.shape), chunk)
    finally:
        native.select_isa("auto")
    with pytest.raises(ValueError):
        native.select_isa("avx512")


def test_isa_environment_override():
    env = {**os.environ, "ANEMOI_DATASETS_CODECS_ISA": "scalar"}
    r = subprocess.run(
        [sys.executable, "-c", "from anemoi_datasets_codecs.icon_lattice import native; print(native.isa())"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert r.stdout.strip() == "scalar"
    env["ANEMOI_DATASETS_CODECS_ISA"] = "foo"  # unknown: the best variant, with a warning
    r = subprocess.run(
        [sys.executable, "-c", "from anemoi_datasets_codecs.icon_lattice import native; print(native.isa())"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert r.stdout.strip() in ("avx2", "scalar") and "ANEMOI_DATASETS_CODECS_ISA='foo' is not available" in r.stderr


# ----------------------------------------------------------------------------------------------- threads
def test_default_threads(monkeypatch):
    monkeypatch.delenv(threads.THREADS_ENV, raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert threads.default_threads() == min(16, threads.available_cpus())
    monkeypatch.setenv(threads.THREADS_ENV, "3")
    assert threads.default_threads() == 3
    monkeypatch.setenv(threads.THREADS_ENV, "0")
    with pytest.raises(ValueError):
        threads.default_threads()
    monkeypatch.delenv(threads.THREADS_ENV)
    # inside a torch DataLoader worker: 1 (torch is only looked at when already imported)
    info = {"worker": object()}
    fake = types.ModuleType("torch")
    fake.utils = types.SimpleNamespace(data=types.SimpleNamespace(get_worker_info=lambda: info["worker"]))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert threads.default_threads() == 1
    info["worker"] = None  # the main process of a training job
    assert threads.default_threads() == min(16, threads.available_cpus())
    monkeypatch.setenv(threads.THREADS_ENV, "5")  # the environment wins
    info["worker"] = object()
    assert threads.default_threads() == 5


def test_decode_in_child_forked_after_threaded_decode():
    """GNU libgomp's thread pool does not survive fork(): a child forked after the parent decoded with several
    threads must decode with 1 thread (it would hang otherwise), even when asked for more."""
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "fork_check.py")
    r = subprocess.run([sys.executable, "-W", "ignore", script, "4"], capture_output=True, text=True, timeout=300)
    assert "child decoded with 1 threads: OK" in r.stdout, r.stdout + r.stderr


# ----------------------------------------------------------------------------------------------- stores
def _write(path, config, chunk, n_dates=2, register=False):
    data = np.stack([chunk] * n_dates)[:, :, None, :]
    g = zarr.open_group(str(path), mode="w")
    if register:
        from anemoi_datasets_codecs.icon_lattice import register_store

        register_store(path)
    codec = IconLatticeCodec(**config)
    a = g.create_array(
        "data", shape=data.shape, chunks=(1, *data.shape[1:]), dtype="float32", serializer=codec, compressors=None
    )
    a[:] = data
    g.attrs["variables"] = list(NAMES)
    return data


def _strip_sidecar(path):
    """Move the store's sidecar files into the cache directory (the writer's side), as if the store had been copied
    without its sidecar (``anemoi-datasets copy``)."""
    cache = sc.cache_dir()
    for kind in ("grids", "statics"):
        d = path / sc.SIDECAR / kind
        for f in d.iterdir() if d.is_dir() else ():
            sc.write(cache, kind, f.name.split(".")[0], f.read_bytes())
    shutil.rmtree(path / sc.SIDECAR)


def test_attach_and_check(tmp_path, lattice, chunk):
    """A store copied without its sidecar (the writer's cache still holds the files): check reports the missing
    sidecar, attach copies lattice and statics from the cache into the store, then check passes and a fresh process
    reads the store with nothing but the store itself."""
    path = tmp_path / "plain.zarr"
    _write(path, make_config(lattice[0]), chunk)
    _strip_sidecar(path)
    r = store.check(path)
    assert r["problems"] and any("missing" in p for p in r["problems"])
    # one line per missing static, naming the chunks by their keys in the array
    missing = [p for p in r["problems"] if p.startswith("static ")]
    assert len(missing) == len(STATIC) and len({p.split()[1] for p in missing}) == len(STATIC)
    assert all("(referenced by c/0/0/0/0, c/1/0/0/0)" in p for p in missing), missing
    rep = store.attach(path)
    assert rep["statics"] == len(STATIC) and rep["missing"] == [] and rep["verified"]
    assert store.check(path)["problems"] == []
    assert {p.name.split(".")[0] for p in (path / sc.SIDECAR / "statics").iterdir()} == {
        statics.sha256_of(chunk[NAMES.index(n)]) for n in STATIC
    }
    m = store.array_meta(path)
    assert m["format"] == 3 and m["prefix"] == "c/"
    assert m["codec"] == {**make_codec(lattice[0]).get_config(), "written_by": config.writer_stamp()}
    assert store.check(path)["written_by"] == {"icon_lattice": config.writer_stamp()}
    # wrong variable order in the store attribute (the group's zarr.json attributes) is reported
    zarr.open_group(str(path), mode="r+").attrs["variables"] = list(reversed(NAMES))
    assert any("variables" in p for p in store.check(path)["problems"])
    with pytest.raises(FileNotFoundError, match="zarr.json"):
        store.array_meta(path, "nope")


def test_corrupt_sidecar_files(tmp_path, lattice, chunk):
    """A damaged or truncated sidecar file (static field or lattice) is a ValueError naming the file (sha256
    mismatch), whatever the damage."""
    from anemoi_datasets_codecs.icon_lattice.layout import TriangleLatticeLayout

    x = chunk[NAMES.index("FIS")]
    sha = statics.sha256_of(x)
    good = statics.static_file_bytes(x)
    lat = lattice[0].save(tmp_path / "lattice.npz").read_bytes()
    for name, data in (("flipped", good[:40] + bytes(4) + good[44:]), ("truncated", good[: len(good) // 2])):
        f = tmp_path / f"{name}.f4.blosc"
        f.write_bytes(data)
        with pytest.raises(ValueError, match=rf"static file .*{name}.f4.blosc is corrupt \(sha256 mismatch\)"):
            statics.read_static_file(f, sha)
    statics.clear_caches()
    with pytest.raises(ValueError, match="sha256"):
        statics.lookup(sha, good[: len(good) // 2])  # inline bytes
    for name, data in (("flipped", lat[:100] + bytes(8) + lat[108:]), ("truncated", lat[: len(lat) // 2])):
        f = tmp_path / f"{name}.npz"
        f.write_bytes(data)
        with pytest.raises(ValueError, match=rf"{name}.npz: lattice file is corrupt.*sha256 mismatch"):
            TriangleLatticeLayout.load(f, lattice[0].sha256)


def test_cli(tmp_path, lattice, chunk):
    path = tmp_path / "cli.zarr"
    _write(path, make_config(lattice[0]), chunk, register=True)

    def run(*args):
        return subprocess.run(
            [sys.executable, "-m", "anemoi_datasets_codecs.cli", *args], capture_output=True, text=True
        )

    r = run("isa")
    assert r.returncode == 0 and json.loads(r.stdout)["in_use"] in ("avx2", "scalar")
    assert "icon-ch1" in json.loads(run("grids").stdout)
    r = run("check", str(path))
    assert r.returncode == 0, r.stdout + r.stderr
    chunk_file = next(store.chunk_paths(path))
    r = run("info", str(chunk_file))
    assert r.returncode == 0 and "static_ref" in r.stdout and "lattice" in r.stdout
    # errors: one line, exit status 1; --debug shows the traceback
    r = run("check", str(tmp_path / "nope.zarr"))
    assert r.returncode == 1 and r.stderr.startswith("anemoi-datasets-codecs check: ") and "Traceback" not in r.stderr
    assert len(r.stderr.strip().splitlines()) == 1, r.stderr
    r = run("info", str(tmp_path / "nope"))
    assert (
        r.returncode == 1
        and r.stderr.strip() == f"anemoi-datasets-codecs info: {tmp_path / 'nope'}: No such file or directory"
    )
    r = run("--debug", "check", str(tmp_path / "nope.zarr"))
    assert r.returncode == 1 and "Traceback" in r.stderr and "FileNotFoundError" in r.stderr


def test_bounds_report(tmp_path, lattice, chunk):
    """scripts/bounds_report.py on a plain anemoi-like store of three hourly dates: stats (with the change over one
    time step for the two dates that have their predecessor), encode at two scales, a deterministic figure."""
    import importlib.util

    from anemoi_datasets_codecs.icon_lattice import grids

    spec = importlib.util.spec_from_file_location(
        "bounds_report", os.path.join(os.path.dirname(__file__), "..", "scripts", "bounds_report.py")
    )
    br = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(br)
    layout = lattice[0]
    grids.save_lattice(layout)
    rng = np.random.default_rng(1)
    data = np.stack([chunk + (rng.standard_normal(chunk.shape) * 0.01 * i).astype(np.float32) for i in range(3)])
    static = [NAMES.index(n) for n in STATIC]
    data[:, static] = chunk[static]
    g = zarr.open_group(str(tmp_path / "plain.zarr"), mode="w")
    g.create_array("data", data=data[:, :, None, :], chunks=(1, len(NAMES), 1, data.shape[-1]))
    g.create_array("dates", data=DATE + np.arange(3) * np.timedelta64(3600, "s"))
    g.attrs.update(variables=list(NAMES), frequency="1h")
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({**make_config(layout), "grid": layout.ref()}))
    st, enc = tmp_path / "stats.json", tmp_path / "encode.json"
    br.main(["stats", str(tmp_path / "plain.zarr"), str(cfg), str(st), "--dates", "0:3", "--threads", "2"])
    s = json.loads(st.read_text())
    lossy = [v.name for v in make_codec(layout).variables if v.lossy]
    assert [v["name"] for v in s["variables"]] == lossy and s["tendency_pairs"] == 2 and s["time_step"] == "1 h"
    assert all(v["edge"] > 0 and v["ulp"] > 0 and v["std"] > 0 for v in s["variables"])
    br.main(["encode", str(tmp_path / "plain.zarr"), str(cfg), str(enc), "--dates", "1", "--scales", "1,2"])
    runs = json.loads(enc.read_text())["runs"]
    assert [r["scale"] for r in runs] == [1.0, 2.0]
    assert all(v["max_err_over_bound"] <= 1 for r in runs for v in r["variables"].values())
    assert sum(v["bytes"] for v in runs[1]["variables"].values()) < sum(
        v["bytes"] for v in runs[0]["variables"].values()
    )
    svgs = []
    for name in ("a.svg", "b.svg"):
        br.main(["draw", str(st), str(tmp_path / name), "--encode", str(enc), "--table", str(tmp_path / "t.csv")])
        svgs.append((tmp_path / name).read_text())
    assert svgs[0] == svgs[1] and all(n in svgs[0] for n in lossy)
    assert len((tmp_path / "t.csv").read_text().splitlines()) == len(lossy) + 1
