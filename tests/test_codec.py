"""The lattice codec engine: structures, encoder, C decoder, container, statics / recipes, configuration, and Zarr 3
store round trips (sidecar resolution in a fresh process), on a synthetic lattice; one real ICON-CH1 date.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pytest
from conftest import (
    DATE,
    GRID_FILE,
    NAMES,
    SAMPLE_CONFIG,
    SAMPLE_STORE,
    STATIC,
    VARS,
    check_bounds,
    make_codec,
    make_config,
    smooth,
)

from anemoi_datasets_codecs import sidecar as sc
from anemoi_datasets_codecs.icon_lattice import (
    LatticeCodec,
    container,
    details,
    gridfile,
    grids,
    inspect_chunk,
    native,
    statics,
)
from anemoi_datasets_codecs.icon_lattice import structure as S
from anemoi_datasets_codecs.icon_lattice.container import (
    MODE_CONSTANT,
    MODE_LATTICE,
    MODE_LOSSLESS,
    MODE_RECIPE,
    MODE_STATIC_REF,
)
from anemoi_datasets_codecs.icon_lattice.encoder import EncodeError, EncodeOptions, encode_field
from anemoi_datasets_codecs.icon_lattice.recipes import RECIPE_IDS
from anemoi_datasets_codecs.physics import cross_plan
from anemoi_datasets_codecs.plan import depth, limit_depth, topo_order


# --------------------------------------------------------------------------------------------- structures
@pytest.mark.parametrize("sid", ["tree", "tree_b", "hc2_k3"])
def test_structure_partitions_cells(lattice, sid):
    layout = lattice[0]
    s = S.get_structure(sid, layout)
    s.validate()  # every cell once, sources decoded earlier (C check)
    assert sum(st.n for st in s.steps) == layout.n_cells
    assert s.steps[0].kind == "delta" and all(st.k == 3 for st in s.steps[1:])
    S._cache.clear()
    assert S.get_structure(sid, layout).sha == s.sha  # disk cache round trip


# the frozen structures on the synthetic test lattice: a builder change shows up here, independently of the chunk
# baselines (structures are integer-only numpy, so the sha does not depend on the machine)
STRUCTURE_SHA256 = {
    "tree": "21ef9db35c4096911db9794927d254f798911cdceac33b1f387e667d9fc333b0",
    "tree_b": "73628680848ccc518d604e5f07ec0939dd17cd1c27494823db15e461bba905c0",
    "hc2_k3": "a5e2b9ad36c338a1d8894f43d464cda0d15805863948775296932d387646cb20",
}


@pytest.mark.parametrize("sid", sorted(STRUCTURE_SHA256))
def test_structure_sha_is_frozen(lattice, sid, monkeypatch):
    monkeypatch.setenv(sc.CACHE_ENV, "off")  # built, not read from the disk cache
    monkeypatch.setattr(S, "_cache", {})
    assert S.get_structure(sid, lattice[0]).sha == STRUCTURE_SHA256[sid]


def test_structure_rejects_other_layouts():
    with pytest.raises(ValueError, match="triangle_lattice"):
        S.get_structure("tree", types.SimpleNamespace(kind="padded_reshape"))
    with pytest.raises(ValueError, match="unknown"):
        S.get_structure("nope", None)


# ---------------------------------------------------------------------------------------------- rANS
def test_rans_roundtrip_with_tables_and_segments():
    rng = np.random.default_rng(0)
    syms = np.minimum(rng.geometric(0.3, 10_001) - 1, 255).astype(np.uint8)
    tid = (np.arange(len(syms)) % 3 == 0).astype(np.uint8)
    seg = np.array([0, 37, 5000, len(syms)], np.uint64)
    freqs = np.stack([native.normalize_freqs(np.bincount(syms[tid == t], minlength=256)) for t in (0, 1)])
    stream = native.rans_encode(syms, tid, seg, freqs)
    out = np.zeros_like(syms)
    buf = np.frombuffer(stream, np.uint8)
    rc = native.lib().lc_rans_decode_test(
        native.ptr(buf), len(buf), native.ptr(seg), 3, native.ptr(freqs), native.ptr(tid), native.ptr(out), 12
    )
    assert rc == 0 and np.array_equal(out, syms)
    f10 = np.stack([native.normalize_freqs(np.bincount(syms[tid == t], minlength=256), 1024) for t in (0, 1)])
    buf = np.frombuffer(native.rans_encode(syms, tid, seg, f10, 10), np.uint8)
    out[:] = 0
    rc = native.lib().lc_rans_decode_test(
        native.ptr(buf), len(buf), native.ptr(seg), 3, native.ptr(f10), native.ptr(tid), native.ptr(out), 10
    )
    assert rc == 0 and np.array_equal(out, syms)  # 10-bit precision (4 KB tables)


@pytest.mark.parametrize("binding", [True, False])
def test_fit_robust_is_numpy_lstsq_bit_for_bit(binding, monkeypatch):
    """fit_robust through native.Lstsq (ctypes dgelsd, or its np.linalg.lstsq fallback) equals the plain IRLS."""
    from anemoi_datasets_codecs.icon_lattice.encoder import fit_robust

    if not binding:
        monkeypatch.setattr(native, "_gelsd_fn", False)
    elif native.gelsd() is None:
        pytest.skip("numpy's LAPACK is not a scipy-openblas build")

    def plain(X, y, it=6):
        w = np.linalg.lstsq(X, y, rcond=None)[0]
        for _ in range(it):
            s = 1.0 / np.sqrt(np.maximum(np.abs(y - X @ w), 0.5))
            w = np.linalg.lstsq(X * s[:, None], y * s, rcond=None)[0]
        return w

    rng = np.random.default_rng(3)
    for m, k, order in ((50_000, 2, "F"), (3000, 5, "C"), (40, 1, "F"), (7, 2, "C")):
        X = np.asarray(rng.integers(-2000, 2000, (m, k)), np.float64, order=order)
        y = np.round(X @ rng.random(k) + rng.laplace(size=m) * 20)
        assert fit_robust(X, y).view(np.int64).tolist() == plain(X, y).view(np.int64).tolist()


def test_gelsd_binding_falls_back_when_its_self_test_fails(monkeypatch):
    """A failing self-test (an exception, not only a mismatch) leaves the encoder on np.linalg.lstsq."""
    from anemoi_datasets_codecs.icon_lattice.encoder import fit_robust

    def boom(self, m, n, fn=None):
        raise RuntimeError("the LAPACK call failed")

    monkeypatch.setattr(native, "_gelsd_fn", None)
    monkeypatch.setattr(native.Lstsq, "__init__", boom)
    assert native.gelsd() is None
    monkeypatch.undo()
    monkeypatch.setattr(native, "_gelsd_fn", False)  # the fallback: fit_robust still fits
    X = np.arange(12.0).reshape(6, 2)
    w = fit_robust(X, X @ np.array([1.5, -2.0]))
    assert np.allclose(w, [1.5, -2.0])


def test_irls_scale_is_numpy_bit_for_bit():
    """The one-pass IRLS reweighting (csrc/encode.c) equals the numpy expressions in both kernel variants."""
    rng = np.random.default_rng(4)
    try:
        for isa in native.available_isas():
            native.select_isa(isa)
            for m, k, order in ((1, 1, "F"), (5, 2, "C"), (513, 3, "F"), (1027, 5, "C")):
                X = np.asarray(rng.standard_normal((m, k)) * 1e3, order=order)
                y = rng.standard_normal(m) * 3
                y[::7] = np.nan
                xw = y + rng.standard_normal(m) * rng.choice([0.1, 1.0, 10.0], m)
                a, b = np.empty((m, k), order="F"), np.empty(m)
                assert native.irls_scale(y, xw, X, a, b)
                s = 1.0 / np.sqrt(np.maximum(np.abs(y - xw), 0.5))
                assert np.array_equal(a.view(np.int64), (X * s[:, None]).view(np.int64)), isa
                assert np.array_equal(b.view(np.int64), (y * s).view(np.int64)), isa
    finally:
        native.select_isa("auto")


# ---------------------------------------------------------------------------------------------- fields
@pytest.mark.parametrize("sid", ["tree", "hc2_k3"])
@pytest.mark.parametrize("contexts", [1, 4])
def test_field_roundtrip_paths(lattice, chunk, sid, contexts):
    layout = lattice[0]
    s = S.get_structure(sid, layout)
    opts = EncodeOptions(contexts=contexts)
    for name, vt in (("PMSL", 2), ("WIDE", 4), ("NOISY", 2)):
        i = NAMES.index(name)
        bound = VARS[i][1]
        enc = encode_field(chunk[i], bound, None, s, opts)
        assert enc.info["vt"] == vt
        assert enc.info["max_err"] <= bound
    enc = encode_field(chunk[NAMES.index("NOISY")], 0.01, None, s, opts)
    assert enc.info["n_esc"] > 0  # residuals beyond the byte alphabet go through the escape stream


def test_field_clip_and_k(lattice, chunk):
    s = S.get_structure("tree", lattice[0])
    i = NAMES.index("CLCT")
    enc = encode_field(chunk[i], 0.5, (0.0, 100.0), s, EncodeOptions())
    assert enc.info["max_err"] <= 0.5
    tight = encode_field(chunk[i], 0.5, (0.0, 100.0), s, EncodeOptions(k=4))
    assert tight.info["max_err"] <= 0.5 / 4 + 1e-6 and len(tight.payload) > len(enc.payload)


def test_field_bound_below_float_resolution_raises(lattice, chunk):
    s = S.get_structure("tree", lattice[0])
    with pytest.raises(EncodeError):
        encode_field(chunk[NAMES.index("PMSL")], 1e-4, None, s, EncodeOptions())


# ---------------------------------------------------------------------------------------------- chunks
@pytest.mark.parametrize("sid", ["tree", "tree_b", "hc2_k3"])
def test_chunk_roundtrip_all_modes(lattice, chunk, sid):
    codec = make_codec(lattice[0], sid)
    enc = codec.encode(chunk, dates=[DATE])
    modes = [r.mode for r in container.inspect_chunk(enc)]
    expect = {
        "FLAT": MODE_CONSTANT,
        "cos_julian_day": MODE_CONSTANT,
        "FIS": MODE_STATIC_REF,
        "cos_longitude": MODE_STATIC_REF,
        "insolation": MODE_RECIPE,
        "cos_local_time": MODE_RECIPE,
        "EXTRA": MODE_LOSSLESS,
        "T_2M": MODE_LATTICE,
        "WIDE": MODE_LATTICE,
        "NOISY": MODE_LATTICE,
    }
    for name, m in expect.items():
        assert modes[NAMES.index(name)] == m, name
    dec = codec.decode(enc).reshape(chunk.shape)
    check_bounds(dec, chunk)
    i = NAMES.index("TOT_PREC_1H")  # the grid is anchored at zero: exact zeros stay exact
    zeros = chunk[i] == 0
    assert zeros.mean() > 0.3 and np.all(dec[i][zeros] == 0)
    # zarr-style decode into a given buffer, and the (1, V, E, cells) shape
    out = np.empty((1, len(NAMES), 1, chunk.shape[1]), np.float32)
    codec.decode(codec.encode(chunk[None, :, None, :], dates=[DATE]), out=out)
    np.testing.assert_array_equal(out[0, :, 0], dec)


def test_cross_terms_are_used_and_pay(lattice, chunk):
    on = make_codec(lattice[0])
    off = LatticeCodec(**make_config(lattice[0]), physics={"cross": "off"})
    i = NAMES.index("T_2M")
    p_on = container.parse(on.encode(chunk, dates=[DATE])).payloads[i]
    p_off = container.parse(off.encode(chunk, dates=[DATE])).payloads[i]
    assert p_on[4] == 2  # T_2M uses FIS (static detail) and PMSL (lattice detail)
    assert len(p_on) < 0.8 * len(p_off)


def test_recipes_without_date_are_lossless(lattice, chunk):
    """With ``recipe_fit: false`` a recipe needs the valid date: without one (zarr) the forcings are lossless."""
    codec = LatticeCodec(**make_config(lattice[0]), physics={"recipe_fit": False})
    enc = codec.encode(chunk)
    recs = container.inspect_chunk(enc)
    assert all(recs[NAMES.index(n)].mode == MODE_LOSSLESS for n in RECIPE_IDS)
    dec = codec.decode(enc).reshape(chunk.shape)
    for n in RECIPE_IDS:
        np.testing.assert_array_equal(dec[NAMES.index(n)], chunk[NAMES.index(n)])
    assert container.parse(enc).dates[0] == container.NO_DATE


def test_recipe_with_wrong_date_falls_back(lattice, chunk):
    codec = make_codec(lattice[0])
    enc = codec.encode(chunk, dates=["2021-01-03T05:00:00"])
    recs = container.inspect_chunk(enc)
    assert recs[NAMES.index("insolation")].mode == MODE_LOSSLESS
    check_bounds(codec.decode(enc).reshape(chunk.shape), chunk)


def test_recipe_fit_without_date(lattice, chunk):
    """physics.recipe_fit (the default): without a date the coefficients come from the fields, the date's own
    coefficients."""
    from anemoi_datasets_codecs.icon_lattice.recipes import recipe_params

    codec = make_codec(lattice[0])
    assert codec.get_config()["physics"]["recipe_fit"] is True
    enc = codec.encode(chunk)
    assert enc == LatticeCodec(**make_config(lattice[0]), physics={"recipe_fit": True}).encode(chunk)
    ch = container.parse(enc)
    assert ch.dates[0] == container.NO_DATE
    secs = int(np.datetime64(DATE, "s").astype(np.int64))
    dated = container.parse(make_codec(lattice[0]).encode(chunk, dates=[DATE]))
    for n, rid in RECIPE_IDS.items():
        i = NAMES.index(n)
        assert ch.modes[i] == MODE_RECIPE, n
        p = bytes(ch.payloads[i])
        np.testing.assert_array_equal(np.frombuffer(p[16:48], "<f8"), recipe_params(rid, secs))
        assert p[:56] == bytes(dated.payloads[i])[:56]  # the dated payload but for its date
        assert int.from_bytes(p[56:64], "little", signed=True) == container.NO_DATE
    check_bounds(codec.decode(enc).reshape(chunk.shape), chunk)
    # off: written into the configuration like every physics key
    assert LatticeCodec(**make_config(lattice[0]), physics={"recipe_fit": False}).physics["recipe_fit"] is False


def test_recipe_fit_falls_back_on_other_data(lattice, chunk):
    codec = LatticeCodec(**make_config(lattice[0]), physics={"recipe_fit": True})
    x = chunk.copy()
    i = NAMES.index("insolation")
    x[i] = np.where(x[i] > 0, x[i] * 1.01, 0)  # not the formula of any date
    recs = container.inspect_chunk(codec.encode(x))
    assert recs[i].mode == MODE_LOSSLESS
    assert recs[NAMES.index("cos_local_time")].mode == MODE_RECIPE


@pytest.mark.parametrize("sid", ["tree", "tree_b", "hc2_k3"])
def test_rounding_direct(lattice, chunk, sid):
    """options.rounding: direct (the default) codes the residuals from the exact values: the full step, within the
    bound, smaller than ``grid`` (the rounding of the first baselines); both are written into the configuration."""
    grid, direct = make_codec(lattice[0], sid, rounding="grid"), make_codec(lattice[0], sid)
    assert grid.get_config()["options"]["rounding"] == "grid"
    assert direct.get_config()["options"]["rounding"] == "direct"
    a, b = grid.encode(chunk, dates=[DATE]), direct.encode(chunk, dates=[DATE])
    assert b == make_codec(lattice[0], sid, rounding="direct").encode(chunk, dates=[DATE])
    check_bounds(direct.decode(b).reshape(chunk.shape), chunk)
    ra, rb = container.inspect_chunk(a), container.inspect_chunk(b)
    i = NAMES.index("WIDE")
    assert ra[i].mode == rb[i].mode == MODE_LATTICE
    assert rb[i].nbytes < ra[i].nbytes
    # a reader configured with either value decodes both (the decoder does not depend on the option)
    np.testing.assert_array_equal(grid.decode(b), direct.decode(b))
    np.testing.assert_array_equal(grid.decode(a), direct.decode(a))
    with pytest.raises(ValueError, match="rounding"):
        make_codec(lattice[0], rounding="nearest")


def test_statics_inline_and_off(lattice, chunk):
    for mode, expect in (("inline", MODE_STATIC_REF), ("off", MODE_LOSSLESS)):
        codec = make_codec(lattice[0], statics=mode)
        enc = codec.encode(chunk, dates=[DATE])
        assert container.inspect_chunk(enc)[NAMES.index("FIS")].mode == expect
        statics.clear_caches()  # a fresh process: inline data must be enough
        os.environ[sc.CACHE_ENV] = "off"  # (restored by the autouse fixture)
        check_bounds(codec.decode(enc).reshape(chunk.shape), chunk)


def test_static_missing_from_store(lattice, chunk, monkeypatch):
    """No sidecar to reference the statics from and nowhere to capture them: LOSSLESS for that chunk."""
    statics.clear_caches()
    monkeypatch.setenv(sc.CACHE_ENV, "off")
    codec = make_codec(lattice[0])
    with pytest.warns(UserWarning, match="could not be written"):
        enc = codec.encode(chunk, dates=[DATE])
    assert container.inspect_chunk(enc)[NAMES.index("FIS")].mode == MODE_LOSSLESS
    check_bounds(codec.decode(enc).reshape(chunk.shape), chunk)


def test_static_ref_unknown_sha_raises(lattice, chunk):
    codec = make_codec(lattice[0])
    enc = bytearray(codec.encode(chunk, dates=[DATE]))
    ch = container.parse(bytes(enc))
    off = ch.header_bytes + sum(len(p) for p in ch.payloads[: NAMES.index("FIS")])
    enc[off] ^= 0xFF
    statics.clear_caches()
    with pytest.raises(ValueError, match="static"):
        codec.decode(bytes(enc))


def test_non_finite_raises(lattice, chunk):
    codec = make_codec(lattice[0])
    bad = chunk.copy()
    bad[1, 5] = np.nan
    with pytest.raises(ValueError, match="non-finite.*PMSL"):
        codec.encode(bad)


def test_multi_date_chunk(lattice, chunk):
    codec = make_codec(lattice[0])
    two = np.stack([chunk, chunk])
    enc = codec.encode(two, dates=[DATE, DATE])
    assert len(inspect_chunk(enc)) == 2 * len(NAMES)
    dec = codec.decode(enc).reshape(two.shape)
    check_bounds(dec[1], chunk)


# ---------------------------------------------------------------------------------------------- container
def test_container_records_sum_to_size(lattice, chunk):
    codec = make_codec(lattice[0])
    enc = codec.encode(chunk, dates=[DATE])
    recs = inspect_chunk(enc)
    assert sum(r.nbytes for r in recs) == len(enc)
    assert {r.mode for r in recs} == set(container.MODE_LABELS)  # every mode occurs in the test chunk
    ch = container.parse(enc)
    assert ch.structure_sha == codec.structure.sha16
    assert ch.dates[0] == DATE.astype("datetime64[s]").astype(np.int64)
    with pytest.raises(ValueError, match="size"):
        container.parse(enc[:-4])
    with pytest.raises(ValueError, match="magic"):
        container.parse(b"XXXX" + enc[4:])


def test_field_references_agree_with_decode(lattice, chunk, monkeypatch):
    """``container.field_references`` names the cross sources ``codec.decode`` reads from the LATTICE headers: decode
    computes the static detail of exactly the STATIC_REF fields that are cross sources of a LATTICE field."""
    codec = make_codec(lattice[0])
    enc = codec.encode(chunk, dates=[DATE])
    ch = container.parse(enc)
    refs = container.field_references(enc)
    plan = codec.cross_plan
    for i, r in enumerate(refs):
        if ch.modes[i] == MODE_LATTICE:
            assert set(r.get("cross", [])) <= {NAMES.index(s) for s in plan.get(NAMES[i], [])}, NAMES[i]
        else:
            assert "cross" not in r
    static_srcs = sorted({j for r in refs for j in r.get("cross", []) if ch.modes[j] == MODE_STATIC_REF})
    assert static_srcs == [NAMES.index("FIS")]  # the test chunk has a static cross source
    asked = []
    real = details.static_detail

    def spy(sha, x, structure, decode_order=False):
        if decode_order:
            asked.append(sha)
        return real(sha, x, structure, decode_order)

    monkeypatch.setattr(details, "static_detail", spy)
    check_bounds(codec.decode(enc).reshape(chunk.shape), chunk)
    assert sorted(asked) == sorted(bytes(ch.payloads[j][:32]).hex() for j in static_srcs)


def test_corrupt_payload_is_an_error_not_a_crash(lattice, chunk):
    codec = make_codec(lattice[0])
    enc = bytearray(codec.encode(chunk, dates=[DATE]))
    ch = container.parse(bytes(enc))
    off = ch.header_bytes + sum(len(p) for p in ch.payloads[: NAMES.index("PMSL")])
    enc[off + 3] = 250  # n_tables of PMSL
    with pytest.raises(ValueError, match="lattice codec decode failed"):
        codec.decode(bytes(enc))


GOLDEN = Path(__file__).parent / "data" / "golden_chunks.json"
GOLDEN_SIDECAR = GOLDEN.parent / "golden_sidecar"


def _complete(cfg: dict) -> dict:
    """A stored lattice configuration as this version reads it: the options and physics keys it lacks (written
    before they existed, or before every key was written) take this version's defaults; the stamp is ignored."""
    from anemoi_datasets_codecs import physics
    from anemoi_datasets_codecs.icon_lattice.config import DEFAULT_OPTIONS

    out = {k: v for k, v in cfg.items() if k != "written_by"}
    out["options"] = {**DEFAULT_OPTIONS, **cfg.get("options", {})}
    out["physics"] = {**physics.PHYSICS_DEFAULTS, **cfg.get("physics", {})}
    return out


def _golden_decode(g: dict, enc: bytes, lattice) -> np.ndarray:
    """Decode a golden chunk from its stored configuration (a lattice codec configuration, or a chain's codec
    entries as in its zarr.json)."""
    from anemoi_datasets_codecs import chain, envelope

    ref = {"name": "test-lattice", "n_cells": lattice[0].n_cells, "sha256": lattice[0].sha256}
    if "chain" not in g:
        cfg = dict(g["configuration"])
        assert cfg.pop("grid") == ref
        codec = LatticeCodec(**cfg, grid=lattice[0])
        assert codec.get_config() == _complete({**g["configuration"], "grid": ref})
        if "modes" in g:  # the chunk exercises what it is kept for
            assert [int(m) for m in container.parse(enc).modes] == g["modes"]
        return codec.decode(enc)
    lat = next(e["configuration"] for e in g["chain"] if e["name"] == "icon_lattice_bytes")
    assert lat["grid"] == ref
    grids.register_lattice(lattice[0])  # the lattice (not in tests/data): this process has it
    env = envelope.parse(enc)
    stages = [e["name"] for e in g["chain"][:-1]]
    assert env is not None and [h[0] for h in env.headers] == stages and stages == ["field_terms", "static_ref"]
    return chain.decode_chunk(g["chain"], enc, g["shape"])


@pytest.mark.parametrize("name", sorted(json.loads(GOLDEN.read_text())))
def test_golden_chunk_decodes_bit_identically(lattice, name, monkeypatch):
    """Chunks written once (tests/data/golden_chunks.json; ``scripts/golden_chunks.py`` wrote all but ``tree``, and
    says what each covers: structures tree / tree_b / hc2_k3, cross terms on lattice and static sources, a mask term,
    RECIPE fields, STATIC_REF fields inline and through a sidecar, a clip, escapes, a constant, the chain
    field_terms + static_ref + icon_lattice_bytes in its FCHN envelope) decode to the same float32 bytes with every
    kernel variant: the decoder side of the frozen formats, checkable without the real-data sample and independent
    of the encoder (golden chunks are decoder fixtures; the encoder that wrote them does not matter). The stored
    bytes are decoded, never regenerated; the static fields they reference come from tests/data/golden_sidecar
    alone. Never rewrite the files or the shas."""
    g = json.loads(GOLDEN.read_text())[name]
    enc = (GOLDEN.parent / g["file"]).read_bytes()
    assert hashlib.sha256(enc).hexdigest() == g["chunk_sha256"]
    monkeypatch.setenv(sc.CACHE_ENV, "off")
    monkeypatch.setenv(sc.PATH_ENV, str(GOLDEN_SIDECAR))
    statics.clear_caches()
    try:
        for isa in native.available_isas():
            native.select_isa(isa)
            dec = np.ascontiguousarray(_golden_decode(g, enc, lattice), np.float32)
            assert hashlib.sha256(dec.tobytes()).hexdigest() == g["decoded_sha256"], isa
    finally:
        native.select_isa(os.environ.get("ANEMOI_DATASETS_CODECS_ISA", "auto"))
        statics.clear_caches()


def test_golden_chain_chunk_reads_through_zarr(lattice, tmp_path, monkeypatch):
    """The golden chain chunk read by zarr through the codecs of its stored entries (from_dict: stamps kept)."""
    import zarr

    from anemoi_datasets_codecs import chain

    g = json.loads(GOLDEN.read_text())["chain"]
    grids.register_lattice(lattice[0])
    monkeypatch.setenv(sc.CACHE_ENV, "off")
    monkeypatch.setenv(sc.PATH_ENV, str(GOLDEN_SIDECAR))
    statics.clear_caches()
    codecs = [chain.codec_class(e["name"]).from_dict(e) for e in g["chain"]]
    assert [c.to_dict() for c in codecs] == g["chain"]  # the stored configurations, stamps included
    a = zarr.open_group(tmp_path / "s.zarr", mode="w").create_array(
        "data", shape=g["shape"], chunks=g["shape"], dtype="float32", compressors=codecs
    )
    (tmp_path / "s.zarr" / "data" / "c" / "0" / "0" / "0").mkdir(parents=True)
    (tmp_path / "s.zarr" / "data" / "c" / "0" / "0" / "0" / "0").write_bytes((GOLDEN.parent / g["file"]).read_bytes())
    dec = np.ascontiguousarray(a[...], np.float32)
    assert hashlib.sha256(dec.tobytes()).hexdigest() == g["decoded_sha256"]
    statics.clear_caches()


def _forged_chunk(codec, n_dates, n_header_dates=None):
    """A well-formed container of ``n_dates`` dates of CONSTANT fields (a header asking for a large array)."""
    n = n_dates * codec.fields_per_date
    dates = np.full(n_dates if n_header_dates is None else n_header_dates, container.NO_DATE, np.int64)
    zero = np.float32(0).tobytes()
    return container.pack([MODE_CONSTANT] * n, [0] * n, [zero] * n, dates, codec.structure.sha16)


def test_corrupt_header_rejected_before_allocating(lattice, chunk, monkeypatch):
    """A chunk header asking for more fields than the chunk holds (decoder hardening F3): ``out`` from the caller's
    chunk shape is checked before anything is allocated from the header, and the header's date count must match
    its field count."""
    codec = make_codec(lattice[0])
    forged = _forged_chunk(codec, 1000)  # 1000 dates where the chunk holds one
    out = np.empty((1, len(NAMES), 1, codec.n_cells), np.float32)
    allocated = []
    real_empty = np.empty
    monkeypatch.setattr(np, "empty", lambda *a, **k: allocated.append(a) or real_empty(*a, **k))
    with pytest.raises(ValueError, match=f"header has {1000 * len(NAMES)} fields"):
        codec.decode(forged, out=out)
    assert not allocated
    monkeypatch.undo()
    with pytest.raises(ValueError, match=f"header has {2 * len(NAMES)} fields for 1 dates"):
        codec.decode(_forged_chunk(codec, 2, n_header_dates=1))
    assert codec.decode(_forged_chunk(codec, 1), out=out) is out  # a valid header of the same form decodes
    assert not out.any()


def test_structure_mismatch_raises(lattice, chunk):
    tree = make_codec(lattice[0])
    hc2 = make_codec(lattice[0], "hc2_k3")
    with pytest.raises(ValueError, match="structure"):
        hc2.decode(tree.encode(chunk))


# ---------------------------------------------------------------------------------------------- determinism
def test_determinism(lattice, chunk, monkeypatch):
    codec = make_codec(lattice[0], "hc2_k3")
    a = codec.encode(chunk, dates=[DATE])
    b = codec.encode(chunk, dates=[DATE])
    assert a == b
    monkeypatch.setenv("ANEMOI_DATASETS_CODECS_THREADS", "1")
    d1 = codec.decode(a)
    monkeypatch.setenv("ANEMOI_DATASETS_CODECS_THREADS", "7")
    d2 = codec.decode(a)
    np.testing.assert_array_equal(d1.view(np.uint32), d2.view(np.uint32))


# ---------------------------------------------------------------------------------------------- plan / DAG
def test_cross_plan_is_a_dag_on_ch1_names():
    lv = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
    names = [f"{f}_{p}" for f in ("FI", "OMEGA", "QV", "T", "U", "V") for p in lv]
    names += ["PMSL", "PS", "T_2M", "TD_2M", "SKT", "U_10M", "V_10M", "FIS", "CAPE_ML", "CAPE_MU"]
    for mode, max_d in (("on", 8), ("shallow", 4)):
        plan = cross_plan(names, mode)
        order = topo_order(names, plan)
        pos = {names[i]: k for k, i in enumerate(order)}
        for tgt, srcs in plan.items():
            assert all(pos[s] < pos[tgt] for s in srcs)
        d = depth(names, plan)
        assert d["FIS"] == 0 and d["PS"] == 3 and max(d.values()) <= max_d
        assert plan["PS"] == ["PMSL", "FIS", "T_2M"]
    assert cross_plan(names, "off") == {}
    # the depth limit (static FIS is free): every kept source is a root of the decode DAG
    flat = limit_depth(names, cross_plan(names, "shallow"), 1, {"FIS"})
    assert flat["PS"] == ["PMSL", "FIS"] and "T_2M" not in flat.get("SKT", [])
    for srcs in flat.values():
        assert all(all(x == "FIS" for x in flat.get(s, [])) for s in srcs)


def test_decoder_dag_handles_reverse_field_order(lattice, chunk):
    """T_2M comes before its sources in the chunk; the C decoder must order the tasks."""
    codec = make_codec(lattice[0])
    enc = codec.encode(chunk, dates=[DATE])
    for threads in ("1", "4"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("ANEMOI_DATASETS_CODECS_THREADS", threads)
            check_bounds(codec.decode(enc).reshape(chunk.shape), chunk)


# ---------------------------------------------------------------------------------------------- configuration
def test_config_roundtrip_and_entry_points(lattice, chunk):
    codec = make_codec(lattice[0], "hc2_k3", k=2.0)
    cfg = json.loads(json.dumps(codec.get_config()))
    assert "id" not in cfg and cfg["structure"] == "hc2_k3" and cfg["options"]["k"] == 2.0
    assert set(cfg) == {
        "version",
        "variables",
        "bounds",
        "clip",
        "static_fields",
        "grid",
        "n_ensembles",
        "structure",
        "lossless",
        "options",
        "physics",
    }
    assert len(json.dumps(cfg)) < 4096  # small: names, bounds, grid reference, options
    assert cfg["grid"] == {"name": "test-lattice", "n_cells": lattice[0].n_cells, "sha256": lattice[0].sha256}
    again = LatticeCodec.from_config(cfg)
    assert again == codec and again.get_config() == cfg
    # a stray identifier (numcodecs-style id, codec entry name) is ignored; anything else is an error
    assert LatticeCodec.from_config({**cfg, "id": "icon_lattice", "name": "icon_lattice"}) == codec
    with pytest.raises(ValueError, match="bogus"):  # also a TypeError
        LatticeCodec.from_config({**cfg, "bogus": 1})
    assert len(repr(codec)) < 300
    back = pickle.loads(pickle.dumps(codec))  # picklable (the config is small): DataLoader workers, spawn
    assert back == codec and hash(back) == hash(codec)
    with pytest.raises(ValueError, match="unknown lattice codec options"):
        LatticeCodec(**make_config(lattice[0], bogus=1))
    import importlib.metadata

    eps = {e.name: e.value for e in importlib.metadata.entry_points(group="zarr.codecs")}
    assert eps["icon_lattice"] == "anemoi_datasets_codecs.icon_lattice.zarr3:IconLatticeCodec"
    assert eps["icon_lattice_bytes"] == "anemoi_datasets_codecs.icon_lattice.zarr3:IconLatticeBytesCodec"
    assert not [e for e in importlib.metadata.entry_points(group="numcodecs.codecs") if e.name == "icon_lattice"]
    # chunk shapes: whole dates of (variables, ensembles, cells)
    n, v = lattice[0].n_cells, len(NAMES)
    codec.validate_chunk_shape((1, v, 1, n))
    codec.validate_chunk_shape((3, v, 1, n))
    for bad in ((1, v - 1, 1, n), (1, v, 1, n - 1), ()):
        with pytest.raises(ValueError, match="whole dates"):
            codec.validate_chunk_shape(bad)


def test_config_validation(lattice):
    """Bounds by name or regular expression, clip, typos and contradictions fail at construction."""
    lay = lattice[0]
    c = LatticeCodec(
        variables=["T_850", "T_500", "U_850", "FIS", "CLCT"],
        grid=lay,
        bounds={"T_.*": 0.1, "T_500": 0.05, "U_850": {"max_error": 0.2, "clip": [None, 50]}, "CLCT": 1},
        clip={"CLCT": [0, 100]},
        static_fields=["FIS"],
    )
    got = {v.name: (v.max_error, v.clip) for v in c.variables}
    assert got == {
        "T_850": (0.1, None),
        "T_500": (0.05, None),  # the exact name wins over the pattern
        "U_850": (0.2, (None, 50.0)),
        "FIS": (None, None),
        "CLCT": (1.0, (0.0, 100.0)),
    }
    cfg = c.get_config()
    assert cfg["bounds"] == {"T_850": 0.1, "T_500": 0.05, "U_850": 0.2, "CLCT": 1.0}  # resolved, exact names
    assert LatticeCodec.from_config(cfg) == c
    for bad, match in (
        ({"bounds": {"T_2M": 0.1}}, "match no variable"),
        ({"bounds": {"T_850": -1}}, "> 0"),
        ({"bounds": {"T_850": 0.1, "FIS": 1.0}, "static_fields": ["FIS"]}, "cannot have a bound"),
        ({"static_fields": ["NOPE"]}, "not variables"),
        ({"clip": {"T_850": [2, 1]}}, "lower bound"),
        ({"structure": "nope"}, "unknown lattice codec structure"),
        ({"variables": ["A", "A"]}, "duplicate"),
    ):
        kw = {"variables": ["T_850", "FIS"], "grid": lay, **bad}
        with pytest.raises(ValueError, match=match):
            LatticeCodec(**kw)
    with pytest.raises(ValueError, match="unknown grid"):
        LatticeCodec(variables=["A"], grid="no-such-grid")


def test_zarr3_write_read(tmp_path, lattice, chunk, caplog):
    """A plain Zarr 3 array with the array-to-bytes codec: written without warnings (the statics captured into the
    store's sidecar are logged at level INFO, not warned), read back within bounds."""
    import zarr

    from anemoi_datasets_codecs.icon_lattice.zarr3 import IconLatticeCodec

    data = np.stack([chunk, chunk])[:, :, None, :]
    z = zarr.create_array(
        str(tmp_path / "x.zarr"),
        shape=data.shape,
        chunks=(1, *data.shape[1:]),
        dtype="float32",
        serializer=IconLatticeCodec(**make_config(lattice[0])),
        compressors=None,
        fill_value=np.nan,
    )
    with warnings.catch_warnings(), caplog.at_level("INFO", logger="anemoi_datasets_codecs.statics"):
        warnings.simplefilter("error")
        z[:] = data
    assert caplog.text.count("captured static field") == len(STATIC)
    assert len(list((tmp_path / "x.zarr" / sc.SIDECAR / "statics").glob("*.f4.blosc"))) == len(STATIC)
    statics.clear_caches()
    back = zarr.open_array(str(tmp_path / "x.zarr"), mode="r")
    assert isinstance(back.serializer, IconLatticeCodec) and back.serializer.codec == make_codec(lattice[0])
    check_bounds(back[1, :, 0], chunk)


# ---------------------------------------------------------------------------------------------- real data
@pytest.mark.realdata
def test_realdata_one_date():
    """Date 6 of the ICON-CH1 sample with the sample's bounds (inline statics, lz4 lossless fields)."""
    import zarr

    if not SAMPLE_STORE.exists():
        pytest.skip(f"sample store {SAMPLE_STORE} missing")
    z = zarr.open_group(str(SAMPLE_STORE), mode="r")
    x = z["data"][6, :, 0, :]
    cfg = json.loads(Path(SAMPLE_CONFIG).read_text())
    assert cfg["variables"] == list(z.attrs["variables"])
    cfg["lossless"] = {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1}
    cfg["options"] = {"statics": "inline"}
    codec = LatticeCodec(**cfg)
    enc = codec.encode(x, dates=[z.attrs["sample"]["source_dates"][6]])
    dec = codec.decode(enc).reshape(x.shape)
    recs = container.inspect_chunk(enc)
    for i, v in enumerate(codec.variables):
        err = np.abs(dec[i].astype(np.float64) - x[i]).max()
        if v.lossy:
            assert err <= v.max_error, v.name
            assert recs[i].mode == MODE_LATTICE, v.name
        elif recs[i].mode != MODE_RECIPE:
            assert err == 0, v.name
    assert len(enc) < 99.5 * 2**20  # the acceptance line, even with inline statics (about 26 MiB of lz4)


@pytest.mark.realdata
def test_realdata_tree_matches_grid_file():
    """The bundled ICON-CH1 lattice reproduces the grid file's edge neighbours, and the coordinate-derived
    refinement tree its parents."""
    if not GRID_FILE.name or not GRID_FILE.exists():
        pytest.skip("grid file missing (ANEMOI_DATASETS_CODECS_GRID_FILE)")
    pytest.importorskip("netCDF4")
    layout = grids.load_lattice("icon-ch1")
    g = gridfile.read_grid_file(GRID_FILE)
    gridfile.check_against_grid_file(layout, g)
    import netCDF4

    with netCDF4.Dataset(str(GRID_FILE)) as ds:
        parent = np.asarray(ds.variables["parent_cell_index"][:], np.int64) - 1
    tree = S.get_structure("tree", layout)
    finest = [s for s in tree.steps if s.name.startswith("L0")]
    assert finest[-1].name == "L0g" and {s.name for s in finest} <= {"L0hg", "L0hr", "L0g"}  # boundary, then regular
    assert tree.regular_groups() is not None and tree.regular_groups()[1] * 4 > 0.85 * layout.n_cells
    for l0 in finest:
        np.testing.assert_array_equal(parent[l0.src[0]], parent[l0.t])  # own central child = sibling


# ---------------------------------------------------------------------------------------------- stores
DECODE_IN_FRESH_PROCESS = """
import sys, numpy as np, zarr
np.save(sys.argv[2], zarr.open_group(sys.argv[1], mode="r")["data"][1])
"""


@pytest.mark.parametrize("statics_mode", ["ref", "inline"])
def test_store_readable_in_fresh_process(tmp_path, lattice, chunk, statics_mode):
    """Write a store with a registered sidecar (lattice and captured statics land in <store>/.codecs), then
    read it in a new process with a cold cache: only ANEMOI_DATASETS_CODECS_PATH=<store> tells where they are."""
    import zarr

    from anemoi_datasets_codecs.icon_lattice import register_store
    from anemoi_datasets_codecs.icon_lattice.zarr3 import IconLatticeCodec

    store = tmp_path / "s.zarr"
    data = np.stack([chunk, chunk])[:, :, None, :]
    g = zarr.open_group(str(store), mode="w")
    register_store(store)
    codec = IconLatticeCodec(**make_config(lattice[0], statics=statics_mode))
    a = g.create_array(
        "data", shape=data.shape, chunks=(1, *data.shape[1:]), dtype="float32", serializer=codec, compressors=None
    )
    a[:] = data
    side = store / sc.SIDECAR
    assert (side / "grids" / f"{lattice[0].sha256}.npz").is_file()
    n_statics = len(list((side / "statics").glob("*.f4.blosc"))) if (side / "statics").exists() else 0
    assert n_statics == (len(STATIC) if statics_mode == "ref" else 0)
    script = tmp_path / "read.py"
    script.write_text(DECODE_IN_FRESH_PROCESS)
    env = {**os.environ, sc.CACHE_ENV: str(tmp_path / "cold-cache"), sc.PATH_ENV: str(store)}
    out = tmp_path / "x.npy"
    subprocess.run([sys.executable, str(script), str(store), str(out)], check=True, env=env)
    check_bounds(np.load(out)[:, 0], chunk)
    # without the pipeline that serves the store and without the store on the search path: a clear error, not garbage
    env.pop(sc.PATH_ENV)
    env["ANEMOI_DATASETS_CODECS_PIPELINE"] = "off"
    r = subprocess.run([sys.executable, str(script), str(store), str(out)], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and ("not found" in r.stderr or "not available" in r.stderr)


def test_variable_subset_without_cross_sources(lattice):
    """A config with only a few T/U levels (no FIS, PS, PMSL, statics, forcings): plan terms whose sources are absent
    are dropped, the rest (T_700, T_925 <- T_850) still applies, and every field round-trips."""
    layout, rr, cc, *_ = lattice
    names = ["T_700", "T_850", "T_925", "U_850", "U_1000"]
    rng = np.random.default_rng(3)
    shared = rng.standard_normal(len(rr))
    x = np.stack([270 + 5 * smooth(rr, cc, k) + 0.2 * shared for k in range(len(names))]).astype(np.float32)
    codec = LatticeCodec(variables=names, grid=layout, bounds={".*": 0.05})
    assert codec.cross_plan == {"T_700": ["T_850"], "T_925": ["T_850"]}  # absent sources dropped
    enc = codec.encode(x)
    dec = codec.decode(enc).reshape(x.shape)
    assert np.abs(dec.astype(np.float64) - x).max() <= 0.05
    ch = container.parse(enc)
    assert ch.payloads[names.index("T_700")][4] == 1  # T_700 uses T_850 (shared signal)
    assert ch.payloads[names.index("T_850")][4] == 0
    # the same data in a different variable order and with an extra unrelated variable
    names2 = ["U_1000", "OTHER", "T_925", "T_850"]
    x2 = np.stack([x[4], x[0] * 0 + 1.5 + rng.standard_normal(len(rr)), x[2], x[1]]).astype(np.float32)
    codec2 = LatticeCodec(variables=names2, grid=layout, bounds={".*": 0.05})
    dec2 = codec2.decode(codec2.encode(x2)).reshape(x2.shape)
    assert np.abs(dec2.astype(np.float64) - x2).max() <= 0.05


@pytest.mark.parametrize("sid", ["tree", "hc2_k3"])
def test_residual_domain_reconstruction(lattice, chunk, sid):
    """recon_bits 3 (default): values on step / 8, residuals coded at the step; bound kept, RMSE lower than the plain
    rule; recon_bits 0 reproduces the plain rule; fields with many exact zeros keep the plain rule."""
    s = S.get_structure(sid, lattice[0])
    i = NAMES.index("PMSL")
    fine = encode_field(chunk[i], 2.0, None, s, EncodeOptions(recon_bits=3))
    plain = encode_field(chunk[i], 2.0, None, s, EncodeOptions(recon_bits=0))
    assert fine.payload[7] == 3 and plain.payload[7] == 0
    assert fine.info["max_err"] <= 2.0 and plain.info["max_err"] <= 2.0
    codec = make_codec(lattice[0], sid)
    dec = codec.decode(codec.encode(chunk, dates=[DATE])).reshape(chunk.shape)
    rmse = {}
    for opt in (0, 3):
        c = make_codec(lattice[0], sid, recon_bits=opt)
        d = c.decode(c.encode(chunk, dates=[DATE])).reshape(chunk.shape)
        check_bounds(d, chunk)
        k = NAMES.index("NOISY")
        rmse[opt] = float(np.sqrt(np.mean((d[k].astype(np.float64) - chunk[k]) ** 2)))
    assert rmse[3] < rmse[0]
    np.testing.assert_array_equal(dec, codec.decode(codec.encode(chunk, dates=[DATE])).reshape(chunk.shape))
    tp = container.parse(codec.encode(chunk, dates=[DATE])).payloads[NAMES.index("TOT_PREC_1H")]
    assert tp[7] == 0  # many exact zeros: plain rule, zeros stay exact
