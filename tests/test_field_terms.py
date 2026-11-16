"""The ``field_terms`` stage: configuration guards, the frozen header, the per-cell verification and its bypass,
decoding in place, and the chains ``[field_terms, icon_lattice_bytes chain: true]`` and ``[field_terms, blosc]``
through ``zarr.create_array(compressors=[...])`` on the synthetic lattice; on real data the six orography terms on
an ICON-CH1 date."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
import zarr
from conftest import NAMES, SAMPLE_CONFIG, SAMPLE_STORE, VARS, check_bounds, make_config
from numcodecs import Blosc

from anemoi_datasets_codecs import config, envelope, statics
from anemoi_datasets_codecs.field_terms import DEFAULT_MARGIN, FieldTerms, FieldTermsCodec, arith, header
from anemoi_datasets_codecs.icon_lattice import container, register_store
from anemoi_datasets_codecs.icon_lattice.zarr3 import IconLatticeBytesCodec

BOUNDS = {n: b for n, b, _ in VARS if b is not None}
TERMS = {"T_2M": ["FIS"], "PMSL": ["FIS"]}  # lossy targets, the static FIS as their exact source
TARGETS = ("T_2M", "PMSL")


def _stage(**kw) -> FieldTerms:
    return FieldTerms(**{"variables": NAMES, "terms": TERMS, "bounds": BOUNDS, **kw})


def _lattice_cfg(layout, stage: FieldTerms) -> dict:
    """The lattice configuration behind the stage: chain mode, the targets at the stage's reduced bound b_L."""
    cfg = make_config(layout)
    cfg.update(chain=True)
    cfg["bounds"] = {**cfg["bounds"], **stage.lattice_bounds}
    return cfg


def _data(chunk, n_dates=3):
    x = np.stack([chunk] * n_dates)[:, :, None, :].copy()
    for d in range(n_dates):  # dates that differ: a shift and a scale of the targets
        for t in TARGETS:
            i = NAMES.index(t)
            x[d, i, 0] = (x[d, i, 0].astype(np.float64) * (1 + 1e-4 * d) + 0.3 * d).astype(np.float32)
    return x


def _check_targets(dec, x, bounds=BOUNDS):
    for t in TARGETS:
        i = NAMES.index(t)
        assert np.abs(dec[i].astype(np.float64) - x[i]).max() <= bounds[t], t


# ---------------------------------------------------------------------------------------------- configuration
def test_config_canonical():
    s = _stage(bounds={"T_2M": 0.05, "PMSL": 2.0, "CLCT": 0.5, "TOT_.*": 1e-5})  # shared bounds: others ignored
    cfg = s.get_config()
    assert cfg == {
        "variables": NAMES,
        "terms": {"T_2M": {"sources": ["FIS"], "intercept": True}, "PMSL": {"sources": ["FIS"], "intercept": True}},
        "bounds": {"T_2M": 0.05, "PMSL": 2.0},
        "margin": DEFAULT_MARGIN,
    }
    assert FieldTerms(**json.loads(json.dumps(cfg))) == s and s.targets == TARGETS
    assert s.lattice_bounds == {"T_2M": 0.05 * (1 - 2**-6), "PMSL": 2.0 * (1 - 2**-6)}
    # regex bounds, per-term and global intercept, clips of targets
    s2 = FieldTerms(
        variables=NAMES,
        terms={"PMSL": {"sources": ["FIS"], "intercept": True}, "T_2M": ["FIS", "EXTRA"]},
        bounds={"T_.*": 0.05, "PMSL": {"max_error": 2.0, "clip": [0, None]}},
        clip={"T_2M": [200, 330]},
        intercept=False,
        margin=0.1,
    )
    cfg2 = s2.get_config()
    assert list(cfg2["terms"]) == ["T_2M", "PMSL"]  # in variable order
    assert cfg2["terms"]["T_2M"] == {"sources": ["FIS", "EXTRA"], "intercept": False}
    assert cfg2["terms"]["PMSL"]["intercept"] is True
    assert cfg2["clip"] == {"T_2M": [200.0, 330.0], "PMSL": [0.0, None]} and cfg2["margin"] == 0.1
    codec = FieldTermsCodec(**cfg2)
    assert FieldTermsCodec.from_dict(codec.to_dict()) == codec and codec.to_dict()["name"] == "field_terms"


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"terms": {"EXTRA": ["FIS"]}}, "target has no bound"),
        ({"terms": {"T_2M": ["PMSL"]}}, "'PMSL' has a bound"),
        ({"terms": {"T_2M": ["FIS"], "PMSL": ["T_2M"]}}, "'T_2M' is a target"),
        ({"terms": {"T_2M": ["insolation"]}}, "forcing recipe"),
        ({"terms": {"T_2M": ["NOPE"]}}, r"sources \['NOPE'\] are not variables"),
        ({"terms": {"NOPE": ["FIS"]}}, "not variables"),
        ({"terms": {"T_.*": ["FIS"]}}, "exact names"),
        ({"terms": {"T_2M": ["FIS", "FIS"]}}, "repeat"),
        ({"terms": {"T_2M": ["T_2M"]}}, "contain the target"),
        ({"terms": {"T_2M": []}}, "non-empty list"),
        ({"terms": {"T_2M": "FIS"}}, "non-empty list"),
        ({"terms": {"T_2M": {"sources": ["FIS"], "fit": "ls"}}}, "unknown keys"),
        ({"terms": {"T_2M": {"sources": ["FIS"], "intercept": 1}}}, "intercept must be"),
        ({"terms": {}}, "non-empty mapping"),
        ({"clip": {"CLCT": [0, 100]}}, "targets only"),
        ({"bounds": {**BOUNDS, "FIS": {"max_error": None, "clip": [0, None]}}}, "targets only"),
        ({"clip": {"T_2M": [300, 200]}}, "lower bound"),
        ({"bounds": {**BOUNDS, "ZZZ": 1.0}}, "match no variable"),
        ({"margin": 1.0}, "margin"),
        ({"margin": -0.1}, "margin"),
        ({"margin": True}, "margin"),
        ({"intercept": "yes"}, "intercept must be"),
        ({"variables": [*NAMES, "T_2M"]}, "duplicate"),
    ],
)
def test_config_guards(kw, match):
    with pytest.raises(ValueError, match=match):
        _stage(**kw)


def test_config_unknown_key():
    with pytest.raises(TypeError, match="unknown field_terms configuration keys"):
        _stage(sources=["FIS"])


def test_nonfinite_raises(chunk):
    x = _data(chunk, 2)
    x[1, NAMES.index("FIS"), 0, 5] = np.nan
    with pytest.raises(ValueError, match="'FIS' has non-finite values \\(date offset 1, member 0\\)"):
        _stage().encode(x)
    x = _data(chunk, 1)
    x[0, NAMES.index("PMSL"), 0, 0] = np.inf
    with pytest.raises(ValueError, match="'PMSL' has non-finite"):
        _stage().encode(x)
    x[0, NAMES.index("PMSL"), 0, 0] = 0
    x[0, NAMES.index("NOISY"), 0, 0] = np.nan  # not the stage's business
    _stage().encode(x)


# ---------------------------------------------------------------------------------------------- arithmetic, header
def test_endpoints_cover_the_accepted_values():
    """Every float32 r' that the lattice's check |r' - r| <= b (float64) accepts lies in [lo, hi]."""
    rng = np.random.default_rng(3)
    r = np.concatenate(
        [rng.standard_normal(2000) * 10.0 ** rng.integers(-30, 30, 2000), [0.0, 1.0, -1.0, 0.5, 2.0**-140]]
    ).astype(np.float32)
    for b in (0.05, 2.0, 1e-6, 0.5):
        lo, hi = arith.endpoints(r, b)
        r64 = r.astype(np.float64)
        assert (lo.dtype, hi.dtype) == (np.float32, np.float32)
        assert (lo <= r).all() and (hi >= r).all()
        # the next float32 value beyond each end is rejected, or the end is within 2 ulp of the ideal end
        for end, way in ((lo, -np.inf), (hi, np.inf)):
            beyond = np.nextafter(end, np.float32(way))
            assert not (np.abs(beyond.astype(np.float64) - r64) <= b).any()


def test_header_pinned_bytes():
    """A fixed input gives fixed header bytes (format version 1): exact relations (fits exact in float32), two
    dates, a term without intercept with two sources, and a bypassed target on date 1 (values near 1e7, whose
    float32 spacing 1 exceeds the bound 0.6 after any perturbation of the residual)."""
    n = 16
    s = np.arange(n, dtype=np.float32)
    c = (np.arange(n) * 7 % 5).astype(np.float32)
    x = np.zeros((2, 4, 1, n), np.float32)
    x[:, 1, 0], x[:, 3, 0] = s, c
    x[0, 0, 0] = 3 + 2 * s
    x[1, 0, 0] = 1e7 + 2 * s
    x[:, 2, 0] = 0.5 * s + 4 * c
    st = FieldTerms(
        variables=["A", "S", "B", "C"],
        terms={"B": {"sources": ["S", "C"], "intercept": False}, "A": ["S"]},
        bounds={"A": 0.6, "B": 0.01},
    )
    y, h = st.encode(x)
    assert h.hex() == (
        "010002000200000000000100010002000200010003000000"  # version, T, S; terms (A: S), (B: S, C)
        "000000000000404000000040"  # date 0, A: c0 3, c1 2
        "00000000000000000000003f00008040"  # date 0, B: c0 0, 0.5, 4
        "010000000000000000000000"  # date 1, A: bypassed
        "00000000000000000000003f00008040"  # date 1, B
    )
    assert len(h) == header.size(((0, (1,)), (2, (1, 3))), 2)
    assert not y[0, 0].any() and not y[:, 2].any()  # exact fits: zero residuals
    assert np.array_equal(y[1, 0], x[1, 0]) and np.array_equal(y[:, [1, 3]], x[:, [1, 3]])
    recs = st.records(h, x.shape)
    assert [[r.bypass for r in row] for row in recs] == [[False, False], [True, False]]
    # info: the header describes itself (term table, slices, bypass flags, coefficients), names given a configuration
    d = FieldTerms.describe_header(h, st.get_config())
    assert (d["terms"], d["slices"], d["bypassed"]) == (2, 2, 1)
    assert [(t["target"], t["sources"], t["bypassed"]) for t in d["per_term"]] == [
        ("A", ["S"], 1),
        ("B", ["S", "C"], 0),
    ]
    assert d["per_term"][0]["coefficients"][1] is None and d["per_term"][1]["coefficients"][0] == [0.0, 0.5, 4.0]
    assert FieldTerms.describe_header(h)["per_term"][1]["sources"] == [1, 3]
    z = y.copy()
    assert np.array_equal(st.decode(z, h, z.shape), x)
    # a configuration with other terms, another slice count, a corrupt header: clear errors
    other = FieldTerms(variables=["A", "S", "B", "C"], terms={"A": ["S"]}, bounds={"A": 0.6, "B": 0.01})
    with pytest.raises(ValueError, match="written with the terms"):
        other.decode(y.copy(), h, y.shape)
    with pytest.raises(ValueError, match="2 slices, the chunk has 1"):
        st.decode(y[:1].copy(), h, (1, 4, 1, n))
    with pytest.raises(ValueError, match="version 2"):
        st.decode(y.copy(), b"\2" + h[1:], y.shape)
    bad = bytearray(h)
    bad[24] = 2  # flags of the first record
    with pytest.raises(ValueError, match="unknown flags"):
        st.decode(y.copy(), bytes(bad), y.shape)
    with pytest.raises(ValueError, match="bytes, expected"):
        st.decode(y.copy(), h + b"\0" * 4, y.shape)


def test_bypass_when_the_margin_fails(chunk):
    """A slice whose target values are so large that float32 rounding of the reconstruction eats the margin is left
    untouched (flag in the header); the other slices keep their residuals."""
    x = _data(chunk, 2)
    t = NAMES.index("T_2M")
    x[1, t, 0] += np.float32(3e5)  # float32 spacing 1/32 at 3e5: the rounding exceeds b - b_L = 0.05 / 64
    st = _stage()
    y, h = st.encode(x)
    recs = st.records(h, x.shape)
    assert [[r.bypass for r in row] for row in recs] == [[False, False], [True, False]]
    assert np.array_equal(y[1, t, 0].view(np.uint32), x[1, t, 0].view(np.uint32))
    assert not np.array_equal(y[0, t, 0], x[0, t, 0])
    assert not recs[1][0].coefs.any()


def _within(y, bound, up):
    """The float32 values farthest from ``y`` (below, or above with ``up``) that are within ``bound`` of it."""
    target = y.astype(np.float64) + (bound if up else -bound)
    v = target.astype(np.float32)
    off = v.astype(np.float64) > target if up else v.astype(np.float64) < target
    return np.where(off, np.nextafter(v, np.float32(-np.inf if up else np.inf)), v)


def test_bypassed_slice_is_clipped(chunk):
    """The stage owns its targets' clips: a bypassed slice comes back from the codec after the stage unclipped
    (anywhere within b_L), and the stage clips it on decode, within the bound. A slice outside its clip by more than
    its bound cannot be coded so: ValueError at encode."""
    x = _data(chunk, 1)
    t = NAMES.index("T_2M")
    x[0, t, 0] += np.float32(3e5)  # bypassed (float32 rounding eats the margin)
    lo, hi = float(x[0, t, 0].min()), float(x[0, t, 0].max())
    st = _stage(clip={"T_2M": [lo, hi]})
    y, h = st.encode(x)
    assert st.records(h, x.shape)[0][0].bypass
    b, b_l = BOUNDS["T_2M"], st.lattice_bounds["T_2M"]
    for up in (False, True):
        z = y.copy()
        z[0, t, 0] = _within(y[0, t, 0], b_l, up)  # what the lattice may return, at the end of its bound
        assert (z[0, t, 0] < np.float32(lo)).any() or (z[0, t, 0] > np.float32(hi)).any()
        dec = st.decode(z, h, z.shape)[0, t, 0]
        assert dec.min() >= np.float32(lo) and dec.max() <= np.float32(hi), up
        assert np.abs(dec.astype(np.float64) - x[0, t, 0]).max() <= b
    near = _stage(clip={"T_2M": [lo + 0.5 * b, None]})  # outside by at most b: clipped within the bound
    y, h = near.encode(x)
    z = y.copy()
    z[0, t, 0] = _within(y[0, t, 0], b_l, False)
    dec = near.decode(z, h, z.shape)[0, t, 0]
    assert np.abs(dec.astype(np.float64) - x[0, t, 0]).max() <= b and dec.min() >= np.float32(lo + 0.5 * b)
    with pytest.raises(ValueError, match="outside its clip"):
        _stage(clip={"T_2M": [lo + 2 * b, None]}).encode(x)


def test_decode_in_place(chunk):
    """The engine decodes into the array it is given; through the adapter the decoded chunk is a view of the
    buffer the codecs after the stage handed on."""
    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.dtype import parse_dtype

    x = _data(chunk, 2)
    st = _stage()
    y, h = st.encode(x)
    z = y.copy()
    out = st.decode(z, h, z.shape)
    assert out is z
    check_bounds(out[0, :, 0], x[0, :, 0])
    codec = FieldTermsCodec(**st.get_config())
    proto = default_buffer_prototype()
    spec = ArraySpec(x.shape, parse_dtype("float32", zarr_format=3), 0.0, ArrayConfig.from_dict({}), proto)
    enc = codec._encode_sync(proto.buffer.from_array_like(x.reshape(-1).view(np.uint8)), spec)
    env = envelope.parse(enc.as_numpy_array())
    assert env.headers == [("field_terms", h)] and np.array_equal(env.array(), y)
    writable = np.frombuffer(enc.to_bytes(), np.uint8).copy()
    dec = codec._decode_sync(proto.buffer.from_array_like(writable), spec).as_numpy_array()
    assert np.shares_memory(dec, writable)
    assert np.array_equal(dec.view("<f4").reshape(x.shape), out)


def test_members_are_separate_slices(chunk):
    """Two dates x two members: one record per (date, member), date-major; each slice fitted on its own."""
    x = np.repeat(_data(chunk, 2), 2, axis=2)
    t = NAMES.index("T_2M")
    x[:, t, 1] += np.float32(1.5)  # member 1 offset: its own intercept
    x[1, t, 1] += np.float32(3e5)  # bypassed in (date 1, member 1) only
    st = _stage()
    y, h = st.encode(x)
    recs = st.records(h, x.shape)
    assert [r[0].bypass for r in recs] == [False, False, False, True]
    assert abs(float(recs[1][0].coefs[0]) - float(recs[0][0].coefs[0]) - 1.5) < 0.01
    z = st.decode(y.copy(), h, y.shape)
    for d in range(2):
        for e in range(2):
            _check_targets(z[d, :, e], x[d, :, e])


# ---------------------------------------------------------------------------------------------- chains
def _store(tmp_path, data, compressors, name="chain.zarr"):
    path = tmp_path / name
    g = zarr.open_group(str(path), mode="w")
    register_store(path)
    a = g.create_array("data", shape=data.shape, chunks=(1, *data.shape[1:]), dtype="float32", compressors=compressors)
    a[:] = data
    statics.clear_caches()
    return path, zarr.open_group(str(path), mode="r")["data"][...]


@pytest.mark.filterwarnings("ignore:lattice codec failed on variable 'T_2M':UserWarning")  # the bypassed date
def test_chain_field_terms_lattice(tmp_path, lattice, chunk):
    """[field_terms, icon_lattice_bytes chain: true] round trips within every bound (the targets coded by the lattice
    at b_L, within b after the stage); a slice whose target the stage bypasses is coded raw by the lattice at b_L."""
    layout = lattice[0]
    st = _stage()
    data = _data(chunk, 3)
    t = NAMES.index("T_2M")
    data[2, t, 0] += np.float32(3e5)  # bypassed on date 2
    path, back = _store(
        tmp_path, data, [FieldTermsCodec(**st.get_config()), IconLatticeBytesCodec(**_lattice_cfg(layout, st))]
    )
    meta = json.loads((path / "data" / "zarr.json").read_text())
    assert [c["name"] for c in meta["codecs"]] == ["bytes", "field_terms", "icon_lattice_bytes"]
    assert {k: v for k, v in meta["codecs"][1]["configuration"].items() if k != "written_by"} == st.get_config()
    assert meta["codecs"][1]["configuration"]["written_by"] == config.writer_stamp()
    for d in range(3):
        raw = (path / "data" / "c" / str(d) / "0" / "0" / "0").read_bytes()
        env = envelope.parse(raw)
        assert env.kind == envelope.KIND_BYTES and [k for k, _ in env.headers] == ["field_terms"]
        recs = st.records(env.headers[0][1], env.shape)
        assert [r.bypass for r in recs[0]] == [d == 2, False]
        assert len(container.parse(env.payload).modes) == len(NAMES)
        dec = back[d, :, 0]
        check_bounds(dec, data[d, :, 0])  # every variable within its bound b, the exact ones exact
    # the stage targets went through the lattice at b_L: residuals, so not bit-exact
    assert not np.array_equal(back[0, t, 0], data[0, t, 0])


def test_chain_field_terms_blosc(tmp_path, chunk):
    """[field_terms, blosc] (the default bytes codec first): Blosc keeps the residual exactly, so a stage target
    decodes to clip(fl32(r + pred)): within its bound but not bit-exact in general (float32 roundings of r and of the
    sum); every other field is exact."""
    st = _stage()
    data = _data(chunk, 2)
    path, back = _store(tmp_path, data, [FieldTermsCodec(**st.get_config()), zarr.codecs.BloscCodec(cname="lz4")])
    raw = Blosc().decode((path / "data" / "c" / "0" / "0" / "0" / "0").read_bytes())
    env = envelope.parse(raw)
    assert env.kind == envelope.KIND_ARRAY and [k for k, _ in env.headers] == ["field_terms"]
    y, h = st.encode(data[:1])
    assert env.headers[0][1] == h and np.array_equal(env.array(), y)
    ti = [NAMES.index(t) for t in TARGETS]
    rest = [i for i in range(len(NAMES)) if i not in ti]
    assert np.array_equal(back[:, rest].view(np.uint32), data[:, rest].view(np.uint32))
    for d in range(2):
        _check_targets(back[d, :, 0], data[d, :, 0])
        for i in ti:  # within the float32 roundings: ulp(r) / 2 + ulp(x) / 2
            err = np.abs(back[d, i, 0].astype(np.float64) - data[d, i, 0]).max()
            assert err <= float(np.spacing(np.abs(data[d, i, 0]).max())), NAMES[i]


# ---------------------------------------------------------------------------------------------- real data
RULE = ("PMSL", "FI_1000", "T_2M", "PS", "SKT", "TD_2M")  # the six orography terms
FORCINGS = ("cos_local_time", "sin_local_time", "insolation")
REFERENCE_TREE_K1 = 38_478_132  # bytes of a reference measurement: tree k = 1, sample date 6, statics removed
STATIC_REF_RECORDS = 624  # the 13 STATIC_REF records this test keeps in the lattice


@pytest.mark.realdata
def test_realdata_orography_terms(tmp_path):
    """Sample date 6, the reference measurement's setup (forcing recipes dropped, sample bounds, lz4, ``tree``
    k = 1): the chain [field_terms (the six orography terms on FIS), icon_lattice_bytes chain: true] with the
    statics kept as STATIC_REF in the lattice and FIS removed from the lattice's convention (the plan of the lattice
    without the statics). Every value within its bound, every exact field exact; the size is printed against the
    reference."""
    if not SAMPLE_STORE.exists():
        pytest.skip(f"sample store {SAMPLE_STORE} missing")
    z = zarr.open_group(str(SAMPLE_STORE), mode="r")
    allnames = list(z.attrs["variables"])
    cfg = json.loads(SAMPLE_CONFIG.read_text())
    names = [v for v in cfg["variables"] if v not in FORCINGS]
    x = np.ascontiguousarray(z["data"][6, [allnames.index(v) for v in names], 0, :])[None, :, None, :]
    stage = FieldTerms(variables=names, terms={t: ["FIS"] for t in RULE}, bounds=cfg["bounds"])
    bounds = {k: v for k, v in cfg["bounds"].items() if k in names}
    no_fis = {"convention": {"extends": "icon-ch1", "surface": {"FIS": None}}}
    lattice_cfg = {
        "variables": names,
        "bounds": {**bounds, **stage.lattice_bounds},
        "clip": {k: v for k, v in cfg["clip"].items() if k in names},
        "static_fields": [v for v in cfg["static_fields"] if v in names],
        "grid": cfg["grid"],
        "structure": "tree",
        "lossless": {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1},
        "options": {"k": 1.0},
        "physics": no_fis,
        "chain": True,
    }
    lattice = IconLatticeBytesCodec(**lattice_cfg)
    assert not any("FIS" in v for v in lattice.codec.cross_plan.values())
    t0 = time.perf_counter()
    path, back = _store(tmp_path, x, [FieldTermsCodec(**stage.get_config()), lattice])
    seconds = time.perf_counter() - t0
    raw = (path / "data" / "c" / "0" / "0" / "0" / "0").read_bytes()
    env = envelope.parse(raw)
    recs = stage.records(env.headers[0][1], env.shape)[0]
    for tm, rec in zip(stage.terms, recs, strict=True):
        print(f"{tm.target}: bypass {rec.bypass}, c0 {rec.coefs[0]:.8g}, c_FIS {rec.coefs[1]:.6g}")
    assert not any(r.bypass for r in recs)
    worst, worst_stage = 0.0, 0.0
    for i, n in enumerate(names):
        err = float(np.abs(back[0, i, 0].astype(np.float64) - x[0, i, 0]).max())
        if n in bounds:
            assert err <= bounds[n], (n, err, bounds[n])
            worst = max(worst, err / bounds[n])
            if n in RULE:
                worst_stage = max(worst_stage, err / bounds[n])
        else:
            assert np.array_equal(back[0, i, 0].view(np.uint32), x[0, i, 0].view(np.uint32)), n
    ref = REFERENCE_TREE_K1 + STATIC_REF_RECORDS
    print(
        f"field_terms + icon_lattice_bytes: {len(raw)} bytes ({len(raw) / 2**20:.3f} MiB), reference + "
        f"STATIC_REF records {ref} ({ref / 2**20:.3f} MiB): {100 * (len(raw) - ref) / ref:+.3f}%; worst error / "
        f"bound {worst:.4f} (stage targets {worst_stage:.4f}); write {seconds:.1f} s"
    )


def _spawn_roundtrip(codec, x):  # in a spawned worker: the codec arrives pickled, as in a DataLoader worker
    from anemoi_datasets_codecs import envelope as env_mod

    y, header = codec.codec.encode(x)
    back = codec.codec.decode(np.array(y), header, x.shape)
    return env_mod.MAGIC, header, back


def test_pickle_and_spawn(chunk):
    """FieldTerms pickles as its configuration, like StaticRef (DataLoader workers started with spawn), and so does
    its Zarr codec; a spawned worker encodes and decodes the same bytes and values as this process."""
    import multiprocessing as mp
    import pickle

    st = _stage()
    assert pickle.loads(pickle.dumps(st)) == st and pickle.loads(pickle.dumps(st)).terms == st.terms
    assert pickle.loads(pickle.dumps(st)).__dict__.keys() == st.__dict__.keys()
    codec = FieldTermsCodec(**st.get_config())
    assert pickle.loads(pickle.dumps(codec)) == codec
    x = _data(chunk, 2)
    y, header = st.encode(x)
    want = st.decode(np.array(y), header, x.shape)
    with mp.get_context("spawn").Pool(1) as pool:
        _, h2, back = pool.apply(_spawn_roundtrip, (codec, x))
    assert h2 == header and np.array_equal(back.view(np.uint32), want.view(np.uint32))
