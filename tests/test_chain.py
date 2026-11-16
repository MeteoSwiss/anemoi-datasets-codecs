"""Codec chains: field stages (``zarr3.FieldStage`` / ``FieldStageBytesCodec``) ahead of ``icon_lattice_bytes`` with
``chain: true``, through ``zarr.create_array(..., compressors=[...])`` on the synthetic lattice; the lattice codec's
chain mode; the plan of a lattice configuration without the static fields."""

from __future__ import annotations

import copy
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import zarr
from conftest import NAMES, STATIC, VARS, check_bounds, make_config
from numcodecs import Blosc
from zarr.registry import register_codec

from anemoi_datasets_codecs import envelope, physics, statics
from anemoi_datasets_codecs import plan as P
from anemoi_datasets_codecs.icon_lattice import LatticeCodec, container, register_store
from anemoi_datasets_codecs.icon_lattice.zarr3 import IconLatticeBytesCodec
from anemoi_datasets_codecs.zarr3 import FieldStage, FieldStageBytesCodec

SAMPLE_CONFIG = Path(__file__).parent / "data" / "icon_ch1_sample.json"


class DropFields(FieldStage):
    """Toy stage that changes the field count: drops ``drop``, their float32 values travel in the header."""

    def __init__(self, variables, drop):
        super().__init__(variables)
        self.drop = tuple(drop)
        self._d = [self.variables.index(n) for n in self.drop]
        self._k = [i for i in range(self.fields_in) if i not in self._d]

    @property
    def variables_out(self):
        return tuple(self.variables[i] for i in self._k)

    def encode(self, x):
        return x[:, self._k], np.ascontiguousarray(x[:, self._d], "<f4").tobytes()

    def decode(self, y, header, out_shape, out=None):
        x = np.empty(out_shape, np.float32) if out is None else out
        x[:, self._k] = y
        x[:, self._d] = np.frombuffer(header, "<f4").reshape(out_shape[0], len(self._d), *out_shape[2:])
        return x

    def get_config(self):
        return {**super().get_config(), "drop": list(self.drop)}


class Residual(FieldStage):
    """Toy stage that rewrites a field in place: ``target -> fl32(target - c source)``, c fitted per chunk (float64 in
    the header); decode ``fl32(r + c source)`` in place."""

    def __init__(self, variables, target, source):
        super().__init__(variables)
        self.target, self.source = target, source
        self._t, self._s = self.variables.index(target), self.variables.index(source)

    def encode(self, x):
        xt, xs = x[:, self._t].astype(np.float64), x[:, self._s].astype(np.float64)
        a = xs - xs.mean()
        c = float((a * (xt - xt.mean())).sum() / (a * a).sum())
        y = x.copy()
        y[:, self._t] = (xt - c * xs).astype(np.float32)
        return y, struct.pack("<d", c)

    def decode(self, y, header, out_shape, out=None):
        (c,) = struct.unpack("<d", header)
        y[:, self._t] = (y[:, self._t].astype(np.float64) + c * y[:, self._s].astype(np.float64)).astype(np.float32)
        return y

    def get_config(self):
        return {**super().get_config(), "target": self.target, "source": self.source}


class DropFieldsBytes(FieldStageBytesCodec):
    codec_name, engine = "test.drop_fields", DropFields


class ResidualBytes(FieldStageBytesCodec):
    codec_name, engine = "test.residual", Residual


register_codec(DropFieldsBytes.codec_name, DropFieldsBytes)
register_codec(ResidualBytes.codec_name, ResidualBytes)

T2M_BOUND = dict((n, b) for n, b, _ in VARS)["T_2M"]
KEPT = [n for n in NAMES if n not in STATIC]


def _lattice_cfg(layout, chain=True):
    """The lattice configuration behind the stages: the fields the drop stage leaves, no static fields, the residual
    target's bound reduced to leave room for the float32 roundings of the residual stage."""
    cfg = make_config(layout)
    cfg.update(variables=list(KEPT), static_fields=[], chain=chain)
    cfg["bounds"] = {**cfg["bounds"], "T_2M": 0.9 * T2M_BOUND}
    return cfg


def _stages():
    return [ResidualBytes(variables=NAMES, target="T_2M", source="FIS"), DropFieldsBytes(variables=NAMES, drop=STATIC)]


def _data(chunk, n_dates=3):
    return np.stack([chunk] * n_dates)[:, :, None, :]


@pytest.mark.parametrize("last", ["icon_lattice_bytes", "blosc"])
def test_chain_round_trip(tmp_path, lattice, chunk, last):
    """[residual (in place), drop (fewer fields), icon_lattice_bytes chain: true] and [residual, drop, blosc] round
    trip through a zarr store, within bounds (exact where nothing is lossy), read back by codecs rebuilt from
    zarr.json."""
    layout = lattice[0]
    path = tmp_path / "chain.zarr"
    g = zarr.open_group(str(path), mode="w")
    register_store(path)
    data = _data(chunk)
    tail = IconLatticeBytesCodec(**_lattice_cfg(layout)) if last != "blosc" else zarr.codecs.BloscCodec(cname="lz4")
    a = g.create_array(
        "data", shape=data.shape, chunks=(1, *data.shape[1:]), dtype="float32", compressors=[*_stages(), tail]
    )
    a[:] = data
    meta = json.loads((path / "data" / "zarr.json").read_text())
    assert [c["name"] for c in meta["codecs"]] == ["bytes", "test.residual", "test.drop_fields", last]
    raw = (path / "data" / "c" / "1" / "0" / "0" / "0").read_bytes()
    if last == "blosc":
        raw = Blosc().decode(raw)
    env = envelope.parse(raw)
    assert env is not None and env.shape == (1, len(KEPT), 1, data.shape[-1])
    assert [h[0] for h in env.headers] == ["test.residual", "test.drop_fields"]
    if last == "blosc":
        assert env.kind == envelope.KIND_ARRAY
    else:
        assert env.kind == envelope.KIND_BYTES
        cfg = meta["codecs"][-1]["configuration"]
        assert cfg["chain"] is True and cfg["variables"] == KEPT
        assert len(container.parse(env.payload).modes) == len(KEPT)  # the lattice coded the reduced field count
    statics.clear_caches()
    back = zarr.open_group(str(path), mode="r")["data"][...]
    t2m = NAMES.index("T_2M")
    for d in range(data.shape[0]):
        dec = back[d, :, 0]
        assert np.abs(dec[t2m].astype(np.float64) - chunk[t2m]).max() <= T2M_BOUND
        rest = [i for i in range(len(NAMES)) if i != t2m]
        if last == "blosc":  # nothing lossy: exact except the residual target (float32 roundings)
            assert np.array_equal(dec[rest].view(np.uint32), chunk[rest].view(np.uint32))
            assert np.abs(dec[t2m].astype(np.float64) - chunk[t2m]).max() <= 1e-4
        else:
            check_bounds(dec, chunk)


def test_chain_mode_validation(tmp_path, lattice, chunk):
    """Without ``chain`` the lattice codec still rejects a chunk grid with another field count at array creation;
    with ``chain: true`` it accepts the full grid and checks the field count per chunk instead."""
    layout = lattice[0]
    data = _data(chunk, 2)
    n = data.shape[-1]
    g = zarr.open_group(str(tmp_path / "v.zarr"), mode="w")
    register_store(tmp_path / "v.zarr")
    kw = {"shape": data.shape, "dtype": "float32"}
    with pytest.raises(ValueError, match="whole dates"):
        g.create_array(
            "strict",
            chunks=(1, 17, 1, n),
            compressors=[*_stages(), IconLatticeBytesCodec(**_lattice_cfg(layout, False))],
            **kw,
        )
    chained = IconLatticeBytesCodec(**_lattice_cfg(layout))
    g.create_array("ok", chunks=(1, 17, 1, n), compressors=[*_stages(), chained], **kw)
    with pytest.raises(ValueError, match="every field of a date"):
        g.create_array("split", chunks=(1, 6, 1, n), compressors=[chained], **kw)
    with pytest.raises(ValueError, match="ensembles, cells"):
        g.create_array(
            "cells", shape=(2, 17, 1, n + 1), chunks=(1, 17, 1, n + 1), dtype="float32", compressors=[chained]
        )
    # the per-chunk field count: the chain lattice alone on the 17-field array, and a stage whose variables differ
    a = g.create_array("alone", chunks=(1, 17, 1, n), compressors=[chained], **kw)
    with pytest.raises(ValueError, match="configured for 12 variables"):
        a[:] = data
    bad = ResidualBytes(variables=KEPT, target="T_2M", source="PMSL")
    b = g.create_array("badstage", chunks=(1, 17, 1, n), compressors=[bad, chained], **kw)
    with pytest.raises(ValueError, match="test.residual.*17 fields per date.*configured for 12"):
        b[:] = data
    # an in-place stage before a lattice codec without chain passes creation (same field count) and fails clearly
    full = make_config(layout)
    c = g.create_array("nochain", chunks=(1, 17, 1, n), compressors=[_stages()[0], IconLatticeBytesCodec(**full)], **kw)
    with pytest.raises(ValueError, match="needs chain: true"):
        c[:] = data


def test_chain_mode_bare_input_same_bytes(tmp_path, lattice, chunk):
    """``chain: true`` on a bare input writes the bytes of the codec without it; the key is in the configuration only
    when true, and survives from_config."""
    cfg = make_config(lattice[0])
    plain, chained = IconLatticeBytesCodec(**cfg), IconLatticeBytesCodec(**cfg, chain=True)
    assert "chain" not in plain.configuration and "chain" not in LatticeCodec(**cfg).get_config()
    assert chained.configuration["chain"] is True and chained != plain
    assert LatticeCodec.from_config(chained.codec.get_config()).chain is True
    assert "chain" in repr(chained.codec) and "chain" not in repr(plain.codec)
    with pytest.raises(ValueError, match="chain must be"):
        LatticeCodec(**cfg, chain="yes")
    data = _data(chunk, 1)
    files = []
    for name, codec in (("plain", plain), ("chained", chained)):
        path = tmp_path / f"{name}.zarr"
        g = zarr.open_group(str(path), mode="w")
        register_store(path)
        a = g.create_array("data", shape=data.shape, chunks=data.shape, dtype="float32", compressors=[codec])
        a[:] = data
        files.append((path / "data" / "c" / "0" / "0" / "0" / "0").read_bytes())
        check_bounds(zarr.open_group(str(path), mode="r")["data"][0, :, 0], chunk)
    assert files[0] == files[1] and files[0][:4] == container.MAGIC


def _strip_statics(p: P.Plan, st: set[str]) -> P.Plan:
    """``p`` without any static field: cross sources removed (a term left without sources dropped), mask and recipe
    terms that use one dropped."""
    cross = {t: [s for s in v if s not in st] for t, v in p.cross.items() if t not in st}
    masks = {t: m for t, m in p.masks.items() if t not in st and not set(P.mask_sources(m)) & st}
    recipes = {t: r for t, r in p.recipes.items() if t not in st and not set(r[1]) & st}
    return P.Plan({t: v for t, v in cross.items() if v}, masks, recipes, copy.deepcopy(p.report))


@pytest.mark.parametrize("physics_cfg", [None, {"below_ground": True}, {"cross": "on"}, {"cross": "chain"}])
def test_plan_without_statics(physics_cfg):
    """The lattice configuration without the static fields (behind a stage that takes them) resolves the plan of the
    full configuration minus every static source, before and after the codec's limits: it needs no ``plan`` key."""
    cfg = json.loads(SAMPLE_CONFIG.read_text())
    for names, st in ((cfg["variables"], set(cfg["static_fields"])), (list(NAMES), set(STATIC))):
        kept = [n for n in names if n not in st]
        full, reduced = physics.resolve(names, physics_cfg), physics.resolve(kept, physics_cfg)
        stripped = _strip_statics(full, st)
        assert (reduced.cross, reduced.masks, reduced.recipes) == (stripped.cross, stripped.masks, stripped.recipes)
        assert any(s in st for v in full.cross.values() for s in v)  # the full plan does use statics
        for max_depth in (1, 2, None):
            a = P.finalize(_strip_statics(full, st), names, max_depth, st)
            b = P.finalize(physics.resolve(kept, physics_cfg), kept, max_depth)
            assert (a.cross, a.masks) == (b.cross, b.masks), max_depth
    # and through the codec, as configured in a chain (sample configuration, grid by alias)
    base = {k: v for k, v in cfg.items() if k != "static_fields"}
    st = set(cfg["static_fields"])
    kept = [n for n in cfg["variables"] if n not in st]
    bounds = {k: v for k, v in cfg["bounds"].items() if k not in st}
    clip = {k: v for k, v in cfg.get("clip", {}).items() if k not in st}
    g = LatticeCodec(**{**base, "variables": kept, "bounds": bounds, "clip": clip, "chain": True}, physics=physics_cfg)
    f = LatticeCodec(
        **cfg, physics=physics_cfg, plan=_strip_statics(physics.resolve(cfg["variables"], physics_cfg), st)
    )
    assert g.cross_plan == f.cross_plan and g.plan.masks == f.plan.masks


# ------------------------------------------------------------------------------------------- chain consistency
def _expanded(layout):
    from anemoi_datasets_codecs.physics import expand

    entries = expand(NAMES, None, make_config(layout))
    assert [e["name"] for e in entries] == ["field_terms", "static_ref", "icon_lattice_bytes"]
    return entries


def _with(entries, i, **changes):
    out = copy.deepcopy(entries)
    out[i]["configuration"].update(changes)
    return out


def test_chain_rules(lattice):
    """chain_problems verifies, from the configurations alone, what no stage sees on its own: field_terms targets at
    most at b_L and unclipped in the lattice (a regular-expression clip counts), field_terms sources exact through
    the later codecs (no bound in the lattice, no recipe of its plan, not a later stage's target), and static_ref's
    output equal to the lattice's variables."""
    from anemoi_datasets_codecs import chain

    entries = _expanded(lattice[0])
    assert chain.chain_problems(entries) == []
    lat = entries[2]["configuration"]
    b = {**lat["bounds"], "T_2M": 0.05}  # the target's full bound b > b_L
    (p,) = chain.chain_problems(_with(entries, 2, bounds=b))
    assert "'T_2M'" in p and "more than b_L" in p
    b = {**lat["bounds"], "T_2M": 0.01}  # tighter than b_L: fine
    assert chain.chain_problems(_with(entries, 2, bounds=b)) == []
    for clip in ({**lat["clip"], "T_2M": [200, None]}, {**lat["clip"], "T_.*": [200, None]}):
        (p,) = chain.chain_problems(_with(entries, 2, clip=clip))
        assert "'T_2M'" in p and "clips it" in p
    # a source with a bound in the lattice: EXTRA is lossless for field_terms, lossy for the lattice
    ft = entries[0]["configuration"]
    bad = _with(entries, 0, terms={**ft["terms"], "PMSL": {"sources": ["FIS", "EXTRA"], "intercept": True}})
    assert chain.chain_problems(bad) == []
    (p,) = chain.chain_problems(_with(bad, 2, bounds={**lat["bounds"], "EXTRA": 0.1}))
    assert "'EXTRA'" in p and "lossy there" in p
    # a source that the lattice evaluates as a forcing recipe (its convention maps the role to EXTRA)
    conv = {"extends": "icon-ch1", "forcings": {"cos_local_time": "EXTRA"}}
    rec = _with(bad, 2, physics={"convention": conv}, static_fields=[])
    rec[2]["configuration"]["variables"] = NAMES  # the recipe inputs (static) stay in the lattice: no static_ref
    del rec[1]
    probs = chain.chain_problems(rec)
    assert any("'EXTRA'" in p and "forcing recipe" in p for p in probs), probs
    # a source that a later field_terms stage replaces by a residual
    later = {"name": "field_terms", "configuration": {"variables": NAMES, "terms": {"EXTRA": ["sin_latitude"]}}}
    later["configuration"]["bounds"] = {"EXTRA": 0.1}
    two = [bad[0], later, *bad[1:]]
    assert any("target of the later stage" in p for p in chain.chain_problems(two))
    # static_ref's output is not the lattice's variables
    short = _with(entries, 2, variables=lat["variables"][:-1])
    short[2]["configuration"]["bounds"] = {k: v for k, v in lat["bounds"].items() if k != lat["variables"][-1]}
    (p,) = chain.chain_problems(short)
    assert "'static_ref' emits the variables" in p


def test_check_reports_chain_rules(tmp_path, lattice, chunk):
    """A store written with a lattice bound above b_L (zarr writes it: no stage sees the other's configuration) is
    reported by check; the expanded chain passes."""
    from anemoi_datasets_codecs import chain
    from anemoi_datasets_codecs.icon_lattice import store

    entries = _expanded(lattice[0])
    data = _data(chunk, 1)
    for label, ents, ok in (
        ("good", entries, True),
        ("bad", _with(entries, 2, bounds={**entries[2]["configuration"]["bounds"], "PMSL": 2.0}), False),
    ):
        path = tmp_path / f"{label}.zarr"
        g = zarr.open_group(str(path), mode="w")
        register_store(path)
        codecs = [chain.codec_class(e["name"])(**e["configuration"]) for e in ents]
        a = g.create_array("data", shape=data.shape, chunks=data.shape, dtype="float32", compressors=codecs)
        a[:] = data
        g.attrs["variables"] = list(NAMES)
        r = store.check(path)
        assert (r["problems"] == []) is ok, r["problems"]
        if not ok:
            assert any("'PMSL'" in p and "b_L" in p for p in r["problems"])


# ------------------------------------------------------------------------------------------- three-codec chains
@pytest.mark.parametrize("last", ["icon_lattice_bytes", "blosc"])
def test_three_codec_chain(tmp_path, lattice, chunk, last):
    """[field_terms, static_ref, icon_lattice_bytes chain: true] and [field_terms, static_ref, blosc] through zarr,
    read back in a fresh process whose only sidecar source is ANEMOI_DATASETS_CODECS_PATH: every value within its
    bound, the static fields and every other exact field bit-identical; the stage targets within their bound (with
    Blosc within the float32 roundings of r + pred, not bit-exact in general). check passes on both."""
    from test_static_ref import _fresh_read

    from anemoi_datasets_codecs import chain
    from anemoi_datasets_codecs.icon_lattice import store

    entries = _expanded(lattice[0])
    ft = entries[0]["configuration"]
    codecs = [chain.codec_class(e["name"])(**e["configuration"]) for e in entries[:2]]
    if last == "blosc":
        codecs.append(zarr.codecs.BloscCodec(cname="lz4", clevel=5, shuffle="shuffle"))
    else:
        codecs.append(IconLatticeBytesCodec(**entries[2]["configuration"]))
    data = _data(chunk, 2)
    data[1, NAMES.index("T_2M")] += np.float32(0.4)
    path = tmp_path / f"three-{last}.zarr"
    g = zarr.open_group(str(path), mode="w")
    register_store(path)
    a = g.create_array("data", shape=data.shape, chunks=(1, *data.shape[1:]), dtype="float32", compressors=codecs)
    a[:] = data
    g.attrs["variables"] = list(NAMES)
    names = [c["name"] for c in json.loads((path / "data" / "zarr.json").read_text())["codecs"]]
    assert names == ["bytes", "field_terms", "static_ref", last]
    statics.clear_caches()
    back = _fresh_read(path, tmp_path / "back.npy", path)
    targets = [NAMES.index(t) for t in ft["terms"]]
    for d in range(2):
        check_bounds(back[d, :, 0], data[d, :, 0], exact_lossless=True)
        for i in targets:
            if last != "blosc":
                assert not np.array_equal(back[d, i, 0], data[d, i, 0])  # lossy residuals
            else:  # the residual kept exactly: within the roundings of r + pred (bit-exact or nearly)
                err = np.abs(back[d, i, 0].astype(np.float64) - data[d, i, 0]).max()
                assert err <= float(np.spacing(np.abs(data[d, i, 0]).max())), NAMES[i]
    if last == "blosc":
        exact = [i for i in range(len(NAMES)) if i not in targets]
        assert np.array_equal(back[:, exact].view(np.uint32), data[:, exact].view(np.uint32))
    r = store.check(path)
    assert r["problems"] == [] and r["chain"][:2] == ["field_terms", "static_ref"], r
    if last == "icon_lattice_bytes":  # info -v describes the field_terms header too
        import subprocess
        import sys

        chunk_file = path / "data" / "c" / "0" / "0" / "0" / "0"
        cmd = [sys.executable, "-m", "anemoi_datasets_codecs.cli", "info", str(chunk_file), "--config", str(path), "-v"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert "stage field_terms: bytes" in r.stdout and f"terms {len(ft['terms'])}, slices 1" in r.stdout
        t0 = next(iter(ft["terms"]))
        assert any(x.split()[:2] == [t0, "<-"] for x in r.stdout.splitlines()), r.stdout


# ------------------------------------------------------------------------------------------- tool robustness
def test_unknown_codec_reported(tmp_path):
    """A codec name zarr does not know is no class (not a crash): chain_problems reports it, and check on a JSON chain
    with it exits 1 with the problem and no traceback."""
    import subprocess
    import sys

    from anemoi_datasets_codecs import chain

    assert chain.codec_class("blosc_typo", missing_ok=True) is None and not chain.is_stage("blosc_typo")
    with pytest.raises(ValueError, match="'blosc_typo' is not known to zarr"):
        chain.codec_class("blosc_typo")
    ref = {"name": "static_ref", "configuration": {"variables": list(NAMES), "static_fields": list(STATIC)}}
    entries = [ref, {"name": "blosc_typo"}]
    (p,) = chain.chain_problems(entries)
    assert "'blosc_typo'" in p and "not known" in p
    with pytest.raises(ValueError, match="blosc_typo"):
        chain.strip_compressors(entries, b"")
    f = tmp_path / "chain.json"
    f.write_text(json.dumps(entries))
    r = subprocess.run([sys.executable, "-m", "anemoi_datasets_codecs.cli", "check", str(f)], capture_output=True)
    assert r.returncode == 1 and b"Traceback" not in r.stderr, r.stderr
    assert any("blosc_typo" in p for p in json.loads(r.stdout)["problems"])


def test_chain_rules_codecs(lattice):
    """The array-to-bytes icon_lattice after the stages, a lattice lossless codec that is not exact (bitround) for a
    field_terms source, and a codec that cannot be built (one problem, its neighbours not compared)."""
    from anemoi_datasets_codecs import chain

    entries = _expanded(lattice[0])
    a2b = copy.deepcopy(entries)
    a2b[2]["name"] = "icon_lattice"
    (p,) = chain.chain_problems(a2b)
    assert "'icon_lattice'" in p and "array-to-bytes" in p
    # EXTRA, a lossless field of the lattice, as a second source: exact only with an exact lossless codec there
    ft = entries[0]["configuration"]
    two = _with(entries, 0, terms={**ft["terms"], "PMSL": {"sources": ["FIS", "EXTRA"], "intercept": True}})
    for lossless in ({"id": "zstd", "level": 3}, {"id": "zlib", "level": 1}, {"id": "lz4"}):
        assert chain.chain_problems(_with(two, 2, lossless=lossless)) == [], lossless
    (p,) = chain.chain_problems(_with(two, 2, lossless={"id": "bitround", "keepbits": 10}))
    assert "'EXTRA'" in p and "'bitround'" in p
    assert chain.chain_problems(_with(entries, 2, lossless={"id": "bitround", "keepbits": 10})) == []  # FIS: static
    (p,) = chain.chain_problems(_with(entries, 1, static_fields=["NOPE"]))
    assert "'static_ref' cannot be built" in p
    (p,) = chain.chain_problems(_with(entries, 0, margin=2.0))
    assert "'field_terms' cannot be built" in p


@pytest.mark.parametrize(
    "entry, codec",
    [
        ({"name": "zstd", "configuration": {"level": 3}}, "Zstd"),
        ({"name": "gzip", "configuration": {"level": 5}}, "GZip"),
        ({"name": "numcodecs.zlib", "configuration": {"level": 1}}, "Zlib"),
        ({"name": "blosc", "configuration": {"cname": "zstd"}}, "Blosc"),
    ],
)
def test_strip_compressors(entry, codec):
    """The tools undo zarr's zstd, gzip, blosc and numcodecs.* compressors at the end of a chain."""
    import numcodecs

    from anemoi_datasets_codecs import chain

    payload = envelope.MAGIC + bytes(range(256)) * 8
    ref = {"name": "static_ref", "configuration": {"variables": list(NAMES), "static_fields": list(STATIC)}}
    rest, buf = chain.strip_compressors([ref, entry], getattr(numcodecs, codec)().encode(payload))
    assert rest == [ref] and bytes(buf) == payload


def _write(path, data, codecs):
    g = zarr.open_group(str(path), mode="w")
    register_store(path)
    a = g.create_array("data", shape=data.shape, chunks=(1, *data.shape[1:]), dtype="float32", compressors=codecs)
    a[:] = data
    g.attrs["variables"] = list(NAMES)
    statics.clear_caches()


def test_check_trailing_compressors(tmp_path, lattice, chunk):
    """check reports, not raises, a codec after the stages that the tools cannot undo ([static_ref, crc32c]); a
    single lattice codec followed by blosc is checked and decoded like a chain (the compressor undone first), and
    check --sample 0 scans every chunk."""
    from anemoi_datasets_codecs import chain
    from anemoi_datasets_codecs.icon_lattice import store
    from anemoi_datasets_codecs.static_ref.zarr3 import StaticRefCodec

    data = _data(chunk, 2)
    path = tmp_path / "crc.zarr"
    _write(path, data, [StaticRefCodec(variables=list(NAMES), static_fields=list(STATIC)), zarr.codecs.Crc32cCodec()])
    r = store.check(path)
    assert any("'crc32c'" in p for p in r["problems"]), r
    cfg = make_config(lattice[0])
    path = tmp_path / "single-blosc.zarr"
    _write(path, data, [IconLatticeBytesCodec(**cfg), zarr.codecs.BloscCodec(cname="lz4")])
    assert store.array_meta(path)["chain"] is None
    r = store.check(path, sample=0)
    assert r["problems"] == [] and r["chunks_checked"] == 2, r
    assert store.check(path, sample=1)["chunks_checked"] == 1
    p = next(store.chunk_paths(path))
    check_bounds(store.decode_chunk_file(path, p)[0, :, 0], chunk)
    assert chain.codec_class("crc32c") is not None
