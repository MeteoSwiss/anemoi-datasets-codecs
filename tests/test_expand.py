"""``physics expand``: the codec chain [field_terms, static_ref, icon_lattice_bytes] of a lattice configuration, on the
ICON-CH1 sample names and on the synthetic names; the degenerate cases (no chain); the margin from the data range;
anemoi's variable order; the command line."""

from __future__ import annotations

import copy
import json
import subprocess
import sys

import numpy as np
import pytest
import zarr
from conftest import NAMES, SAMPLE_CONFIG, STATIC, check_bounds, make_config

from anemoi_datasets_codecs import chain, physics, statics
from anemoi_datasets_codecs.field_terms import DEFAULT_MARGIN, FieldTerms
from anemoi_datasets_codecs.icon_lattice import LatticeCodec, register_store
from anemoi_datasets_codecs.physics.expansion import MARGIN_MAX, margin_for
from anemoi_datasets_codecs.static_ref import StaticRef

LZ4 = {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1}
CH1_TERMS = ("FI_1000", "PMSL", "PS", "SKT", "TD_2M", "T_2M")  # the six orography terms, in variable order


def _ch1() -> dict:
    return {**json.loads(SAMPLE_CONFIG.read_text()), "lossless": dict(LZ4)}


def _by_name(entries) -> dict:
    return {e["name"]: e["configuration"] for e in entries}


def test_expand_ch1_sample():
    """CH1 names: field_terms takes the six orography terms on FIS (bounds and clips of the targets, the default
    margin), static_ref the 13 static fields, the lattice the other 102 fields with the targets at b_L, no clip on
    them, no static fields, chain mode and a plan without FIS; the result is small JSON and consistent."""
    cfg = _ch1()
    entries = physics.expand(cfg["variables"], None, cfg)
    assert [e["name"] for e in entries] == ["field_terms", "static_ref", "icon_lattice_bytes"]
    by = _by_name(entries)
    ft, sr, lat = by["field_terms"], by["static_ref"], by["icon_lattice_bytes"]
    assert ft["variables"] == cfg["variables"] and sr["variables"] == cfg["variables"]
    assert list(ft["terms"]) == list(CH1_TERMS)
    assert all(v == {"sources": ["FIS"], "intercept": True} for v in ft["terms"].values())
    assert ft["bounds"] == {t: cfg["bounds"][t] for t in CH1_TERMS} and ft["margin"] == DEFAULT_MARGIN
    assert "clip" not in ft  # none of the targets has a clip
    assert sr["static_fields"] == sorted(cfg["static_fields"]) == cfg["static_fields"]
    kept = [n for n in cfg["variables"] if n not in cfg["static_fields"]]
    assert lat["variables"] == kept and lat["static_fields"] == [] and lat["chain"] is True
    stage = FieldTerms(**ft)
    for n in kept:
        want = stage.lattice_bounds.get(n, cfg["bounds"].get(n))
        assert lat["bounds"].get(n) == want, n
    assert lat["clip"] == cfg["clip"]  # the clipped fields (clouds, CAPE, LPI) are no targets
    assert lat["grid"]["name"] == "icon-ch1" and lat["lossless"] == LZ4
    codec = LatticeCodec.from_config(lat)
    assert not any("FIS" in s for s in codec.cross_plan.values()) and codec.cross_plan
    assert chain.chain_problems(entries) == []
    assert json.loads(json.dumps(entries)) == entries and len(json.dumps(entries)) < 16_000
    # the margin from the sample's range (PMSL ~ 1e5 Pa at b = 2 Pa needs 4 ulp / b ~ 2^-6) and anemoi's order
    pmsl = (95_000.0, 105_000.0)
    m, need = margin_for({"PMSL": 2.0, "T_2M": 0.05}, {"PMSL": pmsl, "T_2M": (230.0, 320.0)})
    assert m == 2.0**-6 and need["T_2M"] < 2.0**-8
    assert physics.anemoi_order(reversed(cfg["variables"])) == cfg["variables"]


def test_expand_forcing_bound():
    """forcing_bound: the forcings without a bound get it (chain, and single codec without recipe_fit); an exact key,
    even null, stays; without it nothing changes; the single codec with recipe_fit (the default) keeps them as
    recipes."""
    cfg = _ch1()
    forc = ("cos_local_time", "insolation", "sin_local_time")
    plain = _by_name(physics.expand(cfg["variables"], None, cfg))["icon_lattice_bytes"]
    assert not any(n in plain["bounds"] for n in forc)
    lat = _by_name(physics.expand(cfg["variables"], None, cfg, forcing_bound=1e-6))["icon_lattice_bytes"]
    assert {n: lat["bounds"][n] for n in forc} == dict.fromkeys(forc, 1e-6)
    assert {k: v for k, v in lat["bounds"].items() if k not in forc} == plain["bounds"]
    kept = {**cfg, "bounds": {**cfg["bounds"], "insolation": None, "cos_local_time": 1e-5}}
    lat = _by_name(physics.expand(cfg["variables"], None, kept, forcing_bound=1e-6))["icon_lattice_bytes"]
    assert "insolation" not in lat["bounds"] and lat["bounds"]["cos_local_time"] == 1e-5
    assert lat["bounds"]["sin_local_time"] == 1e-6
    off = {**cfg, "options": {"statics": "off"}, "physics": {"recipe_fit": False}}
    single = physics.expand(cfg["variables"], None, off, forcing_bound=1e-6)
    assert [e["name"] for e in single] == ["icon_lattice_bytes"]
    assert all(single[0]["configuration"]["bounds"][n] == 1e-6 for n in forc)
    assert single[0]["configuration"]["physics"]["recipe_fit"] is False
    # the single codec with recipe_fit (the default) stores them as recipes: a lossy field is never a recipe, so no
    # bound there; spelled out or not, the same configuration
    for fit in ({**cfg, "options": {"statics": "off"}}, {**off, "physics": {"recipe_fit": True}}):
        single = physics.expand(cfg["variables"], None, fit, forcing_bound=1e-6)
        assert not any(n in single[0]["configuration"]["bounds"] for n in forc)
        assert single[0]["configuration"]["physics"]["recipe_fit"] is True
    # a forcing whose recipe inputs are absent is no recipe: it gets the bound
    no_lon = [n for n in cfg["variables"] if n not in ("sin_longitude", "cos_longitude")]
    single = physics.expand(no_lon, None, {**cfg, "options": {"statics": "off"}}, forcing_bound=1e-6)
    assert all(single[0]["configuration"]["bounds"][n] == 1e-6 for n in forc)
    # the chain's lattice codec has no latitude / longitude (static_ref takes them): the bound, recipe_fit or not
    for fit in (cfg, {**cfg, "physics": {"recipe_fit": True}}):
        chain = _by_name(physics.expand(cfg["variables"], None, fit, forcing_bound=1e-6))
        assert {n: chain["icon_lattice_bytes"]["bounds"][n] for n in forc} == dict.fromkeys(forc, 1e-6)
    with pytest.raises(ValueError, match="forcing_bound"):
        physics.expand(cfg["variables"], None, cfg, forcing_bound=0)


def test_expand_below_ground_off_when_ps_is_a_stage_target():
    """The chain's lattice codec sees the residual of PS: its below-ground terms (mask PS < level) are turned off; the
    single codec keeps them."""
    cfg = {**_ch1(), "physics": {"below_ground": True}}
    by = _by_name(physics.expand(cfg["variables"], None, cfg))
    assert "PS" in by["field_terms"]["terms"]
    lat = by["icon_lattice_bytes"]
    assert lat["physics"]["below_ground"] is False and LatticeCodec.from_config(lat).plan.masks == {}
    single = physics.expand(cfg["variables"], None, {**cfg, "options": {"statics": "off"}})
    assert single[0]["configuration"]["physics"]["below_ground"] is True


def test_expand_synthetic_round_trip(tmp_path, lattice, chunk):
    """Synthetic names: the terms PMSL and T_2M on FIS (the other lossy fields have no orography term); CLCT keeps
    its clip in the lattice. The three codecs written through zarr and read back in a fresh process with only
    ANEMOI_DATASETS_CODECS_PATH: every value within its bound, every exact field exact."""
    layout = lattice[0]
    cfg = make_config(layout)
    entries = physics.expand(NAMES, None, cfg)
    by = _by_name(entries)
    assert list(by["field_terms"]["terms"]) == ["T_2M", "PMSL"]  # variable order
    assert by["static_ref"]["static_fields"] == list(STATIC)
    lat = by["icon_lattice_bytes"]
    assert lat["clip"] == {"CLCT": [0.0, 100.0]} and "FIS" not in lat["variables"]
    assert lat["bounds"]["T_2M"] == 0.05 * (1 - DEFAULT_MARGIN)
    path = tmp_path / "expand.zarr"
    data = np.stack([chunk, chunk])[:, :, None, :].copy()
    data[1] += np.float32(0.25) * (np.arange(len(NAMES)) % 3 == 0)[:, None, None].astype(np.float32)
    g = zarr.open_group(str(path), mode="w")
    register_store(path)
    codecs = [chain.codec_class(e["name"])(**e["configuration"]) for e in entries]
    a = g.create_array("data", shape=data.shape, chunks=(1, *data.shape[1:]), dtype="float32", compressors=codecs)
    a[:] = data
    g.attrs["variables"] = list(NAMES)
    statics.clear_caches()
    from test_static_ref import _fresh_read

    back = _fresh_read(path, tmp_path / "back.npy", path)
    for d in range(2):
        check_bounds(back[d, :, 0], data[d, :, 0])
    from anemoi_datasets_codecs.icon_lattice import store

    assert store.check(path)["problems"] == []


def test_expand_no_chain():
    """Nothing to reference (no static fields, or statics: off): the single lattice codec, as configured. FIS with a
    bound cannot be a source: no field_terms. Physics off or no FIS terms: static_ref and the lattice only. Inline
    statics: static_ref inline."""
    cfg = _ch1()
    for change in ({"static_fields": []}, {"options": {"statics": "off"}}):
        c = {**copy.deepcopy(cfg), **change}
        entries = physics.expand(None, None, c)
        assert [e["name"] for e in entries] == ["icon_lattice_bytes"]
        assert entries[0]["configuration"] == LatticeCodec(**c).get_config()
    for kw in ({"physics": "off"}, {"physics": {"surface_terms": False, "hydrostatic": False}}):
        names = [e["name"] for e in physics.expand(None, kw["physics"], cfg)]
        assert names == ["static_ref", "icon_lattice_bytes"], kw
    only_hydro = _by_name(physics.expand(None, {"surface_terms": False}, cfg))["field_terms"]
    assert list(only_hydro["terms"]) == ["FI_1000"]
    c = copy.deepcopy(cfg)
    c["static_fields"] = [n for n in c["static_fields"] if n != "FIS"]  # FIS a plain lossless field of the lattice
    assert "field_terms" not in _by_name(physics.expand(None, None, c))
    c = copy.deepcopy(cfg)
    c["options"] = {"statics": "inline"}
    assert _by_name(physics.expand(None, None, c))["static_ref"]["inline"] is True
    with pytest.raises(ValueError, match="cannot have a bound"):
        physics.expand(None, None, {**cfg, "bounds": {**cfg["bounds"], "FIS": 1.0}})


def test_expand_margin_from_ranges():
    """With data ranges: the margin is the power of two at or above 4 ulp(A) / b of every target (at least 2^-10); a
    target whose float32 resolution is too close to its bound stays in the lattice."""
    cfg = _ch1()
    ranges = {n: (-1.0, 1.0) for n in cfg["variables"]}
    ranges.update(PMSL=(9e4, 1.06e5), PS=(5e4, 1.06e5), FI_1000=(-5e3, 3e3), T_2M=(240.0, 320.0))
    ranges.update(SKT=(240.0, 340.0), TD_2M=(230.0, 310.0))
    by = _by_name(physics.expand(None, None, cfg, ranges=ranges))
    assert by["field_terms"]["margin"] == 2.0**-6 and list(by["field_terms"]["terms"]) == list(CH1_TERMS)
    ranges.update(PMSL=(9e6, 1.06e7))  # ulp 1: b = 2 needs 2 > 2^-4
    m, need = margin_for({"PMSL": 2.0}, {"PMSL": ranges["PMSL"]})
    assert need["PMSL"] > MARGIN_MAX
    by = _by_name(physics.expand(None, None, cfg, ranges=ranges))
    assert "PMSL" not in by["field_terms"]["terms"] and by["icon_lattice_bytes"]["bounds"]["PMSL"] == 2.0
    by = _by_name(physics.expand(None, None, cfg, margin=0.125))
    assert by["field_terms"]["margin"] == 0.125 and by["icon_lattice_bytes"]["bounds"]["PMSL"] == 1.75


def test_anemoi_order():
    """Recipe-style lists (names and {param, levelist}) in anemoi's order: sorted by name, so FI_1000 comes between
    FI_100 and FI_150 and lowercase names last."""
    items = ["T_2M", {"param": ["FI", "T"], "levelist": [1000, 100, 150]}, "cos_latitude", {"param": "FIS"}]
    assert physics.anemoi_order(items) == [
        "FIS",
        "FI_100",
        "FI_1000",
        "FI_150",
        "T_100",
        "T_1000",
        "T_150",
        "T_2M",
        "cos_latitude",
    ]
    with pytest.raises(ValueError, match="duplicate"):
        physics.anemoi_order(["T_2M", "T_2M"])
    with pytest.raises(ValueError, match="param"):
        physics.anemoi_order([{"levelist": [1]}])


def test_cli_expand_and_check(tmp_path):
    """``physics expand`` prints the output section (format, chunking with the whole grid, compressors) or, with
    --compressors, the list; --variables takes a recipe-style list. ``check`` on a JSON chain checks the
    configurations alone."""
    cfg = _ch1()
    cfg_file = tmp_path / "lattice.json"
    cfg_file.write_text(json.dumps(cfg))
    cli = [sys.executable, "-m", "anemoi_datasets_codecs.cli"]
    r = subprocess.run([*cli, "physics", "expand", str(cfg_file)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)["output"]
    assert out["format"] == "zarr3" and out["chunking"] == {"dates": 1, "ensembles": 1, "values": 1147980}
    assert out["compressors"] == physics.expand(None, None, cfg)
    assert "field_terms: FI_1000 <- FIS" in r.stderr
    recipe_list = tmp_path / "variables.json"
    recipe_list.write_text(json.dumps(list(reversed(cfg["variables"]))))
    r = subprocess.run(
        [*cli, "physics", "expand", str(cfg_file), "--variables", str(recipe_list), "--sort", "--compressors"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    entries = json.loads(r.stdout)
    assert entries == out["compressors"]
    chain_file = tmp_path / "chain.json"
    chain_file.write_text(json.dumps(entries))
    r = subprocess.run([*cli, "check", str(chain_file)], capture_output=True, text=True)
    assert r.returncode == 0 and json.loads(r.stdout)["problems"] == [], r.stdout + r.stderr
    entries[-1]["configuration"]["variables"] = entries[-1]["configuration"]["variables"][::-1]
    chain_file.write_text(json.dumps(entries))
    r = subprocess.run([*cli, "check", str(chain_file)], capture_output=True, text=True)
    assert r.returncode == 1 and json.loads(r.stdout)["problems"], r.stdout


def test_static_ref_outputs_match():
    """The expanded static_ref emits exactly the lattice's variables (the pairwise chain rule)."""
    cfg = _ch1()
    by = _by_name(physics.expand(None, None, cfg))
    assert list(StaticRef(**by["static_ref"]).variables_out) == by["icon_lattice_bytes"]["variables"]


def test_orography_terms_by_role():
    """The terms are roles mapped through the convention: IFS short names give the same six terms; absent targets and
    a convention without FIS give none."""
    from anemoi_datasets_codecs.physics import parse_convention, parse_physics, terms

    p = parse_physics(None)
    ifs = ["2d", "2t", "msl", "skt", "sp", "z", "z_1000", "z_925", "10u"]
    got = terms.orography_terms(ifs, parse_convention("ifs"), p)
    assert got == {t: ["z"] for t in ("msl", "z_1000", "2t", "sp", "skt", "2d")}
    assert terms.orography_terms(["msl", "2t"], parse_convention("ifs"), p) == {}  # no z
    no_fis = parse_convention({"extends": "icon-ch1", "surface": {"FIS": None}})
    assert terms.orography_terms(["PMSL", "FIS"], no_fis, p) == {}
    assert terms.orography_terms(["PMSL", "FIS"], parse_convention("icon-ch1"), {**p, "cross": "off"}) == {}


@pytest.mark.realdata
def test_realdata_sample_order_and_margin():
    """The sample store was built by anemoi-datasets: its variables are in anemoi's order (sorted by name), which is
    what ``physics expand`` assumes for a recipe-style list; its statistics give the default margin 2^-6 (PMSL and PS
    at ~1e5 Pa with b = 2 Pa need 4 ulp / b ~ 2^-6)."""
    from conftest import SAMPLE_STORE

    from anemoi_datasets_codecs.cli import read_variables

    if not SAMPLE_STORE.exists():
        pytest.skip(f"sample store {SAMPLE_STORE} missing")
    names, ranges = read_variables(SAMPLE_STORE)
    assert physics.anemoi_order(names) == names and ranges is not None and set(ranges) == set(names)
    cfg = _ch1()
    assert names == cfg["variables"]
    by = _by_name(physics.expand(names, None, cfg, ranges=ranges))
    assert by["field_terms"]["margin"] == DEFAULT_MARGIN and list(by["field_terms"]["terms"]) == list(CH1_TERMS)


def test_expand_dict_variables_physics_absent_statics():
    """Variables given as dicts {name, max_error, clip} (alone, or the configuration's settings of a variable the
    dataset names) expand like bounds / clip mappings; the single codec takes the physics argument like the chain;
    configured static fields absent from the dataset are dropped from static_ref."""
    from anemoi_datasets_codecs.config import resolve_variables

    cfg = _ch1()
    want = physics.expand(None, None, cfg)
    res = resolve_variables(cfg["variables"], cfg["bounds"], cfg["clip"])
    as_dicts = {k: v for k, v in cfg.items() if k not in ("bounds", "clip")}
    as_dicts["variables"] = [{"name": v.name, "max_error": v.max_error, "clip": v.clip} for v in res]
    assert physics.expand(None, None, as_dicts) == want
    assert physics.expand(list(cfg["variables"]), None, as_dicts) == want  # names from the dataset, settings here
    single = {**as_dicts, "static_fields": []}
    (e,) = physics.expand(None, None, single)
    assert e["configuration"] == LatticeCodec(**{**cfg, "static_fields": []}).get_config()
    for off in (False, "off"):
        (e,) = physics.expand(None, off, single)
        assert e["configuration"]["physics"]["enabled"] is False
    (e,) = physics.expand(None, {"cross": "on"}, single)
    assert e["configuration"]["physics"]["cross"] == "on"
    absent = cfg["static_fields"][-1]
    names = [n for n in cfg["variables"] if n != absent]
    by = _by_name(physics.expand(names, None, {**cfg, "bounds": {}, "clip": {}}))
    assert by["static_ref"]["static_fields"] == cfg["static_fields"][:-1]
    assert absent not in by["icon_lattice_bytes"]["variables"]


def _cli(*args, env=None, pre=""):
    code = f"import sys{pre}; from anemoi_datasets_codecs.cli import main; sys.exit(main(sys.argv[1:]))"
    return subprocess.run([sys.executable, "-c", code, *map(str, args)], capture_output=True, text=True, env=env)


def test_cli_expand_output_is_input(tmp_path):
    """check, plan and info --config read what physics expand prints: the output section, the compressors list, a
    bare {compressors}, an anemoi recipe with the output section, and its YAML; expand reads its single-codec output."""
    cfg = _ch1()
    cfg_file = tmp_path / "lattice.json"
    cfg_file.write_text(json.dumps(cfg))
    r = _cli("physics", "expand", cfg_file)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    forms = {
        "output.json": out,
        "compressors.json": {"compressors": out["output"]["compressors"]},
        "list.json": out["output"]["compressors"],
        "recipe.json": {"dates": {"start": "2020-01-01", "end": "2020-01-01"}, "input": {}, **out},
    }
    for name, form in forms.items():
        (tmp_path / name).write_text(json.dumps(form))
    r = _cli("physics", "expand", cfg_file, "--yaml")
    assert r.returncode == 0, r.stderr
    (tmp_path / "output.yaml").write_text(r.stdout)
    for name in [*forms, "output.yaml"]:
        r = _cli("check", tmp_path / name)
        assert r.returncode == 0 and json.loads(r.stdout)["problems"] == [], (name, r.stdout, r.stderr)
        assert json.loads(r.stdout)["chain"] == ["field_terms", "static_ref", "icon_lattice_bytes"], name
    r = _cli("plan", tmp_path / "output.json", "--json")
    assert r.returncode == 0 and json.loads(r.stdout)["terms"], r.stderr
    # plan on a configuration without variables (the README's lattice.json): --variables, else a clear message
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({k: v for k, v in cfg.items() if k != "variables"}))
    (tmp_path / "names.json").write_text(json.dumps(cfg["variables"]))
    r = _cli("plan", bare)
    assert r.returncode == 1 and r.stderr.startswith("anemoi-datasets-codecs plan: ") and "--variables" in r.stderr
    r = _cli("plan", bare, "--variables", tmp_path / "names.json", "--json")
    assert r.returncode == 0 and json.loads(r.stdout)["terms"], r.stderr
    r = _cli("physics", "expand", cfg_file, "--forcing-bound", "-1")
    assert r.returncode == 1 and r.stderr.startswith("anemoi-datasets-codecs physics expand: forcing_bound"), r.stderr
    (tmp_path / "no-statics.json").write_text(json.dumps({**cfg, "static_fields": []}))
    r = _cli("physics", "expand", tmp_path / "no-statics.json", "--compressors")
    assert r.returncode == 0, r.stderr
    (tmp_path / "single.json").write_text(r.stdout)
    r = _cli("physics", "expand", tmp_path / "single.json", "--compressors")
    assert r.returncode == 0 and json.loads(r.stdout) == json.loads((tmp_path / "single.json").read_text()), r.stderr
    (tmp_path / "bad.json").write_text(json.dumps({"output": {"format": "zarr3"}}))
    r = _cli("check", tmp_path / "bad.json")
    assert r.returncode == 1 and "no compressors" in r.stderr and "Traceback" not in r.stderr


def test_cli_expand_arguments(tmp_path):
    """--physics on / off / true / false / JSON / a file, --margin auto or a power of two in (0, 2^-4], --yaml
    without PyYAML: invalid values exit with a message, no traceback."""
    cfg = {**_ch1(), "static_fields": []}  # the single codec: its physics group shows what --physics gave
    cfg_file = tmp_path / "lattice.json"
    cfg_file.write_text(json.dumps(cfg))
    phys_file = tmp_path / "physics.json"
    phys_file.write_text(json.dumps({"cross": "on"}))
    for value, key, want in (
        ("off", "enabled", False),
        ("false", "enabled", False),
        ("on", "enabled", True),
        ("true", "enabled", True),
        ('{"cross": "on"}', "cross", "on"),
        (phys_file, "cross", "on"),
    ):
        r = _cli("physics", "expand", cfg_file, "--physics", value, "--compressors")
        assert r.returncode == 0, (value, r.stderr)
        assert json.loads(r.stdout)[0]["configuration"]["physics"][key] == want, value
    for args in (("--physics", "bogus"), ("--physics", '{"nope": 1}'), ("--physics", "[1]")):
        r = _cli("physics", "expand", cfg_file, *args)
        assert r.returncode == 1 and "Traceback" not in r.stderr and r.stderr.strip(), (args, r.stderr)
    full = tmp_path / "full.json"
    full.write_text(json.dumps(_ch1()))
    for m, ok in (("auto", True), ("0.03125", True), ("0.0625", True), ("abc", False), ("1.5", False)):
        r = _cli("physics", "expand", full, "--margin", m, "--compressors")
        assert (r.returncode == 0) is ok and "Traceback" not in r.stderr, (m, r.stderr)
        if ok and m != "auto":
            assert _by_name(json.loads(r.stdout))["field_terms"]["margin"] == float(m)
    for m in ("0.1", "0.125", "0", "-0.015625"):
        r = _cli("physics", "expand", full, "--margin", m)
        assert r.returncode == 2 and "power of two" in r.stderr and "Traceback" not in r.stderr, (m, r.stderr)
    r = _cli("physics", "expand", full, "--yaml", pre="; sys.modules['yaml'] = None")
    assert r.returncode == 1 and r.stderr.strip() == "--yaml needs PyYAML (pip install pyyaml)", r.stderr


def test_expand_refuses_inconsistent(monkeypatch):
    """expand verifies its result with chain_problems and raises on any problem (check=False skips it)."""
    cfg = _ch1()
    monkeypatch.setattr(chain, "chain_problems", lambda entries: ["a planted problem"])
    with pytest.raises(ValueError, match="inconsistent: a planted problem"):
        physics.expand(None, None, cfg)
    assert [e["name"] for e in physics.expand(None, None, cfg, check=False)][-1] == "icon_lattice_bytes"
