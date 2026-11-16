"""The physics layer: plans and chunk bytes against the frozen baseline, switches, name conventions (model levels,
IFS names), the plan report, and configuration validation."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import DATE, NAMES, check_bounds, make_codec, make_config, smooth

from anemoi_datasets_codecs import physics as phys
from anemoi_datasets_codecs.icon_lattice import LatticeCodec, container
from anemoi_datasets_codecs.plan import Plan

DATA = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import plan_baseline as pb  # noqa: E402


def _plan_json(codec: LatticeCodec) -> dict:
    masks = {k: [m, t, [list(f) for f in fs]] for k, (m, t, fs) in codec.plan.masks.items()}
    return json.loads(json.dumps({"cross": codec.cross_plan, "masks": masks, "deps": codec._deps}, sort_keys=True))


# -------------------------------------------------------------------------------------------- frozen baseline
@pytest.mark.parametrize("setting", list(pb.SETTINGS))
def test_plan_identical_to_baseline(lattice, setting):
    """On the 115 ICON-CH1 names, every setting resolves to exactly the baseline plan (tests/data). Same plan and the
    same encoder -> the same chunk bytes."""
    baseline = json.loads((DATA / "plans_ch1_baseline.json").read_text())[setting]
    assert _plan_json(pb.codec(setting, lattice)) == baseline


def test_chunk_bytes_identical_to_baseline(lattice):
    """The chunk bytes of the settings above against the baseline (only meaningful with the numpy the baseline was
    written with: the mask-term fit is float64 least squares; the below-ground setting also needs the linear algebra
    kernels of the baseline, ``plan_baseline.BASELINE_BLAS``, and is left out with others). The first settings pin
    the encoder settings the baseline was first written with (``plan_baseline.LEGACY``), the ``direct-*`` ones the
    defaults of release 0.1.0; ``scripts/plan_baseline.py check`` runs all of them."""
    ref = json.loads((DATA / "chunk_sha256_baseline_numpy-2.5.4.json").read_text())
    if np.__version__ != "2.5.4":
        pytest.skip(f"baseline written with numpy 2.5.4, this is {np.__version__}")
    for setting in (
        "default",
        "cross-off",
        "below-ground",
        "no-recipes",
        "direct-recipe-fit",
        "direct-recipe-fit-nodate",
    ):
        if pb.blas_dependent(setting) and pb.blas_architecture() not in pb.BASELINE_BLAS:
            continue  # other OpenBLAS kernels round the least-squares fit differently (OPENBLAS_CORETYPE=Haswell)
        assert pb.encode(setting) == ref[setting], setting


def test_baseline_settings_pin_the_changed_defaults():
    """Every baseline setting spells out the options whose default has changed, so that the baseline does not move
    with a default."""
    for name, (options, physics, _) in pb.SETTINGS.items():
        assert "rounding" in options and "recipe_fit" in physics, name


def test_defaults_spelled_out_are_the_defaults(lattice, chunk):
    a = make_codec(lattice[0])
    b = LatticeCodec(**make_config(lattice[0]), physics=dict(phys.PHYSICS_DEFAULTS))
    c = LatticeCodec(**make_config(lattice[0]), physics="on")
    assert a.get_config() == b.get_config() == c.get_config()
    assert a.encode(chunk, dates=[DATE]) == b.encode(chunk, dates=[DATE])


# -------------------------------------------------------------------------------------------- a given plan
def test_given_plan_is_the_physics_plan(lattice, chunk):
    """The physics layer's candidate plan, given as ``plan=``, writes the same bytes; the configuration is unchanged
    (a plan is not part of it)."""
    a = make_codec(lattice[0])
    b = LatticeCodec(**make_config(lattice[0]), plan=phys.resolve(NAMES))
    assert a.get_config() == b.get_config() and a.cross_plan == b.cross_plan
    assert a.encode(chunk, dates=[DATE]) == b.encode(chunk, dates=[DATE])
    assert a != b  # same configuration, but b's plan does not come from it


def test_given_plan_without_physics(lattice, chunk):
    """A hand-built plan with the physics layer off: its cross term and recipe are used, the codec's limits apply,
    pickling keeps the plan."""
    inputs = ("sin_latitude", "cos_latitude", "sin_longitude", "cos_longitude")
    plan = Plan(cross={"T_2M": ["PMSL", "FIS"], "WIDE": ["T_2M"]}, recipes={"cos_local_time": (1, inputs)})
    codec = LatticeCodec(**make_config(lattice[0]), physics="off", plan=plan)
    assert codec.cross_plan == {"T_2M": ["PMSL", "FIS"]}  # max_depth 1: WIDE <- T_2M is too deep
    assert plan.cross["WIDE"] == ["T_2M"]  # the given plan is not modified
    enc = codec.encode(chunk, dates=[DATE])
    check_bounds(codec.decode(enc).reshape(chunk.shape), chunk)
    refs = container.field_references(enc)
    assert refs[NAMES.index("cos_local_time")]["recipe"]["id"] == 1
    assert "recipe" not in refs[NAMES.index("sin_local_time")]
    assert set(refs[NAMES.index("T_2M")].get("cross", [])) <= {NAMES.index("PMSL"), NAMES.index("FIS")}
    copy = pickle.loads(pickle.dumps(codec))
    assert copy == codec and copy.cross_plan == codec.cross_plan and "plan" not in codec.get_config()
    assert copy.encode(chunk, dates=[DATE]) == enc


def test_given_plan_is_validated(lattice):
    cfg = make_config(lattice[0])
    with pytest.raises(ValueError, match="not variables.*NOPE"):
        LatticeCodec(**cfg, plan=Plan(cross={"T_2M": ["NOPE"]}))
    with pytest.raises(ValueError, match="recipe"):
        LatticeCodec(**cfg, plan=Plan(recipes={"cos_local_time": (9, ("FIS",) * 4)}))
    with pytest.raises(TypeError, match="plan.Plan"):
        LatticeCodec(**cfg, plan={"cross": {}})


# -------------------------------------------------------------------------------------------- switches
@pytest.mark.parametrize(
    "physics",
    [
        "off",
        {"cross": "off"},
        {"surface_terms": False},
        {"level_terms": False, "hydrostatic": False},
        {"forcings": False},
        {"enabled": False, "below_ground": True},
    ],
)
def test_switches_round_trip(lattice, chunk, physics):
    codec = LatticeCodec(**make_config(lattice[0]), physics=physics)
    enc = codec.encode(chunk, dates=[DATE])
    check_bounds(codec.decode(enc).reshape(chunk.shape), chunk)
    refs = container.field_references(enc)
    if physics == "off" or physics.get("enabled") is False:
        assert codec.cross_plan == {} and codec.plan.recipes == {} and codec.plan.masks == {}
        assert all(not r.get("cross") and "recipe" not in r for r in refs)
    if physics == {"surface_terms": False}:  # T_2M <- FIS, PMSL is a surface term
        assert not refs[NAMES.index("T_2M")].get("cross")
    if physics == {"forcings": False}:
        assert all("recipe" not in r for r in refs)


def test_everything_off_on_ch1_names(lattice):
    """physics off on the full ICON-CH1 variable set: no term at all, every field within its bound."""
    layout = lattice[0]
    x = pb.chunk(lattice)
    codec = LatticeCodec(variables=pb.names(), grid=layout, bounds=pb.bounds(), static_fields=pb.STATIC, physics=False)
    enc = codec.encode(x, dates=[DATE])
    assert all(not r.get("cross") and "recipe" not in r and "mask" not in r for r in container.field_references(enc))
    dec = codec.decode(enc).reshape(x.shape)
    for i, v in enumerate(codec.variables):
        err = float(np.abs(dec[i].astype(np.float64) - x[i]).max())
        assert err <= (v.max_error if v.lossy else 0.0), v.name


# -------------------------------------------------------------------------------------------- conventions
def test_model_levels_get_inter_level_terms(lattice):
    """A model-level naming (t_ml_1..t_ml_12, top to surface) with a custom level family: the inter-level terms
    apply to it, and the chunk uses them."""
    layout, rr, cc, *_ = lattice
    names = [f"t_ml_{k}" for k in range(1, 13)]
    rng = np.random.default_rng(5)
    shared = rng.standard_normal(len(rr))
    x = np.stack([250 + 2 * k + 5 * smooth(rr, cc, 11) + 0.3 * shared for k in range(12)]).astype(np.float32)
    convention = {"families": {"T": "t_ml_{level}"}, "levels": list(range(1, 13)), "vertical": "model"}
    codec = LatticeCodec(variables=names, grid=layout, bounds={"t_ml_.*": 0.05}, physics={"convention": convention})
    assert codec.cross_plan["t_ml_2"] == ["t_ml_1", "t_ml_3"]
    assert codec.cross_plan["t_ml_12"] == ["t_ml_11"]  # the lowest is odd-indexed: its upper neighbour only
    enc = codec.encode(x)
    refs = container.field_references(enc)
    assert refs[names.index("t_ml_2")]["cross"] == [0, 2]
    dec = codec.decode(enc).reshape(x.shape)
    assert float(np.abs(dec.astype(np.float64) - x).max()) <= 0.05
    # the convention travels with the configuration
    again = LatticeCodec.from_config(json.loads(json.dumps(codec.get_config())))
    assert again.cross_plan == codec.cross_plan
    # below-ground terms need pressure levels: silently nothing on model levels
    bg = LatticeCodec(
        variables=names + ["sp"],
        grid=layout,
        bounds={"t_ml_.*": 0.05},
        physics={"convention": {**convention, "surface": {"PS": "sp"}}, "below_ground": True},
    )
    assert bg.plan.masks == {}


def test_ifs_names_get_surface_terms(lattice):
    layout, rr, cc, *_ = lattice
    names = ["2t", "msl", "sp", "z", "10u", "10v", "u_1000", "u_925", "t_850", "t_925", "t_1000"]
    bounds = {n: 0.1 for n in names if n != "z"}  # z (surface geopotential) is static: exact, a free cross source
    codec = LatticeCodec(
        variables=names, grid=layout, bounds=bounds, static_fields=["z"], physics={"convention": "ifs"}
    )
    plan = codec.cross_plan
    assert plan["msl"] == ["z"] and plan["2t"] == ["z", "msl"] and plan["sp"] == ["msl", "z"]
    assert plan["10u"] == ["u_1000", "z"] and plan["t_925"] == ["t_850", "t_1000"]
    rng = np.random.default_rng(9)
    shared = rng.standard_normal(len(rr))
    x = np.stack([10 * smooth(rr, cc, 30 + k) + 0.5 * shared for k in range(len(names))]).astype(np.float32)
    enc = codec.encode(x)
    assert container.inspect_chunk(enc)[names.index("z")].mode == container.MODE_STATIC_REF
    refs = container.field_references(enc)
    assert (
        names.index("z") in refs[names.index("2t")]["cross"] or names.index("msl") in refs[names.index("2t")]["cross"]
    )
    dec = codec.decode(enc).reshape(x.shape)
    assert float(np.abs(dec.astype(np.float64) - x).max()) <= 0.1
    # the same data under the ICON names convention: no term (no variable plays a role)
    assert LatticeCodec(variables=names, grid=layout, bounds=bounds, static_fields=["z"]).cross_plan == {}


def test_custom_surface_mapping_extends_a_profile(lattice):
    names = ["t2m", "pmsl", "hsurf_geop"]
    conv = {"extends": "icon-ch1", "surface": {"T_2M": "t2m", "PMSL": "pmsl", "FIS": "hsurf_geop"}}
    codec = LatticeCodec(
        variables=names,
        grid=lattice[0],
        bounds={"t2m": 0.1, "pmsl": 1.0},
        static_fields=["hsurf_geop"],
        physics={"convention": conv},
    )
    assert codec.cross_plan == {"pmsl": ["hsurf_geop"], "t2m": ["hsurf_geop", "pmsl"]}


# -------------------------------------------------------------------------------------------- report
def test_plan_report_says_what_was_dropped_and_why(lattice):
    names = ["T_2M", "PMSL", "PS", "TD_2M", "QV_1000", "T_850", "T_925", "cos_local_time", "sin_latitude"]
    codec = LatticeCodec(variables=names, grid=lattice[0], bounds={"T_.*|PMSL|PS|TD_2M|QV_.*": 0.1})
    rep = {(t["target"], t["kind"]): t for t in codec.plan_report()}
    assert rep[("T_2M", "surface")]["applied"] == ["PMSL"]
    assert ["FIS", "absent"] in rep[("T_2M", "surface")]["dropped"]
    td = rep[("TD_2M", "surface")]
    assert ["T_2M", "decode depth limit 1"] in td["dropped"] and ["QV_925", "absent"] in td["dropped"]
    assert rep[("T_925", "level")]["applied"] == ["T_850"]
    forcing = rep[("cos_local_time", "forcing")]
    assert forcing["note"].startswith("inputs absent") and ["cos_latitude", "absent"] in forcing["dropped"]
    off = LatticeCodec(variables=names, grid=lattice[0], bounds={"T_.*": 0.1}, physics={"surface_terms": False})
    assert {t["target"]: t.get("note") for t in off.plan_report()}["T_2M"] == "surface_terms off"


def test_cli_plan_and_info(tmp_path, lattice, chunk):
    from anemoi_datasets_codecs.icon_lattice import grids

    grids.save_lattice(lattice[0])
    cfg = make_codec(lattice[0]).get_config()
    path = tmp_path / "codec.json"
    path.write_text(json.dumps({"name": "icon_lattice", "configuration": cfg}))
    r = subprocess.run(
        [sys.executable, "-m", "anemoi_datasets_codecs.cli", "plan", str(path)], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    assert "convention icon-ch1" in r.stdout and "T_2M             surface      FIS, PMSL" in r.stdout
    enc = make_codec(lattice[0]).encode(chunk, dates=[DATE])
    (tmp_path / "chunk").write_bytes(enc)
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "anemoi_datasets_codecs.cli",
            "info",
            str(tmp_path / "chunk"),
            "--config",
            str(path),
            "-v",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    line = next(x for x in r.stdout.splitlines() if " T_2M " in x)
    assert "cross: FIS, PMSL" in line
    assert "recipe 3 of sin_latitude, cos_latitude, sin_longitude, cos_longitude" in r.stdout


# -------------------------------------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "physics, match",
    [
        ({"crosss": "on"}, "unknown physics keys"),
        ({"cross": "deep"}, "physics.cross"),
        ({"below_ground": "yes"}, "true or false"),
        ("sometimes", "on / off"),
        ({"convention": "nwp"}, "unknown name convention"),
        ({"convention": {"surfaces": {}}}, "unknown convention keys"),
        ({"convention": {"surface": {"T2M": "2t"}}}, "unknown role"),
        ({"convention": {"surface": {"T_2M": "x", "SKT": "x"}}}, "mapped to both"),
        ({"convention": {"families": {"T": "t_ml"}, "levels": [1, 2]}}, "one '{level}'"),
        ({"convention": {"families": {"TT": "t_{level}"}}}, "unknown family"),
        ({"convention": {"levels": [1, 1]}}, "not unique"),
        ({"convention": {"levels": [1, 2], "star_level": 3}}, "star_level"),
        ({"convention": {"vertical": "sigma"}}, "vertical"),
        ({"convention": {"extends": "icon-ch1", "families": {"T": "FI_{level}"}}}, "same variables"),
        ({"recipe_bound": 0}, "recipe_bound"),
    ],
)
def test_validation_at_parse_time(lattice, physics, match):
    with pytest.raises(ValueError, match=match):
        LatticeCodec(**make_config(lattice[0]), physics=physics)


@pytest.mark.parametrize("option", ["cross", "below_ground", "recipes", "recipe_bound"])
def test_physics_settings_are_not_options(lattice, option):
    """The physics settings live in the ``physics`` group; as encoder options they are unknown (with a hint)."""
    with pytest.raises(ValueError, match=rf"unknown lattice codec options.*{option}.*physics"):
        LatticeCodec(**make_config(lattice[0], **{option: "off"}))


# -------------------------------------------------------------------------------------------- package layout
def test_package_level_layout():
    """The physics layer, the plan and the shared configuration helpers are package level: importing them (and the
    grid-agnostic stages) does not import the lattice codec; the convention's recipe roles are the lattice codec's
    recipe table, and the lattice codec checks a plan's recipe ids against it."""
    code = (
        "import sys\n"
        "import anemoi_datasets_codecs.physics, anemoi_datasets_codecs.plan, anemoi_datasets_codecs.config\n"
        "import anemoi_datasets_codecs.field_terms, anemoi_datasets_codecs.static_ref\n"
        "print(sorted(m for m in sys.modules if m.startswith('anemoi_datasets_codecs.icon_lattice')))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"
    import anemoi_datasets_codecs.plan as plan
    from anemoi_datasets_codecs import config
    from anemoi_datasets_codecs.icon_lattice import config as lconfig
    from anemoi_datasets_codecs.icon_lattice import recipes
    from anemoi_datasets_codecs.physics import conventions

    bad = plan.Plan(recipes={"A": (99, ("B", "B", "B", "B"))})
    plan.check_names(bad, ["A", "B"])  # the package-level function knows no format
    with pytest.raises(ValueError, match="unknown id 99"):
        plan.check_names(bad, ["A", "B"], recipes.RECIPE_IDS.values())
    plan.check_names(plan.Plan(recipes={"A": (1, ("B", "B", "B", "B"))}), ["A", "B"], recipes.RECIPE_IDS.values())
    assert lconfig.resolve_variables is config.resolve_variables and lconfig.Variable is config.Variable
    assert tuple(recipes.RECIPE_IDS) == conventions.RECIPE_ROLES
    assert recipes.RECIPE_INPUTS == conventions.RECIPE_INPUT_ROLES
