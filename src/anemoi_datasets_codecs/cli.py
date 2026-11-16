"""anemoi-datasets-codecs: tools for stores, chunks and configurations of the codecs (lattice codec and chains).

anemoi-datasets-codecs isa                    kernel variants (avx2 / scalar) and the one in use
anemoi-datasets-codecs grids                  grids bundled with the package
anemoi-datasets-codecs grid build GRID.nc [--name N] [--out DIR] [--no-check]
                                              lattice file of an ICON grid (encoder side; extra "grid")
anemoi-datasets-codecs attach STORE [--array data] [--sample N]
                                              copy the lattice and static fields into STORE/.codecs
anemoi-datasets-codecs check STORE [--array data] [--sample N]
                                              variables vs the store's attribute, chain consistent, sidecar
                                              complete, decodes; the codecs' writer stamps (written_by)
anemoi-datasets-codecs info CHUNK_FILE [--config STORE_OR_JSON] [-v]
                                              field table of one chunk (-v: per field, the terms it uses); for
                                              a chained chunk, its envelope and stage headers first
anemoi-datasets-codecs plan STORE_OR_JSON [--variables SRC] [--sort] [--json]
                                              the physics plan of a configuration: terms applied / dropped, why
anemoi-datasets-codecs physics expand STORE_OR_JSON [--variables SRC] [--sort] [--physics P] [--margin M]
                                              [--compressors] [--forcing-bound B] [--yaml]
                                              the codec chain [field_terms, static_ref, icon_lattice_bytes] of a
                                              lattice configuration for a dataset's variables, as the output
                                              section (or the output.compressors list) of an anemoi recipe
anemoi-datasets-codecs lab TOOL ...          the research toolkit (docs/lab.md); imported only when used

STORE is a local Zarr 3 store; STORE_OR_JSON a store, a JSON file with the codec configuration, or a JSON file with
a codec entry ``{name, configuration}`` (as in ``zarr.json`` or an anemoi recipe), a list of codec entries (a chain,
e.g. ``[static_ref, icon_lattice_bytes]``), ``{compressors: [...]}``, or an anemoi recipe / ``physics expand``'s
output ``{output: {compressors: [...]}}``; JSON files may be YAML (``.yaml``, ``.yml``). ``check`` also takes such a
file: then only the configurations are checked (a chain's consistency). SRC (the dataset's variables, for a
configuration without them): a store, or a JSON / YAML list of names.

An error is reported as ``anemoi-datasets-codecs COMMAND: message`` with exit status 1; ``--debug`` (before the
command) shows the traceback instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import STAMP_KEY


def _isa(_: argparse.Namespace) -> int:
    from .icon_lattice import native

    print(
        json.dumps(
            {"in_use": native.isa(), "available": native.available_isas(), "library": str(native.library_path())}
        )
    )
    return 0


def _grids(_: argparse.Namespace) -> int:
    from .icon_lattice import grids

    print(json.dumps(grids.known_grids(), indent=2))
    return 0


def _grid_build(a: argparse.Namespace) -> int:
    from .icon_lattice import gridfile, grids

    lay = gridfile.lattice_from_grid_file(a.grid_file, a.name, check=not a.no_check)
    path = grids.save_lattice(lay, a.out)
    print(json.dumps({"lattice": str(path), "grid": lay.ref(), "shape": list(lay.shape)}, indent=2))
    return 0


def _attach(a: argparse.Namespace) -> int:
    from .icon_lattice import store

    print(json.dumps(store.attach(a.store, a.array, a.sample), indent=2))
    return 0


def _check(a: argparse.Namespace) -> int:
    from .icon_lattice import store

    if a.store.is_file():  # configurations only
        from . import chain

        cfg = load_config(a.store)
        entries = cfg if isinstance(cfg, list) else [{"name": "icon_lattice_bytes", "configuration": cfg}]
        r = {"config": str(a.store), "chain": [e["name"] for e in entries], "problems": chain.chain_problems(entries)}
        print(json.dumps(r, indent=2))
        return 1 if r["problems"] else 0
    r = store.check(a.store, a.array, a.sample)
    print(json.dumps(r, indent=2))
    return 1 if r["problems"] else 0


def load_config(path: Path) -> dict | list[dict]:
    """Codec configuration from a store (its data array), a JSON file with the configuration, or a JSON file with
    a codec entry ``{name, configuration}`` (as in ``zarr.json`` or an anemoi recipe). For a chained store, a JSON
    list of codec entries, ``{compressors: [...]}`` or a recipe ``{output: {compressors: [...]}}`` (what ``physics
    expand`` prints), the list of entries ``{name, configuration}`` of the chain (encode order)."""
    from .icon_lattice import store

    if path.is_dir():
        m = store.array_meta(path)
        if m["chain"] is not None:
            return [dict(e) for e in m["chain"]]
        cfg = dict(m["codec"])
    else:
        cfg = read_json(path)
        if isinstance(cfg, dict) and "output" in cfg:  # an anemoi recipe, or physics expand's output section
            cfg = cfg["output"]
            if not isinstance(cfg, dict) or "compressors" not in cfg:
                raise SystemExit(f"{path}: the output section has no compressors")
        if isinstance(cfg, dict) and "compressors" in cfg:
            cfg = cfg["compressors"]
            cfg = [cfg] if isinstance(cfg, dict) else cfg
        if isinstance(cfg, list):
            if not all(isinstance(e, dict) and isinstance(e.get("name"), str) for e in cfg):
                raise SystemExit(f"{path}: a codec list must hold entries {{name, configuration}}")
            return [_entry(e) for e in cfg]
        if not isinstance(cfg, dict):
            raise SystemExit(f"{path}: expected a codec configuration, a codec entry or a list of codec entries")
        if "configuration" in cfg and "name" in cfg:
            cfg = dict(cfg["configuration"])
    cfg.pop("id", None)
    return cfg


def read_json(path: Path):
    """A JSON file, or YAML for ``.yaml`` / ``.yml``."""
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        return _yaml(f"{path}: reading YAML").safe_load(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"{path}: not valid JSON ({e}); YAML files need the suffix .yaml or .yml") from None


def _yaml(what: str):
    """PyYAML (installed with zarr, through donfig), or a one-line exit naming ``what`` needs it."""
    try:
        import yaml
    except ImportError:
        raise SystemExit(f"{what} needs PyYAML (pip install pyyaml)") from None
    return yaml


def physics_arg(value: str):
    """The ``--physics`` value: ``on`` / ``true``, ``off`` / ``false``, a JSON object or a JSON / YAML file."""
    v = value.strip()
    if v.lower() in ("on", "true", "off", "false"):
        return "on" if v.lower() in ("on", "true") else "off"
    p = Path(v)
    if p.is_file():
        return read_json(p)
    try:
        out = json.loads(v)
    except json.JSONDecodeError:
        raise SystemExit(
            f"--physics: on, off, true, false, a JSON object or a JSON / YAML file, got {value!r}"
        ) from None
    if not isinstance(out, dict):
        raise SystemExit(f"--physics: on, off, true, false, a JSON object or a JSON / YAML file, got {value!r}")
    return out


def margin_arg(value: str) -> float | None:
    """The ``--margin`` value: ``auto`` (None) or a power of two in (0, 2^-4]."""
    import math

    from .physics.expansion import MARGIN_MAX

    if value == "auto":
        return None
    try:
        m = float(value)
    except ValueError:
        m = math.nan
    if not (0 < m <= MARGIN_MAX and math.frexp(m)[0] == 0.5):
        raise argparse.ArgumentTypeError(f"auto or a power of two in (0, 2^-4], e.g. 0.015625, got {value!r}")
    return m


def read_variables(path: Path, sort: bool = False) -> tuple[list[str], dict[str, tuple[float, float]] | None]:
    """A dataset's variables and, for an anemoi store with statistics, their ranges ``{name: (min, max)}``: from a
    store (its ``variables`` attribute, in the store's order), or a JSON / YAML file with a list of names (the
    dataset's order; ``sort``: anemoi's order), a mapping with ``variables``, or a recipe-style list (names and
    ``{param, levelist}`` mappings, put in anemoi's order)."""
    from .physics import anemoi_order

    if not path.exists():
        raise SystemExit(f"{path}: no such store or file")
    if path.is_dir():
        meta = path / "zarr.json"
        attrs = (json.loads(meta.read_text()).get("attributes") or {}) if meta.is_file() else {}
        if "variables" not in attrs:
            raise SystemExit(f"{path}: no variables attribute (not an anemoi dataset)")
        names = list(attrs["variables"])
        ranges = None
        if (path / "minimum" / "zarr.json").is_file() and (path / "maximum" / "zarr.json").is_file():
            import zarr

            g = zarr.open_group(str(path), mode="r")
            lo, hi = g["minimum"][...], g["maximum"][...]
            ranges = {n: (float(lo[i]), float(hi[i])) for i, n in enumerate(names)}
        return (anemoi_order(names) if sort else names), ranges
    data = read_json(path)
    if isinstance(data, dict):
        data = data.get("variables")
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a list of variables (or a mapping with variables)")
    if sort or any(not isinstance(v, str) for v in data):
        return anemoi_order(data), None
    return list(data), None


def _expand(a: argparse.Namespace) -> int:
    from . import physics

    cfg = load_config(a.config)
    if isinstance(cfg, list) and len(cfg) == 1 and cfg[0]["name"] in ("icon_lattice", "icon_lattice_bytes"):
        cfg = dict(cfg[0]["configuration"])  # a single lattice codec (physics expand's output without statics)
    if isinstance(cfg, list):
        raise SystemExit(f"{a.config} is a chain already; give the lattice configuration (with its static_fields)")
    yaml = _yaml("--yaml") if a.yaml else None
    names, ranges = (None, None) if a.variables is None else read_variables(a.variables, a.sort)
    if names is None and a.sort:
        names = physics.anemoi_order([v if isinstance(v, str) else v["name"] for v in cfg.get("variables") or ()])
    phys = None if a.physics is None else physics_arg(a.physics)
    entries = physics.expand(names, phys, cfg, margin=a.margin, ranges=ranges, forcing_bound=a.forcing_bound)
    out = entries if a.compressors else {"output": physics.output_block(entries)}
    if yaml is not None:
        print(yaml.safe_dump(out, sort_keys=False, default_flow_style=None, width=120), end="")
    else:
        print(json.dumps(out, indent=1))
    by = {e["name"]: e["configuration"] for e in entries}
    summary = []
    if "field_terms" in by:
        ft = by["field_terms"]
        summary.append(
            "field_terms: "
            + ", ".join(f"{t} <- {'+'.join(v['sources'])}" for t, v in ft["terms"].items())
            + f" (margin {ft['margin']:g})"
        )
    if "static_ref" in by:
        summary.append(f"static_ref: {len(by['static_ref']['static_fields'])} static fields")
    lat = by["icon_lattice_bytes"]
    summary.append(f"icon_lattice_bytes: {len(lat['variables'])} variables, chain {lat.get('chain', False)}")
    print("; ".join(summary), file=sys.stderr)
    return 0


def _entry(e: dict) -> dict:
    cfg = dict(e.get("configuration") or {})
    cfg.pop("id", None)
    return {"name": e["name"], "configuration": cfg}


def lattice_config(cfg: dict | list[dict]) -> dict:
    """The lattice codec's configuration of ``load_config``'s result (the chain's ``icon_lattice*`` entry)."""
    if not isinstance(cfg, list):
        return cfg
    for e in cfg:
        if e["name"] in ("icon_lattice", "icon_lattice_bytes"):
            return dict(e["configuration"])
    raise ValueError(f"the chain {[e['name'] for e in cfg]} has no icon_lattice codec")


def _plan(a: argparse.Namespace) -> int:
    from .icon_lattice.codec import LatticeCodec

    cfg = lattice_config(load_config(a.config))
    if a.variables is not None:
        cfg["variables"] = read_variables(a.variables, a.sort)[0]
    elif a.sort and cfg.get("variables"):
        from .physics import anemoi_order

        cfg["variables"] = anemoi_order([v if isinstance(v, str) else v["name"] for v in cfg["variables"]])
    if not cfg.get("variables"):
        raise ValueError(f"{a.config} names no variables: give the dataset's with --variables (a store or a list)")
    codec = LatticeCodec.from_config(cfg)
    report = codec.plan_report()
    if a.json:
        print(json.dumps({"physics": codec.physics, "terms": report}, indent=1))
        return 0
    p = codec.physics
    conv = p["convention"] if isinstance(p["convention"], str) else "custom"
    flags = ", ".join(
        f"{k} {'on' if p[k] else 'off'}"
        for k in ("surface_terms", "level_terms", "hydrostatic", "below_ground", "forcings")
    )
    print(f"physics {'on' if p['enabled'] else 'off'}: convention {conv}, cross {p['cross']}, {flags}")
    kinds: dict[str, list[int]] = {}
    for t in report:
        k = kinds.setdefault(t["kind"], [0, 0])
        k[0 if t["applied"] and not t.get("note", "").endswith("off") else 1] += 1
    print("terms applied / not applied: " + ", ".join(f"{k} {v[0]} / {v[1]}" for k, v in kinds.items()))
    for t in report:
        dropped = ", ".join(f"{s} ({why})" for s, why in t.get("dropped", []))
        line = f"{t['target']:16s} {t['kind']:12s} {', '.join(t['applied']) or '-':40s}"
        extra = "; ".join(x for x in (dropped and "dropped: " + dropped, t.get("note", "")) if x)
        print(line + ("  " + extra if extra else ""))
    return 0


def _info(a: argparse.Namespace) -> int:
    from . import envelope

    buf = Path(a.chunk).read_bytes()
    cfg = load_config(a.config) if a.config is not None else None
    if cfg is not None:
        entries = cfg if isinstance(cfg, list) else [{"name": "icon_lattice", "configuration": cfg}]
        stamps = sorted({str(e["configuration"][STAMP_KEY]) for e in entries if STAMP_KEY in e["configuration"]})
        if stamps:
            print(f"configuration written by {', '.join(stamps)}")
    env = envelope.parse(buf)
    if env is None:
        return _info_container(buf, None if cfg is None else lattice_config(cfg), a.verbose)
    kind = "float32 array" if env.kind == envelope.KIND_ARRAY else "bytes of the last codec"
    print(
        f"{len(buf)} bytes, chain envelope: field array {list(env.shape)}, {len(env.headers)} stage headers, "
        f"payload {kind} ({len(env.payload)} bytes)"
    )
    _info_stages(env, cfg if isinstance(cfg, list) else None, a.verbose)
    if env.kind == envelope.KIND_BYTES:
        from .icon_lattice import container

        if bytes(env.payload[:4]) == container.MAGIC:
            print("lattice container:")
            return _info_container(env.payload, None if cfg is None else lattice_config(cfg), a.verbose)
    return 0


def _info_stages(env, entries: list[dict] | None, verbose: bool) -> None:
    from . import chain

    configs = {e["name"]: e["configuration"] for e in entries or []}
    for sid, header in env.headers:  # encode order
        d = chain.describe_header(sid, header, configs.get(sid), env.shape)
        per_slot = d.pop("per_slot", None)
        per_term = d.pop("per_term", None)
        print(f"stage {sid}: " + ", ".join(f"{k} {v}" for k, v in d.items()))
        if verbose and per_slot:
            for k, s in enumerate(per_slot):
                print(f"  {k:4d} {str(s['field']):16s} {s['kind']:10s} {s['sha256']}")
        if verbose and per_term:
            for t in per_term:
                coefs = "; ".join("bypass" if c is None else " ".join(f"{v:.6g}" for v in c) for c in t["coefficients"])
                src = "+".join(map(str, t["sources"]))
                print(f"  {str(t['target']):16s} <- {src:24s} bypassed {t['bypassed']}, coefficients {coefs}")


def _info_container(buf, cfg: dict | None, verbose: bool) -> int:
    from .icon_lattice import container
    from .icon_lattice.fields import FieldIndex

    ch = container.parse(buf)
    print(f"{len(buf)} bytes, {len(ch.modes)} fields, {len(ch.dates)} dates, structure {ch.structure_sha.hex()}")
    names = None
    if cfg is not None:
        vnames = [v if isinstance(v, str) else v["name"] for v in cfg["variables"]]
        idx = FieldIndex(vnames, int(cfg.get("n_ensembles", 1)))
        names = [idx.label(i) for i in range(len(ch.modes))]
    refs = container.field_references(buf)

    def nm(i: int) -> str:
        return names[i] if names else f"#{i}"

    counts: dict[str, list[int]] = {}
    for i, r in enumerate(container.inspect_chunk(buf)):
        lab = container.MODE_LABELS[r.mode]
        c = counts.setdefault(lab, [0, 0])
        c[0] += 1
        c[1] += r.nbytes
        if verbose:
            ref = refs[i]
            terms = []
            if ref.get("cross"):
                terms.append("cross: " + ", ".join(nm(j) for j in ref["cross"]))
            if "mask" in ref:
                terms.append(f"below-ground mask on {nm(ref['mask'])}")
            if "recipe" in ref:
                terms.append(f"recipe {ref['recipe']['id']} of " + ", ".join(nm(j) for j in ref["recipe"]["inputs"]))
            print(f"{i:4d} {nm(i):16s} {lab:10s} {r.nbytes:10d} fallback {r.fallback}  {'; '.join(terms)}")
    for lab, (n, b) in counts.items():
        print(f"{lab:10s} {n:4d} fields {b / 2**20:9.2f} MiB")
    return 0


LAB_HELP = "research toolkit: bit accounting, residuals, oracles, references, ledger, fuzz"
VARIABLES_HELP = "the dataset's variables: store, or JSON / YAML list"
# errors reported in one line (anything else is a bug: traceback)
USER_ERRORS = (ValueError, LookupError, OSError, TypeError, ImportError)


def _command(a: argparse.Namespace) -> str:
    """The command's name for messages (``check``, ``physics expand``, ``lab fuzz``)."""
    sub = next((getattr(a, k) for k in ("grid_cmd", "physics_cmd", "lab_cmd") if getattr(a, k, None)), None)
    return f"{a.cmd} {sub}" if sub else a.cmd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="anemoi-datasets-codecs", description=__doc__.split("\n")[0])
    ap.add_argument("--debug", action="store_true", help="show the traceback of an error")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("isa", help="kernel variants").set_defaults(fn=_isa)
    sub.add_parser("grids", help="bundled grids").set_defaults(fn=_grids)
    g = sub.add_parser("grid", help="grid tools").add_subparsers(dest="grid_cmd", required=True)
    b = g.add_parser("build", help="lattice file of an ICON grid file")
    b.add_argument("grid_file", type=Path)
    b.add_argument("--name")
    b.add_argument("--out", type=Path, help="sidecar directory (default: the cache)")
    b.add_argument("--no-check", action="store_true", help="skip the neighbour-table validation")
    b.set_defaults(fn=_grid_build)
    for name, fn, default, what, sample in (
        ("attach", _attach, None, "copy the lattice and static fields into STORE/.codecs", "default: all"),
        (
            "check",
            _check,
            4,
            "variables, chain consistent, sidecar complete, decodes from the store alone (or a configuration file)",
            "default: 4; 0: all",
        ),
    ):
        p = sub.add_parser(name, help=what)
        p.add_argument("store", type=Path)
        p.add_argument("--array", default="data")
        p.add_argument("--sample", type=int, default=default, help=f"chunks to scan ({sample})")
        p.set_defaults(fn=fn)
    p = sub.add_parser("info", help="field table of a chunk file")
    p.add_argument("chunk", type=Path)
    p.add_argument("--config", type=Path, help="store or JSON configuration, for variable names")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=_info)
    ph = sub.add_parser("physics", help="physics tools").add_subparsers(dest="physics_cmd", required=True)
    p = ph.add_parser("expand", help="the codec chain of a lattice configuration, for an anemoi recipe")
    p.add_argument(
        "config",
        type=Path,
        help="lattice configuration: store, JSON / YAML file (configuration or {name, configuration})",
    )
    p.add_argument("--variables", type=Path, help=VARIABLES_HELP)
    p.add_argument("--sort", action="store_true", help="put the variables in anemoi's order (sorted by name)")
    p.add_argument(
        "--physics",
        help="the physics group: on, off, true, false, a JSON object or a JSON / YAML file (default: the "
        "configuration's)",
    )
    p.add_argument(
        "--margin",
        type=margin_arg,
        default=None,
        help="field_terms margin m: auto (default: from the store's statistics, else 2^-6) or a power of two in "
        "(0, 2^-4]",
    )
    p.add_argument(
        "--forcing-bound",
        type=float,
        default=None,
        help="a bound for the forcings (cos / sin local time, insolation) without one, e.g. 1e-6: coded lossily "
        "instead of losslessly (zarr passes no valid dates); not applied to the forcings the single codec makes "
        "recipes (physics.recipe_fit, on by default)",
    )
    p.add_argument("--compressors", action="store_true", help="print the output.compressors list only")
    p.add_argument("--yaml", action="store_true", help="YAML instead of JSON")
    p.set_defaults(fn=_expand)
    p = sub.add_parser("plan", help="the physics plan of a configuration (terms applied / dropped)")
    p.add_argument("config", type=Path, help="store, JSON configuration, or JSON {name, configuration}")
    p.add_argument("--variables", type=Path, help=VARIABLES_HELP + " (for a configuration without variables)")
    p.add_argument("--sort", action="store_true", help="put the variables in anemoi's order (sorted by name)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_plan)
    argv = sys.argv[1:] if argv is None else argv
    if [x for x in argv if not x.startswith("-")][:1] == ["lab"]:  # the lab is imported only when used
        from .lab.cli import add_parser as add_lab

        add_lab(sub)
    else:
        sub.add_parser("lab", help=LAB_HELP)
    a = ap.parse_args(argv)
    if a.debug:
        return a.fn(a)
    try:
        return a.fn(a)
    except USER_ERRORS as e:
        msg = f"{e.filename}: {e.strerror}" if isinstance(e, OSError) and e.filename and e.strerror else str(e)
        print(f"anemoi-datasets-codecs {_command(a)}: {msg or type(e).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
