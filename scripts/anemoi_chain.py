"""anemoi-datasets end to end with the codec chain: ``anemoi-datasets create`` from an ``anemoi_dataset`` source (the
ICON-CH1 sample, the first N dates, no GRIB) twice, with the ``output`` section written by ``physics expand`` (the
chain ``[field_terms, static_ref, icon_lattice_bytes]``) and with the single ``icon_lattice_bytes`` codec of the same
configuration; then, per store, ``open_dataset`` in a fresh process with only ``ANEMOI_DATASETS_CODECS_PATH=<store>``
(cache off) reads every date and checks every variable against its bound (the forcing recipes of the single codec,
``physics.recipe_fit`` being the default, against ``recipe_bound``; the others for equality) against the source, and
``anemoi-datasets-codecs check`` runs on the store. Prints and writes (``OUT_DIR/results.json``) bytes per date,
create and read times and the worst error / bound.

    python scripts/anemoi_chain.py OUT_DIR N_DATES [--structure tree] [--k 1] [--only chain|single]
        [--forcing-bound B]

``--forcing-bound B`` builds the chain of ``physics expand --forcing-bound B`` (the forcings lossy within B, checked
against B) instead of the chain that stores them losslessly.

Needs anemoi-datasets with Zarr 3 support (``.venv-anemoi3``) and the sample store
(``ANEMOI_DATASETS_CODECS_SAMPLE``, default ``.local/data/icon-ch1-sample.zarr``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = Path(os.environ.get("ANEMOI_DATASETS_CODECS_SAMPLE", ROOT / ".local/data/icon-ch1-sample.zarr"))
CONFIG = ROOT / "tests/data/icon_ch1_sample.json"
LZ4 = {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1}

READ = r"""
import json, sys, time
import numpy as np, zarr
from anemoi.datasets import open_dataset
store, sample, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
bounds, recipes = json.loads(sys.argv[4]), json.loads(sys.argv[5])
t0 = time.perf_counter()
ds = open_dataset(store)
x = np.asarray(ds[0:n])                                  # (dates, variables, ensembles, cells)
seconds = time.perf_counter() - t0
src = zarr.open_group(sample, mode="r")
names = list(src.attrs["variables"])
assert list(ds.variables) == names, "variables differ from the source's"
ref = np.asarray(src["data"][0:n])
worst, mismatched = 0.0, []
for i, v in enumerate(names):
    err = float(np.abs(x[:, i].astype(np.float64) - ref[:, i]).max())
    if v in bounds:
        worst = max(worst, err / bounds[v])
        if err > bounds[v]:
            mismatched.append([v, err, bounds[v]])
    elif v in recipes:  # a forcing recipe (or exact when stored losslessly)
        if err > recipes[v]:
            mismatched.append([v, err, recipes[v]])
    elif not np.array_equal(x[:, i].view(np.uint32), ref[:, i].view(np.uint32)):
        mismatched.append([v, err, None])
print(json.dumps({"read_seconds": seconds, "dates": n, "worst_err_over_bound": worst, "out_of_bound": mismatched}))
"""


def chunk_bytes(store: Path) -> int:
    return sum(f.stat().st_size for f in (store / "data" / "c").rglob("*") if f.is_file())


def sidecar_bytes(store: Path) -> int:
    side = store / ".codecs"
    return sum(f.stat().st_size for f in side.rglob("*") if f.is_file()) if side.is_dir() else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("n_dates", type=int)
    ap.add_argument("--structure", default="tree")
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--only", choices=("chain", "single"), help="build one of the two stores (results-<name>.json)")
    ap.add_argument("--forcing-bound", type=float, help="the chain of physics expand --forcing-bound")
    a = ap.parse_args()
    import yaml
    import zarr

    from anemoi_datasets_codecs import physics
    from anemoi_datasets_codecs.cli import read_variables
    from anemoi_datasets_codecs.config import resolve_variables
    from anemoi_datasets_codecs.icon_lattice import LatticeCodec

    a.out.mkdir(parents=True, exist_ok=True)
    cfg = {**json.loads(CONFIG.read_text()), "lossless": LZ4, "structure": a.structure, "options": {"k": a.k}}
    names, ranges = read_variables(SAMPLE)
    src = zarr.open_group(str(SAMPLE), mode="r")
    dates = [str(d)[:19] for d in src["dates"][: a.n_dates].astype("datetime64[s]")]
    chain = physics.expand(names, None, cfg, ranges=ranges, forcing_bound=a.forcing_bound)
    # the bound per lossy variable, resolved as the codecs resolve them (regular-expression keys, dict-form variables)
    lossy = {v.name: v.max_error for v in resolve_variables(names, cfg.get("bounds"), cfg.get("clip")) if v.lossy}
    single_codec = LatticeCodec(**{**cfg, "variables": names})
    single = [{"name": "icon_lattice_bytes", "configuration": single_codec.get_config()}]
    # the forcings the single codec stores as recipes (recipe_fit, the default): within recipe_bound
    recipe_tol = dict.fromkeys(single_codec.plan.recipes, float(single_codec.physics["recipe_bound"]))
    forcing_tol = {} if a.forcing_bound is None else dict.fromkeys(recipe_tol, a.forcing_bound)
    results: dict = {"sample": str(SAMPLE), "dates": dates, "structure": a.structure, "k": a.k}
    results["forcing_bound"] = a.forcing_bound
    for label, entries in (("chain", chain), ("single", single)):
        if a.only not in (None, label):
            continue
        recipe = {
            "dates": {"start": dates[0], "end": dates[-1], "frequency": "1h"},
            "input": {"anemoi_dataset": {"dataset": str(SAMPLE)}},
            "output": physics.output_block(entries),
        }
        rp = a.out / f"recipe-{label}.yaml"
        rp.write_text(yaml.safe_dump(recipe, sort_keys=False, width=120))
        target = a.out / f"{label}.zarr"
        env = {**os.environ, "ANEMOI_DATASETS_CODECS_PATH": str(target)}
        log = a.out / f"create-{label}.log"
        t0 = time.perf_counter()
        with log.open("w") as fh:
            p = subprocess.run(
                [sys.executable, "-m", "anemoi.datasets", "create", str(rp), str(target), "--overwrite"],
                stdout=fh,
                stderr=subprocess.STDOUT,
                env=env,
            )
        r: dict = {"create_returncode": p.returncode, "create_seconds": time.perf_counter() - t0, "log": str(log)}
        if p.returncode == 0:
            meta = json.loads((target / "data" / "zarr.json").read_text())
            r["codecs"] = [c["name"] for c in meta["codecs"]]
            r["chunks"] = meta["chunk_grid"]["configuration"]["chunk_shape"]
            r["bytes_per_date"] = chunk_bytes(target) / len(dates)
            r["sidecar_bytes"] = sidecar_bytes(target)
            r["config_bytes"] = len(json.dumps(meta["codecs"]))
            cold = {k: v for k, v in os.environ.items() if not k.startswith("ANEMOI_DATASETS_CODECS_")}
            cold.update(ANEMOI_DATASETS_CODECS_PATH=str(target), ANEMOI_DATASETS_CODECS_CACHE="off")
            for k in ("ANEMOI_DATASETS_CODECS_THREADS",):
                if k in os.environ:
                    cold[k] = os.environ[k]
            bounds, recipes = json.dumps(lossy), json.dumps(recipe_tol if label == "single" else forcing_tol)
            q = subprocess.run(
                [sys.executable, "-c", READ, str(target), str(SAMPLE), str(len(dates)), bounds, recipes],
                capture_output=True,
                text=True,
                env=cold,
            )
            r["read"] = json.loads(q.stdout.strip().splitlines()[-1]) if q.returncode == 0 else q.stderr[-2000:]
            c = subprocess.run(
                [sys.executable, "-m", "anemoi_datasets_codecs.cli", "check", str(target), "--sample", str(len(dates))],
                capture_output=True,
                text=True,
                env=cold,
            )
            r["check"] = json.loads(c.stdout) if c.stdout.strip().startswith("{") else c.stderr[-2000:]
            r["check_returncode"] = c.returncode
        results[label] = r
        print(label, json.dumps(r, default=str)[:2000], flush=True)
    if all("bytes_per_date" in results.get(k, {}) for k in ("chain", "single")):
        c, s = results["chain"]["bytes_per_date"], results["single"]["bytes_per_date"]
        results["chain_vs_single"] = (c - s) / s
        print(f"chain {c / 2**20:.3f} MiB / date, single {s / 2**20:.3f} MiB / date: {100 * (c - s) / s:+.2f}%")
    name = "results.json" if a.only is None else f"results-{a.only}.json"
    (a.out / name).write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
