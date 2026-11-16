"""Worth of the physics layer on one real ICON-CH1 chunk (the local sample store): chunk size and error with the
physics on (default), off, and each part off on its own, for a structure / k. The encoder gets the date's valid date,
so the forcings can be recipes. Every chunk is decoded and checked against the bounds; one JSON line per setting is
printed and appended to OUT.jsonl.

    python scripts/physics_size.py DATE_INDEX STRUCTURE K OUT.jsonl

Inputs as in ``realdata_store.py`` (``$ANEMOI_DATASETS_CODECS_SAMPLE``, ``tests/data/icon_ch1_sample.json``).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from realdata_store import SAMPLE, sample_config  # noqa: E402

from anemoi_datasets_codecs.icon_lattice import LatticeCodec  # noqa: E402

SETTINGS = {
    "default": {},
    "physics off": "off",
    "cross off": {"cross": "off"},
    "surface_terms off": {"surface_terms": False},
    "level_terms off": {"level_terms": False},
    "hydrostatic off": {"hydrostatic": False},
    "forcings off": {"forcings": False},
}


def main() -> None:
    import zarr

    d, structure, k, out = int(sys.argv[1]), sys.argv[2], float(sys.argv[3]), sys.argv[4]
    z = zarr.open_group(str(SAMPLE), mode="r")
    x = z["data"][d, :, 0, :]
    date = z.attrs["sample"]["source_dates"][d]
    cfg = {**sample_config(), "structure": structure, "options": {"k": k}}
    for label, physics in SETTINGS.items():
        codec = LatticeCodec(**cfg, physics=physics)
        t = time.perf_counter()
        enc = codec.encode(x, dates=[date])
        t_enc = time.perf_counter() - t
        dec = codec.decode(enc).reshape(x.shape)
        worst, sq = 0.0, []
        for i, v in enumerate(codec.variables):
            e = dec[i].astype(np.float64) - x[i]
            if v.lossy:
                worst = max(worst, float(np.abs(e).max()) / v.max_error)
                sq.append(float(np.sqrt(np.mean(e**2))) / v.max_error)
            else:
                assert float(np.abs(e).max()) <= 1e-6, v.name
        if worst > 1.0:
            raise SystemExit(f"{label}: worst error / bound {worst}")
        row = {
            "date_index": d,
            "structure": structure,
            "k": k,
            "setting": label,
            "MiB": round(len(enc) / 2**20, 3),
            "worst_err_over_bound": round(worst, 4),
            "mean_rmse_over_bound": round(float(np.mean(sq)), 4),
            "encode_s": round(t_enc, 1),
        }
        print(json.dumps(row), flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
