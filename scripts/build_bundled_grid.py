"""Build the bundled lattice file of a grid (``src/anemoi_datasets_codecs/icon_lattice/data/grids``) from its
ICON grid file.

    uv run --extra grid python scripts/build_bundled_grid.py GRID.nc NAME

The lattice is recovered from the cell centres (scipy), validated against the grid file's neighbour table, and
written as ``<sha256>.npz``; ``index.json`` maps NAME (and the grid's uuid) to the reference. For ICON-CH1 (grid
file ``icon_grid_0001_R19B08_mch.nc``, uuid ``17643da2-5749-59b6-44d2-54a3cd6e2bc0``) the sha256 must stay
``2c359f6e...``: the structure sha of every existing chunk derives from it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from anemoi_datasets_codecs.icon_lattice import gridfile, grids

OUT = Path(__file__).resolve().parents[1] / "src" / "anemoi_datasets_codecs" / "icon_lattice" / "data" / "grids"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("grid_file", type=Path)
    ap.add_argument("name")
    args = ap.parse_args()
    t = time.time()
    lay = gridfile.lattice_from_grid_file(args.grid_file, args.name)
    print(f"lattice of {args.grid_file.name}: {lay} sha256 {lay.sha256} ({time.time() - t:.1f} s, validated)")
    path = grids.save_lattice(lay, OUT.parent)
    index_path = OUT / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index[args.name] = lay.ref()
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB) and {index_path}: {lay.ref()}")


if __name__ == "__main__":
    main()
