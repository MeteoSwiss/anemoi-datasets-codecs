"""lab minigrid: a rectangular cut-out of a grid's lattice with the real data of a store on it, so that the tools of
the toolkit run in seconds on one date instead of minutes.

``build CONFIG STORE OUT_DIR`` cuts a window ``[r0, r1) x [c0, c1)`` of the lattice image of the configuration's grid
(``LatticeCodec.layout``; default: about ``--fraction`` 1/16 of the cells around the centroid of the domain) and
writes into ``OUT_DIR``:

* ``mini.zarr``: a Zarr 3 store laid out as an anemoi dataset (``data`` float32 ``(dates, variables, 1, cells)``
  chunked one date per chunk with Blosc lz4; ``dates``, ``latitudes`` / ``longitudes`` cropped, the per-variable
  statistics copied, the group attributes copied with ``field_shape`` set to the mini cell count and a
  ``minigrid`` entry describing the cut-out), and its sidecar ``mini.zarr/.codecs/`` with the mini lattice and the
  static fields of the configuration (cropped), so that the store is self-contained;
* ``lattice/``: a sidecar directory with the same files (for ``ANEMOI_DATASETS_CODECS_PATH`` without the store);
* ``mini.json``: the codec configuration with ``grid`` replaced by the mini lattice's reference (name
  ``<grid>-mini-<fraction>``), everything else unchanged;
* ``cells.npy``: the native cell index on the full grid of every mini cell (int32, in mini native order);
* ``minigrid.json``: the manifest (source store and configuration, window, alignment, dates, the structure
  comparison with the full grid) and ``README.md``.

Mini native order: the cells of refinement groups cut by the window edge first, then the complete groups (a central
child and its 3 corners), each in full-grid native order. ICON numbers the 4 children of a parent consecutively, so
the last step of ``tree`` keeps a suffix of regular native groups (the decoder's fast path) as on the full grid.
A window inside the domain has no complete groups outside the regular native order (on ICON-CH1 the full grid's
are its lateral-boundary numbering, the ``L0hg`` step of ``tree``), so that step is missing on the mini grid; the
prediction kinds are the same (:func:`compare_structures`, ``same_prediction``).

Alignment rule. The frozen builders lay their patterns on absolute image coordinates:

* ``tree`` / ``tree_b``: central children are ``(r % 4, c % 4) in CENTRAL`` and a central's coordinates one level
  up are ``(r // 2, (c + 1) // 2)``. A shift of the image by ``(R, C)`` with ``R, C`` even shifts level 1 by
  ``(R / 2, C / 2)``, so the pattern is the same at levels ``0 .. L - 1`` (and the snake order of the delta-coded
  level ``L`` has the same row parity) iff ``R`` and ``C`` are multiples of ``2 ** (L + 1)``;
* ``hc2_k3`` (4 levels): the sub-lattices are ``r % u == 0 and (c - r) % 2u == 0`` with ``u = 2 ** 4``, for which
  ``R, C`` multiples of 32 suffice (``R % 16 == 0``, ``(C - R) % 32 == 0``).

So the window's top-left corner ``(r0, c0)`` must be a multiple of ``align = max(32, 2 ** (L + 1))`` with ``L`` the
number of tree levels of the full grid (5 on ICON-CH1: ``align`` 64). Then the cells of every refinement level of
the mini lattice are the full grid's cells of that level inside the window, the steps of every structure are of the
same kinds and the hc2_k3 hierarchy has the same 4 levels. The tree stops refining when a level has at most 2000
cells (``structures.tree.COARSEST``), so a mini grid with a fraction ``f`` of the cells has about ``log4(1 / f)``
fewer tree levels (2 at 1/16): its coarsest level is the full grid's level ``L - 2`` restricted to the window, the
finer levels are the same. The default window also has sides that are multiples of ``align``.

``check OUT_DIR`` builds and validates the three structures on the mini lattice, encodes one date of ``mini.zarr``
with ``LatticeCodec`` from ``mini.json`` and decodes it with nothing but ``OUT_DIR`` (``sidecar.isolated`` on the mini
store's sidecar), checks every lossy field against its bound and every other field for equality (forcing recipes
within the physics ``recipe_bound``), and prints bytes per cell, with ``--full STORE`` next to the same date encoded
on the full grid. Other tools use :func:`use` (registers the mini store's sidecar, returns ``mini.json``) or
``ANEMOI_DATASETS_CODECS_PATH=OUT_DIR/mini.zarr``.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from .. import sidecar, statics
from ..icon_lattice import container, grids
from ..icon_lattice.layout import TriangleLatticeLayout
from ..icon_lattice.structure import Structure, cell_image, get_structure
from ..icon_lattice.structures.tree import CENTRAL, COARSEST, level_images
from . import data

STRUCTURE_IDS = ("tree", "tree_b", "hc2_k3")
HC2_LEVELS = 4  # structures.hc2.build_hc2_k3's default
STATS = ("count", "has_nans", "maximum", "mean", "minimum", "squares", "stdev", "sums")
DEFAULT_VARS = ("T_2M", "PMSL", "U_10M", "T_850", "QV_850", "OMEGA_500", "CLCT", "TOT_PREC_1H", "FI_500")


@dataclass(frozen=True)
class Window:
    """Rows ``[r0, r1)`` and columns ``[c0, c1)`` of a lattice image."""

    r0: int
    r1: int
    c0: int
    c1: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.r1 - self.r0, self.c1 - self.c0

    def __str__(self) -> str:
        return f"{self.r0}:{self.r1},{self.c0}:{self.c1}"

    def as_list(self) -> list[int]:
        return [self.r0, self.r1, self.c0, self.c1]


def parse_window(text: str) -> Window:
    """``"r0:r1,c0:c1"`` -> ``Window``."""
    m = re.fullmatch(r"\s*(\d+):(\d+)\s*,\s*(\d+):(\d+)\s*", text)
    if not m:
        raise ValueError(f"window must be r0:r1,c0:c1, got {text!r}")
    return Window(*(int(v) for v in m.groups()))


def parse_fraction(text: str | float) -> Fraction:
    """``"1/16"`` or ``0.0625`` -> ``Fraction`` in (0, 1]."""
    f = Fraction(text).limit_denominator(1 << 20) if not isinstance(text, str) else Fraction(text.strip())
    if not 0 < f <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {text}")
    return f


def fraction_label(f: Fraction) -> str:
    """``1/16`` -> ``"1-16"`` (the mini lattice's name suffix)."""
    return f"{f.numerator}-{f.denominator}" if f.denominator != 1 else str(f.numerator)


# ---------------------------------------------------------------------------------------------- geometry
def tree_levels(layout: TriangleLatticeLayout) -> int:
    """Number of refinement levels of the ``tree`` structure on ``layout`` (the builder's rule: refine until a level
    has at most ``COARSEST`` cells)."""
    cimg = cell_image(layout.pixel_index, layout.shape)
    imgs = [cimg]
    while (imgs[-1] >= 0).sum() > COARSEST:
        imgs = level_images(cimg, len(imgs))
    return len(imgs) - 1


def alignment(layout: TriangleLatticeLayout) -> int:
    """The alignment of a window's top-left corner on ``layout`` (module docstring): ``max(32, 2 ** (L + 1))``."""
    return max(2 ** (HC2_LEVELS + 1), 2 ** (tree_levels(layout) + 1))


def check_window(layout: TriangleLatticeLayout, win: Window, align: int) -> None:
    """Raise ``ValueError`` unless ``win`` lies in the image, has sides >= 32 and an aligned top-left corner."""
    h, w = layout.shape
    if not (0 <= win.r0 < win.r1 <= h and 0 <= win.c0 < win.c1 <= w):
        raise ValueError(f"window {win} is not inside the image {h} x {w}")
    if min(win.shape) < 32:
        raise ValueError(f"window {win}: both sides must be >= 32")
    if win.r0 % align or win.c0 % align:
        raise ValueError(
            f"window {win}: r0 and c0 must be multiples of {align} so that the structures keep their pattern "
            f"(see lab/minigrid.py)"
        )


def _counts(layout: TriangleLatticeLayout) -> np.ndarray:
    """Summed-area table of the occupied pixels (``(H + 1, W + 1)``)."""
    occ = np.zeros(layout.shape[0] * layout.shape[1], np.int64)
    occ[layout.pixel_index] = 1
    sat = np.zeros((layout.shape[0] + 1, layout.shape[1] + 1), np.int64)
    sat[1:, 1:] = occ.reshape(layout.shape).cumsum(0).cumsum(1)
    return sat


def _n_inside(sat: np.ndarray, win: Window) -> int:
    return int(sat[win.r1, win.c1] - sat[win.r0, win.c1] - sat[win.r1, win.c0] + sat[win.r0, win.c0])


def default_window(
    layout: TriangleLatticeLayout, fraction: Fraction | float = Fraction(1, 16), align: int = 0
) -> Window:
    """A window of about ``fraction`` of the cells around the centroid of the domain, its corners on multiples of
    ``align`` (default :func:`alignment`), its aspect close to the image's."""
    align = align or alignment(layout)
    h, w = layout.shape
    rows, cols = layout.rows_cols()
    rmid, cmid = rows.mean(), cols.mean()
    target = float(fraction) * layout.n_cells
    aspect = h / w
    sat = _counts(layout)
    best: tuple[float, Window] | None = None
    for wh in range(align, h + 1, align):
        for ww in range(align, w + 1, align):
            if min(wh, ww) < 32:
                continue
            r0 = int(np.clip(round((rmid - wh / 2) / align) * align, 0, (h - wh) // align * align))
            c0 = int(np.clip(round((cmid - ww / 2) / align) * align, 0, (w - ww) // align * align))
            win = Window(r0, r0 + wh, c0, c0 + ww)
            n = _n_inside(sat, win)
            if n == 0:
                continue
            score = abs(math.log(n / target)) + 0.5 * abs(math.log((wh / ww) / aspect)) + 0.2 * (1 - n / (wh * ww))
            if best is None or score < best[0]:
                best = (score, win)
    if best is None:
        raise ValueError(f"no aligned window (align {align}) fits the image {h} x {w}")
    return best[1]


def _group_keys(rows: np.ndarray, cols: np.ndarray, w: int) -> np.ndarray:
    """Pixel (``row * w + col``) of the central child of the refinement group of every cell at ``(rows, cols)``
    (lattice convention of ``structures.tree``: every corner has exactly one central among its 3 edge neighbours)."""

    def central(r: np.ndarray, c: np.ndarray) -> np.ndarray:
        m = np.zeros(len(r), bool)
        for a, b in CENTRAL:
            m |= (r % 4 == a) & (c % 4 == b)
        return m

    key = np.full(len(rows), -1, np.int64)
    cen = central(rows, cols)
    key[cen] = rows[cen] * w + cols[cen]
    vert = np.where((rows + cols) % 2 == 0, rows - 1, rows + 1)
    for rr, cc in ((rows, cols - 1), (rows, cols + 1), (vert, cols)):
        hit = (key < 0) & central(rr, cc)
        key[hit] = rr[hit] * w + cc[hit]
    assert (key >= 0).all()
    return key


def crop(
    layout: TriangleLatticeLayout, win: Window, name: str | None = None
) -> tuple[TriangleLatticeLayout, np.ndarray]:
    """The mini lattice of the cells of ``layout`` inside ``win`` and the native index on ``layout`` of every mini cell
    (int32), in mini native order (module docstring)."""
    rows, cols = layout.rows_cols()
    inside = (rows >= win.r0) & (rows < win.r1) & (cols >= win.c0) & (cols < win.c1)
    cells = np.flatnonzero(inside)
    if len(cells) == 0:
        raise ValueError(f"window {win} holds no cell")
    r, c = rows[cells], cols[cells]
    key = _group_keys(r, c, layout.shape[1])
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    complete = cnt[inv] == 4
    cells = cells[np.lexsort((cells, complete))]
    r, c = rows[cells] - win.r0, cols[cells] - win.c0
    wh, ww = win.shape
    meta = {
        "name": name or f"{layout.meta.get('name', 'grid')}-mini",
        "source": f"window {win} of {layout.meta.get('name') or layout.sha256[:12]} (sha256 {layout.sha256})",
        "window": win.as_list(),
        "parent_sha256": layout.sha256,
    }
    mini = TriangleLatticeLayout((r * ww + c).astype(np.int32), (wh, ww), layout.orientation, meta)
    return mini, cells.astype(np.int32)


# ---------------------------------------------------------------------------------------------- structures
def _kind(name: str) -> str:
    """Step kind: the name with its level digits replaced (``L3dg`` -> ``L#dg``)."""
    return re.sub(r"\d+", "#", name)


def _prediction(name: str) -> str:
    """Prediction kind: the step kind without the tree's decoder-order split suffixes (``h`` head of the regular
    native groups, ``g`` parent triples, ``r`` the rest), which depend on the native order, not the prediction."""
    k = _kind(name)
    return re.sub(r"h?[gr]$", "", k) if k.startswith("L#") else k


def structure_summary(s: Structure) -> dict[str, Any]:
    """Steps ``[name, k, n]``, levels, gathers per cell and the regular native groups of a structure."""
    levels = sorted({int(m.group(1)) for st in s.steps if (m := re.match(r"[LAB](\d+)", st.name))})
    reg = s.regular_groups()
    return {
        "n_cells": s.n_cells,
        "steps": [[st.name, st.k, st.n] for st in s.steps],
        "levels": len(levels),
        "gathers_per_cell": round(sum(st.k * st.n for st in s.steps if st.kind == "gather") / s.n_cells, 4),
        "delta_share": round(s.steps[0].n / s.n_cells, 5),
        "regular_groups": 0 if reg is None else reg[1],
    }


def build_structures(layout: TriangleLatticeLayout, ids: tuple[str, ...] = STRUCTURE_IDS) -> dict[str, Structure]:
    """The structures ``ids`` on ``layout`` (``structure.get_structure``: the frozen builders, through the cache when
    it is on), each validated."""
    out = {}
    for sid in ids:
        s = get_structure(sid, layout)
        s.validate()
        out[sid] = s
    return out


def compare_structures(full: dict[str, Structure], mini: dict[str, Structure]) -> dict[str, Any]:
    """Per structure: the summaries on both lattices, whether the mini's step kinds (names with the level digits
    replaced) are those of the full grid (``same_kinds``) and whether its prediction kinds are (``same_prediction``,
    without the native-order split suffixes of the tree steps)."""
    out = {}
    for sid, sm in mini.items():
        a, b = structure_summary(full[sid]), structure_summary(sm)
        kinds_full = sorted({_kind(n) for n, _, _ in a["steps"]})
        kinds_mini = sorted({_kind(n) for n, _, _ in b["steps"]})
        out[sid] = {
            "full": a,
            "mini": b,
            "same_kinds": kinds_full == kinds_mini,
            "kinds_only_full": sorted(set(kinds_full) - set(kinds_mini)),
            "kinds_only_mini": sorted(set(kinds_mini) - set(kinds_full)),
            "same_prediction": {_prediction(n) for n in kinds_full} == {_prediction(n) for n in kinds_mini},
        }
    return out


def level_cells(layout: TriangleLatticeLayout, k: int) -> list[np.ndarray]:
    """Native indices of the cells of refinement levels ``0 .. k`` of ``layout`` (``structures.tree.level_images``)."""
    imgs = level_images(cell_image(layout.pixel_index, layout.shape), k)
    return [np.sort(im[im >= 0]) for im in imgs]


# ---------------------------------------------------------------------------------------------- build
def _blosc():
    from zarr.codecs import BloscCodec

    return BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")


def _fill(a: Any) -> Any:
    f = a.fill_value
    return None if f is None else np.asarray(f, dtype=a.dtype)[()]


def _parse_dates(text: str | None, n: int) -> list[int]:
    if text in (None, "", "all"):
        return list(range(n))
    out = [int(v) for v in str(text).split(",") if v.strip()]
    bad = [i for i in out if not 0 <= i < n]
    if bad:
        raise ValueError(f"dates {bad} out of range (the store has {n})")
    return out


def build(
    config: str | os.PathLike[str] | dict[str, Any],
    store: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    fraction: Fraction | str | float | None = None,
    window: Window | str | None = None,
    dates: str | list[int] | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """Write the mini grid of ``store`` into ``out_dir`` (module docstring). Returns the manifest."""
    import zarr

    t0 = time.perf_counter()
    cfg = dict(config) if isinstance(config, dict) else data.load_lattice_config(config)
    out = Path(out_dir)
    mini_store = out / "mini.zarr"
    if mini_store.exists():
        raise FileExistsError(f"{mini_store} exists")
    codec = data.codec_for(cfg)
    full = codec.layout
    parent = codec.grid
    align = alignment(full)
    if window is not None:
        win = parse_window(window) if isinstance(window, str) else window
        label = f"r{win.r0}-{win.r1}-c{win.c0}-{win.c1}"
        frac = None
    else:
        frac = parse_fraction(fraction if fraction is not None else "1/16")
        win = default_window(full, frac, align)
        label = fraction_label(frac)
    check_window(full, win, align)
    name = f"{parent.get('name') or 'grid'}-mini-{label}"
    mini, cells = crop(full, win, name)
    log(f"window {win} (align {align}) of {full}: {mini.n_cells} cells ({mini.n_cells / full.n_cells:.4f})")

    full_s = build_structures(full)
    mini_s = build_structures(mini)
    comparison = compare_structures(full_s, mini_s)
    for sid, c in comparison.items():
        log(
            f"  {sid}: levels {c['full']['levels']} -> {c['mini']['levels']}, steps {len(c['full']['steps'])} -> "
            f"{len(c['mini']['steps'])}, same step kinds: {c['same_kinds']} (only on the full grid: "
            f"{c['kinds_only_full'] or 'none'}), same prediction kinds: {c['same_prediction']}"
        )

    src = zarr.open_group(str(store), mode="r")
    d = src["data"]
    if d.shape[-1] != full.n_cells:
        raise ValueError(f"{store}: data has {d.shape[-1]} cells, the grid {full.n_cells}")
    if d.shape[1] != len(codec.names):
        raise ValueError(f"{store}: data has {d.shape[1]} variables, the configuration {len(codec.names)}")
    if "variables" in src.attrs:
        names = src.attrs["variables"]
        names = ast.literal_eval(names) if isinstance(names, str) else list(names)
        if list(names) != list(codec.names):
            raise ValueError(f"{store}: the store's variables are not the configuration's (in order)")
    idx = _parse_dates(dates if isinstance(dates, str | None) else ",".join(map(str, dates)), d.shape[0])

    out.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(mini_store), mode="w", zarr_format=3)
    n_vars, n_ens = d.shape[1], d.shape[2]
    md = g.create_array(
        "data",
        shape=(len(idx), n_vars, n_ens, mini.n_cells),
        chunks=(1, n_vars, n_ens, mini.n_cells),
        dtype=d.dtype,
        fill_value=_fill(d),
        compressors=_blosc(),
    )
    md.attrs.update(dict(d.attrs))
    for name_ in sorted(src.array_keys()):
        if name_ == "data":
            continue
        a = src[name_]
        if name_ == "dates":
            values = a[...][idx]
        elif a.ndim == 1 and a.shape[0] == full.n_cells:
            values = a[...][cells]
        elif name_ in STATS or (a.ndim == 1 and a.shape[0] == n_vars):
            values = a[...]
        else:
            log(f"  {name_} {a.shape}: not copied (neither per cell, per date nor per variable)")
            continue
        b = g.create_array(
            name_, shape=values.shape, chunks=values.shape, dtype=a.dtype, fill_value=_fill(a), compressors=_blosc()
        )
        b.attrs.update(dict(a.attrs))
        b[...] = values
    first: np.ndarray | None = None
    for j, i in enumerate(idx):
        t = time.perf_counter()
        x = np.ascontiguousarray(np.asarray(d[i])[..., cells])
        md[j] = x
        if first is None:
            first = x
        else:
            moved = [
                n for v, n in enumerate(codec.names) if n in codec.static_fields and not np.array_equal(x[v], first[v])
            ]
            if moved:
                log(f"  date {i}: static fields differ from the first date: {moved}")
        log(f"  data[{i}] -> mini[{j}] in {time.perf_counter() - t:.1f} s")

    # the sidecar: lattice and static fields, in the store and in OUT_DIR/lattice
    sc_dirs = [mini_store / sidecar.SIDECAR, out / "lattice"]
    assert first is not None
    fields = first.reshape(-1, mini.n_cells)
    for sc in sc_dirs:
        grids.save_lattice(mini, sc)
        for v, n in enumerate(codec.names):
            if n in codec.static_fields:
                for e in range(n_ens):
                    statics.add_static(np.ascontiguousarray(fields[v * n_ens + e], np.float32), sc)

    mini_cfg = {**cfg, "grid": mini.ref()}
    manifest = {
        "source_store": str(Path(store).resolve()),
        "source_config": None if isinstance(config, dict) else str(Path(config).resolve()),
        "parent_grid": parent,
        "grid": mini.ref(),
        "window": win.as_list(),
        "align": align,
        "fraction": None if frac is None else str(frac),
        "n_cells": mini.n_cells,
        "cell_share": mini.n_cells / full.n_cells,
        "dates": idx,
        "config": cfg,
        "structures": {sid: {k: v for k, v in c.items() if k not in ("full", "mini")} for sid, c in comparison.items()},
        "structure_summaries": {sid: {"full": c["full"], "mini": c["mini"]} for sid, c in comparison.items()},
    }
    attrs = dict(src.attrs)
    if "field_shape" in attrs:
        attrs["field_shape"] = str([mini.n_cells]) if isinstance(attrs["field_shape"], str) else [mini.n_cells]
    attrs["minigrid"] = {k: manifest[k] for k in ("source_store", "parent_grid", "grid", "window", "dates")}
    g.attrs.update(attrs)
    (out / "mini.json").write_text(json.dumps(mini_cfg, indent=1) + "\n")
    np.save(out / "cells.npy", cells)
    (out / "minigrid.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    (out / "README.md").write_text(_readme(manifest, out))
    log(f"wrote {out} in {time.perf_counter() - t0:.0f} s")
    log(f"decode on the mini grid: ANEMOI_DATASETS_CODECS_PATH={mini_store.resolve()} (or lab.minigrid.use({out}))")
    return manifest


def _readme(m: dict[str, Any], out: Path) -> str:
    w = m["window"]
    rows = [
        f"# Mini grid {m['grid']['name']}",
        "",
        f"Cut-out of `{m['source_store']}` (grid {m['parent_grid'].get('name')}, {m['parent_grid']['n_cells']} cells): "
        f"lattice rows {w[0]}:{w[1]}, columns {w[2]}:{w[3]} (corner aligned to {m['align']}), {m['n_cells']} cells "
        f"({100 * m['cell_share']:.2f} % of the grid), {len(m['dates'])} dates (source indices {m['dates']}).",
        "Written by `anemoi-datasets-codecs lab minigrid build` (anemoi_datasets_codecs/lab/minigrid.py).",
        "",
        "* `mini.zarr`: the data (Zarr 3, as the source store; its sidecar `mini.zarr/.codecs` holds the mini lattice "
        "and the static fields);",
        "* `mini.json`: the codec configuration on the mini lattice;",
        "* `lattice/`: a sidecar directory with the lattice and the static fields;",
        "* `cells.npy`: native index on the full grid of every mini cell;",
        "* `minigrid.json`: the manifest (window, dates, source configuration, structure comparison).",
        "",
        "Use:",
        "",
        "```bash",
        f"anemoi-datasets-codecs lab minigrid check {out} --date 0",
        f"export ANEMOI_DATASETS_CODECS_PATH={(out / 'mini.zarr').resolve()}   # for any tool that encodes / decodes",
        "```",
        "",
        "```python",
        "from anemoi_datasets_codecs.lab import data, minigrid",
        f"config = minigrid.use({str(out)!r})   # registers the mini store's sidecar, returns mini.json",
        "codec = data.codec_for(config)",
        f"x = data.read_date(data.open_data({str(out / 'mini.zarr')!r})[0], 0)",
        "```",
        "",
    ]
    return "\n".join(rows)


def use(out_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Register the sidecar of ``OUT_DIR/mini.zarr`` in this process and return the mini codec configuration."""
    out = Path(out_dir)
    sidecar.register_store(out / "mini.zarr")
    return json.loads((out / "mini.json").read_text())


def manifest(out_dir: str | os.PathLike[str]) -> dict[str, Any]:
    return json.loads((Path(out_dir) / "minigrid.json").read_text())


# ---------------------------------------------------------------------------------------------- check
def roundtrip(codec: Any, x: np.ndarray, date: Any = None) -> dict[str, Any]:
    """Encode the fields ``x`` (``(fields, cells)``) of one date, decode them and check them: lossy fields within
    their bound, forcing recipes within the physics ``recipe_bound``, every other field bit-identical. Returns
    ``{enc, nbytes (per field), bad: [(label, error, allowed)], encode_s, decode_s}``."""
    t = time.perf_counter()
    enc = codec.encode(x, dates=None if date is None else [date])
    t_enc = time.perf_counter() - t
    t = time.perf_counter()
    dec = np.asarray(codec.decode(enc)).reshape(x.shape)
    t_dec = time.perf_counter() - t
    idx = data.field_index(codec)
    recipes = set(codec.plan.recipes)
    rb = float(codec.physics["recipe_bound"])
    bad = []
    for i in range(len(x)):
        v = codec.variables[idx.variable(i)]
        err = float(np.abs(dec[i].astype(np.float64) - x[i]).max())
        if v.lossy:
            if not err <= v.max_error:
                bad.append((idx.label(i), err, float(v.max_error)))
        elif v.name in recipes:
            if not err <= rb:
                bad.append((idx.label(i), err, rb))
        elif not np.array_equal(dec[i].view(np.uint32), x[i].view(np.uint32)):
            bad.append((idx.label(i), err, 0.0))
    nbytes = np.array([r.nbytes for r in container.inspect_chunk(enc)], np.int64)
    return {"enc": enc, "nbytes": nbytes, "bad": bad, "encode_s": t_enc, "decode_s": t_dec}


def check(
    out_dir: str | os.PathLike[str],
    date: int = 0,
    full: str | os.PathLike[str] | None = None,
    variables: tuple[str, ...] | None = None,
    structure: str | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """Validate the mini grid of ``out_dir`` (module docstring). Returns the results; ``ok`` is False on any failure."""
    out = Path(out_dir)
    m = manifest(out)
    mini_store = out / "mini.zarr"
    cfg = json.loads((out / "mini.json").read_text())
    if structure:
        cfg["structure"] = structure
    ok = True
    res: dict[str, Any] = {"out_dir": str(out), "date": date}
    with sidecar.isolated(mini_store):  # the mini store's own sidecar only: OUT_DIR is self-contained
        layout = grids.load_lattice(cfg["grid"])
        cells = np.load(out / "cells.npy")
        if len(cells) != layout.n_cells:
            raise ValueError(f"cells.npy has {len(cells)} entries, the lattice {layout.n_cells} cells")
        for sid, s in build_structures(layout).items():  # validated
            sm = structure_summary(s)
            log(f"{sid}: valid, {len(sm['steps'])} steps, {sm['levels']} levels, {sm['gathers_per_cell']} gathers/cell")
        codec = data.codec_for(cfg)
        arr, dates = data.open_data(mini_store)
        x = data.read_date(arr, date)
        r = roundtrip(codec, x, None if dates is None else dates[date])
    for label, err, allowed in r["bad"]:
        log(f"FAIL {label}: error {err:.3g} > {allowed:.3g}")
    ok &= not r["bad"]
    n = layout.n_cells
    log(
        f"mini {m['grid']['name']}: {n} cells, date {date}: {len(r['enc'])} bytes = {len(r['enc']) / n:.3f} B/cell; "
        f"encode {r['encode_s']:.2f} s, decode {r['decode_s'] * 1e3:.0f} ms; "
        + ("all fields within bounds" if not r["bad"] else f"{len(r['bad'])} fields FAIL")
    )
    res.update(n_cells=n, bytes=len(r["enc"]), bad=r["bad"], encode_s=r["encode_s"], decode_s=r["decode_s"])
    names = list(codec.names)
    lossy = [v.lossy for v in codec.variables]
    per_var = {nm: int(r["nbytes"][i]) for i, nm in enumerate(names)}  # n_ensembles 1 (anemoi sample layout)
    res["mini_bytes_per_cell"] = {nm: b / n for nm, b in per_var.items()}
    if full is not None:
        src_date = m["dates"][date]
        fcodec = data.codec_for(dict(m["config"], **({"structure": structure} if structure else {})))
        farr, fdates = data.open_data(full)
        fx = data.read_date(farr, src_date)
        if not np.array_equal(fx[:, cells].view(np.uint32), x.view(np.uint32)):
            log("FAIL the mini data is not the full store's data at cells.npy")
            ok = False
        fr = roundtrip(fcodec, fx, None if fdates is None else fdates[src_date])
        del fx
        ok &= not fr["bad"]
        nf = fcodec.n_cells
        res.update(full_bytes=len(fr["enc"]), full_encode_s=fr["encode_s"], full_decode_s=fr["decode_s"])
        res["full_bytes_per_cell"] = {nm: int(fr["nbytes"][i]) / nf for i, nm in enumerate(names)}
        bm, bf = len(r["enc"]) / n, len(fr["enc"]) / nf
        lm = sum(int(r["nbytes"][i]) for i in range(len(names)) if lossy[i]) / n
        lf = sum(int(fr["nbytes"][i]) for i in range(len(names)) if lossy[i]) / nf
        log(
            f"full {fcodec.grid.get('name')}: {nf} cells, source date {src_date}: {len(fr['enc'])} bytes = "
            f"{bf:.3f} B/cell; encode {fr['encode_s']:.1f} s, decode {fr['decode_s'] * 1e3:.0f} ms"
        )
        log(
            f"bytes per cell, mini / full: all fields {bm:.3f} / {bf:.3f} = {bm / bf:.3f}; "
            f"lossy fields {lm:.3f} / {lf:.3f} = {lm / lf:.3f}"
        )
        ratios = np.array(
            [res["mini_bytes_per_cell"][nm] / res["full_bytes_per_cell"][nm] for i, nm in enumerate(names) if lossy[i]]
        )
        q = np.percentile(ratios, [10, 50, 90])
        log(f"lossy fields, ratio mini / full per field: p10 {q[0]:.3f}, median {q[1]:.3f}, p90 {q[2]:.3f}")
        res.update(ratio_all=bm / bf, ratio_lossy=lm / lf, ratio_lossy_p10_50_90=q.tolist())
    show = [v for v in (variables or DEFAULT_VARS) if v in names]
    if show:
        log(f"{'variable':<14}{'bound':>10}{'mini B/cell':>13}" + (f"{'full B/cell':>13}{'ratio':>8}" if full else ""))
        for nm in show:
            v = codec.variables[names.index(nm)]
            b = "lossless" if not v.lossy else f"{v.max_error:.3g}"
            line = f"{nm:<14}{b:>10}{res['mini_bytes_per_cell'][nm]:>13.4f}"
            if full is not None:
                f_ = res["full_bytes_per_cell"][nm]
                line += f"{f_:>13.4f}{res['mini_bytes_per_cell'][nm] / f_ if f_ else float('nan'):>8.3f}"
            log(line)
    res["ok"] = bool(ok)
    log("check OK" if ok else "check FAILED")
    return res


# ---------------------------------------------------------------------------------------------- command line
def _build(a: argparse.Namespace) -> int:
    if a.fraction is not None and a.window is not None:
        raise SystemExit("give --fraction or --window, not both")
    build(a.config, a.store, a.out_dir, a.fraction, a.window, a.dates)
    return 0


def _check(a: argparse.Namespace) -> int:
    variables = tuple(v for v in a.vars.split(",") if v) if a.vars else None
    res = check(a.out_dir, a.date, a.full, variables, a.structure)
    if a.json:
        keep = {k: v for k, v in res.items() if k != "enc"}
        a.json.write_text(json.dumps(keep, indent=1, default=str) + "\n")
    return 0 if res["ok"] else 1


def add_parsers(sub: argparse._SubParsersAction) -> None:
    """Register ``lab minigrid build`` and ``lab minigrid check``."""
    p = sub.add_parser("minigrid", help="a cut-out of the grid with real data, for fast experiments")
    ms = p.add_subparsers(dest="minigrid_cmd", required=True)
    b = ms.add_parser("build", help="cut a window of the lattice and write the mini store, lattice and configuration")
    b.add_argument("config", type=Path, help="lattice codec configuration: JSON / YAML file, codec entry or store")
    b.add_argument("store", type=Path, help="the source store (anemoi layout, data on the configuration's grid)")
    b.add_argument("out_dir", type=Path)
    b.add_argument("--fraction", help="share of the cells of the default window (default 1/16)")
    b.add_argument("--window", help="r0:r1,c0:c1 of the lattice image (r0, c0 multiples of the alignment)")
    b.add_argument("--dates", default="all", help="all (default) or comma-separated date indices of the source")
    b.set_defaults(fn=_build)
    c = ms.add_parser("check", help="structures, roundtrip within bounds and bytes per cell on the mini grid")
    c.add_argument("out_dir", type=Path)
    c.add_argument("--date", type=int, default=0, help="date index in the mini store (default 0)")
    c.add_argument("--full", type=Path, help="the source store: also encode the date on the full grid")
    c.add_argument("--vars", help="comma-separated variables of the per-variable table")
    c.add_argument("--structure", choices=STRUCTURE_IDS, help="override the configuration's structure")
    c.add_argument("--json", type=Path, help="write the results as JSON")
    c.set_defaults(fn=_check)
