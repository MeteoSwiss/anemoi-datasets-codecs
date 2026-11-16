"""``lab refs``: what the lattice codec spends on each lossy variable against reference compressors at the same
absolute error bound.

For one date the chunk is encoded once with the lattice codec (``container.inspect_chunk`` gives the bytes of every
field; the chunk is cached in the output directory). Every reference compresses the same field at the same bound,
in two layouts: ``1d``, the cells in native order (what a generic codec sees in an anemoi chunk), and ``2d``, the
cells placed on the lattice image (``layout.rows_cols()``) with the holes of the cut-out filled with the nearest
cell's value; only the cells are checked against the bound, the bytes are the whole image.

References (a package that is not installed gives "not installed" for its methods):

* lossy scientific compressors, error-bounded at ``tolerance``: ``zfp`` (``zfpy``, fixed accuracy) and ``sz3``
  (``pysz``, absolute error bound). The decoded cells are checked against the bound; a method that misses it is
  retried with the tolerance halved until it meets it, and the result records the tolerance used;
* lossless coders of the codec's own quantised integers (``encoder.quantise(x, bound, k, clip, 0)`` with the
  configuration's ``k`` and clip, int16 when they fit else int32), which measure entropy coding alone at exactly the
  codec's error: ``zstd`` (level 19), ``xz`` (preset 9 extreme), ``blosc-lz4`` and ``blosc-zstd`` (numcodecs Blosc,
  byte shuffle, clevel 5 and 9);
* ``blosc-lz4-f32``: Blosc lz4 (clevel 5, byte shuffle; anemoi's default) on the raw float32 field, lossless.

Every result is verified: lossy decodes against the bound, lossless decodes equal to their input.
"""

from __future__ import annotations

import argparse
import json
import lzma
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..icon_lattice import encoder
from ..icon_lattice.container import MODE_LABELS, inspect_chunk
from . import data as labdata

LAYOUTS = ("1d", "2d")
MAX_HALVINGS = 12


# --------------------------------------------------------------------------------------------------- compressors
# Each compressor takes an array and returns (nbytes, decoded array of the same shape and dtype).
def _zfp(a: np.ndarray, tol: float) -> tuple[int, np.ndarray]:
    import zfpy

    buf = zfpy.compress_numpy(a, tolerance=tol)
    return len(buf), zfpy.decompress_numpy(buf)


def _sz3(a: np.ndarray, tol: float) -> tuple[int, np.ndarray]:
    from pysz import sz, szConfig, szErrorBoundMode

    conf = szConfig(*a.shape)
    conf.errorBoundMode = szErrorBoundMode.ABS
    conf.absErrorBound = tol
    comp, _ = sz.compress(a, conf)
    dec, _ = sz.decompress(comp, a.dtype, a.shape)
    return int(comp.nbytes), np.asarray(dec).reshape(a.shape)


def _zstd(a: np.ndarray) -> tuple[int, np.ndarray]:
    raw = np.ascontiguousarray(a).tobytes()
    try:
        import zstandard

        comp = zstandard.ZstdCompressor(level=19).compress(raw)
        dec = zstandard.ZstdDecompressor().decompress(comp)
    except ImportError:
        exe = shutil.which("zstd")
        if exe is None:
            raise ModuleNotFoundError("zstandard or the zstd command") from None
        comp = subprocess.run([exe, "-19", "-q", "-c"], input=raw, capture_output=True, check=True).stdout
        dec = subprocess.run([exe, "-d", "-q", "-c"], input=comp, capture_output=True, check=True).stdout
    return len(comp), np.frombuffer(dec, a.dtype).reshape(a.shape)


def _xz(a: np.ndarray) -> tuple[int, np.ndarray]:
    comp = lzma.compress(np.ascontiguousarray(a).tobytes(), preset=9 | lzma.PRESET_EXTREME, check=lzma.CHECK_NONE)
    return len(comp), np.frombuffer(lzma.decompress(comp), a.dtype).reshape(a.shape)


def _blosc(cname: str, clevel: int) -> Callable[[np.ndarray], tuple[int, np.ndarray]]:
    def run(a: np.ndarray) -> tuple[int, np.ndarray]:
        from numcodecs import Blosc

        a = np.ascontiguousarray(a)
        c = Blosc(cname=cname, clevel=clevel, shuffle=Blosc.SHUFFLE)
        comp = bytes(c.encode(a))
        return len(comp), np.frombuffer(c.decode(comp), a.dtype).reshape(a.shape)

    return run


@dataclass(frozen=True)
class Method:
    """A reference: ``kind`` is ``lossy`` (float32 field, tolerance), ``int`` (the codec's quantised integers,
    lossless) or ``float`` (the raw float32 field, lossless); ``modules`` are the imports it needs."""

    name: str
    kind: str
    run: Callable[..., tuple[int, np.ndarray]]
    modules: tuple[str, ...] = ()
    label: str = ""

    def missing(self) -> str | None:
        """The name of a missing dependency, None when the method can run."""
        import importlib.util

        for m in self.modules:
            if importlib.util.find_spec(m) is None:
                return m
        if self.name == "zstd" and importlib.util.find_spec("zstandard") is None and shutil.which("zstd") is None:
            return "zstandard"
        return None


METHODS: dict[str, Method] = {
    m.name: m
    for m in (
        Method("zfp", "lossy", _zfp, ("zfpy",), "zfp fixed-accuracy"),
        Method("sz3", "lossy", _sz3, ("pysz",), "SZ3 abs"),
        Method("zstd", "int", _zstd, (), "zstd -19 on ints"),
        Method("xz", "int", _xz, (), "xz -9e on ints"),
        Method("blosc-lz4", "int", _blosc("lz4", 5), ("numcodecs",), "Blosc lz4 shuffle on ints"),
        Method("blosc-zstd", "int", _blosc("zstd", 9), ("numcodecs",), "Blosc zstd shuffle on ints"),
        Method("blosc-lz4-f32", "float", _blosc("lz4", 5), ("numcodecs",), "Blosc lz4 shuffle on float32"),
    )
}


# --------------------------------------------------------------------------------------------------- geometry
class Image:
    """Placement of the cells on the lattice image: ``put`` returns the (H, W) image of a per-cell array with the
    holes filled by the nearest cell's value."""

    def __init__(self, rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int]):
        from scipy import ndimage

        self.rows, self.cols, self.shape = np.asarray(rows), np.asarray(cols), tuple(shape)
        occupied = np.zeros(self.shape, bool)
        occupied[self.rows, self.cols] = True
        self.fill = ndimage.distance_transform_edt(~occupied, return_distances=False, return_indices=True)

    @classmethod
    def of(cls, layout: Any) -> Image:
        rows, cols = layout.rows_cols()
        return cls(rows, cols, layout.shape)

    def put(self, v: np.ndarray) -> np.ndarray:
        img = np.zeros(self.shape, v.dtype)
        img[self.rows, self.cols] = v
        return np.ascontiguousarray(img[self.fill[0], self.fill[1]])

    def take(self, img: np.ndarray) -> np.ndarray:
        """The cells of an image."""
        return img[self.rows, self.cols]


def int_dtype(n: np.ndarray) -> np.dtype:
    """int16 when the quantised integers fit, else int32."""
    return np.dtype(np.int16 if n.size == 0 or (n.min() >= -(2**15) and n.max() < 2**15) else np.int32)


# --------------------------------------------------------------------------------------------------- measuring
def max_error(decoded: np.ndarray, x: np.ndarray) -> float:
    """Largest |decoded - x| over the cells (float64)."""
    return float(np.abs(decoded.astype(np.float64) - x.astype(np.float64)).max())


def measure_field(
    x: np.ndarray,
    bound: float,
    k: float,
    clip: tuple | None,
    image: Image | None,
    methods: list[str],
    layouts: tuple[str, ...] = LAYOUTS,
) -> list[dict[str, Any]]:
    """Every method in every layout on one float32 field: records ``{method, layout, bytes, max_error, tolerance,
    halvings}`` or ``{method, layout, error}``. ``image`` None with ``2d`` in ``layouts``: the ``2d`` records are
    ``not installed (scipy)`` (the image needs ``scipy.ndimage``, see :func:`run`)."""
    x = np.ascontiguousarray(x, np.float32)
    out: list[dict[str, Any]] = []
    quant: dict[str, Any] = {}
    for name in methods:
        m = METHODS[name]
        for lay in layouts:
            rec: dict[str, Any] = {"method": name, "layout": lay}
            miss = m.missing() or ("scipy" if lay == "2d" and image is None else None)
            if miss:
                out.append({**rec, "error": f"not installed ({miss})"})
                continue
            try:
                if m.kind == "lossy":
                    a = x if lay == "1d" else image.put(x)
                    tol, halvings = float(bound), 0
                    while True:
                        nbytes, dec = m.run(a, tol)
                        err = max_error(dec if lay == "1d" else image.take(dec), x)
                        if err <= bound or halvings >= MAX_HALVINGS:
                            break
                        tol, halvings = tol / 2, halvings + 1
                    if err > bound:
                        raise RuntimeError(f"bound missed after {halvings} halvings (max error {err:.6g})")
                    rec.update(bytes=nbytes, max_error=err, tolerance=tol, halvings=halvings)
                elif m.kind == "float":
                    a = x if lay == "1d" else image.put(x)
                    nbytes, dec = m.run(a)
                    if not np.array_equal(dec.view(np.uint32), a.view(np.uint32)):
                        raise RuntimeError("lossless roundtrip differs")
                    rec.update(bytes=nbytes, max_error=0.0, tolerance=0.0, halvings=0)
                else:
                    if not quant:
                        off, step, n = encoder.quantise(x, bound, k, clip, 0)
                        n = n.astype(int_dtype(n))
                        quant.update(n=n, err=max_error(encoder.reconstruct(off, step, n, clip), x))
                    n = quant["n"]
                    a = n if lay == "1d" else image.put(n)
                    nbytes, dec = m.run(a)
                    if not np.array_equal(dec, a):
                        raise RuntimeError("lossless roundtrip differs")
                    rec.update(bytes=nbytes, max_error=quant["err"], tolerance=float(bound), halvings=0)
            except Exception as e:  # a method that fails must not stop the others
                rec = {"method": name, "layout": lay, "error": f"{type(e).__name__}: {e}"}
            out.append(rec)
    return out


def encode_chunk(codec: Any, fields: np.ndarray, cache: Path | None, fresh: bool = False) -> bytes:
    """The encoded chunk of ``fields`` (read from ``cache`` when it is there and ``fresh`` is false)."""
    if cache is not None and cache.exists() and not fresh:
        return cache.read_bytes()
    chunk = codec.encode(fields[None, :, None, :])
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(chunk)
    return chunk


def run(
    store: str,
    config: Any,
    date: int,
    fields: list[str] | None = None,
    methods: list[str] | None = None,
    layouts: tuple[str, ...] = LAYOUTS,
    out_dir: Path | None = None,
    fresh: bool = False,
    log: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Records of every (variable, method, layout) of ``date``, plus the lattice codec's own (``method``
    ``lattice``, layout ``-``)."""
    log = log or (lambda s: None)
    codec = labdata.codec_for(config)
    arr, _ = labdata.open_data(store)
    fx = labdata.read_date(arr, date)
    lossy = labdata.lossy_fields(codec)
    if fields:
        unknown = sorted(set(fields) - {name for _, name, _, _ in lossy})
        if unknown:
            raise SystemExit(f"not lossy variables of the configuration: {', '.join(unknown)}")
        lossy = [f for f in lossy if f[1] in fields]
    cache = None
    if out_dir is not None:
        key = labdata.config_sha256(codec.get_config())[:12]
        cache = out_dir / f"refs-chunk-{Path(store).name}-d{date}-{key}.bin"
    log(f"encoding date {date} ({'cache ' + str(cache) if cache else 'no cache'})")
    chunk = encode_chunk(codec, fx, cache, fresh)
    recs = inspect_chunk(chunk)
    k = float(codec.options["k"])
    image = None
    if "2d" in layouts:
        try:
            image = Image.of(codec.layout)
        except ImportError:  # no scipy: the 2d records say so, 1d goes on
            log("layout 2d: not installed (scipy)")
    env = labdata.environment()
    names = methods or list(METHODS)
    rows: list[dict[str, Any]] = []
    for i, name, bound, clip in lossy:
        base = {
            "store": str(store),
            "date": date,
            "variable": name,
            "bound": bound,
            "n_cells": codec.n_cells,
            "env": env,
        }
        r = recs[i]
        rows.append(
            {**base, "method": "lattice", "layout": "-", "bytes": r.nbytes, "mode": MODE_LABELS.get(r.mode, r.mode)}
        )
        log(f"{name}: lattice {r.nbytes} B")
        for rec in measure_field(fx[i], bound, k, clip, image, names, layouts):
            if "bytes" in rec:
                rec["ratio"] = rec["bytes"] / max(r.nbytes, 1)
            rows.append({**base, **rec})
    return rows


# --------------------------------------------------------------------------------------------------- tables
def _cell(rec: dict[str, Any] | None, lat: int) -> str:
    if rec is None:
        return "-"
    if "error" in rec:
        return "n/i" if rec["error"].startswith("not installed") else "FAIL"
    s = f"{rec['bytes'] / max(lat, 1):.2f}"
    return s + (f"/{2 ** rec['halvings']}" if rec.get("halvings") else "")


def format_tables(rows: list[dict[str, Any]], layouts: tuple[str, ...] = LAYOUTS) -> str:
    """Per layout: one row per variable (lattice bytes, bits per cell, ratio of each method's bytes to the lattice
    codec's; ``/2^h`` marks a tolerance of bound / 2^h), then the totals (bytes, ratio)."""
    lat = {r["variable"]: r for r in rows if r["method"] == "lattice"}
    meths = [m for m in METHODS if any(r["method"] == m for r in rows)]
    by = {(r["variable"], r["method"], r["layout"]): r for r in rows if r["method"] != "lattice"}
    n_cells = next(iter(lat.values()))["n_cells"] if lat else 1
    out: list[str] = []
    for lay in layouts:
        out.append(f"\nlayout {lay}: bytes of each method / bytes of the lattice codec (/2^h: tolerance bound / 2^h)")
        head = f"{'variable':<16}{'lattice B':>11}{'bit/cell':>9}" + "".join(f"{m:>14}" for m in meths)
        out.append(head)
        for v, lr in lat.items():
            line = f"{v:<16}{lr['bytes']:>11}{8 * lr['bytes'] / n_cells:>9.3f}"
            line += "".join(f"{_cell(by.get((v, m, lay)), lr['bytes']):>14}" for m in meths)
            out.append(line)
        tot = f"{'TOTAL':<16}{sum(r['bytes'] for r in lat.values()):>11}"
        tot += f"{8 * sum(r['bytes'] for r in lat.values()) / n_cells / max(len(lat), 1):>9.3f}"
        tl, tm = [], []
        for m in meths:
            ok = [(v, by[(v, m, lay)]) for v in lat if (v, m, lay) in by and "bytes" in by[(v, m, lay)]]
            if ok:
                sm, sl = sum(r["bytes"] for _, r in ok), sum(lat[v]["bytes"] for v, _ in ok)
                tm.append(f"{sm / sl:>9.3f}" + (f" ({len(ok)})" if len(ok) < len(lat) else "").rjust(5))
            else:
                tm.append(f"{'n/i':>14}")
            tl.append(sm if ok else 0)
        out.append(tot + "".join(tm))
        out.append(f"{'total bytes':<36}" + "".join(f"{t:>14}" if t else f"{'-':>14}" for t in tl))
    return "\n".join(out)


# --------------------------------------------------------------------------------------------------- command
def add_parsers(sub: argparse._SubParsersAction) -> None:
    """Register ``lab refs``."""
    p = sub.add_parser("refs", help="lattice codec bytes against reference compressors at the same bound")
    p.add_argument("config", help="lattice codec configuration (store, JSON / YAML file)")
    p.add_argument("store", help="anemoi store (Zarr) with the data array")
    p.add_argument("--date", type=int, required=True, help="date index")
    p.add_argument("--fields", help="comma-separated lossy variables (default all)")
    p.add_argument("--methods", help=f"comma-separated, from {','.join(METHODS)} (default all)")
    p.add_argument("--layouts", default=",".join(LAYOUTS), help="1d, 2d or both (default)")
    p.add_argument("--out", help="JSON lines, one per variable, method and layout; its directory caches the chunk")
    p.add_argument("--fresh", action="store_true", help="ignore the cached chunk")
    p.add_argument("--json", action="store_true", help="print the records as JSON instead of the tables")
    p.set_defaults(fn=_main)


def _main(args: argparse.Namespace) -> int:
    methods = args.methods.split(",") if args.methods else None
    bad = [m for m in methods or () if m not in METHODS]
    layouts = tuple(args.layouts.split(","))
    if bad or not set(layouts) <= set(LAYOUTS):
        print(f"unknown method or layout: {bad or layouts}", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else None
    rows = run(
        args.store,
        args.config,
        args.date,
        args.fields.split(",") if args.fields else None,
        methods,
        layouts,
        out.resolve().parent if out else None,
        args.fresh,
        log=lambda s: print(s, file=sys.stderr, flush=True),
    )
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    if args.json:
        print(json.dumps(rows))
    else:
        print(format_tables(rows, layouts))
        miss = sorted(
            {
                ("layout 2d" if "(scipy)" in r["error"] else r["method"]) + r["error"][len("not installed") :]
                for r in rows
                if str(r.get("error", "")).startswith("not installed")
            }
        )
        if miss:
            print(f"\nnot installed: {', '.join(miss)}")
    return 0


__all__ = ["METHODS", "Image", "format_tables", "measure_field", "run", "add_parsers"]
