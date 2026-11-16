"""lab account / explain / diff: where the bits of a lattice codec chunk go.

``lab account CHUNK [--config STORE_OR_JSON]`` reads an existing chunk file (no re-encode; a chained chunk's lattice
container is found inside its envelope) and splits every field's payload into its parts: for a LATTICE field the
fixed header, the mask block, the steps section, the entropy tables, the rANS streams, the escapes and the padding
(docs/codecs/icon_lattice.md "LATTICE payload"), with the header's n_tables, prob_bits, escape type, recon_bits and
escape units; with a configuration (or when the chunk sits in a store) also the steps section per structure step
(context buckets, rANS bytes of the multi-range steps, escape units). It prints per field and for the chunk the share
of each part.

``lab account --encode CONFIG STORE --date I`` re-encodes date ``I`` of ``STORE`` with ``CONFIG`` and records, per
LATTICE field and step, the symbols, escapes, table ids and spreads the encoder coded (``LatticeCodec.encode(...,
record=)``): actual coded bits against the order-0 entropy of the symbols per step (no contexts), the conditional
entropy given the encoder's tables (``H_ctx``; coder loss = actual rANS bits - H_ctx, split into the loss of the
normalised table frequencies and the per-stream flushes), and given finer contexts (32 quantile buckets of the spread,
per step or with global thresholds and tables shared by the steps: the headroom of more buckets, gross and net of
their table bytes); the bits per target per step (refinement level). The whole date is encoded (cross terms need their
sources); ``--fields`` restricts what is recorded.

``lab explain CHUNK --config CFG`` says why each field is stored as it is: lossless fallback (1: the lossy coding
failed, 2: lossless was smaller), the plain rule because of exact zeros (recon_bits 0), the plan's cross terms that
are absent from the payload or have a zero coefficient in every step, the mask term, escape type, recon bits and the
table layout. ``lab diff CHUNK_A CHUNK_B`` compares two chunks field by field (bytes, mode, the LATTICE parts).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .. import envelope
from ..icon_lattice import container, encoder, native
from ..icon_lattice.container import (
    LATTICE_HEADER,
    MASK_BLOCK,
    MODE_LABELS,
    MODE_LATTICE,
)
from . import data

LATTICE_PARTS = ("lattice header", "mask block", "steps", "tables", "rans", "escapes", "padding")
FALLBACK_TEXT = {0: "", 1: "lossy coding failed or missed the bound", 2: "lossless was smaller"}
ESCAPE_TYPES = {1: "varint", 2: "int16", 4: "int32"}
FINE_BUCKETS = 32


# ------------------------------------------------------------------------------------------- chunk side
def find_container(buf: bytes) -> tuple[memoryview, dict[str, Any]]:
    """The lattice container of a chunk file (bare, or the payload of a chain envelope) and what wraps it
    (``envelope`` bytes before the container, the stage ids)."""
    env = envelope.parse(buf)
    if env is None:
        return memoryview(buf), {"envelope": 0, "stages": []}
    if env.kind == envelope.KIND_BYTES and bytes(env.payload[:4]) == container.MAGIC:
        return memoryview(env.payload), {
            "envelope": len(buf) - len(env.payload),
            "stages": [sid for sid, _ in env.headers],
        }
    raise ValueError("chained chunk without a lattice container (its payload is not coded by the lattice codec)")


def parse_tables(blob: Any, n_tables: int) -> list[dict[str, Any]]:
    """The entropy tables of a LATTICE payload: per table ``M`` (symbols before the escape), its bytes and the
    frequencies (int64 ``(256,)``)."""
    b = bytes(blob)
    p, out = 0, []
    for _ in range(n_tables):
        start, m = p, b[p]
        p += 1
        vals = []
        for _ in range(m + 1):
            v = sh = 0
            while True:
                x = b[p]
                p += 1
                v |= (x & 127) << sh
                sh += 7
                if x < 128:
                    break
            vals.append(v)
        f = np.zeros(256, np.int64)
        f[:m] = vals[:m]
        f[255] = vals[m]
        out.append({"M": m, "bytes": p - start, "freqs": f})
    if p != len(b):
        raise ValueError(f"tables section of {len(b)} bytes, parsed {p}")
    return out


def parse_steps(blob: Any, structure: Any, n_cross: int) -> list[dict[str, Any]]:
    """The steps section of a LATTICE payload under ``structure``: per step ``n`` targets, ``n_ctx``, table map,
    thresholds, weights and cross coefficients (gather steps), ranges ``(rANS bytes, escape units)``."""
    b = memoryview(bytes(blob))
    off, out = 0, []
    for s, st in enumerate(structure.steps):
        n_ctx = b[off]
        off += 1
        tmap = list(b[off : off + n_ctx])
        off += n_ctx
        thr = np.frombuffer(b, "<i4", n_ctx - 1, off).tolist()
        off += 4 * (n_ctx - 1)
        w = cc = None
        if st.kind == "gather":
            w = np.frombuffer(b, "<i2", st.k - 1, off).tolist()
            off += 2 * (st.k - 1)
            cc = np.frombuffer(b, "<i4", n_cross, off).tolist()
            off += 4 * n_cross
        nr = len(structure.step_ranges(s))
        rr = np.frombuffer(b, "<u4", 2 * nr, off).reshape(nr, 2).tolist()
        off += 8 * nr
        out.append(
            {"step": s, "kind": st.kind, "n": st.n, "k": st.k, "n_ctx": n_ctx, "tmap": tmap, "thr": thr, "w": w,
             "cc": cc, "ranges": rr, "rans_bytes": sum(r[0] for r in rr), "esc_units": sum(r[1] for r in rr)}
        )  # fmt: skip
    if off != len(b):
        raise ValueError(f"steps section of {len(b)} bytes does not match the structure ({off} parsed)")
    return out


def lattice_parts(payload: Any) -> tuple[dict[str, int], container.LatticeHeader]:
    """Bytes of each part of a LATTICE payload (:data:`LATTICE_PARTS`, summing to the payload) and its header."""
    p = memoryview(payload).cast("B")
    h = container.lattice_header(p)
    mask = 0
    if h.flags & 4:
        mask = MASK_BLOCK + 24 * p[LATTICE_HEADER + 2]
    head = LATTICE_HEADER + mask + h.steps_bytes + h.tables_bytes
    pad_a = -head % 4
    esc = h.n_esc * (1 if h.et == 1 else h.et)
    rest = len(p) - head - pad_a - h.rans_bytes - esc
    if rest < 0:
        raise ValueError("LATTICE payload shorter than its header says")
    parts = {
        "lattice header": LATTICE_HEADER,
        "mask block": mask,
        "steps": h.steps_bytes,
        "tables": h.tables_bytes,
        "rans": h.rans_bytes,
        "escapes": esc,
        "padding": pad_a + rest,
    }
    return parts, h


def _section(payload: Any, h: container.LatticeHeader, name: str) -> memoryview:
    p = memoryview(payload).cast("B")
    mask = MASK_BLOCK + 24 * p[LATTICE_HEADER + 2] if h.flags & 4 else 0
    a = LATTICE_HEADER + mask
    if name == "steps":
        return p[a : a + h.steps_bytes]
    a += h.steps_bytes
    return p[a : a + h.tables_bytes]


def field_entry(i: int, label: str, mode: int, fallback: int, payload: Any, structure: Any = None) -> dict[str, Any]:
    """The accounting of one field of a chunk (see :func:`account_chunk`)."""
    d: dict[str, Any] = {
        "index": i,
        "field": label,
        "mode": MODE_LABELS[int(mode)],
        "fallback": int(fallback),
        "bytes": len(payload),
    }
    if mode != MODE_LATTICE:
        return d
    parts, h = lattice_parts(payload)
    tables = parse_tables(_section(payload, h, "tables"), h.n_tables)
    d.update(
        parts=parts,
        vt=h.vt,
        et=h.et,
        n_tables=h.n_tables,
        prob_bits=h.prob_bits or 12,
        recon_bits=h.recon_bits,
        n_esc_units=h.n_esc,
        cross=list(h.cross),
        mask=h.mask_src,
        table_bytes=[t["bytes"] for t in tables],
        table_M=[t["M"] for t in tables],
    )
    if structure is not None and h.n_steps == len(structure.steps):
        steps = parse_steps(_section(payload, h, "steps"), structure, h.n_cross)
        d["n_ctx_max"] = max(s["n_ctx"] for s in steps)
        d["serial_rans_bytes"] = h.rans_bytes - sum(s["rans_bytes"] for s in steps)
        d["steps"] = steps
    return d


def config_near(chunk: Path) -> Path | None:
    """The store of a chunk file inside a store's ``data`` array (``STORE/data/c/...``), else None."""
    for p in Path(chunk).resolve().parents:
        meta = p / "zarr.json"
        if meta.is_file():
            try:
                node = json.loads(meta.read_text()).get("node_type")
            except (OSError, ValueError):
                return None
            return p.parent if node == "array" and p.name == "data" else None
    return None


def codec_of(config: Any) -> Any:
    """``LatticeCodec`` of a configuration path / dict / codec (None stays None)."""
    if config is None:
        return None
    from ..icon_lattice.codec import LatticeCodec

    return config if isinstance(config, LatticeCodec) else data.codec_for(config)


def labels_for(codec: Any, n_fields: int) -> list[str]:
    """Field labels of a chunk of ``n_fields`` (``#i`` without a codec or when the count does not match)."""
    if codec is not None and n_fields % codec.fields_per_date == 0:
        return data.field_labels(codec, n_fields // codec.fields_per_date)
    return [f"#{i}" for i in range(n_fields)]


def account_chunk(buf: bytes, codec: Any = None) -> dict[str, Any]:
    """Where the bytes of a chunk go: per field (mode, fallback, bytes, LATTICE parts and header values; per step
    with a codec) and the chunk totals per part (``parts``: bytes, ``share``: fraction of the chunk)."""
    inner, wrap = find_container(buf)
    ch = container.parse(inner)
    structure = None
    if codec is not None:
        structure = codec.structure
        if structure.sha16 != ch.structure_sha:
            raise ValueError("the configuration's structure does not match the chunk's (structure sha)")
    labels = labels_for(codec, len(ch.modes))
    fields = [
        field_entry(i, labels[i], m, ch.fallbacks[i], ch.payloads[i], structure)
        for i, m in enumerate(ch.modes.tolist())
    ]
    totals: dict[str, int] = {"envelope": wrap["envelope"], "container header": ch.header_bytes}
    for name in LATTICE_PARTS:
        totals[name] = 0
    for f in fields:
        if f["mode"] == "lattice":
            for k, v in f["parts"].items():
                totals[k] += v
        else:
            totals[f["mode"]] = totals.get(f["mode"], 0) + f["bytes"]
    if sum(totals.values()) != len(buf):
        raise ValueError(f"parts sum to {sum(totals.values())} bytes, the chunk has {len(buf)}")
    modes: dict[str, list[int]] = {}
    for f in fields:
        c = modes.setdefault(f["mode"], [0, 0])
        c[0] += 1
        c[1] += f["bytes"]
    n_cells = codec.n_cells if codec is not None else None
    if n_cells:
        for f in fields:
            f["bits_per_cell"] = 8 * f["bytes"] / n_cells
    return {
        "bytes": len(buf),
        "n_fields": len(ch.modes),
        "n_dates": len(ch.dates),
        "n_cells": n_cells,
        "structure_sha": ch.structure_sha.hex(),
        "stages": wrap["stages"],
        "parts": {k: v for k, v in totals.items() if v or k in LATTICE_PARTS},
        "share": {k: v / len(buf) for k, v in totals.items() if v or k in LATTICE_PARTS},
        "modes": {k: {"fields": v[0], "bytes": v[1]} for k, v in modes.items()},
        "fields": fields,
    }


# ------------------------------------------------------------------------------------------ encode side
def table_ids(sc: Any, st: Any, s: int) -> np.ndarray:
    """Table id of every target of step ``s`` (what ``_layout_streams`` coded it with): the step's table map at the
    bucket of the spread (number of thresholds strictly below it), the mask table for masked targets."""
    thr = np.asarray(st.thr[s], np.int64)
    if sc.spread is None or len(thr) == 0:
        b = np.zeros(len(sc.sym), np.intp)
    else:
        b = np.searchsorted(thr, sc.spread, side="left")
    tid = np.asarray(st.tmap[s], np.intp)[b]
    if sc.masked is not None:
        tid = np.where(sc.masked, st.mask_table, tid)
    return tid


def residuals_of(sc: Any) -> np.ndarray:
    """The coded residuals ``q`` of a step (int64): zigzag inverse of the symbols, the escapes at symbol 255."""
    z = sc.sym.astype(np.int64)
    q = np.where(z % 2 == 0, z // 2, -(z + 1) // 2)
    esc = sc.sym == 255
    q[esc] = sc.esc
    return q


def _hist(sym: np.ndarray, ctx: np.ndarray, n_ctx: int) -> np.ndarray:
    idx = ctx.astype(np.intp) << 8
    idx |= sym
    return np.bincount(idx, minlength=n_ctx * 256).reshape(n_ctx, 256)


def _cond_bits(h: np.ndarray) -> float:
    """``H(symbol | row)`` in bits of a histogram ``(rows, 256)``."""
    h = h.astype(np.float64)
    tot = h.sum(1, keepdims=True)
    nz = h > 0
    p = np.divide(h, tot, out=np.ones_like(h), where=tot > 0)
    return float(-(h[nz] * np.log2(p[nz])).sum())


def _tables_bytes(h: np.ndarray, pbits: int) -> int:
    """Serialised bytes of the tables of the non-empty rows of a histogram at precision ``pbits``."""
    live = h[h.sum(1) > 0]
    if not len(live):
        return 0
    f = np.stack([native.normalize_freqs(r.astype(np.uint32), 1 << pbits) for r in live])
    return encoder._tables_len(f)


def summarise_record(pieces: dict[str, Any], structure: Any, fine: int = FINE_BUCKETS) -> dict[str, Any]:
    """Per step and per field statistics of one recorded LATTICE coding (``EncodeOptions.record``'s pieces).

    Bits: ``actual`` (8 x rANS bytes; per step only for multi-range steps, the serial stream is the rest),
    ``model`` (the static rANS cost under the coded tables, without the stream flushes), ``h0`` (order-0 entropy per
    step, the "no contexts" layout), ``h_ctx`` (conditional entropy given the table of each target, tables over the
    whole field), ``h_fine`` (given ``fine`` quantile buckets of the spread per step, masked targets apart: tables
    per (step, bucket)) and ``h_fine_shared`` (``fine`` global buckets with tables shared by the gather steps, the
    encoder's ``shared`` layout with more buckets). ``*_net`` ratios count the table and threshold bytes too (the
    encoder also charges ``table_penalty`` per table, a decode cost, which is left out here). ``stream_overhead``:
    ``actual - model`` per rANS stream (the 32-lane state flush, in bytes). ``tables_and_thresholds_bytes``: the
    tables section plus 4 bytes per context threshold (``parts["tables"]`` is the tables section alone)."""
    st, steps_code = pieces["streams"], pieces["steps"]
    pbits, n_tab = int(st.pbits), len(st.freqs)
    tids = [table_ids(sc, st, s) for s, sc in enumerate(steps_code)]
    counts = [_hist(sc.sym, t, n_tab) for sc, t in zip(steps_code, tids, strict=True)]
    tab_hist = np.sum(counts, axis=0).astype(np.float64)
    tab_tot = tab_hist.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.log2(tab_hist / tab_tot)
        log_f = np.log2(st.freqs.astype(np.float64))
    esc_w = 1 if st.et == 1 else st.et
    steps, h0_hists, fine_hists = [], [], []
    for s, (sc, c) in enumerate(zip(steps_code, counts, strict=True)):
        n = len(sc.sym)
        nz = c > 0
        h0_h = c.sum(0, keepdims=True)
        h0_hists.append(h0_h)
        if sc.spread is not None and n and fine > 1:
            thr = encoder._thresholds(sc.spread, fine)
            ctx = np.searchsorted(thr, sc.spread, side="left")
            if sc.masked is not None:
                ctx = np.where(sc.masked, fine, ctx)
            fh = _hist(sc.sym, ctx, fine + 1)
            n_thr = len(thr)
        else:
            fh, n_thr = h0_h, 0
        fine_hists.append((fh, n_thr))
        rr = st.ranges[s]
        multi = len(structure.step_ranges(s)) > 1
        q = residuals_of(sc)
        steps.append(
            {
                "step": s,
                "n": n,
                "n_ctx": int(st.n_ctx[s]),
                "tables": sorted({int(t) for t in np.unique(tids[s])}) if n else [],
                "actual": 8.0 * sum(r[0] for r in rr) if multi else None,
                "model": float((c[nz] * (pbits - log_f[nz])).sum()),
                "h0": _cond_bits(h0_h),
                "h_ctx": float(-(c[nz] * log_p[nz]).sum()),
                "h_fine": _cond_bits(fh),
                "n_esc": int(np.count_nonzero(sc.sym == 255)),
                "esc_bits": 8.0 * esc_w * sum(r[1] for r in rr),
                "mean_abs_q": float(np.abs(q).mean()) if n else 0.0,
                "zero_frac": float(np.count_nonzero(q == 0) / n) if n else 0.0,
                "cross_used": bool(np.any(sc.cc)),
            }
        )
    for d in steps:
        d["bits_per_target"] = (d["model"] + d["esc_bits"]) / d["n"] if d["n"] else 0.0
    # the shared layout at ``fine`` buckets: global thresholds over the gather steps' spreads, tables shared by them
    gath = [s for s, sc in enumerate(steps_code) if sc.spread is not None and len(sc.sym)]
    other = [h0_hists[s] for s in range(len(steps_code)) if s not in gath]
    if gath and fine > 1:
        gthr = encoder._thresholds(np.concatenate([steps_code[s].spread for s in gath]), fine)
        shared = np.zeros((fine + 1, 256), np.int64)
        for s in gath:
            sc = steps_code[s]
            ctx = np.searchsorted(gthr, sc.spread, side="left")
            if sc.masked is not None:
                ctx = np.where(sc.masked, fine, ctx)
            shared += _hist(sc.sym, ctx, fine + 1)
        other.append(shared)
        thr_bytes = 4 * len(gthr) * len(gath)
    else:
        thr_bytes = 0
    h_fine_shared = sum(_cond_bits(h) for h in other)
    tables_fine_shared = sum(_tables_bytes(h, pbits) for h in other) + thr_bytes
    n_streams = 1 + sum(len(structure.step_ranges(s)) for s in range(len(steps_code)) if steps[s]["actual"] is not None)
    tot = {k: sum(d[k] for d in steps) for k in ("model", "h0", "h_ctx", "h_fine", "esc_bits", "n_esc")}
    payload = pieces["payload"]
    parts, _ = lattice_parts(payload)
    n_cells = len(pieces["values"])
    actual = 8.0 * len(st.rans)
    tables0 = sum(_tables_bytes(h, pbits) for h in h0_hists)  # one table per step
    tables_fine = sum(_tables_bytes(h, pbits) + 4 * nt for h, nt in fine_hists)
    tables = len(st.tables_blob) + 4 * sum(len(t) for t in st.thr)
    h0, hc, hf, hs = tot["h0"], tot["h_ctx"], tot["h_fine"], h_fine_shared
    cur = hc / 8 + tables
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "n_cells": n_cells,
        "bits_per_cell": 8 * len(payload) / n_cells,
        "parts": parts,
        "pbits": pbits,
        "recon_bits": int(pieces["m"]),
        "n_tables": n_tab,
        "n_ctx_max": int(max(st.n_ctx)),
        "et": int(st.et),
        "actual": actual,
        **tot,
        "tables_and_thresholds_bytes": tables,
        "tables0_bytes": tables0,
        "tables_fine_bytes": tables_fine,
        "h_fine_shared": hs,
        "tables_fine_shared_bytes": tables_fine_shared,
        "n_streams": n_streams,
        "stream_overhead": (actual - tot["model"]) / 8 / n_streams,
        "escape_share": parts["escapes"] / len(payload),
        "coder_loss": (actual - hc) / hc if hc else 0.0,
        "table_loss": (tot["model"] - hc) / hc if hc else 0.0,
        "context_gain": (h0 - hc) / h0 if h0 else 0.0,
        "context_gain_net": ((h0 / 8 + tables0) - (hc / 8 + tables)) / (h0 / 8 + tables0) if h0 else 0.0,
        "headroom": (hc - hf) / hc if hc else 0.0,
        "headroom_net": (cur - (hf / 8 + tables_fine)) / cur if hc else 0.0,
        "headroom_shared": (hc - hs) / hc if hc else 0.0,
        "headroom_shared_net": (cur - (hs / 8 + tables_fine_shared)) / cur if hc else 0.0,
        "steps": steps,
    }


class Recorder:
    """The ``record`` callback of ``LatticeCodec.encode``: keeps, per recorded field, a summary of every LATTICE
    coding (:func:`summarise_record`, computed in the field's thread) and optionally the raw pieces (``keep``)."""

    def __init__(self, structure: Any, fields: set[int] | None = None, keep: bool = False, fine: int = FINE_BUCKETS):
        self.structure, self.fields, self.keep, self.fine = structure, fields, keep, fine
        self.records: dict[int, list[tuple[dict[str, Any], dict[str, Any] | None]]] = {}
        self._lock = threading.Lock()

    def __call__(self, i: int, pieces: dict[str, Any]) -> None:
        if self.fields is not None and i not in self.fields:
            return
        summary = summarise_record(pieces, self.structure, self.fine)
        with self._lock:
            self.records.setdefault(i, []).append((summary, pieces if self.keep else None))

    def final(self, chunk: bytes) -> dict[int, tuple[dict[str, Any], dict[str, Any] | None, int, int]]:
        """Per recorded field: the summary and pieces of the coding that is in ``chunk`` (the last attempt when the
        field was stored otherwise), its mode and fallback."""
        ch = container.parse(chunk)
        out = {}
        for i, recs in self.records.items():
            want = hashlib.sha256(bytes(ch.payloads[i])).hexdigest()
            pick = next((r for r in recs if r[0]["sha256"] == want), recs[-1])
            out[i] = (*pick, int(ch.modes[i]), int(ch.fallbacks[i]))
        return out


def select_fields(codec: Any, spec: str | None) -> set[int] | None:
    """Field indices (within the first date) of a comma-separated list of names / labels (None: all)."""
    if not spec:
        return None
    labels = data.field_labels(codec)
    want = [s.strip() for s in spec.split(",") if s.strip()]
    out = {i for i, lab in enumerate(labels) if lab in want or codec.names[data.field_index(codec).variable(i)] in want}
    missing = [w for w in want if w not in labels and w not in codec.names]
    if missing:
        raise SystemExit(f"--fields: unknown fields {missing}")
    return out


def encode_recorded(
    codec: Any, store: Any, date: int, fields: set[int] | None, keep: bool = False, valid_date: bool = True,
    fine: int = FINE_BUCKETS,
) -> tuple[bytes, dict[int, Any], dict[str, Any]]:  # fmt: skip
    """Encode date ``date`` of ``store`` with ``codec`` and the recorder: (chunk, :meth:`Recorder.final`, run info)."""
    arr, dates = data.open_data(store)
    x = data.read_date(arr, date)
    rec = Recorder(codec.structure, fields, keep, fine)
    valid = dates[date] if (valid_date and dates is not None) else None
    t0 = time.perf_counter()
    chunk = codec.encode(x, dates=valid, record=rec)
    run = {"date": date, "valid_date": None if valid is None else str(valid), "encode_s": time.perf_counter() - t0}
    return chunk, rec.final(chunk), run


def account_encode(
    codec: Any, store: Any, date: int, fields: set[int] | None = None, fine: int = FINE_BUCKETS,
    valid_date: bool = True, save_chunk: Path | None = None,
) -> dict[str, Any]:  # fmt: skip
    """``lab account --encode``: per recorded LATTICE field the summary of :func:`summarise_record`, plus totals and
    the bits per target per step summed over the fields."""
    chunk, final, run = encode_recorded(codec, store, date, fields, False, valid_date, fine)
    if save_chunk is not None:
        Path(save_chunk).write_bytes(chunk)
        run["chunk"] = str(save_chunk)
    labels = data.field_labels(codec)
    out = []
    for i in sorted(final):
        summary, _, mode, fb = final[i]
        out.append({"index": i, "field": labels[i], "mode": MODE_LABELS[mode], "fallback": fb, **summary})
    lat = [f for f in out if f["mode"] == "lattice"]
    keys = ("actual", "model", "h0", "h_ctx", "h_fine", "h_fine_shared", "esc_bits", "tables_and_thresholds_bytes",
            "tables0_bytes", "tables_fine_bytes", "tables_fine_shared_bytes", "bytes", "n_streams")  # fmt: skip
    tot = {k: float(sum(f[k] for f in lat)) for k in keys}
    if lat:
        tot["coder_loss"] = (tot["actual"] - tot["h_ctx"]) / tot["h_ctx"]
        tot["context_gain"] = (tot["h0"] - tot["h_ctx"]) / tot["h0"]
        tot["table_loss"] = (tot["model"] - tot["h_ctx"]) / tot["h_ctx"]
        tot["stream_overhead"] = (tot["actual"] - tot["model"]) / 8 / tot["n_streams"]
        tot["headroom"] = (tot["h_ctx"] - tot["h_fine"]) / tot["h_ctx"]
        tot["headroom_shared"] = (tot["h_ctx"] - tot["h_fine_shared"]) / tot["h_ctx"]
        a = tot["h_ctx"] / 8 + tot["tables_and_thresholds_bytes"]
        tot["headroom_net"] = (a - tot["h_fine"] / 8 - tot["tables_fine_bytes"]) / a
        tot["headroom_shared_net"] = (a - tot["h_fine_shared"] / 8 - tot["tables_fine_shared_bytes"]) / a
        tot["escape_share"] = sum(f["parts"]["escapes"] for f in lat) / tot["bytes"]
    profile = []
    for s, st in enumerate(codec.structure.steps):
        bits = sum(f["steps"][s]["model"] + f["steps"][s]["esc_bits"] for f in lat)
        profile.append(
            {"step": s, "name": st.name, "kind": st.kind, "n": st.n,
             "bits_per_target": bits / (st.n * len(lat)) if lat and st.n else 0.0,
             "share": bits / max(1.0, sum(f["model"] + f["esc_bits"] for f in lat))}
        )  # fmt: skip
    run["fine"] = fine
    return {"run": run, "chunk_bytes": len(chunk), "fields": out, "totals": tot, "profile": profile}


# ------------------------------------------------------------------------------------------- explain / diff
def explain_chunk(buf: bytes, codec: Any) -> list[dict[str, Any]]:
    """Per field, why it is stored as it is (see the module docstring)."""
    acc = account_chunk(buf, codec)
    labels = [f["field"] for f in acc["fields"]]
    idx = data.field_index(codec)
    plan = codec.plan
    opt = codec.options
    out = []
    for f in acc["fields"]:
        i = f["index"]
        name = idx.name(i) if i < idx.per * acc["n_dates"] else None
        var = codec.variables[idx.variable(i)] if name else None
        why: list[str] = []
        d = {"index": i, "field": f["field"], "mode": f["mode"], "bytes": f["bytes"], "why": why}
        if f["fallback"]:
            why.append(f"lossy variable stored losslessly: {FALLBACK_TEXT[f['fallback']]} (fallback {f['fallback']})")
        if f["mode"] == "lossless" and var is not None and not var.lossy:
            if name in plan.recipes:
                why.append("forcing recipe planned, stored losslessly (no valid date, or the recipe missed its bound)")
            elif name in codec.static_fields:
                why.append("static field stored losslessly (statics off, or it differed from the captured one)")
            else:
                why.append("no bound: lossless variable")
        if f["mode"] in ("constant", "static_ref", "recipe"):
            why.append({"constant": "every value equal", "static_ref": "static field (reference)",
                        "recipe": "analytic forcing"}[f["mode"]])  # fmt: skip
        if f["mode"] == "lattice":
            cfg_m = int(opt["recon_bits"])
            if f["recon_bits"] == 0 and cfg_m:
                why.append(
                    "plain rule (recon_bits 0, configured "
                    f"{cfg_m}): more than 1% exact zeros (the zero rule)" + (" or the rdoq path" if opt["rdoq"] else "")
                )
            planned = list(plan.cross.get(name, []))
            used = [labels[j] if j < len(labels) else f"#{j}" for j in f["cross"]]
            used_names = [idx.name(j) for j in f["cross"] if j < len(labels)]
            absent = [s for s in planned if s not in used_names]
            if absent:
                why.append("planned cross sources not coded as LATTICE / STATIC_REF (no detail): " + ", ".join(absent))
            if "steps" in f:
                nx = len(f["cross"])
                gcc = [s["cc"] for s in f["steps"] if s["kind"] == "gather"]
                cc = np.array(gcc, np.int64).reshape(len(gcc), nx)
                zero = [used[c] for c in range(nx) if not np.any(cc[:, c])]
                nz = [int(np.count_nonzero(cc[:, c])) for c in range(nx)]
                if zero:
                    why.append("cross sources with a zero coefficient in every step (did not pay): " + ", ".join(zero))
                d["cross_steps"] = dict(zip(used, nz, strict=True))
                layout = _table_layout(f["steps"])
                d["n_ctx"] = [s["n_ctx"] for s in f["steps"]]
            else:
                layout = None
            if name in plan.masks and f["mask"] is None:
                why.append("below-ground mask term planned but absent")
            d.update(
                cross=used,
                planned_cross=planned,
                mask=None if f["mask"] is None else labels[f["mask"]],
                escape_type=ESCAPE_TYPES.get(f["et"], f["et"]),
                n_esc_units=f["n_esc_units"],
                recon_bits=f["recon_bits"],
                prob_bits=f["prob_bits"],
                n_tables=f["n_tables"],
                table_layout=layout,
                table_bytes=sum(f["table_bytes"]),
                parts=f["parts"],
            )
        out.append(d)
    return out


def _table_layout(steps: list[dict[str, Any]]) -> str:
    """``none`` (one table per step), ``shared`` (gather steps map their buckets to the same tables) or
    ``per_step``."""
    gather = [s for s in steps if s["kind"] == "gather" and s["n"]]
    if all(s["n_ctx"] == 1 for s in steps):
        return "none"
    maps = {tuple(s["tmap"]) for s in gather if s["n_ctx"] > 1}
    return "shared" if len(maps) == 1 and len(gather) > 1 else "per_step"


def diff_chunks(a: bytes, b: bytes, codec: Any = None) -> dict[str, Any]:
    """Field-by-field comparison of two chunks: bytes and mode per field (sorted by |delta|), totals per part."""
    ca, cb = account_chunk(a, codec), account_chunk(b, codec)
    n = min(ca["n_fields"], cb["n_fields"])
    rows = []
    for fa, fb in zip(ca["fields"][:n], cb["fields"][:n], strict=True):
        r = {"index": fa["index"], "field": fa["field"], "a": fa["bytes"], "b": fb["bytes"],
             "delta": fb["bytes"] - fa["bytes"], "mode_a": fa["mode"], "mode_b": fb["mode"]}  # fmt: skip
        r["rel"] = r["delta"] / fa["bytes"] if fa["bytes"] else 0.0
        for k in ("rans", "escapes", "tables"):
            if fa["mode"] == fb["mode"] == "lattice":
                r[k] = fb["parts"][k] - fa["parts"][k]
        rows.append(r)
    rows.sort(key=lambda r: -abs(r["delta"]))
    parts = sorted(
        set(ca["parts"]) | set(cb["parts"]), key=lambda k: -max(ca["parts"].get(k, 0), cb["parts"].get(k, 0))
    )
    return {
        "a": ca["bytes"],
        "b": cb["bytes"],
        "delta": cb["bytes"] - ca["bytes"],
        "n_fields": [ca["n_fields"], cb["n_fields"]],
        "mode_changes": [r for r in rows if r["mode_a"] != r["mode_b"]],
        "parts": {k: [ca["parts"].get(k, 0), cb["parts"].get(k, 0)] for k in parts},
        "fields": rows,
    }


# ----------------------------------------------------------------------------------------------- output
def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def dump_json(obj: Any) -> None:
    print(json.dumps(_jsonable(obj), indent=1))


def _pct(x: float) -> str:
    return f"{100 * x:6.2f}%"


def print_account(acc: dict[str, Any], top: int, steps: bool) -> None:
    mib = acc["bytes"] / 2**20
    extra = f", chain stages {', '.join(acc['stages'])}" if acc["stages"] else ""
    print(f"{acc['bytes']} bytes ({mib:.2f} MiB), {acc['n_fields']} fields, {acc['n_dates']} dates{extra}")
    print(
        "parts: " + ", ".join(f"{k} {_pct(v).strip()}" for k, v in sorted(acc["share"].items(), key=lambda kv: -kv[1]))
    )
    print(
        "modes: " + ", ".join(f"{k} {v['fields']} fields {v['bytes'] / 2**20:.2f} MiB" for k, v in acc["modes"].items())
    )
    rows = sorted(acc["fields"], key=lambda f: -f["bytes"])
    if top:
        rows = rows[:top]
    bpc = acc["n_cells"] is not None
    print(
        f"{'#':>4} {'field':16s} {'mode':10s} {'fb':>2} {'bytes':>10} {'share':>7}"
        + (f" {'b/cell':>6}" if bpc else "")
        + f" {'rans':>7} {'esc':>7} {'tables':>7} {'steps':>6} {'n_tab':>5} {'ctx':>3} {'pb':>2} {'et':>6} {'m':>1}"
        f" {'n_esc':>8}"
    )
    for f in rows:
        line = f"{f['index']:4d} {f['field']:16s} {f['mode']:10s} {f['fallback']:2d} {f['bytes']:10d} "
        line += _pct(f["bytes"] / acc["bytes"])
        if bpc:
            line += f" {f['bits_per_cell']:6.3f}"
        if f["mode"] == "lattice":
            p, b = f["parts"], f["bytes"]
            line += f" {_pct(p['rans'] / b)} {_pct(p['escapes'] / b)} {_pct(p['tables'] / b)} {p['steps']:6d}"
            line += f" {f['n_tables']:5d} {f.get('n_ctx_max', '-'):>3} {f['prob_bits']:2d}"
            line += f" {ESCAPE_TYPES.get(f['et'], f['et']):>6} {f['recon_bits']:1d} {f['n_esc_units']:8d}"
        print(line)
        if steps and "steps" in f:
            print(f"       {'step':>4} {'kind':6s} {'n':>8} {'ctx':>3} {'rans B':>9} {'esc units':>9} {'ranges':>6}")
            for s in f["steps"]:
                rb = s["rans_bytes"] if len(s["ranges"]) > 1 else "serial"
                print(f"       {s['step']:4d} {s['kind']:6s} {s['n']:8d} {s['n_ctx']:3d} {rb!s:>9} {s['esc_units']:9d}"
                      f" {len(s['ranges']):6d}")  # fmt: skip
            print(f"       serial stream {f['serial_rans_bytes']} bytes")


def print_encode(acc: dict[str, Any], top: int, steps: bool) -> None:
    run, tot, fine = acc["run"], acc["totals"], acc["run"].get("fine", FINE_BUCKETS)
    print(
        f"date {run['date']} (valid date {run['valid_date']}), chunk {acc['chunk_bytes']} bytes, encode "
        f"{run['encode_s']:.1f} s; {len(acc['fields'])} fields recorded"
    )
    if "coder_loss" in tot:
        print(
            f"LATTICE fields {tot['bytes'] / 2**20:.2f} MiB: escapes {_pct(tot['escape_share']).strip()} of the bytes; "
            f"coder loss {_pct(tot['coder_loss']).strip()} of H_ctx (tables {_pct(tot['table_loss']).strip()}, "
            f"stream flushes {tot['stream_overhead']:.0f} B per stream); context gain "
            f"{_pct(tot['context_gain']).strip()} (H0 -> H_ctx); headroom of {fine} buckets: per step "
            f"{_pct(tot['headroom']).strip()} gross / {_pct(tot['headroom_net']).strip()} net of tables, shared "
            f"{_pct(tot['headroom_shared']).strip()} / {_pct(tot['headroom_shared_net']).strip()}"
        )
    print(
        f"{'#':>4} {'field':16s} {'mode':8s} {'bytes':>10} {'b/cell':>6} {'esc':>7} {'tables':>7} {'loss':>7}"
        f" {'tabloss':>7} {'ctxgain':>7} {'netgain':>7} {'hr/step':>7} {'hr/shr':>7} {'shr net':>7} {'n_tab':>5}"
        f" {'ctx':>3}"
    )
    rows = sorted(acc["fields"], key=lambda f: -f["bytes"])
    for f in rows[:top] if top else rows:
        print(
            f"{f['index']:4d} {f['field']:16s} {f['mode']:8s} {f['bytes']:10d} {f['bits_per_cell']:6.3f}"
            f" {_pct(f['escape_share'])} {_pct(f['parts']['tables'] / f['bytes'])} {_pct(f['coder_loss'])}"
            f" {_pct(f['table_loss'])} {_pct(f['context_gain'])} {_pct(f['context_gain_net'])}"
            f" {_pct(f['headroom'])} {_pct(f['headroom_shared'])} {_pct(f['headroom_shared_net'])}"
            f" {f['n_tables']:5d} {f['n_ctx_max']:3d}"
        )
        if steps:
            print(f"       {'step':>4} {'n':>8} {'ctx':>3} {'bits/t':>7} {'H0/t':>7} {'Hctx/t':>7} {'Hfine/t':>7}"
                  f" {'esc':>7} {'mean|q|':>8} {'zero':>6}")  # fmt: skip
            for s in f["steps"]:
                n = max(1, s["n"])
                print(
                    f"       {s['step']:4d} {s['n']:8d} {s['n_ctx']:3d} {s['bits_per_target']:7.3f}"
                    f" {s['h0'] / n:7.3f} {s['h_ctx'] / n:7.3f} {s['h_fine'] / n:7.3f} {s['n_esc']:7d}"
                    f" {s['mean_abs_q']:8.3f} {s['zero_frac']:6.3f}"
                )
    print("bits per target per step over the LATTICE fields (step name, targets, bits / target, share of the bits):")
    for p in acc["profile"]:
        print(f"  {p['step']:3d} {p['name']:24s} {p['n']:8d} {p['bits_per_target']:7.3f} {_pct(p['share'])}")


def print_explain(rows: list[dict[str, Any]], all_fields: bool, verbose: bool) -> None:
    for d in rows:
        if not all_fields and d["mode"] != "lattice" and not d["why"]:
            continue
        line = f"{d['index']:4d} {d['field']:16s} {d['mode']:10s} {d['bytes']:10d}"
        if d["mode"] == "lattice":
            line += (
                f"  m {d['recon_bits']} pb {d['prob_bits']} esc {d['escape_type']}/{d['n_esc_units']} tables "
                f"{d['n_tables']} ({d['table_layout'] or '?'}, {d['table_bytes']} B)"
            )
            if d["cross"]:
                cs = d.get("cross_steps")
                line += "  cross " + ", ".join(f"{c}" + (f" ({cs[c]} steps)" if cs else "") for c in d["cross"])
            if d["mask"]:
                line += f"  mask {d['mask']}"
        print(line)
        for w in d["why"]:
            print(f"       {w}")
        if verbose and d["mode"] == "lattice" and "n_ctx" in d:
            print(f"       n_ctx per step {d['n_ctx']}")


def print_diff(d: dict[str, Any], top: int) -> None:
    print(f"A {d['a']} bytes, B {d['b']} bytes, delta {d['delta']:+d} ({_pct(d['delta'] / d['a']).strip()})")
    if d["n_fields"][0] != d["n_fields"][1]:
        print(f"field counts differ: {d['n_fields']} (compared the first {min(d['n_fields'])})")
    print("parts (A -> B): " + ", ".join(f"{k} {a} -> {b} ({b - a:+d})" for k, (a, b) in d["parts"].items() if a or b))
    for r in d["mode_changes"]:
        print(f"mode change: {r['field']} {r['mode_a']} -> {r['mode_b']}")
    print(f"{'#':>4} {'field':16s} {'A':>10} {'B':>10} {'delta':>9} {'rel':>8} {'rans':>8} {'esc':>7} {'tables':>6}")
    for r in d["fields"][:top] if top else d["fields"]:
        extra = "".join(f" {r[k]:+{w}d}" if k in r else f" {'':>{w}}" for k, w in (("rans", 8), ("escapes", 7),
                                                                                    ("tables", 6)))  # fmt: skip
        modes = "" if r["mode_a"] == r["mode_b"] else f"  {r['mode_a']} -> {r['mode_b']}"
        print(
            f"{r['index']:4d} {r['field']:16s} {r['a']:10d} {r['b']:10d} {r['delta']:+9d} {_pct(r['rel'])}"
            f"{extra}{modes}"
        )


# -------------------------------------------------------------------------------------------- commands
def _config_arg(a: argparse.Namespace, chunk: Path | None) -> Any:
    cfg = a.config if a.config is not None else (config_near(chunk) if chunk is not None else None)
    return codec_of(cfg)


def _account(a: argparse.Namespace) -> int:
    if a.encode is not None:
        cfg, store = a.encode
        if a.date is None:
            raise SystemExit("lab account --encode needs --date")
        if a.threads:
            os.environ["ANEMOI_DATASETS_CODECS_THREADS"] = str(a.threads)
        codec = codec_of(Path(cfg))
        acc = account_encode(
            codec, Path(store), a.date, select_fields(codec, a.fields), a.fine, not a.no_valid_date, a.save_chunk
        )
        acc["environment"] = data.environment()
        dump_json(acc) if a.json else print_encode(acc, a.top, a.steps)
        return 0
    if a.chunk is None:
        raise SystemExit("lab account: give a CHUNK file or --encode CONFIG STORE --date I")
    acc = account_chunk(a.chunk.read_bytes(), _config_arg(a, a.chunk))
    dump_json(acc) if a.json else print_account(acc, a.top, a.steps)
    return 0


def _explain(a: argparse.Namespace) -> int:
    codec = _config_arg(a, a.chunk)
    if codec is None:
        raise SystemExit("lab explain needs --config (or a chunk inside a store)")
    rows = explain_chunk(a.chunk.read_bytes(), codec)
    if a.fields:
        want = {s.strip() for s in a.fields.split(",")}
        rows = [r for r in rows if r["field"] in want]
    dump_json(rows) if a.json else print_explain(rows, a.all, a.verbose)
    return 0


def _diff(a: argparse.Namespace) -> int:
    d = diff_chunks(a.chunk_a.read_bytes(), a.chunk_b.read_bytes(), _config_arg(a, a.chunk_a))
    dump_json(d) if a.json else print_diff(d, a.top)
    return 0


def add_parsers(sub: argparse._SubParsersAction) -> None:
    """Register ``account``, ``explain`` and ``diff`` under ``lab``."""
    p = sub.add_parser("account", help="where the bits of a chunk go (chunk file, or --encode a store date)")
    p.add_argument("chunk", type=Path, nargs="?", help="chunk file (bare lattice container or chained)")
    p.add_argument("--config", type=Path, help="store or JSON configuration (names, per-step parts)")
    p.add_argument("--encode", nargs=2, metavar=("CONFIG", "STORE"), help="re-encode a date of STORE with recording")
    p.add_argument("--date", type=int, help="date index (with --encode)")
    p.add_argument("--fields", help="comma-separated fields to record (with --encode; default: all)")
    p.add_argument("--fine", type=int, default=FINE_BUCKETS, help="quantile buckets of the finer context")
    p.add_argument("--no-valid-date", action="store_true", help="encode without the valid date (forcings lossless)")
    p.add_argument("--save-chunk", type=Path, help="write the encoded chunk (with --encode)")
    p.add_argument("--threads", type=int, help="encoder threads (ANEMOI_DATASETS_CODECS_THREADS)")
    p.add_argument("--top", type=int, default=25, help="fields to print, largest first (0: all)")
    p.add_argument("--steps", action="store_true", help="per step lines for the printed fields")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_account)
    p = sub.add_parser("explain", help="why each field of a chunk is stored as it is")
    p.add_argument("chunk", type=Path)
    p.add_argument("--config", type=Path, help="store or JSON configuration (default: the chunk's store)")
    p.add_argument("--fields", help="comma-separated fields (default: all)")
    p.add_argument("--all", action="store_true", help="also the fields with nothing to explain")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_explain)
    p = sub.add_parser("diff", help="per-field bytes and mode deltas between two chunks")
    p.add_argument("chunk_a", type=Path)
    p.add_argument("chunk_b", type=Path)
    p.add_argument("--config", type=Path, help="store or JSON configuration, for names and per-step parts")
    p.add_argument("--top", type=int, default=25, help="fields to print, largest |delta| first (0: all)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_diff)
