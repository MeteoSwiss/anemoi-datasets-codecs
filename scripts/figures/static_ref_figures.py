"""The figures of docs/codecs/static_ref.md, computed from the code and the synthetic test data.

    python scripts/figures/static_ref_figures.py [--out DIR] [--only NAME ...]

Writes plain SVG (real text, generic fonts, an opaque light background) into docs/codecs/figures/static_ref/, with
the helpers and the palette of ``icon_lattice_figures.py`` (the same colour for the same meaning: green static /
exact, blue the fields the next codec codes):

* ``stage.svg``: the stage in the chain ``[field_terms, static_ref, icon_lattice_bytes]`` and what it does to one
  chunk: the static fields leave as slots of its header (and files of the sidecar), the other fields go on in order,
  the header is pushed on the envelope.
* ``lookup.svg``: the reference rule of the encoder (``stage.py``, ``statics.StaticCapture`` and
  ``statics.ensure_durable``) and the lookup of the decoder (``statics.lookup``, ``sidecar.search_dirs``).
* ``header.svg``: header version 1 (``header.py``) with the bytes of the real headers of the chunk, by reference and
  with ``inline: true``, to scale.
* ``decode.svg``: the decoder's one write of the whole chunk into the ``out`` array of the next envelope
  (``stage.py``, ``zarr3.FieldStageBytesCodec``).

The chunk is the synthetic test chunk of tests/conftest.py on 2 dates and 2 members (the other fields shifted per
date and member, as tests/test_static_ref.py does), written through zarr with the chain that ``physics.expand``
writes for its variables, into a temporary store whose sidecar is the capture target. Every number shown is read
from that store, its chunk and its sidecar.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from icon_lattice_figures import (  # noqa: E402
    BG,
    DECODED,
    DECODED_LIGHT,
    FAINT,
    INK,
    LINE,
    MUTED,
    STATIC,
    STATIC_LIGHT,
    Svg,
    _f,
    fmt_int,
    mix,
    panel_label,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "codecs" / "figures" / "static_ref"
N_DATES, N_ENS = 2, 2

# the three slot kinds (all exact, so all green): referenced, inline (values in the header), repeat (no data)
KIND_FILL = {0: STATIC_LIGHT, 1: STATIC, 2: BG}
KIND_TEXT = {0: INK, 1: BG, 2: INK}
KIND_SHORT = {0: "REF", 1: "INL", 2: "REP"}
STATIC_MID = mix(STATIC, "#ffffff", 0.45)


# ------------------------------------------------------------------------------------------------ data
class Data:
    """The synthetic chunk written through the real chain into a temporary store (the capture target)."""

    def __init__(self, tmp: Path) -> None:
        sys.path.insert(0, str(REPO / "tests"))
        import conftest  # noqa: E402
        import zarr

        from anemoi_datasets_codecs import chain, envelope, physics, sidecar
        from anemoi_datasets_codecs.static_ref import StaticRef
        from anemoi_datasets_codecs.static_ref import header as H

        lat = conftest.make_lattice()
        chunk = conftest.make_chunk(lat)
        names, static = list(conftest.NAMES), tuple(conftest.STATIC)
        x = np.repeat(np.stack([chunk] * N_DATES)[:, :, None, :], N_ENS, axis=2)
        for d in range(N_DATES):  # as tests/test_static_ref.py: only the statics repeat
            for e in range(N_ENS):
                for i, n in enumerate(names):
                    if n not in static:
                        x[d, i, e] += np.float32(0.01 * (d + 2 * e))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            entries = physics.expand(names, None, conftest.make_config(lat[0]))
            for e in entries:
                if e["name"] == "icon_lattice_bytes":
                    e["configuration"]["n_ensembles"] = N_ENS
            codecs = [chain.codec_class(e["name"])(**e["configuration"]) for e in entries]
            store = tmp / "chain.zarr"
            g = zarr.open_group(str(store), mode="w")
            self.target = sidecar.register_store(store)
            arr = g.create_array("data", shape=x.shape, chunks=x.shape, dtype="float32", compressors=codecs)
            arr[:] = x
            sidecar.unregister_store(store)
            self.inline_header = StaticRef(names, list(static), inline=True).encode(x)[1]
        self.chain = [e["name"] for e in entries]
        assert self.chain == ["field_terms", "static_ref", "icon_lattice_bytes"], self.chain
        self.cfg = {e["name"]: e["configuration"] for e in entries}
        self.stage = StaticRef(**self.cfg["static_ref"])
        assert self.stage.static_fields == static
        self.raw = (store / "data" / "c" / "0" / "0" / "0" / "0").read_bytes()
        env = envelope.parse(self.raw)
        assert env is not None and env.kind == envelope.KIND_BYTES
        assert [h for h, _ in env.headers] == ["field_terms", "static_ref"]
        self.env = env
        self.header = env.headers[-1][1]
        self.ft_header = env.headers[0][1]
        self.n_static, self.slots = H.parse(self.header)
        _, self.inline_slots = H.parse(self.inline_header)
        first = list(dict.fromkeys(s.sha for s in self.slots))  # the sidecar files in the order of first use
        on_disk = {p.name.split(".")[0]: p.name for p in (self.target / "statics").iterdir()}
        assert set(on_disk) == set(first), "every static array is a file of the store's sidecar"
        self.files = [on_disk[sha] for sha in first]
        self.x_shape = x.shape
        self.names = names
        self.kept = list(self.stage.variables_out)
        self.s_idx = [names.index(n) for n in static]
        self.k_idx = [names.index(n) for n in self.kept]
        from anemoi_datasets_codecs.static_ref.stage import _runs

        self.runs = _runs(self.k_idx)
        self.y_shape = (x.shape[0], len(self.kept), *x.shape[2:])

    def slot_name(self, k: int) -> tuple[int, str, int]:
        """(date, static field, member) of slot k (C order of x[:, static_fields, :, :])."""
        d, rest = divmod(k, self.n_static * N_ENS)
        i, e = divmod(rest, N_ENS)
        return d, self.stage.static_fields[i], e


def kinds_of(slots) -> dict[int, int]:
    out = {0: 0, 1: 0, 2: 0}
    for s in slots:
        out[s.kind] += 1
    return out


def box(svg: Svg, x, y, w, h, title, rows=(), fill=FAINT, stroke=LINE, sw=1.4, size=12.5, title_size=13.5):
    svg.rect(x, y, w, h, fill=fill, stroke=stroke, sw=sw, rx=6)
    ty = y + 20 if rows else y + h / 2 + 5
    svg.text(x + w / 2, ty, title, size=title_size, anchor="middle", weight="bold")
    for i, s in enumerate(rows):
        svg.text(x + w / 2, y + 38 + 16 * i, s, size=size, anchor="middle")


def field_rows(svg: Svg, x, y, names, statics, row_h=16, col_w=24, gap=8, name_w=112, label=True):
    """Rows of fields: name, then one bar per (date, member). Returns the y of each row's middle."""
    mids = []
    for r, n in enumerate(names):
        yy = y + r * row_h
        st = n in statics
        if label:
            svg.text(x + name_w - 6, yy + row_h - 4, n, size=12, anchor="end", mono=True, fill=INK)
        for d in range(N_DATES):
            for e in range(N_ENS):
                cx = x + name_w + d * (N_ENS * col_w + gap) + e * col_w
                svg.rect(
                    cx + 1,
                    yy + 2,
                    col_w - 2,
                    row_h - 4,
                    fill=STATIC_MID if st else mix(DECODED, "#ffffff", 0.45),
                    stroke="none",
                )
        mids.append(yy + row_h / 2)
    return mids


def col_heads(svg: Svg, x, y, col_w=24, gap=8, name_w=112):
    for d in range(N_DATES):
        cx = x + name_w + d * (N_ENS * col_w + gap)
        svg.text(cx + N_ENS * col_w / 2, y - 16, f"date {d}", size=12, anchor="middle", fill=MUTED)
        for e in range(N_ENS):
            svg.text(cx + e * col_w + col_w / 2, y - 2, f"m{e}", size=12, anchor="middle", fill=MUTED)


def bracket(svg: Svg, x, y0, y1, color, sw=1.4):
    svg.path(f"M{_f(x)},{_f(y0)} L{_f(x + 6)},{_f(y0)} L{_f(x + 6)},{_f(y1)} L{_f(x)},{_f(y1)}", stroke=color, sw=sw)


# ================================================================================================ figure 1
def fig_stage(data: Data, out: Path) -> Path:
    svg = Svg(900, 806, "What the static_ref stage does to a chunk")
    nd, nf, ne, nc = data.x_shape
    nk = len(data.kept)
    ns = data.n_static
    ft_b, sr_b = len(data.ft_header), len(data.header)
    k = kinds_of(data.slots)

    # (a) the chain, encode order
    panel_label(svg, 20, 28, "(a)", "the stage in the chain (encode order; decode runs right to left)")
    bw, bh, gap, x0, y0 = 196, 104, 22, 20, 50
    boxes = [
        ("bytes", ["the float32 chunk,", "little endian (bare)", f"({nd}, {nf}, {ne}, {fmt_int(nc)})"], FAINT, LINE),
        (
            "field_terms",
            ["residuals of", " and ".join(data.cfg["field_terms"]["terms"]), f"pushes {ft_b} B header"],
            FAINT,
            LINE,
        ),
        (
            "static_ref",
            [f"drops the {ns} statics", f"pushes {sr_b} B header", f"emits ({nd}, {nk}, {ne}, ·)"],
            STATIC_LIGHT,
            STATIC,
        ),
        (
            "icon_lattice_bytes",
            ["chain: true; codes the", f"{nk} fields, headers", "passed on unread"],
            DECODED_LIGHT,
            DECODED,
        ),
    ]
    for i, (t, rows, fill, stroke) in enumerate(boxes):
        x = x0 + i * (bw + gap)
        box(svg, x, y0, bw, bh, t, rows, fill, stroke, sw=2.4 if t == "static_ref" else 1.4)
        if i:
            svg.arrow(x - gap, y0 + bh / 2, x, y0 + bh / 2, color=INK, shrink0=1, shrink1=1)
    svg.text(
        x0,
        y0 + bh + 22,
        f"stored chunk: an FCHN envelope (shape ({nd}, {nk}, {ne}, {fmt_int(nc)}), stack [field_terms, static_ref]) "
        f"around the LATC container: {fmt_int(len(data.raw))} bytes",
        size=12.5,
        fill=MUTED,
    )

    # (b) one chunk
    yb = 222
    panel_label(
        svg,
        20,
        yb,
        "(b)",
        f"one chunk: {nd} dates × {nf} fields × {ne} members × {fmt_int(nc)} cells (synthetic test data)",
    )
    yr = yb + 52
    xl = 20
    col_heads(svg, xl, yr)
    svg.text(xl, yr - 16, "x, the chunk in", size=12.5, weight="bold")
    mids = field_rows(svg, xl, yr, data.names, set(data.stage.static_fields))
    xr_end = xl + 112 + N_DATES * N_ENS * 24 + 8
    # brackets: the runs of kept fields, the statics
    s0, s1 = min(data.s_idx), max(data.s_idx)
    assert data.s_idx == list(range(s0, s1 + 1))
    for a, b_, _ in data.runs:
        bracket(svg, xr_end + 4, mids[a] - 7, mids[b_ - 1] + 7, DECODED)
    bracket(svg, xr_end + 4, mids[s0] - 7, mids[s1] + 7, STATIC)

    # the envelope it emits, on the right: prefix, shape, the stack, the payload y
    ex, ew = 596, 284
    ey = yr - 34
    parts = [
        ("FCHN, version 1, kind 0, ndim 4", 24, FAINT, LINE),
        (f"shape ({nd}, {nk}, {ne}, {fmt_int(nc)})", 26, FAINT, LINE),
        (f"header field_terms: {ft_b} B", 26, FAINT, LINE),
        (f"header static_ref: {sr_b} B  (pushed)", 30, STATIC_LIGHT, STATIC),
    ]
    svg.text(ex, ey - 8, "envelope out (to icon_lattice_bytes)", size=12.5, weight="bold")
    yy = ey
    hdr_mid = 0.0
    for t, h, fill, stroke in parts:
        svg.rect(ex, yy, ew, h, fill=fill, stroke=stroke, sw=1.4 if stroke == STATIC else 1)
        svg.text(ex + 10, yy + h / 2 + 4.5, t, size=12.5, weight="bold" if stroke == STATIC else "normal")
        if stroke == STATIC:
            hdr_mid = yy + h / 2
        yy += h
    py = yy
    ph = 12 + nk * 16 + 30
    svg.rect(ex, py, ew, ph, fill=BG, stroke=LINE, sw=1)
    ymids = field_rows(svg, ex + 6, py + 10, data.kept, set(), name_w=128)
    svg.text(ex + 10, py + ph - 10, f"payload y: float32, {fmt_int(4 * nd * nk * ne * nc)} B", size=12.5, fill=MUTED)
    # statics: to the header (one slot per date, static, member) and, on first sight, into the sidecar
    sm = (mids[s0] + mids[s1]) / 2
    svg.arrow(xr_end + 12, sm, ex - 2, hdr_mid, color=STATIC, sw=1.6, shrink1=2)
    svg.lines(
        (xr_end + ex) / 2 + 26,
        sm - 4,
        [f"{nd * ns * ne} slots, one per", "(date, static, member):", "sha256 of its float32", "little-endian bytes"],
        size=12.5,
        anchor="middle",
        fill=STATIC,
    )

    # kept fields: one arrow per run, into their rows of y
    for a, b_, ya in data.runs:
        src = (mids[a] + mids[b_ - 1]) / 2
        dst = (ymids[ya] + ymids[ya + b_ - a - 1]) / 2
        svg.line(xr_end + 12, src, ex - 6, dst + (src - dst) * 4 / (ex - xr_end - 12), stroke=BG, sw=6)
        svg.arrow(xr_end + 12, src, ex - 2, dst, color=DECODED, sw=1.6, shrink1=2)
    t1 = (mids[data.runs[0][0]] + mids[data.runs[0][1] - 1]) / 2
    svg.text(
        (xr_end + ex) / 2 + 10, t1 - 30, f"the other {nk} fields, in order:", size=12.5, anchor="middle", fill=DECODED
    )
    svg.text(
        (xr_end + ex) / 2 + 10, t1 - 15, "y = x[:, kept] (variables_out)", size=12.5, anchor="middle", fill=DECODED
    )
    # the header's slots and the sidecar, below
    yh = max(mids[-1], py + ph) + 44
    svg.text(
        20,
        yh,
        f"the static_ref header: {nd} × {ns} × {ne} = {nd * ns * ne} slots, C order of x[:, static_fields, :, :] "
        f"({k[0]} REFERENCE, {k[2]} REPEAT, {k[1]} INLINE)",
        size=12.5,
        weight="bold",
    )
    cw, ch_ = 41.0, 26
    gx = 20
    slot_y = yh + 30
    for kk, s in enumerate(data.slots):
        d = data.slot_name(kk)[0]
        x = gx + kk * cw + (d * 12)
        svg.rect(
            x, slot_y, cw - 3, ch_, fill=KIND_FILL[s.kind], stroke=STATIC, sw=1, dash="3 2" if s.kind == 2 else None
        )
        svg.text(
            x + (cw - 3) / 2,
            slot_y + 17,
            KIND_SHORT[s.kind],
            size=12,
            anchor="middle",
            fill=KIND_TEXT[s.kind],
            mono=True,
        )
    for d in range(nd):
        xd = gx + d * (ns * ne * cw + 12)
        svg.text(xd, slot_y - 8, f"date {d}: each static on member 0, member 1", size=12, fill=MUTED)
    # the sidecar files
    fy = slot_y + ch_ + 54
    fw, fh = 166, 40
    shas = [f.split(".")[0] for f in data.files]
    for i, f in enumerate(data.files):
        x = 20 + i * (fw + 6)
        name = data.slot_name(next(kk for kk, s in enumerate(data.slots) if s.sha == shas[i]))[1]
        svg.rect(x, fy, fw, fh, fill=STATIC_LIGHT, stroke=STATIC, sw=1)
        svg.text(x + fw / 2, fy + 16, name, size=12, anchor="middle", fill=MUTED, mono=True)
        svg.text(x + fw / 2, fy + 32, f[:12] + "….f4.blosc", size=12, anchor="middle", mono=True)
    svg.text(
        20,
        fy + fh + 20,
        f"the store's sidecar <store>/.codecs/statics/: the {len(data.files)} arrays, captured on first sight "
        "(the store was the capture target)",
        size=12.5,
        weight="bold",
    )
    # REFERENCE slots point to their file
    for kk, s in enumerate(data.slots):
        if s.kind != 0:
            continue
        d, _, _ = data.slot_name(kk)
        sx = gx + kk * cw + d * 12 + (cw - 3) / 2
        j = shas.index(s.sha)
        svg.arrow(sx, slot_y + ch_, 20 + j * (fw + 6) + fw / 2, fy, color=STATIC, sw=1.2, shrink0=1, shrink1=2)
    yl = svg.lines(
        20,
        fy + fh + 48,
        [
            "REF: the sha256 of a sidecar file; REP: the sha256 of an earlier slot of the chunk, whose values it "
            "reuses; INL (none here):",
            f"the values in the header. Whatever the kinds, the stage emits {nf} − {ns} = {nk} fields.",
        ],
        size=12.5,
        gap=18,
    )
    svg.h = round(yl)
    return svg.save(out / "stage.svg")


# ================================================================================================ figure 2
def kind_pill(svg: Svg, x, y, w, h, kind: int, note: str = ""):
    svg.rect(x, y, w, h, fill=KIND_FILL[kind], stroke=STATIC, sw=1.4, rx=h / 2, dash="4 3" if kind == 2 else None)
    label = {0: "REFERENCE", 1: "INLINE", 2: "REPEAT"}[kind]
    svg.text(x + w / 2, y + h / 2 + 4.5, label, size=12.5, anchor="middle", weight="bold", fill=KIND_TEXT[kind])
    if note:
        svg.text(x + w + 8, y + h / 2 + 4.5, note, size=12, fill=MUTED)


def question(svg: Svg, x, y, w, h, rows, fill=FAINT, stroke=LINE):
    svg.rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.2, rx=6)
    y0 = y + h / 2 + 4.5 - (len(rows) - 1) * 8
    for i, r in enumerate(rows):
        svg.text(x + 12, y0 + 16 * i, r, size=12.5)


def yes_no(svg: Svg, x, y, label):
    svg.text(x, y, label, size=12, fill=MUTED, italic=True)


def fig_lookup(data: Data, out: Path) -> Path:
    from anemoi_datasets_codecs import statics

    svg = Svg(900, 760, "How static_ref finds and stores a static array")
    panel_label(
        svg, 20, 28, "(a)", "encoder, for each slot in C order (stage.py, statics.StaticCapture, ensure_durable)"
    )
    qx, qw, qh = 20, 384, 44
    mx, mw = 436, 196
    ox, ow, oh = 664, 108, 28

    def pill_at(ymid, kind, note=""):
        kind_pill(svg, ox, ymid - oh / 2, ow, oh, kind, note)

    def right(x0, ymid, label, kind=None, note=""):
        svg.arrow(x0, ymid, ox, ymid, color=INK, sw=1.2, shrink1=1)
        yes_no(svg, x0 + 8, ymid - 6, label)
        if kind is not None:
            pill_at(ymid, kind, note)

    def down_right(x, y0, ymid, label, kind, note=""):
        """From (x, y0) down to ymid, then right to a pill."""
        svg.path(f"M{_f(x)},{_f(y0)} L{_f(x)},{_f(ymid)} L{_f(ox - 1)},{_f(ymid)}", stroke=INK, sw=1.2, marker=INK)
        yes_no(svg, x + 8, ymid - 6, label)
        pill_at(ymid, kind, note)

    mv = statics.MAX_VERSIONS
    rows = [
        (
            48,
            ["slot (date d, static field i, member e): v = x[d, i, e];", "sha = sha256 of v as little-endian float32"],
        ),
        (120, ["is sha the sha of an earlier slot of this chunk?"]),
        (184, ["inline: true in the configuration?"]),
        (248, [f"has this codec already referenced {mv} arrays", "(MAX_VERSIONS) for this name, none with this sha?"]),
        (
            328,
            ["is there a store target? the first writable", "registered store or $ANEMOI_DATASETS_CODECS_PATH entry"],
        ),
        (
            452,
            ["no target: is sha on the search path? a file in", "3 to 6 of (b), or found there before by this process"],
        ),
        (532, ["capture: write statics/<sha>.f4.blosc into the", "cache directory (unless the cache is off)"]),
    ]
    for i, (y, txt) in enumerate(rows):
        question(svg, qx, y, qw, qh, txt, fill=BG if i == 0 else FAINT, stroke=INK if i == 0 else LINE)
        if i:
            y0 = rows[i - 1][0] + qh
            svg.arrow(qx + 40, y0, qx + 40, y, color=INK, sw=1.2, shrink1=1)
            if i > 1:
                yes_no(svg, qx + 46, (y0 + y) / 2 + 4, "no")
    mid = [y + qh / 2 for y, _ in rows]
    right(qx + qw, mid[1], "yes", 2, "arg = that slot")
    right(qx + qw, mid[2], "yes", 1, "arg = m bytes")
    right(qx + qw, mid[3], "yes", 1, "+ warning")
    # a store target: the file must be in that store's sidecar
    svg.arrow(qx + qw, mid[4], mx, mid[4], color=INK, sw=1.2, shrink1=1)
    yes_no(svg, qx + qw + 6, mid[4] - 6, "yes")
    question(
        svg,
        mx,
        rows[4][0] - 8,
        mw,
        qh + 16,
        ["the file in the target's", "sidecar? if not, write it", "there (not in the cache)"],
    )
    right(mx + mw, mid[4], "", 0)
    down_right(mx + mw / 2, rows[4][0] + qh + 8, mid[4] + 64, "write fails", 1, "+ warning")
    right(qx + qw, mid[5], "yes", 0)
    right(qx + qw, mid[6], "written", 0, "captured")
    down_right(qx + qw / 2, rows[6][0] + qh, mid[6] + 54, "cache off, or the write fails", 1, "+ warning")
    yf = mid[6] + 54
    svg.lines(
        20,
        yf + 40,
        [
            "A REFERENCE is written only once its file is durably stored; its sha then counts as one of the name's "
            "arrays. REPEAT and",
            "INLINE slots never touch a sidecar. Whatever the kind, a static never goes on as a field, and decodes "
            "the same.",
        ],
        size=12.5,
        gap=18,
    )

    # (b) the decoder
    yb = yf + 106
    panel_label(
        svg, 20, yb, "(b)", "decoder, for a REFERENCE or INLINE slot (statics.lookup): the first source that has it"
    )
    steps = [
        ("1 process", ["arrays already", "read, by sha"]),
        ("2 inline", ["INLINE slots:", "the header's bytes"]),
        ("3 registered", ["register_store,", "latest first"]),
        ("4 path", ["the entries of", "the path variable"]),
        ("5 cache", ["the user cache", "directory"]),
        ("6 package", ["directories", "bundled"]),
    ]
    bw, gap, by, bh = 126, 20, yb + 22, 66
    for i, (t, r) in enumerate(steps):
        x = 20 + i * (bw + gap)
        box(svg, x, by, bw, bh, t, r, fill=STATIC_LIGHT if i == 1 else FAINT, stroke=STATIC if i == 1 else LINE, sw=1.2)
        if i:
            svg.arrow(x - gap, by + bh / 2, x, by + bh / 2, color=INK, sw=1.2, shrink1=1)
    x6 = 20 + 5 * (bw + gap)
    my = by + bh + 34
    svg.arrow(x6 + bw / 2, by + bh, x6 + bw / 2, my, color=INK, sw=1.2, shrink1=1)
    yes_no(svg, x6 + bw / 2 + 6, by + bh + 21, "none")
    svg.rect(x6 - 40, my, bw + 40, 30, fill=BG, stroke=INK, sw=1.4, rx=6)
    svg.text(x6 - 40 + (bw + 40) / 2, my + 20, "MissingReference", size=12.5, anchor="middle", weight="bold")
    yb2 = by + bh + 26
    svg.lines(
        20,
        yb2,
        [
            "3 to 6: sidecar directories (statics/<sha>.f4.blosc), in the order of sidecar.search_dirs;",
            "4 is $ANEMOI_DATASETS_CODECS_PATH, 5 $ANEMOI_DATASETS_CODECS_CACHE.",
            "Bytes read (2 to 6) are verified against the sha256, then kept in",
            "the process (1); every array's size is checked against the cells.",
            "A REPEAT slot takes the array of its earlier slot: no lookup.",
            "MissingReference names the sha and the directories searched.",
        ],
        size=12.5,
        gap=18,
    )
    svg.h = round(yb2 + 6 * 18 - 4)
    return svg.save(out / "lookup.svg")


# ================================================================================================ figure 3
def byte_fields(svg: Svg, x0, y0, scale, fields, raw: bytes, h=34, hex_row=True):
    """Fields (offset, size, value, label, type) to scale, with the real bytes under each field."""
    end = 0
    for off, nb, val, lbl, typ in fields:
        x = x0 + off * scale
        svg.rect(x, y0, nb * scale, h, fill=FAINT, stroke=INK, sw=1)
        svg.text(x + nb * scale / 2, y0 + h / 2 + 4.5, val, size=12.5, anchor="middle", mono=True)
        svg.text(x, y0 - 6, str(off), size=12, fill=MUTED)
        svg.text(x + nb * scale / 2, y0 + h + 18, lbl, size=12, anchor="middle")
        svg.text(x + nb * scale / 2, y0 + h + 33, typ, size=12, anchor="middle", fill=MUTED)
        if hex_row:
            hx = raw[off : off + nb].hex(" ")
            if len(hx) * 7.3 > nb * scale - 4:
                hx = raw[off : off + 4].hex(" ") + " …"
            svg.text(x + nb * scale / 2, y0 + h + 50, hx, size=12, anchor="middle", mono=True, fill=MUTED)
        end = off + nb
    svg.text(x0 + end * scale, y0 - 6, str(end), size=12, fill=MUTED, anchor="end")


def fig_header(data: Data, out: Path) -> Path:
    from anemoi_datasets_codecs.static_ref import header as H

    hdr, slots, inl = data.header, data.slots, data.inline_slots
    ns, n = data.n_static, len(slots)
    assert len(hdr) == 16 + 40 * n and all(not s.data for s in slots)
    svg = Svg(900, 900, "The static_ref header, version 1")

    # (a) the fixed part
    panel_label(svg, 20, 28, "(a)", "the fixed part: 16 bytes (integers little endian), and its bytes in this chunk")
    B = 53.0
    x0, y0 = 24, 58
    byte_fields(
        svg,
        x0,
        y0,
        B,
        [
            (0, 1, str(hdr[0]), "version", "u8"),
            (1, 1, "0", "", "u8"),
            (2, 2, "0", "reserved", "u16"),
            (4, 4, str(ns), "n_static", "u32"),
            (8, 4, str(n), "n_slots", "u32"),
            (12, 4, "0", "reserved", "u32"),
        ],
        hdr,
    )
    svg.text(
        x0,
        y0 + 104,
        f"n_slots = dates × n_static × members = {data.x_shape[0]} × {ns} × {data.x_shape[2]}; then the n_slots "
        "records, back to back, no padding",
        size=12.5,
    )

    # (b) one record
    yb = y0 + 146
    k1 = next(k for k, s in enumerate(slots) if s.kind == H.REPEAT)
    off1 = 16 + 40 * k1
    rec = hdr[off1 : off1 + 40]
    panel_label(svg, 20, yb, "(b)", f"a slot record: 40 bytes (slot {k1} of this header), then m bytes for INLINE")
    Bb = 21.0
    yr = yb + 32
    cells = [
        (0, 1, str(rec[0])),
        (1, 1, "0"),
        (2, 2, "0"),
        (4, 4, str(slots[k1].source)),
        (8, 32, slots[k1].sha[:32] + "…"),
    ]
    for off, nb, val in cells:
        svg.rect(x0 + off * Bb, yr, nb * Bb, 34, fill=FAINT, stroke=INK, sw=1)
        svg.text(x0 + (off + nb / 2) * Bb, yr + 21.5, val, size=12.5, anchor="middle", mono=True)
        svg.text(x0 + off * Bb, yr - 6, str(off), size=12, fill=MUTED)
    svg.text(x0 + 40 * Bb, yr - 6, "40", size=12, fill=MUTED, anchor="end")
    svg.text(x0, yr + 52, "kind", size=12)
    svg.text(x0 + 6 * Bb, yr + 52, "arg", size=12, anchor="middle")
    svg.text(x0 + 24 * Bb, yr + 52, "sha256 of the slot's float32 little-endian bytes", size=12, anchor="middle")
    svg.text(x0, yr + 70, "u8 kind, u8 0, u16 0, u32 arg, 32 bytes sha256", size=12, fill=MUTED)
    svg.text(
        x0 + 24 * Bb,
        yr + 70,
        "bytes: " + " | ".join(rec[o : o + nb].hex(" ") for o, nb, _ in cells[:4]) + " | " + rec[8:12].hex(" ") + " …",
        size=12,
        anchor="middle",
        mono=True,
        fill=MUTED,
    )
    d, name, e = data.slot_name(k1)
    svg.lines(
        x0,
        yr + 96,
        [
            f"slot {k1} = ({name}, date {d}, member {e}): kind 2 REPEAT, arg = {slots[k1].source}, the earlier slot "
            "with the same sha256.",
            "kind 0 REFERENCE: arg 0, the values are the sidecar file statics/<sha256>.f4.blosc.",
            "kind 1 INLINE: arg = m > 0, and m bytes follow the record: the numcodecs Blosc frame of the values "
            "(zstd 9, bit shuffle).",
        ],
        size=12.5,
        gap=18,
    )

    # (c) the slots of this chunk, by reference and with inline: true
    yc = yr + 176
    panel_label(svg, 20, yc, "(c)", f"the {n} slots of the chunk, C order of x[:, static_fields, :, :]")
    cw, ch_ = 80.0, 40
    gx = x0 + 50
    ycols = yc + 26
    for i, nm in enumerate(data.stage.static_fields):
        xx = gx + i * 2 * cw
        svg.text(xx + cw, ycols + 12, nm, size=12, anchor="middle", mono=True)
        for e in range(data.x_shape[2]):
            svg.text(xx + e * cw + cw / 2, ycols + 28, f"m{e}", size=12, anchor="middle", fill=MUTED)

    def grid(y, sl, title):
        svg.text(x0, y - 10, title, size=12.5, weight="bold")
        for kk, s in enumerate(sl):
            dd, rest = divmod(kk, ns * data.x_shape[2])
            x = gx + rest * cw
            yy = y + dd * (ch_ + 4)
            svg.rect(
                x + 1, yy, cw - 4, ch_, fill=KIND_FILL[s.kind], stroke=STATIC, sw=1, dash="3 2" if s.kind == 2 else None
            )
            top = f"{kk} {KIND_SHORT[s.kind]}"
            bot = {0: s.sha[:8], 1: f"{fmt_int(len(s.data))} B", 2: f"arg = {s.source}"}[s.kind]
            svg.text(x + cw / 2 - 1, yy + 16, top, size=12, anchor="middle", mono=True, fill=KIND_TEXT[s.kind])
            svg.text(x + cw / 2 - 1, yy + 32, bot, size=12, anchor="middle", mono=True, fill=KIND_TEXT[s.kind])
        for dd in range(data.x_shape[0]):
            svg.text(gx - 8, y + dd * (ch_ + 4) + ch_ / 2 + 4.5, f"date {dd}", size=12, anchor="end", fill=MUTED)
        return y + data.x_shape[0] * (ch_ + 4)

    kr, ki = kinds_of(slots), kinds_of(inl)
    y1 = grid(
        ycols + 58,
        slots,
        f"with a store target: {fmt_int(len(hdr))} bytes = 16 + {n} × 40 ({kr[0]} REFERENCE, {kr[2]} REPEAT)",
    )
    ilen = len(data.inline_header)
    data_b = sum(len(s.data) for s in inl)
    y2 = grid(
        y1 + 34,
        inl,
        f"inline: true: {fmt_int(ilen)} bytes = 16 + {n} × 40 + {fmt_int(data_b)} of Blosc frames "
        f"({ki[1]} INLINE, {ki[2]} REPEAT)",
    )
    assert ilen == 16 + 40 * n + data_b

    # (d) the two headers to scale
    yd = y2 + 40
    panel_label(svg, 20, yd, "(d)", "to scale: each header as a bar of its bytes")
    W = 852

    def bar(y, sl, total, label):
        sx = x0
        svg.text(x0, y - 6, label, size=12, fill=MUTED)
        svg.rect(sx, y, W * 16 / total, 22, fill=FAINT, stroke=INK, sw=0.6)
        sx += W * 16 / total
        for s in sl:
            w = W * 40 / total
            svg.rect(sx, y, w, 22, fill=KIND_FILL[s.kind], stroke=STATIC, sw=0.6)
            sx += w
            if s.data:
                w = W * len(s.data) / total
                svg.rect(sx, y, w, 22, fill=STATIC_MID, stroke=STATIC, sw=0.6)
                if w > 70:
                    svg.text(sx + w / 2, y + 15.5, f"{fmt_int(len(s.data))} B", size=12, anchor="middle", mono=True)
                sx += w
        assert abs(sx - x0 - W) < 1e-6

    bar(yd + 28, slots, len(hdr), f"with a store target: {fmt_int(len(hdr))} bytes (fixed part, then 20 records)")
    bar(
        yd + 78,
        inl,
        ilen,
        f"inline: true: {fmt_int(ilen)} bytes; the records are the hairlines, the frames hold the values",
    )
    yf = yd + 124
    svg.rect(20, yf, 860, 50, fill=FAINT, stroke=INK, sw=1.2, rx=6)
    svg.text(
        32,
        yf + 20,
        "frozen (header version 1): every offset and type above, the kind codes 0, 1, 2, the slot order and the",
        size=12.5,
        weight="bold",
    )
    svg.text(
        32,
        yf + 38,
        "Blosc frame of INLINE data. Not frozen: which kind a slot gets (it depends on the writer's sidecar and "
        "cache).",
        size=12.5,
    )
    svg.h = round(yf + 64)
    return svg.save(out / "header.svg")


# ================================================================================================ figure 4
def envelope_box(svg: Svg, x, y, w, shape, headers, title, popped=None):
    """An envelope drawn top down: prefix, shape, the header stack; returns the y where the payload starts."""
    svg.text(x, y - 8, title, size=12.5, weight="bold")
    parts = [("FCHN, version 1, kind 0, ndim 4", 24), (f"shape ({', '.join(fmt_int(v) for v in shape)})", 26)]
    parts += [(f"header {h}: {fmt_int(len(b))} B", 26) for h, b in headers]
    yy = y
    pop_mid = None
    for t, h in parts:
        svg.rect(x, yy, w, h, fill=FAINT, stroke=LINE, sw=1)
        svg.text(x + 10, yy + h / 2 + 4.5, t, size=12.5)
        yy += h
    if popped is not None:
        h = 30
        svg.rect(x, yy, w, h, fill=STATIC_LIGHT, stroke=STATIC, sw=1.4, dash="5 3")
        svg.text(
            x + 10, yy + h / 2 + 4.5, f"header static_ref: {fmt_int(len(popped))} B, popped", size=12.5, weight="bold"
        )
        pop_mid = yy + h / 2
        yy += h
    return yy, pop_mid


def fig_decode(data: Data, out: Path) -> Path:
    nd, nf, ne, nc = data.x_shape
    nk, ns = len(data.kept), data.n_static
    ft = data.env.headers[0]
    svg = Svg(900, 900, "static_ref decode: the one full-chunk write")
    svg.text(
        20,
        26,
        "decode: pop the header, then write every value of the chunk once, into the out array of the next envelope",
        size=14,
        weight="bold",
    )
    lx, lw, rx, rw = 44, 262, 536, 262
    top = 78
    # the envelope icon_lattice_bytes hands on (it decoded straight into it)
    ly, pop_mid = envelope_box(svg, lx, top, lw, data.y_shape, [ft], "from icon_lattice_bytes: y", popped=data.header)
    ph_l = nk * 16 + 40
    svg.rect(lx, ly, lw, ph_l, fill=BG, stroke=LINE, sw=1)
    ymid = field_rows(svg, lx + 6, ly + 10, data.kept, set(), name_w=128)
    svg.text(lx + 10, ly + ph_l - 10, f"payload y: float32, {fmt_int(4 * nd * nk * ne * nc)} B", size=12.5, fill=MUTED)
    # the envelope the adapter allocates for the stage before (field_terms): out
    ry, _ = envelope_box(svg, rx, top + 30, rw, data.x_shape, [ft], "to field_terms: x = out (envelope.allocate)")
    assert ry == ly
    ph_r = nf * 16 + 40
    svg.rect(rx, ry, rw, ph_r, fill=BG, stroke=LINE, sw=1)
    xmid = field_rows(svg, rx + 6, ry + 10, data.names, set(data.stage.static_fields), name_w=128)
    svg.text(
        rx + 10, ry + ph_r - 10, f"payload out: float32, {fmt_int(4 * nd * nf * ne * nc)} B", size=12.5, fill=MUTED
    )
    # the kept fields: one block copy per run
    for a, b_, ya in data.runs:
        src = (ymid[ya] + ymid[ya + b_ - a - 1]) / 2
        dst = (xmid[a] + xmid[b_ - 1]) / 2
        bracket(svg, lx + lw - 12, ymid[ya] - 7, ymid[ya + b_ - a - 1] + 7, DECODED)
        svg.arrow(lx + lw + 2, src, rx - 2, dst, color=DECODED, sw=1.6)
        svg.text(
            (lx + lw + rx) / 2,
            (src + dst) / 2 - 10,
            f"x[:, {a}:{b_}] = y[:, {ya}:{ya + b_ - a}]",
            size=12.5,
            anchor="middle",
            mono=True,
            fill=DECODED,
            rotate=math.degrees(math.atan2(dst - src, rx - lx - lw - 4)),
        )
    # the slots: from the popped header, through the lookup, into the static rows (from the right)
    kr = kinds_of(data.slots)
    by = max(ly + ph_l, ry + ph_r) + 40
    bx, bw_, bh_ = lx, 420, 92
    svg.path(
        f"M{_f(lx)},{_f(pop_mid)} L24,{_f(pop_mid)} L24,{_f(by + bh_ / 2)} L{_f(bx - 1)},{_f(by + bh_ / 2)}",
        stroke=STATIC,
        sw=1.4,
        marker=STATIC,
    )
    svg.rect(bx, by, bw_, bh_, fill=STATIC_LIGHT, stroke=STATIC, sw=1.4, rx=6)
    svg.lines(
        bx + 12,
        by + 22,
        [
            f"its {len(data.slots)} slots, k = (d · {ns} + i) · {ne} + e:",
            f"{kr[0]} REFERENCE: statics.lookup(sha), from the sidecar",
            f"{kr[2]} REPEAT: the array of slot arg (no lookup)",
            f"{kr[1]} INLINE: statics.lookup(sha, the frame in the header)",
        ],
        size=12.5,
        gap=17,
    )
    # the sidecar files the REFERENCE slots read
    fy = by + bh_ + 34
    fw = 160
    for i, f in enumerate(data.files):
        x = lx + i * (fw + 8)
        svg.rect(x, fy, fw, 26, fill=BG, stroke=STATIC, sw=1)
        svg.text(x + fw / 2, fy + 17, f[:12] + "….f4.blosc", size=12, anchor="middle", mono=True)
    svg.text(
        lx,
        fy + 46,
        "statics/ in a sidecar on the search path (inline: true: none, the frames are in the header)",
        size=12,
        fill=MUTED,
    )
    svg.arrow(lx + 2 * (fw + 8), fy, lx + bw_ / 2, by + bh_, color=STATIC, sw=1.2, shrink0=2, shrink1=2)
    # to the static rows, entering out from the right
    tx = rx + rw + 46
    svg.path(
        f"M{_f(bx + bw_)},{_f(by + bh_ / 2)} L{_f(tx)},{_f(by + bh_ / 2)} L{_f(tx)},{_f(xmid[data.s_idx[0]])}",
        stroke=STATIC,
        sw=1.6,
    )
    for i in data.s_idx:
        svg.arrow(tx, xmid[i], rx + rw - 2, xmid[i], color=STATIC, sw=1.4)
    svg.lines(
        tx - 8,
        by + bh_ / 2 + 22,
        ["x[d, static i, e] = v", "for each slot"],
        size=12.5,
        anchor="end",
        mono=False,
        fill=STATIC,
    )
    yn = fy + 80
    svg.lines(
        20,
        yn,
        [
            f"out has the stage's input shape ({nd}, {nf}, {ne}, ·), so the adapter (zarr3.FieldStageBytesCodec) "
            "passes the array inside the",
            "envelope it allocates for the stages before (here field_terms, which then works on it in place); when "
            "static_ref is the first",
            "stage, out is a new array that zarr gets as the bare chunk. Every value is written once: "
            f"{len(data.runs)} block copies of the kept fields and {len(data.slots)} slot writes.",
        ],
        size=12.5,
        gap=18,
    )
    svg.h = round(yn + 3 * 18 - 2)
    return svg.save(out / "decode.svg")


# ================================================================================================ main
FIGURES = ("stage", "lookup", "header", "decode")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--only", nargs="*", choices=FIGURES)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["ANEMOI_DATASETS_CODECS_CACHE"] = str(Path(tmp) / "cache")  # nothing outside tmp
        os.environ.pop("ANEMOI_DATASETS_CODECS_PATH", None)
        data = Data(Path(tmp))
        for name in a.only or FIGURES:
            t0 = time.perf_counter()
            path = {"stage": fig_stage, "lookup": fig_lookup, "header": fig_header, "decode": fig_decode}[name](
                data, a.out
            )
            print(f"{path.relative_to(REPO) if path.is_relative_to(REPO) else path}: {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
