"""The figures of docs/codecs/envelope.md, computed by running the real codec chain.

    python scripts/figures/envelope_figures.py [--out DIR] [--only NAME ...]

Writes plain SVG (real text, generic fonts, an opaque light background) into docs/codecs/figures/envelope/:

* ``chain.svg``: one chunk through ``compressors=[field_terms, static_ref, icon_lattice_bytes]``: the buffer each
  codec receives and emits, the chunk spec zarr gives every codec, and what each codec checks.
* ``layout.svg``: envelope version 1 byte by byte, for the stored chunk of that chain.
* ``stack.svg``: the header stack, pushed on encode and popped (after the id check) on decode.
* ``copies.svg``: the buffers of a decode and of an encode, to scale, and which code writes them.

The chain is the one of ``tests/test_chain.py::test_three_codec_chain``: ``physics.expand`` on the synthetic test
variables and lattice (tests/conftest.py), one date, written and read back through zarr with every codec call and
every allocating ``envelope`` call recorded. Every size, offset and code path shown comes from that run (the decode
path is asserted), so the figures follow the code. It runs in a temporary directory.

The SVG helpers and the palette are those of ``icon_lattice_figures.py``, with the same meanings: orange (target)
marks the targets of ``field_terms`` and its header (their prediction coefficients), green (static) the static fields
and the header of ``static_ref`` (their sha256), blue (decoded) the coded LATC container.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import types
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from icon_lattice_figures import (
    BG,
    DECODED,
    DECODED_LIGHT,
    FAINT,
    INK,
    LIGHT,
    LINE,
    MUTED,
    STATIC,
    STATIC_LIGHT,
    TARGET,
    TARGET_LIGHT,
    Svg,
    _f,
    fmt_int,
    panel_label,
    text_width,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "codecs" / "figures" / "envelope"
PREFIX = LIGHT  # the envelope's own bytes: magic, sizes, shape, record heads
STAGE_COLOUR = {"field_terms": (TARGET, TARGET_LIGHT), "static_ref": (STATIC, STATIC_LIGHT)}


# ------------------------------------------------------------------------------------------------ the chain run
@dataclass(frozen=True)
class Buf:
    """A buffer a codec received or emitted: an envelope (``kind`` 0 or 1) or a bare chunk (``kind`` None)."""

    nbytes: int
    kind: int | None
    shape: tuple[int, ...] | None
    headers: tuple[tuple[str, int], ...]  # (stage id, header payload bytes), bottom first
    header_bytes: int
    writable: bool
    addr: int


@dataclass(frozen=True)
class Call:
    """One ``_encode_sync`` / ``_decode_sync`` of a codec adapter."""

    codec: str
    op: str
    spec: tuple[int, ...]  # the chunk spec's shape zarr passed
    inp: Buf
    out: Buf
    env: tuple[str, ...]  # the envelope functions it called (allocate, pack, pack_bytes, repack_in_place)


class Run:
    """The three-codec chain on the synthetic test chunk, through zarr, every adapter call recorded."""

    def __init__(self) -> None:
        sys.path.insert(0, str(REPO / "tests"))
        import conftest  # noqa: E402
        import zarr

        from anemoi_datasets_codecs import chain, envelope, zarr3
        from anemoi_datasets_codecs.icon_lattice import register_store
        from anemoi_datasets_codecs.physics import expand
        from anemoi_datasets_codecs.static_ref import header as sr_header

        lat = conftest.make_lattice()
        chunk = conftest.make_chunk(lat)
        entries = expand(conftest.NAMES, None, conftest.make_config(lat[0]))
        assert [e["name"] for e in entries] == ["field_terms", "static_ref", "icon_lattice_bytes"]
        codecs = [chain.codec_class(e["name"])(**e["configuration"]) for e in entries]
        self.names = list(conftest.NAMES)
        self.static = list(entries[1]["configuration"]["static_fields"])
        self.targets = list(entries[0]["configuration"]["terms"])
        self.kept = list(entries[2]["configuration"]["variables"])
        self.sources = sorted({s for t in entries[0]["configuration"]["terms"].values() for s in t["sources"]})

        def describe(buf) -> Buf:
            b = np.asarray(buf.as_numpy_array())
            env = envelope.parse(b)
            addr = b.__array_interface__["data"][0]
            if env is None:
                return Buf(b.nbytes, None, None, (), 0, bool(b.flags.writeable), addr)
            hs = tuple((h, len(p)) for h, p in env.headers)
            return Buf(b.nbytes, env.kind, env.shape, hs, env.header_bytes, bool(b.flags.writeable), addr)

        calls: list[Call] = []
        active: list[list[str]] = []
        depth = [0]
        saved = []

        def wrap_codec(cls, name):
            orig = cls.__dict__[name]

            def f(self, b, spec):
                active.append([])
                inp = describe(b)
                out = orig(self, b, spec)
                env = tuple(active.pop())
                calls.append(Call(self.codec_name, name[1:7], tuple(spec.shape), inp, describe(out), env))
                return out

            saved.append((cls, name, orig))
            setattr(cls, name, f)

        def wrap_env(name):
            orig = getattr(envelope, name)

            def f(*a, **k):
                top = depth[0] == 0
                depth[0] += 1
                try:
                    res = orig(*a, **k)
                finally:
                    depth[0] -= 1
                if top and active:
                    active[-1].append(name if res is not None else f"{name} (None)")
                return res

            saved.append((envelope, name, orig))
            setattr(envelope, name, f)

        for cls in (zarr3.FieldStageBytesCodec, zarr3.ChunkBytesBytesCodec):
            for m in ("_encode_sync", "_decode_sync"):
                wrap_codec(cls, m)
        for fn in ("allocate", "pack", "pack_bytes", "repack_in_place"):
            wrap_env(fn)
        try:
            path = Path(tempfile.mkdtemp(dir=os.environ["ANEMOI_DATASETS_CODECS_CACHE"])) / "chain.zarr"
            g = zarr.open_group(str(path), mode="w")
            register_store(path)
            data = chunk[None, :, None, :]
            a = g.create_array("data", shape=data.shape, chunks=data.shape, dtype="float32", compressors=codecs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a[:] = data
                back = zarr.open_group(str(path), mode="r")["data"][...]
        finally:
            for obj, name, orig in reversed(saved):
                setattr(obj, name, orig)
        self.shape = data.shape
        self.codec_names = [c["name"] for c in entries]
        self.encode = [c for c in calls if c.op == "encode"]
        self.decode = [c for c in calls if c.op == "decode"]
        assert [c.codec for c in self.encode] == self.codec_names
        assert [c.codec for c in self.decode] == self.codec_names[::-1]
        assert all(c.spec == self.shape for c in calls)  # zarr gives every codec the array's chunk spec
        for i in [self.names.index(n) for n in self.static]:
            assert np.array_equal(back[0, i].view(np.uint32), data[0, i].view(np.uint32))
        self.stored = (path / "data" / "c" / "0" / "0" / "0" / "0").read_bytes()
        env = envelope.parse(self.stored)
        assert env is not None and env.kind == envelope.KIND_BYTES and self.stored[env.header_bytes :][:4] == b"LATC"
        self.env = env
        self.header_payloads = dict(env.headers)
        self.n_cells = data.shape[-1]
        recs = codecs[0].codec.records(self.header_payloads["field_terms"], self.shape)
        self.ft_slices, self.ft_terms = len(recs), len(recs[0])
        n_static, slots = sr_header.parse(self.header_payloads["static_ref"])
        self.sr_slots = [sr_header.KIND_LABELS[s.kind] for s in slots]
        assert n_static == len(self.static)
        # what a stage says when the top header is not its own: static_ref given field_terms' envelope
        lone = envelope.pack(np.zeros(self.shape, np.float32), [("field_terms", self.header_payloads["field_terms"])])
        try:
            codecs[1]._decode_sync(
                types.SimpleNamespace(as_numpy_array=lambda: lone), types.SimpleNamespace(shape=self.shape)
            )
        except ValueError as e:
            self.wrong_top = str(e)
        else:
            raise AssertionError("static_ref decoded an envelope whose top header is not its own")

    def call(self, op: str, codec: str) -> Call:
        return next(c for c in (self.encode if op == "encode" else self.decode) if c.codec == codec)


def fmt_shape(shape) -> str:
    return "(" + ", ".join(fmt_int(int(s)) for s in shape) + ")"


def field_fill(name: str, run: Run) -> str:
    return TARGET_LIGHT if name in run.targets else STATIC_LIGHT if name in run.static else FAINT


def stripes(svg: Svg, x: float, y: float, w: float, h: float, names: list[str], run: Run, sw: float = 0.6) -> None:
    """The field array of an envelope payload or a bare chunk: one stripe per field, coloured by kind."""
    k = w / len(names)
    for i, n in enumerate(names):
        svg.rect(x + i * k, y, k, h, fill=field_fill(n, run), stroke=LINE, sw=sw)


def legend(svg: Svg, x: float, y: float, items, size: float = 12.5) -> float:
    """Swatches with an outline in a row; returns the x after the last one."""
    for fill, stroke, label in items:
        svg.rect(x, y - 11, 14, 14, fill=fill, stroke=stroke, sw=1)
        svg.text(x + 20, y, label, size=size)
        x += 20 + text_width(label, size) + 18
    return x


# ================================================================================================ figure 1
def fig_chain(run: Run, out: Path) -> Path:
    svg = Svg(900, 874, "One chunk through a codec chain")
    spec = fmt_shape(run.shape)
    svg.text(
        20, 26, "one chunk through compressors=[field_terms, static_ref, icon_lattice_bytes]", size=14, weight="bold"
    )
    svg.rect(20, 40, 860, 62, fill=FAINT, stroke=LINE, sw=1, rx=6)
    svg.text(
        32, 60, f"zarr gives every codec the same chunk spec, the array's: {spec} float32", size=12.5, weight="bold"
    )
    svg.lines(
        32,
        78,
        [
            'at array creation (validate, against the chunk grid) and for every chunk (chunk_spec, "spec" below). No',
            "codec announces another shape (the stages keep resolve_metadata the identity): it travels in the "
            "envelope.",
        ],
        size=12.5,
        gap=16,
    )
    # columns: codecs (A), the buffer between two codecs (B), the checks (C)
    ax, aw = 20, 200
    bx, bw_max = 244, 252
    cx = 586
    y0, bh, pitch = 156, 40, 116
    svg.text(ax, y0 - 16, "codec", size=12.5, fill=MUTED, weight="bold")
    svg.text(bx, y0 - 16, "what passes between two codecs", size=12.5, fill=MUTED, weight="bold")
    svg.text(cx, y0 - 16, "what the codec checks", size=12.5, fill=MUTED, weight="bold")
    nodes = [
        ("chunk of the array", f"NDBuffer {spec}", True),
        ("bytes", "zarr's array → bytes", False),
        ("field_terms", "targets → residuals", False),
        ("static_ref", "drops the static fields", False),
        ("icon_lattice_bytes", "lattice codec, chain: true", False),
        ("store", "the chunk file", True),
    ]
    ys = [y0 + i * pitch for i in range(len(nodes))]
    for (name, sub, outer), y in zip(nodes, ys, strict=True):
        stroke = STAGE_COLOUR.get(name, (INK, None))[0]
        svg.rect(ax, y, aw, bh, fill=BG if outer else FAINT, stroke=stroke, sw=1.4, rx=6, dash="5 3" if outer else None)
        svg.text(ax + aw / 2, y + 17, name, size=13, anchor="middle", weight="bold", mono=not outer)
        svg.text(ax + aw / 2, y + 33, sub, size=12, anchor="middle", fill=MUTED)
    xe, xd = ax + 44, ax + aw - 44
    for i in range(len(nodes) - 1):
        svg.arrow(xe, ys[i] + bh, xe, ys[i + 1], color=INK, shrink0=2, shrink1=2)
        svg.arrow(xd, ys[i + 1], xd, ys[i] + bh, color=DECODED, shrink0=2, shrink1=2)
    svg.text(xe - 6, ys[0] + bh + 38, "encode", size=12, anchor="end", fill=INK)
    svg.text(xd + 6, ys[0] + bh + 38, "decode", size=12, fill=DECODED)

    # the buffers, from the recorded calls (the decode side sees the same shapes, stacks and sizes)
    enc = {c.codec: c for c in run.encode}
    dec = {c.codec: c for c in run.decode}
    bufs = [enc["field_terms"].inp, enc["field_terms"].out, enc["static_ref"].out, enc["icon_lattice_bytes"].out]
    for b, (src, dst) in zip(
        bufs[1:],
        [("static_ref", "field_terms"), ("icon_lattice_bytes", "static_ref"), (None, "icon_lattice_bytes")],
        strict=True,
    ):
        d_in = dec[dst].inp
        assert (d_in.kind, d_in.shape, d_in.headers, d_in.nbytes) == (b.kind, b.shape, b.headers, b.nbytes)
        if src is not None:
            assert dec[src].out.headers == b.headers
    bare = bufs[0]
    assert bare.kind is None and bare.nbytes == int(np.prod(run.shape)) * 4
    scale = bw_max / bare.nbytes
    svg.text(bx, ys[0] + bh + 40, f"the chunk as a float32 array, shape {spec}", size=12.5)
    for i, b in enumerate(bufs):
        yb = ys[i + 1] + bh + 12
        x = bx
        if b.kind is None:
            stripes(svg, x, yb, bw_max, 18, run.names, run)
            l1, l2 = "bare: little-endian float32, no envelope", f"{fmt_int(b.nbytes)} B; shape only from the spec"
        else:
            svg.rect(x, yb, 36, 18, fill=PREFIX, stroke=INK, sw=0.8)
            svg.text(x + 18, yb + 13.5, "FCHN", size=12, anchor="middle", mono=True)
            x += 36
            for h, _ in b.headers:
                svg.rect(x, yb, 16, 18, fill=STAGE_COLOUR[h][0], stroke=INK, sw=0.8)
                x += 16
            pw = (b.nbytes - b.header_bytes) * scale
            if b.kind == 0:
                names = run.names if b.shape[1] == len(run.names) else run.kept
                assert len(names) == b.shape[1]
                stripes(svg, x, yb, pw, 18, names, run)
            else:
                svg.rect(x, yb, pw, 18, fill=DECODED_LIGHT, stroke=DECODED, sw=1)
                svg.text(x + pw / 2, yb + 13.5, "LATC", size=12, anchor="middle", mono=True)
            l1 = f"kind {b.kind}, shape {fmt_shape(b.shape)}"
            l2 = f"{fmt_int(b.nbytes)} B: {b.header_bytes} B prefix + " + (
                "the field array" if b.kind == 0 else "the LATC container"
            )
        svg.text(bx, yb + 34, l1, size=12.5)
        svg.text(bx, yb + 50, l2, size=12.5, fill=MUTED)

    # the checks, per codec
    n_all, n_kept = len(run.names), len(run.kept)
    checks = {
        1: [("", "writes the chunk as little-endian float32")],
        2: [
            ("create", "float32; 4-D chunks holding every"),
            ("", "field of a date"),
            ("encode", "a bare chunk of the spec's shape;"),
            ("", f"{n_all} fields = its variables"),
            ("decode", "top header 'field_terms'? pop it;"),
            ("", f"{n_all} fields out; the stack is empty:"),
            ("", "the result has the spec's shape"),
        ],
        3: [
            ("create", "the same as field_terms"),
            ("encode", "kind 0; dates, members and cells ="),
            ("", f"the spec's; {n_all} fields = its variables"),
            ("decode", "top header 'static_ref'? pop it;"),
            ("", f"{n_kept} fields in; dates, members and"),
            ("", "cells = the spec's"),
        ],
        4: [
            ("create", "float32; 4-D chunks holding every"),
            ("", "field of a date; cells and members"),
            ("", "of its lattice (not the field count)"),
            ("encode", "kind 0; dates, members, cells = the"),
            ("", f"spec's; {n_kept} fields = its variables"),
            ("decode", "kind 1; the same shape checks"),
        ],
        5: [("", "stores the bytes it is given")],
    }
    for i, rows in checks.items():
        for j, (tag, s) in enumerate(rows):
            y = ys[i] + 14 + j * 15
            if tag:
                svg.text(cx, y, tag, size=12, weight="bold", fill=MUTED)
            svg.text(cx + 54, y, s, size=12)
    yl = ys[-1] + bh + 34
    x = legend(
        svg,
        ax,
        yl,
        [
            (PREFIX, INK, "envelope prefix"),
            (TARGET, INK, "field_terms header"),
            (STATIC, INK, "static_ref header"),
            (DECODED_LIGHT, DECODED, "LATC container"),
        ],
    )
    legend(
        svg,
        ax,
        yl + 22,
        [
            (TARGET_LIGHT, LINE, f"field_terms targets ({', '.join(run.targets)})"),
            (STATIC_LIGHT, LINE, f"static fields ({len(run.static)})"),
            (FAINT, LINE, "other fields"),
        ],
    )
    svg.text(ax, yl + 46, "bars: payload to scale, prefix and headers not to scale", size=12, fill=MUTED)
    return svg.save(out / "chain.svg")


# ================================================================================================ figure 2
def fig_layout(run: Run, out: Path) -> Path:
    env, raw = run.env, run.stored
    hb, pb = env.header_bytes, len(raw) - env.header_bytes
    svg = Svg(900, 668, "Envelope version 1, byte layout")
    x0 = 30

    # (a) the fixed head, bytes 0..24
    panel_label(svg, 20, 28, "(a)", "the stored chunk of the chain: head, bytes 0 to 24 (integers little endian)")
    B = 35.0
    y = 58
    head = [
        (0, 4, raw[:4].decode("ascii"), "magic", "4 B", 1),
        (4, 1, str(raw[4]), "version", "u8", 1),
        (5, 1, str(raw[5]), "ndim", "u8", 2),
        (6, 1, str(raw[6]), "kind", "u8", 1),
        (7, 1, str(raw[7]), "reserved", "u8", 2),
        (8, 2, str(int.from_bytes(raw[8:10], "little")), "n_headers", "u16", 1),
        (10, 2, str(int.from_bytes(raw[10:12], "little")), "reserved", "u16", 2),
        (12, 4, str(int.from_bytes(raw[12:16], "little")), "header_bytes", "u32", 1),
        (16, 8, fmt_int(int.from_bytes(raw[16:24], "little")), "payload_bytes", "u64", 1),
    ]
    assert int(head[7][2]) == hb and head[8][2] == fmt_int(pb) and head[2][2] == str(len(env.shape))
    assert head[3][2] == "1" and head[5][2] == str(len(env.headers))
    for off, nb, val, lbl, typ, row in head:
        svg.rect(x0 + off * B, y, nb * B, 40, fill=PREFIX, stroke=INK, sw=1)
        svg.text(x0 + (off + nb / 2) * B, y + 18, val, size=12.5, anchor="middle", mono=True)
        svg.text(x0 + (off + nb / 2) * B, y + 33, typ, size=12, anchor="middle", fill=MUTED)
        svg.text(x0 + off * B + 2, y - 6, str(off), size=12, fill=MUTED)
        svg.text(x0 + (off + nb / 2) * B, y + 40 + 16 * row, lbl, size=12, anchor="middle")
    svg.text(x0 + 24 * B, y - 6, "24", size=12, fill=MUTED, anchor="end")
    svg.text(
        x0,
        y + 40 + 16 * 3 + 4,
        "kind 1: the payload is opaque bytes a later codec wrote (kind 0: a float32 array of the shape)",
        size=12,
        fill=MUTED,
    )

    # (b) the prefix to scale
    yb = y + 136
    panel_label(svg, 20, yb, "(b)", f"the prefix, bytes 0 to header_bytes = {hb}, to scale")
    W = 840
    s = W / hb
    yb2 = yb + 36
    shape_end = 24 + 8 * len(env.shape)
    segs = [(0, 24, PREFIX, INK, "head"), (24, shape_end, PREFIX, INK, "shape")]
    for off, end, fill, stroke, lbl in segs:
        svg.rect(x0 + off * s, yb2, (end - off) * s, 36, fill=fill, stroke=stroke, sw=1)
        svg.text(x0 + (off + end) / 2 * s, yb2 + 23, lbl, size=12.5, anchor="middle")
    off = shape_end
    recs = []
    for sid, payload in env.headers:
        n, m = len(sid), len(payload)
        end = off + 8 + n + m
        pad = -end % 8
        recs.append((sid, off, n, m, pad))
        col, light = STAGE_COLOUR[sid]
        svg.rect(x0 + off * s, yb2, 8 * s, 36, fill=PREFIX, stroke=INK, sw=1)
        svg.rect(x0 + (off + 8) * s, yb2, n * s, 36, fill=PREFIX, stroke=INK, sw=1)
        svg.rect(x0 + (off + 8 + n) * s, yb2, m * s, 36, fill=light, stroke=col, sw=1.2)
        if pad:
            svg.rect(x0 + end * s, yb2, pad * s, 36, fill=BG, stroke=INK, sw=1)
        svg.text(x0 + (off + 8 + n + m / 2) * s, yb2 + 23, f"{m} B", size=12.5, anchor="middle")
        svg.text(x0 + off * s + 2, yb2 - 6, str(off), size=12, fill=MUTED)
        svg.text(x0 + (off + (end + pad - off) / 2) * s, yb2 + 54, f"record of {sid}", size=12.5, anchor="middle")
        off = end + pad
    assert off == hb
    svg.text(x0, yb2 - 6, "0", size=12, fill=MUTED)
    svg.text(x0 + 24 * s + 2, yb2 - 6, "24", size=12, fill=MUTED)
    svg.text(x0 + W, yb2 - 6, str(hb), size=12, fill=MUTED, anchor="end")
    dims = ", ".join(fmt_int(v) for v in env.shape)
    svg.text(x0 + shape_end * s / 2, yb2 + 54, f"i64 × {len(env.shape)}: {dims}", size=12.5, anchor="middle")

    # (c) the records, field by field (not to scale)
    yc = yb2 + 102
    panel_label(svg, 20, yc, "(c)", "the header records, bottom of the stack first (not to scale)")
    cols = [
        (64, "u16 n"),
        (64, "u16 0"),
        (76, "u32 m"),
        (150, "stage id, n B"),
        (300, "header payload, m B"),
        (100, "0-pad to 8"),
    ]
    yr = yc + 34
    for sid, off, n, m, pad in recs:
        col, light = STAGE_COLOUR[sid]
        payload = run.header_payloads[sid]
        what = (
            f"{run.ft_terms} terms, {run.ft_slices} slice: term table + coefficients"
            if sid == "field_terms"
            else f"{len(run.sr_slots)} slots: {', '.join(sorted(set(run.sr_slots)))} + sha256"
        )
        vals = [str(n), "0", str(m), sid, what, f"{pad} B"]
        offs = [off, off + 2, off + 4, off + 8, off + 8 + n, off + 8 + n + m]
        x = x0
        for (wdt, _), v, o, i in zip(cols, vals, offs, range(6), strict=True):
            fill = light if i == 4 else BG if i == 5 else PREFIX
            svg.rect(x, yr, wdt, 32, fill=fill, stroke=col if i == 4 else INK, sw=1.2 if i == 4 else 1)
            svg.text(x + wdt / 2, yr + 21, v, size=12.5 if i == 4 else 12.5, anchor="middle", mono=i in (0, 1, 2, 3))
            svg.text(x + 2, yr - 5, str(o), size=12, fill=MUTED)
            x += wdt
        svg.text(x, yr - 5, str(off + 8 + n + m + pad), size=12, fill=MUTED, anchor="end")
        assert len(payload) == m
        yr += 54
    x = x0
    for wdt, lbl in cols:
        svg.text(x + wdt / 2, yr - 6, lbl, size=12, anchor="middle", fill=MUTED)
        x += wdt
    svg.text(
        x0,
        yr + 14,
        "a header payload is written and read by its stage only (its own frozen format); the envelope does not read it",
        size=12.5,
    )

    # (d) the whole stored chunk to scale
    yd = yr + 52
    panel_label(svg, 20, yd, "(d)", f"the whole stored chunk, {fmt_int(len(raw))} bytes, to scale")
    yd2 = yd + 18
    svg.rect(x0, yd2, W, 26, fill=DECODED_LIGHT, stroke=DECODED, sw=1)
    svg.rect(x0, yd2, max(hb * W / len(raw), 1.5), 26, fill=INK)
    svg.text(
        x0 + W / 2,
        yd2 + 18,
        f"payload at {hb}, kind 1: the LATC container (container version 3), {fmt_int(pb)} B",
        size=12.5,
        anchor="middle",
    )
    svg.text(
        x0,
        yd2 + 44,
        f"the prefix: {hb} B, {100 * hb / len(raw):.2f} % of the chunk; the payload starts 8-byte aligned",
        size=12,
        fill=MUTED,
    )

    yf = yd2 + 64
    svg.rect(20, yf, 860, 50, fill=FAINT, stroke=INK, sw=1.2, rx=6)
    svg.text(
        32,
        yf + 20,
        "frozen (envelope version 1): the magic, every offset, type and order above, the payload "
        "kinds, the padding to 8.",
        size=12.5,
        weight="bold",
    )
    svg.text(
        32,
        yf + 38,
        "not the envelope's: the header payloads (each stage freezes its own) and a kind 1 payload "
        "(the LATC container).",
        size=12.5,
    )
    return svg.save(out / "layout.svg")


# ================================================================================================ figure 3
def stack(svg: Svg, cx: float, base: float, headers, popped=None, w: float = 150, h: float = 24) -> None:
    """A header stack drawn bottom first, upwards from ``base``; ``popped``: the header just taken off it."""
    if not headers:
        svg.rect(cx - w / 2, base - h, w, h, fill=BG, stroke=LINE, sw=1, dash="4 3")
        svg.text(cx, base - 7.5, "empty", size=12, anchor="middle", fill=MUTED)
    for i, (sid, m) in enumerate(headers):
        col, light = STAGE_COLOUR[sid]
        y = base - (i + 1) * h
        svg.rect(cx - w / 2, y, w, h, fill=light, stroke=col, sw=1.4)
        svg.text(cx, y + 16.5, f"{sid}, {m} B", size=12, anchor="middle", mono=True)
    if popped is not None:
        y = base - (max(len(headers), 1) + 1) * h - 6
        svg.rect(cx - w / 2, y, w, h, fill=BG, stroke=STAGE_COLOUR[popped][0], sw=1.2, dash="4 3")
        svg.text(cx, y + 16.5, f"popped: {popped}", size=12, anchor="middle", fill=MUTED)


def fig_stack(run: Run, out: Path) -> Path:
    svg = Svg(900, 560, "The header stack")
    order = ["bytes", *run.codec_names]
    cw, gap, x0 = 196, 26, 20
    xs = [x0 + i * (cw + gap) for i in range(4)]
    cxs = [x + cw / 2 for x in xs]
    enc = {c.codec: c for c in run.encode}
    dec = {c.codec: c for c in run.decode}

    def boxes(y, arrows_right):
        for x, name in zip(xs, order, strict=True):
            stroke = STAGE_COLOUR.get(name, (INK, None))[0]
            svg.rect(x, y, cw, 30, fill=FAINT, stroke=stroke, sw=1.4, rx=6)
            svg.text(x + cw / 2, y + 20, name, size=13, anchor="middle", weight="bold", mono=True)
        for i in range(3):
            if arrows_right:
                svg.arrow(xs[i] + cw, y + 15, xs[i + 1], y + 15, color=INK, shrink0=2, shrink1=2)
            else:
                svg.arrow(xs[i + 1], y + 15, xs[i] + cw, y + 15, color=DECODED, shrink0=2, shrink1=2)

    # encode
    svg.text(
        20,
        26,
        "encode: a stage pushes its header; the codec after the stages passes the stack through",
        size=14,
        weight="bold",
    )
    ye = 42
    boxes(ye, True)
    ft, sr, lt = enc["field_terms"], enc["static_ref"], enc["icon_lattice_bytes"]
    assert ft.inp.kind is None and ft.out.headers == ft.out.headers[:1] and sr.out.headers[:1] == ft.out.headers
    assert lt.out.headers == sr.out.headers and lt.out.kind == 1
    texts = [
        ["the bare chunk:", "no envelope, no stack"],
        [f"push ('field_terms', {ft.out.headers[0][1]} B)", "onto an empty stack"],
        [f"push ('static_ref', {sr.out.headers[1][1]} B)", f"fields {ft.out.shape[1]} → {sr.out.shape[1]}"],
        ["the stack passes unread;", "the field array → LATC"],
    ]
    for cx, rows in zip(cxs, texts, strict=True):
        svg.lines(cx, ye + 52, rows, size=12.5, anchor="middle", gap=16)
    base = ye + 156
    for cx, b in zip(cxs[1:], [ft.out, sr.out, lt.out], strict=True):
        stack(svg, cx, base, b.headers)
    svg.lines(
        cxs[0], base - 26, ["the stack it hands on,", "bottom first:"], size=12.5, anchor="middle", gap=16, fill=MUTED
    )
    svg.text(cxs[3], base + 18, "stored: kind 1", size=12, anchor="middle", fill=MUTED)

    # decode
    yd = base + 70
    svg.text(20, yd - 16, "decode: a stage checks that the top header is its own and pops it", size=14, weight="bold")
    boxes(yd, False)
    ld, sd, fd = dec["icon_lattice_bytes"], dec["static_ref"], dec["field_terms"]
    assert ld.out.headers == ld.inp.headers and sd.out.headers == ld.out.headers[:-1] and fd.out.kind is None
    texts = [
        ["views the bytes as", "the chunk array"],
        ["top id = 'field_terms'? pop;", "nothing left: it was the", "first stage"],
        ["top id = 'static_ref'? pop", f"fields {sd.inp.shape[1]} → {sd.out.shape[1]}"],
        ["the stack passes unread;", "LATC → the field array", "(kind 1 → kind 0)"],
    ]
    for cx, rows in zip(cxs, texts, strict=True):
        svg.lines(cx, yd + 52, rows, size=12.5, anchor="middle", gap=16)
    base_d = yd + 156
    for cx, b, p in zip(cxs[1:], [fd.out, sd.out, ld.out], ["field_terms", "static_ref", None], strict=True):
        stack(svg, cx, base_d, b.headers, popped=p)
    svg.lines(
        cxs[0], base_d - 26, ["the stack it hands on,", "bottom first:"], size=12.5, anchor="middle", gap=16, fill=MUTED
    )
    assert fd.out.nbytes == int(np.prod(run.shape)) * 4
    svg.lines(
        cxs[1],
        base_d + 20,
        ["hands zarr the bare chunk,", "which must have the spec's", f"shape {fmt_shape(run.shape)}"],
        size=12.5,
        anchor="middle",
        gap=16,
        fill=MUTED,
    )
    yn = base_d + 82
    svg.rect(20, yn, 860, 46, fill=FAINT, stroke=LINE, sw=1, rx=6)
    svg.text(
        32,
        yn + 19,
        "stage id = codec name. A top header that is not the stage's own fails the chunk, e.g. static_ref given an",
        size=12.5,
    )
    svg.text(32, yn + 36, f"envelope whose stack is [field_terms]: “{run.wrong_top}”", size=12.5)
    return svg.save(out / "stack.svg")


# ================================================================================================ figure 4
def fig_copies(run: Run, out: Path) -> Path:
    svg = Svg(900, 908, "Buffers and copies of a chain")
    dec = {c.codec: c for c in run.decode}
    enc = {c.codec: c for c in run.encode}
    ld, sd, fd = dec["icon_lattice_bytes"], dec["static_ref"], dec["field_terms"]
    # the decode path the figure draws, as the code took it
    assert ld.env == ("allocate",) and not ld.inp.writable  # the lattice codec decodes into a new envelope
    assert sd.env == ("allocate",) and sd.inp.addr == ld.out.addr  # static_ref: out inside the next envelope
    assert fd.env == () and fd.inp.addr == sd.out.addr and fd.inp.writable  # field_terms: in place, no allocation
    assert fd.out.kind is None and fd.out.addr == fd.inp.addr + fd.inp.header_bytes  # a view of the payload
    assert enc["field_terms"].env == ("pack",) and enc["static_ref"].env == ("pack",)
    assert enc["icon_lattice_bytes"].env == ("pack_bytes",)

    panel_label(svg, 20, 28, "(a)", "decode: the buffers, to scale, and what writes them")
    lx, x0, W = 20, 128, 520
    tx = x0 + W + 22  # the text column
    k = W / sd.out.nbytes
    bh = 30
    yS, yA, yB = 52, 186, 346
    for lbl, b, y in (("stored chunk", ld.inp, yS), ("envelope A", ld.out, yA), ("envelope B", sd.out, yB)):
        svg.text(lx, y + 13, lbl, size=13, weight="bold")
        svg.text(lx, y + 28, f"{fmt_int(b.nbytes)} B", size=12, fill=MUTED)
    # the stored chunk: prefix + LATC (read-only)
    lw = (ld.inp.nbytes - ld.inp.header_bytes) * k
    svg.rect(x0 + 1.5, yS, lw, bh, fill=DECODED_LIGHT, stroke=DECODED, sw=1)
    svg.rect(x0, yS, 1.5, bh, fill=INK)
    svg.text(x0 + lw / 2, yS + 20, "LATC", size=12.5, anchor="middle", mono=True)
    svg.lines(
        tx,
        yS + 13,
        ["read-only, from the store; the dark", f"hairline: the {ld.inp.header_bytes} B prefix"],
        size=12.5,
        gap=16,
        fill=MUTED,
    )
    # envelope A and B: prefix + stripes
    pa, aw = 1.5, (ld.out.nbytes - ld.out.header_bytes) * k
    pb, bw = 1.5, (sd.out.nbytes - sd.out.header_bytes) * k
    svg.rect(x0, yA, pa, bh, fill=INK)
    stripes(svg, x0 + pa, yA, aw, bh, run.kept, run)
    svg.rect(x0, yB, pb, bh, fill=INK)
    stripes(svg, x0 + pb, yB, bw, bh, run.names, run)
    # 1: the lattice decoder writes A's whole array
    svg.poly(
        [(x0 + 1.5, yS + bh), (x0 + 1.5 + lw, yS + bh), (x0 + pa + aw, yA), (x0 + pa, yA)],
        fill=DECODED_LIGHT,
        stroke=DECODED,
        sw=0.6,
    )
    svg.text(x0 + (lw + aw) / 4 + 8, (yS + yA + bh) / 2 + 5, "C decoder", size=12.5, anchor="middle", fill=DECODED)
    svg.text(tx, yS + bh + 34, "1  icon_lattice_bytes", size=12.5, weight="bold")
    svg.lines(
        tx,
        yS + bh + 50,
        [
            "envelope.allocate makes A (the",
            "same stack); the C decoder",
            f"writes the {len(run.kept)} fields straight into",
            "its array (decode(…, out))",
        ],
        size=12.5,
        gap=16,
    )
    # 2: static_ref copies the runs of kept fields into B's array (out) and writes the static fields
    ka, kb = aw / len(run.kept), bw / len(run.names)
    runs, start = [], None
    for i, n in enumerate(run.names):
        if n not in run.static and start is None:
            start = i
        if (n in run.static or i == len(run.names) - 1) and start is not None:
            runs.append((start, i if n in run.static else i + 1))
            start = None
    ya = 0
    for a, b in runs:
        svg.poly(
            [
                (x0 + pa + ya * ka, yA + bh),
                (x0 + pa + (ya + b - a) * ka, yA + bh),
                (x0 + pb + b * kb, yB),
                (x0 + pb + a * kb, yB),
            ],
            fill=FAINT,
            stroke=LINE,
            sw=0.8,
        )
        ya += b - a
    svg.text(x0 + (runs[0][1] * kb) / 2, (yA + yB + bh) / 2 + 5, "copy", size=12.5, anchor="middle", fill=MUTED)
    si = [run.names.index(n) for n in run.static]
    sx0, sx1 = x0 + pb + min(si) * kb, x0 + pb + (max(si) + 1) * kb
    ysc = yB + bh + 22
    svg.rect((sx0 + sx1) / 2 - 40, ysc, 80, 26, fill=STATIC_LIGHT, stroke=STATIC, sw=1.2, rx=4)
    svg.text((sx0 + sx1) / 2, ysc + 18, "sidecar", size=12.5, anchor="middle")
    svg.arrow((sx0 + sx1) / 2, ysc, (sx0 + sx1) / 2, yB + bh, color=STATIC, sw=1.4, shrink1=2)
    svg.text(tx, yA + bh + 34, f"2  static_ref, {sd.inp.shape[1]} → {sd.out.shape[1]} fields", size=12.5, weight="bold")
    svg.lines(
        tx,
        yA + bh + 50,
        [
            "envelope.allocate makes B (stack",
            "[field_terms]), its array is out;",
            f"the stage copies the {len(runs)} runs of",
            f"kept fields and writes the {len(run.static)}",
            "static fields from the sidecar",
        ],
        size=12.5,
        gap=16,
    )
    # 3: field_terms in place on B's targets; 4: the byte view to zarr
    ti = [run.names.index(t) for t in run.targets]
    t0, t1 = x0 + pb + min(ti) * kb, x0 + pb + (max(ti) + 1) * kb
    svg.path(
        f"M{_f(t0 + 6)},{_f(yB + bh + 2)} C{_f(t0 + 6)},{_f(yB + bh + 30)} {_f(t1 - 6)},{_f(yB + bh + 30)} "
        f"{_f(t1 - 6)},{_f(yB + bh + 4)}",
        stroke=TARGET,
        sw=1.6,
        marker=TARGET,
    )
    svg.text(tx, yB + 13, "3  field_terms", size=12.5, weight="bold")
    svg.text(tx, yB + 29, f"residuals → {', '.join(run.targets)}, in place", size=12.5)
    svg.text(tx, yB + bh + 30, "4  field_terms → zarr", size=12.5, weight="bold")
    yv = yB + bh + 66
    svg.path(
        f"M{_f(x0 + pb)},{_f(yv - 6)} L{_f(x0 + pb)},{_f(yv)} L{_f(x0 + pb + bw)},{_f(yv)} L{_f(x0 + pb + bw)},"
        f"{_f(yv - 6)}",
        stroke=INK,
        sw=1.2,
    )
    svg.text(
        x0 + pb + bw / 2,
        yv + 18,
        f"a byte view of B's array (offset {sd.out.header_bytes}) → zarr",
        size=12.5,
        anchor="middle",
    )
    svg.lines(
        tx,
        yB + bh + 46,
        [
            "the stack is empty: zarr's bytes",
            "codec gets a byte view of B's",
            "array and views it as the chunk",
        ],
        size=12.5,
        gap=16,
    )
    svg.text(
        lx,
        yv + 46,
        "Nothing is copied in steps 1, 3 and 4; step 2, the one full write of a stage that changes "
        "the field count, writes into",
        size=12.5,
    )
    svg.text(
        lx,
        yv + 62,
        "the envelope it hands on. envelope.repack_in_place is not needed: the in-place stage, "
        "field_terms, is the first.",
        size=12.5,
    )

    # (b) the other decode paths of the adapter
    yb = yv + 104
    panel_label(svg, 20, yb, "(b)", "the other decode paths of a stage (none taken by this chain)")
    bx, yb1 = lx + 52, yb + 22
    svg.text(lx, yb1 + 16, "before", size=12, fill=MUTED)
    svg.text(lx, yb1 + 44, "after", size=12, fill=MUTED)
    svg.rect(bx, yb1, 56, 22, fill=PREFIX, stroke=INK, sw=1)
    svg.text(bx + 28, yb1 + 16, "prefix", size=12, anchor="middle")
    svg.rect(bx + 56, yb1, 40, 22, fill=PREFIX, stroke=INK, sw=1)
    svg.text(bx + 76, yb1 + 16, "top", size=12, anchor="middle")
    svg.rect(bx + 96, yb1, 150, 22, fill=FAINT, stroke=INK, sw=1)
    svg.text(bx + 171, yb1 + 16, "array (decoded in place)", size=12, anchor="middle")
    svg.rect(bx, yb1 + 28, 40, 22, fill=BG, stroke=LINE, sw=1, dash="3 2")
    svg.rect(bx + 40, yb1 + 28, 56, 22, fill=PREFIX, stroke=INK, sw=1)
    svg.text(bx + 68, yb1 + 44, "prefix", size=12, anchor="middle")
    svg.rect(bx + 96, yb1 + 28, 150, 22, fill=FAINT, stroke=INK, sw=1)
    svg.text(bx + 171, yb1 + 44, "array, not moved", size=12, anchor="middle")
    svg.lines(
        bx + 266,
        yb1 + 16,
        [
            "an in-place stage with headers left below it: envelope.repack_in_place",
            "writes the shorter prefix just before the array, in the same buffer",
        ],
        size=12.5,
        gap=16,
    )
    svg.lines(
        lx,
        yb1 + 76,
        [
            "a read-only envelope (a compressor such as blosc after the stage): an in-place stage works on one copy;",
            "the first stage of a chain that changes the field count decodes into a new array, handed to zarr as a "
            "byte view.",
        ],
        size=12.5,
        gap=17,
    )

    # (c) encode: one copy per stage
    yc = yb1 + 130
    panel_label(svg, 20, yc, "(c)", "encode: the adapter copies each stage's output once into the next envelope")
    heads = ["reads (a view)", "its engine returns", "the adapter copies it into"]
    cols = [172, 412, 652]
    cw = 196
    kk = cw / enc["field_terms"].inp.nbytes
    for x, h in zip(cols, heads, strict=True):
        svg.text(x, yc + 26, h, size=12.5, fill=MUTED, weight="bold")
    yr = yc + 40
    for c in run.encode:
        stroke = STAGE_COLOUR.get(c.codec, (INK, None))[0]
        svg.text(lx, yr + 15, c.codec, size=12.5, mono=True, weight="bold", fill=INK)
        inb = c.inp
        svg.rect(cols[0], yr, inb.nbytes * kk, 20, fill=FAINT, stroke=INK, sw=0.8)
        if inb.kind is not None:
            svg.rect(cols[0], yr, max(inb.header_bytes * kk, 1.2), 20, fill=INK)
        if c.codec == "icon_lattice_bytes":
            yb_ = c.out.nbytes - c.out.header_bytes
            svg.rect(cols[1], yr, yb_ * kk, 20, fill=DECODED_LIGHT, stroke=DECODED, sw=0.8)
            what = "the LATC container"
            how = "envelope.pack_bytes (kind 1)"
        else:
            yb_ = c.out.nbytes - c.out.header_bytes
            svg.rect(cols[1], yr, yb_ * kk, 20, fill=FAINT, stroke=INK, sw=0.8)
            what = "residuals: a copy of x" if c.codec == "field_terms" else "x[:, kept]: a new array"
            how = "envelope.pack (kind 0)"
        svg.rect(
            cols[2],
            yr,
            c.out.nbytes * kk,
            20,
            fill=FAINT if c.out.kind == 0 else DECODED_LIGHT,
            stroke=INK if c.out.kind == 0 else DECODED,
            sw=0.8,
        )
        svg.rect(cols[2], yr, max(c.out.header_bytes * kk, 1.2), 20, fill=INK)
        svg.arrow(cols[0] + cw + 6, yr + 10, cols[1] - 8, yr + 10, color=stroke, sw=1.3)
        svg.arrow(cols[1] + cw + 6, yr + 10, cols[2] - 8, yr + 10, color=INK, sw=1.3)
        svg.text(cols[1], yr + 34, f"{what}, {fmt_int(yb_)} B", size=12, fill=MUTED)
        svg.text(cols[2], yr + 34, f"{how}, {fmt_int(c.out.nbytes)} B", size=12, fill=MUTED)
        svg.text(
            cols[0], yr + 34, "zarr's bare bytes" if inb.kind is None else "the envelope's array", size=12, fill=MUTED
        )
        yr += 52
    svg.text(
        lx,
        yr + 6,
        "a stage's input is read-only (it can be the caller's array); bars to scale, the dark end "
        "is the envelope prefix",
        size=12,
        fill=MUTED,
    )
    return svg.save(out / "copies.svg")


# ================================================================================================ main
FIGURES = {"chain": fig_chain, "layout": fig_layout, "stack": fig_stack, "copies": fig_copies}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--only", nargs="*", choices=FIGURES)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["ANEMOI_DATASETS_CODECS_CACHE"] = tmp  # the store, structures, captured statics: all in tmp
        os.environ.pop("ANEMOI_DATASETS_CODECS_PATH", None)
        t0 = time.perf_counter()
        run = Run()
        print(f"chain run: {time.perf_counter() - t0:.1f} s")
        for name in a.only or FIGURES:
            t0 = time.perf_counter()
            path = FIGURES[name](run, a.out)
            print(f"{path.relative_to(REPO) if path.is_relative_to(REPO) else path}: {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
