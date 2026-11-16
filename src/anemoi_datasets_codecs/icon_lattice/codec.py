"""``LatticeCodec``: the engine of the ICON lattice codec, for anemoi ``data`` chunks on an ICON triangle lattice.

It works on whole chunks (array in, container bytes out) and knows nothing about zarr; the Zarr 3 codecs
``icon_lattice`` / ``icon_lattice_bytes`` (``zarr3.py``, on the generic adapter ``anemoi_datasets_codecs.zarr3``)
wrap it. Its jobs are the configuration (``config.py``: schema, options and their defaults), the plan (a
``plan.Plan``: produced by the package's physics layer, ``anemoi_datasets_codecs.physics``, from the ``physics``
group, or given as ``plan=``) and the
orchestration of encode and decode.

Encode: per field one mode (``container.py``): CONSTANT, STATIC_REF (``statics.py``), RECIPE (``fields.py``,
``recipes.py``; needs the valid date) or LOSSLESS for the lossless variables, then LATTICE for the lossy ones
(``encoder.py``, verified through the C decoder, LOSSLESS on failure) in waves of the plan's decode DAG, so that cross
and mask terms see the decoded sources. Decode: LOSSLESS fields and static arrays in Python, then one C call per chunk
(``csrc/decode.c``, ``threads.decoder_threads()`` threads); it needs the lattice (``grids.py``) and the referenced
static arrays, both found on the sidecar search path (``sidecar.py``), never the grid file.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import threading
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numcodecs import get_codec
from numcodecs.compat import ensure_contiguous_ndarray, ensure_ndarray

from .. import physics as phys
from .. import plan as P
from ..config import IGNORED_KEYS, UnknownKeys
from . import container, details, grids, native, statics
from .config import (
    CODEC_ID,
    CONFIG_VERSION,
    DEFAULT_LOSSLESS,
    DEFAULT_STRUCTURE,
    STRUCTURES,
    Variable,
    check_options,
    resolve_variables,
)
from .container import (
    FALLBACK_FAILED,
    FALLBACK_NONE,
    FALLBACK_SMALLER,
    MODE_CONSTANT,
    MODE_LATTICE,
    MODE_LOSSLESS,
    MODE_STATIC_REF,
    to_seconds,
)
from .encoder import EncodeError, EncodeOptions, encode_field
from .fields import FieldIndex, encode_recipe, mask_term
from .layout import TriangleLatticeLayout
from .recipes import RECIPE_IDS
from .structure import Structure, get_structure
from .threads import blas_serial, decoder_threads, map_threads


class LatticeCodec:
    """Error-bounded codec for ``(dates, variables, ensembles, cells)`` float32 chunks on a triangle-lattice grid.

    Args:
        variables: Variable names in chunk order (or dicts ``{name, max_error, clip}``).
        grid: Grid alias / uuid of a bundled grid, a grid reference ``{name, uuid, n_cells, sha256}``, or a
            ``TriangleLatticeLayout`` (registered in this process; tests and tools).
        bounds: Max abs error per variable (names or regular expressions; variables without one are lossless).
        clip: Optional ``[lo, hi]`` per variable (names or regular expressions), applied on decode.
        static_fields: Lossless variables identical in every chunk (coded as references).
        n_ensembles: Size of the ensemble axis of a chunk.
        structure: Prediction structure (``tree``, ``tree_b`` or ``hc2_k3``).
        lossless: numcodecs config of the LOSSLESS fields.
        options: Encoder options (``config.DEFAULT_OPTIONS``; docs/user-guide.md).
        physics: The physics layer: on (default) / off, or a dict (``physics.PHYSICS_DEFAULTS``).
        plan: A candidate ``plan.Plan`` to use instead of the one the physics layer resolves (the codec's limits,
            ``max_depth`` and 4 cross sources, still apply; ``physics`` then only supplies ``recipe_bound``). For
            experiments and tools: the plan is not part of the configuration (``get_config`` and the stored metadata
            do not have it). Decoding is unaffected (the decoder reads the terms from the chunk), but a codec rebuilt
            from the metadata (``from_config``) encodes with the physics plan of the configuration. Pickling keeps
            the plan.
        chain: The codec follows field stages in a chain (``anemoi_datasets_codecs.zarr3``): the chunk grid may have
            another field count than ``variables`` (it is checked per chunk instead), and an enveloped input is coded
            inside its envelope. A bare input gives the same bytes as without it.

    A stray ``id`` or ``name`` key (a codec entry's identifier passed along with its configuration) and the writer
    stamp ``written_by`` (``config.STAMP_KEY``) are ignored.
    """

    def __init__(
        self,
        variables: Sequence[str | Mapping[str, Any]],
        grid: Any,
        bounds: Mapping[str, Any] | None = None,
        clip: Mapping[str, Any] | None = None,
        static_fields: Sequence[str] = (),
        n_ensembles: int = 1,
        structure: str = DEFAULT_STRUCTURE,
        lossless: Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
        physics: Any = None,
        version: int = CONFIG_VERSION,
        plan: P.Plan | None = None,
        chain: bool = False,
        **ignored: Any,
    ) -> None:
        extra = set(ignored) - IGNORED_KEYS
        if extra:
            raise UnknownKeys(f"unknown {CODEC_ID} configuration keys: {sorted(extra)}")
        if version != CONFIG_VERSION:
            raise ValueError(f"unsupported {CODEC_ID} configuration version {version} (this is {CONFIG_VERSION})")
        if structure not in STRUCTURES:
            raise ValueError(f"unknown lattice codec structure {structure!r}; known: {list(STRUCTURES)}")
        if not isinstance(chain, bool):
            raise ValueError(f"chain must be true or false, got {chain!r}")
        self._chain = chain
        self._vars: tuple[Variable, ...] = resolve_variables(variables, bounds, clip)
        self._names = tuple(v.name for v in self._vars)
        self._grid = grids.grid_ref(grid)
        self._layout: TriangleLatticeLayout | None = grid if isinstance(grid, TriangleLatticeLayout) else None
        self._n_ens = int(n_ensembles)
        if self._n_ens < 1:
            raise ValueError(f"n_ensembles must be >= 1, got {n_ensembles!r}")
        self._idx = FieldIndex(self._names, self._n_ens)
        self._sid = structure
        self._lossless_cfg = dict(DEFAULT_LOSSLESS if lossless is None else lossless)
        self._lossless_codec = get_codec(dict(self._lossless_cfg))
        self._opt = check_options(options)
        self._physics = phys.parse_physics(physics)
        self._static = tuple(static_fields or ())
        unknown = [n for n in self._static if n not in self._names]
        if unknown:
            raise ValueError(f"static_fields are not variables: {unknown}")
        lossy_static = [v.name for v in self._vars if v.name in self._static and v.lossy]
        if lossy_static:
            raise ValueError(f"static fields are stored exactly and cannot have a bound: {lossy_static}")
        free = set(self._static) if self._opt["statics"] != "off" else set()
        if plan is None:
            candidate = phys.resolve(self._names, self._physics)
        else:
            if not isinstance(plan, P.Plan):
                raise TypeError(f"plan must be a plan.Plan (it is not part of the configuration), got {type(plan)}")
            P.check_names(plan, self._names, RECIPE_IDS.values())
            candidate = copy.deepcopy(plan)  # finalize edits the report
        self._given_plan = None if plan is None else copy.deepcopy(plan)
        self._resolved = P.finalize(candidate, self._names, self._opt["max_depth"], free)
        self._plan = self._resolved.cross
        self._masks = self._resolved.masks
        self._recipes = self._resolved.recipes
        self._deps = P.dependencies(self._plan, self._masks)
        self._structure: Structure | None = None
        self._capture = statics.StaticCapture()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------------------------------- properties
    @property
    def variables(self) -> tuple[Variable, ...]:
        return self._vars

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def grid(self) -> dict[str, Any]:
        return dict(self._grid)

    @property
    def options(self) -> dict[str, Any]:
        """Effective options (defaults filled in)."""
        return dict(self._opt)

    @property
    def static_fields(self) -> tuple[str, ...]:
        return self._static

    @property
    def lossless_config(self) -> dict[str, Any]:
        """The numcodecs configuration of the codec of the losslessly stored fields (configuration key ``lossless``)."""
        return dict(self._lossless_cfg)

    @property
    def layout(self) -> TriangleLatticeLayout:
        """The lattice (loaded from the sidecar search path on first use)."""
        if self._layout is None:
            with self._lock:
                if self._layout is None:
                    self._layout = grids.load_lattice(self._grid)
        return self._layout

    @property
    def structure(self) -> Structure:
        if self._structure is None:
            layout = self.layout
            with self._lock:
                if self._structure is None:
                    self._structure = get_structure(self._sid, layout)
        return self._structure

    @property
    def n_cells(self) -> int:
        return int(self._grid["n_cells"])

    @property
    def fields_per_date(self) -> int:
        return len(self._vars) * self._n_ens

    @property
    def chain(self) -> bool:
        """Whether the codec follows field stages in a chain (configuration key ``chain``)."""
        return self._chain

    def validate_chunk_shape(self, chunks: Sequence[int]) -> None:
        """Raise ``ValueError`` unless a chunk of this shape holds whole dates of (variables, ensembles, cells). In
        chain mode the chunk grid is the array's, before the stages that change the field count: only the cells and
        ensembles of ``(dates, fields, ensembles, cells)`` chunks are checked here (the adapter checks that a chunk
        holds every field of a date, and the field count per chunk when it codes one)."""
        chunks = tuple(int(c) for c in chunks)
        if self._chain:
            if len(chunks) != 4 or chunks[3] != self.n_cells or chunks[2] != self._n_ens:
                raise ValueError(
                    f"chunks {chunks} must be (dates, fields, ensembles, cells) with (ensembles, cells) = "
                    f"({self._n_ens}, {self.n_cells})"
                )
            return
        per = self.fields_per_date * self.n_cells
        if not chunks or chunks[-1] != self.n_cells or int(np.prod(chunks)) % per:
            why = ""
            if len(chunks) == 4 and chunks[3] == self.n_cells and chunks[1] != len(self._vars):
                why = f": the array has {chunks[1]} variables per chunk, the configuration {len(self._vars)}"
            elif chunks and chunks[-1] != self.n_cells:
                why = f": the array has {chunks[-1]} cells per chunk, the grid {self.n_cells}"
            raise ValueError(
                f"chunks {chunks} must hold whole dates of (variables, ensembles, cells) = "
                f"({len(self._vars)}, {self._n_ens}, {self.n_cells}){why}"
            )

    @property
    def cross_plan(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self._plan.items()}

    @property
    def physics(self) -> dict[str, Any]:
        """Effective physics group (defaults filled in)."""
        return dict(self._physics)

    @property
    def plan(self) -> P.Plan:
        """The resolved plan (cross terms, mask terms, recipes; the physics layer's or the given one, after the
        codec's limits) and its report (``plan_report``)."""
        return self._resolved

    def plan_report(self) -> list[dict[str, Any]]:
        """Per candidate term: target, kind, the sources applied and those dropped with the reason. (Whether an
        applied term pays is decided per chunk by the encoder; ``anemoi-datasets-codecs info`` shows what a chunk
        uses.)"""
        return [t.to_dict() for t in self._resolved.report]

    def _encode_options(self) -> EncodeOptions:
        o = self._opt
        return EncodeOptions(
            k=float(o["k"]),
            contexts=int(o["contexts"]),
            fit=o["fit"],
            tables=o["tables"],
            table_penalty=int(o["table_penalty"]),
            rdoq=bool(o["rdoq"]),
            prob_bits=o["prob_bits"],
            recon_bits=int(o["recon_bits"]),
            rounding=o["rounding"],
        )

    def _as_fields(self, buf: Any) -> np.ndarray:
        arr = ensure_contiguous_ndarray(buf)
        if arr.dtype != np.float32:
            if arr.dtype.itemsize != 1:
                raise ValueError(f"the lattice codec encodes float32 data, got {arr.dtype}")
            arr = arr.view(np.float32)
        per_date = self.fields_per_date * self.n_cells
        if arr.size == 0 or arr.size % per_date:
            raise ValueError(
                f"chunk of {arr.size} values is not a multiple of (variables, ensembles, cells) = "
                f"({len(self._vars)}, {self._n_ens}, {self.n_cells})"
            )
        return arr.reshape(-1, self.n_cells)

    # ------------------------------------------------------------------------------------------------ encode
    def encode(self, buf: Any, dates: Any = None, record: Any = None) -> bytes:
        """Encode a chunk. ``dates``: the valid date(s) of the chunk (numpy datetime64 or ISO strings; one per date
        in the chunk), needed for RECIPE fields; zarr never passes it (forcings are then stored losslessly).
        ``record``: for the lab (``anemoi_datasets_codecs.lab.account``), ``record(i, pieces)`` is called for every
        LATTICE coding of field ``i`` (``EncodeOptions.record``, from the field threads); the bytes do not depend on
        it."""
        fields = self._as_fields(buf)
        n_fields = len(fields)
        idx = self._idx
        secs = to_seconds(dates, n_fields // idx.per)
        S = self.structure
        grids.persist_lattice(self.layout)  # capture targets (the store's sidecar) get the lattice (a stat when there)
        opts = self._encode_options()
        modes = [MODE_LOSSLESS] * n_fields
        fallbacks = [FALLBACK_NONE] * n_fields
        payloads: list[bytes] = [b""] * n_fields
        shas: dict[int, str] = {}
        static_names = set(self._static)
        for i, x in enumerate(fields):  # finiteness first: never encode garbage
            lo, hi = x.min(), x.max()
            if not (np.isfinite(lo) and np.isfinite(hi)):
                raise ValueError(f"non-finite values in {idx.describe(i)}")

        def lossless(i: int) -> bytes:
            return bytes(ensure_contiguous_ndarray(self._lossless_codec.encode(fields[i])))

        # phase 1: constant / static / lossless (recipes need the statics, so they come after)
        recipe_fields = []
        for i, x in enumerate(fields):
            var = self._vars[idx.variable(i)]
            name = var.name
            bits = x.view(np.uint32)
            if x[0] == x.min() == x.max() and (bits == bits[0]).all():
                modes[i], payloads[i] = MODE_CONSTANT, x[:1].astype("<f4").tobytes()
                continue
            if var.lossy:
                continue
            if name in static_names and self._opt["statics"] != "off":
                sha = statics.sha256_of(x)
                if self._opt["statics"] == "inline":
                    modes[i], payloads[i] = MODE_STATIC_REF, bytes.fromhex(sha) + statics.inline_payload(x)
                    statics.add_static(x)
                elif self._capture.ok(name, sha, x):
                    modes[i], payloads[i] = MODE_STATIC_REF, bytes.fromhex(sha)
                if modes[i] == MODE_STATIC_REF:
                    shas[i] = sha
                    continue
            if name in self._recipes:
                recipe_fields.append(i)
                continue
            modes[i], payloads[i] = MODE_LOSSLESS, lossless(i)
        bound, fit = self._physics["recipe_bound"], self._physics["recipe_fit"]
        for i in recipe_fields:
            spec = self._recipes[idx.name(i)]
            modes[i], payloads[i] = encode_recipe(fields, i, spec, secs[i // idx.per], idx, modes, bound, fit)
            if modes[i] == MODE_LOSSLESS:
                payloads[i] = lossless(i)

        # phase 2: lossy fields in DAG waves (sources first)
        details_of: dict[int, np.ndarray] = {}
        decoded: dict[int, np.ndarray] = {}  # decoded floats of lattice fields that feed mask terms
        lossy = [i for i in range(n_fields) if modes[i] != MODE_CONSTANT and self._vars[idx.variable(i)].lossy]
        dep = P.depth(self._names, self._deps)
        users = {s for srcs in self._plan.values() for s in srcs}
        mask_users = {s for t, (m, _, f) in self._masks.items() for s in [m, *[a for a, _ in f], *[b for _, b in f]]}

        def encode_one(i: int) -> tuple[int, int, bytes, np.ndarray | None]:
            var = self._vars[idx.variable(i)]
            name = var.name
            cross = []
            for src in self._plan.get(name, []):
                j = idx.sibling(i, src)
                if modes[j] == MODE_LATTICE and j in details_of:
                    cross.append((j, details_of[j]))
                elif modes[j] == MODE_STATIC_REF:
                    cross.append((j, details.static_detail(shas[j], fields[j], S)))
            assert var.max_error is not None
            plain = lossless(i)
            mask = mask_term(fields, i, self._masks[name], idx, modes, decoded) if name in self._masks else None
            fopts = opts if record is None else dataclasses.replace(opts, record=functools.partial(record, i))
            try:
                enc = encode_field(
                    fields[i], var.max_error, var.clip, S, fopts, cross, want_detail=name in users, mask=mask
                )
            except EncodeError as e:
                warnings.warn(f"lattice codec failed on {idx.describe(i)}, stored losslessly: {e}", stacklevel=2)
                return MODE_LOSSLESS, FALLBACK_FAILED, plain, None
            if len(plain) < len(enc.payload):
                return MODE_LOSSLESS, FALLBACK_SMALLER, plain, None
            if name in mask_users:
                decoded[i] = enc.decoded
            return MODE_LATTICE, FALLBACK_NONE, enc.payload, enc.detail

        with blas_serial():  # the fits are small: BLAS threads only cost (threads.py)
            for level in sorted({dep[idx.name(i)] for i in lossy}):
                wave = [i for i in lossy if dep[idx.name(i)] == level]
                results = map_threads(lambda k, wave=wave: encode_one(wave[k]), len(wave))
                for i, (m, fb, p, d) in zip(wave, results, strict=True):
                    modes[i], fallbacks[i], payloads[i] = m, fb, p
                    if d is not None:
                        details_of[i] = d
        return container.pack(modes, fallbacks, payloads, secs, S.sha16)

    # ------------------------------------------------------------------------------------------------ decode
    def decode(self, buf: Any, out: Any = None) -> Any:
        """Decode a chunk into ``out`` (any shape / bytes-like) or a new flat float32 array.

        Without ``out`` the size of the new array comes from the chunk header (fields x cells): a caller that knows
        the chunk shape (the Zarr adapters, the tools) passes ``out``, so that a corrupt header is rejected
        (``ValueError``) before anything is allocated from it."""
        ch = container.parse(buf)
        S = self.structure
        if ch.structure_sha != S.sha16:
            raise ValueError("lattice codec chunk was written with a different structure (sha mismatch)")
        n_fields, n_cells, per = len(ch.modes), self.n_cells, self.fields_per_date
        if n_fields % per:
            raise ValueError(f"chunk has {n_fields} fields, not a multiple of {per}")
        if len(ch.dates) * per != n_fields:
            raise ValueError(f"chunk header has {n_fields} fields for {len(ch.dates)} dates of {per} fields")
        if out is None:
            dest = np.empty(n_fields * n_cells, np.float32)
        else:
            dest = ensure_contiguous_ndarray(out)
            if dest.dtype != np.float32:
                dest = dest.view(np.float32)
            if dest.size != n_fields * n_cells:
                raise ValueError(
                    f"lattice codec chunk header has {n_fields} fields ({n_fields * n_cells} values), the chunk "
                    f"holds {dest.size} values: a corrupt chunk or another array's"
                )
        fields = dest.reshape(n_fields, n_cells)
        static_arr: list[np.ndarray | None] = [None] * n_fields
        ext: list[np.ndarray | None] = [None] * n_fields
        shas: dict[int, str] = {}
        for i, m in enumerate(ch.modes):
            p = ch.payloads[i]
            if m == MODE_STATIC_REF:
                sha = bytes(p[:32]).hex()
                static_arr[i] = statics.lookup(sha, p[32:])
                if static_arr[i].size != n_cells:
                    raise ValueError(f"static field {sha[:12]} has {static_arr[i].size} values, not {n_cells}")
                shas[i] = sha
            elif m == MODE_LOSSLESS:
                values = ensure_ndarray(self._lossless_codec.decode(p)).view(np.float32)
                if values.size != n_cells:
                    raise ValueError(f"lossless payload of {self._idx.describe(i)} has {values.size} values")
                fields[i] = values
        for i, m in enumerate(ch.modes):  # static details for cross terms on non-lattice fields
            if m != MODE_LATTICE:
                continue
            for j in container.lattice_header(ch.payloads[i]).cross:
                if j < n_fields and ch.modes[j] == MODE_STATIC_REF and ext[j] is None:
                    ext[j] = details.static_detail(shas[j], static_arr[j], S, decode_order=True)
        sp, ep = native.ptr_array(static_arr), native.ptr_array(ext)
        buf_arr = np.frombuffer(ch.buffer, np.uint8)
        rc = native.lib().lc_decode_chunk(
            native.ptr(buf_arr), len(buf_arr), S.c_struct(), native.ptr(dest), decoder_threads(), sp, ep, None, None
        )
        native.check(rc, "decode")
        return dest if out is None else out

    # ------------------------------------------------------------------------------------------- configuration
    def get_config(self) -> dict[str, Any]:
        """Canonical configuration (small JSON: names, resolved bounds, grid reference, options, physics): the
        ``configuration`` of the Zarr 3 codec."""
        cfg: dict[str, Any] = {
            "version": CONFIG_VERSION,
            "variables": list(self._names),
            "bounds": {v.name: v.max_error for v in self._vars if v.lossy},
        }
        clips = {v.name: list(v.clip) for v in self._vars if v.clip is not None}
        if clips:
            cfg["clip"] = clips
        cfg.update(
            static_fields=list(self._static),
            grid=dict(self._grid),
            n_ensembles=self._n_ens,
            structure=self._sid,
            lossless=dict(self._lossless_cfg),
            # every option and every physics key, defaults included: the defaults may change between releases, and
            # a store must keep encoding as configured (a key missing from a stored configuration takes the default
            # of the reading version)
            options=dict(self._opt),
            physics=dict(self._physics),
        )
        if self._chain:  # only when set: absent means a codec used alone, which never reads an envelope
            cfg["chain"] = True
        return cfg

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> LatticeCodec:
        """The codec of a stored configuration (``get_config()``)."""
        return cls(**config)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, LatticeCodec)
            and other.get_config() == self.get_config()
            and other._given_plan == self._given_plan
        )

    def __hash__(self) -> int:
        return hash((CODEC_ID, self._names, self._grid["sha256"], self._sid))

    def __repr__(self) -> str:
        n_lossy = sum(v.lossy for v in self._vars)
        name = self._grid.get("name") or self._grid.get("uuid") or self._grid["sha256"][:12]
        return (
            f"LatticeCodec(variables={len(self._vars)} ({n_lossy} lossy), grid={name}, structure={self._sid}, "
            f"k={self._opt['k']}, physics={self._physics_label()}, statics={self._opt['statics']}"
            f"{', chain' if self._chain else ''})"
        )

    def _physics_label(self) -> str:
        p = self._physics
        if self._given_plan is not None:
            return "given plan"
        if not p["enabled"]:
            return "off"
        conv = p["convention"] if isinstance(p["convention"], str) else "custom"
        return f"{conv}/cross={p['cross']}"

    def __getstate__(self) -> dict[str, Any]:  # the configuration (locks, lattice and structure are not pickled)
        state = self.get_config()
        if self._given_plan is not None:
            state["plan"] = self._given_plan
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(**state)  # type: ignore[misc]
