"""``StaticRef``: the engine of the ``static_ref`` field stage (grid-agnostic, exact).

Encode: every static field of the chunk (one array per date and member) is replaced by a reference: the sha256 of
its float32 bytes, the rule of ``anemoi_datasets_codecs.statics``. An array not seen yet is captured into the
sidecar on first sight (the registered store's sidecar or a writable ``$ANEMOI_DATASETS_CODECS_PATH`` entry; the
cache only without one; :class:`~anemoi_datasets_codecs.statics.StaticCapture`) and referenced only once it is durably
stored; an
array that cannot be (nowhere to write it, or more than ``statics.MAX_VERSIONS`` different arrays for one name) is
stored inline in the header instead, so the stage always emits ``len(variables) - len(static_fields)`` fields. The
fields are dropped from the array; the header (``header.py``) lists, per slot, the reference or the inline data.

Decode: the other fields go back to their places and each static field is restored from the header (inline) or the
sidecar search path (``sidecar.MissingReference`` naming the searched directories when it is not there), into the
``out`` array the adapter passes: one write of the whole chunk.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

from .. import statics
from ..config import IGNORED_KEYS, UnknownKeys
from ..zarr3 import FieldStage
from . import header as H

__all__ = ["StaticRef"]

F4 = np.dtype("<f4")


def _runs(kept: Sequence[int]) -> list[tuple[int, int, int]]:
    """Maximal runs of consecutive indices: ``(start in x, stop in x, start in y)``."""
    out: list[tuple[int, int, int]] = []
    for j, i in enumerate(kept):
        if out and out[-1][1] == i:
            a, _, ya = out[-1]
            out[-1] = (a, i + 1, ya)
        else:
            out.append((i, i + 1, j))
    return out


class StaticRef(FieldStage):
    """Field stage that replaces the static fields of a chunk by references into the sidecar.

    Args:
        variables: Names of the chunk's fields (chunk order).
        static_fields: The variables that are static (identical in every chunk, or nearly: up to
            ``statics.MAX_VERSIONS`` different arrays per name are referenced, the others stored inline). Stored in
            chunk order.
        inline: Store every static field inline in the header (self-contained chunks, no sidecar needed; the
            equivalent of the lattice codec's ``statics: inline``). Default false; always in the configuration (an
            encoder setting: a stored configuration records it, whatever a later default).

    A stray ``id`` or ``name`` key and the writer stamp ``written_by`` are ignored.
    """

    def __init__(
        self, variables: Sequence[str], static_fields: Sequence[str], inline: bool = False, **ignored: Any
    ) -> None:
        extra = set(ignored) - IGNORED_KEYS
        if extra:
            raise UnknownKeys(f"unknown static_ref configuration keys: {sorted(extra)}")
        super().__init__(variables)
        if isinstance(static_fields, str) or not isinstance(static_fields, Sequence) or not static_fields:
            raise ValueError("static_fields must be a non-empty list of variable names")
        given = list(static_fields)
        if len(set(given)) != len(given):
            raise ValueError(f"duplicate static_fields: {sorted({n for n in given if given.count(n) > 1})}")
        unknown = [n for n in given if n not in self.variables]
        if unknown:
            raise ValueError(f"static_fields are not variables: {unknown}")
        if not isinstance(inline, bool):
            raise ValueError(f"inline must be true or false, got {inline!r}")
        st = set(given)
        self._static = tuple(n for n in self.variables if n in st)
        self._s_idx = [i for i, n in enumerate(self.variables) if n in st]
        self._k_idx = [i for i, n in enumerate(self.variables) if n not in st]
        self._runs = _runs(self._k_idx)
        self._inline = inline
        self._capture = statics.StaticCapture("stored inline in the static_ref header")
        self._lock = threading.Lock()

    @property
    def static_fields(self) -> tuple[str, ...]:
        """The static fields, in chunk order."""
        return self._static

    @property
    def inline(self) -> bool:
        return self._inline

    @property
    def variables_out(self) -> tuple[str, ...]:
        return tuple(self.variables[i] for i in self._k_idx)

    def get_config(self) -> dict[str, Any]:
        """The configuration: ``variables``, ``static_fields`` and ``inline`` (always: an encoder setting)."""
        return {**super().get_config(), "static_fields": list(self._static), "inline": self._inline}

    def __getstate__(self) -> dict[str, Any]:  # the configuration (the lock and the capture state are not pickled)
        return self.get_config()

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(**state)  # type: ignore[misc]

    # ------------------------------------------------------------------------------------------------ encode
    def encode(self, x: np.ndarray) -> tuple[np.ndarray, bytes]:
        """The chunk without its static fields, and the header with one slot per (date, member, static field):
        a reference, a repeat of an earlier slot or the inline array (module docstring)."""
        n_dates, _, n_ens, _ = x.shape
        slots: list[H.Slot] = []
        first: dict[str, int] = {}  # sha -> its first slot in this chunk
        for d in range(n_dates):
            for name, i in zip(self._static, self._s_idx, strict=True):
                for e in range(n_ens):
                    v = x[d, i, e]
                    sha = statics.sha256_of(v)
                    if sha in first:
                        slots.append(H.Slot(H.REPEAT, sha, source=first[sha]))
                        continue
                    first[sha] = len(slots)
                    if not self._inline:
                        with self._lock:
                            ok = self._capture.ok(name, sha, v)
                        if ok:
                            slots.append(H.Slot(H.REFERENCE, sha))
                            continue
                    slots.append(H.Slot(H.INLINE, sha, statics.static_file_bytes(v)))
        return x[:, self._k_idx], H.pack(len(self._static), slots)

    # ------------------------------------------------------------------------------------------------ decode
    def decode(
        self, y: np.ndarray, header: bytes, out_shape: tuple[int, ...], out: np.ndarray | None = None
    ) -> np.ndarray:
        """The full chunk: the fields of ``y`` back in their places and every static field restored from its slot,
        into ``out`` (raises ``sidecar.MissingReference`` when a referenced static is not on the search path)."""
        n_static, slots = H.parse(header)
        n_dates, _, n_ens, n_cells = out_shape
        if n_static != len(self._static) or len(slots) != n_dates * n_static * n_ens:
            raise ValueError(
                f"static_ref header holds {len(slots)} slots of {n_static} static fields, the configuration has "
                f"{len(self._static)} static fields and the chunk {n_dates} dates x {n_ens} members"
            )
        x = np.empty(out_shape, F4) if out is None else out
        for a, b, ya in self._runs:
            x[:, a:b] = y[:, ya : ya + b - a]
        arrays: list[np.ndarray] = []
        for k, s in enumerate(slots):
            if s.kind == H.REPEAT:
                v = arrays[s.source]
            else:
                v = statics.lookup(s.sha, s.data if s.kind == H.INLINE else None)
                if v.size != n_cells:
                    raise ValueError(f"static field {s.sha[:12]} has {v.size} values, the chunk {n_cells} cells")
            arrays.append(v)
            d, rest = divmod(k, n_static * n_ens)
            i, e = divmod(rest, n_ens)
            x[d, self._s_idx[i], e] = v
        return x

    # ------------------------------------------------------------------------------------------------ tools
    @staticmethod
    def header_references(header: bytes) -> dict[str, set[str]]:
        """The sidecar files a chunk with this header needs: ``{"statics": {sha256, ...}}`` (its REFERENCE slots)."""
        _, slots = H.parse(header)
        return {"statics": {s.sha for s in slots if s.kind == H.REFERENCE}}

    @staticmethod
    def describe_header(
        header: bytes, configuration: dict[str, Any] | None = None, shape: tuple[int, ...] | None = None
    ) -> dict[str, Any]:
        """A summary of the header for ``anemoi-datasets-codecs info``: version, static fields, slots per kind, bytes
        of inline data, and per slot ``{field, kind, sha256}`` (``field``: the static field's name, with ``[member]``
        when there are several members, given the stage's ``configuration`` and the envelope ``shape``; else the
        slot index)."""
        n_static, slots = H.parse(header)
        out: dict[str, Any] = {"version": H.VERSION, "static_fields": n_static, "slots": len(slots)}
        for kind, label in H.KIND_LABELS.items():
            out[label] = sum(s.kind == kind for s in slots)
        out["inline_bytes"] = sum(len(s.data) for s in slots)
        names = None
        n_ens = shape[2] if shape is not None and len(shape) == 4 else 1
        if configuration is not None and len(configuration.get("static_fields", ())) == n_static:
            names = list(StaticRef(**configuration).static_fields)
        per_slot = []
        for k, s in enumerate(slots):
            field: Any = k
            if names and n_static:
                i, e = divmod(k % (n_static * n_ens), n_ens)
                field = names[i] + (f"[{e}]" if n_ens > 1 else "")
            per_slot.append({"field": field, "kind": H.KIND_LABELS[s.kind], "sha256": s.sha})
        out["per_slot"] = per_slot
        return out
