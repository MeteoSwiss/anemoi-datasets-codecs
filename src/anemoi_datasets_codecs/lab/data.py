"""Shared input side of the tools: stores, configurations, dates, field labels, and the environment of a run."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..cli import lattice_config, load_config
from ..icon_lattice.codec import LatticeCodec
from ..icon_lattice.fields import FieldIndex


def load_lattice_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """The lattice codec configuration of a store, a JSON / YAML configuration file, a codec entry or a chain."""
    return lattice_config(load_config(Path(path)))


def codec_for(config: dict[str, Any] | str | os.PathLike[str], **overrides: Any) -> LatticeCodec:
    """``LatticeCodec`` of a configuration (dict or path) with top-level overrides (``structure``, ``options``,
    ``physics`` ...); ``options`` and ``physics`` overrides are merged into the configuration's."""
    cfg = dict(config) if isinstance(config, dict) else load_lattice_config(config)
    for key, value in overrides.items():
        if key in ("options", "physics") and isinstance(cfg.get(key), dict) and isinstance(value, dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return LatticeCodec.from_config(cfg)


def config_sha256(config: dict[str, Any]) -> str:
    """sha256 of a configuration (pretty JSON, sorted keys)."""
    return hashlib.sha256(json.dumps(config, indent=1, sort_keys=True).encode()).hexdigest()


def open_data(store: str | os.PathLike[str], array: str = "data") -> tuple[Any, np.ndarray | None]:
    """The data array of a store and its dates (``datetime64[s]``; None without a ``dates`` array)."""
    import zarr

    g = zarr.open_group(str(store), mode="r")
    dates = g["dates"][:].astype("datetime64[s]") if "dates" in g else None
    return g[array], dates


def read_date(data: Any, i: int) -> np.ndarray:
    """The fields of date ``i`` of an anemoi array ``(dates, variables, ensembles, cells)`` as float32
    ``(variables * ensembles, cells)`` in chunk order."""
    x = np.asarray(data[i])
    return np.ascontiguousarray(x.reshape(-1, x.shape[-1]), np.float32)


def field_index(codec: LatticeCodec) -> FieldIndex:
    """Field indices of one date of ``codec``'s chunk."""
    return FieldIndex(list(codec.names), int(codec.get_config().get("n_ensembles", 1)))


def field_labels(codec: LatticeCodec, n_dates: int = 1) -> list[str]:
    """Labels of every field of a chunk with ``n_dates`` dates (``name`` or ``name[e]``, with ``@d`` after the first
    date)."""
    idx = field_index(codec)
    return [idx.label(i) for i in range(idx.per * n_dates)]


def lossy_fields(codec: LatticeCodec) -> list[tuple[int, str, float, tuple | None]]:
    """``(field index within a date, name, bound, clip)`` of every lossy variable (``n_ensembles`` 1)."""
    idx = field_index(codec)
    out = []
    for i in range(idx.per):
        v = codec.variables[idx.variable(i)]
        if v.lossy:
            out.append((i, v.name, float(v.max_error), v.clip))
    return out


def git_sha(path: str | os.PathLike[str] | None = None) -> str | None:
    """The git commit of ``path`` (default: this package's checkout), ``+dirty`` when modified; None outside git."""
    cwd = Path(path) if path is not None else Path(__file__).resolve().parent
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha.stdout.strip()[:12] + ("+dirty" if dirty.stdout.strip() else "")


def environment() -> dict[str, Any]:
    """What a measurement depends on: package and numpy versions, kernel variant, threads, host, git commit."""
    from ..icon_lattice import native, threads

    return {
        "package": __version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "isa": native.isa(),
        "threads": threads.default_threads(),
        "host": platform.node(),
        "git": git_sha(),
    }
