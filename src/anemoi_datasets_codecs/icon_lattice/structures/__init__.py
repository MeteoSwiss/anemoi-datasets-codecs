"""The frozen structure builders, by id: ``(cell image) -> Structure`` (``structure.cell_image``: native cell index
per lattice pixel, -1 on padding). Every chunk stores the sha of its structure, so a builder's output never changes;
a new structure gets a new id (here and in ``config.STRUCTURES``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..structure import Structure
from .hc2 import build_hc2_k3
from .tree import build_tree

STRUCTURES: dict[str, Callable[[np.ndarray], Structure]] = {
    "tree": build_tree,
    "tree_b": lambda cimg: build_tree(cimg, bpass=True),
    "hc2_k3": build_hc2_k3,
}

__all__ = ["STRUCTURES", "build_hc2_k3", "build_tree"]
