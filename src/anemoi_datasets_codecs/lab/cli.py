"""The ``lab`` command group: every tool module registers its subcommands with ``add_parsers(sub)``."""

from __future__ import annotations

import argparse
import importlib

from ..cli import LAB_HELP

# one module per tool; each defines ``add_parsers(sub: argparse._SubParsersAction) -> None`` and sets ``fn`` on its
# parsers (``fn(args) -> int``)
MODULES = ("account", "residuals", "minigrid", "ledger", "oracles", "refs", "fuzz")


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Register ``lab`` and its subcommands on the main parser's subparsers."""
    lab = sub.add_parser("lab", help=LAB_HELP)
    lsub = lab.add_subparsers(dest="lab_cmd", required=True)
    for name in MODULES:
        importlib.import_module(f"{__package__}.{name}").add_parsers(lsub)
