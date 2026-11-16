"""The lab command group is wired into the command line."""

from __future__ import annotations

import sys

import pytest

from anemoi_datasets_codecs import cli


def test_lab_group_registered(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    assert "research toolkit" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        cli.main(["lab", "--help"])
    assert e.value.code == 0


def test_main_help_does_not_import_the_lab(monkeypatch, capsys):
    for m in [m for m in sys.modules if m.startswith("anemoi_datasets_codecs.lab.")]:
        monkeypatch.delitem(sys.modules, m)
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    assert "research toolkit" in capsys.readouterr().out
    assert "anemoi_datasets_codecs.lab.cli" not in sys.modules
