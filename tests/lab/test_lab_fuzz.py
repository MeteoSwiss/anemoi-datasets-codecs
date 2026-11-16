"""lab fuzz: mutation determinism, the worker pool's classification of crashes, hangs, sanitizer reports and errors,
a short run of the real fuzzer on the synthetic chunk (plain and chain: no crash, no hang), and the parser fuzzer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from anemoi_datasets_codecs.lab import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/conftest.py, for the synthetic chunk


def _run(spec, scopes, base, n, out, seed=1, jobs=2, timeout=20.0):
    return fuzz.run_campaign(
        spec, lambda i: fuzz.make_mutant(scopes, seed, i), n, out, timeout=timeout, jobs=jobs, base=base
    )


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    work = tmp_path_factory.mktemp("fuzz-base")
    out = {}
    for chain in (False, True):
        spec = fuzz.make_synthetic(work / ("chain" if chain else "plain"), chain=chain)
        base = Path(spec["chunk"]).read_bytes()
        out[chain] = (spec, base, fuzz.scopes_of(base, chain))
    return out


@pytest.mark.parametrize("chain", [False, True])
def test_mutants_are_deterministic_and_differ(synthetic, chain):
    _, base, scopes = synthetic[chain]
    a = [fuzz.make_mutant(scopes, 7, i) for i in range(60)]
    assert a == [fuzz.make_mutant(scopes, 7, i) for i in range(60)]
    assert a != [fuzz.make_mutant(scopes, 8, i) for i in range(60)]
    assert sum(m[0] != base for m in a) > 45
    assert len({m[1] for m in a}) > 10  # several kinds


def test_chain_scopes_rewrap_a_consistent_envelope(synthetic):
    from anemoi_datasets_codecs import envelope

    _, base, scopes = synthetic[True]
    assert [s.name for s in scopes] == ["envelope", "payload", "hdr:field_terms", "hdr:static_ref"]
    for s in scopes[1:]:
        assert s.wrap(s.buf) == base  # unchanged inner buffer: the same chunk
        grown = s.wrap(s.buf + b"\0" * 8)
        assert envelope.parse(grown) is not None and len(grown) == len(base) + 8


@pytest.mark.parametrize("chain", [False, True])
def test_fuzz_synthetic_no_crash_no_hang(synthetic, tmp_path, chain):
    spec, base, scopes = synthetic[chain]
    camp = _run(spec, scopes, base, 250, tmp_path)
    tot = camp.totals()
    assert sum(tot.values()) == 250
    for bad in ("crash", "hang", "crash_unconfirmed", "hang_unconfirmed", "sanitizer"):
        assert tot[bad] == 0, (bad, camp.saved[:3])
    assert tot["clean_error"] > 50  # the checks fire
    assert "TOTAL" in fuzz.summary_table(camp)


def test_pool_classifies_crash_hang_sanitizer_and_errors(tmp_path):
    spec = {"selftest": True}
    cases = {
        0: (b"\x01", "abort"),
        1: (b"\x02", "hang"),
        2: (b"\x03", "error"),
        3: (b"\x04", "other"),
        4: (b"\x05", "asan"),
        5: (b"\x06", "diff"),
        6: (b"\x00", "same"),
    }
    camp = fuzz.run_campaign(spec, lambda i: (cases[i][0], cases[i][1], "toy"), len(cases), tmp_path, timeout=1, jobs=2)
    got = {k[0]: k[1] for k in camp.counts}
    assert got == {
        "abort": "crash",
        "hang": "hang",
        "error": "clean_error",
        "other": "error_other",
        "asan": "sanitizer",
        "diff": "accepted_diff",
        "same": "accepted_same",
    }
    meta = json.loads(next((tmp_path / "crash").glob("*.json")).read_text())
    assert "SIGABRT" in meta["detail"] and meta["mutation"] == "toy"
    assert (tmp_path / "hang").is_dir() and (tmp_path / "sanitizer").is_dir()
    assert "AddressSanitizer" in json.loads(next((tmp_path / "sanitizer").glob("*.json")).read_text())["stderr"]


def test_fuzz_parsers_only_raise_value_error():
    res = fuzz.fuzz_parsers(1500, seed=3)
    assert set(res) >= {"envelope.parse", "static_ref.header.parse", "field_terms.header.parse", "container.parse"}
    for name, r in res.items():
        assert r["findings"] == [], (name, r["findings"][:1])
        assert r["rejected"] > 0 and r["cases"] == 1500
