"""The entropy-coding helpers of the LATTICE encoder against their reference formulas (the earlier, slower code):
zigzag symbols, the order-0 estimate, LEB128, serialised tables, context thresholds and the table layout. The
encoder output depends on every count, threshold and cost, so they must agree exactly."""

from __future__ import annotations

import numpy as np
import pytest

from anemoi_datasets_codecs.icon_lattice import encoder as E


# ----------------------------------------------------------------------------------------------- references
def ref_leb128(values):
    out = bytearray()
    for x in values:
        while True:
            b = x & 127
            x >>= 7
            out.append(b | (128 if x else 0))
            if not x:
                break
    return bytes(out)


def ref_symbols(r):
    z = np.where(r >= 0, 2 * r, -2 * r - 1)
    return np.minimum(z, 255).astype(np.uint8), r[z >= 255]


def ref_entropy_bits(r):
    if len(r) == 0:
        return 0.0
    sym, esc = ref_symbols(r)
    cnt = np.bincount(sym, minlength=256).astype(np.float64)
    p = cnt[cnt > 0] / len(r)
    return float(-(cnt[cnt > 0] * np.log2(p)).sum() + 16 * len(esc))


def ref_table_bytes(freqs):
    nz = np.flatnonzero(freqs[:255])
    m = int(nz[-1]) + 1 if len(nz) else 0
    return bytes([m]) + ref_leb128([*freqs[:m].tolist(), int(freqs[255])])


def ref_thresholds(spread, nb):
    if nb <= 1 or len(spread) == 0:
        return np.zeros(0, np.int64)
    q = np.quantile(spread, np.arange(1, nb) / nb, method="lower")
    return np.unique(q.astype(np.int64))


def ref_choose_tables(steps_code, opts, bits=12):
    nb = max(1, min(8, int(opts.contexts)))
    modes = {"none": ["none"], "shared": ["shared"], "per_step": ["per_step"]}.get(opts.tables)
    if modes is None:
        modes = ["none", "per_step", "shared"]
    if nb == 1:
        modes = ["none"]
    gather = [sc.spread for sc in steps_code if sc.spread is not None]
    gthr = ref_thresholds(np.concatenate(gather), nb) if gather else np.zeros(0, np.int64)
    candidates = []
    for mode in modes:
        n_ctx, thr, buckets, keys = [], [], [], []
        for s, sc in enumerate(steps_code):
            if sc.spread is None or mode == "none":
                th = np.zeros(0, np.int64)
            else:
                th = gthr if mode == "shared" else ref_thresholds(sc.spread, nb)
            n_ctx.append(len(th) + 1)
            thr.append(th)
            buckets.append(
                np.searchsorted(th, sc.spread, side="left").astype(np.int64)
                if len(th)
                else np.zeros(len(sc.sym), np.int64)
            )
            if mode == "shared" and sc.spread is not None:
                keys.append([("g", b) for b in range(len(th) + 1)])
            else:
                keys.append([("s", s, b) for b in range(len(th) + 1)])
        hist = {}
        for s, sc in enumerate(steps_code):
            free = ~sc.masked if sc.masked is not None else np.ones(len(sc.sym), bool)
            for b, key in enumerate(keys[s]):
                h = np.bincount(sc.sym[(buckets[s] == b) & free], minlength=256)
                hist[key] = hist[key] + h if key in hist else h
            if sc.masked is not None:
                h = np.bincount(sc.sym[sc.masked], minlength=256)
                hist[("m",)] = hist[("m",)] + h if ("m",) in hist else h
        live = [key for key, h in hist.items() if h.sum() > 0]
        if len(live) > 255:
            continue
        ids = {key: i for i, key in enumerate(live)}
        tmap = [[ids.get(key, 0) for key in keys[s]] for s in range(len(steps_code))]
        tid_steps = [np.asarray(tmap[s], np.int64)[buckets[s]] for s in range(len(steps_code))]
        mtab = ids.get(("m",), 0)
        for s, sc in enumerate(steps_code):
            if sc.masked is not None:
                tid_steps[s][sc.masked] = mtab
        coded, freqs, blob = 0.0, [], bytearray()
        for key in live:
            b, f = E._coded_bits(hist[key].astype(np.uint32), bits)
            coded += b
            freqs.append(f)
            blob += ref_table_bytes(f)
        cost = coded / 8 + len(blob) + opts.table_penalty * len(live) + 4 * sum(len(t) for t in thr)
        candidates.append((cost, n_ctx, thr, tmap, tid_steps, np.stack(freqs), bytes(blob), mtab))
    return min(candidates, key=lambda c: c[0])


def ref_cost_layout(steps_code, opts, bits, n_ctx, thr, tmap, tid_steps, mtab):
    n_tab = 1 + max(int(t.max()) if len(t) else 0 for t in tid_steps)
    hists = np.zeros((n_tab, 256), np.int64)
    for sc, tid in zip(steps_code, tid_steps, strict=True):
        np.add.at(hists, (tid.astype(np.int64), sc.sym), 1)
    coded, freqs, blob = 0.0, [], bytearray()
    for h in hists:
        b, f = E._coded_bits(h.astype(np.uint32), bits)
        coded += b
        freqs.append(f)
        blob += ref_table_bytes(f)
    cost = coded / 8 + len(blob) + opts.table_penalty * n_tab + 4 * sum(len(t) for t in thr)
    return (cost, n_ctx, thr, tmap, tid_steps, np.stack(freqs), bytes(blob), mtab)


# ----------------------------------------------------------------------------------------------- tests
def _residuals(rng, n):
    """Laplacian-like residuals with escapes, zeros and both signs."""
    r = np.rint(rng.laplace(0, rng.uniform(0.3, 80), n)).astype(np.int64)
    r[rng.integers(0, n, max(1, n // 50))] = rng.integers(-(2**31), 2**31, max(1, n // 50))
    return r


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16, np.int8])
def test_zigzag_full_range(dtype):
    info = np.iinfo(dtype)
    rng = np.random.default_rng(0)
    r = np.concatenate(
        [
            np.array([0, 1, -1, 127, -127, 128, -128, 254, -255, info.max, info.min, info.max // 2, info.min // 2]),
            rng.integers(info.min, info.max, 10_000, endpoint=True),
        ]
    ).astype(dtype)
    with np.errstate(over="ignore"):
        assert np.array_equal(E._zigzag(r), np.where(r >= 0, 2 * r, -2 * r - 1))
        assert E._zigzag(r).dtype == dtype


def test_symbols_and_entropy_bits():
    rng = np.random.default_rng(1)
    edges = np.array([0, 1, -1, 126, -126, 127, -127, 128, -128, 129, -129, 2**40, -(2**40), 2**61, -(2**61)])
    for n in (0, 1, 2, 7, 1000, 50_000, -1):
        r = edges if n < 0 else _residuals(rng, n) if n else np.zeros(0, np.int64)
        for dtype in (np.int64, np.int32, np.int16):
            rr = (
                r if dtype is np.int64 else np.clip(r, np.iinfo(dtype).min // 2, np.iinfo(dtype).max // 2).astype(dtype)
            )
            sym, esc = E._symbols(rr)
            rsym, resc = ref_symbols(rr)
            assert sym.dtype == rsym.dtype and esc.dtype == resc.dtype
            assert np.array_equal(sym, rsym) and np.array_equal(esc, resc)
            assert E._entropy_bits(rr) == ref_entropy_bits(rr)
    assert E._entropy_bits(np.full(10, 3, np.int64)) == ref_entropy_bits(np.full(10, 3, np.int64))
    assert E._entropy_bits(np.full(10, -300, np.int64)) == ref_entropy_bits(np.full(10, -300, np.int64))


def test_varints_and_table_bytes():
    rng = np.random.default_rng(2)
    edges = [0, 1, 127, 128, 16383, 16384, 2**21 - 1, 2**21, 2**28, 2**35, 2**49, 2**56 - 1, 2**63 - 1]
    for v in (np.zeros(0, np.int64), np.array(edges, np.int64), rng.integers(0, 2**40, 5000), rng.integers(0, 300, 99)):
        assert E._varints(v) == ref_leb128(v.tolist())
    for _ in range(50):
        f = np.zeros(256, np.uint32)
        k = int(rng.integers(0, 256))
        f[rng.integers(0, 256, k)] = rng.integers(0, 4097, k)
        assert E._table_bytes(f) == ref_table_bytes(f)
    assert E._table_bytes(np.zeros(256, np.uint32)) == ref_table_bytes(np.zeros(256, np.uint32))
    tables = np.zeros((40, 256), np.uint32)
    for i, f in enumerate(tables[1:]):  # tables[0] empty
        k = int(rng.integers(1, 256))
        f[rng.integers(0, 256, k)] = rng.integers(0, 2**32, k) >> rng.integers(0, 32, k)
        f[254 if i % 3 == 0 else 255] = 2**31 + i
    assert E._tables_len(tables) == sum(len(ref_table_bytes(f)) for f in tables)


def test_thresholds_counted_equal_quantile():
    rng = np.random.default_rng(3)
    cases = [
        np.zeros(1000, np.int64),
        np.full(17, 5, np.int64),
        np.array([3], np.int64),
        np.array([0, 9, 2], np.int64),  # n < nb
        rng.integers(0, 4, 999),  # ties
        rng.geometric(0.05, 123_457).astype(np.int64) - 1,
        rng.integers(0, 2**30, 5000),  # large range: np.quantile fallback
        np.concatenate([rng.integers(0, 10, 4000), [2**27]]),  # one outlier: fallback
        np.zeros(0, np.int64),
    ]
    for spread in cases:
        for nb in (1, 2, 3, 5, 8):
            assert np.array_equal(E._thresholds(spread, nb), ref_thresholds(spread, nb)), (spread[:5], nb)
    for n in range(1, 40):  # every rank rule near small n
        spread = rng.integers(0, 6, n)
        for nb in range(2, 9):
            assert np.array_equal(E._thresholds(spread, nb), ref_thresholds(spread, nb))
    parts = [rng.integers(0, m, size) for m, size in ((5, 300), (60, 7000), (1, 3), (900, 20_000))]
    total = np.zeros(max(int(p.max()) + 1 for p in parts), np.int64)
    for p in parts:
        total[: int(p.max()) + 1] += np.bincount(p)
    allp = np.concatenate(parts)
    assert np.array_equal(E._thresholds_counted(total, len(allp), 8), ref_thresholds(allp, 8))


def _steps(rng, masked=True, contexts=True, wide=False):
    """Synthetic coded steps: a delta step, gather steps with small spreads (or, ``wide``, some with spreads too
    wide to count, and one with only zero spreads), masked targets on two of them and one fully masked step."""
    steps = []
    for k, n in enumerate((300, 1, 5000, 20_000, 64, 12_000)):
        r = _residuals(rng, n)
        sym, esc = ref_symbols(r)
        spread = None if (k == 0 or not contexts) else rng.geometric(0.1, n).astype(np.int64) - 1
        if wide and spread is not None and k in (2, 3):
            spread = rng.integers(0, 2**24, n) if k == 2 else np.where(rng.random(n) < 0.5, spread, 2**21)
        if wide and spread is not None and k == 4:
            spread = np.zeros(n, np.int64)
        msk = rng.random(n) < 0.2 if masked and k in (3, 5) else None
        steps.append(E._StepCode(sym, esc, spread, np.zeros(1, np.int64), np.zeros(0, np.int64), msk))
    if masked:  # a step whose targets are all masked
        r = _residuals(rng, 500)
        sym, esc = ref_symbols(r)
        sp = rng.integers(0, 3, 500)
        steps.append(E._StepCode(sym, esc, sp, np.zeros(1, np.int64), np.zeros(0, np.int64), np.ones(500, bool)))
    return steps


@pytest.mark.parametrize("tables", ["auto", "none", "shared", "per_step"])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("contexts", [1, 4, 8])
@pytest.mark.parametrize("wide", [False, True])
def test_choose_tables_equal_reference(tables, masked, contexts, wide):
    rng = np.random.default_rng([["auto", "none", "shared", "per_step"].index(tables), masked, contexts, wide])
    opts = E.EncodeOptions(tables=tables, contexts=contexts)
    steps = _steps(rng, masked, contexts > 1, wide)
    new = E._choose_tables(steps, opts, 12)
    ref = ref_choose_tables(steps, opts, 12)
    assert new[0] == ref[0]
    assert new[1] == ref[1] and new[3] == ref[3] and new[6] == ref[6] and new[7] == ref[7]
    assert all(np.array_equal(a, b) for a, b in zip(new[2], ref[2], strict=True))
    assert all(np.array_equal(a, b) for a, b in zip(new[4], ref[4], strict=True))
    assert np.array_equal(new[5], ref[5])
    for bits in (10, 12):
        a = E._choose_tables(steps, opts, bits, fixed=new)
        b = ref_cost_layout(steps, opts, bits, *ref[1:5], ref[7])
        assert a[0] == b[0] and a[6] == b[6] and np.array_equal(a[5], b[5])
        c = E._cost_layout(steps, opts, bits, *new[1:5], new[7])  # histograms counted from the table ids
        assert c[0] == b[0] and c[6] == b[6] and np.array_equal(c[8], a[8])
