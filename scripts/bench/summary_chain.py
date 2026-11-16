"""Tables of ``bench_chain.py`` output: size and write time per store, and zarr read time per store and thread count
(best of the rounds, median of the medians); checks that every thread count decoded each store to the same bits.

    python scripts/bench/summary_chain.py OUT.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict


def main() -> None:
    rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    writes = [r for r in rows if "write_s" in r]
    reads = [r for r in rows if "best_ms" in r]
    print("| store | MiB | write s | worst error / bound |\n|---|---|---|---|")
    for r in writes:
        print(f"| {r['store']} | {r['MiB']:.3f} | {r['write_s']} | {r['worst_err_over_bound']} |")
    by = defaultdict(list)
    for r in reads:
        by[(r["store"], r["threads"])].append(r)
    print("\n| store | threads | best ms | median ms |\n|---|---|---|---|")
    bad = []
    for (store, t), rs in sorted(by.items()):
        best = min(r["best_ms"] for r in rs)
        med = statistics.median(r["median_ms"] for r in rs)
        print(f"| {store} | {t} | {best:.1f} | {med:.1f} |")
    for store in {s for s, _ in by}:
        shas = {r["sha256"] for (s, _), rs in by.items() if s == store for r in rs}
        if len(shas) != 1:
            bad.append(store)
    print("\ndecoded bits identical across thread counts:", "yes" if not bad else f"NO for {bad}")


if __name__ == "__main__":
    main()
