"""Table of bench_decode.py rows (JSON lines): per (kernel variant, threads), the best of all rounds and the median
of the per-round medians, over every chunk, relative to AVX2 at the same thread count; and whether every variant /
thread count decoded each chunk to the same bits (exit 1 if not).

    python scripts/bench/summary.py OUT.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict


def main() -> int:
    rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    shas: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        groups[(r["isa"], r["threads"])].append(r)
        shas[r["chunk"]].add(r["decoded_sha256"])
    chunks = sorted(shas)
    print(f"host {', '.join(sorted({r['host'] for r in rows}))}; store {rows[0]['store']}; chunks {', '.join(chunks)}")
    print("| kernels | threads | best ms | median ms (median over rounds) | rounds x chunks |")
    print("|---|---|---|---|---|")
    ref = {t: statistics.median(r["median_ms"] for r in rs) for (isa, t), rs in groups.items() if isa == "avx2"}
    for (isa, t), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0] != "avx2", kv[0][0])):
        best = min(r["best_ms"] for r in rs)
        med = statistics.median(r["median_ms"] for r in rs)
        rel = f" ({med / ref[t]:.2f}x avx2)" if isa != "avx2" and t in ref else ""
        print(f"| {isa} | {t} | {best:.1f} | {med:.1f}{rel} | {len(rs)} |")
    differ = [c for c in chunks if len(shas[c]) > 1]
    print("decoded bits:", f"DIFFER for {differ}" if differ else "identical for every variant and thread count")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
