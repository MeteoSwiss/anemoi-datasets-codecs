"""Does a decode with several threads work in a process forked after the parent decoded with several threads?
(GNU libgomp's thread pool does not survive fork.) Prints the child's outcome.

    python scripts/fork_check.py [CHILD_THREADS]
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import conftest  # noqa: E402

from anemoi_datasets_codecs.icon_lattice import threads  # noqa: E402


def child(q, codec, enc, n_threads):
    if n_threads:
        os.environ["ANEMOI_DATASETS_CODECS_THREADS"] = n_threads
    else:
        os.environ.pop("ANEMOI_DATASETS_CODECS_THREADS", None)
    q.put((threads.decoder_threads(), float(codec.decode(enc)[:10].sum())))


def main() -> None:
    lattice = conftest.make_lattice()
    chunk = conftest.make_chunk(lattice)
    codec = conftest.make_codec(lattice[0])
    enc = codec.encode(chunk, dates=[conftest.DATE])
    os.environ["ANEMOI_DATASETS_CODECS_THREADS"] = "8"
    ref = float(codec.decode(enc)[:10].sum())  # the parent's OpenMP pool now exists
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=child, args=(q, codec, enc, sys.argv[1] if len(sys.argv) > 1 else ""))
    p.start()
    p.join(30)
    if p.is_alive():
        p.kill()
        print("child HUNG (killed after 30 s)")
    else:
        t, got = q.get()
        print(f"child decoded with {t} threads: {'OK' if got == ref else 'WRONG'} (exit {p.exitcode})")


if __name__ == "__main__":
    main()
