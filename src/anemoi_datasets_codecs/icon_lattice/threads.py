"""Thread count of the codec (decoder OpenMP team, encoder field pool).

``default_threads()`` is read at every encode / decode call:

1. ``$ANEMOI_DATASETS_CODECS_THREADS`` when set (an integer >= 1; 1 = serial);
2. 1 inside a torch ``DataLoader`` worker process (``torch.utils.data.get_worker_info()`` is not None; torch is only
   looked at when it is already imported): the loader already runs one process per worker, and a 16-thread decoder
   per worker oversubscribes a training node (measured in a DataLoader benchmark);
3. otherwise ``min(16, CPUs available to this process)``.

A decode in a process forked after its parent decoded with several threads runs on 1 thread whatever the setting
(``decoder_threads``): GNU libgomp's thread pool does not survive fork(), and a team of several threads would hang.

It is a runtime knob, never stored in the codec configuration.

The encoder's least-squares fits (the LAPACK routine of ``np.linalg.lstsq`` in numpy's BLAS library, called
without the GIL by ``native.Lstsq``, on a few hundred thousand rows and 2-7 columns) run inside ``blas_serial()``:
numpy's BLAS (OpenBLAS by default) is limited to one thread for the duration of the chunk encode, restored when the
last concurrent encode leaves. OpenBLAS's pthreads build serialises concurrent calls on one global pool and spawns a
thread per core for problems this small, which made the field threads slower than one thread and every fit several
times slower on a many-core node. The fits are then deterministic (one thread, one summation order); the baselines
check that the chunk bytes are those of the earlier encoder, whose fits ran on the BLAS thread count of the machine
(for the integer-valued fit inputs of the lattice steps no dependence was seen, a large float-valued fit could in
principle have differed by a rounding).
"""

from __future__ import annotations

import os
import sys
import threading
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

from threadpoolctl import threadpool_limits

THREADS_ENV = "ANEMOI_DATASETS_CODECS_THREADS"
MAX_DEFAULT_THREADS = 16


def available_cpus() -> int:
    """CPUs this process may run on (affinity / cgroup aware where the platform tells)."""
    if hasattr(os, "process_cpu_count"):  # Python >= 3.13
        n = os.process_cpu_count()
        if n:
            return n
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0)) or 1
        except OSError:
            pass
    return os.cpu_count() or 1


def in_dataloader_worker() -> bool:
    """True inside a torch DataLoader worker (without importing torch)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    try:
        return torch.utils.data.get_worker_info() is not None
    except Exception:  # an incomplete or unusual torch install: never fail a decode over this
        return False


def default_threads() -> int:
    """Thread count for the next encode / decode (see the module docstring)."""
    value = os.environ.get(THREADS_ENV, "").strip()
    if value:
        try:
            n = int(value)
        except ValueError:
            raise ValueError(f"{THREADS_ENV} must be an integer >= 1, got {value!r}") from None
        if n < 1:
            raise ValueError(f"{THREADS_ENV} must be >= 1, got {value!r}")
        return n
    if in_dataloader_worker():
        return 1
    return min(MAX_DEFAULT_THREADS, available_cpus())


_executors: dict[int, ThreadPoolExecutor] = {}
_executors_lock = threading.Lock()


def _executor(n: int) -> ThreadPoolExecutor:
    """Shared pool per thread count (creating one per chunk would cost more than encoding a small field)."""
    with _executors_lock:
        if n not in _executors:
            _executors[n] = ThreadPoolExecutor(n, thread_name_prefix="anemoi_datasets_codecs")
        return _executors[n]


_team_used = False  # this process ran a decoder team of > 1 OpenMP threads (libgomp's thread pool exists)
_forked_after_team = False  # ... and this process is a fork of such a process
_warned_fork = False


def _after_fork_in_child() -> None:
    """A forked child inherits the executor objects but not their threads, and GNU libgomp's thread pool does not
    survive fork(): a team of > 1 threads in the child would wait forever for threads that do not exist."""
    global _forked_after_team
    _executors.clear()
    if _team_used:
        _forked_after_team = True


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


def decoder_threads() -> int:
    """Threads for the next chunk decode (an OpenMP team): ``default_threads()``, except 1 in a process forked after
    its parent decoded with several threads (libgomp cannot start a team there; a one-time warning says so)."""
    global _team_used, _warned_fork
    n = default_threads()
    if n > 1 and _forked_after_team:
        if not _warned_fork:
            _warned_fork = True
            warnings.warn(
                "this process was forked after its parent decoded with several OpenMP threads, and GNU OpenMP cannot "
                "start a thread team after fork(): decoding with 1 thread. Start the workers with the 'spawn' or "
                "'forkserver' method, or keep the parent at ANEMOI_DATASETS_CODECS_THREADS=1, to use more.",
                RuntimeWarning,
                stacklevel=3,
            )
        return 1
    if n > 1:
        _team_used = True
    return n


def map_threads(fn: Callable[[int], Any], n: int) -> list[Any]:
    """``[fn(0), ..., fn(n - 1)]`` on up to ``default_threads()`` threads (in order)."""
    threads = min(default_threads(), n)
    if threads <= 1:
        return [fn(i) for i in range(n)]
    return list(_executor(threads).map(fn, range(n)))


_blas_lock = threading.Lock()
_blas_users = 0
_blas_limits: Any = None


@contextmanager
def blas_serial() -> Iterator[None]:
    """Run the block with numpy's BLAS pools limited to one thread (``threadpoolctl``). Reference counted: the first
    concurrent user sets the limit and the last one restores the previous thread counts, so overlapping chunk encodes
    on several threads see the limit for their whole duration and the rest of the process gets its BLAS threads back
    afterwards."""
    global _blas_users, _blas_limits
    with _blas_lock:
        if _blas_users == 0:
            _blas_limits = threadpool_limits(limits=1, user_api="blas")
        _blas_users += 1
    try:
        yield
    finally:
        with _blas_lock:
            _blas_users -= 1
            if _blas_users == 0 and _blas_limits is not None:
                _blas_limits.unregister()
                _blas_limits = None
