"""Parallel sweep helper.

Runs independent RCWA computations (library/wavelength/angle sweeps) across
multiple cores. It takes a function + a list of arguments because it needs a
top-level worker.

Usage:
    from rcwa.parallel import parallel_map
    results = parallel_map(worker, arg_list, n_jobs=4)

Workers are started with the "spawn" method and with the BLAS/OpenMP thread count limited to
``blas_threads`` (default 1). Without that limit every worker opens one BLAS thread per core
and the processes compete: measured on 2 cores, 2 workers were 2x SLOWER than serial; with the
limit they were 1.54x faster. Results are identical to the serial run (each solve is
independent and deterministic).

Note: on Windows the main script must be under `if __name__ == "__main__":`.
n_jobs=1 -> serial (safe default).
"""
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

_THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def parallel_map(func, args_list, n_jobs=1, blas_threads=1):
    """Applies func to each element of args_list. n_jobs>1 -> parallel process pool whose
    workers use at most blas_threads BLAS threads each."""
    args_list = list(args_list)
    if n_jobs is None or n_jobs <= 1 or len(args_list) <= 1:
        return [func(a) for a in args_list]
    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    try:
        for k in _THREAD_VARS:                 # inherited by the spawned workers at start-up
            os.environ[k] = str(int(blas_threads))
        with ProcessPoolExecutor(max_workers=min(int(n_jobs), len(args_list)),
                                 mp_context=mp.get_context("spawn")) as ex:
            return list(ex.map(func, args_list))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
