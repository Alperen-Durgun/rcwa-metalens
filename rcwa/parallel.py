"""Parallel sweep helper — B7.

Runs independent RCWA computations (library/wavelength/angle sweeps) across
multiple cores. It takes a function + a list of arguments because it needs a
top-level worker.

Usage:
    from rcwa.parallel import parallel_map
    results = parallel_map(worker, arg_list, n_jobs=4)

Note: on Windows the main script must be under `if __name__ == "__main__":`.
n_jobs=1 -> serial (safe default).
"""
from concurrent.futures import ProcessPoolExecutor


def parallel_map(func, args_list, n_jobs=1):
    """Applies func to each element of args_list. n_jobs>1 -> parallel process pool."""
    if n_jobs is None or n_jobs <= 1:
        return [func(a) for a in args_list]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(func, args_list))
