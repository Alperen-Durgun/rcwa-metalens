"""Disk cache — B4 (library) & B5 (eigenmode).

Stores heavy computations (phase libraries, layer eigenmodes) as .npz keyed
by parameters. Seconds on the second call.

Cache folder: $RCWA_CACHE_DIR if set, otherwise ~/.cache/rcwa-metalens (never inside
the installed package). The folder is created lazily; if it is not writable the
result is simply computed without caching.
"""
import os, hashlib, numpy as np

CACHE_DIR = os.environ.get("RCWA_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "rcwa-metalens")


CACHE_VERSION = 2   # bump whenever solver defaults change (v2: Li factorization + fine raster)


def _key(prefix, params):
    s = repr(sorted(params.items())) + f"|v{CACHE_VERSION}"
    h = hashlib.md5(s.encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{prefix}_{h}.npz")


def cached_library(prefix, params, compute):
    """A dict-result cache keyed by params.
    compute() -> {name: ndarray}. Loads if present, else computes and stores."""
    path = _key(prefix, params)
    if os.path.exists(path):
        try:
            with np.load(path, allow_pickle=False) as d:
                return {k: d[k] for k in d.files}
        except Exception:
            pass                                  # corrupt/old entry -> recompute
    out = compute()
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = path[:-4] + f".{os.getpid()}.tmp.npz"
        np.savez(tmp, **out); os.replace(tmp, path)   # atomic: no half-written entries
    except Exception:
        pass                                      # read-only location: just skip caching
    return out


def clear_cache():
    n = 0
    if not os.path.isdir(CACHE_DIR):
        return 0
    for f in os.listdir(CACHE_DIR):
        if f.endswith(".npz"):
            os.remove(os.path.join(CACHE_DIR, f)); n += 1
    return n
