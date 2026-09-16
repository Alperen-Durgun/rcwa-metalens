"""Disk önbelleği — B4 (kütüphane) & B5 (özmod).

Ağır hesapları (faz kütüphaneleri, katman özmodları) parametre anahtarıyla
.npz olarak saklar. İkinci çağrıda saniyeler. Önbellek klasörü: rcwa_py/.cache/
"""
import os, hashlib, numpy as np

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _key(prefix, params):
    s = repr(sorted(params.items()))
    h = hashlib.md5(s.encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{prefix}_{h}.npz")


def cached_library(prefix, params, compute):
    """params ile anahtarlanan bir sözlük-sonuç önbelleği.
    compute() -> {isim: ndarray}. Varsa yükler, yoksa hesaplar ve saklar."""
    path = _key(prefix, params)
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files}
    out = compute()
    np.savez(path, **out)
    return out


def clear_cache():
    n = 0
    for f in os.listdir(CACHE_DIR):
        if f.endswith(".npz"):
            os.remove(os.path.join(CACHE_DIR, f)); n += 1
    return n
