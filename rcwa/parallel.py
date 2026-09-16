"""Paralel sweep aracı — B7.

Bağımsız RCWA hesaplarını (kütüphane/dalga boyu/açı taramaları) çok çekirdekte
koşturur. Top-level worker gerektiği için işlev + argüman listesi alır.

Kullanım:
    from rcwa.parallel import parallel_map
    sonuçlar = parallel_map(worker, arg_listesi, n_jobs=4)

Not: Windows'ta ana betik `if __name__ == "__main__":` altında olmalı.
n_jobs=1 -> seri (güvenli varsayılan).
"""
from concurrent.futures import ProcessPoolExecutor


def parallel_map(func, args_list, n_jobs=1):
    """func'ı args_list'in her elemanına uygular. n_jobs>1 -> paralel süreç havuzu."""
    if n_jobs is None or n_jobs <= 1:
        return [func(a) for a in args_list]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(func, args_list))
