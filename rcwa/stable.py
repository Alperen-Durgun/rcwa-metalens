"""Kararlı otomatik çözücü — A1.

Kalın+yüksek-indis süper-hücrede RCWA, belirli mertebe sayılarında (M) arayüz
matrisi tekilleştiği için (cond(A)~1e16, rehberli-mod rezonansı) patlar. Bu bize
özgü değildir (grcwa da patlar). Kanıtlanmış çözüm: birkaç M dene, ENERJİ
KORUNUMU (ΣDE≈1) sağlayanları seç — geçerli çözümlerin hepsi aynı fiziği verir.

Not: Tekil-M için nihai düzeltme "enhanced transmittance matrix" (Moharam 1995b)
olurdu; katman bölme İŞE YARAMAZ (denendi, daha kötü). Bu yüzden pratik ve
güvenilir yol koşul/enerji korumalı otomatik taramadır.
"""
import numpy as np
from .solver import solve_rcwa_1d


def solve_1d_auto(lam0, theta_deg, er_ref, er_trn, period, layers, base_M,
                  pol=(0., 1.), M_list=None, tol=0.02):
    """Enerji-korumalı otomatik-M 1B RCWA. Kararlı çözümü döndürür.

    Döndürülen dict'e eklenir: stable_M, sigDE, n_stable, M_denenen.
    Kararlı çözüm yoksa RuntimeError (FDTD öner).
    """
    if M_list is None:
        cand = {base_M - 20, base_M - 10, base_M, base_M + 10, base_M + 20}
        M_list = sorted(m for m in cand if m >= 8)
    good = []
    for M in M_list:
        r = solve_rcwa_1d(lam0, theta_deg, er_ref, er_trn, period, layers, 2 * M + 1, pol=pol)
        sig = r["Rtot"] + r["Ttot"]
        if np.isfinite(sig) and abs(sig - 1) < tol:
            good.append((abs(sig - 1), M, r, sig))
    if not good:
        raise RuntimeError("Kararlı M bulunamadı (hepsi enerji korumadı). "
                           "M_list'i genişlet ya da FDTD/GPU kullan.")
    good.sort(key=lambda t: t[0])
    _, M, r, sig = good[0]
    r["stable_M"] = M; r["sigDE"] = sig; r["n_stable"] = len(good)
    r["M_denenen"] = M_list
    return r


def solve_2d_auto(lam0, theta_deg, phi_deg, er_ref, er_trn, Lx, Ly, layers, base_M,
                  pol=(0., 1.), M_list=None, tol=0.02):
    """Enerji-korumalı otomatik-M 2B RCWA (A1'in 2B karşılığı)."""
    from .solver2d import solve_rcwa_2d
    if M_list is None:
        M_list = sorted({m for m in (base_M - 2, base_M - 1, base_M, base_M + 1) if m >= 3})
    good = []
    for M in M_list:
        r = solve_rcwa_2d(lam0, theta_deg, phi_deg, er_ref, er_trn, Lx, Ly, layers, M, M, pol=pol)
        sig = r["Rtot"] + r["Ttot"]
        if np.isfinite(sig) and abs(sig - 1) < tol:
            good.append((abs(sig - 1), M, r, sig))
    if not good:
        raise RuntimeError("Kararlı M bulunamadı (2B); M_list genişlet ya da FDTD.")
    good.sort(key=lambda t: t[0]); _, M, r, sig = good[0]
    r["stable_M"] = M; r["sigDE"] = sig; r["n_stable"] = len(good)
    return r
