"""Örnek 1 — Doğrulama: Homojen film RCWA sonucu analitik Fresnel'e eşit mi?
Çalıştır:  python examples/01_slab_validation.py
Beklenen: ΔR ~ 1e-16 (makine hassasiyeti), ΣDE = 1.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import solve_rcwa_1d


def film_analytic(lam0, theta_deg, n_ref, n1, n_trn, d, pol):
    th = np.deg2rad(theta_deg); s = n_ref * np.sin(th)
    c_r = np.sqrt(1 - (s / n_ref) ** 2 + 0j)
    c_1 = np.sqrt(1 - (s / n1) ** 2 + 0j)
    c_t = np.sqrt(1 - (s / n_trn) ** 2 + 0j)
    if pol == 'TE':
        er, e1, et = n_ref * c_r, n1 * c_1, n_trn * c_t
    else:
        er, e1, et = n_ref / c_r, n1 / c_1, n_trn / c_t
    dl = 2 * np.pi / lam0 * n1 * c_1 * d
    m11 = np.cos(dl); m12 = 1j * np.sin(dl) / e1
    m21 = 1j * e1 * np.sin(dl); m22 = np.cos(dl)
    r = (er * (m11 + m12 * et) - (m21 + m22 * et)) / (er * (m11 + m12 * et) + (m21 + m22 * et))
    return abs(r) ** 2


if __name__ == "__main__":
    lam0, period, P, n1, d = 0.55, 0.30, 21, 1.5, 0.40
    print("pol  θ     RCWA_R0    Fresnel_R   ΔR        ΣDE")
    for pol, pv in [('TE', (0., 1.)), ('TM', (1., 0.))]:
        for th in (0., 30., 55.):
            eps = np.full(512, n1 ** 2, complex)
            res = solve_rcwa_1d(lam0, th, 1.0, 1.0, period, [(eps, d)], P, pol=pv)
            R0 = res['R'][res['zeroth_index']]
            Ra = film_analytic(lam0, th, 1.0, n1, 1.0, d, pol)
            print(f"{pol}  {th:4.0f}  {R0:.6f}  {Ra:.6f}  {abs(R0-Ra):.1e}  {res['Rtot']+res['Ttot']:.6f}")
