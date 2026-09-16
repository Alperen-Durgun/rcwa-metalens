"""Örnek 2 — İkili (binary) grating: enerji korunumu ve mertebe sayısına yakınsama.
Çalıştır:  python examples/02_binary_grating.py
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import solve_rcwa_1d

if __name__ == "__main__":
    lam0, period, d = 0.55, 1.20, 0.30
    x = (np.arange(1024) + 0.5) / 1024
    eps = np.where(x < 0.5, 4.0, 1.0).astype(complex)   # duty 0.5, n=2 / hava
    for pol, pv in [('TE', (0., 1.)), ('TM', (1., 0.))]:
        print(f"-- {pol} --  (θ=15°, alt ortam n=1.5)")
        for P in (11, 21, 41, 81):
            r = solve_rcwa_1d(lam0, 15., 1.0, 2.25, period, [(eps, d)], P, pol=pv)
            print(f"  P={P:3d}  Rtot={r['Rtot']:.5f}  Ttot={r['Ttot']:.5f}  ΣDE={r['Rtot']+r['Ttot']:.6f}")
