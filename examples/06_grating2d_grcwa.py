"""Örnek 6 — 2B doğrulama: solve_rcwa_2d vs grcwa.
Kare latiste dielektrik sütun; toplam R/T ve enerji korunumu karşılaştırılır.
Kurulum:  pip install grcwa autograd
Çalıştır: python examples/06_grating2d_grcwa.py
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import solve_rcwa_2d
try:
    import grcwa; grcwa.set_backend('numpy')
except ImportError:
    print("grcwa kurulu değil: pip install grcwa autograd"); sys.exit(1)

lam0, Lx, Ly, d = 0.55, 1.0, 1.0, 0.30
er_ref, er_trn, eps_hi, eps_lo, side = 1.0, 2.25, 4.0, 1.0, 0.5

def eps_grid(N):
    x = (np.arange(N) + 0.5) / N * Lx
    X, Y = np.meshgrid(x, x, indexing='ij')
    e = np.ones((N, N)) * eps_lo
    e[(np.abs(X - Lx/2) < side/2) & (np.abs(Y - Ly/2) < side/2)] = eps_hi
    return e

print("=== 2B: rcwa_py vs grcwa (kare sütun, normal geliş) ===\n")
eN = eps_grid(120)
for M in (5, 7, 9):
    m = solve_rcwa_2d(lam0, 0, 0, er_ref, er_trn, Lx, Ly, [(eN, d)], M, M, pol=(0., 1.))
    print(f"rcwa_py  M={M}: Rtot={m['Rtot']:.5f} Ttot={m['Ttot']:.5f} ΣDE={m['Rtot']+m['Ttot']:.5f}")

obj = grcwa.obj(225, [Lx, 0.], [0., Ly], 1.0/lam0, 0., 0., verbose=0)
obj.Add_LayerUniform(1.0, er_ref); obj.Add_LayerGrid(d, 120, 120); obj.Add_LayerUniform(1.0, er_trn)
obj.Init_Setup(); obj.GridLayer_geteps(eps_grid(120).flatten())
obj.MakeExcitationPlanewave(0., 0., 1., 0., order=0)
R, T = obj.RT_Solve(normalize=1)
print(f"grcwa   nG=225: Rtot={float(R):.5f} Ttot={float(T):.5f} ΣDE={float(R)+float(T):.5f}")
print(f"\n|fark| (M=9): {abs(m['Rtot']-float(R)):.2e}  -> ✅" if abs(m['Rtot']-float(R))<2e-3 else "⚠️")
