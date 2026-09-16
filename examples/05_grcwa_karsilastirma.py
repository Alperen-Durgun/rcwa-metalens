"""Örnek 5 — Bağımsız doğrulama: rcwa_py vs grcwa.
Senin kodunun sonuçlarını, hakemli/açık kaynak bir RCWA çözücüsü olan
grcwa (Fan grubu, Stanford) ile yan yana karşılaştırır. İki bağımsız kod
aynı sayıyı veriyorsa doğruluk bağımsız olarak kanıtlanmış olur.

Kurulum (bir kez):  pip install grcwa autograd
Çalıştır:           python examples/05_grcwa_karsilastirma.py

Not: grcwa 2B periyodik bir çözücü. 1B grating'i taklit etmek için y ekseninde
subwavelength (kırınımsız) bir periyot kullanılır. Kalan ~1e-4 fark, grcwa'nın
deseni gerçek-uzay gridinde örneklemesinden kaynaklanır (ayrıklaştırma).
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import solve_rcwa_1d

try:
    import grcwa
    grcwa.set_backend('numpy')
except ImportError:
    print("grcwa kurulu değil. Kur:  pip install grcwa autograd"); sys.exit(1)

# --- ortak problem ---
lam0, period, theta, d = 0.55, 1.2, 15.0, 0.30
er_ref, er_trn, eps_hi, eps_lo, duty = 1.0, 2.25, 4.0, 1.0, 0.5


def mine(pol):
    x = (np.arange(1000) + 0.5) / 1000
    eps = np.where(x < duty, eps_hi, eps_lo).astype(complex)
    r = solve_rcwa_1d(lam0, theta, er_ref, er_trn, period, [(eps, d)], 81, pol=pol)
    return r['Rtot'], r['Ttot']


def ref(spol):
    obj = grcwa.obj(121, [period, 0.0], [0.0, 0.20], 1.0/lam0,
                    np.deg2rad(theta), 0.0, verbose=0)
    obj.Add_LayerUniform(1.0, er_ref)
    Nx, Ny = 256, 64
    obj.Add_LayerGrid(d, Nx, Ny)
    obj.Add_LayerUniform(1.0, er_trn)
    obj.Init_Setup()
    xx = (np.arange(Nx) + 0.5) / Nx
    epg = np.ones((Nx, Ny)) * eps_lo
    epg[xx < duty, :] = eps_hi
    obj.GridLayer_geteps(epg.flatten())
    if spol:
        obj.MakeExcitationPlanewave(0.0, 0.0, 1.0, 0.0, order=0)   # s = TE
    else:
        obj.MakeExcitationPlanewave(1.0, 0.0, 0.0, 0.0, order=0)   # p = TM
    R, T = obj.RT_Solve(normalize=1)
    return float(R), float(T)


if __name__ == "__main__":
    print("=== rcwa_py  vs  grcwa (bağımsız referans), θ=15° binary grating ===\n")
    ok = True
    for name, pol, spol in [("TE / s-pol", (0., 1.), True), ("TM / p-pol", (1., 0.), False)]:
        Rm, Tm = mine(pol); Rr, Tr = ref(spol)
        dR, dT = abs(Rm - Rr), abs(Tm - Tr)
        ok &= (dR < 2e-3 and dT < 2e-3)
        print(f"--- {name} ---")
        print(f"{'':18s}{'Rtot':>10s}{'Ttot':>10s}")
        print(f"{'rcwa_py':18s}{Rm:>10.5f}{Tm:>10.5f}")
        print(f"{'grcwa':18s}{Rr:>10.5f}{Tr:>10.5f}")
        print(f"{'|fark|':18s}{dR:>10.2e}{dT:>10.2e}\n")
    print("SONUÇ:", "✅ Bağımsız doğrulama başarılı (fark < 2e-3)" if ok else "⚠️ Fark büyük, incele")
