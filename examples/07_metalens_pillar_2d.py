"""Örnek 7 — 2B meta-atom faz kütüphanesi (gerçek metalens birim hücresi).
Kare kesitli nano-sütunun kenarını tarayıp 0. mertebe geçiş fazını/genliğini
çıkarır. 2B = gerçek metalens (x-y'de sütun). Grafik Simulasyonlar/'a kaydedilir.
Çalıştır: python examples/07_metalens_pillar_2d.py
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import solve_rcwa_2d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

lam0, L, H, n_pillar, M, N = 0.633, 0.35, 0.60, 2.4, 7, 96
sides = np.linspace(0.05, 0.32, 24)
phase, amp = [], []
for s in sides:
    x = (np.arange(N) + 0.5) / N * L
    X, Y = np.meshgrid(x, x, indexing='ij')
    eps = np.where((np.abs(X - L/2) < s/2) & (np.abs(Y - L/2) < s/2), n_pillar**2, 1.0)
    r = solve_rcwa_2d(lam0, 0, 0, 1.0, 1.0, L, L, [(eps, H)], M, M, pol=(0., 1.))
    t0 = r['ty'][r['i0']]; phase.append(np.angle(t0)); amp.append(np.abs(t0))
phase = np.unwrap(np.array(phase)); phase -= phase[0]; amp = np.array(amp)
print(f"2B meta-atom: faz kapsaması {np.ptp(phase)/np.pi:.2f}π, ort. genlik {amp.mean():.3f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(sides*1000, phase/np.pi, 'o-'); ax[0].set_xlabel("Sütun kenarı (nm)")
ax[0].set_ylabel("Faz (π)"); ax[0].set_title("2B faz kütüphanesi"); ax[0].grid(True, alpha=.3)
ax[1].plot(sides*1000, amp, 's-', color='C1'); ax[1].set_xlabel("Sütun kenarı (nm)")
ax[1].set_ylabel("|t0|"); ax[1].set_ylim(0, 1.05); ax[1].set_title("Genlik"); ax[1].grid(True, alpha=.3)
fig.suptitle(f"2B meta-atom (λ={lam0}, Λ={L}, H={H}, n={n_pillar})"); fig.tight_layout()
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "Simulasyonlar", "meta_atom_2b_faz_kutuphanesi.png"))
fig.savefig(out, dpi=130); print("Grafik ->", out)
