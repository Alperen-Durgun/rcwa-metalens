"""Örnek 3 — Metalens meta-atom faz kütüphanesi (Ders 07 & Kod Dersi 06).
Bir nano-sütunun genişliğini tarayıp 0. mertebe geçiş fazını ve genliğini
hesaplar. İyi bir kütüphane: faz 0-2π'yi kapsamalı, genlik yüksek kalmalı.
Çıktı grafiği projedeki Simulasyonlar/ klasörüne kaydedilir.
Çalıştır:  python examples/03_metalens_library.py
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import solve_rcwa_1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    lam0 = 0.633            # tasarım dalga boyu (um)
    period = 0.30           # subwavelength birim hücre -> sadece 0. mertebe
    H = 0.60                # sütun yüksekliği (um)
    n_pillar = 2.4          # TiO2 benzeri yüksek indis
    P = 21
    Nx = 512
    widths = np.linspace(0.03, 0.29, 40)   # sütun genişliği taraması (um)

    phase, amp = [], []
    for w in widths:
        x = (np.arange(Nx) + 0.5) / Nx * period
        eps = np.where(np.abs(x - period / 2) < w / 2, n_pillar ** 2, 1.0).astype(complex)
        r = solve_rcwa_1d(lam0, 0.0, 1.0, 1.0, period, [(eps, H)], P, pol=(0., 1.))
        t0 = r['ty'][r['zeroth_index']]     # TE 0. mertebe geçiş genliği
        phase.append(np.angle(t0)); amp.append(np.abs(t0))

    phase = np.unwrap(np.array(phase)); phase -= phase[0]
    amp = np.array(amp)
    print(f"Faz aralığı: {np.ptp(phase)/np.pi:.2f}π  |  Ortalama genlik: {amp.mean():.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(widths * 1000, phase / np.pi, 'o-'); ax[0].set_xlabel("Sütun genişliği (nm)")
    ax[0].set_ylabel("Geçiş fazı (π)"); ax[0].set_title("Faz kütüphanesi"); ax[0].grid(True, alpha=.3)
    ax[1].plot(widths * 1000, amp, 's-', color='C1'); ax[1].set_xlabel("Sütun genişliği (nm)")
    ax[1].set_ylabel("|t0|"); ax[1].set_title("Geçiş genliği"); ax[1].set_ylim(0, 1.05); ax[1].grid(True, alpha=.3)
    fig.suptitle(f"Meta-atom taraması  (λ={lam0}um, Λ={period}um, H={H}um, n={n_pillar})")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Simulasyonlar", "meta_atom_faz_kutuphanesi.png")
    out = os.path.abspath(out)
    fig.savefig(out, dpi=130)
    print("Grafik kaydedildi ->", out)
