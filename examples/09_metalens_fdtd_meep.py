"""Örnek 9 — TAM-DALGA metalens, FDTD (Meep) ile — GERÇEK sonlu yapı referansı.

Neden bu dosya: Kalın+yüksek-indisli metalens süper-hücresi RCWA için kötü-koşullu
olabilir (bkz. 08 ve Kod Dersi 08). FDTD sonlu yapıyı doğrudan, kararlı biçimde
çözer ve topluluğun tam-dalga metalens doğrulamasında altın standardıdır.

DURUM: Bu bir ŞABLONdur. Meep sandbox'ta pip ile kurulamaz (conda gerekir), bu
yüzden burada çalıştırılıp test EDİLMEDİ. Kendi makinende/HPC'de çalıştır:

    conda create -n mp -c conda-forge pymeep
    conda activate mp
    python examples/09_metalens_fdtd_meep.py

Kaynak: Oskooi ve ark. 2010, Comput. Phys. Commun. 181, 687 (Meep). Ayrıntı:
[[📚 RCWA ve Metalens Kaynakçası]]. Büyük 3B lensler için GPU-RCWA (TORCWA) de bir seçenek.

Bu senaryo: 1B (silindirik) metalensin 2B-FDTD kesiti — RCWA sonucuyla (08)
kıyaslanabilir: aynı D_lens, f, Lcell, H, n_pillar kullan.
"""
try:
    import meep as mp
except ImportError:
    raise SystemExit("Meep kurulu değil. Kur:  conda install -c conda-forge pymeep")
import numpy as np

# ---- lens parametreleri (08 ile aynı tut) ----
lam0 = 0.633; Lcell = 0.35; H = 0.60; n_pillar = 2.4
D_lens = 6.0; f = 6.0
res = 40                      # piksel/um (yakınsama için artır)
dpml = 1.0; gap_in = 1.5; gap_out = 2.2*f + 2.0
fcen = 1.0/lam0

Ncell = int(round(D_lens/Lcell)); D_lens = Ncell*Lcell
sx = D_lens + 4.0            # enine pencere (+ kenar boşluğu)
sy = dpml + gap_in + H + gap_out + dpml
cell = mp.Vector3(sx, sy)
pml = [mp.PML(dpml)]

# ---- meta-atom faz kütüphanesi ----
# Doğru yol: her genişlik için tek hücreyi (periyodik) RCWA/FDTD ile tarayıp
# faz(w) çıkar (bkz. Kod Dersi 06). Basitlik için burada rcwa_py kütüphanesini
# içe aktarıp genişlikleri seçebilirsin:
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import solve_rcwa_1d
widths = np.linspace(0.04, Lcell-0.04, 60); ph = []
for w in widths:
    xg = (np.arange(256)+0.5)/256*Lcell
    eps = np.where(np.abs(xg-Lcell/2) < w/2, n_pillar**2, 1.0).astype(complex)
    r = solve_rcwa_1d(lam0, 0, 1, 1, Lcell, [(eps, H)], 21, pol=(0., 1.))
    ph.append(np.angle(r['ty'][r['zeroth_index']]))
ph = np.unwrap(np.array(ph)); ph -= ph.min(); phm = np.mod(ph, 2*np.pi)
xc = (np.arange(Ncell)+0.5)*Lcell - D_lens/2
phi = np.mod(-(2*np.pi/lam0)*(np.sqrt(xc**2+f**2)-f), 2*np.pi)
chosen = np.array([widths[np.argmin(np.abs(np.angle(np.exp(1j*(phm-p)))))] for p in phi])

# ---- geometri: nano-sütunlar ----
y_lens = -sy/2 + dpml + gap_in + H/2
geom = []
for i in range(Ncell):
    cx = xc[i]
    geom.append(mp.Block(size=mp.Vector3(chosen[i], H, mp.inf),
                         center=mp.Vector3(cx, y_lens),
                         material=mp.Medium(index=n_pillar)))

# ---- kaynak: düzlem dalga (Ez), lensin hemen üstünde ----
y_src = -sy/2 + dpml + 0.3
sources = [mp.Source(mp.ContinuousSource(frequency=fcen), component=mp.Ez,
                     center=mp.Vector3(0, y_src), size=mp.Vector3(sx, 0))]

sim = mp.Simulation(cell_size=cell, boundary_layers=pml, geometry=geom,
                    sources=sources, resolution=res, force_complex_fields=True)
sim.run(until=200)

# ---- alanı al, odak eksenini bul ----
ez = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
I = np.abs(ez)**2
nx, ny = I.shape
yv = np.linspace(-sy/2, sy/2, ny)
y0 = -sy/2 + dpml + gap_in + H     # lens çıkış yüzeyi
axis = I[nx//2, :]
mask = yv > y0
zf = yv[mask][np.argmax(axis[mask])] - y0
print(f"FDTD: gerçek odak (lens yüzeyinden) z ≈ {zf:.2f} um  (tasarım f={f} um)")
print("Karşılaştır: RCWA süper-hücre (08) ile aynı D_lens,f için.")
# İstersen matplotlib ile I haritasını kaydet.
"""

Notlar / iyi uygulama:
- `res` (çözünürlük) yakınsama parametresidir; 30-60 arası dene, odak/FWHM sabitlenene dek artır.
- PML kalınlığı en az ~1 dalga boyu olsun (yansımasız açık sınır).
- Bellek/zaman: 2B-FDTD kesit dizüstünde çalışır; tam 3B (gerçek 2B-periyodik lens)
  disk/GPU ister -> Meep MPI veya GPU-RCWA (TORCWA).
- Bu FDTD sonucunu 08'deki RCWA süper-hücre odağıyla kıyasla: yakın çıkmaları,
  metalens tasarımının tam-dalga doğrulamasıdır.
"""
