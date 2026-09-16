"""Örnek 11 — GERÇEK 2B metalens, 3B-FDTD (Meep) TAM-DALGA doğrulama ŞABLONU.

Gerçek metalens x-y'de sütun dizisidir; onu tam-dalga doğrulamak 3B-FDTD ister.
Bu, `examples/10_metalens_layout_2d.py`'nin (yerel-periyodik tasarım) NİHAİ
tam-dalga doğrulamasıdır: aynı sütun haritasını 3B'de birebir simüle eder.

DURUM: ŞABLON. Meep sandbox'ta yok (conda gerekir) — burada çalıştırılıp test
EDİLMEDİ. 3B-FDTD PAHALIDIR: küçük lens (≈10-20 λ) çok çekirdekli iş istasyonu,
büyük lens HPC/GPU ister. Küçük başla, çözünürlüğü/lens boyutunu kademeli artır.

Kurulum:
    conda create -n mp -c conda-forge pymeep pymeep-extras
    conda activate mp
    python examples/11_metalens_meep_3d.py

Kaynak: Oskooi ve ark. 2010 (Meep). Alternatif GPU-RCWA: TORCWA (Kim & Lee 2023).
Bkz. [[🔬 Tam-Dalga Doğrulama Ortamı (FDTD ve GPU-RCWA)]].
"""
try:
    import meep as mp
except ImportError:
    raise SystemExit("Meep yok. Kur: conda install -c conda-forge pymeep")
import numpy as np, os, sys

# ---- 08/10 ile AYNI lens parametreleri (kıyas için) ----
lam0 = 0.633; Lcell = 0.35; H = 0.60; n_pillar = 2.4; n_sub = 1.5
D_lens = 6.0; f = 6.0            # küçük tut (3B-FDTD pahalı!)
resolution = 20                 # piksel/um (yakınsama için artır: 30-40)
fcen = 1.0/lam0

# --- meta-atom kütüphanesi (rcwa_py, kararlı birim hücre) ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import solve_rcwa_2d
sides = np.linspace(0.06, Lcell-0.05, 16); libph = []
for s in sides:
    N = 48; x = (np.arange(N)+0.5)/N*Lcell; X, Y = np.meshgrid(x, x, indexing='ij')
    eps = np.where((np.abs(X-Lcell/2) < s/2) & (np.abs(Y-Lcell/2) < s/2), n_pillar**2, 1.0)
    r = solve_rcwa_2d(lam0, 0, 0, 1, 1, Lcell, Lcell, [(eps, H)], 7, 7, pol=(0., 1.))
    libph.append(np.angle(r['ty'][r['i0']]))
libph = np.unwrap(np.array(libph)); libph -= libph.min(); libphm = np.mod(libph, 2*np.pi)

# --- dairesel açıklıkta sütun haritası ---
Nc = int(round(D_lens/Lcell)); D_lens = Nc*Lcell
ci = (np.arange(Nc)-(Nc-1)/2)*Lcell
pillars = []
for i in range(Nc):
    for j in range(Nc):
        xx, yy = ci[i], ci[j]; rr = np.hypot(xx, yy)
        if rr <= D_lens/2:
            phi = np.mod(-(2*np.pi/lam0)*(np.sqrt(rr**2+f**2)-f), 2*np.pi)
            s = sides[np.argmin(np.abs(np.angle(np.exp(1j*(libphm-phi)))))]
            pillars.append((xx, yy, s))

# ---- Meep 3B sahne ----
dpml = 1.0; air_in = 1.0; air_out = 1.5*f + 1.0
sx = D_lens + 3.0; sy = sx
sz = dpml + air_in + H + air_out + dpml
cell = mp.Vector3(sx, sy, sz)
z_lens = -sz/2 + dpml + air_in + H/2
geometry = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, dpml+air_out),  # alt yarı-uzay (hava)
                     center=mp.Vector3(0, 0, sz/2 - (dpml+air_out)/2), material=mp.Medium(index=1.0))]
for (xx, yy, s) in pillars:
    geometry.append(mp.Block(size=mp.Vector3(s, s, H),
                             center=mp.Vector3(xx, yy, z_lens),
                             material=mp.Medium(index=n_pillar)))

z_src = -sz/2 + dpml + 0.3
sources = [mp.Source(mp.ContinuousSource(frequency=fcen), component=mp.Ey,
                     center=mp.Vector3(0, 0, z_src), size=mp.Vector3(sx, sy, 0))]

sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(dpml)],
                    geometry=geometry, sources=sources, resolution=resolution,
                    force_complex_fields=True)
sim.run(until=150)

# ---- odak: lens altında xz kesitinde |E|² ----
z0 = -sz/2 + dpml + air_in + H            # lens çıkış yüzeyi
ey = sim.get_array(center=mp.Vector3(0, 0, (z0 + sz/2 - dpml)/2),
                   size=mp.Vector3(sx, 0, (sz/2 - dpml) - z0), component=mp.Ey)
I = np.abs(ey)**2
nz = I.shape[-1]; zaxis = np.linspace(z0, sz/2 - dpml, nz) - z0
axis = I[I.shape[0]//2, :]
zf = zaxis[int(np.argmax(axis))]
print(f"Meep 3B-FDTD: gerçek odak z ≈ {zf:.2f} um (tasarım f={f}). Kıyas: 10_metalens_layout_2d (yerel-periyodik).")
print("Not: çözünürlüğü artırıp yakınsamayı doğrula. Büyük lens için HPC/GPU.")
