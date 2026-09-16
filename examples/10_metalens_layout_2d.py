"""Örnek 10 — 2B METALENS TASARIMI (yerleşim + odak tahmini).

Boru hattı:
  1) 2B meta-atom faz kütüphanesi (solve_rcwa_2d, tek hücre periyodik)
  2) dairesel açıklıkta hedef faz -> her hücreye sütun kenarı ata (2B harita)
  3) yerel-periyodik near-field -> 2B açısal spektrum ile odak düzlemine yayılım
     -> gerçek odak, FWHM, odaklama verimi

DÜRÜST SINIR: Bu YEREL PERİYODİK yaklaşımdır (her hücre yerel olarak periyodik
kabul edilir; komşu kuplajı ihmal). Hızlıdır ve dizüstünde çalışır; tasarım ve
ön-değerlendirme için standarttır. NİHAİ tam-dalga doğrulama için FDTD/GPU:
-> 09_metalens_fdtd_meep.py (Meep) veya TORCWA. Bkz. [[Kod Dersi 2B-Yerlesim ...]]

Çalıştır:  python examples/10_metalens_layout_2d.py
"""
import numpy as np, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import solve_rcwa_2d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

lam0, Lcell, H, n_pillar = 0.633, 0.35, 0.60, 2.4
D_lens, f = 20.0, 25.0
M = 7                       # 2B birim-hücre mertebe yarı-sayısı
k0 = 2*np.pi/lam0

# 1) 2B faz kütüphanesi
t = time.time()
sides = np.linspace(0.06, 0.30, 20); libph = []; libam = []
for s in sides:
    N = 64; x = (np.arange(N)+0.5)/N*Lcell; X, Y = np.meshgrid(x, x, indexing='ij')
    eps = np.where((np.abs(X-Lcell/2) < s/2) & (np.abs(Y-Lcell/2) < s/2), n_pillar**2, 1.0)
    r = solve_rcwa_2d(lam0, 0, 0, 1, 1, Lcell, Lcell, [(eps, H)], M, M, pol=(0., 1.))
    t0 = r['ty'][r['i0']]; libph.append(np.angle(t0)); libam.append(abs(t0))
libph = np.unwrap(np.array(libph)); libph -= libph.min(); libam = np.array(libam)
libphm = np.mod(libph, 2*np.pi)
print(f"Kütüphane: faz {np.ptp(libph)/np.pi:.2f}π, ort. genlik {libam.mean():.2f}  ({time.time()-t:.0f}s)")

# 2) dairesel açıklıkta yerleşim
Nc = int(round(D_lens/Lcell)); D_lens = Nc*Lcell
ci = (np.arange(Nc)-(Nc-1)/2)*Lcell
XX, YY = np.meshgrid(ci, ci, indexing='ij'); R = np.sqrt(XX**2+YY**2)
inside = R <= D_lens/2
phi = np.mod(-(2*np.pi/lam0)*(np.sqrt(R**2+f**2)-f), 2*np.pi)
side_map = np.zeros_like(R); amp_map = np.zeros_like(R); ph_map = np.zeros_like(R)
for i in range(Nc):
    for j in range(Nc):
        if inside[i, j]:
            k = np.argmin(np.abs(np.angle(np.exp(1j*(libphm-phi[i, j])))))
            side_map[i, j] = sides[k]; amp_map[i, j] = libam[k]; ph_map[i, j] = libphm[k]
NA = np.sin(np.arctan(D_lens/2/f))
print(f"Yerleşim: {int(inside.sum())} sütun (dairesel açıklık), D={D_lens:.1f}um, f={f}um, NA={NA:.2f}")

# 3) yerel-periyodik near-field -> 2B açısal spektrum
E0 = np.where(inside, amp_map*np.exp(1j*ph_map), 0.0)
u = 4; E0f = np.kron(E0, np.ones((u, u))); dx = Lcell/u; Ncf = Nc*u
pad = int(Ncf*1.6); E0p = np.zeros((pad, pad), complex); o = (pad-Ncf)//2
E0p[o:o+Ncf, o:o+Ncf] = E0f
kxx = 2*np.pi*np.fft.fftfreq(pad, d=dx); KX, KY = np.meshgrid(kxx, kxx, indexing='ij')
KZ = np.sqrt((k0**2-KX**2-KY**2).astype(complex)); KZ = np.where(np.imag(KZ) < 0, -KZ, KZ)
A0 = np.fft.fft2(E0p)
def prop(z): return np.fft.ifft2(A0*np.exp(1j*KZ*z))
zs = np.linspace(5, 2*f, 80); Iax = [np.abs(prop(z)[pad//2, pad//2])**2 for z in zs]
zf = zs[int(np.argmax(Iax))]
I = np.abs(prop(zf))**2
xax = (np.arange(pad)-pad//2)*dx; cut = I[:, pad//2]/I.max()
hw = xax[cut >= 0.5]; fwhm = hw.max()-hw.min()
rr = np.sqrt((xax[:, None])**2 + (xax[None, :])**2)
eff = I[rr <= 1.5*fwhm].sum()/(np.abs(E0p)**2).sum()
print(f"Yerel-periyodik odak: z={zf:.1f}um (tasarım {f}) | FWHM={fwhm*1000:.0f}nm | λ/(2NA)={lam0/(2*NA)*1000:.0f}nm | odak verimi ~{eff*100:.0f}%")

# --- figür ---
fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
sm = np.where(inside, side_map*1000, np.nan)
im0 = ax[0].imshow(sm, origin="lower", extent=[ci[0], ci[-1], ci[0], ci[-1]], cmap="viridis")
ax[0].set_title("Sütun kenarı haritası (nm)"); ax[0].set_xlabel("x (um)"); ax[0].set_ylabel("y (um)")
plt.colorbar(im0, ax=ax[0], fraction=0.046)
ext = xax.max()
im1 = ax[1].imshow(I/I.max(), origin="lower", extent=[-ext, ext, -ext, ext], cmap="inferno")
ax[1].set_xlim(-3, 3); ax[1].set_ylim(-3, 3)
ax[1].set_title(f"Odak düzlemi I(x,y) @ z={zf:.1f}um"); ax[1].set_xlabel("x (um)"); ax[1].set_ylabel("y (um)")
plt.colorbar(im1, ax=ax[1], fraction=0.046)
ax[2].plot(xax*1000, cut); ax[2].axhline(0.5, color="gray", ls=":"); ax[2].set_xlim(-2500, 2500)
ax[2].set_xlabel("x (nm)"); ax[2].set_ylabel("I (norm)"); ax[2].set_title(f"Odak kesiti — FWHM={fwhm*1000:.0f}nm")
fig.suptitle(f"2B METALENS tasarımı — D={D_lens:.0f}um f={f}um NA={NA:.2f} | odak z={zf:.1f}um, verim~{eff*100:.0f}% (yerel-periyodik)")
fig.tight_layout()
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "Simulasyonlar", "metalens_layout_2d.png"))
fig.savefig(out, dpi=130); print("Grafik ->", out)
