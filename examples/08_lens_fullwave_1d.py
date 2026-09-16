"""Örnek 8 — TAM-DALGA metalens (1B/silindirik), SAĞLAM sürüm.

Tüm lens tek RCWA süper-hücresi olarak (yerel yaklaşım YOK) Maxwell ile çözülür,
çıkan alan odak düzlemine EXACT taşınır.

ÖNEMLİ (dürüst uyarı): Kalın + yüksek-indisli metalens süper-hücresi RCWA için
sayısal olarak KÖTÜ-KOŞULLUdur. Belirli mertebe sayılarında (M) çözüm patlar —
bu bize özgü değil; hakemli grcwa da aynı yapıda patlar (rehberli-mod rezonansları).
Çözüm: birden çok M dene, ENERJİ KORUNUMU (ΣDE≈1) sağlayanları seç; bu geçerli
çözümlerin hepsi aynı odağı verir (tutarlılık = güven). Büyük/gerçek-boyut lens
için tam-dalga referansı FDTD'dir -> 09_metalens_fdtd_meep.py.

Çalıştır:  python examples/08_lens_fullwave_1d.py
"""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import solve_rcwa_1d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

lam0, Lcell, H, n_pillar = 0.633, 0.35, 0.60, 2.4
D_lens, f, guard = 6.0, 6.0, 3.0
M_list = [120, 130, 140, 160]
k0 = 2*np.pi/lam0
Ncell = int(round(D_lens/Lcell)); D_lens = Ncell*Lcell
D = D_lens + 2*guard; X0 = D/2
NA = np.sin(np.arctan(D_lens/2/f))
print(f"Lens D={D_lens:.2f}um f={f}um NA={NA:.2f} | süper-hücre {D:.1f}um, {Ncell} sütun")

# 1) faz kütüphanesi (tek hücre — kararlı)
widths = np.linspace(0.04, Lcell-0.04, 60); ph = []
for w in widths:
    x = (np.arange(256)+0.5)/256*Lcell
    eps = np.where(np.abs(x-Lcell/2) < w/2, n_pillar**2, 1.0).astype(complex)
    r = solve_rcwa_1d(lam0, 0, 1, 1, Lcell, [(eps, H)], 21, pol=(0., 1.))
    ph.append(np.angle(r['ty'][r['zeroth_index']]))
ph = np.unwrap(np.array(ph)); ph -= ph.min(); phm = np.mod(ph, 2*np.pi)

# 2) tasarım: hedef faz -> sütun genişliği
xc = X0 + ((np.arange(Ncell)+0.5)*Lcell - D_lens/2)
phi = np.mod(-(2*np.pi/lam0)*(np.sqrt((xc-X0)**2 + f**2) - f), 2*np.pi)
chosen = np.array([widths[np.argmin(np.abs(np.angle(np.exp(1j*(phm-p)))))] for p in phi])

# 3) tüm lensi tek süper-hücre olarak kur
Nx = 4096; xg = (np.arange(Nx)+0.5)/Nx*D
col = np.ones(Nx); xs = X0 - D_lens/2
for i in range(Ncell):
    c = xs + (i+0.5)*Lcell
    col[np.abs(xg-c) < chosen[i]/2] = n_pillar**2

# 4) M tara, enerji koruyanları (kararlı) seç
def propagate(res):
    m = res['orders']; ty = res['ty']; kx = -m*(lam0/D)
    kz = np.sqrt((1.0-kx**2).astype(complex)); kz = np.where(np.imag(kz) < 0, -kz, kz)
    return (lambda z, xo: np.exp(1j*k0*(np.outer(xo, kx) + z*kz[None, :])) @ ty)

stable = []
for M in M_list:
    res = solve_rcwa_1d(lam0, 0, 1, 1, D, [(col.astype(complex), H)], 2*M+1, pol=(0., 1.))
    sig = res['Rtot'] + res['Ttot']
    field = propagate(res)
    zc = np.linspace(2, 2.5*f, 200)
    Iax = np.array([abs(field(z, np.array([X0]))[0])**2 for z in zc]); zf = zc[np.argmax(Iax)]
    ok = abs(sig-1) < 0.03
    print(f"  M={M}: ΣDE={sig:.4f} " + (f"✓ kararlı, odak z={zf:.2f}" if ok else "✗ patladı (atlandı)"))
    if ok:
        stable.append((M, res, field, zf, sig))

if not stable:
    print("UYARI: kararlı M bulunamadı. M_list'i genişlet ya da FDTD kullan (09)."); sys.exit(1)
zfs = [s[3] for s in stable]
print(f"\nKararlı çözüm: {len(stable)} adet | odak tutarlılığı z = {np.mean(zfs):.2f} ± {np.std(zfs):.3f} um")

# 5) en iyi kararlı M ile figür
M, res, field, zf, sig = min(stable, key=lambda s: abs(s[4]-1))
xo = np.linspace(X0-4, X0+4, 1401); If = abs(field(zf, xo))**2; If /= If.max()
hw = xo[If >= 0.5]; fwhm = hw.max()-hw.min()
print(f"Seçilen M={M} (ΣDE={sig:.4f}) | ODAK z={zf:.2f}um (tasarım {f}) | FWHM={fwhm*1000:.0f}nm | λ/(2NA)={lam0/(2*NA)*1000:.0f}nm")
zg = np.linspace(1, 2.2*f, 220); xm = np.linspace(X0-4, X0+4, 260)
Imap = np.array([abs(field(z, xm))**2 for z in zg])
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
ax[0].bar(xc-X0, chosen*1000, width=Lcell*0.9); ax[0].set_xlabel("Konum (um)"); ax[0].set_ylabel("Sütun (nm)"); ax[0].set_title("Lens yerleşimi")
ax[1].imshow(Imap.T, origin="lower", aspect="auto", extent=[zg.min(), zg.max(), xm.min()-X0, xm.max()-X0], cmap="inferno")
ax[1].axvline(f, color="cyan", ls="--", lw=1); ax[1].axvline(zf, color="w", ls=":", lw=1)
ax[1].set_xlabel("z (um)"); ax[1].set_ylabel("x (um)"); ax[1].set_title("Tam-dalga I(x,z)")
ax[2].plot((xo-X0)*1000, If); ax[2].axhline(0.5, color="gray", ls=":"); ax[2].set_xlim(-2000, 2000)
ax[2].set_xlabel("x (nm)"); ax[2].set_ylabel("I"); ax[2].set_title(f"Odak — FWHM={fwhm*1000:.0f}nm")
fig.suptitle(f"TAM-DALGA metalens (1B) D={D_lens:.1f}um f={f}um NA={NA:.2f} | {len(stable)} kararlı M, odak z={zf:.2f}um")
fig.tight_layout()
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "Simulasyonlar", "lens_fullwave_1d.png"))
fig.savefig(out, dpi=130); print("Grafik ->", out)
