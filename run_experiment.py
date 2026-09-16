"""RCWA Otomasyon Motoru — Obsidian deney döngüsü.

Akış:  Deneyler/*.md (parametreler)  ->  RCWA çalıştır  ->  Simulasyonlar/*.png
       ->  Raporlar/Rapor - ....md (inceleme + tablo + figür)  ->  deney 'tamamlandi'.

Kullanım:
    python run_experiment.py --all                 # bekleyen tüm deneyleri çalıştır
    python run_experiment.py "Deneyler/xxx.md"     # tek deney dosyası
"""
import os, sys, re, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcwa import solve_rcwa_1d, solve_rcwa_2d, solve_1d_auto, solve_scalar_1d, materials, cache, metrics, shapes, gds

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))          # ...\1_RCWA_Metalens_Projesi
DENEYLER = os.path.join(PROJECT, "Deneyler")
RAPORLAR = os.path.join(PROJECT, "Raporlar")
SIM = os.path.join(PROJECT, "Simulasyonlar")
for d in (DENEYLER, RAPORLAR, SIM):
    os.makedirs(d, exist_ok=True)


# ----------------------------- yardımcılar -----------------------------
def parse_frontmatter(text):
    """Basit YAML-benzeri frontmatter ayrıştırıcı (key: value)."""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
    if not m:
        return {}, text
    fm, body = {}, m.group(1)
    for line in body.splitlines():
        if not line.strip() or line.strip().startswith("#") or line.strip().startswith("-"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip().strip('"').strip("'")
    return fm, m.group(2)


def num(fm, key, default=None, cast=float):
    if key in fm and fm[key] != "":
        try:
            return cast(fm[key])
        except ValueError:
            return default
    return default


def slugify(s):
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.U).strip("_")
    return s[:60] if s else "deney"


def eps_binary(period, eps_hi, eps_lo, duty, Nx=1024):
    x = (np.arange(Nx) + 0.5) / Nx
    return np.where(x < duty, eps_hi, eps_lo).astype(complex)


def eps_pillar(period, w, n_pillar, Nx=512):
    x = (np.arange(Nx) + 0.5) / Nx * period
    return np.where(np.abs(x - period / 2) < w / 2, n_pillar ** 2, 1.0).astype(complex)


def pol_vec(pol):
    return (0.0, 1.0) if str(pol).upper() == "TE" else (1.0, 0.0)


# ----------------------------- deney tipleri -----------------------------
def run_grating(fm, slug):
    lam0 = num(fm, "lam0", 0.55); period = num(fm, "period", 1.2)
    theta = num(fm, "theta", 0.0); pol = fm.get("pol", "TE")
    er_ref = num(fm, "er_ref", 1.0); er_trn = num(fm, "er_trn", 1.0)
    eps_hi = num(fm, "eps_hi", 4.0); eps_lo = num(fm, "eps_lo", 1.0)
    duty = num(fm, "duty", 0.5); d = num(fm, "thickness", 0.3)
    P = int(num(fm, "P", 41, int))
    eps = eps_binary(period, eps_hi, eps_lo, duty)
    r = solve_rcwa_1d(lam0, theta, er_ref, er_trn, period, [(eps, d)], P, pol=pol_vec(pol))
    orders, R, T = r["orders"], r["R"], r["T"]
    prop = (R + T) > 1e-6
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.4
    ax.bar(orders[prop] - w/2, R[prop], w, label="R", color="C3")
    ax.bar(orders[prop] + w/2, T[prop], w, label="T", color="C0")
    ax.set_xlabel("Kırınım mertebesi m"); ax.set_ylabel("Verim (DE)")
    ax.set_title(f"Grating verimleri — {pol}, θ={theta}°"); ax.legend(); ax.grid(True, alpha=.3)
    figpath = os.path.join(SIM, f"{slug}.png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    rows = "".join(f"| {int(m)} | {R[i]:.4f} | {T[i]:.4f} |\n"
                   for i, m in enumerate(orders) if prop[i])
    body = (f"| m | R | T |\n|---|---|---|\n{rows}\n"
            f"**Toplam:** Rtot = {r['Rtot']:.4f}, Ttot = {r['Ttot']:.4f}, "
            f"ΣDE = {r['Rtot']+r['Ttot']:.6f} {'✅' if abs(r['Rtot']+r['Ttot']-1)<1e-4 else '⚠️'}")
    summary = f"ΣDE={r['Rtot']+r['Ttot']:.6f}, Ttot={r['Ttot']:.4f}"
    return figpath, body, summary


def run_metalens_library(fm, slug):
    lam0 = num(fm, "lam0", 0.633); period = num(fm, "period", 0.30)
    H = num(fm, "height", 0.60); n_pillar = num(fm, "n_pillar", 2.4)
    P = int(num(fm, "P", 21, int))
    w_min = num(fm, "w_min", 0.03); w_max = num(fm, "w_max", period - 0.01)
    nW = int(num(fm, "n_width", 40, int))
    widths = np.linspace(w_min, w_max, nW)
    phase, amp = [], []
    for w in widths:
        eps = eps_pillar(period, w, n_pillar)
        r = solve_rcwa_1d(lam0, 0.0, 1.0, 1.0, period, [(eps, H)], P, pol=(0., 1.))
        t0 = r["ty"][r["zeroth_index"]]; phase.append(np.angle(t0)); amp.append(np.abs(t0))
    phase = np.unwrap(np.array(phase)); phase -= phase[0]; amp = np.array(amp)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(widths*1000, phase/np.pi, "o-"); ax[0].set_xlabel("Sütun genişliği (nm)")
    ax[0].set_ylabel("Faz (π)"); ax[0].set_title("Faz kütüphanesi"); ax[0].grid(True, alpha=.3)
    ax[1].plot(widths*1000, amp, "s-", color="C1"); ax[1].set_xlabel("Sütun genişliği (nm)")
    ax[1].set_ylabel("|t0|"); ax[1].set_ylim(0, 1.05); ax[1].set_title("Genlik"); ax[1].grid(True, alpha=.3)
    fig.suptitle(f"Meta-atom taraması (λ={lam0}, Λ={period}, H={H}, n={n_pillar})")
    figpath = os.path.join(SIM, f"{slug}.png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    cov = np.ptp(phase)/np.pi
    body = (f"- **Faz kapsaması:** {cov:.2f}π  {'✅ (≥2π, tam lens mümkün)' if cov>=2 else '⚠️ (<2π, H veya n artır)'}\n"
            f"- **Ortalama genlik:** {amp.mean():.3f}\n"
            f"- Taranan genişlik: {w_min*1000:.0f}–{w_max*1000:.0f} nm ({nW} nokta)")
    summary = f"faz={cov:.2f}π, <|t|>={amp.mean():.3f}"
    return figpath, body, summary


def run_angle_sweep(fm, slug):
    lam0 = num(fm, "lam0", 0.55); period = num(fm, "period", 1.2)
    pol = fm.get("pol", "TE"); er_trn = num(fm, "er_trn", 1.0)
    eps_hi = num(fm, "eps_hi", 4.0); eps_lo = num(fm, "eps_lo", 1.0)
    duty = num(fm, "duty", 0.5); d = num(fm, "thickness", 0.3)
    P = int(num(fm, "P", 41, int))
    a0 = num(fm, "theta_min", 0.0); a1 = num(fm, "theta_max", 60.0); na = int(num(fm, "n_angle", 31, int))
    thetas = np.linspace(a0, a1, na); eps = eps_binary(period, eps_hi, eps_lo, duty)
    Ttot, Rtot = [], []
    for th in thetas:
        r = solve_rcwa_1d(lam0, th, 1.0, er_trn, period, [(eps, d)], P, pol=pol_vec(pol))
        Ttot.append(r["Ttot"]); Rtot.append(r["Rtot"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thetas, Ttot, label="T toplam"); ax.plot(thetas, Rtot, label="R toplam")
    ax.set_xlabel("Geliş açısı θ (°)"); ax.set_ylabel("Verim"); ax.set_ylim(0, 1.02)
    ax.set_title(f"Açı taraması — {pol}"); ax.legend(); ax.grid(True, alpha=.3)
    figpath = os.path.join(SIM, f"{slug}.png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    body = (f"- Açı aralığı: {a0:.0f}–{a1:.0f}° ({na} nokta)\n"
            f"- T aralığı: {min(Ttot):.3f} – {max(Ttot):.3f}\n"
            f"- Enerji: ΣDE ≈ {np.mean(np.array(Ttot)+np.array(Rtot)):.6f}")
    summary = f"T∈[{min(Ttot):.3f},{max(Ttot):.3f}]"
    return figpath, body, summary



def _eps_square_pillar(L, side, eps_hi, eps_lo, N=100):
    x = (np.arange(N) + 0.5) / N * L
    X, Y = np.meshgrid(x, x, indexing='ij')
    e = np.ones((N, N)) * eps_lo
    e[(np.abs(X - L/2) < side/2) & (np.abs(Y - L/2) < side/2)] = eps_hi
    return e


def run_grating2d(fm, slug):
    lam0 = num(fm, "lam0", 0.633); Lx = num(fm, "Lx", num(fm, "period", 1.0))
    Ly = num(fm, "Ly", Lx); theta = num(fm, "theta", 0.0); phi = num(fm, "phi", 0.0)
    pol = fm.get("pol", "TE"); er_ref = num(fm, "er_ref", 1.0); er_trn = num(fm, "er_trn", 1.0)
    eps_hi = num(fm, "eps_hi", 4.0); eps_lo = num(fm, "eps_lo", 1.0)
    side = num(fm, "side", 0.5); d = num(fm, "thickness", 0.3)
    M = int(num(fm, "M", 7, int))
    eps = _eps_square_pillar(Lx, side, eps_hi, eps_lo, 100)
    r = solve_rcwa_2d(lam0, theta, phi, er_ref, er_trn, Lx, Ly, [(eps, d)], M, M, pol=pol_vec(pol))
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].imshow(eps.real.T, origin="lower", extent=[0, Lx, 0, Ly], cmap="viridis")
    ax[0].set_title("Birim hücre ε(x,y)"); ax[0].set_xlabel("x (um)"); ax[0].set_ylabel("y (um)")
    MXf, MYf = r["orders"]; DE = r["R"] + r["T"]
    sc = ax[1].scatter(MXf, MYf, c=DE, s=80, cmap="magma")
    ax[1].set_title("Mertebe verimleri (R+T)"); ax[1].set_xlabel("m"); ax[1].set_ylabel("n")
    plt.colorbar(sc, ax=ax[1]); fig.suptitle(f"2B grating — {pol}, θ={theta}°")
    figpath = os.path.join(SIM, f"{slug}.png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    ok = "✅" if abs(r['Rtot']+r['Ttot']-1) < 1e-4 else "⚠️"
    body = (f"- Rtot = {r['Rtot']:.4f}, Ttot = {r['Ttot']:.4f}, ΣDE = {r['Rtot']+r['Ttot']:.6f} {ok}\n"
            f"- Mertebe sayısı: {(2*M+1)**2}  ((2·{M}+1)²)")
    summary = f"2B ΣDE={r['Rtot']+r['Ttot']:.6f}, Ttot={r['Ttot']:.4f}"
    return figpath, body, summary


def run_metalens_pillar_2d(fm, slug):
    lam0 = num(fm, "lam0", 0.633); L = num(fm, "period", 0.35)
    H = num(fm, "height", 0.60); n_pillar = num(fm, "n_pillar", 2.4)
    M = int(num(fm, "M", 7, int)); s_min = num(fm, "s_min", 0.05)
    s_max = num(fm, "s_max", L - 0.03); nS = int(num(fm, "n_side", 24, int))
    sides = np.linspace(s_min, s_max, nS)
    phase, amp = [], []
    for sdv in sides:
        eps = _eps_square_pillar(L, sdv, n_pillar**2, 1.0, 96)
        r = solve_rcwa_2d(lam0, 0, 0, 1.0, 1.0, L, L, [(eps, H)], M, M, pol=(0., 1.))
        t0 = r["ty"][r["i0"]]; phase.append(np.angle(t0)); amp.append(np.abs(t0))
    phase = np.unwrap(np.array(phase)); phase -= phase[0]; amp = np.array(amp)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(sides*1000, phase/np.pi, "o-"); ax[0].set_xlabel("Sütun kenarı (nm)")
    ax[0].set_ylabel("Faz (π)"); ax[0].set_title("2B faz kütüphanesi"); ax[0].grid(True, alpha=.3)
    ax[1].plot(sides*1000, amp, "s-", color="C1"); ax[1].set_xlabel("Sütun kenarı (nm)")
    ax[1].set_ylabel("|t0|"); ax[1].set_ylim(0, 1.05); ax[1].set_title("Genlik"); ax[1].grid(True, alpha=.3)
    fig.suptitle(f"2B meta-atom (λ={lam0}, Λ={L}, H={H}, n={n_pillar})")
    figpath = os.path.join(SIM, f"{slug}.png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    cov = np.ptp(phase)/np.pi
    body = (f"- **Faz kapsaması:** {cov:.2f}π {'✅ (≥2π)' if cov>=2 else '⚠️ (<2π; H veya n artır)'}\n"
            f"- **Ortalama genlik:** {amp.mean():.3f}\n"
            f"- Kenar taraması: {s_min*1000:.0f}–{s_max*1000:.0f} nm ({nS} nokta), M={M}")
    summary = f"2B faz={cov:.2f}π, <|t|>={amp.mean():.3f}"
    return figpath, body, summary


def run_lens_fullwave_1d(fm, slug):
    """TAM-DALGA 1B metalens: tüm lens tek süper-hücre RCWA + exact yayılım.
    Kötü-koşulluluk için M taranır; enerji koruyan (ΣDE≈1) kararlı çözümler seçilir."""
    lam0 = num(fm, "lam0", 0.633); Lcell = num(fm, "Lcell", 0.35)
    H = num(fm, "height", 0.60); n_pillar = num(fm, "n_pillar", 2.4)
    D_lens = num(fm, "D_lens", 6.0); f = num(fm, "f", 6.0)
    guard = num(fm, "guard", 3.0); Mc = int(num(fm, "M", 140, int))
    k0 = 2 * np.pi / lam0
    Ncell = int(round(D_lens / Lcell)); D_lens = Ncell * Lcell
    D = D_lens + 2 * guard; X0 = D / 2
    NA = np.sin(np.arctan(D_lens / 2 / f))
    widths = np.linspace(0.04, Lcell - 0.04, 60); ph = []
    for w in widths:
        x = (np.arange(256) + 0.5) / 256 * Lcell
        eps = np.where(np.abs(x - Lcell / 2) < w / 2, n_pillar ** 2, 1.0).astype(complex)
        r = solve_rcwa_1d(lam0, 0, 1.0, 1.0, Lcell, [(eps, H)], 21, pol=(0., 1.))
        ph.append(np.angle(r["ty"][r["zeroth_index"]]))
    ph = np.unwrap(np.array(ph)); ph -= ph.min(); phm = np.mod(ph, 2 * np.pi)
    xc = X0 + ((np.arange(Ncell) + 0.5) * Lcell - D_lens / 2)
    phi = np.mod(-(2 * np.pi / lam0) * (np.sqrt((xc - X0) ** 2 + f ** 2) - f), 2 * np.pi)
    chosen = np.array([widths[np.argmin(np.abs(np.angle(np.exp(1j * (phm - p)))))] for p in phi])
    Nx = 4096; xg = (np.arange(Nx) + 0.5) / Nx * D
    col = np.ones(Nx); xs = X0 - D_lens / 2
    for i in range(Ncell):
        c = xs + (i + 0.5) * Lcell
        col[np.abs(xg - c) < chosen[i] / 2] = n_pillar ** 2
    def propagate(res):
        m = res["orders"]; ty = res["ty"]; kx = -m * (lam0 / D)
        kz = np.sqrt((1.0 - kx ** 2).astype(complex)); kz = np.where(np.imag(kz) < 0, -kz, kz)
        return (lambda z, xo: np.exp(1j * k0 * (np.outer(xo, kx) + z * kz[None, :])) @ ty)
    Mset = sorted({max(20, Mc - 20), max(20, Mc - 10), Mc, Mc + 20})
    stable = []
    for M in Mset:
        res = solve_rcwa_1d(lam0, 0, 1.0, 1.0, D, [(col.astype(complex), H)], 2 * M + 1, pol=(0., 1.))
        sig = res["Rtot"] + res["Ttot"]; field = propagate(res)
        zc = np.linspace(2, 2.5 * f, 200)
        zf = zc[int(np.argmax([abs(field(z, np.array([X0]))[0]) ** 2 for z in zc]))]
        if abs(sig - 1) < 0.03:
            stable.append((M, field, zf, sig))
    if not stable:
        figpath = os.path.join(SIM, slug + ".png")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Kararlı M bulunamadı\nFDTD (Meep) kullan", ha="center"); ax.axis("off")
        fig.savefig(figpath, dpi=110); plt.close(fig)
        return figpath, "- ⚠️ Hiçbir mertebede enerji korunmadı (kötü-koşullu). FDTD gerekli.", "kararsız — FDTD gerekli"
    zfs = [s[2] for s in stable]
    M, field, zf, sig = min(stable, key=lambda s: abs(s[3] - 1))
    xo = np.linspace(X0 - 4, X0 + 4, 1401); If = abs(field(zf, xo)) ** 2; If /= If.max()
    hw = xo[If >= 0.5]; fwhm = hw.max() - hw.min()
    zg = np.linspace(1, 2.2 * f, 200); xm = np.linspace(X0 - 4, X0 + 4, 240)
    Imap = np.array([abs(field(z, xm)) ** 2 for z in zg])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].imshow(Imap.T, origin="lower", aspect="auto", extent=[zg.min(), zg.max(), xm.min() - X0, xm.max() - X0], cmap="inferno")
    ax[0].axvline(f, color="cyan", ls="--", lw=1); ax[0].axvline(zf, color="w", ls=":", lw=1)
    ax[0].set_xlabel("z (um)"); ax[0].set_ylabel("x (um)"); ax[0].set_title("Tam-dalga I(x,z)")
    ax[1].plot((xo - X0) * 1000, If); ax[1].set_xlim(-2000, 2000); ax[1].axhline(0.5, color="gray", ls=":")
    ax[1].set_xlabel("x (nm)"); ax[1].set_ylabel("I"); ax[1].set_title(f"Odak — FWHM={fwhm*1000:.0f}nm")
    fig.suptitle(f"TAM-DALGA metalens (1B) D={D_lens:.1f}um f={f}um NA={NA:.2f} | {len(stable)} kararlı M, odak z={zf:.2f}um")
    figpath = os.path.join(SIM, slug + ".png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    body = (f"- **Yöntem:** tüm lens tek süper-hücre RCWA (yerel yaklaşım YOK) + exact yayılım\n"
            f"- **Kararlılık:** {len(stable)}/{len(Mset)} mertebe enerji korudu; odak tutarlılığı z = {np.mean(zfs):.2f} ± {np.std(zfs):.3f} um\n"
            f"- **Gerçek odak:** z = {zf:.2f} um (tasarım f = {f} um) | **FWHM:** {fwhm*1000:.0f} nm (λ/(2NA)={lam0/(2*NA)*1000:.0f} nm)\n"
            f"- **NA:** {NA:.2f} | Sütun: {Ncell} | Seçilen M: {M} | ΣDE: {sig:.4f}\n"
            f"- Not: kalın+yüksek-indis süper-hücre kötü-koşulludur; büyük lens için FDTD (Meep, örn. 09) önerilir.")
    summary = f"odak z={zf:.2f}um (f={f}), FWHM={fwhm*1000:.0f}nm, {len(stable)} kararlı M"
    return figpath, body, summary


def run_metalens_layout_2d(fm, slug):
    """2B metalens tasarımı: faz kütüphanesi -> dairesel yerleşim -> yerel-periyodik odak."""
    lam0 = num(fm, "lam0", 0.633); Lcell = num(fm, "period", 0.35)
    H = num(fm, "height", 0.60); n_pillar = num(fm, "n_pillar", 2.4)
    D_lens = num(fm, "D_lens", 20.0); f = num(fm, "f", 25.0); M = int(num(fm, "M", 7, int))
    k0 = 2 * np.pi / lam0
    sides = np.linspace(0.06, Lcell - 0.05, 20)
    def _lib2d():
        lp = []; la = []
        for s in sides:
            N = 64; x = (np.arange(N) + 0.5) / N * Lcell; X, Y = np.meshgrid(x, x, indexing="ij")
            eps = np.where((np.abs(X - Lcell / 2) < s / 2) & (np.abs(Y - Lcell / 2) < s / 2), n_pillar ** 2, 1.0)
            r = solve_rcwa_2d(lam0, 0, 0, 1, 1, Lcell, Lcell, [(eps, H)], M, M, pol=(0., 1.))
            t0 = r["ty"][r["i0"]]; lp.append(np.angle(t0)); la.append(abs(t0))
        lp = np.unwrap(np.array(lp)); lp -= lp.min()
        return {"libphm": np.mod(lp, 2 * np.pi), "libam": np.array(la)}
    _c = cache.cached_library("lib2d_metalens", {"lam0": lam0, "Lcell": Lcell, "H": H, "n": n_pillar, "M": M, "ns": 20}, _lib2d)
    libphm = _c["libphm"]; libam = _c["libam"]
    # KONVANSİYON DÜZELTMESİ (bkz. metalens_design_full): solve_rcwa_2d geçiş fazı fiziksel
    # (FDTD/üretim) konvansiyonun EŞLENİĞİ → fabrike lensin odaklaması için eşitle. (1B çözücü
    # farklı konvansiyonda; orada eşitleme YAPILMAZ — süper-hücre RCWA ile doğrulandı.)
    libphm = np.mod(-libphm, 2 * np.pi)
    Nc = int(round(D_lens / Lcell)); D_lens = Nc * Lcell
    ci = (np.arange(Nc) - (Nc - 1) / 2) * Lcell
    XX, YY = np.meshgrid(ci, ci, indexing="ij"); R = np.sqrt(XX ** 2 + YY ** 2)
    inside = R <= D_lens / 2
    phi = np.mod(-(2 * np.pi / lam0) * (np.sqrt(R ** 2 + f ** 2) - f), 2 * np.pi)
    side_map = np.zeros_like(R); amp_map = np.zeros_like(R); ph_map = np.zeros_like(R)
    for i in range(Nc):
        for j in range(Nc):
            if inside[i, j]:
                k = np.argmin(np.abs(np.angle(np.exp(1j * (libphm - phi[i, j])))))
                side_map[i, j] = sides[k]; amp_map[i, j] = libam[k]; ph_map[i, j] = libphm[k]
    NA = np.sin(np.arctan(D_lens / 2 / f))
    E0 = np.where(inside, amp_map * np.exp(1j * ph_map), 0.0)
    u = 4; E0f = np.kron(E0, np.ones((u, u))); dx = Lcell / u; Ncf = Nc * u
    pad = int(Ncf * 1.6); E0p = np.zeros((pad, pad), complex); o = (pad - Ncf) // 2; E0p[o:o + Ncf, o:o + Ncf] = E0f
    kxx = 2 * np.pi * np.fft.fftfreq(pad, d=dx); KX, KY = np.meshgrid(kxx, kxx, indexing="ij")
    KZ = np.sqrt((k0 ** 2 - KX ** 2 - KY ** 2).astype(complex)); KZ = np.where(np.imag(KZ) < 0, -KZ, KZ); A0 = np.fft.fft2(E0p)
    prop = lambda z: np.fft.ifft2(A0 * np.exp(1j * KZ * z))
    zs = np.linspace(5, 2 * f, 80); zf = zs[int(np.argmax([np.abs(prop(z)[pad // 2, pad // 2]) ** 2 for z in zs]))]
    I = np.abs(prop(zf)) ** 2; xax = (np.arange(pad) - pad // 2) * dx; cut = I[:, pad // 2] / I.max()
    hw = xax[cut >= 0.5]; fwhm = hw.max() - hw.min()
    rr = np.sqrt((xax[:, None]) ** 2 + (xax[None, :]) ** 2); eff = I[rr <= 1.5 * fwhm].sum() / (np.abs(E0p) ** 2).sum()
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    sm = np.where(inside, side_map * 1000, np.nan)
    im0 = ax[0].imshow(sm, origin="lower", extent=[ci[0], ci[-1], ci[0], ci[-1]], cmap="viridis")
    ax[0].set_title("Sütun kenarı (nm)"); ax[0].set_xlabel("x (um)"); ax[0].set_ylabel("y (um)"); plt.colorbar(im0, ax=ax[0], fraction=0.046)
    ext = xax.max(); im1 = ax[1].imshow(I / I.max(), origin="lower", extent=[-ext, ext, -ext, ext], cmap="inferno")
    ax[1].set_xlim(-3, 3); ax[1].set_ylim(-3, 3); ax[1].set_title(f"Odak I(x,y) z={zf:.1f}um"); ax[1].set_xlabel("x (um)"); ax[1].set_ylabel("y (um)"); plt.colorbar(im1, ax=ax[1], fraction=0.046)
    ax[2].plot(xax * 1000, cut); ax[2].axhline(0.5, color="gray", ls=":"); ax[2].set_xlim(-2500, 2500); ax[2].set_xlabel("x (nm)"); ax[2].set_ylabel("I"); ax[2].set_title(f"FWHM={fwhm*1000:.0f}nm")
    fig.suptitle(f"2B METALENS D={D_lens:.0f}um f={f}um NA={NA:.2f} odak z={zf:.1f}um verim~{eff*100:.0f}%")
    figpath = os.path.join(SIM, slug + ".png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    body = (f"- **Yöntem:** 2B faz kütüphanesi -> dairesel yerleşim -> yerel-periyodik yayılım (tam-dalga DEĞİL)\n"
            f"- **Açıklık:** {int(inside.sum())} sütun, D={D_lens:.1f}um, NA={NA:.2f}\n"
            f"- **Odak:** z={zf:.1f}um (tasarım f={f}um) | FWHM={fwhm*1000:.0f}nm (λ/(2NA)={lam0/(2*NA)*1000:.0f}nm)\n"
            f"- **Odaklama verimi (yerel-periyodik):** ~{eff*100:.0f}%\n"
            f"- Nihai doğrulama için FDTD/GPU (bkz. 09).")
    summary = f"2B lens odak z={zf:.1f}um, FWHM={fwhm*1000:.0f}nm, verim~{eff*100:.0f}%"
    return figpath, body, summary


def _eps_rect(wx, wy, ang, L, n_pillar, N=80):
    x = (np.arange(N) + 0.5) / N * L
    X, Y = np.meshgrid(x - L / 2, x - L / 2, indexing="ij")
    ca, sa = np.cos(ang), np.sin(ang)
    Xr = ca * X + sa * Y; Yr = -sa * X + ca * Y
    return np.where((np.abs(Xr) < wx / 2) & (np.abs(Yr) < wy / 2), n_pillar ** 2, 1.0)


def _jones(eps, lam0, L, H, M):
    rx = solve_rcwa_2d(lam0, 0, 0, 1, 1, L, L, [(eps, H)], M, M, pol=(1., 0.))
    ry = solve_rcwa_2d(lam0, 0, 0, 1, 1, L, L, [(eps, H)], M, M, pol=(0., 1.))
    txx = rx["tx"][rx["i0"]]; tyx = rx["ty"][rx["i0"]]
    txy = ry["tx"][ry["i0"]]; tyy = ry["ty"][ry["i0"]]
    return np.array([[txx, txy], [tyx, tyy]])


def run_wavelength_sweep(fm, slug):
    L = num(fm, "period", 0.35); H = num(fm, "height", 0.60)
    n_pillar = num(fm, "n_pillar", 2.4); side = num(fm, "side", 0.18); M = int(num(fm, "M", 8, int))
    mat = fm.get("material", "")   # örn "TiO2","Si","SiO2"; boşsa sabit n_pillar
    l0 = num(fm, "lam_min", 0.45); l1 = num(fm, "lam_max", 0.75); nl = int(num(fm, "n_lam", 25, int))
    lams = np.linspace(l0, l1, nl); amp = []; ph = []; Tt = []
    for lam in lams:
        eps_hi = materials.eps_of(mat, lam) if mat else n_pillar ** 2
        N = 80; x = (np.arange(N) + 0.5) / N * L; X, Y = np.meshgrid(x, x, indexing="ij")
        eps = np.where((np.abs(X - L / 2) < side / 2) & (np.abs(Y - L / 2) < side / 2), eps_hi, 1.0)
        r = solve_rcwa_2d(lam, 0, 0, 1, 1, L, L, [(eps, H)], M, M, pol=(0., 1.))
        t0 = r["ty"][r["i0"]]; amp.append(abs(t0)); ph.append(np.angle(t0)); Tt.append(r["Ttot"])
    ph = np.unwrap(np.array(ph))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(lams * 1000, Tt, label="T toplam"); ax[0].plot(lams * 1000, amp, "--", label="|t0|")
    ax[0].set_xlabel("Dalga boyu (nm)"); ax[0].set_ylabel("Verim/genlik"); ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title("Spektral yanıt")
    ax[1].plot(lams * 1000, (ph - ph[0]) / np.pi, color="C2"); ax[1].set_xlabel("Dalga boyu (nm)"); ax[1].set_ylabel("Faz (π)"); ax[1].grid(alpha=.3); ax[1].set_title("0. mertebe faz")
    fig.suptitle(f"Dalga boyu taraması — sütun {side*1000:.0f}nm, H={H}um, n={n_pillar}"); fig.tight_layout()
    figpath = os.path.join(SIM, slug + ".png"); fig.savefig(figpath, dpi=130); plt.close(fig)
    body = (f"- Aralık: {l0*1000:.0f}–{l1*1000:.0f} nm ({nl} nokta) | malzeme: {mat or ('n=%.2f'%n_pillar)}\n"
            f"- T: {min(Tt):.3f}–{max(Tt):.3f} | faz kapsaması: {np.ptp(ph)/np.pi:.2f}π")
    return figpath, body, f"λ∈[{l0*1000:.0f},{l1*1000:.0f}]nm, T∈[{min(Tt):.2f},{max(Tt):.2f}]"


def run_birefringent_library(fm, slug):
    L = num(fm, "period", 0.35); H = num(fm, "height", 0.60); lam0 = num(fm, "lam0", 0.633)
    n_pillar = num(fm, "n_pillar", 2.4); M = int(num(fm, "M", 6, int))
    w0 = num(fm, "w_min", 0.06); w1 = num(fm, "w_max", 0.30); nw = int(num(fm, "n_w", 8, int))
    ws = np.linspace(w0, w1, nw)
    PX = np.zeros((nw, nw)); PY = np.zeros((nw, nw)); AX = np.zeros((nw, nw))
    for i, wx in enumerate(ws):
        for j, wy in enumerate(ws):
            J = _jones(_eps_rect(wx, wy, 0.0, L, n_pillar), lam0, L, H, M)
            PX[i, j] = np.angle(J[0, 0]); PY[i, j] = np.angle(J[1, 1]); AX[i, j] = abs(J[0, 0])
    ret = np.angle(np.exp(1j * (PX - PY)))
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, D, tit in zip(ax, [PX / np.pi, PY / np.pi, ret / np.pi], ["faz_x (π)", "faz_y (π)", "retardans φx-φy (π)"]):
        im = a.imshow(D, origin="lower", extent=[w0*1000, w1*1000, w0*1000, w1*1000], cmap="twilight")
        a.set_xlabel("wy (nm)"); a.set_ylabel("wx (nm)"); a.set_title(tit); plt.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"Çift-kırılımlı meta-atom kütüphanesi (λ={lam0}, H={H}, n={n_pillar})"); fig.tight_layout()
    figpath = os.path.join(SIM, slug + ".png"); fig.savefig(figpath, dpi=130); plt.close(fig)
    body = (f"- wx,wy ∈ {w0*1000:.0f}–{w1*1000:.0f} nm ({nw}×{nw} ızgara)\n"
            f"- Maks retardans: {np.max(np.abs(ret))/np.pi:.2f}π (yarım-dalga için ~1π gerekir)")
    return figpath, body, f"çift-kırılım {nw}x{nw}, maks retardans {np.max(np.abs(ret))/np.pi:.2f}π"


def run_jones_matrix(fm, slug):
    L = num(fm, "period", 0.35); H = num(fm, "height", 0.60); lam0 = num(fm, "lam0", 0.633)
    n_pillar = num(fm, "n_pillar", 2.4); M = int(num(fm, "M", 8, int))
    wx = num(fm, "wx", 0.26); wy = num(fm, "wy", 0.10); ang = np.deg2rad(num(fm, "angle", 0.0))
    J = _jones(_eps_rect(wx, wy, ang, L, n_pillar), lam0, L, H, M)
    Lc = np.array([1, 1j]) / np.sqrt(2); Rc = np.array([1, -1j]) / np.sqrt(2)
    tRL = Rc.conj() @ (J @ Lc); tLL = Lc.conj() @ (J @ Lc)
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["txx", "txy", "tyx", "tyy"]; vals = [abs(J[0,0]), abs(J[0,1]), abs(J[1,0]), abs(J[1,1])]
    ax.bar(labels, vals, color="C0"); ax.set_ylim(0, 1.05); ax.set_ylabel("|t|"); ax.set_title("Jones matris genlikleri")
    figpath = os.path.join(SIM, slug + ".png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    body = (f"- **Jones (genlik∠faz°):**\n"
            f"  - txx={abs(J[0,0]):.2f}∠{np.angle(J[0,0],deg=True):.0f}  txy={abs(J[0,1]):.2f}∠{np.angle(J[0,1],deg=True):.0f}\n"
            f"  - tyx={abs(J[1,0]):.2f}∠{np.angle(J[1,0],deg=True):.0f}  tyy={abs(J[1,1]):.2f}∠{np.angle(J[1,1],deg=True):.0f}\n"
            f"- Retardans (φx−φy): {(np.angle(J[0,0])-np.angle(J[1,1]))*180/np.pi:.0f}°\n"
            f"- Çapraz-dairesel |tRL|={abs(tRL):.2f}, eş-dairesel |tLL|={abs(tLL):.2f}  (iyi PB için |tRL|→1)")
    return figpath, body, f"retardans {(np.angle(J[0,0])-np.angle(J[1,1]))*180/np.pi:.0f}°, |tRL|={abs(tRL):.2f}"


def run_pb_phase_library(fm, slug):
    L = num(fm, "period", 0.35); H = num(fm, "height", 0.60); lam0 = num(fm, "lam0", 0.633)
    n_pillar = num(fm, "n_pillar", 2.4); M = int(num(fm, "M", 6, int))
    wx = num(fm, "wx", 0.28); wy = num(fm, "wy", 0.09); na = int(num(fm, "n_angle", 13, int))
    angs = np.linspace(0, 180, na); phase = []; amp = []
    Lc = np.array([1, 1j]) / np.sqrt(2); Rc = np.array([1, -1j]) / np.sqrt(2)
    for a in angs:
        J = _jones(_eps_rect(wx, wy, np.deg2rad(a), L, n_pillar), lam0, L, H, M)
        tRL = Rc.conj() @ (J @ Lc); phase.append(np.angle(tRL)); amp.append(abs(tRL))
    phase = np.unwrap(np.array(phase)); phase -= phase[0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(angs, phase / np.pi, "o-", label="ölçülen"); ax[0].plot(angs, 2 * angs / 180, "--", label="2θ (teori)")
    ax[0].set_xlabel("Dönme θ (°)"); ax[0].set_ylabel("Çapraz-dairesel faz (π)"); ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title("PB geometrik faz")
    ax[1].plot(angs, amp, "s-", color="C1"); ax[1].set_ylim(0, 1.05); ax[1].set_xlabel("θ (°)"); ax[1].set_ylabel("|tRL|"); ax[1].grid(alpha=.3); ax[1].set_title("Dönüşüm verimi")
    fig.suptitle(f"PB (geometrik) faz kütüphanesi — çubuk {wx*1000:.0f}×{wy*1000:.0f}nm"); fig.tight_layout()
    figpath = os.path.join(SIM, slug + ".png"); fig.savefig(figpath, dpi=130); plt.close(fig)
    lin = np.corrcoef(phase / np.pi, 2 * angs / 180)[0, 1]
    body = (f"- Çubuk: {wx*1000:.0f}×{wy*1000:.0f} nm, {na} açı\n"
            f"- 2θ ile doğrusallık (korelasyon): {lin:.4f}  | ort. |tRL|: {np.mean(amp):.2f}")
    return figpath, body, f"PB doğrusallık r={lin:.3f}, <|tRL|>={np.mean(amp):.2f}"


def run_param_map_2d(fm, slug):
    L = num(fm, "period", 0.35); lam0 = num(fm, "lam0", 0.633); n_pillar = num(fm, "n_pillar", 2.4); M = int(num(fm, "M", 6, int))
    h0 = num(fm, "h_min", 0.30); h1 = num(fm, "h_max", 0.90); nh = int(num(fm, "n_h", 8, int))
    s0 = num(fm, "s_min", 0.06); s1 = num(fm, "s_max", 0.30); ns = int(num(fm, "n_s", 8, int))
    hs = np.linspace(h0, h1, nh); ss = np.linspace(s0, s1, ns)
    PH = np.zeros((nh, ns)); AM = np.zeros((nh, ns))
    for i, H in enumerate(hs):
        for j, s in enumerate(ss):
            N = 64; x = (np.arange(N) + 0.5) / N * L; X, Y = np.meshgrid(x, x, indexing="ij")
            eps = np.where((np.abs(X - L / 2) < s / 2) & (np.abs(Y - L / 2) < s / 2), n_pillar ** 2, 1.0)
            r = solve_rcwa_2d(lam0, 0, 0, 1, 1, L, L, [(eps, H)], M, M, pol=(0., 1.))
            t0 = r["ty"][r["i0"]]; PH[i, j] = np.angle(t0); AM[i, j] = abs(t0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    im0 = ax[0].imshow(PH / np.pi, origin="lower", extent=[s0*1000, s1*1000, h0, h1], aspect="auto", cmap="twilight")
    ax[0].set_xlabel("Sütun kenarı (nm)"); ax[0].set_ylabel("Yükseklik (um)"); ax[0].set_title("0. mertebe faz (π)"); plt.colorbar(im0, ax=ax[0], fraction=0.046)
    im1 = ax[1].imshow(AM, origin="lower", extent=[s0*1000, s1*1000, h0, h1], aspect="auto", cmap="viridis")
    ax[1].set_xlabel("Sütun kenarı (nm)"); ax[1].set_ylabel("Yükseklik (um)"); ax[1].set_title("|t0|"); plt.colorbar(im1, ax=ax[1], fraction=0.046)
    fig.suptitle(f"2-parametre harita (yükseklik×genişlik) — λ={lam0}, n={n_pillar}"); fig.tight_layout()
    figpath = os.path.join(SIM, slug + ".png"); fig.savefig(figpath, dpi=130); plt.close(fig)
    body = (f"- Yükseklik {h0}–{h1}um × kenar {s0*1000:.0f}–{s1*1000:.0f}nm ({nh}×{ns})\n"
            f"- Faz kapsaması: {np.ptp(PH)/np.pi:.2f}π | genlik {AM.min():.2f}–{AM.max():.2f}")
    return figpath, body, f"harita {nh}x{ns}, faz {np.ptp(PH)/np.pi:.2f}π"


def _prop2d(E0, dx, k0):
    N = E0.shape[0]
    kxx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    KX, KY = np.meshgrid(kxx, kxx, indexing="ij")
    KZ = np.sqrt((k0 ** 2 - KX ** 2 - KY ** 2).astype(complex)); KZ = np.where(np.imag(KZ) < 0, -KZ, KZ)
    A0 = np.fft.fft2(E0)
    return lambda z: np.fft.ifft2(A0 * np.exp(1j * KZ * z))


def run_inverse_design_deflector(fm, slug):
    """C8: bir birim hücreyi hedef mertebeye (ışın saptırıcı) topoloji optimizasyonu."""
    from scipy.optimize import differential_evolution
    lam0 = num(fm, "lam0", 0.633); period = num(fm, "period", 1.30); H = num(fm, "height", 0.60)
    eps_hi = num(fm, "eps_hi", 6.25); Nseg = int(num(fm, "n_seg", 12, int))
    target = int(num(fm, "target_order", 1, int)); Mo = int(num(fm, "M", 15, int))
    maxiter = int(num(fm, "maxiter", 18, int)); Nx = Nseg * 40
    def make(dens): return np.repeat(1.0 + np.clip(dens, 0, 1) * (eps_hi - 1.0), Nx // Nseg)
    def Tt(dens):
        r = solve_scalar_1d(lam0, 0, 1, 1, period, make(dens), H, Mo, "TE", "laurent")
        return r["T"][np.where(r["orders"] == target)[0][0]]
    import numpy as _np; _np.random.seed(0); T0 = Tt(_np.random.rand(Nseg))
    res = differential_evolution(lambda d: -Tt(d), [(0, 1)] * Nseg, maxiter=maxiter,
                                 popsize=8, seed=1, polish=False, tol=1e-3)
    dens = res.x; eps = make(dens)
    r = solve_scalar_1d(lam0, 0, 1, 1, period, eps, H, Mo, "TE", "laurent")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].step(np.arange(Nseg), np.clip(dens, 0, 1), where="mid"); ax[0].set_ylim(-0.05, 1.05)
    ax[0].set_xlabel("Segment"); ax[0].set_ylabel("Dolgu (0=hava,1=dielektrik)"); ax[0].set_title("Optimize birim hücre")
    o = r["orders"]; T = r["T"]; sel = T > 0.005
    ax[1].bar(o[sel], T[sel]); ax[1].set_xlabel("Mertebe"); ax[1].set_ylabel("T"); ax[1].set_title(f"Mertebe verimleri (hedef {target:+d})")
    fig.suptitle(f"Ters tasarım — {target:+d}. mertebe saptırıcı"); fig.tight_layout()
    figpath = os.path.join(SIM, slug + ".png"); fig.savefig(figpath, dpi=130); plt.close(fig)
    body = (f"- **Başlangıç (rastgele) T[{target:+d}]:** {T0:.3f}\n"
            f"- **Optimize T[{target:+d}]:** {-res.fun:.3f}  ({res.nfev} değerlendirme)\n"
            f"- ΣDE = {r['Rtot']+r['Ttot']:.4f} | segment: {Nseg}, mertebe: {Mo}\n"
            f"- Yöntem: differential_evolution (gradyansız); ileri model = skaler RCWA.")
    return figpath, body, f"ters tasarım T[{target:+d}]: {T0:.2f}->{-res.fun:.2f}"


def run_achromatic_analysis(fm, slug):
    """C10: 1B silindirik metalensin dalga boyuna gore odak kaymasi (kromatik aberasyon).
    Her lambda'da gercek RCWA fazi (solve_rcwa_1d karmasik ty) kullanilir."""
    Lcell = num(fm, "period", 0.35); H = num(fm, "height", 0.60); n_pillar = num(fm, "n_pillar", 2.4)
    D_lens = num(fm, "D_lens", 12.0); f0 = num(fm, "f", 15.0); lam0 = num(fm, "lam0", 0.633)
    l0 = num(fm, "lam_min", 0.55); l1 = num(fm, "lam_max", 0.72); nl = int(num(fm, "n_lam", 9, int))
    Ncell = int(round(D_lens / Lcell)); D_lens = Ncell * Lcell
    widths = np.linspace(0.05, Lcell - 0.05, 28)
    lams = np.linspace(l0, l1, nl)
    # her lambda'da (genislik -> 0. mertebe faz) kutuphanesi
    def phase_at(lam):
        ph = []
        for w in widths:
            x = (np.arange(128) + 0.5) / 128 * Lcell
            eps = np.where(np.abs(x - Lcell/2) < w/2, n_pillar**2, 1.0).astype(complex)
            r = solve_rcwa_1d(lam, 0, 1.0, 1.0, Lcell, [(eps, H)], 15, pol=(0., 1.))
            ph.append(np.angle(r["ty"][r["zeroth_index"]]))
        return np.array(ph)
    PH = {round(lam, 5): phase_at(lam) for lam in lams}
    ph0 = np.unwrap(PH[round(lam0 if lam0 in lams else lams[np.argmin(abs(lams-lam0))],5)]) if False else np.unwrap(phase_at(lam0))
    ph0 -= ph0.min(); ph0m = np.mod(ph0, 2*np.pi)
    xc = (np.arange(Ncell) - (Ncell-1)/2) * Lcell
    phi_need = np.mod(-(2*np.pi/lam0) * (np.sqrt(xc**2 + f0**2) - f0), 2*np.pi)
    chosen = np.array([np.argmin(np.abs(np.angle(np.exp(1j*(ph0m - p))))) for p in phi_need])
    focus = []
    for lam in lams:
        atom_ph = PH[round(lam,5)][chosen]           # secilen atomlarin BU lambda'daki fazi
        E0 = np.exp(1j*atom_ph)
        Npad = 4096; dx = Lcell; Ec = np.zeros(Npad, complex); o=(Npad-Ncell)//2; Ec[o:o+Ncell]=E0
        kx = 2*np.pi*np.fft.fftfreq(Npad, d=dx); kz=np.sqrt(((2*np.pi/lam)**2-kx**2).astype(complex)); kz=np.where(np.imag(kz)<0,-kz,kz)
        A0=np.fft.fft(Ec); zc=np.linspace(0.4*f0, 1.8*f0, 140)
        Ia=[np.abs(np.fft.ifft(A0*np.exp(1j*kz*z))[Npad//2])**2 for z in zc]
        focus.append(zc[int(np.argmax(Ia))])
    focus=np.array(focus)
    fig, ax = plt.subplots(figsize=(7,4.2))
    ax.plot(lams*1000, focus, "o-"); ax.axhline(f0,color="gray",ls=":",label="tasarim f")
    ax.axvline(lam0*1000,color="C3",ls="--",label="tasarim lambda")
    ax.set_xlabel("Dalga boyu (nm)"); ax.set_ylabel("Odak z (um)")
    ax.set_title("Kromatik odak kaymasi (RCWA fazli)"); ax.legend(); ax.grid(alpha=.3)
    figpath=os.path.join(SIM, slug+".png"); fig.tight_layout(); fig.savefig(figpath,dpi=130); plt.close(fig)
    shift=focus.max()-focus.min()
    body=(f"- Tasarim: D={D_lens:.1f}um, f={f0}um @ lambda={lam0*1000:.0f}nm (her lambda'da gercek RCWA fazi)\n"
          f"- lambda {l0*1000:.0f}-{l1*1000:.0f}nm'de odak {focus.min():.1f}-{focus.max():.1f}um -> **kromatik kayma {shift:.1f}um**\n"
          f"- Akromatik tasarim: meta-atom dispersiyonu (dphi/dlambda) esitlenerek kayma giderilir.")
    return figpath, body, f"kromatik odak kaymasi {shift:.1f}um"


def run_vortex_beam(fm, slug):
    """C11: OAM vorteks üreteci — azimutal faz exp(i·l·θ) (PB ile gerçeklenebilir)."""
    lam0 = num(fm, "lam0", 0.633); D = num(fm, "aperture", 12.0); ell = int(num(fm, "charge", 1, int))
    zf = num(fm, "z", 20.0); k0 = 2 * np.pi / lam0
    N = 512; ext = D * 1.6; dx = ext / N
    x = (np.arange(N) - N / 2) * dx; X, Y = np.meshgrid(x, x, indexing="ij")
    R = np.hypot(X, Y); TH = np.arctan2(Y, X)
    E0 = np.where(R <= D / 2, np.exp(1j * ell * TH), 0.0)
    prop = _prop2d(E0, dx, k0); Ez = prop(zf); I = np.abs(Ez) ** 2; ph = np.angle(Ez)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].imshow(np.angle(E0), extent=[-ext/2, ext/2, -ext/2, ext/2], cmap="twilight"); ax[0].set_title(f"Girdi faz (l={ell})")
    ax[1].imshow(I / I.max(), extent=[-ext/2, ext/2, -ext/2, ext/2], cmap="inferno"); ax[1].set_xlim(-4, 4); ax[1].set_ylim(-4, 4); ax[1].set_title(f"Şiddet @ z={zf}um (donut)")
    ax[2].imshow(ph, extent=[-ext/2, ext/2, -ext/2, ext/2], cmap="twilight"); ax[2].set_xlim(-4, 4); ax[2].set_ylim(-4, 4); ax[2].set_title("Faz (tekillik)")
    fig.suptitle(f"OAM vorteks demeti — topolojik yük l={ell}"); fig.tight_layout()
    figpath = os.path.join(SIM, slug + ".png"); fig.savefig(figpath, dpi=130); plt.close(fig)
    cen = I[N//2, N//2] / I.max()
    body = (f"- Topolojik yük l={ell}, açıklık D={D}um\n"
            f"- Merkez şiddeti/maks = {cen:.3f} (vorteks için ~0; donut halka)\n"
            f"- Faz merkezde 2πl'lik tekillik. PB (dönen çubuk) meta-atomlarla gerçeklenir.")
    return figpath, body, f"vorteks l={ell}, merkez I~{cen:.2f}"


def run_hologram(fm, slug):
    """C11: faz-yalnız hologram (Gerchberg-Saxton) — hedef deseni yeniden oluşturur."""
    N = int(num(fm, "N", 128, int)); iters = int(num(fm, "iters", 30, int))
    # hedef: basit bir 'L' ya da halka deseni
    target = np.zeros((N, N)); c = N // 2
    yy, xx = np.mgrid[0:N, 0:N]; rr = np.hypot(xx - c, yy - c)
    target[(rr > N * 0.18) & (rr < N * 0.24)] = 1.0  # halka
    target[(np.abs(xx - c) < 2) & (yy > c)] = 1.0     # çizgi
    amp = np.sqrt(target); 
    field = np.exp(1j * 2 * np.pi * np.random.rand(N, N))
    for _ in range(iters):
        F = np.fft.fftshift(np.fft.fft2(field))
        F = amp * np.exp(1j * np.angle(F))
        field = np.fft.ifft2(np.fft.ifftshift(F))
        field = np.exp(1j * np.angle(field))   # faz-yalnız kısıt
    recon = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2; recon /= recon.max()
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].imshow(target, cmap="gray"); ax[0].set_title("Hedef desen")
    ax[1].imshow(np.angle(field), cmap="twilight"); ax[1].set_title("Faz maskesi (metayüzey)")
    ax[2].imshow(recon, cmap="inferno"); ax[2].set_title("Yeniden oluşum")
    fig.suptitle("Faz-yalnız hologram (Gerchberg-Saxton)"); fig.tight_layout()
    figpath = os.path.join(SIM, slug + ".png"); fig.savefig(figpath, dpi=130); plt.close(fig)
    err = np.mean((recon - target / target.max()) ** 2)
    body = (f"- {N}×{N} faz maskesi, {iters} GS iterasyonu\n"
            f"- Yeniden oluşum ortalama kare hata ~ {err:.3f}\n"
            f"- Faz maskesi meta-atom faz kütüphanesiyle gerçeklenir.")
    return figpath, body, f"hologram GS, hata~{err:.3f}"


def run_polarization_multiplexed(fm, slug):
    """C11: polarizasyon-çoğullama — x ve y polarizasyonu farklı yöne saptır (çift-kırılım)."""
    lam0 = num(fm, "lam0", 0.633); D = num(fm, "aperture", 12.0)
    gx = num(fm, "grad_x", 0.15); gy = num(fm, "grad_y", -0.15); zf = num(fm, "z", 18.0)
    k0 = 2 * np.pi / lam0; N = 512; ext = D * 1.8; dx = ext / N
    x = (np.arange(N) - N / 2) * dx; X, Y = np.meshgrid(x, x, indexing="ij"); R = np.hypot(X, Y)
    mask = R <= D / 2
    Ex = np.where(mask, np.exp(1j * k0 * gx * X), 0.0)   # x-pol: +yöne
    Ey = np.where(mask, np.exp(1j * k0 * gy * X), 0.0)   # y-pol: -yöne
    Ix = np.abs(_prop2d(Ex, dx, k0)(zf)) ** 2; Iy = np.abs(_prop2d(Ey, dx, k0)(zf)) ** 2
    def peak_x(I): 
        j = np.unravel_index(np.argmax(I), I.shape); return x[j[0]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].imshow(Ix / Ix.max(), extent=[-ext/2, ext/2, -ext/2, ext/2], cmap="inferno"); ax[0].set_title(f"x-pol @ z={zf}um (tepe x={peak_x(Ix):+.1f})")
    ax[1].imshow(Iy / Iy.max(), extent=[-ext/2, ext/2, -ext/2, ext/2], cmap="inferno"); ax[1].set_title(f"y-pol @ z={zf}um (tepe x={peak_x(Iy):+.1f})")
    fig.suptitle("Polarizasyon-çoğullama: x ve y farklı yöne (çift-kırılımlı meta-atom)"); fig.tight_layout()
    figpath = os.path.join(SIM, slug + ".png"); fig.savefig(figpath, dpi=130); plt.close(fig)
    body = (f"- x-pol tepe konumu x={peak_x(Ix):+.1f}um, y-pol x={peak_x(Iy):+.1f}um → zıt yönler\n"
            f"- Çift-kırılımlı (wx≠wy) meta-atomlar iki polarizasyona bağımsız faz verir (Jones).")
    return figpath, body, f"polmux: x->{peak_x(Ix):+.1f}, y->{peak_x(Iy):+.1f}"


def run_metalens_design_full(fm, slug):
    """Madde 2+3+4 amiral gemisi: şekil+malzeme kütüphanesi -> 2B lens tasarımı ->
    metrikler (FWHM, verim, Strehl, MTF) -> üretilebilir GDS. Tam paket."""
    lam0 = num(fm, "lam0", 0.633); L = num(fm, "period", 0.35); H = num(fm, "height", 0.60)
    mat = fm.get("material", "TiO2_t"); shape = fm.get("shape", "circle")
    D_lens = num(fm, "D_lens", 16.0); f = num(fm, "f", 20.0); M = int(num(fm, "M", 7, int))
    k0 = 2 * np.pi / lam0
    n_pillar = float(np.real(np.sqrt(materials.eps_of(mat, lam0)))) if mat else num(fm, "n_pillar", 2.4)
    sizes = np.linspace(0.06, L - 0.05, 20)
    def _lib():
        lp = []; la = []
        for s in sizes:
            eps = shapes.rasterize(shape, s, L, 64, n_pillar=n_pillar)
            r = solve_rcwa_2d(lam0, 0, 0, 1, 1, L, L, [(eps, H)], M, M, pol=(0., 1.))
            t0 = r["ty"][r["i0"]]; lp.append(np.angle(t0)); la.append(abs(t0))
        lp = np.unwrap(np.array(lp)); lp -= lp.min()
        return {"phm": np.mod(lp, 2 * np.pi), "am": np.array(la)}
    c = cache.cached_library("libdesign", {"lam0": lam0, "L": L, "H": H, "mat": mat, "shape": shape, "M": M, "n": round(n_pillar, 3)}, _lib)
    libphm = c["phm"]; libam = c["am"]
    # KONVANSİYON DÜZELTMESİ (Tidy3D FDTD ile doğrulandı, D=6um lens):
    # solve_rcwa_2d'nin geçiş fazı, fiziksel (FDTD/üretim) konvansiyonun EŞLENİĞİdir
    # (birim-hücre testi: boyut-faz eğilimi RCWA vs FDTD zıt). Eşleniklemezsek fiziksel
    # yapı IRAKSAK lens olur (kenara saçılır). Eşleniklenince odak difraksiyon-sınırlı.
    libphm = np.mod(-libphm, 2 * np.pi)
    Nc = int(round(D_lens / L)); D_lens = Nc * L
    ci = (np.arange(Nc) - (Nc - 1) / 2) * L
    XX, YY = np.meshgrid(ci, ci, indexing="ij"); R = np.hypot(XX, YY)
    inside = R <= D_lens / 2
    phi_t = np.mod(-(2 * np.pi / lam0) * (np.sqrt(R ** 2 + f ** 2) - f), 2 * np.pi)
    size_map = np.zeros_like(R); E0 = np.zeros_like(R, complex); E0_ideal = np.zeros_like(R, complex)
    placements = []
    for i in range(Nc):
        for j in range(Nc):
            if inside[i, j]:
                idx = np.argmin(np.abs(np.angle(np.exp(1j * (libphm - phi_t[i, j])))))
                size_map[i, j] = sizes[idx]
                E0[i, j] = libam[idx] * np.exp(1j * libphm[idx])
                E0_ideal[i, j] = libam[idx] * np.exp(1j * phi_t[i, j])
                placements.append({"x": ci[i], "y": ci[j], "shape": shape, "params": float(sizes[idx]), "angle": 0.0})
    # yayılım (ince grid)
    u = 4; dx = L / u
    def prop_stack(Efield):
        Ef = np.kron(Efield, np.ones((u, u))); Ncf = Nc * u
        pad = int(Ncf * 1.5); Ep = np.zeros((pad, pad), complex); o = (pad - Ncf) // 2; Ep[o:o + Ncf, o:o + Ncf] = Ef
        return Ep, pad
    Ep, pad = prop_stack(E0); Epi, _ = prop_stack(E0_ideal)
    xax = (np.arange(pad) - pad // 2) * dx
    kx = 2 * np.pi * np.fft.fftfreq(pad, d=dx); KX, KY = np.meshgrid(kx, kx, indexing="ij")
    KZ = np.sqrt((k0 ** 2 - KX ** 2 - KY ** 2).astype(complex)); KZ = np.where(np.imag(KZ) < 0, -KZ, KZ)
    A0 = np.fft.fft2(Ep); A0i = np.fft.fft2(Epi)
    zc = np.linspace(0.4 * f, 1.8 * f, 60)
    Iz_axis = np.array([np.abs(np.fft.ifft2(A0 * np.exp(1j * KZ * z))[pad // 2, pad // 2]) ** 2 for z in zc])
    zf = zc[int(np.argmax(Iz_axis))]
    I = np.abs(np.fft.ifft2(A0 * np.exp(1j * KZ * zf))) ** 2
    zfi = zc[int(np.argmax([np.abs(np.fft.ifft2(A0i * np.exp(1j * KZ * z))[pad // 2, pad // 2]) ** 2 for z in zc]))]
    Iideal = np.abs(np.fft.ifft2(A0i * np.exp(1j * KZ * zfi))) ** 2
    met = metrics.focal_metrics(I, xax); fwhm = met["fwhm_x"]
    eff = metrics.focusing_efficiency(I, xax, fwhm)
    St = float(I.max() / Iideal.max())   # tepe/tepe (aynı genlik, ideal faz) -> <=1
    NA = np.sin(np.arctan(D_lens / 2 / f))
    # --- PROFESYONEL METRİKLER (literatür-standardı) ---
    P_inc = float(inside.sum()) * (u * u)                          # gelen güç (birim genlik açıklık)
    esuite = metrics.efficiency_suite(I, xax, fwhm, P_incident=P_inc)
    ee80 = metrics.encircled_energy(I, xax, 0.8)                   # %80 enerji yarıçapı (um)
    slobe = metrics.sidelobe_level(I[:, met["peak_ij"][1]])       # yan-lob (dB)
    dof = metrics.depth_of_focus(Iz_axis, zc)                      # odak derinliği (um)
    st_mtf = metrics.strehl_mtf(metrics.mtf_2d(I), metrics.mtf_2d(Iideal))  # MTF-tabanlı Strehl
    dl_fwhm = lam0 / (2 * NA) * 1000                               # difraksiyon sınırı (nm)
    # GDS
    gdspath = os.path.join(SIM, slug + ".gds"); npoly = gds.layout_to_gds(placements, gdspath)
    import json
    design = {"lam0": lam0, "Lcell": L, "height": H, "n_pillar": n_pillar, "shape": shape,
              "D_lens": D_lens, "f": f, "material": mat,
              "placements": [{"x": float(p["x"]), "y": float(p["y"]), "params": float(p["params"]), "angle": 0.0} for p in placements]}
    jsonpath = os.path.join(SIM, slug + "_design.json")
    with open(jsonpath, "w") as _jf: json.dump(design, _jf)
    # figür
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    sm = np.where(inside, size_map * 1000, np.nan)
    im0 = ax[0].imshow(sm, origin="lower", extent=[ci[0], ci[-1], ci[0], ci[-1]], cmap="viridis")
    ax[0].set_title(f"{shape} boyut (nm)"); ax[0].set_xlabel("x(um)"); ax[0].set_ylabel("y(um)"); plt.colorbar(im0, ax=ax[0], fraction=0.046)
    ext = xax.max(); ax[1].imshow(I / I.max(), origin="lower", extent=[-ext, ext, -ext, ext], cmap="inferno"); ax[1].set_xlim(-3, 3); ax[1].set_ylim(-3, 3)
    ax[1].set_title(f"Odak I(x,y) z={zf:.1f}um"); ax[1].set_xlabel("x(um)"); ax[1].set_ylabel("y(um)")
    mtf = metrics.mtf_1d(I[:, met["peak_ij"][1]]); fr = np.fft.fftshift(np.fft.fftfreq(pad, d=dx))
    ax[2].plot(fr[pad // 2:], mtf[pad // 2:]); ax[2].set_xlim(0, 2 / lam0); ax[2].set_xlabel("Uzaysal frekans (1/um)"); ax[2].set_ylabel("MTF"); ax[2].set_title("MTF"); ax[2].grid(alpha=.3)
    fig.suptitle(f"TAM PAKET tasarım — {mat}/{shape}, D={D_lens:.0f}um f={f}um NA={NA:.2f}")
    figpath = os.path.join(SIM, slug + ".png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    abs_pct = esuite["absolute"] * 100 if esuite["absolute"] else 0.0
    body = (f"- **Malzeme:** {mat} (n={n_pillar:.2f}) · **Şekil:** {shape} · **Sütun:** {int(inside.sum())} · **NA:** {NA:.2f} (f/{f/D_lens:.1f})\n"
            f"- **Odak:** z={zf:.1f}um (tasarım {f}um) · **FWHM:** {fwhm*1000:.0f}nm (difraksiyon λ/2NA={dl_fwhm:.0f}nm)\n"
            f"\n**📐 Profesyonel metrikler (literatür-standardı):**\n\n"
            f"| Metrik | Değer | Not |\n|---|---|---|\n"
            f"| Verim — mutlak (3×FWHM/gelen) | **{abs_pct:.0f}%** | standart raporlama |\n"
            f"| Verim — bağıl (3×FWHM/geçen) | {esuite['relative']*100:.0f}% | transmisyona normalize |\n"
            f"| Verim — legacy (1.5×FWHM) | {eff*100:.0f}% | eski tanım |\n"
            f"| Strehl (tepe) | {St:.2f} | ≥0.8 difraksiyon-sınırlı |\n"
            f"| Strehl (MTF-alan) | {st_mtf:.2f} | alternatif tanım |\n"
            f"| FWHM / difraksiyon | {fwhm*1000:.0f} / {dl_fwhm:.0f} nm | oran {fwhm*1000/dl_fwhm:.2f} |\n"
            f"| Kuşatılmış enerji %80 | {ee80*1000:.0f} nm | leke konsantrasyonu |\n"
            f"| Yan-lob seviyesi | {slobe['dB']:.1f} dB | düşük = temiz leke |\n"
            f"| Odak derinliği (DOF) | {dof:.1f} um | eksen I(z) FWHM |\n\n"
            f"- **GDS (üretilebilir):** `{os.path.basename(gdspath)}` — {npoly} polygon\n"
            f"- **3B doğrulama için:** `{slug}_design.json` → fullwave (Meep/Tidy3D)\n"
            f"- Yerel-periyodik yaklaşım; nihai doğrulama FDTD. (Metrik tanımları: [[📐 Metalens Değerlendirme ve Ölçüm Metrikleri (Profesyonel Katalog)]])")
    return figpath, body, f"z={zf:.1f}um FWHM={fwhm*1000:.0f}nm verim(mutlak){abs_pct:.0f}% Strehl{St:.2f} DOF{dof:.1f}um GDS✓"


def run_tolerance_analysis(fm, slug):
    """Madde 4: üretim toleransı Monte-Carlo. Sütun boyu hatasının verime etkisi."""
    lam0 = num(fm, "lam0", 0.633); period = num(fm, "period", 1.30); H = num(fm, "height", 0.60)
    eps_hi = num(fm, "eps_hi", 6.25); Nseg = int(num(fm, "n_seg", 12, int))
    sigma = num(fm, "sigma_nm", 15.0) / 1000.0; ntrial = int(num(fm, "n_trial", 60, int))
    target = int(num(fm, "target_order", 1, int)); Mo = int(num(fm, "M", 13, int))
    Nx = Nseg * 40
    from scipy.optimize import differential_evolution
    rng = np.random.default_rng(0)
    def Tt(dens):
        eps = np.repeat(1.0 + np.clip(dens, 0, 1) * (eps_hi - 1.0), Nx // Nseg)
        r = solve_rcwa_1d(lam0, 0, 1, 1, period, [(eps.astype(complex), H)], 2 * Mo + 1, pol=(0., 1.))
        return r["T"][np.where(r["orders"] == target)[0][0]]
    # önce iyi bir tasarım kur (kısa optimizasyon), sonra o tasarımın toleransını ölç
    _res = differential_evolution(lambda d: -Tt(d), [(0, 1)] * Nseg, maxiter=12, popsize=8, seed=1, polish=False, tol=1e-3)
    base = _res.x; T0 = -_res.fun
    vals = [Tt(base + rng.normal(0, sigma / (period / Nseg), Nseg)) for _ in range(ntrial)]
    vals = np.array(vals)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(vals * 100, bins=15, color="C0", alpha=.8)
    ax.axvline(T0 * 100, color="C3", ls="--", label=f"hatasız {T0*100:.0f}%")
    ax.axvline(vals.mean() * 100, color="k", ls=":", label=f"ortalama {vals.mean()*100:.0f}%")
    ax.set_xlabel(f"T[{target:+d}] verimi (%)"); ax.set_ylabel("Deney sayısı")
    ax.set_title(f"Üretim toleransı — σ={sigma*1000:.0f}nm sütun hatası, {ntrial} deneme"); ax.legend()
    figpath = os.path.join(SIM, slug + ".png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    body = (f"- Hatasız verim: {T0*100:.0f}% · Hata σ={sigma*1000:.0f}nm ({ntrial} deneme)\n"
            f"- **Ortalama: {vals.mean()*100:.0f}% ± {vals.std()*100:.0f}%** · en kötü: {vals.min()*100:.0f}%\n"
            f"- Verim düşüşü (ort): {(T0-vals.mean())*100:.0f} puan → üretim gürbüzlüğü göstergesi.")
    return figpath, body, f"tolerans: {T0*100:.0f}%->{vals.mean()*100:.0f}±{vals.std()*100:.0f}%"


def run_achromatic_design(fm, slug):
    """AKROMATİK METALENS — grup gecikmesi (GD) mühendisliği. Meta-atomları hem
    faz φ(λ0) hem GD=dφ/dω eşleşecek biçimde seçer → tüm dalga boyları aynı z'de
    odaklanır (kromatik kayma azalır). (boyut×yükseklik) kütüphanesi GD çeşitliliği verir.
    Basit sütunların GD aralığı sınırlı → akromatik açıklık sınırlıdır (fizik: açıklık×bant
    çarpımı maks-GD ile sınırlı). Bu üreteç bunu somut gösterir + singlet ile kıyaslar."""
    lam0 = num(fm, "lam0", 0.55); band = num(fm, "band", 0.10)
    lmin = num(fm, "lam_min", lam0 - band / 2); lmax = num(fm, "lam_max", lam0 + band / 2)
    L = num(fm, "period", 0.35); mat = fm.get("material", "TiO2_t"); shape = fm.get("shape", "circle")
    D = num(fm, "D_lens", 12.0); f = num(fm, "f", 20.0); M = int(num(fm, "M", 6, int))
    nsz = int(num(fm, "n_size", 8, int)); nh = int(num(fm, "n_h", 5, int))
    hmin = num(fm, "h_min", 0.4); hmax = num(fm, "h_max", 1.4)
    gdw = num(fm, "gd_weight", 1.0)
    c0 = 299792458.0
    n_pillar = float(np.real(np.sqrt(materials.eps_of(mat, lam0)))) if mat else num(fm, "n_pillar", 2.4)
    lams = np.array([lmin, lam0, lmax]); om = 2 * np.pi * c0 / (lams * 1e-6)
    sizes = np.linspace(0.06, L - 0.05, nsz); heights = np.linspace(hmin, hmax, nh)
    def _lib():
        PH = np.zeros((nsz, nh, 3)); AM = np.zeros((nsz, nh, 3))
        for a, s in enumerate(sizes):
            for b, H in enumerate(heights):
                eps = shapes.rasterize(shape, s, L, 48, n_pillar=n_pillar)
                for k, lam in enumerate(lams):
                    r = solve_rcwa_2d(lam, 0, 0, 1, 1, L, L, [(eps, H)], M, M, pol=(0., 1.))
                    t0 = r["ty"][r["i0"]]; PH[a, b, k] = np.angle(t0); AM[a, b, k] = abs(t0)
        return {"PH": PH, "AM": AM}
    c = cache.cached_library("achrolib", {"lam0": lam0, "band": band, "L": L, "H": (hmin, hmax, nh),
                             "mat": mat, "shape": shape, "M": M, "ns": nsz}, _lib)
    PH = c["PH"]; AM = c["AM"]
    # her atom: fiziksel faz (konvansiyon: -φ_rcwa) ve GD=dφ/dω (merkezî fark)
    phi_phys = np.mod(-PH, 2 * np.pi)                       # (nsz,nh,3)
    ph0 = phi_phys[:, :, 1].ravel()                        # λ0 fazı
    unwrapped = np.unwrap(-PH, axis=2)                     # GD için sürekli faz
    GD = ((unwrapped[:, :, 2] - unwrapped[:, :, 0]) / (om[2] - om[0])).ravel()
    amp = AM[:, :, 1].ravel()
    S, Hh = np.meshgrid(sizes, heights, indexing="ij")
    size_flat = S.ravel(); h_flat = Hh.ravel()
    # hedef profil
    Nc = int(round(D / L)); D = Nc * L; ci = (np.arange(Nc) - (Nc - 1) / 2) * L
    XX, YY = np.meshgrid(ci, ci, indexing="ij"); R = np.hypot(XX, YY); inside = R <= D / 2
    phi_req = np.mod(-(2 * np.pi / lam0) * (np.sqrt(R ** 2 + f ** 2) - f), 2 * np.pi)
    gd_req = -(np.sqrt(R ** 2 + f ** 2) - f) * 1e-6 / c0    # s (metre/c) — GEREKEN GD ŞEKLİ
    gr = gd_req[inside]
    gd_span_need = float(gr.max() - gr.min())
    gd_span_have = float(GD.max() - GD.min())
    # bağıl GD önemli (global piston serbest): gereken GD aralığını atomların GD aralığının
    # MERKEZİNE hizala → hücreler tüm GD aralığında atom bulabilsin
    gd_req = gd_req - (gr.max() + gr.min()) / 2.0 + (GD.max() + GD.min()) / 2.0
    # atom ata: |Δφ|/π + w*|ΔGD|/GD_gereken (her iki terim O(1) → GD gerçekten etkiler)
    gdsc = max(gd_span_need, 1e-18)
    size_map = np.zeros((Nc, Nc)); h_map = np.zeros((Nc, Nc))
    ins_ij = [(i, j) for i in range(Nc) for j in range(Nc) if inside[i, j]]
    idxA = -np.ones((Nc, Nc), int); idxS = -np.ones((Nc, Nc), int)   # akromatik / singlet(faz-yalnız)
    phflat = phi_phys.reshape(-1, 3); amflat = AM.reshape(-1, 3)
    for (i, j) in ins_ij:
        dphi = np.abs(np.angle(np.exp(1j * (ph0 - phi_req[i, j])))) / np.pi
        dgd = np.abs(GD - gd_req[i, j]) / gdsc
        kA = int(np.argmin(dphi + gdw * dgd)); kS = int(np.argmin(dphi))
        idxA[i, j] = kA; idxS[i, j] = kS
        size_map[i, j] = size_flat[kA]; h_map[i, j] = h_flat[kA]
    u = 3; dx = L / u
    def _prop_multilam(idxmap):
        fz = []; fw = []
        for k, lam in enumerate(lams):
            phk = phflat[:, k]; amk = amflat[:, k]; Ek = np.zeros((Nc, Nc), complex)
            for (i, j) in ins_ij:
                Ek[i, j] = amk[idxmap[i, j]] * np.exp(1j * phk[idxmap[i, j]])
            Ef = np.kron(Ek, np.ones((u, u))); Ncf = Nc * u; pad = int(Ncf * 1.6)
            Ep = np.zeros((pad, pad), complex); o = (pad - Ncf) // 2; Ep[o:o + Ncf, o:o + Ncf] = Ef
            kx = 2 * np.pi * np.fft.fftfreq(pad, d=dx); KX, KY = np.meshgrid(kx, kx, indexing="ij")
            k0 = 2 * np.pi / lam; KZ = np.sqrt((k0 ** 2 - KX ** 2 - KY ** 2).astype(complex)); KZ = np.where(np.imag(KZ) < 0, -KZ, KZ)
            A = np.fft.fft2(Ep); zc = np.linspace(0.5 * f, 1.6 * f, 40)
            Iax = [np.abs(np.fft.ifft2(A * np.exp(1j * KZ * z))[pad // 2, pad // 2]) ** 2 for z in zc]
            zf = zc[int(np.argmax(Iax))]; I = np.abs(np.fft.ifft2(A * np.exp(1j * KZ * zf))) ** 2
            xax = (np.arange(pad) - pad // 2) * dx; cut = I[:, pad // 2] / I[:, pad // 2].max()
            ab = np.where(cut >= 0.5)[0]; fw.append((xax[ab[-1]] - xax[ab[0]]) * 1000 if ab.size >= 2 else float("nan"))
            fz.append(zf)
        return fz, fw
    lam_focus, lam_fwhm = _prop_multilam(idxA)
    sfz, _ = _prop_multilam(idxS)
    chroma = max(lam_focus) - min(lam_focus)
    singlet_chroma = max(sfz) - min(sfz)                   # ÖLÇÜLEN singlet (faz-yalnız, aynı atomlar) — adil kıyas
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(lams * 1000, lam_focus, "o-", label="akromatik tasarım")
    ax[0].axhline(f, color="gray", ls=":", label=f"tasarım f={f}um")
    ax[0].set_xlabel("dalga boyu (nm)"); ax[0].set_ylabel("odak z (um)")
    ax[0].set_title(f"Odak vs λ · kromatik kayma {chroma:.2f}um (singlet~{singlet_chroma:.1f}um)"); ax[0].legend(); ax[0].grid(alpha=.3)
    sm = np.where(inside, h_map, np.nan)
    im = ax[1].imshow(sm.T, origin="lower", extent=[ci[0], ci[-1], ci[0], ci[-1]], cmap="plasma")
    ax[1].set_title("Sütun yüksekliği (um) — GD kontrolü"); ax[1].set_xlabel("x(um)"); ax[1].set_ylabel("y(um)"); plt.colorbar(im, ax=ax[1], fraction=0.046)
    fig.suptitle(f"AKROMATİK METALENS — {mat}, D={D:.0f}um f={f}um, {lmin*1000:.0f}-{lmax*1000:.0f}nm")
    figpath = os.path.join(SIM, slug + ".png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    cover = 100 * min(1.0, gd_span_have / max(gd_span_need, 1e-18))
    body = (f"- **Band:** {lmin*1000:.0f}-{lmax*1000:.0f}nm · **D={D:.0f}um f={f}um** · (boyut×yükseklik) {nsz}×{nh} kütüphane\n"
            f"- **Kromatik kayma:** akromatik **{chroma:.2f}um** vs basit singlet ~{singlet_chroma:.1f}um → **{singlet_chroma/max(chroma,1e-6):.1f}× iyileşme**\n"
            f"- **Grup gecikmesi:** gereken aralık {gd_span_need*1e15:.1f} fs, kütüphane {gd_span_have*1e15:.1f} fs → **kapsama %{cover:.0f}**\n"
            f"- Kapsama <%100 ise açıklık×bant fiziksel sınırı: daha büyük D/geniş bant için daha yüksek/rezonanslı meta-atom (daha çok GD) gerekir.\n"
            f"- λ başına odak: {['%.1f'%z for z in lam_focus]} um, FWHM {['%.0f'%w for w in lam_fwhm]} nm")
    return figpath, body, f"akromatik: kromatik {chroma:.2f}um (singlet~{singlet_chroma:.1f}um), GD kapsama %{cover:.0f}"


def _build_lens(fm):
    """Ortak lens kurucu (FOV + n-gürbüzlüğü analizleri için). Kütüphane (konvansiyon-
    düzeltmeli) + dairesel yerleşim -> açıklık alanı E0 + seçilen boyutlar."""
    lam0 = num(fm, "lam0", 0.633); L = num(fm, "period", 0.35); H = num(fm, "height", 0.60)
    mat = fm.get("material", "TiO2_t"); shape = fm.get("shape", "circle")
    D_lens = num(fm, "D_lens", 10.0); f = num(fm, "f", 12.0); M = int(num(fm, "M", 7, int))
    n_pillar = float(np.real(np.sqrt(materials.eps_of(mat, lam0)))) if mat else num(fm, "n_pillar", 2.4)
    sizes = np.linspace(0.06, L - 0.05, 20)
    def _lib():
        lp = []; la = []
        for s in sizes:
            eps = shapes.rasterize(shape, s, L, 64, n_pillar=n_pillar)
            r = solve_rcwa_2d(lam0, 0, 0, 1, 1, L, L, [(eps, H)], M, M, pol=(0., 1.))
            t0 = r["ty"][r["i0"]]; lp.append(np.angle(t0)); la.append(abs(t0))
        lp = np.unwrap(np.array(lp)); lp -= lp.min()
        return {"phm": np.mod(lp, 2 * np.pi), "am": np.array(la)}
    c = cache.cached_library("libdesign", {"lam0": lam0, "L": L, "H": H, "mat": mat, "shape": shape, "M": M, "n": round(n_pillar, 3)}, _lib)
    libphm = np.mod(-c["phm"], 2 * np.pi); libam = c["am"]      # konvansiyon düzeltmesi
    Nc = int(round(D_lens / L)); D_lens = Nc * L
    ci = (np.arange(Nc) - (Nc - 1) / 2) * L
    XX, YY = np.meshgrid(ci, ci, indexing="ij"); R = np.hypot(XX, YY); inside = R <= D_lens / 2
    phi_t = np.mod(-(2 * np.pi / lam0) * (np.sqrt(R ** 2 + f ** 2) - f), 2 * np.pi)
    idx_map = -np.ones((Nc, Nc), int); E0 = np.zeros((Nc, Nc), complex)
    for i in range(Nc):
        for j in range(Nc):
            if inside[i, j]:
                k = np.argmin(np.abs(np.angle(np.exp(1j * (libphm - phi_t[i, j])))))
                idx_map[i, j] = k; E0[i, j] = libam[k] * np.exp(1j * libphm[k])
    NA = np.sin(np.arctan(D_lens / 2 / f))
    return dict(lam0=lam0, L=L, H=H, f=f, D_lens=D_lens, M=M, n_pillar=n_pillar, shape=shape,
                sizes=sizes, Nc=Nc, ci=ci, inside=inside, idx_map=idx_map, E0=E0, NA=NA)


def _prop_focus(E0, Nc, L, lam0, f, tilt_deg=0.0, zwin=(0.6, 1.4), nz=25, u=4, padf=1.7, zfix=None):
    """Açıklık alanını (opsiyonel eğik gelişle) yay, odak düzleminde tepe/konum/FWHM bul.
    zfix verilirse yalnız o z-düzleminde (düz-sensör); yoksa en iyi z'yi arar (best-focus)."""
    k0 = 2 * np.pi / lam0; dx = L / u
    Ef = np.kron(E0, np.ones((u, u))); Ncf = Nc * u; pad = int(Ncf * padf)
    Ep = np.zeros((pad, pad), complex); o = (pad - Ncf) // 2; Ep[o:o + Ncf, o:o + Ncf] = Ef
    xf = (np.arange(pad) - pad // 2) * dx
    Xf, _Yf = np.meshgrid(xf, xf, indexing="ij")
    Ep = Ep * np.exp(1j * k0 * np.sin(np.deg2rad(tilt_deg)) * Xf)
    kx = 2 * np.pi * np.fft.fftfreq(pad, d=dx); KX, KY = np.meshgrid(kx, kx, indexing="ij")
    KZ = np.sqrt((k0 ** 2 - KX ** 2 - KY ** 2).astype(complex)); KZ = np.where(np.imag(KZ) < 0, -KZ, KZ)
    A = np.fft.fft2(Ep)
    zc = [zfix] if zfix is not None else np.linspace(zwin[0] * f, zwin[1] * f, nz)
    best = None
    for z in zc:
        I = np.abs(np.fft.ifft2(A * np.exp(1j * KZ * z))) ** 2
        if best is None or I.max() > best[0]:
            best = (I.max(), z, I)
    peak, zf, I = best
    j = np.unravel_index(np.argmax(I), I.shape); x_peak = xf[j[0]]
    cut = I[:, j[1]] / I[:, j[1]].max(); ab = np.where(cut >= 0.5)[0]
    fwhm = float(xf[ab[-1]] - xf[ab[0]]) if ab.size >= 2 else float("nan")
    return dict(peak=float(peak), zf=float(zf), x_peak=float(x_peak), fwhm=fwhm, I=I, xf=xf)


def run_fov_analysis(fm, slug):
    """Eğik-geliş GÖRÜNTÜ KALİTESİ: açıya karşı odak kayması (distorsiyon), leke bozulması
    (koma/alan aberasyonu ~ FWHM artışı), bağıl verim (Strehl), alan eğriliği (odak z kayması).
    FOV = difraksiyon-sınırlı kalınan yarı-açı."""
    d = _build_lens(fm); f = d["f"]; lam0 = d["lam0"]; NA = d["NA"]; dl = lam0 / (2 * NA)
    amax = num(fm, "angle_max", 15.0); na = int(num(fm, "n_angle", 8, int))
    angles = np.linspace(0, amax, na)
    r0 = _prop_focus(d["E0"], d["Nc"], d["L"], lam0, f, tilt_deg=0.0)
    peak0 = r0["peak"]; z0 = r0["zf"]
    th, xpk, zpk, fw, rstrehl, dist = [], [], [], [], [], []
    for a in angles:
        r = _prop_focus(d["E0"], d["Nc"], d["L"], lam0, f, tilt_deg=a, zfix=z0)   # DÜZ sensör (sabit z0)
        rc = _prop_focus(d["E0"], d["Nc"], d["L"], lam0, f, tilt_deg=a)            # en iyi odak (alan eğriliği)
        ideal = f * np.tan(np.deg2rad(a))
        th.append(a); xpk.append(r["x_peak"]); zpk.append(rc["zf"]); fw.append(r["fwhm"] * 1000)
        rstrehl.append(r["peak"] / peak0)
        dist.append(100 * (r["x_peak"] - ideal) / ideal if ideal > 1e-6 else 0.0)
    fw = np.array(fw); rstrehl = np.array(rstrehl); th = np.array(th)
    ok = (fw <= 1.5 * dl * 1000) & (rstrehl >= 0.5)
    fov_half = float(th[ok].max()) if ok.any() else 0.0
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].plot(th, fw, "o-"); ax[0].axhline(dl * 1000, color="gray", ls=":", label=f"difraksiyon {dl*1000:.0f}nm")
    ax[0].set_xlabel("geliş açısı (°)"); ax[0].set_ylabel("odak FWHM (nm)"); ax[0].set_title("Leke bozulması (alan aberasyonu)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(th, rstrehl, "o-"); ax[1].axhline(0.8, color="gray", ls=":", label="0.8 (difr.-sınırlı)")
    ax[1].set_xlabel("geliş açısı (°)"); ax[1].set_ylabel("bağıl Strehl (Iθ/I0)"); ax[1].set_title(f"Açısal verim · yarı-FOV≈{fov_half:.0f}°"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].plot(th, dist, "o-"); ax[2].axhline(0, color="gray", ls=":")
    ax[2].set_xlabel("geliş açısı (°)"); ax[2].set_ylabel("distorsiyon (%)"); ax[2].set_title("Distorsiyon (x vs f·tanθ)"); ax[2].grid(alpha=.3)
    fig.suptitle(f"EĞİK-GELİŞ GÖRÜNTÜ KALİTESİ — {d['shape']}/{fm.get('material','')}, D={d['D_lens']:.0f}um f={f}um NA={NA:.2f}")
    figpath = os.path.join(SIM, slug + ".png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    fc = float(np.max(np.abs(np.array(zpk) - z0)))
    body = (f"- **NA:** {NA:.2f} · difraksiyon FWHM {dl*1000:.0f}nm · normal-geliş odak z={z0:.1f}um\n"
            f"- **Yarı-FOV (difraksiyon-sınırlı):** ~{fov_half:.0f}° (FWHM≤1.5×difr. & bağıl Strehl≥0.5)\n"
            f"- **Maks açıda ({amax:.0f}°):** FWHM={fw[-1]:.0f}nm, bağıl Strehl={rstrehl[-1]:.2f}, distorsiyon={dist[-1]:.1f}%\n"
            f"- **Alan eğriliği (odak z sapması):** {fc:.2f}um\n"
            f"- Düz singlet metalens: FOV sınırlı (koma). Geniş-FOV için ikili/kuadratik tasarım.")
    return figpath, body, f"yarı-FOV~{fov_half:.0f}°, {amax:.0f}°'de FWHM={fw[-1]:.0f}nm rStrehl={rstrehl[-1]:.2f}"


def run_index_robustness(fm, slug):
    """KIRILMA-İNDİSİ GÜRBÜZLÜĞÜ: sütun indisi n±Δn saptığında (malzeme/proses hatası)
    odaklama verimi/FWHM nasıl değişir. Seçilen boyutlar sabit; faz kütüphanesi n'de yeniden."""
    d = _build_lens(fm); f = d["f"]; lam0 = d["lam0"]; L = d["L"]; H = d["H"]
    shape = d["shape"]; M = d["M"]; sizes = d["sizes"]; NA = d["NA"]; dl = lam0 / (2 * NA)
    n0 = d["n_pillar"]; dn_max = num(fm, "dn_max", 0.15); ndn = int(num(fm, "n_dn", 5, int))
    inside = d["inside"]; idx_map = d["idx_map"]; Nc = d["Nc"]
    from rcwa import metrics as _m
    used = np.unique(idx_map[inside])                      # yalnız kullanılan benzersiz boyutlar
    dns = np.linspace(-dn_max, dn_max, ndn); effs = []; fwhms = []
    for dn in dns:
        np_ = n0 + dn
        # yalnız kullanılan boyutların bu n'deki fazı+genliği (hız)
        lp = np.zeros(len(sizes)); la = np.zeros(len(sizes)); raw = {}
        for k in used:
            eps = shapes.rasterize(shape, sizes[k], L, 48, n_pillar=np_)
            r = solve_rcwa_2d(lam0, 0, 0, 1, 1, L, L, [(eps, H)], M, M, pol=(0., 1.))
            t0 = r["ty"][r["i0"]]; raw[k] = (np.angle(t0), abs(t0))
        ph_used = np.unwrap(np.array([raw[k][0] for k in used])); ph_used -= ph_used.min()
        for ii, k in enumerate(used):
            lp[k] = ph_used[ii]; la[k] = raw[k][1]
        lp = np.mod(-lp, 2 * np.pi)
        E0 = np.zeros((Nc, Nc), complex)
        for i in range(Nc):
            for j in range(Nc):
                if inside[i, j]:
                    k = idx_map[i, j]; E0[i, j] = la[k] * np.exp(1j * lp[k])
        pr = _prop_focus(E0, Nc, L, lam0, f)
        xf = pr["xf"]; I = pr["I"]; met = _m.focal_metrics(I, xf); fw = met["fwhm_x"]
        effs.append(_m.focusing_efficiency(I, xf, fw)); fwhms.append(fw * 1000)
    effs = np.array(effs) * 100; fwhms = np.array(fwhms)
    i0 = int(np.argmin(np.abs(dns)))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(dns, effs, "o-"); ax[0].axvline(0, color="gray", ls=":"); ax[0].set_xlabel("Δn (indis hatası)"); ax[0].set_ylabel("odaklama verimi (%)"); ax[0].set_title("Verim vs indis hatası"); ax[0].grid(alpha=.3)
    ax[1].plot(dns, fwhms, "o-"); ax[1].axhline(dl * 1000, color="gray", ls=":", label=f"difraksiyon {dl*1000:.0f}nm"); ax[1].axvline(0, color="gray", ls=":")
    ax[1].set_xlabel("Δn"); ax[1].set_ylabel("FWHM (nm)"); ax[1].set_title("Leke vs indis hatası"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle(f"KIRILMA-İNDİSİ GÜRBÜZLÜĞÜ — {shape}/{fm.get('material','')} n0={n0:.2f}, D={d['D_lens']:.0f}um f={f}um")
    figpath = os.path.join(SIM, slug + ".png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    # duyarlılık: verim düşüşü / birim Δn (uçlarda)
    sens = float((effs[i0] - min(effs[0], effs[-1])) / dn_max)
    body = (f"- **Nominal n={n0:.2f}:** verim {effs[i0]:.0f}%, FWHM {fwhms[i0]:.0f}nm (difraksiyon {dl*1000:.0f}nm)\n"
            f"- **±{dn_max:.2f} indis hatasında:** verim {effs[0]:.0f}%…{effs[-1]:.0f}%, FWHM {fwhms.min():.0f}…{fwhms.max():.0f}nm\n"
            f"- **Duyarlılık:** ~{sens:.0f} puan verim / birim Δn → üretim/malzeme indis toleransı göstergesi\n"
            f"- Yüksek duyarlılık = indis-hassas tasarım; düşük NA / gürbüz meta-atom seçimi iyileştirir.")
    return figpath, body, f"n-gürbüzlük: ±{dn_max:.2f}Δn -> verim {effs[0]:.0f}…{effs[-1]:.0f}%, duyarlılık {sens:.0f}p/Δn"


def run_metalens_large_tiled(fm, slug):
    """E6: cm-ÖLÇEK KUTUCUKLAMALI büyük lens. Vektörize boyut seçimi (döngüsüz) +
    kutucuk-kutucuk GDS (bellek sınırlı) + önizleme + cm ölçek projeksiyonu."""
    from rcwa import largelens as LL
    import time
    lam0 = num(fm, "lam0", 0.633); L = num(fm, "period", 0.35); H = num(fm, "height", 0.60)
    mat = fm.get("material", "TiO2_t"); shape = fm.get("shape", "circle")
    D_lens = num(fm, "D_lens", 100.0); f = num(fm, "f", 200.0); M = int(num(fm, "M", 7, int))
    tile = int(num(fm, "tile_cells", 300, int))
    do_gds = str(fm.get("write_gds", "1")).lower() not in ("0", "false", "hayir", "no")
    gdsdir = os.path.join(SIM, slug + "_gds") if do_gds else None
    t0 = time.time()
    st = LL.design_large_lens(lam0, L, H, mat, shape, D_lens, f, M, gds_dir=gdsdir, tile_cells=tile)
    dt = time.time() - t0
    ci = st["ci_ds"]; sm = np.where(st["preview_size_nm"] > 0, st["preview_size_nm"], np.nan)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    im = ax[0].imshow(sm.T, origin="lower", extent=[ci[0], ci[-1], ci[0], ci[-1]], cmap="viridis")
    ax[0].set_title("Sütun boyutu (nm) — önizleme"); ax[0].set_xlabel("x(um)"); ax[0].set_ylabel("y(um)"); plt.colorbar(im, ax=ax[0], fraction=0.046)
    r = np.linspace(0, st["D_lens"] / 2, 400); phi = np.mod(-(2 * np.pi / lam0) * (np.sqrt(r ** 2 + f ** 2) - f), 2 * np.pi)
    ax[1].plot(r, phi); ax[1].set_xlabel("r (um)"); ax[1].set_ylabel("hedef faz (rad)"); ax[1].set_title("Radyal faz profili"); ax[1].grid(alpha=.3)
    fig.suptitle(f"BÜYÜK LENS (kutucuklu) — {mat}/{shape} D={st['D_lens']:.0f}um NA={st['NA']:.2f}")
    figpath = os.path.join(SIM, slug + ".png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    gb_now = LL.scale_projection(st["n_atoms"])
    na_cm = int(np.pi / 4 * (10000.0 / L) ** 2); gb_cm = LL.scale_projection(na_cm)
    body = (f"- **Açıklık:** D={st['D_lens']:.0f}um, f={f}um, NA={st['NA']:.2f} · **{st['n_atoms']:,} meta-atom** (vektörize/döngüsüz seçim)\n"
            f"- **Kutucuklu GDS:** {st['n_tiles']} kutucuk, {st['npoly']:,} poligon, ~{gb_now:.2f} GB → `{os.path.basename(gdsdir) if gdsdir else '(yazılmadı)'}`\n"
            f"- **Süre:** {dt:.1f}s · bellek kutucuk boyutuyla sınırlı (cm-ölçek mümkün)\n"
            f"- **Ölçek projeksiyonu:** 1cm açıklık ≈ {na_cm:,} atom, ~{gb_cm:.0f} GB GDS — hat çalışır; kalan yalnız disk/süre.\n"
            f"- Doğrulama: LUT seçimi `metalens_design_full` ile birebir; 3B için temsili kesit ya da Tidy3D bulut.")
    return figpath, body, f"D={st['D_lens']:.0f}um {st['n_atoms']:,} atom, {st['n_tiles']} kutucuk ~{gb_now:.2f}GB"


RUNNERS = {"grating": run_grating, "metalens_library": run_metalens_library,
           "angle_sweep": run_angle_sweep,
           "grating2d": run_grating2d, "metalens_pillar_2d": run_metalens_pillar_2d,
           "lens_fullwave_1d": run_lens_fullwave_1d,
           "metalens_layout_2d": run_metalens_layout_2d,
           "wavelength_sweep": run_wavelength_sweep, "birefringent_library": run_birefringent_library,
           "jones_matrix": run_jones_matrix, "pb_phase_library": run_pb_phase_library,
           "param_map_2d": run_param_map_2d,
           "inverse_design_deflector": run_inverse_design_deflector,
           "achromatic_analysis": run_achromatic_analysis, "vortex_beam": run_vortex_beam,
           "hologram": run_hologram, "polarization_multiplexed": run_polarization_multiplexed,
           "metalens_design_full": run_metalens_design_full, "tolerance_analysis": run_tolerance_analysis,
           "fov_analysis": run_fov_analysis, "index_robustness": run_index_robustness,
           "metalens_large_tiled": run_metalens_large_tiled,
           "achromatic_design": run_achromatic_design}


# ----------------------------- rapor + güncelleme -----------------------------
def write_report(exp_name, fm, figpath, body, summary):
    ts = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = slugify(exp_name)
    title = f"Rapor - {exp_name} - {stamp}"
    figname = os.path.basename(figpath)
    params = "\n".join(f"| {k} | {v} |" for k, v in fm.items()
                       if k not in ("etiketler","ilgili_proje_veya_alan","id","tur","durum","olusturulma","son_rapor"))
    md = f"""---
id: {stamp}
olusturulma: {ts}
tur: rapor
durum: 🟢 tamamlandi
deney: "[[{exp_name}]]"
etiketler:
  - konu/rcwa
  - tip/rapor
ilgili_proje_veya_alan: "[[🧪 RCWA Deney Panosu]]"
---

# 📊 {title}

⬅️ [[🧪 RCWA Deney Panosu]] · Deney: [[{exp_name}]]  ·  Çalıştırma: {ts}

> **Özet:** {summary}

## 🔧 Parametreler
| Anahtar | Değer |
|---|---|
{params}

## 📈 Sonuçlar
{body}

## 🖼️ Figür
![[{figname}]]

## 🔍 İnceleme / Notlar
*(Buraya kendi yorumunu ekle: beklenen mi? bir sonraki deney ne olmalı?)*
- 

---
*Flora Status: 🟢 Otomatik üretildi. İnceleme kısmını sen doldur.*
"""
    path = os.path.join(RAPORLAR, title + ".md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return title


def mark_done(exp_path, report_title):
    with open(exp_path, "r", encoding="utf-8") as f:
        txt = f.read()
    txt = re.sub(r"(?m)^durum:.*$", "durum: 🟢 tamamlandi", txt, count=1)
    if "son_rapor:" in txt:
        txt = re.sub(r"(?m)^son_rapor:.*$", f'son_rapor: "[[{report_title}]]"', txt, count=1)
    else:
        txt = re.sub(r"(?m)^(durum: .*)$", rf'\1\nson_rapor: "[[{report_title}]]"', txt, count=1)
    with open(exp_path, "w", encoding="utf-8") as f:
        f.write(txt)


def run_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    fm, _ = parse_frontmatter(text)
    tip = fm.get("deney_tipi", "grating")
    name = os.path.splitext(os.path.basename(path))[0]
    if tip not in RUNNERS:
        print(f"  ⚠️ bilinmeyen deney_tipi '{tip}' — atlandı: {name}")
        return False
    print(f"  ▶ çalışıyor: {name}  (tip={tip})")
    figpath, body, summary = RUNNERS[tip](fm, slugify(name))
    title = write_report(name, fm, figpath, body, summary)
    mark_done(path, title)
    print(f"    ✅ {summary}\n    rapor -> {title}")
    return True


def main(argv):
    args = argv[1:]
    if args and args[0] != "--all":
        run_file(args[0]); return
    # --all: bekleyen deneyler
    files = [os.path.join(DENEYLER, f) for f in os.listdir(DENEYLER) if f.endswith(".md")]
    pending = []
    for p in files:
        fm, _ = parse_frontmatter(open(p, encoding="utf-8").read())
        if "bekliyor" in fm.get("durum", "") or "bekliyor" in fm.get("etiketler", ""):
            pending.append(p)
    if not pending:
        print("Bekleyen deney yok. (durum: 🔬 bekliyor olan deney ekle.)"); return
    print(f"{len(pending)} bekleyen deney bulundu.")
    for p in pending:
        try:
            run_file(p)
        except Exception as e:
            print(f"  ❌ hata ({os.path.basename(p)}): {e}")


if __name__ == "__main__":
    main(sys.argv)


