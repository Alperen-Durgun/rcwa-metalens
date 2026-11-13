"""Experiment generators + the Markdown experiment loop.

Every experiment type is a function ``run_<type>(params, slug) -> (figure_path, report_body,
summary)`` registered in ``RUNNERS``. They can be driven three ways:

  * command line:   rcwa-metalens <type> key=value ...        (see rcwa/cli.py)
  * Python:         from rcwa.experiments import RUNNERS
  * note workflow:  experiment notes (*.md with an ``experiment_type`` — or Turkish
                    ``deney_tipi`` — front-matter field and ``status: pending``) are run
                    and a report note is written next to them (Obsidian-vault friendly):
                        python run_experiment.py --all
                        python run_experiment.py "path/to/experiment.md"

Output location — see ``project_dirs()``.
"""
import os, sys, re, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import solve_rcwa_1d, solve_rcwa_2d, solve_scalar_1d, materials, cache, metrics, shapes, gds
from . import design as dz

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # repository root (when run from a clone)


# Default unit-cell recipe of the lens generators (docs/recipes): ALD TiO2 (Siefke data) on fused
# silica, H = 600 nm, L = 375 nm, at 633 nm - certified and unit-cell FDTD-validated.
RECIPE = {"material": "TiO2_Siefke", "period": 0.375, "height": 0.60, "substrate": "SiO2_Malitson"}


def _substrate_index(fm, lam0):
    """Substrate index: n_sub=... if given, else the substrate material (default fused silica,
    'air' or 'none' = 1)."""
    if "n_sub" in fm:
        return float(num(fm, "n_sub", 1.0))
    sub = fm.get("substrate", RECIPE["substrate"])
    if sub in ("", "air", "none"):
        return 1.0
    return float(np.real(np.sqrt(materials.eps_of(sub, num(fm, "lam0", 0.633)))))


def project_dirs():
    """Where experiment notes, reports and figures/GDS live -> (root, experiments, reports, outputs).

    1) $RCWA_PROJECT_DIR if set (sub-folders experiments/ reports/ outputs/);
    2) the Obsidian vault two levels above the repository, if this copy sits inside one —
       recognised by its 'Deneyler' folder; the vault's own folder names are then used;
    3) otherwise ./workspace under the current directory.
    Nothing is created at import time; folders appear when something is written."""
    env = os.environ.get("RCWA_PROJECT_DIR")
    vault = os.path.dirname(os.path.dirname(HERE))
    if not env and os.path.isdir(os.path.join(vault, "Deneyler")):
        return vault, os.path.join(vault, "Deneyler"), os.path.join(vault, "Raporlar"), os.path.join(vault, "Simulasyonlar")
    root = os.path.abspath(env) if env else os.path.join(os.getcwd(), "workspace")
    return root, os.path.join(root, "experiments"), os.path.join(root, "reports"), os.path.join(root, "outputs")


PROJECT, DENEYLER, RAPORLAR, SIM = project_dirs()


def _out(name):
    """Path of an output file (figure/GDS/JSON); creates the outputs folder on first use."""
    os.makedirs(SIM, exist_ok=True)
    return os.path.join(SIM, name)


# ----------------------------- helpers -----------------------------
def parse_frontmatter(text):
    """Simple YAML-like frontmatter parser (key: value)."""
    m = re.match(r"^\ufeff?---\s*\n(.*?)\n---\s*\n?(.*)$", text.replace("\r\n", "\n"), re.S)
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


# ----------------------------- experiment types -----------------------------
def run_grating(fm, slug):
    """1D binary grating: diffraction efficiencies per order (R, T) and energy check."""
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
    ax.set_xlabel("Diffraction order m"); ax.set_ylabel("Efficiency (DE)")
    ax.set_title(f"Grating efficiencies — {pol}, θ={theta}°"); ax.legend(); ax.grid(True, alpha=.3)
    figpath = _out(f"{slug}.png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    rows = "".join(f"| {int(m)} | {R[i]:.4f} | {T[i]:.4f} |\n"
                   for i, m in enumerate(orders) if prop[i])
    body = (f"| m | R | T |\n|---|---|---|\n{rows}\n"
            f"**Total:** Rtot = {r['Rtot']:.4f}, Ttot = {r['Ttot']:.4f}, "
            f"ΣDE = {r['Rtot']+r['Ttot']:.6f} {'✅' if abs(r['Rtot']+r['Ttot']-1)<1e-4 else '⚠️'}")
    summary = f"ΣDE={r['Rtot']+r['Ttot']:.6f}, Ttot={r['Ttot']:.4f}"
    return figpath, body, summary


def run_metalens_library(fm, slug):
    """1D meta-atom library: transmission phase/amplitude versus pillar width."""
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
        t0 = r["ty"][r["zeroth_index"]]; phase.append(-np.angle(t0)); amp.append(np.abs(t0))   # physical convention
    phase = np.unwrap(np.array(phase)); phase -= phase[0]; amp = np.array(amp)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(widths*1000, phase/np.pi, "o-"); ax[0].set_xlabel("Pillar width (nm)")
    ax[0].set_ylabel("Phase (π)"); ax[0].set_title("Phase library"); ax[0].grid(True, alpha=.3)
    ax[1].plot(widths*1000, amp, "s-", color="C1"); ax[1].set_xlabel("Pillar width (nm)")
    ax[1].set_ylabel("|t0|"); ax[1].set_ylim(0, 1.05); ax[1].set_title("Amplitude"); ax[1].grid(True, alpha=.3)
    fig.suptitle(f"Meta-atom sweep (λ={lam0}, Λ={period}, H={H}, n={n_pillar})")
    figpath = _out(f"{slug}.png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    cov = np.ptp(phase)/np.pi
    body = (f"- **Phase coverage:** {cov:.2f}π  {'✅ (≥2π, full lens possible)' if cov>=2 else '⚠️ (<2π, increase H or n)'}\n"
            f"- **Mean amplitude:** {amp.mean():.3f}\n"
            f"- Swept width: {w_min*1000:.0f}–{w_max*1000:.0f} nm ({nW} points)")
    summary = f"phase={cov:.2f}π, <|t|>={amp.mean():.3f}"
    return figpath, body, summary


def run_angle_sweep(fm, slug):
    """1D grating: total R/T versus incidence angle."""
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
    ax.plot(thetas, Ttot, label="T total"); ax.plot(thetas, Rtot, label="R total")
    ax.set_xlabel("Incidence angle θ (°)"); ax.set_ylabel("Efficiency"); ax.set_ylim(0, 1.02)
    ax.set_title(f"Angle sweep — {pol}"); ax.legend(); ax.grid(True, alpha=.3)
    figpath = _out(f"{slug}.png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    body = (f"- Angle range: {a0:.0f}–{a1:.0f}° ({na} points)\n"
            f"- T range: {min(Ttot):.3f} – {max(Ttot):.3f}\n"
            f"- Energy: ΣDE ≈ {np.mean(np.array(Ttot)+np.array(Rtot)):.6f}")
    summary = f"T∈[{min(Ttot):.3f},{max(Ttot):.3f}]"
    return figpath, body, summary



def _eps_square_pillar(L, side, eps_hi, eps_lo, N=None):
    """Square pillar (side) centred in an L x L cell, fine raster (see shapes.rasterize)."""
    return shapes.rasterize("square", side, L, N or shapes.DEFAULT_N,
                            n_pillar=np.sqrt(complex(eps_hi)), n_bg=np.sqrt(complex(eps_lo)))


def run_grating2d(fm, slug):
    """2D (crossed) grating: per-order efficiencies of a square pillar lattice."""
    lam0 = num(fm, "lam0", 0.633); Lx = num(fm, "Lx", num(fm, "period", 1.0))
    Ly = num(fm, "Ly", Lx); theta = num(fm, "theta", 0.0); phi = num(fm, "phi", 0.0)
    pol = fm.get("pol", "TE"); er_ref = num(fm, "er_ref", 1.0); er_trn = num(fm, "er_trn", 1.0)
    eps_hi = num(fm, "eps_hi", 4.0); eps_lo = num(fm, "eps_lo", 1.0)
    side = num(fm, "side", 0.5); d = num(fm, "thickness", 0.3)
    M = int(num(fm, "M", 7, int))
    eps = _eps_square_pillar(Lx, side, eps_hi, eps_lo)
    r = solve_rcwa_2d(lam0, theta, phi, er_ref, er_trn, Lx, Ly, [(eps, d)], M, M, pol=pol_vec(pol))
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].imshow(eps.real.T, origin="lower", extent=[0, Lx, 0, Ly], cmap="viridis")
    ax[0].set_title("Unit cell ε(x,y)"); ax[0].set_xlabel("x (um)"); ax[0].set_ylabel("y (um)")
    MXf, MYf = r["orders"]; DE = r["R"] + r["T"]
    sc = ax[1].scatter(MXf, MYf, c=DE, s=80, cmap="magma")
    ax[1].set_title("Order efficiencies (R+T)"); ax[1].set_xlabel("m"); ax[1].set_ylabel("n")
    plt.colorbar(sc, ax=ax[1]); fig.suptitle(f"2D grating — {pol}, θ={theta}°")
    figpath = _out(f"{slug}.png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    ok = "✅" if abs(r['Rtot']+r['Ttot']-1) < 1e-4 else "⚠️"
    body = (f"- Rtot = {r['Rtot']:.4f}, Ttot = {r['Ttot']:.4f}, ΣDE = {r['Rtot']+r['Ttot']:.6f} {ok}\n"
            f"- Number of orders: {(2*M+1)**2}  ((2·{M}+1)²)")
    summary = f"2D ΣDE={r['Rtot']+r['Ttot']:.6f}, Ttot={r['Ttot']:.4f}"
    return figpath, body, summary


def run_metalens_pillar_2d(fm, slug):
    """2D meta-atom library: zeroth-order transmission phase/amplitude versus pillar size."""
    lam0 = num(fm, "lam0", 0.633); L = num(fm, "period", 0.35)
    H = num(fm, "height", 0.60); n_pillar = num(fm, "n_pillar", 2.4)
    mat = fm.get("material", ""); shape = fm.get("shape", "square"); n_sub = num(fm, "n_sub", 1.0)
    M = int(num(fm, "M", 7, int)); s_min = num(fm, "s_min", 0.05)
    s_max = num(fm, "s_max", L - 0.03); nS = int(num(fm, "n_side", 24, int))
    sides = np.linspace(s_min, s_max, nS)
    lib = dz.meta_atom_library(lam0, L, H, sizes=sides, shape=shape, material=mat, n_pillar=n_pillar, M=M, n_sub=n_sub)
    phase = lib["phase"]; amp = lib["amp"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(sides*1000, phase/np.pi, "o-"); ax[0].set_xlabel("Pillar size (nm)")
    ax[0].set_ylabel("Phase (π)"); ax[0].set_title("2D phase library"); ax[0].grid(True, alpha=.3)
    ax[1].plot(sides*1000, amp, "s-", color="C1"); ax[1].set_xlabel("Pillar size (nm)")
    ax[1].set_ylabel("|t0|"); ax[1].set_ylim(0, 1.05); ax[1].set_title("Amplitude"); ax[1].grid(True, alpha=.3)
    fig.suptitle(f"2D meta-atom — {shape} (λ={lam0}, Λ={L}, H={H}, n={lib['n'].real:.2f})")
    figpath = _out(f"{slug}.png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    cov = 2 * lib["coverage"]
    body = (f"- **Phase coverage:** {cov:.2f}π {'✅ (≥2π)' if cov>=2 else '⚠️ (<2π; increase H or n)'}\n"
            f"- **Mean amplitude:** {amp.mean():.3f} (min {amp.min():.3f})\n"
            f"- Size sweep: {s_min*1000:.0f}–{s_max*1000:.0f} nm ({nS} points), {shape}, M={M}, substrate n={n_sub}\n"
            f"- Phase is reported in the physics convention (grows with optical delay).")
    summary = f"2D phase={cov:.2f}π, <|t|>={amp.mean():.3f}"
    return figpath, body, summary


def run_lens_fullwave_1d(fm, slug):
    """Full-wave 1D metalens: the whole lens as one super-cell RCWA solve + exact propagation.
    M is scanned because of ill-conditioning; stable energy-conserving (ΣDE≈1) solutions are selected."""
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
    ph = np.unwrap(np.array(ph)); ph -= ph.min()
    # CONVENTION: solver phases are exp(+jwt); the lens profile below is exp(-iwt) -> negate
    # (same fix as the 2D designers; see rcwa/convention.py). Without it the lens DIVERGES.
    phm = np.mod(-ph, 2 * np.pi)
    xc = X0 + ((np.arange(Ncell) + 0.5) * Lcell - D_lens / 2)
    phi = np.mod(-(2 * np.pi / lam0) * (np.sqrt((xc - X0) ** 2 + f ** 2) - f), 2 * np.pi)
    chosen = np.array([widths[np.argmin(np.abs(np.angle(np.exp(1j * (phm - p)))))] for p in phi])
    Nx = 4096; xg = (np.arange(Nx) + 0.5) / Nx * D
    col = np.ones(Nx); xs = X0 - D_lens / 2
    for i in range(Ncell):
        c = xs + (i + 0.5) * Lcell
        col[np.abs(xg - c) < chosen[i] / 2] = n_pillar ** 2
    def propagate(res):
        """Physical field behind the lens: the full-wave near field of the super-cell at the exit
        plane, windowed to the LENS APERTURE, then exact angular-spectrum propagation. The window
        removes the plane wave leaking through the guard gaps and the periodic replicas."""
        m = res["orders"]; ty = np.conj(res["ty"]); kx = -m * (lam0 / D)   # physical-convention amplitudes
        Nap = 1024; dxa = D_lens / Nap; xa = X0 - D_lens / 2 + (np.arange(Nap) + 0.5) * dxa
        Enear = np.exp(1j * k0 * np.outer(xa, kx)) @ ty
        Npad = 16384; Ep = np.zeros(Npad, complex); o = (Npad - Nap) // 2; Ep[o:o + Nap] = Enear
        kxf = 2 * np.pi * np.fft.fftfreq(Npad, d=dxa)
        kzf = np.sqrt((k0 ** 2 - kxf ** 2).astype(complex)); kzf = np.where(np.imag(kzf) < 0, -kzf, kzf)
        A = np.fft.fft(Ep); xfull = X0 + (np.arange(Npad) - Npad / 2 + 0.5) * dxa

        def field(z, xo):
            E = np.fft.ifft(A * np.exp(1j * kzf * z))
            return np.interp(xo, xfull, E.real) + 1j * np.interp(xo, xfull, E.imag)
        return field
    Mset = sorted({max(20, Mc - 20), max(20, Mc - 10), Mc, Mc + 20})
    stable = []
    for M in Mset:
        res = solve_rcwa_1d(lam0, 0, 1.0, 1.0, D, [(col.astype(complex), H)], 2 * M + 1, pol=(0., 1.))
        sig = res["Rtot"] + res["Ttot"]; field = propagate(res)
        zc = np.linspace(0.5 * f, 1.6 * f, 200)      # search around the design focus (excludes near field / periodic replicas)
        zf = zc[int(np.argmax([abs(field(z, np.array([X0]))[0]) ** 2 for z in zc]))]
        if abs(sig - 1) < 0.03:
            stable.append((M, field, zf, sig))
    if not stable:
        figpath = _out(slug + ".png")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No stable M found\nUse FDTD (Meep)", ha="center"); ax.axis("off")
        fig.savefig(figpath, dpi=110); plt.close(fig)
        return figpath, "- ⚠️ Energy was not conserved at any order (ill-conditioned). FDTD required.", "unstable — FDTD required"
    zfs = [s[2] for s in stable]
    M, field, zf, sig = max(stable, key=lambda s: s[0])   # largest stable M = best converged
    xo = np.linspace(X0 - 4, X0 + 4, 1401); If = abs(field(zf, xo)) ** 2; If /= If.max()
    fwhm = metrics.fwhm_1d(If, xo)                   # main-lobe FWHM (interpolated)
    zg = np.linspace(1, 2.2 * f, 200); xm = np.linspace(X0 - 4, X0 + 4, 240)
    Imap = np.array([abs(field(z, xm)) ** 2 for z in zg])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].imshow(Imap.T, origin="lower", aspect="auto", extent=[zg.min(), zg.max(), xm.min() - X0, xm.max() - X0], cmap="inferno")
    ax[0].axvline(f, color="cyan", ls="--", lw=1); ax[0].axvline(zf, color="w", ls=":", lw=1)
    ax[0].set_xlabel("z (um)"); ax[0].set_ylabel("x (um)"); ax[0].set_title("Full-wave I(x,z)")
    ax[1].plot((xo - X0) * 1000, If); ax[1].set_xlim(-2000, 2000); ax[1].axhline(0.5, color="gray", ls=":")
    ax[1].set_xlabel("x (nm)"); ax[1].set_ylabel("I"); ax[1].set_title(f"Focus — FWHM={fwhm*1000:.0f}nm")
    fig.suptitle(f"FULL-WAVE metalens (1D) D={D_lens:.1f}um f={f}um NA={NA:.2f} | {len(stable)} stable M, focus z={zf:.2f}um")
    figpath = _out(slug + ".png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    body = (f"- **Method:** whole lens as a single super-cell RCWA (NO local approximation); exit field windowed to the lens aperture + exact propagation\n"
            f"- **Stability:** {len(stable)}/{len(Mset)} orders conserved energy; focus consistency z = {np.mean(zfs):.2f} ± {np.std(zfs):.3f} um\n"
            f"- **Actual focus:** z = {zf:.2f} um (design f = {f} um) | **FWHM:** {fwhm*1000:.0f} nm (λ/(2NA)={lam0/(2*NA)*1000:.0f} nm)\n"
            f"- **NA:** {NA:.2f} | Pillars: {Ncell} | Selected M: {M} | ΣDE: {sig:.4f}\n"
            f"- **Fresnel number:** N = a²/(λf) = {(D_lens/2)**2/(lam0*f):.1f} — for N ≲ 5 the intensity maximum sits closer to the lens than the geometric focus (focal shift), so z < f is expected for small lenses.\n"
            f"- **Convergence:** the super-cell needs M ≳ 1.5·n·D/λ = {1.5*n_pillar*D/lam0:.0f} orders (used M = {M}) "
            f"{'✅' if M >= 1.5*n_pillar*D/lam0 else '⚠️ under-resolved — increase M; the focus position is not converged yet'}\n"
            f"- Note: a thick + high-index super-cell is ill-conditioned; for large lenses FDTD (Meep, e.g. example 09) is recommended.")
    summary = f"focus z={zf:.2f}um (f={f}), FWHM={fwhm*1000:.0f}nm, {len(stable)} stable M"
    return figpath, body, summary


def run_metalens_layout_2d(fm, slug):
    """2D metalens design: phase library -> circular layout -> locally-periodic focus."""
    lam0 = num(fm, "lam0", 0.633); Lcell = num(fm, "period", 0.35)
    H = num(fm, "height", 0.60); n_pillar = num(fm, "n_pillar", 2.4)
    mat = fm.get("material", ""); shape = fm.get("shape", "square"); n_sub = num(fm, "n_sub", 1.0)
    D_lens = num(fm, "D_lens", 20.0); f = num(fm, "f", 25.0); M = int(num(fm, "M", 9, int))
    d = dz.design_lens(lam0, Lcell, H, D_lens, f, shape=shape, material=mat, n_pillar=n_pillar, M=M, n_sub=n_sub)
    a = dz.analyze_focus(d)
    D_lens = d["D_lens"]; NA = d["NA"]; ci = d["ci"]; inside = d["inside"]
    zf = a["z_focus"]; fwhm = a["fwhm"]; I = a["I"]; xax = a["x"]; eff = a["eff_legacy"]
    cut = I[:, a["peak_ij"][1]] / I.max()
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    sm = np.where(inside, d["size_map"] * 1000, np.nan)
    im0 = ax[0].imshow(sm.T, origin="lower", extent=[ci[0], ci[-1], ci[0], ci[-1]], cmap="viridis")
    ax[0].set_title("Pillar size (nm)"); ax[0].set_xlabel("x (um)"); ax[0].set_ylabel("y (um)"); plt.colorbar(im0, ax=ax[0], fraction=0.046)
    im1 = ax[1].imshow((I / I.max()).T, origin="lower", extent=[xax[0], xax[-1], xax[0], xax[-1]], cmap="inferno")
    ax[1].set_xlim(-3, 3); ax[1].set_ylim(-3, 3); ax[1].set_title(f"Focus I(x,y) z={zf:.1f}um"); ax[1].set_xlabel("x (um)"); ax[1].set_ylabel("y (um)"); plt.colorbar(im1, ax=ax[1], fraction=0.046)
    ax[2].plot(xax * 1000, cut); ax[2].axhline(0.5, color="gray", ls=":"); ax[2].set_xlim(-2500, 2500); ax[2].set_xlabel("x (nm)"); ax[2].set_ylabel("I"); ax[2].set_title(f"FWHM={fwhm*1000:.0f}nm")
    fig.suptitle(f"2D METALENS D={D_lens:.0f}um f={f}um NA={NA:.2f} focus z={zf:.1f}um efficiency~{eff*100:.0f}%")
    figpath = _out(slug + ".png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    body = (f"- **Method:** 2D phase library -> circular layout -> locally-periodic propagation (NOT full-wave)\n"
            f"- **Aperture:** {int(inside.sum())} pillars, D={D_lens:.1f}um, NA={NA:.2f}, Fresnel number {a['fresnel_number']:.1f}\n"
            f"- **Focus:** z={zf:.1f}um (design f={f}um) | FWHM={fwhm*1000:.0f}nm (λ/(2NA)={a['diffraction_fwhm']*1000:.0f}nm) | Strehl {a['strehl']:.2f}\n"
            f"- **Focusing efficiency (locally-periodic):** ~{eff*100:.0f}% (1.5×FWHM disk / transmitted)\n"
            f"- For final verification use FDTD/GPU (see example 09).")
    summary = f"2D lens focus z={zf:.1f}um, FWHM={fwhm*1000:.0f}nm, efficiency~{eff*100:.0f}%"
    return figpath, body, summary


def _eps_rect(wx, wy, ang, L, n_pillar, N=None):
    """Rotated rectangular bar (wx x wy, angle in rad) centred in an L x L cell."""
    return shapes.rasterize("rect", (wx, wy), L, N or shapes.DEFAULT_N, n_pillar=n_pillar, angle=ang)


def _jones(eps, lam0, L, H, M):
    rx = solve_rcwa_2d(lam0, 0, 0, 1, 1, L, L, [(eps, H)], M, M, pol=(1., 0.))
    ry = solve_rcwa_2d(lam0, 0, 0, 1, 1, L, L, [(eps, H)], M, M, pol=(0., 1.))
    txx = rx["tx"][rx["i0"]]; tyx = rx["ty"][rx["i0"]]
    txy = ry["tx"][ry["i0"]]; tyy = ry["ty"][ry["i0"]]
    return np.array([[txx, txy], [tyx, tyy]])


def run_wavelength_sweep(fm, slug):
    """Spectral response of one 2D meta-atom (dispersive material supported)."""
    L = num(fm, "period", 0.35); H = num(fm, "height", 0.60)
    n_pillar = num(fm, "n_pillar", 2.4); side = num(fm, "side", 0.18); M = int(num(fm, "M", 8, int))
    mat = fm.get("material", "")   # e.g. "TiO2","Si","SiO2"; if empty, constant n_pillar
    l0 = num(fm, "lam_min", 0.45); l1 = num(fm, "lam_max", 0.75); nl = int(num(fm, "n_lam", 25, int))
    lams = np.linspace(l0, l1, nl); amp = []; ph = []; Tt = []
    for lam in lams:
        eps_hi = materials.eps_of(mat, lam) if mat else n_pillar ** 2      # complex: absorption included
        eps = _eps_square_pillar(L, side, eps_hi, 1.0)
        r = solve_rcwa_2d(lam, 0, 0, 1, 1, L, L, [(eps, H)], M, M, pol=(0., 1.))
        t0 = r["ty"][r["i0"]]; amp.append(abs(t0)); ph.append(-np.angle(t0)); Tt.append(r["Ttot"])   # physical phase
    ph = np.unwrap(np.array(ph))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(lams * 1000, Tt, label="T total"); ax[0].plot(lams * 1000, amp, "--", label="|t0|")
    ax[0].set_xlabel("Wavelength (nm)"); ax[0].set_ylabel("Efficiency/amplitude"); ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title("Spectral response")
    ax[1].plot(lams * 1000, (ph - ph[0]) / np.pi, color="C2"); ax[1].set_xlabel("Wavelength (nm)"); ax[1].set_ylabel("Phase (π)"); ax[1].grid(alpha=.3); ax[1].set_title("0th-order phase")
    fig.suptitle(f"Wavelength sweep — pillar {side*1000:.0f}nm, H={H}um, n={n_pillar}"); fig.tight_layout()
    figpath = _out(slug + ".png"); fig.savefig(figpath, dpi=130); plt.close(fig)
    body = (f"- Range: {l0*1000:.0f}–{l1*1000:.0f} nm ({nl} points) | material: {mat or ('n=%.2f'%n_pillar)}\n"
            f"- T: {min(Tt):.3f}–{max(Tt):.3f} | phase coverage: {np.ptp(ph)/np.pi:.2f}π")
    return figpath, body, f"λ∈[{l0*1000:.0f},{l1*1000:.0f}]nm, T∈[{min(Tt):.2f},{max(Tt):.2f}]"


def run_birefringent_library(fm, slug):
    """Birefringent (wx, wy) meta-atom library: x/y phases and retardance map."""
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
    for a, D, tit in zip(ax, [PX / np.pi, PY / np.pi, ret / np.pi], ["phase_x (π)", "phase_y (π)", "retardance φx-φy (π)"]):
        im = a.imshow(D, origin="lower", extent=[w0*1000, w1*1000, w0*1000, w1*1000], cmap="twilight")
        a.set_xlabel("wy (nm)"); a.set_ylabel("wx (nm)"); a.set_title(tit); plt.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"Birefringent meta-atom library (λ={lam0}, H={H}, n={n_pillar})"); fig.tight_layout()
    figpath = _out(slug + ".png"); fig.savefig(figpath, dpi=130); plt.close(fig)
    body = (f"- wx,wy ∈ {w0*1000:.0f}–{w1*1000:.0f} nm ({nw}×{nw} grid)\n"
            f"- Max retardance: {np.max(np.abs(ret))/np.pi:.2f}π (a half-wave plate needs ~1π)")
    return figpath, body, f"birefringence {nw}x{nw}, max retardance {np.max(np.abs(ret))/np.pi:.2f}π"


def run_jones_matrix(fm, slug):
    """Jones matrix of a rotated rectangular meta-atom + circular-polarization conversion."""
    L = num(fm, "period", 0.35); H = num(fm, "height", 0.60); lam0 = num(fm, "lam0", 0.633)
    n_pillar = num(fm, "n_pillar", 2.4); M = int(num(fm, "M", 8, int))
    wx = num(fm, "wx", 0.26); wy = num(fm, "wy", 0.10); ang = np.deg2rad(num(fm, "angle", 0.0))
    J = _jones(_eps_rect(wx, wy, ang, L, n_pillar), lam0, L, H, M)
    Lc = np.array([1, 1j]) / np.sqrt(2); Rc = np.array([1, -1j]) / np.sqrt(2)
    tRL = Rc.conj() @ (J @ Lc); tLL = Lc.conj() @ (J @ Lc)
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["txx", "txy", "tyx", "tyy"]; vals = [abs(J[0,0]), abs(J[0,1]), abs(J[1,0]), abs(J[1,1])]
    ax.bar(labels, vals, color="C0"); ax.set_ylim(0, 1.05); ax.set_ylabel("|t|"); ax.set_title("Jones matrix amplitudes")
    figpath = _out(slug + ".png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    body = (f"- **Jones (amplitude∠phase°):**\n"
            f"  - txx={abs(J[0,0]):.2f}∠{np.angle(J[0,0],deg=True):.0f}  txy={abs(J[0,1]):.2f}∠{np.angle(J[0,1],deg=True):.0f}\n"
            f"  - tyx={abs(J[1,0]):.2f}∠{np.angle(J[1,0],deg=True):.0f}  tyy={abs(J[1,1]):.2f}∠{np.angle(J[1,1],deg=True):.0f}\n"
            f"- Retardance (φx−φy): {(np.angle(J[0,0])-np.angle(J[1,1]))*180/np.pi:.0f}°\n"
            f"- Cross-circular |tRL|={abs(tRL):.2f}, co-circular |tLL|={abs(tLL):.2f}  (for good PB, |tRL|→1)")
    return figpath, body, f"retardance {(np.angle(J[0,0])-np.angle(J[1,1]))*180/np.pi:.0f}°, |tRL|={abs(tRL):.2f}"


def run_pb_phase_library(fm, slug):
    """Pancharatnam-Berry (geometric) phase versus rotation angle of a bar."""
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
    ax[0].plot(angs, phase / np.pi, "o-", label="measured"); ax[0].plot(angs, 2 * angs / 180, "--", label="2θ (theory)")
    ax[0].set_xlabel("Rotation θ (°)"); ax[0].set_ylabel("Cross-circular phase (π)"); ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title("PB geometric phase")
    ax[1].plot(angs, amp, "s-", color="C1"); ax[1].set_ylim(0, 1.05); ax[1].set_xlabel("θ (°)"); ax[1].set_ylabel("|tRL|"); ax[1].grid(alpha=.3); ax[1].set_title("Conversion efficiency")
    fig.suptitle(f"PB (geometric) phase library — bar {wx*1000:.0f}×{wy*1000:.0f}nm"); fig.tight_layout()
    figpath = _out(slug + ".png"); fig.savefig(figpath, dpi=130); plt.close(fig)
    lin = np.corrcoef(phase / np.pi, 2 * angs / 180)[0, 1]
    body = (f"- Bar: {wx*1000:.0f}×{wy*1000:.0f} nm, {na} angles\n"
            f"- Linearity with 2θ (correlation): {lin:.4f}  | mean |tRL|: {np.mean(amp):.2f}")
    return figpath, body, f"PB linearity r={lin:.3f}, <|tRL|>={np.mean(amp):.2f}"


def run_param_map_2d(fm, slug):
    """Two-parameter map (height x size) of the meta-atom phase and amplitude."""
    L = num(fm, "period", 0.35); lam0 = num(fm, "lam0", 0.633); n_pillar = num(fm, "n_pillar", 2.4); M = int(num(fm, "M", 6, int))
    h0 = num(fm, "h_min", 0.30); h1 = num(fm, "h_max", 0.90); nh = int(num(fm, "n_h", 8, int))
    s0 = num(fm, "s_min", 0.06); s1 = num(fm, "s_max", 0.30); ns = int(num(fm, "n_s", 8, int))
    hs = np.linspace(h0, h1, nh); ss = np.linspace(s0, s1, ns)
    PH = np.zeros((nh, ns)); AM = np.zeros((nh, ns))
    for i, H in enumerate(hs):
        for j, s in enumerate(ss):
            eps = _eps_square_pillar(L, s, n_pillar ** 2, 1.0)
            r = solve_rcwa_2d(lam0, 0, 0, 1, 1, L, L, [(eps, H)], M, M, pol=(0., 1.))
            t0 = r["ty"][r["i0"]]; PH[i, j] = -np.angle(t0); AM[i, j] = abs(t0)   # physical phase
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    im0 = ax[0].imshow(PH / np.pi, origin="lower", extent=[s0*1000, s1*1000, h0, h1], aspect="auto", cmap="twilight")
    ax[0].set_xlabel("Pillar side (nm)"); ax[0].set_ylabel("Height (um)"); ax[0].set_title("0th-order phase (π)"); plt.colorbar(im0, ax=ax[0], fraction=0.046)
    im1 = ax[1].imshow(AM, origin="lower", extent=[s0*1000, s1*1000, h0, h1], aspect="auto", cmap="viridis")
    ax[1].set_xlabel("Pillar side (nm)"); ax[1].set_ylabel("Height (um)"); ax[1].set_title("|t0|"); plt.colorbar(im1, ax=ax[1], fraction=0.046)
    fig.suptitle(f"2-parameter map (height×width) — λ={lam0}, n={n_pillar}"); fig.tight_layout()
    figpath = _out(slug + ".png"); fig.savefig(figpath, dpi=130); plt.close(fig)
    body = (f"- Height {h0}–{h1}um × side {s0*1000:.0f}–{s1*1000:.0f}nm ({nh}×{ns})\n"
            f"- Phase coverage: {np.ptp(PH)/np.pi:.2f}π | amplitude {AM.min():.2f}–{AM.max():.2f}")
    return figpath, body, f"map {nh}x{ns}, phase {np.ptp(PH)/np.pi:.2f}π"


def _prop2d(E0, dx, k0):
    N = E0.shape[0]
    kxx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    KX, KY = np.meshgrid(kxx, kxx, indexing="ij")
    KZ = np.sqrt((k0 ** 2 - KX ** 2 - KY ** 2).astype(complex)); KZ = np.where(np.imag(KZ) < 0, -KZ, KZ)
    A0 = np.fft.fft2(E0)
    return lambda z: np.fft.ifft2(A0 * np.exp(1j * KZ * z))


def run_inverse_design_deflector(fm, slug):
    """Inverse design: topology optimization of a unit cell toward a target order (beam deflector)."""
    from scipy.optimize import differential_evolution
    lam0 = num(fm, "lam0", 0.633); period = num(fm, "period", 1.30); H = num(fm, "height", 0.60)
    eps_hi = num(fm, "eps_hi", 6.25); Nseg = int(num(fm, "n_seg", 12, int))
    target = int(num(fm, "target_order", 1, int)); Mo = int(num(fm, "M", 15, int))
    maxiter = int(num(fm, "maxiter", 18, int)); Nx = Nseg * 40
    def make(dens): return np.repeat(1.0 + np.clip(dens, 0, 1) * (eps_hi - 1.0), Nx // Nseg)
    def Tt(dens):
        r = solve_scalar_1d(lam0, 0, 1, 1, period, make(dens), H, Mo, "TE", "laurent")
        return r["T"][np.where(r["orders"] == target)[0][0]]
    T0 = Tt(np.random.default_rng(0).random(Nseg))          # local generator: never reseed the global one
    res = differential_evolution(lambda d: -Tt(d), [(0, 1)] * Nseg, maxiter=maxiter,
                                 popsize=8, seed=1, polish=False, tol=1e-3)
    dens = res.x; eps = make(dens)
    r = solve_scalar_1d(lam0, 0, 1, 1, period, eps, H, Mo, "TE", "laurent")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].step(np.arange(Nseg), np.clip(dens, 0, 1), where="mid"); ax[0].set_ylim(-0.05, 1.05)
    ax[0].set_xlabel("Segment"); ax[0].set_ylabel("Fill (0=air,1=dielectric)"); ax[0].set_title("Optimized unit cell")
    o = r["orders"]; T = r["T"]; sel = T > 0.005
    ax[1].bar(o[sel], T[sel]); ax[1].set_xlabel("Order"); ax[1].set_ylabel("T"); ax[1].set_title(f"Order efficiencies (target {target:+d})")
    fig.suptitle(f"Inverse design — order {target:+d} deflector"); fig.tight_layout()
    figpath = _out(slug + ".png"); fig.savefig(figpath, dpi=130); plt.close(fig)
    body = (f"- **Initial (random) T[{target:+d}]:** {T0:.3f}\n"
            f"- **Optimized T[{target:+d}]:** {-res.fun:.3f}  ({res.nfev} evaluations)\n"
            f"- ΣDE = {r['Rtot']+r['Ttot']:.4f} | segments: {Nseg}, orders: {Mo}\n"
            f"- Method: differential_evolution (gradient-free); forward model = scalar RCWA.")
    return figpath, body, f"inverse design T[{target:+d}]: {T0:.2f}->{-res.fun:.2f}"


def run_achromatic_analysis(fm, slug):
    """Chromatic focal shift of a 1D cylindrical metalens vs wavelength (chromatic aberration).
    The actual RCWA phase (complex ty from solve_rcwa_1d) is used at each wavelength."""
    Lcell = num(fm, "period", 0.35); H = num(fm, "height", 0.60); n_pillar = num(fm, "n_pillar", 2.4)
    D_lens = num(fm, "D_lens", 12.0); f0 = num(fm, "f", 15.0); lam0 = num(fm, "lam0", 0.633)
    l0 = num(fm, "lam_min", 0.55); l1 = num(fm, "lam_max", 0.72); nl = int(num(fm, "n_lam", 9, int))
    Ncell = int(round(D_lens / Lcell)); D_lens = Ncell * Lcell
    widths = np.linspace(0.05, Lcell - 0.05, 28)
    lams = np.linspace(l0, l1, nl)
    # (width -> 0th-order phase) library at each wavelength
    def phase_at(lam):
        ph = []
        for w in widths:
            x = (np.arange(128) + 0.5) / 128 * Lcell
            eps = np.where(np.abs(x - Lcell/2) < w/2, n_pillar**2, 1.0).astype(complex)
            r = solve_rcwa_1d(lam, 0, 1.0, 1.0, Lcell, [(eps, H)], 15, pol=(0., 1.))
            ph.append(-np.angle(r["ty"][r["zeroth_index"]]))   # physical convention (rcwa/convention.py)
        return np.array(ph)
    PH = {round(lam, 5): phase_at(lam) for lam in lams}
    ph0 = np.unwrap(PH[round(lam0 if lam0 in lams else lams[np.argmin(abs(lams-lam0))],5)]) if False else np.unwrap(phase_at(lam0))
    ph0 -= ph0.min(); ph0m = np.mod(ph0, 2*np.pi)
    xc = (np.arange(Ncell) - (Ncell-1)/2) * Lcell
    phi_need = np.mod(-(2*np.pi/lam0) * (np.sqrt(xc**2 + f0**2) - f0), 2*np.pi)
    chosen = np.array([np.argmin(np.abs(np.angle(np.exp(1j*(ph0m - p))))) for p in phi_need])
    focus = []
    for lam in lams:
        atom_ph = PH[round(lam,5)][chosen]           # phase of the selected atoms at THIS wavelength
        E0 = np.exp(1j*atom_ph)
        Npad = 4096; dx = Lcell; Ec = np.zeros(Npad, complex); o=(Npad-Ncell)//2; Ec[o:o+Ncell]=E0
        kx = 2*np.pi*np.fft.fftfreq(Npad, d=dx); kz=np.sqrt(((2*np.pi/lam)**2-kx**2).astype(complex)); kz=np.where(np.imag(kz)<0,-kz,kz)
        A0=np.fft.fft(Ec); zc=np.linspace(0.4*f0, 1.8*f0, 140)
        Ia=[np.abs(np.fft.ifft(A0*np.exp(1j*kz*z))[Npad//2])**2 for z in zc]
        focus.append(zc[int(np.argmax(Ia))])
    focus=np.array(focus)
    fig, ax = plt.subplots(figsize=(7,4.2))
    ax.plot(lams*1000, focus, "o-"); ax.axhline(f0,color="gray",ls=":",label="design f")
    ax.axvline(lam0*1000,color="C3",ls="--",label="design wavelength")
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Focus z (um)")
    ax.set_title("Chromatic focal shift (RCWA phase)"); ax.legend(); ax.grid(alpha=.3)
    figpath=_out(slug+".png"); fig.tight_layout(); fig.savefig(figpath,dpi=130); plt.close(fig)
    shift=focus.max()-focus.min()
    body=(f"- Design: D={D_lens:.1f}um, f={f0}um @ lambda={lam0*1000:.0f}nm (actual RCWA phase at each wavelength)\n"
          f"- over lambda {l0*1000:.0f}-{l1*1000:.0f}nm the focus spans {focus.min():.1f}-{focus.max():.1f}um -> **chromatic shift {shift:.1f}um**\n"
          f"- Achromatic design: the shift is removed by matching the meta-atom dispersion (dphi/dlambda).")
    return figpath, body, f"chromatic focal shift {shift:.1f}um"


def run_vortex_beam(fm, slug):
    """OAM vortex generator — azimuthal phase exp(i·l·θ) (realizable with PB)."""
    lam0 = num(fm, "lam0", 0.633); D = num(fm, "aperture", 12.0); ell = int(num(fm, "charge", 1, int))
    zf = num(fm, "z", 20.0); k0 = 2 * np.pi / lam0
    N = 512; ext = D * 1.6; dx = ext / N
    x = (np.arange(N) - N / 2) * dx; X, Y = np.meshgrid(x, x, indexing="ij")
    R = np.hypot(X, Y); TH = np.arctan2(Y, X)
    E0 = np.where(R <= D / 2, np.exp(1j * ell * TH), 0.0)
    prop = _prop2d(E0, dx, k0); Ez = prop(zf); I = np.abs(Ez) ** 2; ph = np.angle(Ez)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].imshow(np.angle(E0), extent=[-ext/2, ext/2, -ext/2, ext/2], cmap="twilight"); ax[0].set_title(f"Input phase (l={ell})")
    ax[1].imshow(I / I.max(), extent=[-ext/2, ext/2, -ext/2, ext/2], cmap="inferno"); ax[1].set_xlim(-4, 4); ax[1].set_ylim(-4, 4); ax[1].set_title(f"Intensity @ z={zf}um (donut)")
    ax[2].imshow(ph, extent=[-ext/2, ext/2, -ext/2, ext/2], cmap="twilight"); ax[2].set_xlim(-4, 4); ax[2].set_ylim(-4, 4); ax[2].set_title("Phase (singularity)")
    fig.suptitle(f"OAM vortex beam — topological charge l={ell}"); fig.tight_layout()
    figpath = _out(slug + ".png"); fig.savefig(figpath, dpi=130); plt.close(fig)
    cen = I[N//2, N//2] / I.max()
    body = (f"- Topological charge l={ell}, aperture D={D}um\n"
            f"- Center intensity/max = {cen:.3f} (~0 for a vortex; donut ring)\n"
            f"- 2πl phase singularity at the center. Realized with PB (rotated-bar) meta-atoms.")
    return figpath, body, f"vortex l={ell}, center I~{cen:.2f}"


def run_hologram(fm, slug):
    """Phase-only hologram (Gerchberg-Saxton) — reconstructs the target pattern."""
    N = int(num(fm, "N", 128, int)); iters = int(num(fm, "iters", 30, int))
    # target: a simple 'L' or ring pattern
    target = np.zeros((N, N)); c = N // 2
    yy, xx = np.mgrid[0:N, 0:N]; rr = np.hypot(xx - c, yy - c)
    target[(rr > N * 0.18) & (rr < N * 0.24)] = 1.0  # ring
    target[(np.abs(xx - c) < 2) & (yy > c)] = 1.0     # line
    amp = np.sqrt(target); 
    field = np.exp(1j * 2 * np.pi * np.random.default_rng(int(num(fm, "seed", 0, int))).random((N, N)))
    for _ in range(iters):
        F = np.fft.fftshift(np.fft.fft2(field))
        F = amp * np.exp(1j * np.angle(F))
        field = np.fft.ifft2(np.fft.ifftshift(F))
        field = np.exp(1j * np.angle(field))   # phase-only constraint
    recon = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2; recon /= recon.max()
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].imshow(target, cmap="gray"); ax[0].set_title("Target pattern")
    ax[1].imshow(np.angle(field), cmap="twilight"); ax[1].set_title("Phase mask (metasurface)")
    ax[2].imshow(recon, cmap="inferno"); ax[2].set_title("Reconstruction")
    fig.suptitle("Phase-only hologram (Gerchberg-Saxton)"); fig.tight_layout()
    figpath = _out(slug + ".png"); fig.savefig(figpath, dpi=130); plt.close(fig)
    err = np.mean((recon - target / target.max()) ** 2)
    body = (f"- {N}×{N} phase mask, {iters} GS iterations\n"
            f"- Reconstruction mean-squared error ~ {err:.3f}\n"
            f"- The phase mask is realized with a meta-atom phase library.")
    return figpath, body, f"hologram GS, error~{err:.3f}"


def run_polarization_multiplexed(fm, slug):
    """Polarization multiplexing — deflect x and y polarization in different directions (birefringence)."""
    lam0 = num(fm, "lam0", 0.633); D = num(fm, "aperture", 12.0)
    gx = num(fm, "grad_x", 0.15); gy = num(fm, "grad_y", -0.15); zf = num(fm, "z", 18.0)
    k0 = 2 * np.pi / lam0; N = 512; ext = D * 1.8; dx = ext / N
    x = (np.arange(N) - N / 2) * dx; X, Y = np.meshgrid(x, x, indexing="ij"); R = np.hypot(X, Y)
    mask = R <= D / 2
    Ex = np.where(mask, np.exp(1j * k0 * gx * X), 0.0)   # x-pol: toward +x
    Ey = np.where(mask, np.exp(1j * k0 * gy * X), 0.0)   # y-pol: toward -x
    Ix = np.abs(_prop2d(Ex, dx, k0)(zf)) ** 2; Iy = np.abs(_prop2d(Ey, dx, k0)(zf)) ** 2
    def peak_x(I): 
        j = np.unravel_index(np.argmax(I), I.shape); return x[j[0]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].imshow(Ix / Ix.max(), extent=[-ext/2, ext/2, -ext/2, ext/2], cmap="inferno"); ax[0].set_title(f"x-pol @ z={zf}um (peak x={peak_x(Ix):+.1f})")
    ax[1].imshow(Iy / Iy.max(), extent=[-ext/2, ext/2, -ext/2, ext/2], cmap="inferno"); ax[1].set_title(f"y-pol @ z={zf}um (peak x={peak_x(Iy):+.1f})")
    fig.suptitle("Polarization multiplexing: x and y in different directions (birefringent meta-atom)"); fig.tight_layout()
    figpath = _out(slug + ".png"); fig.savefig(figpath, dpi=130); plt.close(fig)
    body = (f"- x-pol peak position x={peak_x(Ix):+.1f}um, y-pol x={peak_x(Iy):+.1f}um → opposite directions\n"
            f"- Birefringent (wx≠wy) meta-atoms impart independent phases to the two polarizations (Jones).")
    return figpath, body, f"polmux: x->{peak_x(Ix):+.1f}, y->{peak_x(Iy):+.1f}"


def run_metalens_design_full(fm, slug):
    """Flagship: shape+material library -> 2D lens design -> metrics (FWHM, efficiency,
    Strehl, MTF, ...) -> fabrication-ready GDS + design JSON for the full-wave bridges."""
    lam0 = num(fm, "lam0", 0.633); L = num(fm, "period", 0.375); H = num(fm, "height", 0.60)
    mat = fm.get("material", "TiO2_Siefke"); shape = fm.get("shape", "circle")
    D_lens = num(fm, "D_lens", 16.0); f = num(fm, "f", 20.0); M = int(num(fm, "M", 9, int))
    n_sub = _substrate_index(fm, lam0); sel = fm.get("selection", "robust")
    sig = num(fm, "sigma_nm", 3.0); cert = bool(int(num(fm, "certify", 0, int)))
    d = dz.design_lens(lam0, L, H, D_lens, f, shape=shape, material=mat,
                       n_pillar=num(fm, "n_pillar", 2.4), M=M, n_sub=n_sub, mode=sel,
                       sigma_nm=sig, refine=(sel == "robust"), certify=cert)
    a = dz.analyze_focus(d)
    lib = d["lib"]; n_pillar = float(lib["n"].real); k_pillar = float(lib["n"].imag)
    D_lens = d["D_lens"]; NA = d["NA"]; ci = d["ci"]; inside = d["inside"]; placements = d["placements"]
    zf = a["z_focus"]; fwhm = a["fwhm"]; I = a["I"]; xax = a["x"]; St = a["strehl"]
    dl_fwhm = a["diffraction_fwhm"] * 1000; dof = a["dof"]; ee80 = a["ee80"]
    ff = a["field"]; pj = a["peak_ij"][1]
    Iideal = dz.FocalField(dz.ideal_aperture(d["E0"], L, lam0, zf), L, lam0).intensity(zf)
    st_mtf = metrics.strehl_mtf(metrics.mtf_2d(I), metrics.mtf_2d(Iideal))
    # GDS + design JSON
    gdspath = _out(slug + ".gds"); npoly = gds.layout_to_gds(placements, gdspath)
    from . import exporting
    min_feat = num(fm, "min_feature", 0.05); min_gap = num(fm, "min_gap", 0.04)
    drc = exporting.drc_check(placements, min_feature=min_feat, min_gap=min_gap)
    mnote = ""
    if mat:
        mi = materials.info(mat)
        mnote = (f"- **Material data:** {mi['note']}"
                 + (" — ⚠️ approximate: use measured n,k (`materials.register_yaml`) before fabrication" if mi["approximate"] else "") + "\n")
    import json
    design = {"lam0": lam0, "Lcell": L, "height": H, "n_pillar": n_pillar, "shape": shape,
              "D_lens": D_lens, "f": f, "material": mat, "n_sub": n_sub,
              "rcwa_focus_um": round(zf, 3), "rcwa_fwhm_nm": round(fwhm * 1000, 1),
              "placements": [{"x": p["x"], "y": p["y"], "params": p["params"], "angle": 0.0} for p in placements]}
    with open(_out(slug + "_design.json"), "w") as _jf:
        json.dump(design, _jf)
    # figure
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    sm = np.where(inside, d["size_map"] * 1000, np.nan)
    im0 = ax[0].imshow(sm.T, origin="lower", extent=[ci[0], ci[-1], ci[0], ci[-1]], cmap="viridis")
    ax[0].set_title(f"{shape} size (nm)"); ax[0].set_xlabel("x(um)"); ax[0].set_ylabel("y(um)"); plt.colorbar(im0, ax=ax[0], fraction=0.046)
    ax[1].imshow((I / I.max()).T, origin="lower", extent=[xax[0], xax[-1], xax[0], xax[-1]], cmap="inferno"); ax[1].set_xlim(-3, 3); ax[1].set_ylim(-3, 3)
    ax[1].set_title(f"Focus I(x,y) z={zf:.1f}um"); ax[1].set_xlabel("x(um)"); ax[1].set_ylabel("y(um)")
    mtf = metrics.mtf_1d(I[:, pj]); fr = np.fft.fftshift(np.fft.fftfreq(ff.N, d=ff.dx))
    ax[2].plot(fr[ff.N // 2:], mtf[ff.N // 2:]); ax[2].set_xlim(0, 2 / lam0); ax[2].set_xlabel("Spatial frequency (1/um)"); ax[2].set_ylabel("MTF"); ax[2].set_title("MTF"); ax[2].grid(alpha=.3)
    fig.suptitle(f"FULL PACKAGE design — {mat}/{shape}, D={D_lens:.0f}um f={f}um NA={NA:.2f}")
    figpath = _out(slug + ".png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    abs_pct = a["eff_absolute"] * 100
    cov = 2 * lib["coverage"]
    body = (f"- **Material:** {mat or 'constant index'} (n={n_pillar:.2f}{f', k={k_pillar:.3f}' if k_pillar > 1e-6 else ''}) · **Shape:** {shape} · **Pillars:** {int(inside.sum())} · **NA:** {NA:.2f} (f/{f/D_lens:.1f}) · substrate n={n_sub}\n"
            f"- **Library:** phase coverage {cov:.2f}π {'✅' if cov >= 1.9 else '⚠️ below 2π — increase the height or the index'} · |t| {lib['amp'].min():.2f}–{lib['amp'].max():.2f} · selection: {sel}\n"
            f"- **Focus:** z={zf:.2f}um (design {f}um; Fresnel number {a['fresnel_number']:.1f}{' — small lens, focal shift toward the lens is expected' if a['fresnel_number'] < 5 else ''}) · **FWHM:** {fwhm*1000:.0f}nm (diffraction λ/2NA={dl_fwhm:.0f}nm)\n"
            f"\n**📐 Professional metrics (literature-standard):**\n\n"
            f"| Metric | Value | Note |\n|---|---|---|\n"
            f"| Efficiency — absolute (3×FWHM/incident) | **{abs_pct:.0f}%** | standard reporting |\n"
            f"| Efficiency — relative (3×FWHM/transmitted) | {a['eff_relative']*100:.0f}% | normalized to transmission |\n"
            f"| Efficiency — legacy (1.5×FWHM) | {a['eff_legacy']*100:.0f}% | old definition |\n"
            f"| Strehl (peak, same-plane reference) | {St:.2f} | ≥0.8 diffraction-limited |\n"
            f"| Strehl (MTF area) | {st_mtf:.2f} | alternative definition |\n"
            f"| FWHM / diffraction | {fwhm*1000:.0f} / {dl_fwhm:.0f} nm | ratio {fwhm*1000/dl_fwhm:.2f} |\n"
            f"| Encircled energy 80% | {ee80*1000:.0f} nm | spot concentration |\n"
            f"| Sidelobe level | {a['sidelobe_dB']:.1f} dB | lower = cleaner spot |\n"
            f"| Depth of focus (DOF) | {dof:.1f} um | on-axis I(z) FWHM |\n\n"
            f"- **GDS:** `{os.path.basename(gdspath)}` — {npoly} pillars, {os.path.getsize(gdspath)/1e6:.2f} MB (hierarchical)\n"
            f"- **DRC** (feature ≥ {min_feat*1000:.0f}nm, gap ≥ {min_gap*1000:.0f}nm): {'✅ clean' if drc['ok'] else '❌ ' + str(drc['n_feature_viol']) + ' feature / ' + str(drc['n_gap_viol']) + ' gap violations'} · smallest pillar {drc['min_size']*1000:.0f}nm, smallest gap {drc['worst_gap']*1000:.0f}nm\n"
            + mnote +
            f"- **For 3D verification:** `{slug}_design.json` → fullwave (Meep/Tidy3D)\n"
            f"- Locally-periodic approximation; final verification is FDTD. (Metric definitions: see `rcwa/metrics.py`.)")
    return figpath, body, f"z={zf:.1f}um FWHM={fwhm*1000:.0f}nm eff(abs){abs_pct:.0f}% Strehl{St:.2f} DOF{dof:.1f}um GDS✓"


def run_tolerance_analysis(fm, slug):
    """Fabrication-tolerance Monte Carlo. Effect of pillar-size error on efficiency."""
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
    # first build a good design (short optimization), then measure the tolerance of that design
    _res = differential_evolution(lambda d: -Tt(d), [(0, 1)] * Nseg, maxiter=12, popsize=8, seed=1, polish=False, tol=1e-3)
    base = _res.x; T0 = -_res.fun
    vals = [Tt(base + rng.normal(0, sigma / (period / Nseg), Nseg)) for _ in range(ntrial)]
    vals = np.array(vals)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(vals * 100, bins=15, color="C0", alpha=.8)
    ax.axvline(T0 * 100, color="C3", ls="--", label=f"error-free {T0*100:.0f}%")
    ax.axvline(vals.mean() * 100, color="k", ls=":", label=f"mean {vals.mean()*100:.0f}%")
    ax.set_xlabel(f"T[{target:+d}] efficiency (%)"); ax.set_ylabel("Number of trials")
    ax.set_title(f"Fabrication tolerance — σ={sigma*1000:.0f}nm pillar error, {ntrial} trials"); ax.legend()
    figpath = _out(slug + ".png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    body = (f"- Error-free efficiency: {T0*100:.0f}% · Error σ={sigma*1000:.0f}nm ({ntrial} trials)\n"
            f"- **Mean: {vals.mean()*100:.0f}% ± {vals.std()*100:.0f}%** · worst: {vals.min()*100:.0f}%\n"
            f"- Efficiency drop (mean): {(T0-vals.mean())*100:.0f} points → indicator of fabrication robustness.")
    return figpath, body, f"tolerance: {T0*100:.0f}%->{vals.mean()*100:.0f}±{vals.std()*100:.0f}%"


def run_achromatic_design(fm, slug):
    """ACHROMATIC METALENS — group-delay (GD) engineering. Selects meta-atoms so that both
    the phase φ(λ0) and GD=dφ/dω match → all wavelengths focus at the same z (the chromatic
    shift is reduced). The (size×height) library provides GD diversity. The GD range of simple
    pillars is limited → the achromatic aperture is limited (physics: the aperture×bandwidth
    product is bounded by the max GD). This generator shows that concretely + compares with a singlet."""
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
                eps = shapes.rasterize(shape, s, L, shapes.DEFAULT_N, n_pillar=n_pillar)
                for k, lam in enumerate(lams):
                    r = solve_rcwa_2d(lam, 0, 0, 1, 1, L, L, [(eps, H)], M, M, pol=(0., 1.))
                    t0 = r["ty"][r["i0"]]; PH[a, b, k] = np.angle(t0); AM[a, b, k] = abs(t0)
        return {"PH": PH, "AM": AM}
    c = cache.cached_library("achrolib", {"lam0": lam0, "band": band, "L": L, "H": (hmin, hmax, nh),
                             "mat": mat, "shape": shape, "M": M, "ns": nsz}, _lib)
    PH = c["PH"]; AM = c["AM"]
    # each atom: physical phase (convention: -φ_rcwa) and GD=dφ/dω (central difference)
    phi_phys = np.mod(-PH, 2 * np.pi)                       # (nsz,nh,3)
    ph0 = phi_phys[:, :, 1].ravel()                        # phase at λ0
    unwrapped = np.unwrap(-PH, axis=2)                     # continuous phase for GD
    GD = ((unwrapped[:, :, 2] - unwrapped[:, :, 0]) / (om[2] - om[0])).ravel()
    S, Hh = np.meshgrid(sizes, heights, indexing="ij")
    size_flat = S.ravel(); h_flat = Hh.ravel()
    # target profile
    Nc = int(round(D / L)); D = Nc * L; ci = (np.arange(Nc) - (Nc - 1) / 2) * L
    XX, YY = np.meshgrid(ci, ci, indexing="ij"); R = np.hypot(XX, YY); inside = R <= D / 2
    phi_req = np.mod(-(2 * np.pi / lam0) * (np.sqrt(R ** 2 + f ** 2) - f), 2 * np.pi)
    gd_req = -(np.sqrt(R ** 2 + f ** 2) - f) * 1e-6 / c0    # s (meters/c) — REQUIRED GD SHAPE
    gr = gd_req[inside]
    gd_span_need = float(gr.max() - gr.min())
    gd_span_have = float(GD.max() - GD.min())
    # relative GD is what matters (global piston is free): align the required GD range with the
    # CENTER of the atoms' GD range → so cells can find atoms across the whole GD range
    gd_req = gd_req - (gr.max() + gr.min()) / 2.0 + (GD.max() + GD.min()) / 2.0
    # assign atoms: |Δφ|/π + w*|ΔGD|/GD_required (both terms O(1) → GD really matters)
    gdsc = max(gd_span_need, 1e-18)
    size_map = np.zeros((Nc, Nc)); h_map = np.zeros((Nc, Nc))
    ins_ij = [(i, j) for i in range(Nc) for j in range(Nc) if inside[i, j]]
    idxA = -np.ones((Nc, Nc), int); idxS = -np.ones((Nc, Nc), int)   # achromatic / singlet (phase-only)
    phflat = phi_phys.reshape(-1, 3); amflat = AM.reshape(-1, 3)
    for (i, j) in ins_ij:
        dphi = np.abs(np.angle(np.exp(1j * (ph0 - phi_req[i, j])))) / np.pi
        dgd = np.abs(GD - gd_req[i, j]) / gdsc
        kA = int(np.argmin(dphi + gdw * dgd)); kS = int(np.argmin(dphi))
        idxA[i, j] = kA; idxS[i, j] = kS
        size_map[i, j] = size_flat[kA]; h_map[i, j] = h_flat[kA]
    u = 3
    def _prop_multilam(idxmap):
        fz = []; fw = []
        for k, lam in enumerate(lams):
            phk = phflat[:, k]; amk = amflat[:, k]; Ek = np.zeros((Nc, Nc), complex)
            for (i, j) in ins_ij:
                Ek[i, j] = amk[idxmap[i, j]] * np.exp(1j * phk[idxmap[i, j]])
            ff = dz.FocalField(Ek, L, lam, u=u)          # exact on-axis focus, padded plane field
            zf = ff.find_focus(f, zwin=(0.5, 1.6), nz=40)[0]; I = ff.intensity(zf)
            xax = ff.x; jc = int(np.argmin(np.abs(xax))); cut = I[:, jc] / I[:, jc].max()
            ab = np.where(cut >= 0.5)[0]; fw.append((xax[ab[-1]] - xax[ab[0]]) * 1000 if ab.size >= 2 else float("nan"))
            fz.append(zf)
        return fz, fw
    lam_focus, lam_fwhm = _prop_multilam(idxA)
    sfz, _ = _prop_multilam(idxS)
    chroma = max(lam_focus) - min(lam_focus)
    singlet_chroma = max(sfz) - min(sfz)                   # MEASURED singlet (phase-only, same atoms) — fair comparison
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(lams * 1000, lam_focus, "o-", label="achromatic design")
    ax[0].axhline(f, color="gray", ls=":", label=f"design f={f}um")
    ax[0].set_xlabel("wavelength (nm)"); ax[0].set_ylabel("focus z (um)")
    ax[0].set_title(f"Focus vs λ · chromatic shift {chroma:.2f}um (singlet~{singlet_chroma:.1f}um)"); ax[0].legend(); ax[0].grid(alpha=.3)
    sm = np.where(inside, h_map, np.nan)
    im = ax[1].imshow(sm.T, origin="lower", extent=[ci[0], ci[-1], ci[0], ci[-1]], cmap="plasma")
    ax[1].set_title("Pillar height (um) — GD control"); ax[1].set_xlabel("x(um)"); ax[1].set_ylabel("y(um)"); plt.colorbar(im, ax=ax[1], fraction=0.046)
    fig.suptitle(f"ACHROMATIC METALENS — {mat}, D={D:.0f}um f={f}um, {lmin*1000:.0f}-{lmax*1000:.0f}nm")
    figpath = _out(slug + ".png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    cover = 100 * min(1.0, gd_span_have / max(gd_span_need, 1e-18))
    body = (f"- **Band:** {lmin*1000:.0f}-{lmax*1000:.0f}nm · **D={D:.0f}um f={f}um** · (size×height) {nsz}×{nh} library\n"
            f"- **Chromatic shift:** achromatic **{chroma:.2f}um** vs simple singlet ~{singlet_chroma:.1f}um → **{singlet_chroma/max(chroma,1e-6):.1f}× improvement**\n"
            f"- **Group delay:** required range {gd_span_need*1e15:.1f} fs, library {gd_span_have*1e15:.1f} fs → **coverage {cover:.0f}%**\n"
            f"- Coverage <100% reflects the physical aperture×bandwidth limit: a larger D/wider band needs taller/resonant meta-atoms (more GD).\n"
            f"- Focus per λ: {['%.1f'%z for z in lam_focus]} um, FWHM {['%.0f'%w for w in lam_fwhm]} nm")
    return figpath, body, f"achromatic: chromatic {chroma:.2f}um (singlet~{singlet_chroma:.1f}um), GD coverage {cover:.0f}%"


def _build_lens(fm):
    """Shared lens builder for the FOV / index-robustness analyses (design.design_lens)."""
    lam0 = num(fm, "lam0", 0.633); L = num(fm, "period", 0.375); H = num(fm, "height", 0.60)
    mat = fm.get("material", "TiO2_Siefke"); shape = fm.get("shape", "circle")
    D_lens = num(fm, "D_lens", 10.0); f = num(fm, "f", 12.0); M = int(num(fm, "M", 9, int))
    return dz.design_lens(lam0, L, H, D_lens, f, shape=shape, material=mat,
                          n_pillar=num(fm, "n_pillar", 2.4), M=M, n_sub=_substrate_index(fm, lam0))


def _prop_focus(E0, L, lam0, f, tilt_deg=0.0, zfix=None):
    """Propagate the aperture field (optionally tilted) and return peak / position / FWHM.
    zfix: evaluate in that plane only (flat sensor); otherwise search the best focus."""
    ff = dz.FocalField(E0, L, lam0, tilt_deg=tilt_deg)
    zf = zfix if zfix is not None else ff.find_focus(f, zwin=(0.6, 1.4), nz=25, off_axis=bool(tilt_deg))[0]
    I = ff.intensity(zf); j = np.unravel_index(np.argmax(I), I.shape)
    return dict(peak=float(I.max()), zf=float(zf), x_peak=float(ff.x[j[0]]),
                fwhm=metrics.fwhm_1d(I[:, j[1]], ff.x), I=I, xf=ff.x)


def run_fov_analysis(fm, slug):
    """Off-axis IMAGE QUALITY: focal shift vs angle (distortion), spot degradation
    (coma/field aberration ~ FWHM growth), relative efficiency (Strehl), field curvature (focus z shift).
    FOV = the half-angle within which the lens stays diffraction-limited.
    atom_angle=1 solves every used pillar size again with RCWA at each angle (rcwa/angular.py,
    roadmap P2) and compares with the former assumption of an angle-independent atom response."""
    d = _build_lens(fm); f = d["f"]; lam0 = d["lam0"]; NA = d["NA"]; dl = lam0 / (2 * NA)
    amax = num(fm, "angle_max", 15.0); na = int(num(fm, "n_angle", 8, int))
    atom_angle = bool(int(num(fm, "atom_angle", 0, int))); pol = fm.get("pol", "TE")
    angles = np.linspace(0, amax, na)
    from . import angular
    a_sub = angular.first_order_angle(lam0, d["L"], d["lib"]["n_sub"])
    r0 = _prop_focus(d["E0"], d["L"], lam0, f)
    peak0 = r0["peak"]; z0 = r0["zf"]
    th, xpk, zpk, fw, rstrehl, dist = [], [], [], [], [], []
    for a in angles:
        r = _prop_focus(d["E0"], d["L"], lam0, f, tilt_deg=a, zfix=z0)   # FLAT sensor (fixed z0)
        rc = _prop_focus(d["E0"], d["L"], lam0, f, tilt_deg=a)            # best focus (field curvature)
        ideal = z0 * np.tan(np.deg2rad(a))                         # chief ray on the sensor plane z0
        th.append(a); xpk.append(r["x_peak"]); zpk.append(rc["zf"]); fw.append(r["fwhm"] * 1000)
        rstrehl.append(r["peak"] / peak0)
        dist.append(100 * (r["x_peak"] - ideal) / ideal if ideal > 1e-6 else 0.0)
    fw = np.array(fw); rstrehl = np.array(rstrehl); th = np.array(th)
    cmp = angular.fov_compare(d, angles, pol=pol) if atom_angle else None
    if cmp is not None:                               # FOV judged with the angle-dependent atoms
        fw_a = np.array(cmp["atoms"]["fwhm"]) * 1000; rs_a = np.array(cmp["atoms"]["rel_peak"])
        ok = (fw_a <= 1.5 * dl * 1000) & (rs_a >= 0.5)
    else:
        ok = (fw <= 1.5 * dl * 1000) & (rstrehl >= 0.5)
    fov_half = float(th[ok].max()) if ok.any() else 0.0
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].plot(th, fw, "o-", label="atom response fixed" if cmp else None)
    ax[1].plot(th, rstrehl, "o-", label="atom response fixed" if cmp else None)
    if cmp is not None:
        ax[0].plot(th, fw_a, "s--", label="angle-dependent atoms")
        ax[1].plot(th, rs_a, "s--", label="angle-dependent atoms")
    for axx in ax[:2]:
        if a_sub is not None and a_sub <= amax:
            axx.axvline(a_sub, color="red", ls=":", lw=1, label=f"substrate order opens ({a_sub:.1f}°)")
    ax[0].axhline(dl * 1000, color="gray", ls=":", label=f"diffraction {dl*1000:.0f}nm")
    ax[0].set_xlabel("incidence angle (°)"); ax[0].set_ylabel("focal FWHM (nm)"); ax[0].set_title("Spot degradation (field aberration)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[1].axhline(0.8, color="gray", ls=":", label="0.8 (diffraction-limited)")
    ax[1].set_xlabel("incidence angle (°)"); ax[1].set_ylabel("relative peak (Iθ/I0)"); ax[1].set_title(f"Angular efficiency · half-FOV≈{fov_half:.0f}°"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    ax[2].plot(th, dist, "o-"); ax[2].axhline(0, color="gray", ls=":")
    ax[2].set_xlabel("incidence angle (°)"); ax[2].set_ylabel("distortion (%)"); ax[2].set_title("Distortion (x vs z0·tanθ)"); ax[2].grid(alpha=.3)
    fig.suptitle(f"OFF-AXIS IMAGE QUALITY — {d['shape']}/{fm.get('material', RECIPE['material'])}, D={d['D_lens']:.0f}um f={f}um NA={NA:.2f}")
    figpath = _out(slug + ".png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    fc = float(np.max(np.abs(np.array(zpk) - z0)))
    body = (f"- **NA:** {NA:.2f} · diffraction FWHM {dl*1000:.0f}nm · normal-incidence focus z={z0:.1f}um\n"
            f"- **Half-FOV (diffraction-limited):** ~{fov_half:.0f}° (FWHM≤1.5×diffraction & relative peak≥0.5"
            f"{', angle-dependent atoms' if cmp else ''})\n"
            f"- **At the max angle ({amax:.0f}°):** FWHM={fw[-1]:.0f}nm, relative peak={rstrehl[-1]:.2f}, distortion={dist[-1]:.1f}% "
            f"(peak position on the FFT grid, step {d['L'] / 4 * 1000:.0f}nm)"
            f"{' (fixed atom response)' if cmp else ''}\n"
            f"- **Field curvature (focus z deviation):** {fc:.2f}um\n")
    if cmp is not None:
        body += (f"- **Angle-dependent atoms ({pol}):** at {amax:.0f}° relative peak {rs_a[-1]:.2f} (fixed response {rstrehl[-1]:.2f}), "
                 f"FWHM {fw_a[-1]:.0f}nm, mean pillar transmission {cmp['mean_T'][-1]:.3f} (0°: {cmp['mean_T'][0]:.3f})\n")
    else:
        body += ("- Atom response assumed independent of the angle (normal-incidence library + tilt phase); "
                 "use atom_angle=1 for the RCWA response at each angle.\n")
    if a_sub is not None:
        body += (f"- A first diffraction order of the period propagates in the substrate above {a_sub:.1f}°: "
                 "the fixed-response model is optimistic there.\n")
    body += "- Flat singlet metalens: limited FOV (coma). For wide FOV use a doublet/quadratic design."
    return figpath, body, f"half-FOV~{fov_half:.0f}°, at {amax:.0f}° FWHM={fw[-1]:.0f}nm rel. peak={rstrehl[-1]:.2f}"


def run_index_robustness(fm, slug):
    """REFRACTIVE-INDEX ROBUSTNESS: how the focusing efficiency/FWHM change when the pillar index
    deviates by n±Δn (material/process error). The layout (selected sizes) stays fixed; the
    transmission of the used sizes is recomputed at each index."""
    d = _build_lens(fm); f = d["f"]; lam0 = d["lam0"]; L = d["L"]; H = d["H"]
    lib = d["lib"]; shape = d["shape"]; NA = d["NA"]; dl = lam0 / (2 * NA)
    n0 = lib["n"]; dn_max = num(fm, "dn_max", 0.15); ndn = int(num(fm, "n_dn", 5, int))
    inside = d["inside"]; idx = d["idx"]
    used = np.unique(idx[inside])                          # only the unique sizes actually used
    dns = np.linspace(-dn_max, dn_max, ndn); effs = []; fwhms = []
    for dn in dns:
        lb = dz.meta_atom_library(lam0, L, H, sizes=lib["sizes"][used], shape=shape, n_pillar=n0 + dn,
                                  M=lib["M"], n_sub=lib["n_sub"])
        t_full = np.zeros(len(lib["sizes"]), complex); t_full[used] = lb["t"]
        ff = dz.FocalField(dz.aperture_field(t_full, idx), L, lam0)
        zf = ff.find_focus(f)[0]; I = ff.intensity(zf)
        fw = metrics.focal_metrics(I, ff.x)["fwhm_x"]
        effs.append(metrics.focusing_efficiency(I, ff.x, fw)); fwhms.append(fw * 1000)
    effs = np.array(effs) * 100; fwhms = np.array(fwhms)
    i0 = int(np.argmin(np.abs(dns))); n0r = float(n0.real)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(dns, effs, "o-"); ax[0].axvline(0, color="gray", ls=":"); ax[0].set_xlabel("Δn (index error)"); ax[0].set_ylabel("focusing efficiency (%)"); ax[0].set_title("Efficiency vs index error"); ax[0].grid(alpha=.3)
    ax[1].plot(dns, fwhms, "o-"); ax[1].axhline(dl * 1000, color="gray", ls=":", label=f"diffraction {dl*1000:.0f}nm"); ax[1].axvline(0, color="gray", ls=":")
    ax[1].set_xlabel("Δn"); ax[1].set_ylabel("FWHM (nm)"); ax[1].set_title("Spot vs index error"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle(f"REFRACTIVE-INDEX ROBUSTNESS — {shape}/{fm.get('material','')} n0={n0r:.2f}, D={d['D_lens']:.0f}um f={f}um")
    figpath = _out(slug + ".png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    # sensitivity: efficiency drop / unit Δn (at the extremes)
    sens = float((effs[i0] - min(effs[0], effs[-1])) / dn_max)
    body = (f"- **Nominal n={n0r:.2f}:** efficiency {effs[i0]:.0f}%, FWHM {fwhms[i0]:.0f}nm (diffraction {dl*1000:.0f}nm)\n"
            f"- **At ±{dn_max:.2f} index error:** efficiency {effs[0]:.0f}%…{effs[-1]:.0f}%, FWHM {fwhms.min():.0f}…{fwhms.max():.0f}nm\n"
            f"- **Sensitivity:** ~{sens:.0f} efficiency points / unit Δn → indicator of fabrication/material index tolerance\n"
            f"- High sensitivity = index-sensitive design; a lower NA / robust meta-atom choice improves it.")
    return figpath, body, f"n-robustness: ±{dn_max:.2f}Δn -> efficiency {effs[0]:.0f}…{effs[-1]:.0f}%, sensitivity {sens:.0f}p/Δn"


def run_lens_tolerance(fm, slug):
    """LENS fabrication tolerance: uniform size bias + random per-pillar error (Monte Carlo) ->
    Strehl / absolute-efficiency distribution, default vs robust atom selection side by side.
    The size response comes from an adaptively refined library (resonances resolved)."""
    lam0 = num(fm, "lam0", 0.633); L = num(fm, "period", 0.375); H = num(fm, "height", 0.60)
    mat = fm.get("material", "TiO2_Siefke"); shape = fm.get("shape", "circle")
    D_lens = num(fm, "D_lens", 16.0); f = num(fm, "f", 20.0); M = int(num(fm, "M", 9, int))
    n_sub = _substrate_index(fm, lam0); n_pillar = num(fm, "n_pillar", 2.4)
    bmax = num(fm, "bias_max_nm", 6.0); nb = int(num(fm, "n_bias", 7, int))
    sigma = num(fm, "sigma_nm", 2.0); ntr = int(num(fm, "n_trial", 10, int))
    rsig = num(fm, "robust_sigma_nm", 3.0); guard = num(fm, "guard_nm", -1.0)
    rmax = int(num(fm, "refine_max_solves", 80, int)); seed = int(num(fm, "seed", 0, int))
    import warnings
    base = dz.meta_atom_library(lam0, L, H, shape=shape, material=mat, n_pillar=n_pillar, M=M, n_sub=n_sub)
    ref = dz.refine_library(base, max_solves=rmax)
    kw = dict(shape=shape, material=mat, n_pillar=n_pillar, M=M, n_sub=n_sub)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        designs = {"complex (0.2 rule)": dz.design_lens(lam0, L, H, D_lens, f, lib=base, mode="complex",
                                                       refine=False, **kw),
                   f"robust σ={rsig:g}nm": dz.design_lens(lam0, L, H, D_lens, f, lib=ref, mode="robust",
                                                         sigma_nm=rsig, guard_nm=guard if guard >= 0 else None, **kw)}
    biases = np.linspace(-bmax, bmax, nb)
    tol = {k: dz.lens_tolerance(d, ref, bias_nm=biases, sigma_nm=sigma, n_trial=ntr, seed=seed)
           for k, d in designs.items()}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for c, (k, r) in enumerate(tol.items()):
        for a_, key in ((ax[0], "eff_stats"), (ax[1], "strehl_stats")):
            m, sd = r[key]["mean"], r[key]["std"]
            a_.plot(biases, m, "o-", color=f"C{c}", label=k)
            a_.fill_between(biases, m - sd, m + sd, color=f"C{c}", alpha=.2)
    for a_, yl in ((ax[0], "absolute efficiency"), (ax[1], "Strehl")):
        a_.set_xlabel("uniform size bias (nm)"); a_.set_ylabel(yl); a_.grid(alpha=.3); a_.legend()
        a_.axvline(0, color="gray", ls=":")
    ax[0].set_title(f"Efficiency (mean ± std, random σ={sigma:g}nm, {next(iter(tol.values()))['n_trial']} trials)")
    ax[1].set_title("Strehl")
    d0 = next(iter(designs.values()))
    fig.suptitle(f"LENS TOLERANCE — {mat}/{shape}, D={d0['D_lens']:.0f}um f={f}um NA={d0['NA']:.2f}")
    figpath = _out(slug + ".png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    i0 = int(np.argmin(np.abs(biases)))
    rows = "".join(f"| {b:+.1f} | " + " | ".join(
        f"{r['eff_stats']['mean'][i]:.3f} ± {r['eff_stats']['std'][i]:.3f} / {r['strehl_stats']['mean'][i]:.3f}"
        for r in tol.values()) + " |\n" for i, b in enumerate(biases))
    drop = {k: float(r["eff_stats"]["mean"][i0] - r["eff_stats"]["mean"].min()) for k, r in tol.items()}
    res = ", ".join(f"{x['center']*1000:.1f} nm (|t|={x['t_min']:.2f}, width {x['width']*1000:.1f} nm)"
                    for x in ref["resonances"]) or "none found"
    oor = max(r["out_of_range"] for r in tol.values())
    names = list(tol)
    nom = {k: dz.analyze_focus(d) for k, d in designs.items()}
    nomtxt = " · ".join(f"{k}: eff {a['eff_absolute']:.3f}, Strehl {a['strehl']:.3f}" for k, a in nom.items())
    body = (f"- **Library:** {len(base['sizes'])} sizes refined to {len(ref['sizes'])} "
            f"({ref['n_refine_solves']} extra solves{'' if ref['refine_converged'] else ', budget exhausted'}) · "
            f"**resonances:** {res}\n"
            f"- **Error model:** size = nominal + bias + N(0, ({sigma:g} nm)²) per pillar; t(s) interpolated "
            f"on the refined library (no re-solve). Pillars pushed outside the library range: up to "
            f"{oor*100:.1f}% (clamped to the end sizes).\n"
            f"- **Error-free:** {nomtxt}\n\n"
            f"| bias (nm) | {names[0]}: eff ± std / Strehl | {names[1]}: eff ± std / Strehl |\n|---|---|---|\n"
            + rows +
            f"\n- **Largest mean-efficiency drop over ±{bmax:g} nm:** "
            + " · ".join(f"{k}: {v*100:.1f} points" for k, v in drop.items()) + "\n"
            "- Locally-periodic approximation; near-resonance atoms are where it is least reliable. "
            "Final verification is FDTD.")
    return figpath, body, ("tolerance ±%gnm: " % bmax) + ", ".join(f"{k} drop {v*100:.1f}pt" for k, v in drop.items())


def run_recipe_search(fm, slug):
    """UNIT-CELL RECIPE SEARCH: pillar height x lattice period for a material and wavelength under
    fabrication limits (min feature, min gap, aspect ratio). Scores each refined library by the
    focal efficiency of an ideal lens built from it with the robust rule (size error sigma)."""
    from . import recipe
    lam0 = num(fm, "lam0", 0.633); mat = fm.get("material", "TiO2_Siefke"); shape = fm.get("shape", "circle")
    sub = fm.get("substrate", "SiO2_Malitson")
    n_sub = float(np.real(np.sqrt(materials.eps_of(sub, lam0)))) if sub not in ("", "air", "none") else 1.0
    Hs = np.round(np.linspace(num(fm, "H_min", 0.55), num(fm, "H_max", 0.80), int(num(fm, "n_H", 6, int))), 6)
    Ls = np.round(np.linspace(num(fm, "L_min", 0.30), num(fm, "L_max", 0.40), int(num(fm, "n_L", 5, int))), 6)
    kw = dict(material=mat, shape=shape, n_sub=n_sub, min_feature=num(fm, "min_feature", 0.06),
              min_gap=num(fm, "min_gap", 0.06), max_aspect=num(fm, "max_aspect", 15.0),
              M=int(num(fm, "M", 9, int)), n_sizes=int(num(fm, "n_sizes", 16, int)),
              sigma_nm=num(fm, "sigma_nm", 3.0), certify=bool(int(num(fm, "certify", 0, int))))
    max_gap = num(fm, "max_phase_gap", float(np.pi / 4))
    rows = recipe.search_unit_cell(lam0, Hs, Ls, **kw)
    F = np.full((len(Hs), len(Ls)), np.nan)
    for r in rows:
        if r["feasible"]:
            F[int(np.argmin(np.abs(Hs - r["H"]))), int(np.argmin(np.abs(Ls - r["L"])))] = r["fom_robust"]
    best = recipe.recommend(rows, max_gap=max_gap, tie=num(fm, "fom_tie", 0.01))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    hw = lambda v: 0.5 * (v[1] - v[0]) if len(v) > 1 else 0.005           # noqa: E731  half cell
    im = ax.imshow(F, origin="lower", aspect="auto", cmap="viridis",
                   extent=[(Ls[0] - hw(Ls)) * 1000, (Ls[-1] + hw(Ls)) * 1000,
                           (Hs[0] - hw(Hs)) * 1000, (Hs[-1] + hw(Hs)) * 1000])
    plt.colorbar(im, ax=ax, label="robust FOM (focal efficiency of an ideal lens)")
    for r in rows:
        if r["feasible"] and r["max_phase_gap"] > max_gap:
            ax.plot(r["L"] * 1000, r["H"] * 1000, "x", color="w")
    if best:
        ax.plot(best["L"] * 1000, best["H"] * 1000, "*", color="r", ms=14, label="recommended")
        ax.legend(loc="upper left")
    ax.set_xlabel("period L (nm)"); ax.set_ylabel("height H (nm)")
    ax.set_title(f"Recipe search — {mat}/{shape}, λ = {lam0*1000:.0f} nm, substrate n = {n_sub:.3f}\n"
                 f"(x: phase gap > {max_gap:.2f} rad; ★: recommended)", fontsize=10)
    figpath = _out(slug + ".png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    idn = num(fm, "index_dn", 0.0); sens_txt = ""
    if best and idn > 0:                        # material-index sensitivity of the recommended recipe
        sens = recipe.index_sensitivity(lam0, best["H"], best["L"], material=mat, shape=shape,
                                        dn=(-2 * idn, -idn, idn, 2 * idn), n_sub=n_sub,
                                        min_feature=kw["min_feature"], min_gap=kw["min_gap"], M=kw["M"],
                                        n_sizes=kw["n_sizes"], sigma_nm=kw["sigma_nm"])
        sens_txt = ("- **Index sensitivity of the recommended recipe** (atoms fixed at the nominal index, used sizes re-solved): "
                    f"n0 = {sens['n0'].real:.3f}, FOM {sens['fom_nominal']:.3f} → "
                    + ", ".join(f"Δn {d:+.3f}: {v:.3f}" for d, v in sorted(sens["fom"].items())) + "\n")
    feas = [r for r in rows if r["feasible"]][:8]
    tab = "".join(f"| {r['H']*1000:.0f} | {r['L']*1000:.0f} | {r['s_min']*1000:.0f}–{r['s_max']*1000:.0f} | {r['aspect']:.1f} | "
                  f"{r['coverage_2pi']:.2f} | {r['max_phase_gap']:.2f} | {r['min_abs_t']:.2f} | "
                  f"{', '.join(f'{c*1000:.0f} ({d:.2f})' for c, d in zip(r['resonances'], r['resonance_depth'])) or '—'} | "
                  f"{r['na_max_sampling']:.2f} | {r['fom_nominal']:.3f} | **{r['fom_robust']:.3f}** |\n"
                  for r in feas)
    infeas = [r for r in rows if not r["feasible"]]
    body = (f"- **Platform:** {mat} ({shape}) at λ = {lam0*1000:.0f} nm, substrate n = {n_sub:.3f} ({sub}), exit medium air\n"
            f"- **Limits:** min feature {kw['min_feature']*1000:.0f} nm, min gap {kw['min_gap']*1000:.0f} nm, aspect ratio ≤ {kw['max_aspect']:g}; "
            f"size error σ = {kw['sigma_nm']:g} nm; RCWA M = {kw['M']}\n"
            + (f"- **Recommended:** H = {best['H']*1000:.0f} nm, L = {best['L']*1000:.0f} nm, sizes {best['s_min']*1000:.0f}–{best['s_max']*1000:.0f} nm "
               f"— robust FOM {best['fom_robust']:.3f}, phase gap {best['max_phase_gap']:.2f} rad, "
               f"{best['n_deep_resonances']} deep resonance(s), sampling allows NA ≤ {best['na_max_sampling']:.2f} "
               f"(rule: within {num(fm, 'fom_tie', 0.01):.2f} of the best FOM, fewest deep resonances, then highest NA)\n" if best else
               f"- **No candidate** closes the phase circle within {max_gap:.2f} rad under these limits: relax a limit or widen the scan.\n")
            + sens_txt
            + "\n| H (nm) | L (nm) | sizes (nm) | aspect | coverage (2π) | max phase gap (rad) | min \\|t\\| | resonances nm (depth \\|t\\|) | max NA (λ/2L) | FOM nominal | FOM robust |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n" + tab
            + ("\n- Convergence certificate (M vs M+2) applied: uncertified sizes are excluded from the FOM and the phase gap. "
               + (f"Recommended recipe: {best['n_uncertified']} uncertified size(s)"
                  + (f" ({', '.join(f'{v:.1f}' for v in best['uncertified_nm'])} nm) — do not use them" if best['n_uncertified'] else "") + ".\n"
                  if best else "\n") if kw["certify"] else "")
            + (f"\n- Infeasible candidates: {len(infeas)} ({'; '.join(sorted(set(r['reason'] for r in infeas)))})\n" if infeas else "")
            + "- FOM = focal intensity of an ideal lens from this library relative to a perfect phase plate; "
              "locally periodic approximation, normal incidence. Verify the chosen recipe with FDTD (examples/17).")
    summ = (f"recommended H={best['H']*1000:.0f}nm L={best['L']*1000:.0f}nm FOM {best['fom_robust']:.3f}" if best
            else "no recipe closes the phase circle")
    return figpath, body, summ


def run_metalens_large_tiled(fm, slug):
    """cm-scale TILED large lens. Vectorized size selection (loop-free) +
    tile-by-tile GDS (bounded memory) + preview + cm-scale projection."""
    from rcwa import largelens as LL
    import time
    lam0 = num(fm, "lam0", 0.633); L = num(fm, "period", 0.375); H = num(fm, "height", 0.60)
    mat = fm.get("material", "TiO2_Siefke"); shape = fm.get("shape", "circle")
    n_sub = _substrate_index(fm, lam0)
    D_lens = num(fm, "D_lens", 100.0); f = num(fm, "f", 200.0); M = int(num(fm, "M", 7, int))
    tile = int(num(fm, "tile_cells", 300, int))
    do_gds = str(fm.get("write_gds", "1")).lower() not in ("0", "false", "hayir", "no")
    gdsdir = _out(slug + "_gds") if do_gds else None
    t0 = time.time()
    st = LL.design_large_lens(lam0, L, H, mat, shape, D_lens, f, M, gds_dir=gdsdir, tile_cells=tile, n_sub=n_sub)
    dt = time.time() - t0
    ci = st["ci_ds"]; sm = np.where(st["preview_size_nm"] > 0, st["preview_size_nm"], np.nan)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    im = ax[0].imshow(sm.T, origin="lower", extent=[ci[0], ci[-1], ci[0], ci[-1]], cmap="viridis")
    ax[0].set_title("Pillar size (nm) — preview"); ax[0].set_xlabel("x(um)"); ax[0].set_ylabel("y(um)"); plt.colorbar(im, ax=ax[0], fraction=0.046)
    r = np.linspace(0, st["D_lens"] / 2, 400); phi = np.mod(-(2 * np.pi / lam0) * (np.sqrt(r ** 2 + f ** 2) - f), 2 * np.pi)
    ax[1].plot(r, phi); ax[1].set_xlabel("r (um)"); ax[1].set_ylabel("target phase (rad)"); ax[1].set_title("Radial phase profile"); ax[1].grid(alpha=.3)
    fig.suptitle(f"LARGE LENS (tiled) — {mat}/{shape} D={st['D_lens']:.0f}um NA={st['NA']:.2f}")
    figpath = _out(slug + ".png"); fig.tight_layout(); fig.savefig(figpath, dpi=130); plt.close(fig)
    gb_now = LL.scale_projection(st["n_atoms"])
    na_cm = int(np.pi / 4 * (10000.0 / L) ** 2); gb_cm = LL.scale_projection(na_cm)
    body = (f"- **Aperture:** D={st['D_lens']:.0f}um, f={f}um, NA={st['NA']:.2f} · **{st['n_atoms']:,} meta-atoms** (vectorized/loop-free selection)\n"
            f"- **Tiled GDS:** {st['n_tiles']} tiles, {st['npoly']:,} pillars, ~{gb_now:.3f} GB (hierarchical) → `{os.path.basename(gdsdir) if gdsdir else '(not written)'}`\n"
            f"- **Time:** {dt:.1f}s · memory bounded by the tile size (cm-scale feasible)\n"
            f"- **Scale projection:** 1cm aperture ≈ {na_cm:,} atoms, ~{gb_cm:.0f} GB GDS — the pipeline works; only disk/time remain.\n"
            f"- Verification: the LUT selection is the same library and LUT rule as `metalens_design_full` (global phase offset fixed at 0); for 3D use a representative section or the Tidy3D cloud.")
    return figpath, body, f"D={st['D_lens']:.0f}um {st['n_atoms']:,} atoms, {st['n_tiles']} tiles ~{gb_now:.2f}GB"


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
           "lens_tolerance": run_lens_tolerance,
           "recipe_search": run_recipe_search,
           "fov_analysis": run_fov_analysis, "index_robustness": run_index_robustness,
           "metalens_large_tiled": run_metalens_large_tiled,
           "achromatic_design": run_achromatic_design}


# ----------------------------- report + update -----------------------------
_META_KEYS = ("etiketler", "ilgili_proje_veya_alan", "id", "tur", "durum", "olusturulma", "son_rapor", "deney_tipi",
              "tags", "kind", "status", "created", "last_report", "experiment_type")


def _is_english(fm):
    """Notes may use English front-matter keys (experiment_type / status: pending) or the
    original Turkish ones (deney_tipi / durum: bekliyor). The report follows the note's style."""
    return "experiment_type" in fm


def write_report(exp_name, fm, figpath, body, summary):
    ts = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    en = _is_english(fm)
    title = f"{'Report' if en else 'Rapor'} - {exp_name} - {stamp}"
    figname = os.path.basename(figpath)
    params = "\n".join(f"| {k} | {v} |" for k, v in fm.items() if k not in _META_KEYS)
    if en:
        head = f"""---
id: {stamp}
created: {ts}
kind: report
status: 🟢 done
experiment: "[[{exp_name}]]"
tags:
  - rcwa
  - report
---

# 📊 {title}

Experiment: [[{exp_name}]]  ·  Run: {ts}
"""
    else:
        head = f"""---
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

⬅️ [[🧪 RCWA Deney Panosu]] · Experiment: [[{exp_name}]]  ·  Run: {ts}
"""
    md = head + f"""
> **Summary:** {summary}

## 🔧 Parameters
| Key | Value |
|---|---|
{params}

## 📈 Results
{body}

## 🖼️ Figure
![[{figname}]]

## 🔍 Review / Notes
*(Add your own comments here: is this as expected? what should the next experiment be?)*
- 
"""
    os.makedirs(RAPORLAR, exist_ok=True)
    path = os.path.join(RAPORLAR, title + ".md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return title


def mark_done(exp_path, report_title, english=False):
    """Set the note's status to done and link the report (keys follow the note's language)."""
    k_status, v_done, k_rep = ("status", "🟢 done", "last_report") if english else ("durum", "🟢 tamamlandi", "son_rapor")
    with open(exp_path, "r", encoding="utf-8") as f:
        txt = f.read()
    if re.search(rf"(?m)^{k_status}:", txt):
        txt = re.sub(rf"(?m)^{k_status}:.*$", f"{k_status}: {v_done}", txt, count=1)
    else:                                                    # no status line yet: add one after the opening ---
        txt = re.sub(r"\A---[ \t]*\r?\n", f"---\n{k_status}: {v_done}\n", txt, count=1)
    link = f'{k_rep}: "[[{report_title}]]"'
    if re.search(rf"(?m)^{k_rep}:", txt):
        txt = re.sub(rf"(?m)^{k_rep}:.*$", lambda m: link, txt, count=1)
    else:
        txt = re.sub(rf"(?m)^({k_status}: .*)$", lambda m: m.group(1) + "\n" + link, txt, count=1)
    with open(exp_path, "w", encoding="utf-8") as f:
        f.write(txt)


def run_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    fm, _ = parse_frontmatter(text)
    name = os.path.splitext(os.path.basename(path))[0]
    tip = fm.get("experiment_type") or fm.get("deney_tipi")
    if not tip:
        print(f"  ⚠️ no 'experiment_type' (or 'deney_tipi') in the front matter — skipped: {name}")
        return False
    if tip not in RUNNERS:
        import difflib
        near = difflib.get_close_matches(tip, list(RUNNERS), n=2)
        print(f"  ⚠️ unknown experiment type '{tip}'" + (f" (did you mean {', '.join(near)}?)" if near else "") + f" — skipped: {name}")
        return False
    print(f"  ▶ running: {name}  (type={tip})")
    figpath, body, summary = RUNNERS[tip](fm, slugify(name))
    title = write_report(name, fm, figpath, body, summary)
    mark_done(path, title, english=_is_english(fm))
    print(f"    ✅ {summary}\n    report -> {title}")
    return True


def _is_pending(fm):
    s = " ".join(str(fm.get(k, "")) for k in ("status", "durum", "tags", "etiketler")).lower()
    return "pending" in s or "bekliyor" in s


def main(argv):
    from .cli import utf8_stdio
    utf8_stdio()
    args = argv[1:]
    if args and args[0] != "--all":
        run_file(args[0]); return
    # --all: every note in the experiments folder whose status is pending
    os.makedirs(DENEYLER, exist_ok=True)
    files = sorted(os.path.join(DENEYLER, f) for f in os.listdir(DENEYLER) if f.endswith(".md"))
    pending = [p for p in files if _is_pending(parse_frontmatter(open(p, encoding="utf-8").read())[0])]
    if not pending:
        print(f"No pending experiments in {DENEYLER}\n(add a note with 'status: pending' — or 'durum: 🔬 bekliyor' — in its front matter)."); return
    print(f"{len(pending)} pending experiment(s) found.")
    for p in pending:
        try:
            run_file(p)
        except Exception as e:
            print(f"  ❌ error ({os.path.basename(p)}): {type(e).__name__}: {e}")


if __name__ == "__main__":
    main(sys.argv)


