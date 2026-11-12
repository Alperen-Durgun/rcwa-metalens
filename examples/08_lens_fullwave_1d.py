"""Example 8 — FULL-WAVE metalens (1D/cylindrical), ROBUST version.

The whole lens is solved with Maxwell as a single RCWA super-cell (NO local
approximation), and the outgoing field is propagated EXACTLY to the focal plane.

IMPORTANT (honest caveat): a thick + high-index metalens super-cell is numerically
ILL-CONDITIONED for RCWA. At certain numbers of orders (M) the solution blows up —
this is not specific to this code; peer-reviewed grcwa blows up on the same structure
(guided-mode resonances). Remedy: try several M and keep those satisfying ENERGY
CONSERVATION (ΣDE≈1); all such valid solutions give the same focus (consistency =
confidence). For large/realistic lenses the full-wave reference is FDTD
-> 09_metalens_fdtd_meep.py.

Run:  python examples/08_lens_fullwave_1d.py
"""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import solve_rcwa_1d, metrics
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

lam0, Lcell, H, n_pillar = 0.633, 0.35, 0.60, 2.4
D_lens, f, guard = 6.0, 6.0, 3.0
M_list = [120, 130, 140, 160]
k0 = 2*np.pi/lam0
Ncell = int(round(D_lens/Lcell)); D_lens = Ncell*Lcell
D = D_lens + 2*guard; X0 = D/2
NA = np.sin(np.arctan(D_lens/2/f))
print(f"Lens D={D_lens:.2f}um f={f}um NA={NA:.2f} | super-cell {D:.1f}um, {Ncell} pillars")

# 1) phase library (single cell — stable)
widths = np.linspace(0.04, Lcell-0.04, 60); ph = []
for w in widths:
    x = (np.arange(256)+0.5)/256*Lcell
    eps = np.where(np.abs(x-Lcell/2) < w/2, n_pillar**2, 1.0).astype(complex)
    r = solve_rcwa_1d(lam0, 0, 1, 1, Lcell, [(eps, H)], 21, pol=(0., 1.))
    ph.append(np.angle(r['ty'][r['zeroth_index']]))
ph = np.unwrap(np.array(ph)); ph -= ph.min()
phm = np.mod(-ph, 2*np.pi)   # solver phase is exp(+jwt); the lens profile is exp(-iwt) -> negate (rcwa/convention.py)

# 2) design: target phase -> pillar width
xc = X0 + ((np.arange(Ncell)+0.5)*Lcell - D_lens/2)
phi = np.mod(-(2*np.pi/lam0)*(np.sqrt((xc-X0)**2 + f**2) - f), 2*np.pi)
chosen = np.array([widths[np.argmin(np.abs(np.angle(np.exp(1j*(phm-p)))))] for p in phi])

# 3) build the whole lens as a single super-cell
Nx = 4096; xg = (np.arange(Nx)+0.5)/Nx*D
col = np.ones(Nx); xs = X0 - D_lens/2
for i in range(Ncell):
    c = xs + (i+0.5)*Lcell
    col[np.abs(xg-c) < chosen[i]/2] = n_pillar**2

# 4) scan M, keep the energy-conserving (stable) ones
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

stable = []
for M in M_list:
    res = solve_rcwa_1d(lam0, 0, 1, 1, D, [(col.astype(complex), H)], 2*M+1, pol=(0., 1.))
    sig = res['Rtot'] + res['Ttot']
    field = propagate(res)
    zc = np.linspace(0.5*f, 1.6*f, 200)      # search around the design focus
    Iax = np.array([abs(field(z, np.array([X0]))[0])**2 for z in zc]); zf = zc[np.argmax(Iax)]
    ok = abs(sig-1) < 0.03
    print(f"  M={M}: ΣDE={sig:.4f} " + (f"✓ stable, focus z={zf:.2f}" if ok else "✗ blew up (skipped)"))
    if ok:
        stable.append((M, res, field, zf, sig))

if not stable:
    print("WARNING: no stable M found. Extend M_list or use FDTD (09)."); sys.exit(1)
zfs = [s[3] for s in stable]
print(f"\nStable solutions: {len(stable)} | focus consistency z = {np.mean(zfs):.2f} ± {np.std(zfs):.3f} um")

# 5) figure with the largest stable M
M, res, field, zf, sig = max(stable, key=lambda s: s[0])   # largest stable M = best converged
xo = np.linspace(X0-4, X0+4, 1401); If = abs(field(zf, xo))**2; If /= If.max()
fwhm = metrics.fwhm_1d(If, xo)   # main-lobe FWHM (interpolated)
print(f"Selected M={M} (ΣDE={sig:.4f}) | FOCUS z={zf:.2f}um (design {f}) | FWHM={fwhm*1000:.0f}nm | λ/(2NA)={lam0/(2*NA)*1000:.0f}nm")
zg = np.linspace(1, 2.2*f, 220); xm = np.linspace(X0-4, X0+4, 260)
Imap = np.array([abs(field(z, xm))**2 for z in zg])
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
ax[0].bar(xc-X0, chosen*1000, width=Lcell*0.9); ax[0].set_xlabel("Position (um)"); ax[0].set_ylabel("Pillar (nm)"); ax[0].set_title("Lens layout")
ax[1].imshow(Imap.T, origin="lower", aspect="auto", extent=[zg.min(), zg.max(), xm.min()-X0, xm.max()-X0], cmap="inferno")
ax[1].axvline(f, color="cyan", ls="--", lw=1); ax[1].axvline(zf, color="w", ls=":", lw=1)
ax[1].set_xlabel("z (um)"); ax[1].set_ylabel("x (um)"); ax[1].set_title("Full-wave I(x,z)")
ax[2].plot((xo-X0)*1000, If); ax[2].axhline(0.5, color="gray", ls=":"); ax[2].set_xlim(-2000, 2000)
ax[2].set_xlabel("x (nm)"); ax[2].set_ylabel("I"); ax[2].set_title(f"Focus — FWHM={fwhm*1000:.0f}nm")
fig.suptitle(f"FULL-WAVE metalens (1D) D={D_lens:.1f}um f={f}um NA={NA:.2f} | {len(stable)} stable M, focus z={zf:.2f}um")
fig.tight_layout()
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "Simulasyonlar", "lens_fullwave_1d.png"))
fig.savefig(out, dpi=130); print("Plot ->", out)
