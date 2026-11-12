"""Example 3 — Metalens meta-atom phase library.
Sweeps the width of a nano-pillar and computes the 0th-order transmission phase
and amplitude. A good library: the phase should cover 0-2π and the amplitude stay high.
The output plot is saved to examples/output/.
Run:  python examples/03_metalens_library.py
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import solve_rcwa_1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    lam0 = 0.633            # design wavelength (um)
    period = 0.30           # subwavelength unit cell -> 0th order only
    H = 0.60                # pillar height (um)
    n_pillar = 2.4          # TiO2-like high index
    P = 21
    Nx = 512
    widths = np.linspace(0.03, 0.29, 40)   # pillar width sweep (um)

    phase, amp = [], []
    for w in widths:
        x = (np.arange(Nx) + 0.5) / Nx * period
        eps = np.where(np.abs(x - period / 2) < w / 2, n_pillar ** 2, 1.0).astype(complex)
        r = solve_rcwa_1d(lam0, 0.0, 1.0, 1.0, period, [(eps, H)], P, pol=(0., 1.))
        t0 = r['ty'][r['zeroth_index']]     # TE 0th-order transmission amplitude
        phase.append(np.angle(t0)); amp.append(np.abs(t0))

    phase = np.unwrap(np.array(phase)); phase -= phase[0]
    amp = np.array(amp)
    print(f"Phase range: {np.ptp(phase)/np.pi:.2f}π  |  Mean amplitude: {amp.mean():.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(widths * 1000, phase / np.pi, 'o-'); ax[0].set_xlabel("Pillar width (nm)")
    ax[0].set_ylabel("Transmission phase (π)"); ax[0].set_title("Phase library"); ax[0].grid(True, alpha=.3)
    ax[1].plot(widths * 1000, amp, 's-', color='C1'); ax[1].set_xlabel("Pillar width (nm)")
    ax[1].set_ylabel("|t0|"); ax[1].set_title("Transmission amplitude"); ax[1].set_ylim(0, 1.05); ax[1].grid(True, alpha=.3)
    fig.suptitle(f"Meta-atom sweep  (λ={lam0}um, Λ={period}um, H={H}um, n={n_pillar})")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "meta_atom_phase_library.png")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out, dpi=130)
    print("Plot saved ->", out)
