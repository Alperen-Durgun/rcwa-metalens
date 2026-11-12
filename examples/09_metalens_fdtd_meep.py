"""Example 9 — FULL-WAVE metalens with FDTD (Meep) — reference for the REAL finite structure.

Why this file: a thick + high-index metalens super-cell can be ill-conditioned for RCWA
(see example 08). FDTD solves the finite structure directly and stably, and is the
community's gold standard for full-wave metalens verification.

STATUS: this is a TEMPLATE. Meep cannot be installed with pip (conda is required), so
it is not covered by the automated tests. Run it on your own machine/HPC:

    conda create -n mp -c conda-forge pymeep
    conda activate mp
    python examples/09_metalens_fdtd_meep.py

Reference: Oskooi et al. 2010, Comput. Phys. Commun. 181, 687 (Meep).
For large 3D lenses GPU-RCWA (TORCWA) is also an option.

This scenario: a 2D-FDTD cross-section of a 1D (cylindrical) metalens — comparable with
the RCWA result (08): use the same D_lens, f, Lcell, H, n_pillar.
"""
try:
    import meep as mp
except ImportError:
    raise SystemExit("Meep is not installed. Install:  conda install -c conda-forge pymeep")
import numpy as np

# ---- lens parameters (keep identical to 08) ----
lam0 = 0.633; Lcell = 0.35; H = 0.60; n_pillar = 2.4
D_lens = 6.0; f = 6.0
res = 40                      # pixels/um (increase for convergence)
dpml = 1.0; gap_in = 1.5; gap_out = 2.2*f + 2.0
fcen = 1.0/lam0

Ncell = int(round(D_lens/Lcell)); D_lens = Ncell*Lcell
sx = D_lens + 4.0            # transverse window (+ margin)
sy = dpml + gap_in + H + gap_out + dpml
cell = mp.Vector3(sx, sy)
pml = [mp.PML(dpml)]

# ---- meta-atom phase library ----
# The proper way: sweep a single (periodic) cell over each width with RCWA/FDTD and
# extract phase(w). For simplicity we import the rcwa library here and pick the
# widths with it:
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import solve_rcwa_1d
widths = np.linspace(0.04, Lcell-0.04, 60); ph = []
for w in widths:
    xg = (np.arange(256)+0.5)/256*Lcell
    eps = np.where(np.abs(xg-Lcell/2) < w/2, n_pillar**2, 1.0).astype(complex)
    r = solve_rcwa_1d(lam0, 0, 1, 1, Lcell, [(eps, H)], 21, pol=(0., 1.))
    ph.append(np.angle(r['ty'][r['zeroth_index']]))
ph = np.unwrap(np.array(ph)); ph -= ph.min()
phm = np.mod(-ph, 2*np.pi)   # solver phase is exp(+jwt); the lens profile is exp(-iwt) -> negate (rcwa/convention.py)
xc = (np.arange(Ncell)+0.5)*Lcell - D_lens/2
phi = np.mod(-(2*np.pi/lam0)*(np.sqrt(xc**2+f**2)-f), 2*np.pi)
chosen = np.array([widths[np.argmin(np.abs(np.angle(np.exp(1j*(phm-p)))))] for p in phi])

# ---- geometry: nano-pillars ----
y_lens = -sy/2 + dpml + gap_in + H/2
geom = []
for i in range(Ncell):
    cx = xc[i]
    geom.append(mp.Block(size=mp.Vector3(chosen[i], H, mp.inf),
                         center=mp.Vector3(cx, y_lens),
                         material=mp.Medium(index=n_pillar)))

# ---- source: plane wave (Ez), just above the lens ----
y_src = -sy/2 + dpml + 0.3
sources = [mp.Source(mp.ContinuousSource(frequency=fcen), component=mp.Ez,
                     center=mp.Vector3(0, y_src), size=mp.Vector3(sx, 0))]

sim = mp.Simulation(cell_size=cell, boundary_layers=pml, geometry=geom,
                    sources=sources, resolution=res, force_complex_fields=True)
sim.run(until=200)

# ---- grab the field, find the focus on the axis ----
ez = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
I = np.abs(ez)**2
nx, ny = I.shape
yv = np.linspace(-sy/2, sy/2, ny)
y0 = -sy/2 + dpml + gap_in + H     # lens exit surface
axis = I[nx//2, :]
mask = yv > y0
zf = yv[mask][np.argmax(axis[mask])] - y0
print(f"FDTD: actual focus (from the lens surface) z ≈ {zf:.2f} um  (design f={f} um)")
print("Compare with the RCWA super-cell (08) for the same D_lens,f.")
# Optionally save the I map with matplotlib.
"""

Notes / good practice:
- `res` (resolution) is the convergence parameter; try 30-60 and increase until the focus/FWHM settle.
- The PML should be at least ~1 wavelength thick (reflection-free open boundary).
- Memory/time: a 2D-FDTD cross-section runs on a laptop; full 3D (a real 2D-periodic lens)
  needs disk/GPU -> Meep MPI or GPU-RCWA (TORCWA).
- Compare this FDTD result with the RCWA super-cell focus from 08: their agreement
  is the full-wave verification of the metalens design.
"""
