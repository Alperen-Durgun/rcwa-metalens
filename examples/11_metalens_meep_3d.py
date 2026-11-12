"""Example 11 — REAL 2D metalens, 3D-FDTD (Meep) FULL-WAVE verification TEMPLATE.

A real metalens is an array of pillars in x-y; verifying it full-wave requires 3D-FDTD.
This is the FINAL full-wave verification of `examples/10_metalens_layout_2d.py`
(the locally-periodic design): it simulates the very same pillar map in 3D.

STATUS: TEMPLATE. Meep requires conda, so this is not covered by the automated tests.
3D-FDTD is EXPENSIVE: a small lens (≈10-20 λ) needs a multi-core workstation, a large
lens needs HPC/GPU. Start small, then increase the resolution/lens size gradually.

Install:
    conda create -n mp -c conda-forge pymeep pymeep-extras
    conda activate mp
    python examples/11_metalens_meep_3d.py

Reference: Oskooi et al. 2010 (Meep). Alternative GPU-RCWA: TORCWA (Kim & Lee 2023).
"""
try:
    import meep as mp
except ImportError:
    raise SystemExit("Meep not found. Install: conda install -c conda-forge pymeep")
import numpy as np, os, sys

# ---- SAME lens parameters as 08/10 (for comparison) ----
lam0 = 0.633; Lcell = 0.35; H = 0.60; n_pillar = 2.4; n_sub = 1.5
D_lens = 6.0; f = 6.0            # keep it small (3D-FDTD is expensive!)
resolution = 20                 # pixels/um (increase for convergence: 30-40)
fcen = 1.0/lam0

# --- meta-atom library (rcwa, stable unit cell) ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import solve_rcwa_2d
sides = np.linspace(0.06, Lcell-0.05, 16); libph = []
for s in sides:
    N = 48; x = (np.arange(N)+0.5)/N*Lcell; X, Y = np.meshgrid(x, x, indexing='ij')
    eps = np.where((np.abs(X-Lcell/2) < s/2) & (np.abs(Y-Lcell/2) < s/2), n_pillar**2, 1.0)
    r = solve_rcwa_2d(lam0, 0, 0, 1, 1, Lcell, Lcell, [(eps, H)], 7, 7, pol=(0., 1.))
    libph.append(np.angle(r['ty'][r['i0']]))
libph = np.unwrap(np.array(libph)); libph -= libph.min(); libphm = np.mod(-libph, 2*np.pi)   # physical convention (rcwa/convention.py)

# --- pillar map over a circular aperture ---
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

# ---- Meep 3D scene ----
dpml = 1.0; air_in = 1.0; air_out = 1.5*f + 1.0
sx = D_lens + 3.0; sy = sx
sz = dpml + air_in + H + air_out + dpml
cell = mp.Vector3(sx, sy, sz)
z_lens = -sz/2 + dpml + air_in + H/2
geometry = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, dpml+air_out),  # lower half-space (air)
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

# ---- focus: |E|² in the xz cross-section below the lens ----
z0 = -sz/2 + dpml + air_in + H            # lens exit surface
ey = sim.get_array(center=mp.Vector3(0, 0, (z0 + sz/2 - dpml)/2),
                   size=mp.Vector3(sx, 0, (sz/2 - dpml) - z0), component=mp.Ey)
I = np.abs(ey)**2
nz = I.shape[-1]; zaxis = np.linspace(z0, sz/2 - dpml, nz) - z0
axis = I[I.shape[0]//2, :]
zf = zaxis[int(np.argmax(axis))]
print(f"Meep 3D-FDTD: actual focus z ≈ {zf:.2f} um (design f={f}). Compare: 10_metalens_layout_2d (locally-periodic).")
print("Note: increase the resolution to verify convergence. Use HPC/GPU for large lenses.")
