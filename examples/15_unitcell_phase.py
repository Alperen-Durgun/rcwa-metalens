"""Example 15 — Unit-cell PHASE verification: RCWA vs Tidy3D FDTD.

Compares the 0th-order transmission COEFFICIENT (|t| and phase). FDTD needs two runs for the phase:
  (1) with the structure, (2) without it (air reference). The ratio Ey_struct/Ey_ref = t
(transmission coefficient), same convention as the RCWA ty phase.
Cost ~0.05 FlexCredit (two small runs).

Usage:
  python examples/15_unitcell_phase.py            # RCWA + cost estimate
  python examples/15_unitcell_phase.py run        # FDTD (spends credits)
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import solve_rcwa_2d, shapes, materials

lam0 = 0.633; L = 0.35; H = 0.60; size = 0.20; mat = "TiO2_t"
n = float(np.real(np.sqrt(materials.eps_of(mat, lam0))))

# --- RCWA ---
eps = shapes.rasterize("circle", size, L, shapes.DEFAULT_N, n_pillar=n)
r = solve_rcwa_2d(lam0, 0, 0, 1, 1, L, L, [(eps, H)], 8, 8, pol=(0., 1.))
t0 = r["ty"][r["i0"]]
print(f"RCWA:  |t|={abs(t0):.4f}  phase={np.angle(t0, deg=True):.1f}°")

import tidy3d as td
freq0 = td.C_0 / lam0
dpml = 0.8; airin = 0.8; airout = 1.2
sz = dpml + airin + H + airout + dpml
z_lens = -sz/2 + dpml + airin + H/2
z_mon = z_lens + H/2 + 0.5
src = td.PlaneWave(center=(0, 0, -sz/2 + dpml + 0.2), size=(td.inf, td.inf, 0),
                   source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0/8), direction="+")
fmon = td.FieldMonitor(center=(0, 0, z_mon), size=(td.inf, td.inf, 0), freqs=[freq0], name="field")
bs = td.BoundarySpec(x=td.Boundary.periodic(), y=td.Boundary.periodic(), z=td.Boundary.pml())
common = dict(size=(L, L, sz), grid_spec=td.GridSpec.auto(min_steps_per_wvl=20),
             sources=[src], monitors=[fmon], run_time=1.2e-12, boundary_spec=bs)
pillar = td.Structure(geometry=td.Cylinder(center=(0, 0, z_lens), radius=size/2, length=H, axis=2),
                      medium=td.Medium(permittivity=n**2))
sim_with = td.Simulation(structures=[pillar], **common)
sim_ref = td.Simulation(structures=[], **common)
print("Two Simulations (structure + reference) built.")

def center_field(data):
    def g(c): return complex(np.asarray(getattr(data["field"], c).sel(x=0, y=0, method="nearest").values).ravel()[0])
    return g("Ex"), g("Ey")

def dom(data):
    fx, fy = center_field(data); return fx if abs(fx) >= abs(fy) else fy

def sim_for(sz_pillar):
    p = td.Structure(geometry=td.Cylinder(center=(0, 0, z_lens), radius=sz_pillar/2, length=H, axis=2),
                     medium=td.Medium(permittivity=n**2))
    return td.Simulation(structures=[p], **common)

if len(sys.argv) > 1 and sys.argv[1] == "run":
    from tidy3d import web
    sizeB = 0.28
    epsB = shapes.rasterize("circle", sizeB, L, shapes.DEFAULT_N, n_pillar=n)
    rB = solve_rcwa_2d(lam0, 0, 0, 1, 1, L, L, [(epsB, H)], 8, 8, pol=(0., 1.)); t0B = rB["ty"][rB["i0"]]
    dA = web.Job(simulation=sim_with, task_name="phase_A").run()          # size=0.20 (cached)
    dB = web.Job(simulation=sim_for(sizeB), task_name="phase_B").run()    # size=0.28
    dR = web.Job(simulation=sim_ref, task_name="phase_ref").run()         # for |t| (cached)
    fr = dom(dR); tA = dom(dA)/fr; tB = dom(dB)/fr
    dphi_rcwa = (np.angle(t0) - np.angle(t0B)) * 180/np.pi
    dphi_fdtd = (np.angle(tA) - np.angle(tB)) * 180/np.pi
    def wrap(a): return (a + 180) % 360 - 180
    rel_same = abs(wrap(dphi_rcwa - dphi_fdtd))         # same convention
    rel_conj = abs(wrap(dphi_rcwa + dphi_fdtd))         # opposite convention (φ->-φ)
    rel = min(rel_same, rel_conj); conv = "same" if rel_same <= rel_conj else "opposite (e^-iwt vs e^+iwt)"
    print("\nRESULT (two sizes: A=200nm, B=280nm):")
    print(f"  |t|   A: RCWA={abs(t0):.4f} FDTD={abs(tA):.4f} | B: RCWA={abs(t0B):.4f} FDTD={abs(tB):.4f}")
    print(f"  RELATIVE PHASE (φA-φB):  RCWA={dphi_rcwa:.1f}°   FDTD={dphi_fdtd:.1f}°")
    print(f"  Convention: {conv}  ->  CORRECTED |diff| = {rel:.1f}°")
    print("  -> If the relative phase difference is ~a few degrees, the library is FDTD-confirmed in magnitude+phase.")
else:
    from tidy3d import web
    c = web.Job(simulation=sim_with, task_name="phase_est").estimate_cost()
    print(f"Estimated cost (per run): {c:.3f} FlexCredit x2 runs")
    print("Run: python examples/15_unitcell_phase.py run")
