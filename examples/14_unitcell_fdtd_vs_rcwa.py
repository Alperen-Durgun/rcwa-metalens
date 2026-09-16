"""Örnek 14 — Birim-hücre: RCWA vs Tidy3D FDTD (UCUZ doğrulama).

Tek meta-atomun 0. mertebe geçişini RCWA ve FDTD ile karşılaştırır. Subwavelength
periyot -> yalnız 0. mertebe -> T karşılaştırması temiz. Maliyet ~kuruşluk kredi.
Bu, tüm tasarımın dayandığı meta-atom kütüphanesini doğrular.

Kullanım:
  python examples/14_unitcell_fdtd_vs_rcwa.py            # RCWA + Tidy3D maliyet TAHMİNİ
  python examples/14_unitcell_fdtd_vs_rcwa.py run        # FDTD'yi çalıştır (kredi harcar!)
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import solve_rcwa_2d, shapes, materials

lam0 = 0.633; L = 0.35; H = 0.60; size = 0.20; mat = "TiO2_t"
n = float(np.real(np.sqrt(materials.eps_of(mat, lam0))))

# --- RCWA (yerel, ücretsiz) ---
eps = shapes.rasterize("circle", size, L, 64, n_pillar=n)
r = solve_rcwa_2d(lam0, 0, 0, 1, 1, L, L, [(eps, H)], 8, 8, pol=(0., 1.))
T_rcwa = float(r["Ttot"]); t0 = r["ty"][r["i0"]]
print(f"RCWA:  T={T_rcwa:.4f}  faz(0.mertebe)={np.angle(t0, deg=True):.1f}°  (n_{mat}={n:.2f})")

# --- Tidy3D birim hücre (periyodik) ---
import tidy3d as td
freq0 = td.C_0 / lam0
dpml = 0.8; airin = 0.8; airout = 1.2
sz = dpml + airin + H + airout + dpml
z_lens = -sz/2 + dpml + airin + H/2
pillar = td.Structure(geometry=td.Cylinder(center=(0, 0, z_lens), radius=size/2, length=H, axis=2),
                      medium=td.Medium(permittivity=n**2))
src = td.PlaneWave(center=(0, 0, -sz/2 + dpml + 0.2), size=(td.inf, td.inf, 0),
                   source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0/8), direction="+")
flux = td.FluxMonitor(center=(0, 0, z_lens + H/2 + 0.5), size=(td.inf, td.inf, 0),
                      freqs=[freq0], name="T")
sim = td.Simulation(
    size=(L, L, sz), grid_spec=td.GridSpec.auto(min_steps_per_wvl=20),
    structures=[pillar], sources=[src], monitors=[flux], run_time=1.2e-12,
    boundary_spec=td.BoundarySpec(x=td.Boundary.periodic(), y=td.Boundary.periodic(),
                                  z=td.Boundary.pml()))
print("Tidy3D birim-hücre Simulation kuruldu.")

from tidy3d import web
job = web.Job(simulation=sim, task_name="unitcell_verify")
cost = job.estimate_cost()   # ücretsiz: kredi harcamaz
print(f"Tahmini maliyet: {cost:.3f} FlexCredit")
if len(sys.argv) > 1 and sys.argv[1] == "run":
    data = job.run()          # kredi HARCAR
    T_fdtd = float(np.abs(np.asarray(data["T"].flux.values).ravel()[0]))
    print(f"\nSONUÇ:  RCWA T={T_rcwa:.4f}  |  FDTD T={T_fdtd:.4f}  |  fark={abs(T_rcwa-T_fdtd):.4f}")
else:
    print("Çalıştırmak (kredi harcar): python examples/14_unitcell_fdtd_vs_rcwa.py run")
