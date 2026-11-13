"""Example 19 - LPA test in 2D: metagrating supercells, RCWA vs Tidy3D FDTD (roadmap P3).

N different pillars (phases 2 pi k / N from the recipe library, robust rule) in an N x 1
supercell deflect normal incidence into one order at sin(theta) = lam0 / (N L) - the local
structure of a lens at that NA. For each N three numbers are compared:

  * LPA   : periodic-library t of each pillar, cell-wise constant field (rcwa.lpa)
  * RCWA  : full supercell solve (memory-limited orders; reported as is)
  * FDTD  : Tidy3D supercell, power per diffraction order (DiffractionMonitor) scaled by the
            total transmission (FluxMonitor) - the reference

Usage:  python examples/19_metagrating_lpa_fdtd.py [--N 4 6 8] [--run | --load]
"""
import sys, os, json, argparse, datetime
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import design, materials, lpa
from rcwa.cli import utf8_stdio
utf8_stdio()

ap = argparse.ArgumentParser()
ap.add_argument("--material", default="TiO2_Siefke"); ap.add_argument("--lam0", type=float, default=0.633)
ap.add_argument("--H", type=float, default=0.60); ap.add_argument("--L", type=float, default=0.375)
ap.add_argument("--substrate", default="SiO2_Malitson")
ap.add_argument("--N", type=int, nargs="*", default=[4, 6, 8])
ap.add_argument("--m-per-cell", type=int, default=4); ap.add_argument("--no-rcwa", action="store_true")
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--run", action="store_true"); ap.add_argument("--load", action="store_true")
a = ap.parse_args()
LAM0, L, H = a.lam0, a.L, a.H
n_sub = float(np.sqrt(materials.eps_of(a.substrate, LAM0)).real)
lib = design.refine_library(design.meta_atom_library(LAM0, L, H, sizes=np.linspace(0.06, L - 0.06, 16),
                                                     material=a.material, n_sub=n_sub))
rows = {}
for N in a.N:
    idx = lpa.metagrating_cells(lib, N)
    lp = lpa.lpa_deflection(lib, idx)
    rc = None if a.no_rcwa else lpa.supercell_deflection(lib, idx, M_per_cell=a.m_per_cell)
    rows[N] = {"sizes_um": [float(lib["sizes"][i]) for i in idx], "sin_theta": LAM0 / (N * L),
               "lpa_cellwise": lp["cellwise"], "lpa_ideal": lp["ideal_sampling"], "rcwa": rc}
    print(f"N={N} (sin theta {LAM0/(N*L):.3f}): sizes {[round(s*1000) for s in rows[N]['sizes_um']]} nm  "
          f"LPA {lp['cellwise']:.3f} (ideal sampling {lp['ideal_sampling']:.3f})"
          + (f"  RCWA(M/cell={a.m_per_cell}) {rc['T_design']:.3f} [other {rc['T_other']:.3f}, 0th {rc['T0']:.3f}, R {rc['R_total']:.3f}]" if rc else ""), flush=True)

try:
    import tidy3d as td
    from tidy3d import web
except ImportError:
    sys.exit("tidy3d not installed")
freq0 = td.C_0 / LAM0; n_p = complex(lib["n"])
dpml, sub_in, air_out = 0.8, 0.8, 1.2
sz = dpml + sub_in + H + air_out + dpml
z_base = -sz / 2 + dpml + sub_in; z_p = z_base + H / 2; z_mon = z_base + H + 0.6
pm = td.Medium.from_nk(n=n_p.real, k=max(n_p.imag, 0.0), freq=freq0)
sub = td.Structure(geometry=td.Box(center=(0, 0, z_base - sz), size=(td.inf, td.inf, 2 * sz)),
                   medium=td.Medium(permittivity=n_sub ** 2))


def make_sim(N, sizes):
    Lx = N * L
    pillars = [td.Structure(geometry=td.Cylinder(center=(-Lx / 2 + (k + 0.5) * L, 0, z_p), radius=s / 2, length=H, axis=2),
                            medium=pm) for k, s in enumerate(sizes)]
    return td.Simulation(
        size=(Lx, L, sz), grid_spec=td.GridSpec.auto(min_steps_per_wvl=a.steps), structures=[sub] + pillars,
        sources=[td.PlaneWave(center=(0, 0, -sz / 2 + dpml + 0.2), size=(td.inf, td.inf, 0),
                              source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 8), direction="+",
                              pol_angle=np.pi / 2)],
        monitors=[td.DiffractionMonitor(center=(0, 0, z_mon), size=(td.inf, td.inf, 0), freqs=[freq0], name="diff"),
                  td.FluxMonitor(center=(0, 0, z_mon), size=(td.inf, td.inf, 0), freqs=[freq0], name="flux")],
        run_time=2e-12,
        boundary_spec=td.BoundarySpec(x=td.Boundary.periodic(), y=td.Boundary.periodic(), z=td.Boundary.pml()))


sims = {f"N{N}": make_sim(N, rows[N]["sizes_um"]) for N in a.N}
PATH_DIR = os.path.join("workspace", "tidy3d_metagrating")
if a.load:
    batch = web.Batch.from_file(os.path.join(PATH_DIR, "batch.hdf5")); data = batch.load(path_dir=PATH_DIR); cost = float("nan")
else:
    batch = web.Batch(simulations=sims, folder_name="rcwa_metagrating")
    cost = batch.estimate_cost(); print(f"\n{len(sims)} simulations. *** Estimated cost: {cost:.3f} FlexCredit ***")
    if not a.run:
        sys.exit("Estimate only. Add --run to spend credits.")
    data = batch.run(path_dir=PATH_DIR)

print("\nRESULT (power fraction of the incident light in the design order)")
for N in a.N:
    d = data[f"N{N}"]
    P = d["diff"].power.isel(f=0)            # keep orders_y (length 1): squeeze() would drop it
    tot = float(np.real(np.asarray(d["flux"].flux.values).ravel()[0]))
    frac = P / float(P.sum())

    def order(m):
        try:
            return float(frac.sel(orders_x=m, orders_y=0)) * tot
        except Exception:
            return float("nan")
    fd = {"T_plus1": order(1), "T_minus1": order(-1), "T0": order(0), "T_total": tot,
          "decay": float(getattr(d, "final_decay_value", np.nan))}
    rows[N]["fdtd"] = fd
    rc = rows[N]["rcwa"]
    print(f"  N={N} sin={rows[N]['sin_theta']:.3f}:  LPA {rows[N]['lpa_cellwise']:.3f}  "
          + (f"RCWA {rc['T_design']:.3f}  " if rc else "")
          + f"FDTD order+1 {fd['T_plus1']:.3f} (order-1 {fd['T_minus1']:.3f}, 0th {fd['T0']:.3f}, T {tot:.3f}, decay {fd['decay']:.1e})")
os.makedirs("validation", exist_ok=True)
out = os.path.join("validation", f"P3_metagrating_{a.material}_{LAM0*1000:.0f}_{H*1000:.0f}_{L*1000:.0f}_{datetime.date.today().isoformat()}.json")
json.dump({"material": a.material, "lam0": LAM0, "H": H, "L": L, "n_sub": n_sub, "m_per_cell": a.m_per_cell,
           "fdtd_steps": a.steps, "estimated_cost_fc": float(cost), "tidy3d_version": td.__version__,
           "rows": {str(k): v for k, v in rows.items()}}, open(out, "w"), indent=1)
print(f"Record written: {out}")