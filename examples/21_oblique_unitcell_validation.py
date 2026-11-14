"""Example 21 - FDTD validation of the angle-dependent library (roadmap P2): unit cells at
oblique incidence.

Same structure as example 18 (light from the substrate, bare-interface reference) at field
angles theta (in air before the substrate; Tidy3D's source angle is the substrate angle
asin(sin(theta) / n_sub), plane of incidence x-z, TE = E along y). The cell is Bloch-periodic
in x; the plane-wave angle is exact at the analysis frequency.

At oblique incidence the zeroth order of the transmitted field varies as exp(+-i kx x) over
the cell, so t is the zeroth Fourier coefficient: <E_y exp(-+i kx x)>_cell (pillar) divided by
the same for the bare interface. Which sign is the physical one depends on the solver's
time convention; both projections are computed and the dominant one (the other is ~0) is
used and recorded. RCWA side: rcwa.angular at M = 11, divided by its own bare interface.

Acceptance as in V1: |d|t|| < 0.02 and |d phase| < 0.1 rad on every size, per angle; a
constant phase per angle is fitted and reported (expected ~0).

Usage (inside the .venv, repository root):
  python examples/21_oblique_unitcell_validation.py                 # estimate only
  python examples/21_oblique_unitcell_validation.py --run           # SPENDS credits
  python examples/21_oblique_unitcell_validation.py --load          # re-analyse a finished batch
Record: validation/V1a_<material>_<lam>_<H>_<L>_<date>.json
"""
import sys, os, json, argparse, datetime
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import design, materials, angular
from rcwa.convention import to_physical
from rcwa.cli import utf8_stdio
utf8_stdio()

ap = argparse.ArgumentParser()
ap.add_argument("--material", default="TiO2_Siefke"); ap.add_argument("--lam0", type=float, default=0.633)
ap.add_argument("--H", type=float, default=0.60); ap.add_argument("--L", type=float, default=0.375)
ap.add_argument("--substrate", default="SiO2_Malitson")
ap.add_argument("--angles", type=float, nargs="+", default=[10.0, 20.0])
ap.add_argument("--sizes", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.25, 0.29])
ap.add_argument("--M", type=int, default=11); ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--run-time", type=float, default=2.0e-12)
ap.add_argument("--run", action="store_true"); ap.add_argument("--load", action="store_true")
ap.add_argument("--tag", default="")
a = ap.parse_args()
LAM0, L, H, MAT = a.lam0, a.L, a.H, a.material
n_sub = float(np.sqrt(materials.eps_of(a.substrate, LAM0)).real)
n_p = design.pillar_index(MAT, LAM0)
TOL_AMP, TOL_PHASE = 0.02, 0.1
N = 512

# ----------------------------------------------------------------------------- RCWA (free)
t_rcwa = {}
for th in a.angles:
    ts = angular.substrate_angle(th, n_sub)
    t_bare = to_physical(angular._atom_oblique((LAM0, L, H, "circle", 0.1, N, 1.0 + 0j, 1.0, n_sub, a.M,
                                                (90.0, 1, None), ts, "TE"))[0])
    t_p = np.array([to_physical(angular._atom_oblique((LAM0, L, H, "circle", s, N, n_p, 1.0, n_sub, a.M,
                                                       (90.0, 1, None), ts, "TE"))[0]) for s in a.sizes])
    t_rcwa[th] = t_p / t_bare
    o = angular.orders_at(th, LAM0, L, n_sub)
    print(f"theta {th:5.1f} deg (substrate {ts:.2f} deg), extra orders: substrate {o['substrate']}, air {o['exit']}")
    for s, t in zip(a.sizes, t_rcwa[th]):
        print(f"   {s*1000:6.1f} nm  RCWA M={a.M}: |t|={abs(t):.4f} phase={np.angle(t):+.4f}")

try:
    import tidy3d as td
    from tidy3d import web
except ImportError:
    sys.exit("\ntidy3d is not installed in this environment: pip install tidy3d")

freq0 = td.C_0 / LAM0
dpml, sub_in, air_out = 0.8, 0.8, 1.2
sz = dpml + sub_in + H + air_out + dpml
z_base = -sz / 2 + dpml + sub_in
z_p = z_base + H / 2; z_mon = z_base + H + 0.5
pillar_medium = td.Medium.from_nk(n=n_p.real, k=max(n_p.imag, 0.0), freq=freq0)
sub_medium = td.Medium(permittivity=n_sub ** 2)
substrate = td.Structure(geometry=td.Box(center=(0, 0, z_base - sz), size=(td.inf, td.inf, 2 * sz)),
                         medium=sub_medium)
monitor = td.FieldMonitor(center=(0, 0, z_mon), size=(td.inf, td.inf, 0), freqs=[freq0], name="field")


def make_sim(size, th):
    ts = np.deg2rad(angular.substrate_angle(th, n_sub))
    src = td.PlaneWave(center=(0, 0, -sz / 2 + dpml + 0.2), size=(td.inf, td.inf, 0),
                       source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 8), direction="+",
                       angle_theta=float(ts), angle_phi=0.0, pol_angle=np.pi / 2)        # s = E along y
    dom = (L, L, sz)
    # the source sits in the substrate: the Bloch vector must use its index (kx = k0 n_sub sin(theta_sub))
    bounds = td.BoundarySpec(x=td.Boundary.bloch_from_source(source=src, domain_size=L, axis=0, medium=sub_medium),
                             y=td.Boundary.periodic(), z=td.Boundary.pml())
    st = [substrate] + ([] if size is None else [td.Structure(
        geometry=td.Cylinder(center=(0, 0, z_p), radius=size / 2, length=H, axis=2), medium=pillar_medium)])
    return td.Simulation(size=dom, grid_spec=td.GridSpec.auto(min_steps_per_wvl=a.steps), structures=st,
                         sources=[src], monitors=[monitor], run_time=a.run_time, boundary_spec=bounds)


sims = {}
for th in a.angles:
    sims[f"a{th:g}_bare"] = make_sim(None, th)
    for s in a.sizes:
        sims[f"a{th:g}_s{s*1000:.1f}"] = make_sim(s, th)
PATH_DIR = os.path.join("workspace", "tidy3d_oblique" + (f"_{a.tag}" if a.tag else ""))
if a.load:
    batch = web.Batch.from_file(os.path.join(PATH_DIR, "batch.hdf5"))
    data = batch.load(path_dir=PATH_DIR); cost = float("nan")
else:
    batch = web.Batch(simulations=sims, folder_name="rcwa_oblique_unitcell" + (f"_{a.tag}" if a.tag else ""))
    cost = batch.estimate_cost()
    print(f"\n{len(sims)} simulations. *** Estimated cost: {cost:.3f} FlexCredit ***")
    if not a.run:
        sys.exit("Estimate only. Add --run to spend credits.")
    data = batch.run(path_dir=PATH_DIR)

uni = -L / 2 + (np.arange(64) + 0.5) * L / 64


def zeroth(d, th):
    """Zeroth Fourier coefficient of E_y for both signs of exp(i kx x): (c_plus, c_minus)."""
    v = np.asarray(d["field"].Ey.isel(f=0).squeeze().interp(x=uni, y=uni, kwargs={"fill_value": "extrapolate"}).values)
    if not np.all(np.isfinite(v)):
        raise RuntimeError("non-finite field after interpolation")
    kx = 2 * np.pi / LAM0 * np.sin(np.deg2rad(th))
    ph = np.exp(1j * kx * uni)[:, None]
    return complex(np.mean(v * np.conj(ph))), complex(np.mean(v * ph))


results = {}; ok_all = True
for th in a.angles:
    rp, rm = zeroth(data[f"a{th:g}_bare"], th)
    sign = +1 if abs(rp) >= abs(rm) else -1
    ref = rp if sign > 0 else rm
    tf = np.array([zeroth(data[f"a{th:g}_s{s*1000:.1f}"], th)[0 if sign > 0 else 1] / ref for s in a.sizes])
    tr = t_rcwa[th]
    offset = np.angle(np.sum(tr * np.conj(tf)))
    dphi = np.angle(tr * np.conj(tf) * np.exp(-1j * offset)); damp = np.abs(tr) - np.abs(tf)
    ok = bool(np.all(np.abs(damp) < TOL_AMP) and np.all(np.abs(dphi) < TOL_PHASE)); ok_all &= ok
    ratio = min(abs(rp), abs(rm)) / max(abs(rp), abs(rm))
    print(f"\ntheta {th:g} deg: projection sign {sign:+d} (other projection / dominant = {ratio:.1e}); "
          f"fitted constant phase {offset:+.4f} rad -> {'PASSED' if ok else 'FAILED'}")
    for i, s in enumerate(a.sizes):
        print(f"  {s*1000:6.1f} nm  |t| RCWA {abs(tr[i]):.4f} FDTD {abs(tf[i]):.4f}  d|t| {damp[i]:+.4f}  dphase {dphi[i]:+.4f}")
    results[f"{th:g}"] = {"theta_deg": th, "theta_sub_deg": angular.substrate_angle(th, n_sub), "passed": ok,
                          "projection_sign": sign, "other_projection_ratio": ratio, "phase_offset_rad": float(offset),
                          "t_rcwa": [[z.real, z.imag] for z in tr], "t_fdtd": [[z.real, z.imag] for z in tf],
                          "d_amp": damp.tolist(), "d_phase_rad": dphi.tolist(),
                          "extra_orders": angular.orders_at(th, LAM0, L, n_sub)}
decay = {k: getattr(data[k], "final_decay_value", None) for k in sims}
print("final field decay per task: " + ", ".join(f"{k}={v:.1e}" for k, v in decay.items() if v is not None))
print(f"\nOBLIQUE UNIT-CELL VALIDATION {'PASSED' if ok_all else 'FAILED'}")
os.makedirs("validation", exist_ok=True)
out = os.path.join("validation", f"V1a_{MAT}_{LAM0*1000:.0f}_{H*1000:.0f}_{L*1000:.0f}"
                   f"{'_' + a.tag if a.tag else ''}_{datetime.date.today().isoformat()}.json")
json.dump({"passed": ok_all, "kind": "oblique unit cell (TE)", "material": MAT, "lam0": LAM0, "H": H, "L": L,
           "n_sub": n_sub, "M": a.M, "fdtd_steps": a.steps, "sizes_um": list(a.sizes), "angles": results,
           "estimated_cost_fc": float(cost), "run_time_s": a.run_time,
           "final_decay": {k: (float(v) if v is not None else None) for k, v in decay.items()},
           "tidy3d_version": td.__version__}, open(out, "w"), indent=1)
print(f"Record written: {out}")
