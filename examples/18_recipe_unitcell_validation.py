"""Example 18 — FDTD validation of a unit-cell RECIPE (any material, wavelength, H, L, substrate).

Generalizes example 17 (V1) to the recipes found by the `recipe_search` generator:

  * light comes from the substrate (n_sub, e.g. fused silica) and exits into air, as in the
    recipe search; the reference is the bare substrate/air interface (no pillar), so the
    interface transmission cancels on both sides;
  * the test sizes are picked from the refined library: `--n` sizes spread over the phase
    circle, at least `--keep-away` nm from any resonance band, plus one size next to the
    strongest resonance (reported, not judged);
  * t = <E_y>(pillar) / <E_y>(bare) in a plane 0.5 um behind the pillar, cell-averaged
    (zeroth order); both sides in the physics convention. A constant phase is still fitted
    and reported - with the bare-interface reference it should be close to zero.

Acceptance as in V1: |d|t|| < 0.02 and |d phase| < 0.1 rad on the judged sizes.

Usage (inside the .venv, repository root):
  python examples/18_recipe_unitcell_validation.py --material TiO2_Siefke --lam0 0.633 --H 0.60 --L 0.375
  python examples/18_recipe_unitcell_validation.py ... --run       # SPENDS credits
Results of a run go to validation/V1r_<material>_<lam>_<H>_<L>_<date>.json.
"""
import sys, os, json, argparse, datetime
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import design, materials

ap = argparse.ArgumentParser()
ap.add_argument("--material", default="TiO2_Siefke"); ap.add_argument("--lam0", type=float, default=0.633)
ap.add_argument("--H", type=float, default=0.60); ap.add_argument("--L", type=float, default=0.375)
ap.add_argument("--substrate", default="SiO2_Malitson"); ap.add_argument("--shape", default="circle")
ap.add_argument("--s-min", type=float, default=0.06); ap.add_argument("--min-gap", type=float, default=0.06)
ap.add_argument("--n", type=int, default=5); ap.add_argument("--keep-away", type=float, default=8.0)
ap.add_argument("--M", type=int, default=11); ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--run", action="store_true")
a = ap.parse_args()
if a.shape != "circle":
    sys.exit("only circular pillars are set up for FDTD here")
LAM0, L, H, MAT = a.lam0, a.L, a.H, a.material
n_sub = float(np.sqrt(materials.eps_of(a.substrate, LAM0)).real) if a.substrate not in ("", "air") else 1.0
TOL_AMP, TOL_PHASE = 0.02, 0.1

# ----------------------------------------------------------------------------- RCWA (free)
base = design.meta_atom_library(LAM0, L, H, sizes=np.linspace(a.s_min, L - a.min_gap, 16),
                                material=MAT, n_sub=n_sub)
lib = design.refine_library(base)
res = lib["resonances"]
far = np.ones(len(lib["sizes"]), bool)
for r in res:
    far &= (lib["sizes"] < r["lo"] - a.keep_away * 1e-3) | (lib["sizes"] > r["hi"] + a.keep_away * 1e-3)
cand = np.nonzero(far)[0]
ph = np.mod(np.angle(lib["t"][cand]), 2 * np.pi)
targets = (np.arange(a.n) + 0.5) * 2 * np.pi / a.n
pick = sorted({int(cand[np.argmin(np.abs(np.angle(np.exp(1j * (ph - g)))))]) for g in targets})
sizes = [float(lib["sizes"][i]) for i in pick]
near = None
if res:
    r0 = min(res, key=lambda r: r["t_min"])
    near = round(r0["lo"] - 0.004, 4)                    # 4 nm below the band edge
judged = list(sizes)
all_sizes = sizes + ([near] if near else [])
t_hi = design.meta_atom_library(LAM0, L, H, sizes=all_sizes, material=MAT, n_sub=n_sub, M=a.M)["t"]
t_bare = design.meta_atom_library(LAM0, L, H, sizes=[0.1], n_pillar=1.0, n_sub=n_sub, M=a.M)["t"][0]
t_rcwa = t_hi / t_bare
n_p = complex(lib["n"])
print(f"Recipe: {MAT} (n={n_p.real:.3f}{n_p.imag:+.3f}i) {a.shape}, lam={LAM0} um, H={H} um, L={L} um, "
      f"substrate n={n_sub:.4f}; resonances: {[round(r['center']*1000, 1) for r in res]} nm")
for s, t in zip(all_sizes, t_rcwa):
    print(f"  {s*1000:6.1f} nm  RCWA (M={a.M}, relative to bare interface): |t|={abs(t):.4f} phase={np.angle(t):+.4f}"
          + ("" if s in judged else "   [near resonance, reported only]"))

try:
    import tidy3d as td
    from tidy3d import web
except ImportError:
    sys.exit("\ntidy3d is not installed in this environment: pip install tidy3d")

freq0 = td.C_0 / LAM0
dpml, sub_in, air_out = 0.8, 0.8, 1.2
sz = dpml + sub_in + H + air_out + dpml
z_base = -sz / 2 + dpml + sub_in                       # substrate top = pillar base
z_p = z_base + H / 2
z_mon = z_base + H + 0.5
pillar_medium = td.Medium.from_nk(n=n_p.real, k=max(n_p.imag, 0.0), freq=freq0)
substrate = td.Structure(geometry=td.Box(center=(0, 0, z_base - sz), size=(td.inf, td.inf, 2 * sz)),
                         medium=td.Medium(permittivity=n_sub ** 2))
source = td.PlaneWave(center=(0, 0, -sz / 2 + dpml + 0.2), size=(td.inf, td.inf, 0),
                      source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 8),
                      direction="+", pol_angle=np.pi / 2)
monitor = td.FieldMonitor(center=(0, 0, z_mon), size=(td.inf, td.inf, 0), freqs=[freq0], name="field")
bounds = td.BoundarySpec(x=td.Boundary.periodic(), y=td.Boundary.periodic(), z=td.Boundary.pml())


def make_sim(size, steps):
    st = [substrate] + ([] if size is None else [td.Structure(
        geometry=td.Cylinder(center=(0, 0, z_p), radius=size / 2, length=H, axis=2), medium=pillar_medium)])
    return td.Simulation(size=(L, L, sz), grid_spec=td.GridSpec.auto(min_steps_per_wvl=steps),
                         structures=st, sources=[source], monitors=[monitor], run_time=2.0e-12,
                         boundary_spec=bounds)


sims = {"bare": make_sim(None, a.steps)}
for s in all_sizes:
    sims[f"s_{s*1000:.1f}"] = make_sim(s, a.steps)
batch = web.Batch(simulations=sims, folder_name="rcwa_recipe_unitcell")
cost = batch.estimate_cost()
print(f"\n{len(sims)} simulations. *** Estimated cost: {cost:.3f} FlexCredit ***")
if not a.run:
    sys.exit("Estimate only. Add --run to spend credits.")

data = batch.run(path_dir=os.path.join("workspace", "tidy3d_recipe"))
uni = -L / 2 + (np.arange(64) + 0.5) * L / 64


def mean_Ey(d):
    v = np.asarray(d["field"].Ey.isel(f=0).squeeze().interp(x=uni, y=uni, kwargs={"fill_value": "extrapolate"}).values)
    if not np.all(np.isfinite(v)):
        raise RuntimeError("non-finite field after interpolation")
    return complex(v.mean())


ref = mean_Ey(data["bare"])
t_fdtd = np.array([mean_Ey(data[f"s_{s*1000:.1f}"]) / ref for s in all_sizes])
jm = np.array([s in judged for s in all_sizes])
offset = np.angle(np.sum(t_rcwa[jm] * np.conj(t_fdtd[jm])))
dphi = np.angle(t_rcwa * np.conj(t_fdtd) * np.exp(-1j * offset)); damp = np.abs(t_rcwa) - np.abs(t_fdtd)
ok = bool(np.all(np.abs(damp[jm]) < TOL_AMP) and np.all(np.abs(dphi[jm]) < TOL_PHASE))
print(f"\nRESULT (fitted constant phase {offset:+.4f} rad; expected ~0 with the bare-interface reference)")
for i, s in enumerate(all_sizes):
    v = ("PASS" if abs(damp[i]) < TOL_AMP and abs(dphi[i]) < TOL_PHASE else "FAIL") if jm[i] else "reported"
    print(f"  {s*1000:6.1f} nm  |t| RCWA {abs(t_rcwa[i]):.4f} FDTD {abs(t_fdtd[i]):.4f}  d|t| {damp[i]:+.4f}  dphase {dphi[i]:+.4f}  {v}")
print(f"\nRECIPE UNIT-CELL VALIDATION {'PASSED' if ok else 'FAILED'}")
os.makedirs("validation", exist_ok=True)
out = os.path.join("validation", f"V1r_{MAT}_{LAM0*1000:.0f}_{H*1000:.0f}_{L*1000:.0f}_{datetime.date.today().isoformat()}.json")
json.dump({"passed": ok, "material": MAT, "lam0": LAM0, "H": H, "L": L, "n_sub": n_sub, "M": a.M,
           "fdtd_steps": a.steps, "sizes_um": all_sizes, "judged": jm.tolist(),
           "t_rcwa": [[z.real, z.imag] for z in t_rcwa], "t_fdtd": [[z.real, z.imag] for z in t_fdtd],
           "phase_offset_rad": float(offset), "d_amp": damp.tolist(), "d_phase_rad": dphi.tolist(),
           "estimated_cost_fc": float(cost), "tidy3d_version": td.__version__}, open(out, "w"), indent=1)
print(f"Record written: {out}")
