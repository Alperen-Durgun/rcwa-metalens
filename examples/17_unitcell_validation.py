"""Example 17 — V1: unit-cell validation of the meta-atom library against Tidy3D FDTD.

Roadmap item V1 (see CHANGELOG). Compares the zeroth-order transmission coefficient t of
single pillars (periodic unit cell, normal incidence) between the RCWA library and FDTD:

  * t_FDTD = <E_y>(structure) / <E_y>(air reference), field averaged over the unit cell
    (= the zeroth Fourier order) in a plane 0.5 um behind the pillar.
  * Both sides are in the PHYSICS convention exp(-i*omega*t): the RCWA library is converted
    by rcwa.design; Tidy3D uses it natively. No sign is guessed.
  * The two sides differ by one constant phase (reference planes), which is fitted and removed;
    the acceptance test is on |t| and on the remaining (relative) phase.
  * One size is repeated at a finer FDTD grid to show that FDTD itself is converged.

Acceptance (roadmap V1): off-resonance sizes |d|t|| < 0.02 and relative phase < 0.1 rad.
The near-resonance size is reported, not judged.

Usage (inside the .venv, repository root):
  python examples/17_unitcell_validation.py              # RCWA + cost ESTIMATE only (no credits)
  python examples/17_unitcell_validation.py run          # runs the FDTD batch (SPENDS credits)
Results of a run are written to validation/V1_unitcell_<date>.json (commit that file).
"""
import sys, os, json, datetime
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import design
from rcwa.cli import utf8_stdio
utf8_stdio()        # Tidy3D prints arrows; a redirected Windows console is cp1252 by default

# ----------------------------------------------------------------------------- set-up
LAM0, L, H, MAT, SHAPE = 0.633, 0.35, 0.60, "TiO2_t", "circle"
SIZES = [0.10, 0.15, 0.19, 0.25, 0.29]        # off-resonance, spanning the phase range
SIZE_NEAR = 0.215                             # 6 nm below the 221 nm resonance: reported only
SIZE_FINE = 0.25                              # repeated on a finer FDTD grid
STEPS, STEPS_FINE = 20, 30                    # FDTD min_steps_per_wvl
RUN_TIME = 2.0e-12                            # s; the auto-shutoff usually ends earlier
TOL_AMP, TOL_PHASE = 0.02, 0.1

do_run = len(sys.argv) > 1 and sys.argv[1] == "run"
all_sizes = SIZES + [SIZE_NEAR]

# ----------------------------------------------------------------------------- RCWA (free)
lib7 = design.meta_atom_library(LAM0, L, H, sizes=all_sizes, shape=SHAPE, material=MAT, M=7)
lib9 = design.meta_atom_library(LAM0, L, H, sizes=all_sizes, shape=SHAPE, material=MAT, M=9)
t_rcwa = lib9["t"]
n_pillar = complex(lib7["n"])
print(f"RCWA library ({MAT}, n={n_pillar.real:.3f}{'+%.3fi' % n_pillar.imag if n_pillar.imag else ''}), "
      "physics convention:")
for s, a, b in zip(all_sizes, lib7["t"], lib9["t"]):
    print(f"  {s*1000:5.0f} nm  M=9: |t|={abs(b):.4f} phase={np.angle(b):+.4f} rad   "
          f"(M=7 -> M=9 phase change {abs(np.angle(a * np.conj(b))):.4f} rad)")

# ----------------------------------------------------------------------------- Tidy3D
try:
    import tidy3d as td
    from tidy3d import web
except ImportError:
    print("\ntidy3d is not installed in this environment: pip install tidy3d"); sys.exit(0)

freq0 = td.C_0 / LAM0
dpml, air_in, air_out = 0.8, 0.8, 1.2
sz = dpml + air_in + H + air_out + dpml
z_p = -sz / 2 + dpml + air_in + H / 2
z_mon = z_p + H / 2 + 0.5
medium = td.Medium.from_nk(n=n_pillar.real, k=n_pillar.imag, freq=freq0)
source = td.PlaneWave(center=(0, 0, -sz / 2 + dpml + 0.2), size=(td.inf, td.inf, 0),
                      source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 8),
                      direction="+", pol_angle=np.pi / 2)          # E along y, as the RCWA library
monitor = td.FieldMonitor(center=(0, 0, z_mon), size=(td.inf, td.inf, 0), freqs=[freq0], name="field")
bounds = td.BoundarySpec(x=td.Boundary.periodic(), y=td.Boundary.periodic(), z=td.Boundary.pml())


def make_sim(size, steps):
    structs = [] if size is None else [td.Structure(
        geometry=td.Cylinder(center=(0, 0, z_p), radius=size / 2, length=H, axis=2), medium=medium)]
    return td.Simulation(size=(L, L, sz), grid_spec=td.GridSpec.auto(min_steps_per_wvl=steps),
                         structures=structs, sources=[source], monitors=[monitor],
                         run_time=RUN_TIME, boundary_spec=bounds)


sims = {"ref": make_sim(None, STEPS), "ref_fine": make_sim(None, STEPS_FINE),
        f"fine_{SIZE_FINE*1000:.0f}": make_sim(SIZE_FINE, STEPS_FINE)}
for s in all_sizes:
    sims[f"s_{s*1000:.0f}"] = make_sim(s, STEPS)
print(f"\n{len(sims)} Tidy3D simulations built: {', '.join(sims)}")

batch = web.Batch(simulations=sims, folder_name="rcwa_V1_unitcell")
cost = batch.estimate_cost()
print(f"*** Estimated cost of the whole batch: {cost:.3f} FlexCredit ***  (budget for V1: 1.5)")
if not do_run:
    print("\nEstimate only. To run (spends credits): python examples/17_unitcell_validation.py run")
    sys.exit(0)

# ----------------------------------------------------------------------------- run + compare
results = batch.run(path_dir=os.path.join("workspace", "tidy3d_V1"))
uni = -L / 2 + (np.arange(64) + 0.5) * L / 64


def mean_Ey(data):
    ey = data["field"].Ey.isel(f=0).squeeze()
    v = np.asarray(ey.interp(x=uni, y=uni, kwargs={"fill_value": "extrapolate"}).values)
    if not np.all(np.isfinite(v)):
        raise RuntimeError("non-finite field after interpolation; check the monitor data")
    return complex(v.mean())


ref, ref_fine = mean_Ey(results["ref"]), mean_Ey(results["ref_fine"])
t_fdtd = np.array([mean_Ey(results[f"s_{s*1000:.0f}"]) / ref for s in all_sizes])
t_fine = mean_Ey(results[f"fine_{SIZE_FINE*1000:.0f}"]) / ref_fine

off = np.arange(len(SIZES))                               # judged sizes (off-resonance)
offset = np.angle(np.sum(t_rcwa[off] * np.conj(t_fdtd[off])))   # constant reference-plane phase
dphi = np.angle(t_rcwa * np.conj(t_fdtd) * np.exp(-1j * offset))
damp = np.abs(t_rcwa) - np.abs(t_fdtd)
k_fine = all_sizes.index(SIZE_FINE)
fdtd_grid = {"amp": float(abs(abs(t_fine) - abs(t_fdtd[k_fine]))),
             "phase": float(abs(np.angle(t_fine * np.conj(t_fdtd[k_fine]))))}

print(f"\nRESULT  (constant phase offset removed: {offset:+.4f} rad; k0*H = {2*np.pi/LAM0*H % (2*np.pi):.4f} rad)")
print(" size    |t| RCWA  |t| FDTD   d|t|     d(phase) rad   verdict")
ok_all = True
for i, s in enumerate(all_sizes):
    judged = i in off
    ok = abs(damp[i]) < TOL_AMP and abs(dphi[i]) < TOL_PHASE
    ok_all &= ok or not judged
    print(f" {s*1000:4.0f} nm  {abs(t_rcwa[i]):.4f}    {abs(t_fdtd[i]):.4f}   {damp[i]:+.4f}   {dphi[i]:+.4f}        "
          + (("PASS" if ok else "FAIL") if judged else "reported (near resonance)"))
print(f" FDTD grid check at {SIZE_FINE*1000:.0f} nm ({STEPS} -> {STEPS_FINE} steps/wvl): "
      f"d|t|={fdtd_grid['amp']:.4f}, d(phase)={fdtd_grid['phase']:.4f} rad")
print(f"\nV1 {'PASSED' if ok_all else 'FAILED'}  (|d|t|| < {TOL_AMP}, |d phase| < {TOL_PHASE} rad, off-resonance sizes)")

os.makedirs("validation", exist_ok=True)
out = os.path.join("validation", f"V1_unitcell_{datetime.date.today().isoformat()}.json")
json.dump({"item": "V1", "passed": bool(ok_all), "lam0": LAM0, "L": L, "H": H, "material": MAT,
           "n": [n_pillar.real, n_pillar.imag], "shape": SHAPE, "rcwa_M": 9, "fdtd_steps": STEPS,
           "sizes_um": all_sizes, "judged": [i in off for i in range(len(all_sizes))],
           "t_rcwa": [[z.real, z.imag] for z in t_rcwa], "t_fdtd": [[z.real, z.imag] for z in t_fdtd],
           "phase_offset_rad": float(offset), "d_amp": damp.tolist(), "d_phase_rad": dphi.tolist(),
           "fdtd_grid_check": fdtd_grid, "estimated_cost_fc": float(cost),
           "tidy3d_version": td.__version__}, open(out, "w"), indent=1)
print(f"Record written: {out}  (commit it)")
