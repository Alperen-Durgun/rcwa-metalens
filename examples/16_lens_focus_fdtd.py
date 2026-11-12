"""16 — SMALL-LENS FULL-WAVE FOCUS VERIFICATION (RCWA ↔ Tidy3D FDTD).

Sets up the full-lens FDTD PROPERLY: instead of ASSUMING the focus, it finds the TRUE
focus (z*, FWHM) with an xz monitor in the y=0 plane. Small lens (D~6um) => cheap + fast,
easy to validate the setup. The physics then generalizes to larger lenses.

Usage (inside the .venv, from the repository root):
  # 1) FREE cost estimate:
  python examples\\16_lens_focus_fdtd.py <design.json>
  # 2) RUN (spends credits):
  python examples\\16_lens_focus_fdtd.py <design.json> run
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import fullwave

if len(sys.argv) < 2:
    print("Usage: python examples\\16_lens_focus_fdtd.py <design.json> [run]"); sys.exit(1)
path = sys.argv[1]; do_run = len(sys.argv) > 2 and sys.argv[2] == "run"
design = json.load(open(path, encoding="utf-8"))

lam0 = design["lam0"]; f = design["f"]; D = design["D_lens"]
rz = design.get("rcwa_focus_um"); rfw = design.get("rcwa_fwhm_nm")
NA = np.sin(np.arctan(D / 2 / f)); dl = lam0 / (2 * NA) * 1000
print(f"Design: {design.get('shape')} / {design.get('material')}, "
      f"{len(design['placements'])} pillars, D={D:.2f}um f={f}um NA={NA:.2f}")
if rz is not None:
    print(f"RCWA prediction: focus z={rz}um, FWHM={rfw}nm (λ/2NA={dl:.0f}nm)")

import tidy3d as td
from tidy3d import web

# run_time: ~4 passes so the focus settles in the small domain (cheap)
sim = fullwave.build_tidy3d(design, add_xz_monitor=True, run_time=2.0e-13, min_steps_per_wvl=15)
print(f"Tidy3D Simulation built: size={tuple(round(s,1) for s in sim.size)}um, "
      f"monitors={[m.name for m in sim.monitors]}")

job = web.Job(simulation=sim, task_name="lens_focus_verify")
cost = job.estimate_cost()
print(f"*** Estimated cost: {cost:.2f} FlexCredit ***  (check that you have enough credits!)")
if not do_run:
    print("\n(Estimate only. Append 'run' to actually run.)"); sys.exit(0)

data = job.run(path="simulation_data.hdf5")

# --- find the TRUE focus from the xz monitor ---
xz = data["xz"]
Ex = np.asarray(xz.Ex.values).squeeze()
Ey = np.asarray(xz.Ey.values).squeeze()
Ez = np.asarray(xz.Ez.values).squeeze()
I = np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2   # (Nx, Nz)
xs = np.asarray(xz.Ex.coords["x"].values)
zs = np.asarray(xz.Ex.coords["z"].values)
if I.shape != (xs.size, zs.size):
    I = I.reshape(xs.size, zs.size)

# brightest z on the axis (x~0) = focus
ix0 = int(np.argmin(np.abs(xs)))
Iaxis = I[ix0, :]
iz = int(np.argmax(Iaxis))
z_focus = float(zs[iz])
# lens exit plane z ~ first monitor point; focal distance = z_focus - z_exit
z_exit = float(zs[0])
f_fdtd = z_focus - z_exit

# x-cut in the focal plane -> FWHM
cut = I[:, iz]; cut = cut / cut.max()
above = np.where(cut >= 0.5)[0]
fwhm = float(xs[above[-1]] - xs[above[0]]) * 1000 if above.size >= 2 else float("nan")
x_peak = float(xs[int(np.argmax(cut))])

print("\n=== FDTD FOCUS (xz monitor, true focus) ===")
print(f"  Focal distance f_FDTD ≈ {f_fdtd:.2f}um (design f={f}um)")
print(f"  Focus x-peak = {x_peak:+.2f}um (should be on-axis ~0)")
print(f"  FWHM = {fwhm:.0f}nm   (diffraction limit λ/2NA={dl:.0f}nm)")
if rz is not None:
    print(f"\n  COMPARE  RCWA: f≈{rz}um, FWHM={rfw}nm   |   FDTD: f≈{f_fdtd:.2f}um, FWHM={fwhm:.0f}nm")

# save the figure
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].imshow(I.T / I.max(), origin="lower", aspect="auto",
                 extent=[xs[0], xs[-1], zs[0], zs[-1]], cmap="inferno")
    ax[0].axhline(z_focus, color="c", ls="--", lw=.8); ax[0].set_xlabel("x(um)"); ax[0].set_ylabel("z(um)")
    ax[0].set_title(f"FDTD I(x,z) — focus z*={z_focus:.1f}um")
    ax[1].plot(xs * 1000, cut); ax[1].axhline(0.5, color="k", ls=":", lw=.7)
    ax[1].set_xlim(-1500, 1500); ax[1].set_xlabel("x(nm)"); ax[1].set_ylabel("I/Imax")
    ax[1].set_title(f"Focal cut — FWHM={fwhm:.0f}nm"); ax[1].grid(alpha=.3)
    out = os.path.splitext(path)[0] + "_FDTD_focus.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); print(f"\nFigure: {out}")
except Exception as e:
    print("figure skipped:", e)
try:
    print("Actual cost:", web.real_cost(job.task_id), "FlexCredit")
except Exception:
    pass
