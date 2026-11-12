"""16b — FOCUS ANALYSIS (NO CREDITS). Re-analyzes an already downloaded
simulation_data.hdf5; it does NOT re-run the FDTD (no cost).

Fix: exclude the NEAR FIELD at the lens surface when searching for the focus. The focus
is an on-axis (x≈0) intensity peak far from the lens exit (z_exit + ~f); since the near
field is brighter, an argmax over all z gives the wrong answer.

Usage (inside the .venv, from the repository root; the hdf5 must be here):
  python examples\\16b_focus_analyze.py <design.json> [simulation_data.hdf5]
"""
import sys, os, json
import numpy as np
import tidy3d as td

if len(sys.argv) < 2:
    print("Usage: python examples\\16b_focus_analyze.py <design.json> [hdf5]"); sys.exit(1)
design = json.load(open(sys.argv[1], encoding="utf-8"))
h5 = sys.argv[2] if len(sys.argv) > 2 else "simulation_data.hdf5"
if not os.path.exists(h5):
    print(f"ERROR: {h5} not found. It should have been produced by 16_lens_focus_fdtd.py ... run."); sys.exit(1)

lam0 = design["lam0"]; f = design["f"]; D = design["D_lens"]
rz = design.get("rcwa_focus_um"); rfw = design.get("rcwa_fwhm_nm")
NA = np.sin(np.arctan(D / 2 / f)); dl = lam0 / (2 * NA) * 1000

data = td.SimulationData.from_file(h5)

# --- (0) DESIGN FOCAL PLANE (xy monitor, z=f): is there a focus at the center? (definitive test) ---
try:
    fo = data["focus"]
    Exf = np.asarray(fo.Ex.values).squeeze(); Eyf = np.asarray(fo.Ey.values).squeeze()
    Ezf = np.asarray(fo.Ez.values).squeeze()
    If = np.abs(Exf) ** 2 + np.abs(Eyf) ** 2 + np.abs(Ezf) ** 2
    xf = np.asarray(fo.Ex.coords["x"].values); yf = np.asarray(fo.Ex.coords["y"].values)
    if If.shape != (xf.size, yf.size): If = If.reshape(xf.size, yf.size)
    icx = int(np.argmin(np.abs(xf))); icy = int(np.argmin(np.abs(yf)))
    pk = np.unravel_index(int(np.argmax(If)), If.shape)
    Icenter = If[icx, icy]; Ipeak = If[pk]
    cutx = If[:, icy]; cutx = cutx / cutx.max(); ab = np.where(cutx >= 0.5)[0]
    fwx = (xf[ab[-1]] - xf[ab[0]]) * 1000 if ab.size >= 2 else float("nan")
    print(f"[z=f={f}um PLANE]  center/peak = {Icenter/Ipeak:.2f} "
          f"(1.0=focus exactly on-axis)  peak=({xf[pk[0]]:+.2f},{yf[pk[1]]:+.2f})um  "
          f"center-cut FWHM={fwx:.0f}nm")
except Exception as e:
    print("focal-plane analysis skipped:", e)

xz = data["xz"]
Ex = np.asarray(xz.Ex.values).squeeze()
Ey = np.asarray(xz.Ey.values).squeeze()
Ez = np.asarray(xz.Ez.values).squeeze()
I = np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2
xs = np.asarray(xz.Ex.coords["x"].values)
zs = np.asarray(xz.Ex.coords["z"].values)
if I.shape != (xs.size, zs.size):
    I = I.reshape(xs.size, zs.size)

z_exit = float(zs[0]) - 0.05                   # the monitor starts ~0.05um below the lens exit
z_win = z_exit + 0.40 * f                       # exclude the NEAR FIELD: focus window
win = zs >= z_win
ix0 = int(np.argmin(np.abs(xs)))                # axis (x≈0)
Iaxis = I[ix0, :].copy()
Iaxis_win = np.where(win, Iaxis, -np.inf)
iz = int(np.argmax(Iaxis_win))
z_focus = float(zs[iz]); f_fdtd = z_focus - z_exit

cut = I[:, iz]; cut = cut / cut.max()
above = np.where(cut >= 0.5)[0]
fwhm = float(xs[above[-1]] - xs[above[0]]) * 1000 if above.size >= 2 else float("nan")
x_peak = float(xs[int(np.argmax(cut))])

# focusing quality: focal-plane peak / on-axis mean at the lens exit (rough contrast)
print("=== FDTD FOCUS (no-credit analysis; near field excluded) ===")
print(f"Design: D={D:.2f}um f={f}um NA={NA:.2f}  |  {len(design['placements'])} pillars")
print(f"  Focal distance f_FDTD ≈ {f_fdtd:.2f} um   (design f={f}um)")
print(f"  Focus x-peak   = {x_peak:+.2f} um   (should be on-axis ~0)")
print(f"  Focus FWHM     = {fwhm:.0f} nm    (λ/2NA={dl:.0f}nm)")
if rz is not None:
    print(f"\n  COMPARE  RCWA:  f≈{rz}um, FWHM={rfw}nm")
    print(f"           FDTD:  f≈{f_fdtd:.2f}um, FWHM={fwhm:.0f}nm")
    print(f"           Δf={abs(f_fdtd-rz):.2f}um, ΔFWHM={abs(fwhm-rfw):.0f}nm")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    # normalize to the focus window -> the focus becomes visible
    Iw = I[:, win]; zw = zs[win]
    ax[0].imshow((I / Iw.max()).clip(0, 1).T, origin="lower", aspect="auto",
                 extent=[xs[0], xs[-1], zs[0] - z_exit, zs[-1] - z_exit], cmap="inferno", vmax=1)
    ax[0].axhline(f_fdtd, color="c", ls="--", lw=.9)
    ax[0].set_xlabel("x(um)"); ax[0].set_ylabel("z from lens exit (um)")
    ax[0].set_title(f"FDTD I(x,z) — focus f*≈{f_fdtd:.1f}um")
    ax[1].plot(zs - z_exit, Iaxis / Iaxis[win].max()); ax[1].axvline(f_fdtd, color="c", ls="--", lw=.9)
    ax[1].axvline(f, color="k", ls=":", lw=.8, label=f"design f={f}")
    ax[1].set_xlabel("z from lens exit (um)"); ax[1].set_ylabel("on-axis I (norm)")
    ax[1].set_title("On-axis I(z) — focal peak"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].plot(xs * 1000, cut); ax[2].axhline(0.5, color="k", ls=":", lw=.7)
    ax[2].set_xlim(-1500, 1500); ax[2].set_xlabel("x(nm)"); ax[2].set_ylabel("I/Imax")
    ax[2].set_title(f"Focal cut — FWHM={fwhm:.0f}nm"); ax[2].grid(alpha=.3)
    out = os.path.splitext(sys.argv[1])[0] + "_FDTD_focus_fix.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); print(f"\nFigure: {out}")
except Exception as e:
    print("figure skipped:", e)
