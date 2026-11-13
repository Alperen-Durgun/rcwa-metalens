"""Example 20 - V2: a small 3D metalens, LPA prediction vs Tidy3D FDTD, with an opaque aperture.

Real metalenses are fabricated with an opaque (e.g. chromium) aperture around the lens, so only
light through the lens reaches the focus; the LPA model (design.FocalField) assumes the same
(field = 0 outside the cells). Here the aperture is a PEC film on the substrate with a circular
hole of radius D/2 + L/2 (edge cells fully open). Light comes from the fused-silica substrate.

Compared: focus position (from the lens exit), FWHM in the focal plane (x cut, y-polarized
light), relative efficiency (power in a disk of radius 3 FWHM / power in the focal plane, |E|^2).
Acceptance (roadmap V2): focus within 3 %, FWHM within 5 %; efficiency difference reported.

Usage:  python examples/20_lens_fdtd_aperture.py [--D 6 --f 6] [--run | --load]
"""
import sys, os, json, argparse, datetime
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import design, materials, metrics
from rcwa.cli import utf8_stdio
utf8_stdio()

ap = argparse.ArgumentParser()
ap.add_argument("--D", type=float, default=6.0); ap.add_argument("--f", type=float, default=6.0)
ap.add_argument("--material", default="TiO2_Siefke"); ap.add_argument("--lam0", type=float, default=0.633)
ap.add_argument("--H", type=float, default=0.60); ap.add_argument("--L", type=float, default=0.375)
ap.add_argument("--substrate", default="SiO2_Malitson"); ap.add_argument("--steps", type=int, default=15)
ap.add_argument("--run-time", type=float, default=3e-13)
ap.add_argument("--run", action="store_true"); ap.add_argument("--load", action="store_true")
a = ap.parse_args()
LAM0, L, H, D, F = a.lam0, a.L, a.H, a.D, a.f
n_sub = float(np.sqrt(materials.eps_of(a.substrate, LAM0)).real)

d = design.design_lens(LAM0, L, H, D, F, material=a.material, n_sub=n_sub, certify=True)
an = design.analyze_focus(d)
lpa = {"z_focus": an["z_focus"], "fwhm": an["fwhm"], "eff_relative": an["eff_relative"],
       "eff_absolute": an["eff_absolute"], "strehl": an["strehl"], "NA": d["NA"]}
print(f"Lens: D={d['D_lens']:.3f} um, f={F} um, NA={d['NA']:.3f}, {len(d['placements'])} pillars, {a.material} on n_sub={n_sub:.4f}")
print(f"LPA: focus {lpa['z_focus']:.3f} um, FWHM {lpa['fwhm']*1000:.0f} nm, eff_rel {lpa['eff_relative']:.3f}, "
      f"eff_abs {lpa['eff_absolute']:.3f}, Strehl {lpa['strehl']:.3f}", flush=True)

try:
    import tidy3d as td
    from tidy3d import web
except ImportError:
    sys.exit("tidy3d not installed")
freq0 = td.C_0 / LAM0; n_p = complex(d["lib"]["n"])
Dl = d["D_lens"]; sx = Dl + 2.0
sub_in, air_out = 0.6, 1.6 * F + 1.0
sz = sub_in + H + air_out
z_base = -sz / 2 + sub_in; z_p = z_base + H / 2; z_exit = z_base + H
pm = td.Medium.from_nk(n=n_p.real, k=max(n_p.imag, 0.0), freq=freq0)
structs = [td.Structure(geometry=td.Box.from_bounds(rmin=(-1e3, -1e3, -1e3), rmax=(1e3, 1e3, z_base)),
                        medium=td.Medium(permittivity=n_sub ** 2))]
t_ap = 0.1
slab = td.Box(center=(0, 0, z_base + t_ap / 2), size=(td.inf, td.inf, t_ap))
hole = td.Cylinder(center=(0, 0, z_base + t_ap / 2), radius=Dl / 2 + L / 2, length=2 * t_ap, axis=2)
structs.append(td.Structure(geometry=td.ClipOperation(operation="difference", geometry_a=slab, geometry_b=hole),
                            medium=td.PEC))
geoms = [td.Cylinder(center=(p["x"], p["y"], z_p), radius=float(p["params"]) / 2, length=H, axis=2) for p in d["placements"]]
structs.append(td.Structure(geometry=td.GeometryGroup(geometries=geoms), medium=pm))
zf_lpa = z_exit + lpa["z_focus"]
Z_NEAR = 0.05   # near-field plane above the pillar tops (um)
mons = [td.FieldMonitor(center=(0, 0, (z_exit + 0.05 + z_exit + 1.6 * F) / 2), size=(sx, 0, 1.6 * F - 0.05),
                        freqs=[freq0], name="xz"),
        td.FieldMonitor(center=(0, 0, zf_lpa), size=(sx, sx, 0), freqs=[freq0], name="xy"),
        td.FieldMonitor(center=(0, 0, z_exit + Z_NEAR), size=(sx, sx, 0), freqs=[freq0], name="near")]
sim = td.Simulation(size=(sx, sx, sz), grid_spec=td.GridSpec.auto(min_steps_per_wvl=a.steps), structures=structs,
                    sources=[td.PlaneWave(center=(0, 0, -sz / 2 + 0.2), size=(td.inf, td.inf, 0),
                                          source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 8),
                                          direction="+", pol_angle=np.pi / 2)],
                    monitors=mons, run_time=a.run_time, boundary_spec=td.BoundarySpec.all_sides(td.PML()))
PATH = os.path.join("workspace", f"tidy3d_V2_D{D:g}_f{F:g}.hdf5")
if a.load:
    data = td.SimulationData.from_file(PATH); cost = float("nan")
else:
    job = web.Job(simulation=sim, task_name=f"rcwa_V2_D{D:g}_f{F:g}")
    cost = job.estimate_cost(); print(f"*** Estimated cost: {cost:.3f} FlexCredit ***", flush=True)
    if not a.run:
        sys.exit("Estimate only. Add --run to spend credits.")
    data = job.run(path=PATH)


def intensity(mon):
    f = data[mon]
    return sum(np.abs(np.asarray(getattr(f, c).isel(f=0).values).squeeze()) ** 2 for c in ("Ex", "Ey", "Ez")), f


Ixz, fxz = intensity("xz")
xs = np.asarray(fxz.Ey.coords["x"].values); zs = np.asarray(fxz.Ey.coords["z"].values)
Ixz = Ixz.reshape(len(xs), len(zs)) if Ixz.ndim == 2 else Ixz
ix0 = int(np.argmin(np.abs(xs)))
Iax = Ixz[ix0, :]; k = int(np.argmax(Iax)); zf = float(zs[k])
if 0 < k < len(zs) - 1:
    a1, b1, c1 = Iax[k - 1], Iax[k], Iax[k + 1]; den = a1 - 2 * b1 + c1
    if den < 0:
        zf = float(zs[k] + 0.5 * (a1 - c1) / den * (zs[k + 1] - zs[k]))
Ixy, fxy = intensity("xy")
xx = np.asarray(fxy.Ey.coords["x"].values); yy = np.asarray(fxy.Ey.coords["y"].values)
Ixy = Ixy.reshape(len(xx), len(yy))
u = np.linspace(-sx / 2, sx / 2, 801)
from scipy.interpolate import RegularGridInterpolator
Iu = RegularGridInterpolator((xx, yy), Ixy, bounds_error=False, fill_value=0.0)(np.stack(np.meshgrid(u, u, indexing="ij"), -1))
fw = metrics.focal_metrics(Iu, u)["fwhm_x"]
es = metrics.efficiency_suite(Iu, u, fw, P_incident=None) if "P_incident" in metrics.efficiency_suite.__code__.co_varnames else None
U, V = np.meshgrid(u, u, indexing="ij"); pk = np.unravel_index(np.argmax(Iu), Iu.shape)
r = np.hypot(U - u[pk[0]], V - u[pk[1]])
eff_rel = float(Iu[r <= 3 * fw].sum() / Iu.sum())
fd = {"z_focus": zf - z_exit, "fwhm": fw, "eff_relative": eff_rel, "xy_plane_z": lpa["z_focus"]}
dz_rel = (fd["z_focus"] - lpa["z_focus"]) / lpa["z_focus"]; dfw = (fd["fwhm"] - lpa["fwhm"]) / lpa["fwhm"]
ok = abs(dz_rel) < 0.03 and abs(dfw) < 0.05

# ---- diagnostic: FDTD near field (a) propagated with the LPA angular-spectrum method, (b) per-cell phase vs design
diag = {}
if "near" in data.monitor_data:
    fn = data["near"]; xn = np.asarray(fn.Ey.coords["x"].values); yn = np.asarray(fn.Ey.coords["y"].values)
    Ey_n = np.asarray(fn.Ey.isel(f=0).values).reshape(len(xn), len(yn))
    Ex_n = np.asarray(fn.Ex.isel(f=0).values).reshape(len(xn), len(yn))
    dx = L / 16; Nw = int(np.ceil(sx / dx)); Nw += Nw % 2; g = (np.arange(Nw) - Nw / 2 + 0.5) * dx
    G = np.stack(np.meshgrid(g, g, indexing="ij"), -1)
    def grid(F):
        re = RegularGridInterpolator((xn, yn), F.real, bounds_error=False, fill_value=0.0)(G)
        im = RegularGridInterpolator((xn, yn), F.imag, bounds_error=False, fill_value=0.0)(G)
        return re + 1j * im
    Eg = grid(Ey_n); Np = 2 * Nw; Ep = np.zeros((Np, Np), complex); o = (Np - Nw) // 2; Ep[o:o + Nw, o:o + Nw] = Eg
    k0 = 2 * np.pi / LAM0; kx = 2 * np.pi * np.fft.fftfreq(Np, d=dx); KX, KY = np.meshgrid(kx, kx, indexing="ij")
    KZ = np.sqrt((k0 ** 2 - KX ** 2 - KY ** 2).astype(complex)); KZ = np.where(np.imag(KZ) < 0, -KZ, KZ)
    xc = o + Nw / 2 - 0.5; ramp = np.exp(2j * np.pi * np.fft.fftfreq(Np) * xc)
    best = None
    for sgn in (+1, -1):                       # time convention: field as is, or its conjugate (physics branch)
        A = np.fft.fft2(Ep if sgn > 0 else np.conj(Ep)); Ac = A * ramp[:, None] * ramp[None, :] / Np ** 2
        zc = np.linspace(0.4 * F, 1.8 * F, 281) - Z_NEAR
        Iz = np.array([abs(np.sum(Ac * np.exp(1j * KZ * z))) ** 2 for z in zc])
        i = int(np.argmax(Iz)); zq = float(zc[i])
        if 0 < i < len(zc) - 1:
            a1, b1, c1 = Iz[i - 1], Iz[i], Iz[i + 1]; den = a1 - 2 * b1 + c1
            if den < 0:
                zq = float(zc[i] + 0.5 * (a1 - c1) / den * (zc[1] - zc[0]))
        if best is None or Iz.max() > best[1]:
            best = (zq + Z_NEAR, float(Iz.max()), sgn)
    diag["z_focus_fdtd_nearfield_propagated"] = best[0]; diag["time_sign"] = best[2]
    # per-cell average of Ey vs the design aperture field
    E0 = np.asarray(d["E0"]); Nc = E0.shape[0]; ci = (np.arange(Nc) - (Nc - 1) / 2) * L
    Ec = np.zeros_like(E0)
    for i_, cx in enumerate(ci):
        mx = np.abs(g - cx) < L / 2
        for j_, cy in enumerate(ci):
            if E0[i_, j_] != 0:
                Ec[i_, j_] = Eg[np.ix_(mx, np.abs(g - cy) < L / 2)].mean()
    if best[2] < 0:
        Ec = np.conj(Ec)
    m = np.abs(E0) > 0; ref = np.sum(Ec[m] * np.conj(E0[m])); Ec = Ec * np.exp(-1j * np.angle(ref))
    dphi = np.angle(Ec[m] / E0[m]); XX, YY = np.meshgrid(ci, ci, indexing="ij"); rr = np.hypot(XX, YY)[m]
    bins = np.linspace(0, D / 2 + L, 6); prof = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        sel = (rr >= b0) & (rr < b1)
        if sel.any():
            prof.append({"r_um": [float(b0), float(b1)], "rms_phase_err_rad": float(np.sqrt(np.mean(dphi[sel] ** 2))),
                         "mean_amp_ratio": float(np.mean(np.abs(Ec[m][sel]) / np.abs(E0[m][sel])))})
    diag["cell_phase_rms_rad"] = float(np.sqrt(np.mean(dphi ** 2))); diag["cell_profile"] = prof
    diag["cross_pol_ratio"] = float(np.sum(np.abs(Ex_n) ** 2) / np.sum(np.abs(Ey_n) ** 2))
    # (c) LPA propagation of the FDTD per-cell field: isolates sampling (per-cell constant) vs near-field content
    ffc = design.FocalField(Ec, L, LAM0, n_bg=1.0); diag["z_focus_lpa_of_fdtd_cells"] = ffc.find_focus(F, nz=281)[0]
    # (d) FDTD near field kept only over the lens cells (light through the open rim outside the cells removed)
    on = np.zeros((Nw, Nw), bool)
    for i_, cx in enumerate(ci):
        mx = np.abs(g - cx) < L / 2
        for j_, cy in enumerate(ci):
            if E0[i_, j_] != 0:
                on[np.ix_(mx, np.abs(g - cy) < L / 2)] = True
    Em = np.where(on, Eg, 0) if best[2] > 0 else np.where(on, np.conj(Eg), 0)
    Epm = np.zeros((Np, Np), complex); Epm[o:o + Nw, o:o + Nw] = Em
    Acm = np.fft.fft2(Epm) * ramp[:, None] * ramp[None, :] / Np ** 2
    zc = np.linspace(0.4 * F, 1.8 * F, 281) - Z_NEAR
    Izm = np.array([abs(np.sum(Acm * np.exp(1j * KZ * z))) ** 2 for z in zc]); i = int(np.argmax(Izm)); zq = float(zc[i])
    a1, b1, c1 = Izm[i - 1], Izm[i], Izm[i + 1]; den = a1 - 2 * b1 + c1
    zq = float(zc[i] + 0.5 * (a1 - c1) / den * (zc[1] - zc[0])) if den < 0 else zq
    diag["z_focus_fdtd_nearfield_cells_only"] = zq + Z_NEAR
    diag["rim_power_fraction"] = float(np.sum(np.abs(Eg[~on]) ** 2) / np.sum(np.abs(Eg) ** 2))
    # (e) propagating part only (|k| < k0) of the FDTD near field, then cell-averaged -> LPA propagator.
    #     This is the quantity the library t represents (zero order); its error map is the coupling error.
    Ef = np.zeros((Np, Np), complex); Ef[o:o + Nw, o:o + Nw] = Eg if best[2] > 0 else np.conj(Eg)
    Elp = np.fft.ifft2(np.fft.fft2(Ef) * (KX ** 2 + KY ** 2 < k0 ** 2))[o:o + Nw, o:o + Nw]
    Ecl = np.zeros_like(E0)
    for i_, cx in enumerate(ci):
        mx = np.abs(g - cx) < L / 2
        for j_, cy in enumerate(ci):
            if E0[i_, j_] != 0:
                Ecl[i_, j_] = Elp[np.ix_(mx, np.abs(g - cy) < L / 2)].mean()
    diag["z_focus_lpa_of_fdtd_cells_lowpass"] = design.FocalField(Ecl, L, LAM0, n_bg=1.0).find_focus(F, nz=281)[0]
    refl = np.sum(Ecl[m] * np.conj(E0[m])); Ecl = Ecl * np.exp(-1j * np.angle(refl))
    dpl = np.angle(Ecl[m] / E0[m]); Arl = np.abs(Ecl[m]) / np.abs(E0[m]); Arl = Arl / np.median(Arl)
    profl = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        sel = (rr >= b0) & (rr < b1)
        if sel.any():
            profl.append({"r_um": [float(b0), float(b1)], "mean_phase_err_rad": float(np.mean(dpl[sel])),
                          "rms_phase_err_rad": float(np.sqrt(np.mean(dpl[sel] ** 2))), "rel_amp": float(np.mean(Arl[sel]))})
    diag["cell_profile_lowpass"] = profl; diag["cell_phase_rms_lowpass_rad"] = float(np.sqrt(np.mean(dpl ** 2)))
    print("\nDIAGNOSTIC")
    print(f"  FDTD near field, angular-spectrum propagated: focus {diag['z_focus_fdtd_nearfield_propagated']:.3f} um")
    print(f"  FDTD cell-averaged field, LPA propagator:     focus {diag['z_focus_lpa_of_fdtd_cells']:.3f} um")
    print(f"  FDTD near field over lens cells only:         focus {diag['z_focus_fdtd_nearfield_cells_only']:.3f} um"
          f"  (power outside the cells: {diag['rim_power_fraction']:.3f})")
    print(f"  per-cell phase error vs design: rms {diag['cell_phase_rms_rad']:.3f} rad; cross-pol {diag['cross_pol_ratio']:.2e}")
    for q in prof:
        print(f"    r {q['r_um'][0]:.2f}-{q['r_um'][1]:.2f} um: rms {q['rms_phase_err_rad']:.3f} rad, |E|/|E0| {q['mean_amp_ratio']:.3f}")
    print(f"  low-passed (propagating) cell field, LPA propagator: focus {diag['z_focus_lpa_of_fdtd_cells_lowpass']:.3f} um;"
          f" phase rms {diag['cell_phase_rms_lowpass_rad']:.3f} rad")
    for q in profl:
        print(f"    r {q['r_um'][0]:.2f}-{q['r_um'][1]:.2f} um: mean {q['mean_phase_err_rad']:+.3f} rms {q['rms_phase_err_rad']:.3f} rad,"
              f" rel |E| {q['rel_amp']:.3f}")
print("\nRESULT  (LPA vs FDTD)")
print(f"  focus  {lpa['z_focus']:.3f} vs {fd['z_focus']:.3f} um   ({dz_rel*100:+.1f} %)")
print(f"  FWHM   {lpa['fwhm']*1000:.0f} vs {fd['fwhm']*1000:.0f} nm   ({dfw*100:+.1f} %)  [FDTD FWHM in the plane of the LPA focus]")
print(f"  eff_rel {lpa['eff_relative']:.3f} vs {fd['eff_relative']:.3f}")
print(f"  final field decay {getattr(data, 'final_decay_value', float('nan')):.1e}")
print(f"\nV2 {'PASSED' if ok else 'FAILED'} (focus within 3 %, FWHM within 5 %)")
os.makedirs("validation", exist_ok=True)
out = os.path.join("validation", f"V2_lens_D{D:g}_f{F:g}_{a.material}_{datetime.date.today().isoformat()}.json")
json.dump({"passed": ok, "lpa": lpa, "fdtd": fd, "D": Dl, "f": F, "L": L, "H": H, "n_sub": n_sub,
           "material": a.material, "steps": a.steps, "run_time": a.run_time, "estimated_cost_fc": float(cost),
           "aperture": "PEC film, hole radius D/2 + L/2", "tidy3d_version": td.__version__,
           "final_decay": float(getattr(data, "final_decay_value", np.nan)), "diagnostic": diag}, open(out, "w"), indent=1)
print(f"Record written: {out}")