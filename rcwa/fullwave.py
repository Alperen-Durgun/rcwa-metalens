"""3D FULL-WAVE BRIDGE — Item 1. Two backends, the user chooses.

Bridges the same design (pillar layout) to two full-wave solvers:
  - 'meep'   : local, offline FDTD (installed via conda; heavy).
  - 'tidy3d' : GPU cloud FDTD (requires an account/Internet; very fast).

design dictionary (compatible with the metalens_design_full output):
  {lam0, Lcell, height, n_pillar, shape, D_lens, f,
   placements: [{x,y,params(size),angle}], n_sub(optional, substrate index; default 1)}

If n_sub != 1 a substrate half-space is placed under the pillars and the source sits inside
it (same geometry the RCWA library assumes: incidence from the substrate).

Usage:
  from rcwa import fullwave
  sim = fullwave.build_tidy3d(design)      # Simulation object (without submitting)
  fullwave.run_fullwave(design, backend="meep", resolution=25)
"""
import numpy as np


def _pillars(design):
    return design["placements"], design.get("shape", "circle")


# ---------------- Meep (local) ----------------
def meep_geometry(design):
    """Pure-python geometry description (testable without meep): list of blocks/cylinders."""
    H = design["height"]; n = design["n_pillar"]; shape = design.get("shape", "circle")
    out = []
    for p in design["placements"]:
        s = p["params"] if np.iterable(p["params"]) else (p["params"],)
        out.append({"kind": "cylinder" if shape in ("circle",) else "block",
                    "center": (p["x"], p["y"], 0.0), "size": s, "height": H, "n": n,
                    "angle": p.get("angle", 0.0)})
    return out


def build_meep(design, resolution=25, dpml=1.0, air_out=None):
    """Build a Meep 3D Simulation (requires meep). Converts meep_geometry into mp objects."""
    import meep as mp
    lam0 = design["lam0"]; H = design["height"]; n = design["n_pillar"]
    D = design["D_lens"]; f = design["f"]; shape = design.get("shape", "circle")
    air_out = air_out or (1.6 * f + 1.0)
    sx = D + 3.0; sy = sx; sz = dpml + 1.0 + H + air_out + dpml
    z_lens = -sz/2 + dpml + 1.0 + H/2
    geom = []
    for p in design["placements"]:
        s = p["params"] if np.iterable(p["params"]) else (p["params"],)
        c = mp.Vector3(p["x"], p["y"], z_lens)
        if shape == "circle":
            geom.append(mp.Cylinder(radius=s[0]/2, height=H, center=c, material=mp.Medium(index=n)))
        else:
            wx = s[0]; wy = s[0] if len(s) == 1 else s[1]
            geom.append(mp.Block(size=mp.Vector3(wx, wy, H), center=c, material=mp.Medium(index=n)))
    n_sub = float(design.get("n_sub", 1.0))
    if n_sub != 1.0:                         # substrate half-space below the pillars
        z0 = z_lens - H / 2
        geom.insert(0, mp.Block(size=mp.Vector3(mp.inf, mp.inf, z0 + sz / 2),
                                center=mp.Vector3(0, 0, (z0 - sz / 2) / 2), material=mp.Medium(index=n_sub)))
    src = [mp.Source(mp.ContinuousSource(frequency=1/lam0), component=mp.Ey,
                     center=mp.Vector3(0, 0, -sz/2 + dpml + 0.3), size=mp.Vector3(sx, sy, 0))]
    return mp.Simulation(cell_size=mp.Vector3(sx, sy, sz), boundary_layers=[mp.PML(dpml)],
                         geometry=geom, sources=src, resolution=resolution, force_complex_fields=True)


# ---------------- Tidy3D (GPU cloud) ----------------
def build_tidy3d(design, resolution_um=0.03, run_time=2e-13, add_xz_monitor=False,
                 min_steps_per_wvl=15):
    """Build a Tidy3D 3D Simulation (requires tidy3d; does not submit). Geometry/source/monitor ready.
    add_xz_monitor=True: adds an I(x,z) monitor in the y=0 plane -> to find the TRUE focus."""
    import tidy3d as td
    lam0 = design["lam0"]; H = design["height"]; n = design["n_pillar"]
    D = design["D_lens"]; f = design["f"]; shape = design.get("shape", "circle")
    n_sub = float(design.get("n_sub", 1.0))
    freq0 = td.C_0 / lam0
    dpml = 1.0; air_out = 1.6 * f + 1.0
    sx = D + 3.0; sz = dpml + 1.0 + H + air_out + dpml
    z_lens = -sz/2 + dpml + 1.0 + H/2
    med = td.Medium(permittivity=n**2)
    geoms = []
    for p in design["placements"]:
        s = p["params"] if np.iterable(p["params"]) else (p["params"],)
        if shape == "circle":
            geoms.append(td.Cylinder(center=(p["x"], p["y"], z_lens), radius=s[0]/2, length=H, axis=2))
        else:
            wx = s[0]; wy = s[0] if len(s) == 1 else s[1]
            geoms.append(td.Box(center=(p["x"], p["y"], z_lens), size=(wx, wy, H)))
    # Collect all same-material pillars into a single GeometryGroup (performance, recommended)
    structures = [td.Structure(geometry=td.GeometryGroup(geometries=geoms), medium=med)]
    if n_sub != 1.0:                         # substrate half-space below the pillars
        sub = td.Box.from_bounds(rmin=(-td.inf, -td.inf, -td.inf), rmax=(td.inf, td.inf, z_lens - H / 2))
        structures.insert(0, td.Structure(geometry=sub, medium=td.Medium(permittivity=n_sub ** 2)))
    src = td.PlaneWave(center=(0, 0, -sz/2 + dpml + 0.3), size=(td.inf, td.inf, 0),
                       source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0/10),
                       direction="+", pol_angle=0)
    z_exit = z_lens + H/2
    mon = td.FieldMonitor(center=(0, 0, z_exit + f), size=(sx, sx, 0),
                          freqs=[freq0], name="focus")
    monitors = [mon]
    if add_xz_monitor:
        # I(x,z) in the y=0 plane: from the lens exit to ~1.6f beyond -> find the TRUE focus
        z0 = z_exit + 0.05          # just below the lens exit
        z1 = z_exit + 1.6 * f       # beyond the focus
        monitors.append(td.FieldMonitor(center=(0, 0, (z0 + z1) / 2),
                                        size=(sx, 0, (z1 - z0)), freqs=[freq0], name="xz"))
    return td.Simulation(size=(sx, sx, sz),
                         grid_spec=td.GridSpec.auto(min_steps_per_wvl=min_steps_per_wvl),
                         structures=structures, sources=[src], monitors=monitors,
                         run_time=run_time, boundary_spec=td.BoundarySpec.all_sides(td.PML()))


def run_fullwave(design, backend="meep", **kw):
    """Run the full-wave verification with the selected backend.
    backend='meep' -> runs local FDTD and returns the simulation.
    backend='tidy3d' -> builds the Simulation; web.run requires an account/API key."""
    if backend == "meep":
        kw = dict(kw); until = kw.pop("until", 150)      # 'until' belongs to run(), not build_meep()
        sim = build_meep(design, **kw); sim.run(until=until)
        return {"backend": "meep", "sim": sim}
    elif backend == "tidy3d":
        sim = build_tidy3d(design, **kw)
        # Actual run (requires an account): data = web.run(sim, task_name="metalens")
        return {"backend": "tidy3d", "sim": sim,
                "note": "web.run(sim, task_name='...') requires a Tidy3D account + API key."}
    raise ValueError("backend must be 'meep' or 'tidy3d'")
