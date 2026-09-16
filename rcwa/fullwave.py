"""3B TAM-DALGA KÖPRÜSÜ — Madde 1. İki backend, kullanıcı seçer.

Aynı tasarımı (sütun yerleşimi) iki tam-dalga çözücüsüne köprüler:
  - 'meep'   : yerel, çevrimdışı FDTD (conda ile kurulur; ağır).
  - 'tidy3d' : GPU bulut FDTD (hesap/İnternet gerekir; çok hızlı).

design sözlüğü (metalens_design_full çıktısı ile uyumlu):
  {lam0, Lcell, height, n_pillar, shape, D_lens, f,
   placements: [{x,y,params(size),angle}], er_sub(opsiyonel)}

Kullanım:
  from rcwa import fullwave
  sim = fullwave.build_tidy3d(design)      # Simulation nesnesi (göndermeden)
  fullwave.run_fullwave(design, backend="meep", resolution=25)
"""
import numpy as np


def _pillars(design):
    return design["placements"], design.get("shape", "circle")


# ---------------- Meep (yerel) ----------------
def meep_geometry(design):
    """Saf-python geometri betimi (meep olmadan test edilebilir): blok/silindir listesi."""
    H = design["height"]; n = design["n_pillar"]; shape = design.get("shape", "circle")
    out = []
    for p in design["placements"]:
        s = p["params"] if np.iterable(p["params"]) else (p["params"],)
        out.append({"kind": "cylinder" if shape in ("circle",) else "block",
                    "center": (p["x"], p["y"], 0.0), "size": s, "height": H, "n": n,
                    "angle": p.get("angle", 0.0)})
    return out


def build_meep(design, resolution=25, dpml=1.0, air_out=None):
    """Meep 3B Simulation kur (meep gerekir). meep_geometry'yi mp nesnelerine çevirir."""
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
    src = [mp.Source(mp.ContinuousSource(frequency=1/lam0), component=mp.Ey,
                     center=mp.Vector3(0, 0, -sz/2 + dpml + 0.3), size=mp.Vector3(sx, sy, 0))]
    return mp.Simulation(cell_size=mp.Vector3(sx, sy, sz), boundary_layers=[mp.PML(dpml)],
                         geometry=geom, sources=src, resolution=resolution, force_complex_fields=True)


# ---------------- Tidy3D (GPU bulut) ----------------
def build_tidy3d(design, resolution_um=0.03, run_time=2e-13, add_xz_monitor=False,
                 min_steps_per_wvl=15):
    """Tidy3D 3B Simulation kur (tidy3d gerekir; göndermez). Geometri/kaynak/monitör hazır.
    add_xz_monitor=True: y=0 kesitinde I(x,z) monitörü ekler -> GERÇEK odağı bulmak için."""
    import tidy3d as td
    lam0 = design["lam0"]; H = design["height"]; n = design["n_pillar"]
    D = design["D_lens"]; f = design["f"]; shape = design.get("shape", "circle")
    er_sub = design.get("er_sub", 1.0)
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
    # Aynı malzemeli tüm sütunları tek GeometryGroup'ta topla (performans, önerilen)
    structures = [td.Structure(geometry=td.GeometryGroup(geometries=geoms), medium=med)]
    src = td.PlaneWave(center=(0, 0, -sz/2 + dpml + 0.3), size=(td.inf, td.inf, 0),
                       source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0/10),
                       direction="+", pol_angle=0)
    z_exit = z_lens + H/2
    mon = td.FieldMonitor(center=(0, 0, z_exit + f), size=(sx, sx, 0),
                          freqs=[freq0], name="focus")
    monitors = [mon]
    if add_xz_monitor:
        # y=0 kesitinde I(x,z): lens çıkışından odağın ~1.6f ötesine kadar -> GERÇEK odağı bul
        z0 = z_exit + 0.05          # lens çıkışının hemen altı
        z1 = z_exit + 1.6 * f       # odağın ötesi
        monitors.append(td.FieldMonitor(center=(0, 0, (z0 + z1) / 2),
                                        size=(sx, 0, (z1 - z0)), freqs=[freq0], name="xz"))
    return td.Simulation(size=(sx, sx, sz),
                         grid_spec=td.GridSpec.auto(min_steps_per_wvl=min_steps_per_wvl),
                         structures=structures, sources=[src], monitors=monitors,
                         run_time=run_time, boundary_spec=td.BoundarySpec.all_sides(td.PML()))


def run_fullwave(design, backend="meep", **kw):
    """Tam-dalga doğrulamayı seçilen backend ile çalıştır.
    backend='meep' -> yerel FDTD çalıştırır ve odağı döndürür.
    backend='tidy3d' -> Simulation kurar; web.run için hesap/API anahtarı gerekir."""
    if backend == "meep":
        import meep as mp
        sim = build_meep(design, **kw); sim.run(until=kw.get("until", 150))
        return {"backend": "meep", "sim": sim}
    elif backend == "tidy3d":
        import tidy3d as td
        from tidy3d import web
        sim = build_tidy3d(design)
        # Gerçek çalıştırma (hesap gerekir): data = web.run(sim, task_name="metalens")
        return {"backend": "tidy3d", "sim": sim,
                "not": "web.run(sim, task_name='...') için Tidy3D hesabı + API anahtarı gerekir."}
    raise ValueError("backend 'meep' veya 'tidy3d' olmalı")
