"""E5/E6 — ENTEGRE 3B KÖPRÜ İŞ AKIŞI + KUTUCUKLAMA + ZEMAX DEVRİ.

- estimate_and_run(design): fullwave.build_tidy3d ile sim kur -> maliyet TAHMİN et ->
  (run=True ise) gönder/bekle/getir -> odak metriklerini döndür. Tek çağrıda uçtan uca.
  (Bulut kısmı tidy3d + API anahtarı gerektirir; anahtarsız yalnız tahmin.)
- tile_design(design, tile_um): büyük lens yerleşimini kutucuklara böler (kutucuk-FDTD
  veya bellek için).
- zemax_phase_csv(design): metalens ideal faz profilini φ(r) Zemax'a devir için CSV yazar
  (Grid Phase / makro import). Sistem-ölçek ışın-optiğine köprü.
"""
import os
import numpy as np


def estimate_and_run(design, run=False, add_xz_monitor=True, run_time=2.0e-13,
                     min_steps_per_wvl=15, task_name="metalens_pipeline", hdf5="pipeline_data.hdf5"):
    """3B Tidy3D iş akışı. run=False -> yalnız maliyet tahmini (ücretsiz).
    run=True -> gönder/bekle/getir (kredi harcar) + odak analizi.
    Döndürür: {estimate, ...(+run ise) focus_um, fwhm_nm, real_cost}."""
    from . import fullwave
    import tidy3d as td
    from tidy3d import web
    sim = fullwave.build_tidy3d(design, add_xz_monitor=add_xz_monitor,
                                run_time=run_time, min_steps_per_wvl=min_steps_per_wvl)
    job = web.Job(simulation=sim, task_name=task_name)
    out = {"estimate": float(job.estimate_cost()), "size_um": tuple(round(s, 2) for s in sim.size)}
    if not run:
        out["note"] = "Yalnız tahmin. run=True ile çalıştır (kredi harcar)."
        return out
    data = job.run(path=hdf5)
    out.update(analyze_focus_data(data, design))
    try:
        out["real_cost"] = float(web.real_cost(job.task_id))
    except Exception:
        pass
    return out


def analyze_focus_data(data, design):
    """xz-monitörden gerçek odağı (yakın-alanı dışlayarak) çıkar. examples/16b ile aynı mantık."""
    f = design["f"]
    xz = data["xz"]
    I = (np.abs(np.asarray(xz.Ex.values).squeeze()) ** 2 +
         np.abs(np.asarray(xz.Ey.values).squeeze()) ** 2 +
         np.abs(np.asarray(xz.Ez.values).squeeze()) ** 2)
    xs = np.asarray(xz.Ex.coords["x"].values); zs = np.asarray(xz.Ex.coords["z"].values)
    if I.shape != (xs.size, zs.size):
        I = I.reshape(xs.size, zs.size)
    z_exit = float(zs[0]) - 0.05
    win = zs >= z_exit + 0.4 * f
    ix0 = int(np.argmin(np.abs(xs)))
    iz = int(np.argmax(np.where(win, I[ix0, :], -np.inf)))
    z_focus = float(zs[iz]); cut = I[:, iz] / I[:, iz].max()
    ab = np.where(cut >= 0.5)[0]
    fwhm = float(xs[ab[-1]] - xs[ab[0]]) * 1000 if ab.size >= 2 else float("nan")
    return {"focus_um": z_focus - z_exit, "fwhm_nm": fwhm, "x_peak_um": float(xs[int(np.argmax(cut))])}


def tile_design(design, tile_um):
    """Büyük lens yerleşimini kare kutucuklara böl (kutucuk-FDTD/bellek için).
    Döndürür: list[design], her biri aynı meta + o kutucuğun placements'i."""
    pl = design["placements"]
    xs = np.array([p["x"] for p in pl]); ys = np.array([p["y"] for p in pl])
    x0, y0 = xs.min(), ys.min()
    tiles = {}
    for p in pl:
        ix = int((p["x"] - x0) // tile_um); iy = int((p["y"] - y0) // tile_um)
        tiles.setdefault((ix, iy), []).append(p)
    out = []
    for (ix, iy), ps in sorted(tiles.items()):
        d = dict(design); d = {k: v for k, v in design.items() if k != "placements"}
        d["placements"] = ps; d["tile"] = (ix, iy)
        out.append(d)
    return out


def zemax_phase_csv(design, csv_path, n_samples=256):
    """Metalens ideal faz profili φ(r) = -(2π/λ)(√(r²+f²)-f) — Zemax'a devir için CSV.
    Zemax 'Grid Phase' yüzeyi / makro ile radyal faz olarak içe alınır (sistem-ölçek ışın izleme)."""
    lam = design["lam0"]; f = design["f"]; D = design["D_lens"]
    r = np.linspace(0, D / 2, n_samples)
    phi = -(2 * np.pi / lam) * (np.sqrt(r ** 2 + f ** 2) - f)     # radyan
    phi_wrapped = np.mod(phi, 2 * np.pi)
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("# Zemax Grid Phase devri — metalens radyal faz\n")
        fh.write(f"# lam_um={lam}, f_um={f}, D_um={D}, NA={np.sin(np.arctan(D/2/f)):.4f}\n")
        fh.write("r_um,phase_rad,phase_wrapped_rad,phase_waves\n")
        for i in range(n_samples):
            fh.write(f"{r[i]:.6f},{phi[i]:.6f},{phi_wrapped[i]:.6f},{phi[i]/(2*np.pi):.6f}\n")
    return {"samples": n_samples, "NA": float(np.sin(np.arctan(D / 2 / f))), "path": csv_path}
