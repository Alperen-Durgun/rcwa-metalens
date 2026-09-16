"""16b — ODAK ANALİZİ (KREDİSİZ). Zaten indirilmiş simulation_data.hdf5'i tekrar
çözer; FDTD'yi TEKRAR ÇALIŞTIRMAZ (ücret yok).

Düzeltme: odağı ararken lens yüzeyindeki YAKIN-ALANI dışla. Odak, lens çıkışından
uzakta (z_exit + ~f) bir eksen-üstü (x≈0) parlaklık tepesidir; yakın-alan daha
parlak olduğu için tüm-z argmax yanlış sonuç verir.

Kullanım (.venv, rcwa_py klasöründen; hdf5 burada olmalı):
  python examples\\16b_focus_analyze.py <design.json> [simulation_data.hdf5]
"""
import sys, os, json
import numpy as np
import tidy3d as td

if len(sys.argv) < 2:
    print("Kullanım: python examples\\16b_focus_analyze.py <design.json> [hdf5]"); sys.exit(1)
design = json.load(open(sys.argv[1], encoding="utf-8"))
h5 = sys.argv[2] if len(sys.argv) > 2 else "simulation_data.hdf5"
if not os.path.exists(h5):
    print(f"HATA: {h5} bulunamadı. 16_lens_focus_fdtd.py ... run ile üretilmiş olmalı."); sys.exit(1)

lam0 = design["lam0"]; f = design["f"]; D = design["D_lens"]
rz = design.get("rcwa_focus_um"); rfw = design.get("rcwa_fwhm_nm")
NA = np.sin(np.arctan(D / 2 / f)); dl = lam0 / (2 * NA) * 1000

data = td.SimulationData.from_file(h5)

# --- (0) TASARIM ODAK DÜZLEMİ (xy monitör, z=f): merkezde odak var mı? (kesin test) ---
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
    print(f"[z=f={f}um DÜZLEMİ]  merkez/tepe = {Icenter/Ipeak:.2f} "
          f"(1.0=odak tam eksende)  tepe=({xf[pk[0]]:+.2f},{yf[pk[1]]:+.2f})um  "
          f"merkez-kesit FWHM={fwx:.0f}nm")
except Exception as e:
    print("odak-düzlemi analizi atlandı:", e)

xz = data["xz"]
Ex = np.asarray(xz.Ex.values).squeeze()
Ey = np.asarray(xz.Ey.values).squeeze()
Ez = np.asarray(xz.Ez.values).squeeze()
I = np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2
xs = np.asarray(xz.Ex.coords["x"].values)
zs = np.asarray(xz.Ex.coords["z"].values)
if I.shape != (xs.size, zs.size):
    I = I.reshape(xs.size, zs.size)

z_exit = float(zs[0]) - 0.05                   # monitör lens çıkışının ~0.05um altından başlar
z_win = z_exit + 0.40 * f                       # YAKIN-ALANI dışla: odak penceresi
win = zs >= z_win
ix0 = int(np.argmin(np.abs(xs)))                # eksen (x≈0)
Iaxis = I[ix0, :].copy()
Iaxis_win = np.where(win, Iaxis, -np.inf)
iz = int(np.argmax(Iaxis_win))
z_focus = float(zs[iz]); f_fdtd = z_focus - z_exit

cut = I[:, iz]; cut = cut / cut.max()
above = np.where(cut >= 0.5)[0]
fwhm = float(xs[above[-1]] - xs[above[0]]) * 1000 if above.size >= 2 else float("nan")
x_peak = float(xs[int(np.argmax(cut))])

# odaklama kalitesi: odak düzlemi tepe / lens-çıkış eksen ortalaması (kaba kontrast)
print("=== FDTD ODAK (kredisiz analiz; yakın-alan dışlanmış) ===")
print(f"Tasarım: D={D:.2f}um f={f}um NA={NA:.2f}  |  {len(design['placements'])} sütun")
print(f"  Odak uzaklığı  f_FDTD ≈ {f_fdtd:.2f} um   (tasarım f={f}um)")
print(f"  Odak x-tepe    = {x_peak:+.2f} um   (eksende olmalı ~0)")
print(f"  Odak FWHM      = {fwhm:.0f} nm    (λ/2NA={dl:.0f}nm)")
if rz is not None:
    print(f"\n  KIYAS  RCWA:  f≈{rz}um, FWHM={rfw}nm")
    print(f"         FDTD:  f≈{f_fdtd:.2f}um, FWHM={fwhm:.0f}nm")
    print(f"         Δf={abs(f_fdtd-rz):.2f}um, ΔFWHM={abs(fwhm-rfw):.0f}nm")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    # odak-penceresine göre normalize -> odak görünür
    Iw = I[:, win]; zw = zs[win]
    ax[0].imshow((I / Iw.max()).clip(0, 1).T, origin="lower", aspect="auto",
                 extent=[xs[0], xs[-1], zs[0] - z_exit, zs[-1] - z_exit], cmap="inferno", vmax=1)
    ax[0].axhline(f_fdtd, color="c", ls="--", lw=.9)
    ax[0].set_xlabel("x(um)"); ax[0].set_ylabel("lens çıkışından z(um)")
    ax[0].set_title(f"FDTD I(x,z) — odak f*≈{f_fdtd:.1f}um")
    ax[1].plot(zs - z_exit, Iaxis / Iaxis[win].max()); ax[1].axvline(f_fdtd, color="c", ls="--", lw=.9)
    ax[1].axvline(f, color="k", ls=":", lw=.8, label=f"tasarım f={f}")
    ax[1].set_xlabel("lens çıkışından z(um)"); ax[1].set_ylabel("eksen I (norm)")
    ax[1].set_title("Eksen-üstü I(z) — odak tepesi"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].plot(xs * 1000, cut); ax[2].axhline(0.5, color="k", ls=":", lw=.7)
    ax[2].set_xlim(-1500, 1500); ax[2].set_xlabel("x(nm)"); ax[2].set_ylabel("I/Imax")
    ax[2].set_title(f"Odak kesiti — FWHM={fwhm:.0f}nm"); ax[2].grid(alpha=.3)
    out = os.path.splitext(sys.argv[1])[0] + "_FDTD_focus_fix.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); print(f"\nFigür: {out}")
except Exception as e:
    print("figür atlandı:", e)
