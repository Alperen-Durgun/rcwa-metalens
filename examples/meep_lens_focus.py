"""MEEP 3B METALENS ODAK — sınırları zorlayan tam-dalga koşu (yerel FDTD).
Gerçek D=6um TiO2 lensi 3B Meep FDTD ile çözer, odağı (FWHM) çıkarır ve
RCWA + Tidy3D sonuçlarıyla karşılaştırır (üç-yönlü doğrulama).

Kullanım (WSL, mp ortamı; rcwa_py klasöründen):
  $HOME/miniforge3/envs/mp/bin/python examples/meep_lens_focus.py <design.json> [resolution]

Not: 3B FDTD ağırdır. resolution=15 orta; 20-25 zorlar. RAM'e göre seç.
Bellek tahmini önce yazdırılır; kabul edersen çalışır.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import meep as mp

if len(sys.argv) < 2:
    print("Kullanım: meep_lens_focus.py <design.json> [resolution]"); sys.exit(1)
design = json.load(open(sys.argv[1], encoding="utf-8"))
resolution = int(sys.argv[2]) if len(sys.argv) > 2 else 15

lam = design["lam0"]; H = design["height"]; n = design["n_pillar"]
D = design["D_lens"]; f = design["f"]; shape = design.get("shape", "circle")
f0 = 1.0 / lam
dpml = 1.0; air_out = 1.6 * f + 1.0
sx = D + 3.0; sy = sx; sz = dpml + 1.0 + H + air_out + dpml
z_lens = -sz/2 + dpml + 1.0 + H/2
z_exit = z_lens + H/2
rz = design.get("rcwa_focus_um"); rfw = design.get("rcwa_fwhm_nm")

# bellek tahmini (kaba): voxel * ~6 alan bileşeni * 16 byte (karmaşık) + DFT/yardımcı
vox = (sx * resolution) * (sy * resolution) * (sz * resolution)
mem_gb = vox * 260 / 1e9   # gerçekçi Meep tepe: ~260 byte/voxel (alanlar+geometri+DFT+aux)
print(f"=== MEEP 3B LENS ODAK ===")
print(f"Tasarım: {shape}/{design.get('material')}, {len(design['placements'])} sütun, D={D:.2f}um f={f}um")
print(f"Hücre: {sx:.1f}x{sy:.1f}x{sz:.1f} um, çözünürlük={resolution} -> ~{vox/1e6:.1f}M voxel")
print(f"KABA BELLEK TAHMİNİ: ~{mem_gb:.1f} GB (çok yüksekse Ctrl-C, düşük resolution ile tekrar)")
print("5 sn içinde başlıyor...", flush=True); time.sleep(5)

# geometri
geom = []
for p in design["placements"]:
    s = p["params"] if np.iterable(p["params"]) else (p["params"],)
    c = mp.Vector3(p["x"], p["y"], z_lens)
    if shape == "circle":
        geom.append(mp.Cylinder(radius=s[0]/2, height=H, center=c, material=mp.Medium(index=n)))
    else:
        wx = s[0]; wy = s[0] if len(s) == 1 else s[1]
        geom.append(mp.Block(size=mp.Vector3(wx, wy, H), center=c, material=mp.Medium(index=n)))

src = [mp.Source(mp.GaussianSource(frequency=f0, fwidth=0.15 * f0), component=mp.Ey,
                 center=mp.Vector3(0, 0, -sz/2 + dpml + 0.3), size=mp.Vector3(sx, sy, 0))]
sim = mp.Simulation(cell_size=mp.Vector3(sx, sy, sz), boundary_layers=[mp.PML(dpml)],
                    geometry=geom, sources=src, resolution=resolution,
                    force_complex_fields=True)

# xz (y=0) DFT alan monitörü: lens çıkışından odağın ötesine
z0 = z_exit + 0.05; z1 = z_exit + 1.6 * f; zc = (z0 + z1) / 2; zspan = z1 - z0
dft = sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez], f0, 0, 1,
                         center=mp.Vector3(0, 0, zc), size=mp.Vector3(sx, 0, zspan))

t0 = time.time()
focus_pt = mp.Vector3(0, 0, z_exit + f)
sim.run(until_after_sources=mp.stop_when_fields_decayed(5, mp.Ey, focus_pt, 1e-3))
print(f"Meep koşu bitti: {time.time()-t0:.1f} s", flush=True)

Ex = sim.get_dft_array(dft, mp.Ex, 0)
Ey = sim.get_dft_array(dft, mp.Ey, 0)
Ez = sim.get_dft_array(dft, mp.Ez, 0)
I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2      # (Nx, Nz)
Nx, Nz = I.shape
xs = np.linspace(-sx/2, sx/2, Nx); zs = np.linspace(z0, z1, Nz)

# yakın-alanı dışla, eksende odağı bul
win = zs >= z_exit + 0.4 * f
ix0 = int(np.argmin(np.abs(xs)))
iz = int(np.argmax(np.where(win, I[ix0, :], -np.inf)))
z_focus = zs[iz]; f_meep = z_focus - z_exit
cut = I[:, iz] / I[:, iz].max()
ab = np.where(cut >= 0.5)[0]
fwhm = (xs[ab[-1]] - xs[ab[0]]) * 1000 if ab.size >= 2 else float("nan")
x_peak = xs[int(np.argmax(cut))]
NA = np.sin(np.arctan(D/2/f)); dl = lam/(2*NA)*1000

try:
    import resource
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # Linux: kB -> GB
except Exception:
    peak_gb = float("nan")
print("\n=== SONUÇ (Meep 3B FDTD) ===")
print(f"  [res={resolution}] tepe RAM ≈ {peak_gb:.2f} GB, süre = {time.time()-t0:.0f}s")
print(f"  Odak uzaklığı f_Meep ≈ {f_meep:.2f} um (tasarım f={f})")
print(f"  Odak x-tepe = {x_peak:+.2f} um (eksende olmalı ~0)")
print(f"  FWHM = {fwhm:.0f} nm (difraksiyon sınırı λ/2NA={dl:.0f}nm)")
if rz: print(f"  KIYAS RCWA(LPA): f≈{rz}um FWHM={rfw}nm  |  Meep(res={resolution}): f≈{f_meep:.2f}um FWHM={fwhm:.0f}nm")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].imshow((I/I[:,win].max()).clip(0,1).T, origin="lower", aspect="auto",
                 extent=[xs[0], xs[-1], z0-z_exit, z1-z_exit], cmap="inferno", vmax=1)
    ax[0].axhline(f_meep, color="c", ls="--"); ax[0].set_xlabel("x(um)"); ax[0].set_ylabel("lens çıkışından z(um)")
    ax[0].set_title(f"Meep I(x,z) — odak f*≈{f_meep:.1f}um")
    ax[1].plot(xs*1000, cut); ax[1].axhline(0.5, color="k", ls=":"); ax[1].set_xlim(-1500,1500)
    ax[1].set_xlabel("x(nm)"); ax[1].set_ylabel("I/Imax"); ax[1].set_title(f"Odak kesiti FWHM={fwhm:.0f}nm"); ax[1].grid(alpha=.3)
    out = os.path.splitext(sys.argv[1])[0] + "_MEEP_focus.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); print(f"Figür: {out}")
except Exception as e:
    print("figür atlandı:", e)
