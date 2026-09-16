"""Örnek 13 — Tasarımı 3B TAM-DALGA ile doğrula (backend SEÇ) — Madde 1.

metalens_design_full çıktısı `*_design.json`'u alır, seçtiğin backend ile
3B FDTD doğrulaması kurar. İki seçenek:
  BACKEND = "tidy3d"  -> GPU bulut (hesap/API anahtarı + internet gerekir; çok hızlı)
  BACKEND = "meep"    -> yerel/çevrimdışı FDTD (conda ile kurulur; ağır)

Kurulum:
  Tidy3D: pip install tidy3d   (sonra: tidy3d configure  ile API anahtarı)
  Meep:   conda install -c conda-forge pymeep

Çalıştır:  python examples/13_fullwave_verify.py  <design.json>
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import fullwave

BACKEND = "tidy3d"   # "tidy3d" ya da "meep" — SENİN SEÇİMİN

def main(path):
    design = json.load(open(path))
    print(f"Tasarım: {design['shape']} / {design.get('material','?')}, "
          f"{len(design['placements'])} sütun, D={design['D_lens']}um f={design['f']}um")
    print(f"Backend: {BACKEND}")
    if BACKEND == "tidy3d":
        from tidy3d import web
        import numpy as np
        sim = fullwave.build_tidy3d(design, run_time=2.5e-13)  # odak için yeterli süre
        ng = len(sim.structures[0].geometry.geometries)
        print(f"Tidy3D Simulation kuruldu: {ng} sütun, boyut={tuple(round(x,1) for x in sim.size)}um.")
        job = web.Job(simulation=sim, task_name="metalens_verify")
        cost = job.estimate_cost()   # ücretsiz
        print(f"*** Tahmini maliyet: {cost:.2f} FlexCredit ***  (kredin yeterli mi kontrol et!)")
        if len(sys.argv) > 2 and sys.argv[2] == "run":
            data = job.run()          # kredi HARCAR
            fd = data["focus"]
            I = (np.abs(fd.Ex.values) ** 2 + np.abs(fd.Ey.values) ** 2 + np.abs(fd.Ez.values) ** 2).squeeze()
            xs = np.asarray(fd.Ex.coords["x"].values)
            j = np.unravel_index(np.argmax(I), I.shape)
            cut = I[:, j[1]] / I.max()
            hw = xs[cut >= 0.5]; fwhm = (hw.max() - hw.min()) * 1000
            print(f"\nFDTD odak düzlemi (z=f={design['f']}um):")
            print(f"  Tepe konumu x={xs[j[0]]:.2f}um  |  FWHM={fwhm:.0f}nm")
            NA = np.sin(np.arctan((design['D_lens'])/2/design['f']))
            print(f"  Difraksiyon sınırı λ/(2NA)={design['lam0']/(2*NA)*1000:.0f}nm")
            print(f"  (RCWA tahmini: odak z=24.8um, FWHM=788nm) -> karşılaştır")
        else:
            print("Çalıştırmak (kredi harcar): python examples/13_fullwave_verify.py <json> run")
    else:
        res = fullwave.run_fullwave(design, backend="meep", resolution=20, until=150)
        print("Meep FDTD çalıştı; alanı işleyip odağı çıkar (bkz. examples/11).")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python examples/13_fullwave_verify.py <design.json>"); sys.exit(1)
    main(sys.argv[1])
