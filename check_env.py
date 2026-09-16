"""ORTAM DENETİMİ — hangi paket kurulu, ne açıyor, ne eksik?
Kullanım: python check_env.py
"""
import importlib

# (modül, zorunlu mu, hangi özelliği açar)
CHECKS = [
    ("numpy",       True,  "Çekirdek RCWA (tüm çözücüler)"),
    ("scipy",       True,  "Optimizasyon (deflektör/tolerans), interpolasyon"),
    ("matplotlib",  True,  "Tüm figürler/raporlar"),
    ("grcwa",       True,  "Doğrulama + E1 adjoint + E3 metal-2B + E4 non-ortogonal kafes"),
    ("autograd",    True,  "E1 adjoint/autodiff gradyanları"),
    ("gdstk",       True,  "GDS üretimi (üretilebilir maske)"),
    ("yaml",        True,  "E7 refractiveindex.info YAML malzeme içe aktarma"),
    ("tidy3d",      False, "3B FDTD bulut doğrulama (E5/E6) — akademik ücretsiz kredi"),
    ("jax",         False, "E2 GPU + autodiff (NVIDIA GPU gerekir)"),
    ("cupy",        False, "E2 GPU dizilim (numpy-uyumlu, NVIDIA GPU)"),
    ("meep",        False, "Yerel/çevrimdışı 3B FDTD (conda/WSL; ağır)"),
]


def main():
    print("=" * 62)
    print("RCWA METALENS — ORTAM DENETİMİ")
    print("=" * 62)
    miss_req, miss_opt = [], []
    for mod, req, feat in CHECKS:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  ✅ {mod:12s} {str(ver):10s} — {feat}")
        except Exception:
            tag = "ZORUNLU" if req else "opsiyonel"
            print(f"  ❌ {mod:12s} {'(eksik)':10s} — [{tag}] {feat}")
            (miss_req if req else miss_opt).append(mod)
    # GPU gerçekten var mı?
    gpu = False
    try:
        import jax
        gpu = any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        pass
    if not gpu:
        try:
            import cupy
            gpu = cupy.cuda.runtime.getDeviceCount() > 0
        except Exception:
            pass
    print("-" * 62)
    print(f"GPU aktif mi: {'EVET' if gpu else 'HAYIR (CPU)'}")
    if miss_req:
        yaml_pkg = "pyyaml" if "yaml" in miss_req else ""
        pkgs = " ".join("pyyaml" if p == "yaml" else p for p in miss_req)
        print(f"\n⚠️ ZORUNLU eksik: {pkgs}")
        print(f"   Kur:  pip install {pkgs}")
    else:
        print("\n✅ Tüm ZORUNLU paketler kurulu — sistem tam çalışır.")
    if miss_opt:
        print(f"ℹ️ Opsiyonel eksik: {', '.join(miss_opt)} (ilgili özellik gerektiğinde kur).")
    print("=" * 62)
    return 0 if not miss_req else 1


if __name__ == "__main__":
    raise SystemExit(main())
