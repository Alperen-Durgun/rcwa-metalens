"""ENVIRONMENT CHECK — which packages are installed, what they enable, what is missing?
Usage: python check_env.py
"""
import importlib

# (module, required?, which feature it enables)
CHECKS = [
    ("numpy",       True,  "Core RCWA (all solvers)"),
    ("scipy",       True,  "Optimization (deflector/tolerance), interpolation"),
    ("matplotlib",  True,  "All figures/reports"),
    ("grcwa",       True,  "Validation + E1 adjoint + E3 metal-2D + E4 non-orthogonal lattice"),
    ("autograd",    True,  "E1 adjoint/autodiff gradients"),
    ("gdstk",       True,  "GDS reading/verification (fabrication mask)"),
    ("yaml",        True,  "E7 refractiveindex.info YAML material import"),
    ("tidy3d",      False, "3D FDTD cloud verification (E5/E6) — free academic credits"),
    ("jax",         False, "E2 GPU + autodiff (requires an NVIDIA GPU)"),
    ("cupy",        False, "E2 GPU arrays (numpy-compatible, NVIDIA GPU)"),
    ("meep",        False, "Local/offline 3D FDTD (conda/WSL; heavy)"),
]


def main():
    print("=" * 62)
    print("RCWA METALENS — ENVIRONMENT CHECK")
    print("=" * 62)
    miss_req, miss_opt = [], []
    for mod, req, feat in CHECKS:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  ✅ {mod:12s} {str(ver):10s} — {feat}")
        except Exception:
            tag = "REQUIRED" if req else "optional"
            print(f"  ❌ {mod:12s} {'(missing)':10s} — [{tag}] {feat}")
            (miss_req if req else miss_opt).append(mod)
    # is there really a GPU?
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
    print(f"GPU active: {'YES' if gpu else 'NO (CPU)'}")
    if miss_req:
        pkgs = " ".join("pyyaml" if p == "yaml" else p for p in miss_req)
        print(f"\n⚠️ REQUIRED missing: {pkgs}")
        print(f"   Install:  pip install {pkgs}")
    else:
        print("\n✅ All REQUIRED packages are installed — the system is fully functional.")
    if miss_opt:
        print(f"ℹ️ Optional missing: {', '.join(miss_opt)} (install when the related feature is needed).")
    print("=" * 62)
    return 0 if not miss_req else 1


if __name__ == "__main__":
    raise SystemExit(main())
