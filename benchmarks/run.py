"""Benchmark suite: fixed problems, time + accuracy (roadmap N5).

Every problem returns a few numbers; the first run stores them in benchmarks/baseline.json.
Later runs compare against it:

  python benchmarks/run.py            # print table (time + values + drift vs baseline)
  python benchmarks/run.py --check    # exit 1 if any VALUE drifted beyond its tolerance
  python benchmarks/run.py --update   # rewrite the baseline (after an intended change)

Values are checked strictly (they are deterministic up to BLAS rounding). Times are only
reported, with a warning above 3x the baseline: CI machines differ too much to fail on time.
Nothing is cached (each run solves from scratch).
"""
import os, sys, json, time, platform, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("RCWA_CACHE_DIR", tempfile.mkdtemp(prefix="rcwa_bench_"))
from rcwa import solve_rcwa_1d, shapes, design as dz           # noqa: E402
from rcwa.solver2d import solve_rcwa_2d                       # noqa: E402
from rcwa.convention import to_physical                       # noqa: E402

BASELINE = os.path.join(HERE, "baseline.json")


def _fresnel_R(lam0, n1, d):
    r01 = (1 - n1) / (1 + n1); dl = 2 * np.pi / lam0 * n1 * d
    r = (r01 - r01 * np.exp(2j * dl)) / (1 - r01 ** 2 * np.exp(2j * dl))
    return abs(r) ** 2


def p_slab_1d():
    eps = np.full(512, 1.5 ** 2, complex)
    r = solve_rcwa_1d(0.55, 0.0, 1.0, 1.0, 0.30, [(eps, 0.40)], 21, pol=(0., 1.))
    return {"R0": float(r["R"][r["zeroth_index"]]), "err_vs_fresnel": abs(r["R"][r["zeroth_index"]] - _fresnel_R(0.55, 1.5, 0.40))}


def p_pillar_2d():
    eps = shapes.rasterize("circle", 0.20, 0.35, 512, n_pillar=2.5)
    t0 = time.perf_counter(); a = solve_rcwa_2d(0.633, 0, 0, 1, 1, 0.35, 0.35, [(eps, 0.6)], 9, 9); ts = time.perf_counter() - t0
    t0 = time.perf_counter(); b = solve_rcwa_2d(0.633, 0, 0, 1, 1, 0.35, 0.35, [(eps, 0.6)], 9, 9, symmetry=False); tf = time.perf_counter() - t0
    t = to_physical(a["ty"][a["i0"]])
    return {"abs_t": float(abs(t)), "phase_t": float(np.angle(t)), "Ttot": a["Ttot"],
            "sym_vs_full": float(np.abs(a["ty"] - b["ty"]).max()), "_time_sym_s": ts, "_time_full_s": tf}


def p_library():
    lib = dz.meta_atom_library(0.633, 0.35, 0.60, material="TiO2_t", use_cache=False)
    return {"coverage": lib["coverage"], "max_phase_gap": dz.max_phase_gap(lib["t"]),
            "min_abs_t": float(lib["amp"].min())}


def p_lens():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = dz.design_lens(0.633, 0.35, 0.60, 16.0, 20.0, material="TiO2_t")
    a = dz.analyze_focus(d)
    return {"z_focus": a["z_focus"], "fwhm": a["fwhm"], "strehl": a["strehl"],
            "eff_absolute": a["eff_absolute"], "n_lib": len(d["lib"]["sizes"])}


PROBLEMS = {"slab_1d": (p_slab_1d, 1e-10), "pillar_2d_M9": (p_pillar_2d, 1e-8),
            "library_20_M9": (p_library, 1e-8), "lens_16um_default": (p_lens, 1e-6)}


def main(argv):
    base = json.load(open(BASELINE, encoding="utf-8")) if os.path.exists(BASELINE) else None
    res = {"_meta": {"python": platform.python_version(), "numpy": np.__version__,
                     "machine": platform.platform()}}
    bad = []
    for name, (fn, tol) in PROBLEMS.items():
        t0 = time.perf_counter(); vals = fn(); el = time.perf_counter() - t0
        vals["_time_s"] = el; res[name] = vals
        line = f"{name:20s} {el:7.2f} s"
        if base and name in base:
            b = base[name]
            drift = max((abs(vals[k] - b[k]) for k in vals if not k.startswith("_") and k in b), default=0.0)
            slow = el / max(b.get("_time_s", el), 1e-9)
            line += f"   max drift {drift:.1e} (tol {tol:.0e}){'  <-- DRIFT' if drift > tol else ''}"
            line += f"   time x{slow:.2f} vs baseline{'  (warning: > 3x)' if slow > 3 else ''}"
            if drift > tol:
                bad.append(name)
        print(line, flush=True)
        print("    " + ", ".join(f"{k}={v:.6g}" for k, v in vals.items()), flush=True)
    if "--update" in argv or base is None:
        json.dump(res, open(BASELINE, "w", encoding="utf-8"), indent=1)
        print(f"baseline written: {BASELINE}")
    if "--check" in argv and bad:
        print(f"FAILED: values drifted in {', '.join(bad)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
