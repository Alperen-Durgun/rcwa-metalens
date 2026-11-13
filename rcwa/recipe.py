"""Unit-cell recipe search (roadmap U1/U2).

For a material, wavelength and pillar shape, scan pillar height H and lattice period L and
score each candidate library for lens design:

  * fabrication limits: smallest pillar >= min_feature, smallest gap >= min_gap, aspect ratio
    H / min(min_feature, min_gap) <= max_aspect;
  * physics limits: the period must be sub-wavelength in the substrate and in the exit medium
    (no propagating diffraction order at normal incidence: L < lam0 / max(n_sub, n_bg));
  * the library is refined adaptively (design.refine_library) so resonances are located;
  * figure of merit (FOM): the focal intensity of an ideal lens built from the library with the
    robust rule under a Gaussian size error sigma_nm, relative to a perfect lossless phase plate:

        FOM = (n_bg / n_sub) * max_offset | < E[t_k(phi)] exp(-i (phi + offset)) >_phi |^2

    averaged over uniformly distributed target phases. It combines phase coverage, transmission
    and tolerance in one number (1 = perfect). FOM_nominal is the same without size error.

All numbers come from the locally periodic approximation (single-pillar periodic RCWA).
"""
import numpy as np
from . import design as dz


def library_fom(lib, sigma_nm=3.0, nbin=72, allowed=None):
    """(FOM_robust, FOM_nominal) of a library; see module docstring. The target phases cover
    the circle uniformly, so the result does not depend on a global phase offset."""
    t = np.asarray(lib["t"], complex)
    phases = (np.arange(nbin) + 0.5) * 2 * np.pi / nbin
    scale = lib.get("n_bg", 1.0) / lib.get("n_sub", 1.0)
    stats = dz.size_error_stats(lib, sigma_nm) if sigma_nm else (t, np.zeros(len(t)))
    lut_r, _ = dz.phase_lut(t, nbin=nbin, mode="robust", allowed=allowed, t_stats=stats)
    lut_n, _ = dz.phase_lut(t, nbin=nbin, mode="complex", allowed=allowed)
    fom_r = scale * abs(np.mean(stats[0][lut_r] * np.exp(-1j * phases))) ** 2
    fom_n = scale * abs(np.mean(t[lut_n] * np.exp(-1j * phases))) ** 2
    return float(fom_r), float(fom_n)


def evaluate_cell(lam0, H, L, material=None, n_pillar=2.4, shape="circle", min_feature=0.06,
                  min_gap=0.06, max_aspect=10.0, M=9, n_sizes=16, sigma_nm=3.0, n_sub=1.0,
                  n_bg=1.0, refine=True, n_jobs=None):
    """Score one (H, L) candidate. Returns a dict with the constraints, the library summary
    and the FOM; 'feasible' is False (with 'reason') when a hard limit is violated."""
    row = {"H": float(H), "L": float(L), "lam0": float(lam0), "material": material, "shape": shape,
           "s_min": float(min_feature), "s_max": float(L - min_gap),
           "aspect": float(H / min(min_feature, min_gap))}
    if L >= lam0 / max(n_sub, n_bg):
        return {**row, "feasible": False, "reason": "period not sub-wavelength (diffraction)"}
    if row["s_max"] <= row["s_min"]:
        return {**row, "feasible": False, "reason": "no size range between min_feature and L - min_gap"}
    if row["aspect"] > max_aspect:
        return {**row, "feasible": False, "reason": f"aspect ratio {row['aspect']:.1f} > {max_aspect}"}
    sizes = np.linspace(row["s_min"], row["s_max"], int(n_sizes))
    lib = dz.meta_atom_library(lam0, L, H, sizes=sizes, shape=shape, material=material,
                               n_pillar=n_pillar, M=M, n_sub=n_sub, n_bg=n_bg, n_jobs=n_jobs)
    if refine:
        lib = dz.refine_library(lib, n_jobs=n_jobs)
    fom_r, fom_n = library_fom(lib, sigma_nm=sigma_nm)
    gap = dz.max_phase_gap(lib["t"])
    res = lib.get("resonances", [])
    return {**row, "feasible": True, "reason": "", "coverage_2pi": float(lib["coverage"]),
            "max_phase_gap": float(gap), "min_abs_t": float(lib["amp"].min()),
            "resonances": [round(r["center"], 5) for r in res],
            "resonance_depth": [round(r["t_min"], 3) for r in res],
            "n_deep_resonances": int(sum(r["t_min"] < 0.5 for r in res)),
            "na_max_sampling": float(min(1.0, lam0 / (2 * L))),
            "n_library": int(len(lib["sizes"])), "fom_robust": fom_r, "fom_nominal": fom_n}


def recommend(rows, max_gap=np.pi / 4, tie=0.01):
    """Production-minded choice among the evaluated rows: candidates whose phase gap is at most
    max_gap and whose robust FOM is within `tie` of the best such candidate; among them the one
    with the fewest deep resonances (|t| < 0.5 relative), then the highest NA that the period can
    sample (lam0 / 2L), then the highest FOM. Returns the row, or None if no candidate closes
    the phase circle. Rationale: FOMs within ~1 point are equivalent within the model, whereas
    each deep resonance is a place where a fabrication error and the locally periodic
    approximation both hurt most."""
    ok = [r for r in rows if r.get("feasible") and r["max_phase_gap"] <= max_gap]
    if not ok:
        return None
    top = max(r["fom_robust"] for r in ok)
    near = [r for r in ok if r["fom_robust"] >= top - tie]
    return sorted(near, key=lambda r: (r["n_deep_resonances"], -r["na_max_sampling"], -r["fom_robust"]))[0]


def search_unit_cell(lam0, H_list, L_list, **kw):
    """Evaluate every (H, L) pair (see evaluate_cell for the keywords). Returns the rows,
    feasible ones first, sorted by robust FOM (best first)."""
    rows = [evaluate_cell(lam0, H, L, **kw) for H in H_list for L in L_list]
    return sorted(rows, key=lambda r: (not r["feasible"], -r.get("fom_robust", 0.0)))
