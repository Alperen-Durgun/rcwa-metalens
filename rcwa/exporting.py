"""E8 — FAR-FIELD / BSDF EXPORT + SIMPLE DRC (fabrication/system bridge).

- bsdf_orders(result, lam, Lx, Ly): converts diffraction orders into an angle + efficiency
  (BSDF-like) table; can be written to CSV (hand-off to system/ray-optics tools).
- far_field(E0, dx, lam): far-field angular intensity distribution from the aperture field.
- drc_check(placements, min_feature, min_gap): design-rule check (min feature/gap).
"""
import numpy as np


def bsdf_orders(result, lam, Lx, Ly, csv_path=None):
    """From a 2D solve result (orders=(MXf,MYf), R, T): table of scattering angle (θ,φ)
    + efficiency for each order. Writes a CSV if csv_path is given.
    Returns: list[dict]."""
    MXf, MYf = result["orders"]; R = result["R"]; T = result["T"]
    rows = []
    for i in range(len(MXf)):
        kx = -MXf[i] * (lam / Lx); ky = -MYf[i] * (lam / Ly)
        kt = np.hypot(kx, ky)
        theta = np.degrees(np.arcsin(min(kt, 1.0)))
        phi = np.degrees(np.arctan2(ky, kx))
        prop = kt <= 1.0
        rows.append({"m": int(MXf[i]), "n": int(MYf[i]), "theta_deg": round(float(theta), 3),
                     "phi_deg": round(float(phi), 3), "R": float(R[i]), "T": float(T[i]),
                     "propagating": bool(prop)})
    if csv_path:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("m,n,theta_deg,phi_deg,R,T,propagating\n")
            for r in rows:
                f.write(f"{r['m']},{r['n']},{r['theta_deg']},{r['phi_deg']},"
                        f"{r['R']:.6e},{r['T']:.6e},{int(r['propagating'])}\n")
    return rows


def far_field(E0, dx, lam):
    """Aperture field E0(x,y) -> far-field angular intensity I(θx,θy) (Fraunhofer/FFT).
    Returns: (I_norm, theta_x_deg, theta_y_deg)."""
    E0 = np.asarray(E0, complex); N = E0.shape[0]
    F = np.fft.fftshift(np.fft.fft2(E0)); I = np.abs(F) ** 2
    fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))       # 1/um
    theta = np.degrees(np.arcsin(np.clip(fx * lam, -1, 1)))
    return I / I.max(), theta, theta


DRC_PROFILES = {
    # Built-in profiles are this package's own defaults, NOT foundry rules. Put your process's
    # numbers in a YAML file (docs/drc_profile_template.yaml) and pass its path as `profile`.
    "rcwa_default": {"min_feature_um": 0.05, "min_gap_um": 0.04, "max_aspect": None,
                     "notes": "historic defaults of drc_check"},
    "recipe_search_default": {"min_feature_um": 0.06, "min_gap_um": 0.06, "max_aspect": 15.0,
                              "notes": "limits used by recipe_search / docs/recipes"},
}


def load_drc_profile(profile):
    """A DRC profile by built-in name or YAML path -> dict(min_feature_um, min_gap_um,
    max_aspect, notes). Missing keys are refused rather than guessed."""
    if isinstance(profile, dict):
        prof = dict(profile)
    elif profile in DRC_PROFILES:
        prof = dict(DRC_PROFILES[profile])
    else:
        import yaml
        with open(profile, encoding="utf-8") as f:
            prof = yaml.safe_load(f) or {}
    missing = [k for k in ("min_feature_um", "min_gap_um") if prof.get(k) is None]
    if missing:
        raise ValueError(f"DRC profile {profile!r} lacks {missing}")
    prof.setdefault("max_aspect", None); prof.setdefault("notes", "")
    return prof


def drc_check(placements, min_feature=0.05, min_gap=0.04, shape_key="params", max_report=50,
              profile=None, height=None, max_aspect=None):
    """Simple design-rule check (DRC) — scales to millions of pillars (k-d tree):
      - min feature: every pillar size >= min_feature (um)
      - min gap: smallest edge-to-edge spacing to the 8 nearest neighbours >= min_gap (um)
      - aspect ratio (if height and max_aspect are given): height / min(pillar size, its
        smallest gap) <= max_aspect — both the pillar and the trench next to it must be etched
    profile: a built-in name (DRC_PROFILES) or a YAML path; its values replace min_feature,
    min_gap and max_aspect. The gap uses the first size parameter as the pillar extent (exact for
    circles and for axis-aligned squares on a square lattice; conservative for other shapes).
    Returns: {ok, n, n_feature_viol, n_gap_viol, n_aspect_viol, worst_gap, min_size,
    worst_aspect, profile, violations[list]}."""
    from scipy.spatial import cKDTree
    prof = None
    if profile is not None:
        prof = load_drc_profile(profile)
        min_feature, min_gap = prof["min_feature_um"], prof["min_gap_um"]
        max_aspect = prof["max_aspect"] if max_aspect is None else max_aspect
    n = len(placements)
    if n == 0:
        return {"ok": True, "n": 0, "n_feature_viol": 0, "n_gap_viol": 0, "n_aspect_viol": 0,
                "worst_gap": -1.0, "min_size": 0.0, "worst_aspect": 0.0, "profile": prof, "violations": []}
    xy = np.array([[p["x"], p["y"]] for p in placements], float)
    sizes = np.array([float(np.atleast_1d(p[shape_key])[0]) for p in placements])
    viol = []
    fv = np.where(sizes < min_feature)[0]
    for i in fv[:max_report]:
        viol.append({"type": "min_feature", "i": int(i), "value": float(sizes[i])})
    worst = np.inf; gv = np.zeros(0, int); gap = np.full(n, np.inf)
    if n > 1:
        kq = min(n, 9)                                    # self + the 8 lattice neighbours
        dist, j = cKDTree(xy).query(xy, k=kq)
        gaps = dist[:, 1:] - (sizes[:, None] + sizes[j[:, 1:]]) / 2.0
        a = np.argmin(gaps, axis=1); gap = gaps[np.arange(n), a]; jn = j[np.arange(n), a + 1]
        worst = float(gap.min()); gv = np.where(gap < min_gap)[0]
        for i in gv[:max(0, max_report - len(viol))]:
            viol.append({"type": "min_gap", "i": int(i), "j": int(jn[i]), "gap": round(float(gap[i]), 4)})
    av = np.zeros(0, int); worst_ar = 0.0
    if height is not None:
        ar = float(height) / np.maximum(np.minimum(sizes, np.where(np.isfinite(gap), gap, np.inf)), 1e-12)
        worst_ar = float(ar.max())
        if max_aspect is not None:
            av = np.where(ar > max_aspect * (1 + 1e-9))[0]
            for i in av[:max(0, max_report - len(viol))]:
                viol.append({"type": "aspect", "i": int(i), "value": round(float(ar[i]), 2)})
    return {"ok": len(fv) == 0 and len(gv) == 0 and len(av) == 0, "n": n,
            "n_feature_viol": int(len(fv)), "n_gap_viol": int(len(gv)), "n_aspect_viol": int(len(av)),
            "worst_gap": float(worst if np.isfinite(worst) else -1), "min_size": float(sizes.min()),
            "worst_aspect": worst_ar, "profile": prof, "violations": viol}


