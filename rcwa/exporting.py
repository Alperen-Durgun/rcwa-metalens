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


def drc_check(placements, min_feature=0.05, min_gap=0.04, shape_key="params"):
    """Simple design-rule check (DRC):
      - min feature: every pillar size >= min_feature (um)
      - min gap: edge-to-edge spacing between neighboring pillars >= min_gap (um)
    Returns: {ok, n_feature_viol, n_gap_viol, worst_gap, violations[list]}."""
    xs = np.array([p["x"] for p in placements]); ys = np.array([p["y"] for p in placements])
    sizes = np.array([float(np.atleast_1d(p[shape_key])[0]) for p in placements])
    viol = []
    # feature size
    fv = np.where(sizes < min_feature)[0]
    for i in fv:
        viol.append({"type": "min_feature", "i": int(i), "value": float(sizes[i])})
    # gap: nearest neighbor (edge-to-edge). O(n^2) for small layouts; a KDTree is recommended for large ones.
    n = len(placements); worst = np.inf; gv = 0
    if n <= 4000:
        for i in range(n):
            d = np.hypot(xs - xs[i], ys - ys[i]); d[i] = np.inf
            j = int(np.argmin(d))
            gap = d[j] - (sizes[i] + sizes[j]) / 2.0
            worst = min(worst, gap)
            if gap < min_gap:
                gv += 1
                if len(viol) < 50:
                    viol.append({"type": "min_gap", "i": int(i), "j": j, "gap": round(float(gap), 4)})
    return {"ok": len(fv) == 0 and gv == 0, "n_feature_viol": int(len(fv)),
            "n_gap_viol": int(gv), "worst_gap": float(worst if np.isfinite(worst) else -1),
            "violations": viol}
