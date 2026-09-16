"""E8 — UZAK-ALAN / BSDF DIŞA AKTARIM + BASİT DRC (üretim/sistem köprüsü).

- bsdf_orders(result, lam, Lx, Ly): difraksiyon mertebelerini açı + verim (BSDF-benzeri)
  tablosuna çevirir; CSV'ye yazılabilir (sistem/ışın-optiği araçlarına devir).
- far_field(E0, dx, lam): açıklık alanından uzak-alan açısal şiddet dağılımı.
- drc_check(placements, min_feature, min_gap): üretim kuralı denetimi (min öznitelik/aralık).
"""
import numpy as np


def bsdf_orders(result, lam, Lx, Ly, csv_path=None):
    """2B çözüm sonucundan (orders=(MXf,MYf), R, T) her mertebe için
    saçılma açısı (θ,φ) + verim tablosu. csv_path verilirse CSV yazar.
    Döndürür: list[dict]."""
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
    """Açıklık alanı E0(x,y) -> uzak-alan açısal şiddet I(θx,θy) (Fraunhofer/FFT).
    Döndürür: (I_norm, theta_x_deg, theta_y_deg)."""
    E0 = np.asarray(E0, complex); N = E0.shape[0]
    F = np.fft.fftshift(np.fft.fft2(E0)); I = np.abs(F) ** 2
    fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))       # 1/um
    theta = np.degrees(np.arcsin(np.clip(fx * lam, -1, 1)))
    return I / I.max(), theta, theta


def drc_check(placements, min_feature=0.05, min_gap=0.04, shape_key="params"):
    """Basit üretim kuralı denetimi (DRC):
      - min öznitelik: her sütun boyutu >= min_feature (um)
      - min aralık: komşu sütunlar arası kenar-kenar boşluk >= min_gap (um)
    Döndürür: {ok, n_feature_viol, n_gap_viol, worst_gap, violations[list]}."""
    xs = np.array([p["x"] for p in placements]); ys = np.array([p["y"] for p in placements])
    sizes = np.array([float(np.atleast_1d(p[shape_key])[0]) for p in placements])
    viol = []
    # öznitelik
    fv = np.where(sizes < min_feature)[0]
    for i in fv:
        viol.append({"tip": "min_feature", "i": int(i), "deger": float(sizes[i])})
    # aralık: en yakın komşu (kenar-kenar). O(n^2) küçük yerleşimler için; büyükte KDTree önerilir.
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
                    viol.append({"tip": "min_gap", "i": int(i), "j": j, "gap": round(float(gap), 4)})
    return {"ok": len(fv) == 0 and gv == 0, "n_feature_viol": int(len(fv)),
            "n_gap_viol": int(gv), "worst_gap": float(worst if np.isfinite(worst) else -1),
            "violations": viol}
