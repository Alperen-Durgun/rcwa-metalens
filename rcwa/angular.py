"""Angle-dependent meta-atom library and off-axis lens analysis (roadmap P2).

The normal-incidence library assumes that a pillar transmits the same complex amplitude at
every angle; the off-axis (field-of-view) analysis then only adds the linear phase of the
tilted wave. Here every pillar size used by a lens is solved again with RCWA at oblique
incidence, and the off-axis focus is computed with that response.

Geometry and conventions
  * The field angle theta is the angle in air on the incidence side (light passes a flat
    substrate first; the transverse wave vector kx = k0 sin(theta) is conserved). Inside the
    substrate the RCWA incidence angle is asin(sin(theta) / n_sub). Plane of incidence x-z.
  * pol="TE": E along y (s). pol="TM": E in the x-z plane (p). t is the ratio of the
    transverse (tangential) field components out / in: ty for TE and tx / cos(theta_sub) for
    TM. At theta = 0 both equal the normal-incidence library value. These definitions are
    checked against an analytic thin-film transfer matrix (tests).
  * t is the zeroth-order amplitude, physics convention, the same quantity as in
    design.meta_atom_library; it does not depend on the pillar's position in the cell, so the
    locally periodic aperture field at angle theta is t(s(x, y), theta) exp(i kx x).
  * For a period L, extra diffraction orders appear in the substrate (reflected) once
    |sin(theta) - lam0/L| < n_sub and in air (transmitted) once |sin(theta) - lam0/L| < n_bg;
    their power is lost from the zeroth order and included in t. orders_at() lists them.
"""
import numpy as np

from . import cache
from . import design as dz
from .convention import to_physical
from .parallel import parallel_map
from .solver2d import solve_rcwa_2d


def substrate_angle(theta_deg, n_sub):
    """RCWA incidence angle in the substrate (deg) for a field angle theta in air."""
    s = np.sin(np.deg2rad(theta_deg)) / n_sub
    if abs(s) >= 1:
        raise ValueError(f"angle {theta_deg} deg cannot enter a substrate of index {n_sub}")
    return float(np.rad2deg(np.arcsin(s)))


def orders_at(theta_deg, lam0, L, n_sub, n_bg=1.0, m_max=3):
    """Propagating diffraction orders other than (0,0) at field angle theta (x-z plane):
    {"substrate": [(mx, my), ...] (reflected), "exit": [...] (transmitted)}."""
    sx = np.sin(np.deg2rad(theta_deg)); g = lam0 / L
    out = {"substrate": [], "exit": []}
    for mx in range(-m_max, m_max + 1):
        for my in range(-m_max, m_max + 1):
            if mx == 0 and my == 0:
                continue
            q2 = (sx - mx * g) ** 2 + (my * g) ** 2
            if q2 < n_sub ** 2:
                out["substrate"].append((mx, my))
            if q2 < n_bg ** 2:
                out["exit"].append((mx, my))
    return out


def first_order_angle(lam0, L, n):
    """Smallest field angle (deg, in air) at which a first diffraction order propagates in a
    medium of index n (None if no angle up to 90 deg)."""
    s = lam0 / L - n
    return None if s >= 1 else float(np.rad2deg(np.arcsin(max(s, 0.0))))


def _atom_oblique(job):
    """Worker: (t_raw, Ttot, Rtot) of one pillar at one angle, raw solver convention."""
    lam0, L, H, shape, s, N, n, n_bg, n_sub, M, geom, th_sub, pol = job
    sidewall_deg, n_steps, residual = geom
    layers = dz.pillar_layers(shape, s, L, H, N, n, n_bg, sidewall_deg, n_steps, residual)
    pv = (0.0, 1.0) if pol == "TE" else (1.0, 0.0)
    r = solve_rcwa_2d(lam0, th_sub, 0.0, n_sub ** 2, n_bg ** 2, L, L, layers, M, M, pol=pv)
    i0 = r["i0"]
    t = r["ty"][i0] if pol == "TE" else r["tx"][i0] / np.cos(np.deg2rad(th_sub))
    return complex(t), float(r["Ttot"]), float(r["Rtot"])


def angle_library(lib, angles_deg, sizes=None, pol="TE", M=None, n_jobs=None, use_cache=True):
    """t(size, angle) for the sizes of a library (or the given subset) at the field angles.

    lib: a design.meta_atom_library result (its geometry, material index and M are used).
    Returns dict: angles_deg, theta_sub_deg, sizes, t (n_angles, n_sizes, physics convention),
    T, R (total power transmission / reflection of the periodic array), pol, M, orders
    (propagating extra orders per angle), first_order_substrate_deg, first_order_exit_deg."""
    if pol not in ("TE", "TM"):
        raise ValueError('pol must be "TE" or "TM"')
    angles = np.atleast_1d(np.asarray(angles_deg, float))
    sizes = np.asarray(lib["sizes"] if sizes is None else sizes, float)
    M = int(lib["M"] if M is None else M)
    n = complex(lib["n"]); n_sub = float(lib["n_sub"]); n_bg = float(lib["n_bg"])
    N = int(lib.get("raster_n") or 512)
    steps = int(lib.get("n_steps") or 1); res = lib.get("residual")
    geom = (float(lib.get("sidewall_deg", 90.0)), steps, res)
    th_sub = np.array([substrate_angle(a, n_sub) for a in angles])

    def _compute():
        jobs = [(lib["lam0"], lib["L"], lib["H"], lib["shape"], float(s), N, n, n_bg, n_sub, M, geom, float(ts), pol)
                for ts in th_sub for s in sizes]
        nj = min(dz.resolve_jobs(n_jobs), len(jobs)) if jobs else 1
        out = parallel_map(_atom_oblique, jobs, n_jobs=nj)
        t = np.array([o[0] for o in out], complex).reshape(len(angles), len(sizes))
        T = np.array([o[1] for o in out]).reshape(len(angles), len(sizes))
        R = np.array([o[2] for o in out]).reshape(len(angles), len(sizes))
        return {"t_re": t.real, "t_im": t.imag, "T": T, "R": R}

    key = {"lam0": round(float(lib["lam0"]), 9), "L": round(float(lib["L"]), 9), "H": round(float(lib["H"]), 9),
           "shape": lib["shape"], "M": M, "n": (round(n.real, 5), round(n.imag, 5)), "n_sub": n_sub, "n_bg": n_bg,
           "N": N, "sizes": tuple(np.round(sizes, 6)), "angles": tuple(np.round(angles, 6)), "pol": pol,
           "geom": (round(geom[0], 4), steps, None if res is None else (round(complex(res[0]).real, 5),
                                                                        round(complex(res[0]).imag, 5), round(res[1], 6)))}
    c = cache.cached_library("anglelib", key, _compute) if use_cache else _compute()
    return {"angles_deg": angles, "theta_sub_deg": th_sub, "sizes": sizes, "pol": pol, "M": M,
            "t": to_physical(c["t_re"] + 1j * c["t_im"]), "T": np.asarray(c["T"]), "R": np.asarray(c["R"]),
            "orders": [orders_at(a, lib["lam0"], lib["L"], n_sub, n_bg) for a in angles],
            "first_order_substrate_deg": first_order_angle(lib["lam0"], lib["L"], n_sub),
            "first_order_exit_deg": first_order_angle(lib["lam0"], lib["L"], n_bg)}


def _field_at_angle(d, alib, k):
    """Aperture field of design d with the atom response of angle index k of alib."""
    ins = d["inside"]
    sz = np.round(np.asarray(alib["sizes"], float), 12)
    q = np.round(d["size_map"], 12)
    j = np.searchsorted(sz, q)
    jj = np.clip(j, 0, len(sz) - 1)
    if not np.all(sz[jj[ins]] == q[ins]):
        raise ValueError("angle library does not contain every pillar size of the design")
    return np.where(ins, alib["t"][k][jj], 0.0)


def _spot(E0, d, theta_deg, z):
    """Peak intensity, peak position x, FWHM (x cut through the peak) in the plane z."""
    from . import metrics
    ff = dz.FocalField(E0, d["L"], d["lam0"], n_bg=d["n_bg"], tilt_deg=theta_deg)
    I = ff.intensity(z); i, j = np.unravel_index(np.argmax(I), I.shape)
    return {"peak": float(I.max()), "x_peak": float(ff.x[i]), "fwhm": float(metrics.fwhm_1d(I[:, j], ff.x))}


def fov_compare(d, angles_deg, pol="TE", alib=None, n_jobs=None):
    """Off-axis spot of a lens with angle-dependent atoms ("atoms") and with the
    normal-incidence response at every angle ("fixed", the former assumption), in the plane of
    the normal-incidence focus (flat sensor). Relative peak = peak(theta) / peak(0) of the same
    model. Returns dict(angles_deg, z0, atoms, fixed, alib, mean_T) where atoms/fixed hold
    lists of rel_peak, x_peak, fwhm, distortion_pct (x_peak vs z0 tan(theta)); x_peak is
    resolved to the FFT grid step L/4."""
    angles = np.atleast_1d(np.asarray(angles_deg, float))
    if alib is None:
        used = np.unique(np.round(d["size_map"][d["inside"]], 12))
        alib = angle_library(d["lib"], angles, sizes=used, pol=pol, n_jobs=n_jobs)
    z0 = dz.FocalField(d["E0"], d["L"], d["lam0"], n_bg=d["n_bg"]).find_focus(d["f"], nz=81)[0]
    out = {"angles_deg": angles, "z0": float(z0), "atoms": {}, "fixed": {}, "alib": alib}
    base = {}
    for name in ("atoms", "fixed"):
        rows = []
        for k, a in enumerate(angles):
            E0 = _field_at_angle(d, alib, k) if name == "atoms" else d["E0"]
            rows.append(_spot(E0, d, a, z0))
        base[name] = rows
    E00 = d["E0"]
    p0 = _spot(E00, d, 0.0, z0)["peak"]
    for name, rows in base.items():
        ideal = z0 * np.tan(np.deg2rad(angles))              # chief ray on the sensor plane z0
        out[name] = {"rel_peak": [r["peak"] / p0 for r in rows], "x_peak": [r["x_peak"] for r in rows],
                     "fwhm": [r["fwhm"] for r in rows],
                     "distortion_pct": [100 * (r["x_peak"] - xi) / xi if xi > 1e-9 else 0.0 for r, xi in zip(rows, ideal)]}
    ins = d["inside"]; sz = np.round(np.asarray(alib["sizes"], float), 12)
    jj = np.searchsorted(sz, np.round(d["size_map"][ins], 12))
    out["mean_T"] = [float(np.mean(alib["T"][k][jj])) for k in range(len(angles))]
    return out
