"""Shared metalens design core.

One implementation of the steps every lens designer in this package needs:

    meta_atom_library  ->  lens_phase / lens_grid  ->  select_atoms  ->  FocalField

Everything here works in the PHYSICS convention exp(-i*omega*t) (see ``convention.py``):
library transmission coefficients are converted once, right after the RCWA solve, so the
hyperbolic lens phase, the angular-spectrum propagator and FDTD tools can be combined with
them directly.
"""
import os
import numpy as np
from . import materials, shapes, cache
from .parallel import parallel_map
from .solver2d import solve_rcwa_2d
from .convention import to_physical
from . import metrics


# --------------------------------------------------------------------------- materials
def pillar_index(material=None, lam0=None, n_pillar=2.4):
    """Complex refractive index of the pillar: from the material library if a name is
    given (absorption included), otherwise the constant ``n_pillar``."""
    if material:
        return complex(np.sqrt(complex(materials.eps_of(material, lam0))))
    return complex(n_pillar)


def default_sizes(L, n=20, s_min=0.06, margin=0.06):
    """Standard size sweep: from s_min up to (L - margin), n samples (um).
    margin = smallest gap between neighbouring pillars. 0.06 since 0.3.0: at a 50 nm gap
    (the old default) the 633 nm TiO2 library sits on the shoulder of a gap resonance at
    ~302 nm that M = 7 does not resolve (0.16 rad phase error against M = 15)."""
    return np.linspace(s_min, L - margin, int(n))


# --------------------------------------------------------------------------- library
def resolve_jobs(n_jobs=None):
    """Worker count for library solves: n_jobs if given, else the RCWA_N_JOBS environment
    variable, else 1 (serial). 0 or a negative value means all cores. On Windows a script that
    uses n_jobs > 1 must keep its top-level code under ``if __name__ == "__main__":``."""
    if n_jobs is None:
        try:
            n_jobs = int(os.environ.get("RCWA_N_JOBS", "1") or 1)
        except ValueError:
            n_jobs = 1
    n_jobs = int(n_jobs)
    return (os.cpu_count() or 1) if n_jobs <= 0 else n_jobs


def pillar_layers(shape, s, L, H, N, n, n_bg, sidewall_deg=90.0, n_steps=1, residual=None):
    """Layer stack of one pillar, from the substrate side up: an optional uniform residual
    film (residual = (complex index, thickness um)), then the pillar. sidewall_deg < 90 makes
    a tapered pillar (wider at the bottom, the usual etch profile), approximated by n_steps
    slices whose width changes linearly; `s` is the width at mid-height. Width change over the
    full height: 2 H / tan(sidewall). Returns [(eps2d, thickness), ...]."""
    layers = []
    if residual is not None and residual[1] > 0:
        layers.append((np.full((N, N), complex(residual[0]) ** 2), float(residual[1])))
    steps = max(1, int(n_steps))
    taper = 0.0 if sidewall_deg >= 90 else 2 * H / np.tan(np.deg2rad(sidewall_deg))
    for i in range(steps):
        frac = (i + 0.5) / steps                                  # 0 = bottom, 1 = top
        w = max(float(s) + (0.5 - frac) * taper, 0.0)
        layers.append((shapes.rasterize(shape, w, L, N, n_pillar=n, n_bg=n_bg), H / steps))
    return layers


def _atom_t(job):
    """Top-level (picklable) worker: zeroth-order co-polarized t of one pillar, raw solver
    convention. job = (lam0, L, H, shape, size, raster_n, n_pillar, n_bg, n_sub, M, geometry)
    with geometry = (sidewall_deg, n_steps, residual) (see pillar_layers)."""
    lam0, L, H, shape, s, N, n, n_bg, n_sub, M = job[:10]
    sidewall_deg, n_steps, residual = job[10] if len(job) > 10 else (90.0, 1, None)
    layers = pillar_layers(shape, s, L, H, N, n, n_bg, sidewall_deg, n_steps, residual)
    r = solve_rcwa_2d(lam0, 0, 0, n_sub ** 2, n_bg ** 2, L, L, layers, M, M, pol=(0., 1.))
    return complex(r["ty"][r["i0"]])


def meta_atom_library(lam0, L, H, sizes=None, shape="circle", material=None, n_pillar=2.4,
                      M=9, n_sub=1.0, n_bg=1.0, raster_n=None, use_cache=True, n_jobs=None,
                      sidewall_deg=90.0, n_steps=None, residual=None):
    """Transmission library of a pillar on a square lattice versus its size.

    Light is incident from the substrate (index n_sub) and exits into the background
    (index n_bg); the zeroth-order co-polarized transmission coefficient is returned.

    Returns dict:
      sizes    : (K,) sizes (um)
      t        : (K,) complex transmission, PHYSICS convention (phase grows with delay)
      phase    : (K,) unwrapped phase of t, shifted so that min = 0
      amp      : (K,) |t|   (field amplitude; power transmission into order 0 is
                 |t|^2 * n_bg / n_sub)
      coverage : phase span in units of 2*pi (>= 1 means a full 2*pi library)
      n        : complex pillar index used
    n_jobs: worker processes for the solves (see resolve_jobs); results do not depend on it.
    Fabrication geometry (roadmap M4, P5): sidewall_deg < 90 tapers the pillar (n_steps slices,
    default 5; the size is the mid-height width); residual = (material name or index,
    thickness um) adds a uniform film between substrate and pillars (incomplete etch).
    """
    sizes = default_sizes(L) if sizes is None else np.asarray(sizes, float)
    n = pillar_index(material, lam0, n_pillar)
    N = int(raster_n or shapes.DEFAULT_N)

    steps = int(n_steps) if n_steps else (5 if sidewall_deg < 90 else 1)
    res = None
    if residual is not None:
        rn = residual[0]
        rn = complex(np.sqrt(complex(materials.eps_of(rn, lam0)))) if isinstance(rn, str) else complex(rn)
        res = (rn, float(residual[1]))
    geom = (float(sidewall_deg), steps, res)

    def _compute():
        jobs = [(lam0, L, H, shape, float(s), N, n, n_bg, n_sub, int(M), geom) for s in sizes]
        nj = min(resolve_jobs(n_jobs), len(jobs)) if jobs else 1
        t = np.array(parallel_map(_atom_t, jobs, n_jobs=nj), complex).reshape(len(sizes))
        return {"t_re": t.real, "t_im": t.imag}

    key = {"lam0": round(float(lam0), 9), "L": round(float(L), 9), "H": round(float(H), 9),
           "shape": shape, "M": int(M), "n": (round(n.real, 5), round(n.imag, 5)),
           "n_sub": float(n_sub), "n_bg": float(n_bg),
           "N": N, "sizes": tuple(np.round(sizes, 6))}
    if sidewall_deg < 90 or steps != 1:                  # only non-default geometry changes the key
        key["sidewall"] = (round(float(sidewall_deg), 4), steps)
    if res is not None and res[1] > 0:
        key["residual"] = (round(res[0].real, 5), round(res[0].imag, 5), round(res[1], 6))
    c = cache.cached_library("atomlib", key, _compute) if use_cache else _compute()
    t = to_physical(c["t_re"] + 1j * c["t_im"])
    ph = np.unwrap(np.angle(t)); ph = ph - ph.min()
    return {"sizes": sizes, "t": t, "phase": ph, "amp": np.abs(t),
            "coverage": float(np.ptp(ph) / (2 * np.pi)), "n": n,
            "shape": shape, "L": L, "H": H, "lam0": lam0, "M": M, "n_sub": n_sub, "n_bg": n_bg,
            "raster_n": N, "sidewall_deg": float(sidewall_deg), "n_steps": steps, "residual": res}


def _library_dict(sizes, t, base):
    ph = np.unwrap(np.angle(t)); ph = ph - ph.min()
    out = {k: base[k] for k in ("n", "shape", "L", "H", "lam0", "M", "n_sub", "n_bg", "raster_n",
                                "sidewall_deg", "n_steps", "residual") if k in base}
    out.update({"sizes": sizes, "t": t, "phase": ph, "amp": np.abs(t),
                "coverage": float(np.ptp(ph) / (2 * np.pi))})
    return out


def find_resonances(sizes, t, prominence=0.2):
    """Transmission dips of a (dense) library: local minima of |t|/max|t| with at least the
    given prominence. For a lossless sub-wavelength lattice (one propagating order) a guided-mode
    resonance drives |t| toward zero, so a deep |t| dip is the signature that matters.
    Returns a list of dicts: center, lo, hi (half-depth crossings, um), width (hi - lo),
    t_min (|t|/max at the centre), prominence."""
    from scipy.signal import find_peaks, peak_widths
    sizes = np.asarray(sizes, float); a = np.abs(np.asarray(t, complex))
    if len(a) < 3:
        return []
    a = a / a.max()
    pk, pr = find_peaks(-a, prominence=prominence)
    if len(pk) == 0:
        return []
    w = peak_widths(-a, pk, rel_height=0.5, prominence_data=(pr["prominences"], pr["left_bases"],
                                                             pr["right_bases"]))
    ix = np.arange(len(sizes))
    lo = np.interp(w[2], ix, sizes); hi = np.interp(w[3], ix, sizes)
    return [{"center": float(sizes[k]), "lo": float(l), "hi": float(h), "width": float(h - l),
             "t_min": float(a[k]), "prominence": float(p)}
            for k, l, h, p in zip(pk, lo, hi, pr["prominences"])]


def refine_library(lib, tol=0.05, min_step=0.5e-3, max_solves=80, prominence=0.2,
                   use_cache=True, n_jobs=None):
    """Adaptive refinement of a meta-atom library (resonance-aware sampling).

    Every interval between neighbouring sizes is checked at its midpoint: if the solved t
    differs from the complex linear interpolation by more than tol (relative to max|t|), the
    interval is split and both halves are checked again, down to min_step (um). Narrow
    resonances that a uniform sweep steps over are resolved this way. Limitation: a resonance
    much narrower than an interval that leaves no trace at the interval's ends or midpoint can
    still be missed; a finer starting sweep is the remedy.

    Returns a library dict (same keys as meta_atom_library) on the merged, sorted size grid,
    plus: resonances (see find_resonances), refined=True, n_refine_solves."""
    sizes = list(np.asarray(lib["sizes"], float)); t = list(np.asarray(lib["t"], complex))
    pts = {round(s, 9): v for s, v in zip(sizes, t)}
    A = float(np.max(np.abs(t)))
    kw = dict(lam0=lib["lam0"], L=lib["L"], H=lib["H"], shape=lib["shape"], material=None,
              n_pillar=lib["n"], M=lib["M"], n_sub=lib["n_sub"], n_bg=lib["n_bg"],
              raster_n=lib.get("raster_n"), use_cache=use_cache, n_jobs=n_jobs,
              sidewall_deg=lib.get("sidewall_deg", 90.0), n_steps=lib.get("n_steps"),
              residual=lib.get("residual"))
    ss = sorted(pts)
    todo = [(a, b) for a, b in zip(ss[:-1], ss[1:]) if b - a >= 2 * min_step]
    n_solved = 0
    while todo and n_solved < max_solves:
        todo = todo[:max_solves - n_solved]
        mids = np.array([(a + b) / 2 for a, b in todo])
        tm = meta_atom_library(sizes=mids, **kw)["t"]; n_solved += len(mids)
        nxt = []
        for (a, b), m, v in zip(todo, mids, tm):
            m = round(float(m), 9); pts[m] = v
            if abs(v - (pts[a] + pts[b]) / 2) / A > tol and (b - a) / 2 >= 2 * min_step:
                nxt += [(a, m), (m, b)]
        todo = nxt
    ss = np.array(sorted(pts)); tt = np.array([pts[k] for k in ss])
    out = _library_dict(ss, tt, lib)
    out.update({"resonances": find_resonances(ss, tt, prominence=prominence), "refined": True,
                "n_refine_solves": n_solved, "refine_tol": tol,
                "refine_converged": not todo})
    return out


def resonance_mask(lib, guard_nm=0.0):
    """Boolean mask of library atoms that are NOT within guard_nm of a detected resonance
    (the resonance band is [lo, hi] from find_resonances). Requires lib["resonances"]."""
    s = np.asarray(lib["sizes"], float); ok = np.ones(len(s), bool)
    g = float(guard_nm) * 1e-3
    for r in lib.get("resonances", []):
        ok &= ~((s >= r["lo"] - g) & (s <= r["hi"] + g))
    return ok


def max_phase_gap(t_lib, allowed=None):
    """Largest empty arc (rad) of the transmission phases on the unit circle: 0 for a dense
    full-2pi library, 2pi - span for a library that does not close the circle."""
    ph = np.sort(np.mod(np.angle(np.asarray(t_lib, complex)), 2 * np.pi))
    if allowed is not None:
        ph = np.sort(np.mod(np.angle(np.asarray(t_lib, complex)[np.asarray(allowed, bool)]),
                            2 * np.pi))
    if len(ph) == 0:
        return 2 * np.pi
    return float(np.max(np.diff(np.r_[ph, ph[0] + 2 * np.pi])))



def certify_library(lib, dM=2, tol_phase=0.02, tol_amp=0.01, tol_res=0.05, sizes=None,
                    use_cache=True, n_jobs=None):
    """Convergence certificate of a library in the Fourier order count (roadmap N1).

    Every size (or the given subset) is solved again at M + dM; the phase and |t| changes are
    the certificate. Energy conservation, which solve_2d_auto checks, is necessary but not
    sufficient: an under-resolved solve can conserve energy and still have the wrong phase.
    Sizes inside a detected resonance band (lib["resonances"], if any) use the looser phase
    tolerance tol_res, because the phase there changes by radians per nanometre.

    Returns a copy of lib with lib["convergence"] = {
      M, M_check, sizes, d_phase (rad), d_amp, in_resonance (bool per size),
      max_d_phase, max_d_amp (outside resonances), max_d_phase_resonance,
      tol_phase, tol_amp, tol_res, ok (per size), passed }.
    The M vs M+dM change is an estimate of the remaining error, not a bound: where the
    convergence is slow (e.g. next to a gap resonance of nearly touching pillars) the true error
    can be several times larger. Use dM=4 for a stricter check."""
    s_all = np.asarray(lib["sizes"], float); t_all = np.asarray(lib["t"], complex)
    if sizes is None:
        sel = np.arange(len(s_all))
    else:
        want = np.round(np.asarray(sizes, float), 9)
        sel = np.nonzero(np.isin(np.round(s_all, 9), want))[0]
        if len(sel) != len(np.unique(want)):
            raise ValueError("certify_library: sizes must be a subset of lib['sizes']")
    s = s_all[sel]; t = t_all[sel]
    M2 = int(lib["M"]) + int(dM)
    hi = meta_atom_library(lib["lam0"], lib["L"], lib["H"], sizes=s, shape=lib["shape"],
                           material=None, n_pillar=lib["n"], M=M2, n_sub=lib["n_sub"],
                           n_bg=lib["n_bg"], raster_n=lib.get("raster_n"), use_cache=use_cache,
                           n_jobs=n_jobs, sidewall_deg=lib.get("sidewall_deg", 90.0),
                           n_steps=lib.get("n_steps"), residual=lib.get("residual"))["t"]
    dphi = np.abs(np.angle(t * np.conj(hi)))
    damp = np.abs(np.abs(t) - np.abs(hi))
    inres = np.zeros(len(s), bool)
    for r in lib.get("resonances", []):
        inres |= (s >= r["lo"]) & (s <= r["hi"])
    out_ok = ~inres
    ok = np.where(inres, dphi < tol_res, (dphi < tol_phase) & (damp < tol_amp))
    mx = lambda a, m: float(a[m].max()) if m.any() else 0.0          # noqa: E731
    cert = {"M": int(lib["M"]), "M_check": M2, "sizes": s, "d_phase": dphi, "d_amp": damp,
            "in_resonance": inres, "max_d_phase": mx(dphi, out_ok), "max_d_amp": mx(damp, out_ok),
            "max_d_phase_resonance": mx(dphi, inres), "tol_phase": tol_phase, "tol_amp": tol_amp,
            "tol_res": tol_res, "ok": ok}
    cert["passed"] = bool(cert["max_d_phase"] < tol_phase and cert["max_d_amp"] < tol_amp
                          and cert["max_d_phase_resonance"] < tol_res)
    out = dict(lib); out["convergence"] = cert
    return out

# --------------------------------------------------------------------------- targets
def lens_phase(r, lam0, f):
    """Hyperbolic (aberration-free on axis) lens phase, physics convention, wrapped to [0, 2pi)."""
    return np.mod(-(2 * np.pi / lam0) * (np.sqrt(np.asarray(r, float) ** 2 + f ** 2) - f), 2 * np.pi)


def lens_grid(D_lens, L):
    """Cell centres of a square lattice covering a circular aperture.
    Returns (ci, R, inside, D_actual): 1D centre coordinates, radius map, aperture mask."""
    Nc = max(1, int(round(D_lens / L))); D = Nc * L
    ci = (np.arange(Nc) - (Nc - 1) / 2) * L
    XX, YY = np.meshgrid(ci, ci, indexing="ij"); R = np.hypot(XX, YY)
    return ci, R, R <= D / 2 + 1e-9, D


# --------------------------------------------------------------------------- selection
def size_error_stats(lib, sigma_nm, bias_nm=0.0, nq=9):
    """Mean and variance of each atom's transmission under a size error delta ~ N(bias, sigma^2)
    (nm), evaluated by Gauss-Hermite quadrature on the complex-linear interpolation of the
    library. No extra solves; the library must be dense where t varies fast, i.e. refined
    (refine_library) near resonances. Returns (mean, var), arrays of len(lib["sizes"])."""
    s = np.asarray(lib["sizes"], float); t = np.asarray(lib["t"], complex)
    x, w = np.polynomial.hermite_e.hermegauss(int(nq)); w = w / w.sum()
    d = (float(bias_nm) + float(sigma_nm) * x) * 1e-3
    q = s[:, None] + d[None, :]
    tq = np.interp(q, s, t.real) + 1j * np.interp(q, s, t.imag)
    mean = tq @ w
    var = (np.abs(tq - mean[:, None]) ** 2) @ w
    return mean, var


def phase_lut(t_lib, nbin=720, mode="complex", allowed=None, t_stats=None):
    """Look-up table: for each target phase bin, the best library atom.

    mode="complex" (default): minimizes |t_k/|t|max - exp(i*phi)| — an atom with the right
    phase but poor transmission (a resonance dip) loses against a slightly detuned bright one.
    mode="phase": nearest phase only (classic rule).
    mode="robust": minimizes the EXPECTED squared error under a size error,
    |E[t_k]/A - exp(i*phi)|^2 + Var[t_k]/A^2, with t_stats = (mean, var) from size_error_stats.
    allowed: optional boolean mask of usable atoms (e.g. resonance_mask).
    Returns (lut_k, phases): atom index per bin and the bin-centre phases."""
    t = np.asarray(t_lib, complex)
    ok = np.ones(len(t), bool) if allowed is None else np.asarray(allowed, bool)
    if not ok.any():
        raise ValueError("phase_lut: no allowed atoms")
    A = np.abs(t[ok]).max()
    phases = (np.arange(nbin) + 0.5) * 2 * np.pi / nbin
    if mode == "phase":
        d = np.abs(np.angle(np.exp(1j * (np.angle(t)[None, :] - phases[:, None]))))
    elif mode == "robust":
        if t_stats is None:
            raise ValueError('mode="robust" needs t_stats=(mean, var), see size_error_stats')
        mean, var = (np.asarray(v) for v in t_stats)
        d = np.abs(mean[None, :] / A - np.exp(1j * phases)[:, None]) ** 2 + var[None, :] / A ** 2
    else:
        d = np.abs(t[None, :] / A - np.exp(1j * phases)[:, None])
    d = np.where(ok[None, :], d, np.inf)
    return np.argmin(d, axis=1), phases


def select_atoms(t_lib, phi_target, mask=None, mode="complex", nbin=720, n_offset=24,
                 allowed=None, t_stats=None):
    """Assign a library atom to every cell (vectorized, no Python loop over cells).

    A lens tolerates any global phase offset, so n_offset offsets are tried and the one that
    maximizes the coherent sum toward the target (i.e. the on-axis focal amplitude) is kept.
    allowed / t_stats: see phase_lut (resonance guard band, robust selection).
    Returns (idx, offset): atom index per cell (-1 outside the mask) and the chosen offset."""
    t = np.asarray(t_lib, complex); phi = np.asarray(phi_target, float)
    mask = np.ones(phi.shape, bool) if mask is None else np.asarray(mask, bool)
    lut_k, _ = phase_lut(t, nbin=nbin, mode=mode, allowed=allowed, t_stats=t_stats)
    best = (-np.inf, None, 0.0)
    pm = phi[mask]
    for off in np.arange(max(1, int(n_offset))) * 2 * np.pi / max(1, int(n_offset)):
        b = (np.mod(pm + off, 2 * np.pi) / (2 * np.pi) * nbin).astype(int) % nbin
        k = lut_k[b]
        score = float(np.real(np.sum(t[k] * np.exp(-1j * (pm + off)))))
        if score > best[0]:
            best = (score, k, off)
    idx = -np.ones(phi.shape, int); idx[mask] = best[1]
    return idx, float(best[2])


def aperture_field(t_lib, idx):
    """Complex aperture field (one sample per cell) for an index map (0 outside)."""
    t = np.asarray(t_lib, complex)
    return np.where(idx >= 0, t[np.clip(idx, 0, None)], 0.0)


# --------------------------------------------------------------------------- propagation
FFT_PAD = 8.0          # default zero-padding factor of the FFT window (plane fields)
FFT_MAX_N = 2048       # cap on the FFT window size (samples per side)


class FocalField:
    """Free-space propagation of a cell-sampled aperture field (physics convention).

    E0   : (Nc, Nc) complex field, one sample per lattice cell (0 outside the aperture)
    L    : lattice period (um);  lam0: wavelength (um);  n_bg: index behind the lens
    u    : sub-samples per cell
    pad  : zero-padding factor of the FFT window used for plane fields (at / intensity);
           None = FFT_PAD, reduced if the window would exceed FFT_MAX_N samples (pad_limited)
    tilt_deg : incidence tilt in the x-z plane (adds a linear phase to the aperture)

    On-axis values (on_axis, find_focus, Strehl) use the Rayleigh-Sommerfeld integral of the
    aperture samples: exact for this aperture and independent of any window. The FFT angular
    spectrum is periodic: light leaving the aperture at steep angles re-enters from the
    neighbouring periodic images, and with a window of 1.6 x the aperture (the old default)
    this moved the on-axis focus of an NA 0.45 lens by up to 5 % (validation V2). Plane
    metrics (FWHM, efficiencies) converge to < 1 % at pad >= 8.
    """

    def __init__(self, E0, L, lam0, n_bg=1.0, u=4, pad=None, tilt_deg=0.0):
        E0 = np.asarray(E0, complex); Nc = E0.shape[0]
        self.lam0 = lam0; self.k = 2 * np.pi / lam0 * n_bg; self.dx = L / u
        Ef = np.kron(E0, np.ones((u, u))); Ncf = Nc * u
        self.pad_limited = False
        if pad is None:
            pad = FFT_PAD
            if Ncf * pad > FFT_MAX_N:
                pad = max(2.0, FFT_MAX_N / Ncf); self.pad_limited = True
                import warnings
                warnings.warn(f"FocalField: FFT window limited to pad = {pad:.2f} (aperture "
                              f"{Ncf} samples); plane metrics may carry window errors of a few "
                              "% - on-axis values are exact", stacklevel=2)
        self.pad = float(pad)
        N = int(Ncf * pad); N += N % 2                     # even window
        o = (N - Ncf) // 2
        self.N = N; self._o = o; self._Ncf = Ncf
        self.x = (np.arange(N) - o - Ncf / 2 + 0.5) * self.dx      # aperture centred on x = 0
        xa = self.x[o:o + Ncf]
        if tilt_deg:
            Ef = Ef * np.exp(1j * self.k * np.sin(np.deg2rad(tilt_deg)) * xa[:, None])
        self._Ef = Ef
        self.P_incident = float(np.sum(np.abs(E0) > 0)) * u * u    # unit-amplitude illumination
        self.P_aperture = float(np.sum(np.abs(Ef) ** 2))
        nz_i, nz_j = np.nonzero(Ef)                                  # Rayleigh-Sommerfeld samples
        self._rs_U = Ef[nz_i, nz_j]; self._rs_rho2 = xa[nz_i] ** 2 + xa[nz_j] ** 2
        self._A = None; self._KZ = None

    def _spectrum(self):
        if self._A is None:
            N, o, Ncf = self.N, self._o, self._Ncf
            Ep = np.zeros((N, N), complex); Ep[o:o + Ncf, o:o + Ncf] = self._Ef
            kx = 2 * np.pi * np.fft.fftfreq(N, d=self.dx); KX, KY = np.meshgrid(kx, kx, indexing="ij")
            KZ = np.sqrt((self.k ** 2 - KX ** 2 - KY ** 2).astype(complex))
            self._KZ = np.where(np.imag(KZ) < 0, -KZ, KZ)
            self._A = np.fft.fft2(Ep)
        return self._A, self._KZ

    @property
    def A(self):
        return self._spectrum()[0]

    @property
    def KZ(self):
        return self._spectrum()[1]

    def at(self, z):
        """Complex field in the plane z (N x N), FFT angular spectrum (window: pad)."""
        A, KZ = self._spectrum()
        return np.fft.ifft2(A * np.exp(1j * KZ * z))

    def intensity(self, z):
        return np.abs(self.at(z)) ** 2

    def on_axis_field(self, z):
        """Complex on-axis field at one or many z: Rayleigh-Sommerfeld (first kind) sum over
        the aperture samples, U(0,0,z) = (1/2pi) sum U0 (z/r)(1/r - ik) exp(ikr)/r dA."""
        zz = np.atleast_1d(np.asarray(z, float)); dA = self.dx * self.dx; k = self.k
        out = np.empty(len(zz), complex)
        for n, zi in enumerate(zz):
            r = np.sqrt(self._rs_rho2 + zi * zi)
            out[n] = np.sum(self._rs_U * (zi / r) * (1 / r - 1j * k) * np.exp(1j * k * r) / r)
        out *= dA / (2 * np.pi)
        return out if np.ndim(z) else complex(out[0])

    def on_axis(self, z):
        """On-axis intensity at one or many z (exact, no window; see on_axis_field)."""
        return np.abs(self.on_axis_field(z)) ** 2

    def find_focus(self, f, zwin=(0.4, 1.8), nz=81, off_axis=False):
        """Best-focus search around the design focal length f.
        Returns (z_focus, zc, I_axis): refined focus position and the coarse on-axis scan.
        off_axis=True tracks the global intensity maximum of each plane (tilted incidence)."""
        zc = np.linspace(zwin[0] * f, zwin[1] * f, int(nz))
        if off_axis:
            Iz = np.array([self.intensity(z).max() for z in zc])
        else:
            Iz = self.on_axis(zc)
        i = int(np.argmax(Iz)); zf = float(zc[i])
        if 0 < i < len(zc) - 1:                                     # parabolic refinement
            a, b, c = Iz[i - 1], Iz[i], Iz[i + 1]; den = a - 2 * b + c
            if den < 0:
                zf = float(zc[i] + 0.5 * (a - c) / den * (zc[1] - zc[0]))
        return zf, zc, Iz


def ideal_aperture(E0, L, lam0, zf, n_bg=1.0):
    """Aberration-free reference: same aperture amplitudes, exact hyperbolic phase for focus zf."""
    E0 = np.asarray(E0, complex); Nc = E0.shape[0]
    ci = (np.arange(Nc) - (Nc - 1) / 2) * L
    XX, YY = np.meshgrid(ci, ci, indexing="ij")
    return np.abs(E0) * np.exp(1j * lens_phase(np.hypot(XX, YY), lam0 / n_bg, zf))


def strehl_at_focus(E0, L, lam0, zf, ff=None, n_bg=1.0, u=4, pad=None):
    """Strehl ratio referenced to the SAME plane: on-axis intensity of the design at its own
    focus zf divided by that of an aberration-free lens with the same aperture amplitudes
    focused at zf. By construction <= 1 (up to sampling), also for low Fresnel numbers where
    the intensity maximum is shifted away from the geometric focus."""
    ideal = ideal_aperture(E0, L, lam0, zf, n_bg=n_bg)
    ff = FocalField(E0, L, lam0, n_bg=n_bg, u=u, pad=pad) if ff is None else ff
    ref = FocalField(ideal, L, lam0, n_bg=n_bg, u=u, pad=pad)
    return float(ff.on_axis(zf) / ref.on_axis(zf))


# --------------------------------------------------------------------------- one-call designer
def design_lens(lam0, L, H, D_lens, f, shape="circle", material=None, n_pillar=2.4, M=9,
                n_sub=1.0, n_bg=1.0, sizes=None, mode="robust", lib=None, refine=True,
                guard_nm=None, sigma_nm=3.0, max_gap=np.pi / 4, certify=False,
                sidewall_deg=90.0, n_steps=None, residual=None):
    """Design a singlet metalens: library -> hyperbolic phase -> atom selection.

    Defaults since 0.3.0: M = 9, refined library, robust selection for a 3 nm size error
    (measured: Strehl 0.963 -> 0.989 and absolute efficiency 0.816 -> 0.830 for the 16 um
    TiO2 test lens, and a far smaller loss under size bias). The 0.2 behaviour is
    mode="complex", refine=False, M=7 (with sizes=default_sizes(L, margin=0.05)).

    Resonance-aware options:
      refine   : refine the library adaptively (refine_library) and detect resonances
      guard_nm : exclude atoms within guard_nm of a detected resonance (needs a refined library)
      sigma_nm : with mode="robust", the standard deviation of the expected size error; atoms
                 are chosen by expected error over that distribution (size_error_stats)
      max_gap  : warn if the allowed atoms leave a phase gap larger than this (rad)
      certify  : run certify_library (M vs M+2) and use only sizes that pass it

    Returns dict with: lib, ci, R, inside, D_lens (snapped to the lattice), NA, idx (atom index
    map), size_map (um, 0 outside), E0 (aperture field, physics convention), phi_target,
    offset, placements (list for GDS / full-wave bridges)."""
    lib = lib or meta_atom_library(lam0, L, H, sizes=sizes, shape=shape, material=material,
                                   n_pillar=n_pillar, M=M, n_sub=n_sub, n_bg=n_bg,
                                   sidewall_deg=sidewall_deg, n_steps=n_steps, residual=residual)
    if (refine or guard_nm is not None) and not lib.get("refined"):
        lib = refine_library(lib)
    allowed = resonance_mask(lib, guard_nm) if guard_nm is not None else None
    if certify:
        if "convergence" not in lib or len(lib["convergence"]["ok"]) != len(lib["sizes"]):
            lib = certify_library(lib)
        ok = np.asarray(lib["convergence"]["ok"], bool)
        allowed = ok if allowed is None else (allowed & ok)
        if not allowed.any():
            raise ValueError("design_lens: no library size passes the convergence certificate "
                             f"(M = {lib['M']} vs {lib['convergence']['M_check']}); increase M")
    gap = max_phase_gap(lib["t"], allowed)
    if gap > max_gap:
        import warnings
        warnings.warn(f"design_lens: allowed atoms leave a phase gap of {gap:.2f} rad "
                      f"(> {max_gap:.2f}); the library does not cover 2*pi well", stacklevel=2)
    t_stats = None
    if mode == "robust":
        if sigma_nm is None:
            raise ValueError('mode="robust" needs sigma_nm')
        if not lib.get("refined"):
            import warnings
            warnings.warn("robust selection on an unrefined library: the interpolated size "
                          "response may miss resonances (use refine=True)", stacklevel=2)
        t_stats = size_error_stats(lib, sigma_nm)
    ci, R, inside, D = lens_grid(D_lens, L)
    phi = lens_phase(R, lam0 / n_bg, f)
    idx, off = select_atoms(lib["t"], phi, inside, mode=mode, allowed=allowed, t_stats=t_stats)
    E0 = aperture_field(lib["t"], idx)
    size_map = np.where(idx >= 0, lib["sizes"][np.clip(idx, 0, None)], 0.0)
    ii, jj = np.nonzero(inside)
    placements = [{"x": float(ci[i]), "y": float(ci[j]), "shape": shape,
                   "params": float(size_map[i, j]), "angle": 0.0} for i, j in zip(ii, jj)]
    return {"lib": lib, "ci": ci, "R": R, "inside": inside, "D_lens": D, "f": f, "lam0": lam0,
            "L": L, "H": H, "shape": shape, "NA": float(n_bg * np.sin(np.arctan(D / 2 / f))),
            "idx": idx, "size_map": size_map, "E0": E0, "phi_target": phi, "offset": off,
            "placements": placements, "n_bg": n_bg, "n_sub": n_sub, "allowed": allowed,
            "phase_gap": gap, "mode": mode, "sigma_nm": sigma_nm, "guard_nm": guard_nm}


def analyze_focus(design, u=4, pad=None, nz=81):
    """Focal analysis of a design_lens() result with the standard metric set.
    Returns dict: z_focus, fwhm (um), strehl, efficiency suite, encircled-energy radius,
    sidelobe, depth of focus, plus the fields needed for plotting (I, x, zc, I_axis)."""
    lam0, f, L, n_bg = design["lam0"], design["f"], design["L"], design["n_bg"]
    ff = FocalField(design["E0"], L, lam0, n_bg=n_bg, u=u, pad=pad)
    zf, zc, Iz = ff.find_focus(f, nz=nz)
    I = ff.intensity(zf); x = ff.x
    met = metrics.focal_metrics(I, x); fwhm = met["fwhm_x"]
    # incident power in the substrate vs transmitted power behind the lens: P ~ n * |E|^2
    es = metrics.efficiency_suite(I, x, fwhm, P_incident=ff.P_incident * design["n_sub"] / n_bg)
    dl = lam0 / (2 * design["NA"])
    return {"z_focus": zf, "fwhm": fwhm, "diffraction_fwhm": dl, "peak_ij": met["peak_ij"],
            "strehl": strehl_at_focus(design["E0"], L, lam0, zf, ff=ff, n_bg=n_bg, u=u, pad=pad),
            "eff_absolute": es["absolute"], "eff_relative": es["relative"], "eff_legacy": es["legacy"],
            "ee80": metrics.encircled_energy(I, x, 0.8),
            "sidelobe_dB": metrics.sidelobe_level(I[:, met["peak_ij"][1]])["dB"],
            "dof": metrics.depth_of_focus(Iz, zc),
            "fresnel_number": (design["D_lens"] / 2) ** 2 / (lam0 / n_bg * f),
            "I": I, "x": x, "zc": zc, "I_axis": Iz, "field": ff}


# --------------------------------------------------------------------------- tolerance
def perturbed_field(design, lib_dense, delta):
    """Aperture field of a design whose pillar sizes are changed by delta (um; scalar or a map
    like design["size_map"]). t(s + delta) is the complex-linear interpolation of lib_dense,
    which should be refined (refine_library) so that resonances are resolved.
    Returns (E0, out_of_range): field and the fraction of pillars outside the library range."""
    s = np.asarray(lib_dense["sizes"], float); t = np.asarray(lib_dense["t"], complex)
    q = design["size_map"] + delta; ins = design["inside"]
    E = np.interp(q, s, t.real) + 1j * np.interp(q, s, t.imag)
    oor = float(np.mean((q[ins] < s[0] - 1e-9) | (q[ins] > s[-1] + 1e-9))) if ins.any() else 0.0
    return np.where(ins, E, 0.0), oor


def lens_tolerance(design, lib_dense=None, bias_nm=(-4, -2, 0, 2, 4), sigma_nm=0.0, n_trial=20,
                   seed=0, u=4, pad=None, nz=41):
    """Fabrication tolerance of a lens: every pillar size becomes s + bias + N(0, sigma^2) (nm),
    for each uniform bias in bias_nm; sigma_nm = 0 gives the bias-only (deterministic) case.
    Uses the interpolated size response of a refined library (no extra solves). Against
    re-solved RCWA for uniform bias (D = 16 um TiO2 lens, +-8 nm) the interpolated absolute
    efficiency agreed within 0.7 points and was slightly pessimistic.
    Returns dict: bias_nm, sigma_nm, n_trial, strehl and eff ((n_bias, n_trial) arrays),
    mean/std/p10 of both, out_of_range (max fraction of pillars outside the library)."""
    lib_dense = lib_dense or design["lib"]
    if not lib_dense.get("refined"):
        lib_dense = refine_library(lib_dense)
    b = np.atleast_1d(np.asarray(bias_nm, float)); sig = float(sigma_nm)
    nt = max(1, int(n_trial)) if sig > 0 else 1
    rng = np.random.default_rng(seed); ins = design["inside"]
    St = np.zeros((len(b), nt)); Ef = np.zeros((len(b), nt)); oor = 0.0
    for i, bi in enumerate(b):
        for j in range(nt):
            dmap = np.full(ins.shape, bi * 1e-3)
            if sig > 0:
                dmap = dmap + rng.normal(0.0, sig * 1e-3, ins.shape)
            E0, o = perturbed_field(design, lib_dense, dmap); oor = max(oor, o)
            dd = dict(design); dd["E0"] = E0
            a = analyze_focus(dd, u=u, pad=pad, nz=nz)
            St[i, j] = a["strehl"]; Ef[i, j] = a["eff_absolute"]
    stat = lambda A: {"mean": A.mean(1), "std": A.std(1), "p10": np.percentile(A, 10, axis=1)}  # noqa: E731
    return {"bias_nm": b, "sigma_nm": sig, "n_trial": nt, "strehl": St, "eff": Ef,
            "strehl_stats": stat(St), "eff_stats": stat(Ef), "out_of_range": oor}
