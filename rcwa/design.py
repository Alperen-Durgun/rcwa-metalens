"""Shared metalens design core.

One implementation of the steps every lens designer in this package needs:

    meta_atom_library  ->  lens_phase / lens_grid  ->  select_atoms  ->  FocalField

Everything here works in the PHYSICS convention exp(-i*omega*t) (see ``convention.py``):
library transmission coefficients are converted once, right after the RCWA solve, so the
hyperbolic lens phase, the angular-spectrum propagator and FDTD tools can be combined with
them directly.
"""
import numpy as np
from . import materials, shapes, cache
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


def default_sizes(L, n=20, s_min=0.06, margin=0.05):
    """Standard size sweep: from s_min up to (L - margin), n samples (um)."""
    return np.linspace(s_min, L - margin, int(n))


# --------------------------------------------------------------------------- library
def meta_atom_library(lam0, L, H, sizes=None, shape="circle", material=None, n_pillar=2.4,
                      M=7, n_sub=1.0, n_bg=1.0, raster_n=None, use_cache=True):
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
    """
    sizes = default_sizes(L) if sizes is None else np.asarray(sizes, float)
    n = pillar_index(material, lam0, n_pillar)
    N = int(raster_n or shapes.DEFAULT_N)

    def _compute():
        t = np.zeros(len(sizes), complex)
        for k, s in enumerate(sizes):
            eps = shapes.rasterize(shape, s, L, N, n_pillar=n, n_bg=n_bg)
            r = solve_rcwa_2d(lam0, 0, 0, n_sub ** 2, n_bg ** 2, L, L, [(eps, H)], M, M, pol=(0., 1.))
            t[k] = r["ty"][r["i0"]]
        return {"t_re": t.real, "t_im": t.imag}

    key = {"lam0": float(lam0), "L": float(L), "H": float(H), "shape": shape, "M": int(M),
           "n": (round(n.real, 5), round(n.imag, 5)), "n_sub": float(n_sub), "n_bg": float(n_bg),
           "N": N, "sizes": tuple(np.round(sizes, 6))}
    c = cache.cached_library("atomlib", key, _compute) if use_cache else _compute()
    t = to_physical(c["t_re"] + 1j * c["t_im"])
    ph = np.unwrap(np.angle(t)); ph = ph - ph.min()
    return {"sizes": sizes, "t": t, "phase": ph, "amp": np.abs(t),
            "coverage": float(np.ptp(ph) / (2 * np.pi)), "n": n,
            "shape": shape, "L": L, "H": H, "lam0": lam0, "M": M, "n_sub": n_sub, "n_bg": n_bg}


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
def phase_lut(t_lib, nbin=720, mode="complex"):
    """Look-up table: for each target phase bin, the best library atom.

    mode="complex" (default): minimizes |t_k/|t|max - exp(i*phi)| — an atom with the right
    phase but poor transmission (a resonance dip) loses against a slightly detuned bright one.
    mode="phase": nearest phase only (classic rule).
    Returns (lut_k, phases): atom index per bin and the bin-centre phases."""
    t = np.asarray(t_lib, complex)
    phases = (np.arange(nbin) + 0.5) * 2 * np.pi / nbin
    if mode == "phase":
        d = np.abs(np.angle(np.exp(1j * (np.angle(t)[None, :] - phases[:, None]))))
    else:
        d = np.abs(t[None, :] / np.abs(t).max() - np.exp(1j * phases)[:, None])
    return np.argmin(d, axis=1), phases


def select_atoms(t_lib, phi_target, mask=None, mode="complex", nbin=720, n_offset=24):
    """Assign a library atom to every cell (vectorized, no Python loop over cells).

    A lens tolerates any global phase offset, so n_offset offsets are tried and the one that
    maximizes the coherent sum toward the target (i.e. the on-axis focal amplitude) is kept.
    Returns (idx, offset): atom index per cell (-1 outside the mask) and the chosen offset."""
    t = np.asarray(t_lib, complex); phi = np.asarray(phi_target, float)
    mask = np.ones(phi.shape, bool) if mask is None else np.asarray(mask, bool)
    lut_k, _ = phase_lut(t, nbin=nbin, mode=mode)
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
class FocalField:
    """Angular-spectrum propagation of a cell-sampled aperture field (physics convention).

    E0   : (Nc, Nc) complex field, one sample per lattice cell (0 outside the aperture)
    L    : lattice period (um);  lam0: wavelength (um);  n_bg: index behind the lens
    u    : sub-samples per cell;  pad: zero-padding factor of the computation window
    tilt_deg : incidence tilt in the x-z plane (adds a linear phase to the aperture)
    """

    def __init__(self, E0, L, lam0, n_bg=1.0, u=4, pad=1.6, tilt_deg=0.0):
        E0 = np.asarray(E0, complex); Nc = E0.shape[0]
        self.lam0 = lam0; self.k = 2 * np.pi / lam0 * n_bg; self.dx = L / u
        Ef = np.kron(E0, np.ones((u, u))); Ncf = Nc * u
        N = int(Ncf * pad); N += N % 2                     # even window
        Ep = np.zeros((N, N), complex); o = (N - Ncf) // 2; Ep[o:o + Ncf, o:o + Ncf] = Ef
        self.N = N
        self.x = (np.arange(N) - o - Ncf / 2 + 0.5) * self.dx      # aperture centred on x = 0
        if tilt_deg:
            Ep = Ep * np.exp(1j * self.k * np.sin(np.deg2rad(tilt_deg)) * self.x[:, None])
        self.P_incident = float(np.sum(np.abs(E0) > 0)) * u * u    # unit-amplitude illumination
        self.P_aperture = float(np.sum(np.abs(Ep) ** 2))
        kx = 2 * np.pi * np.fft.fftfreq(N, d=self.dx); KX, KY = np.meshgrid(kx, kx, indexing="ij")
        KZ = np.sqrt((self.k ** 2 - KX ** 2 - KY ** 2).astype(complex))
        self.KZ = np.where(np.imag(KZ) < 0, -KZ, KZ)
        self.A = np.fft.fft2(Ep)
        # phase ramp that evaluates the inverse FFT at the aperture centre without an FFT
        xc = (o + Ncf / 2 - 0.5)                                    # centre in (fractional) samples
        ramp = np.exp(2j * np.pi * np.fft.fftfreq(N) * xc)
        self._Ac = self.A * ramp[:, None] * ramp[None, :] / (N * N)

    def at(self, z):
        """Complex field in the plane z (N x N)."""
        return np.fft.ifft2(self.A * np.exp(1j * self.KZ * z))

    def intensity(self, z):
        return np.abs(self.at(z)) ** 2

    def on_axis(self, z):
        """On-axis intensity at one or many z (no FFT per plane — fast)."""
        zz = np.atleast_1d(np.asarray(z, float))
        out = np.array([abs(np.sum(self._Ac * np.exp(1j * self.KZ * zi))) ** 2 for zi in zz])
        return out if np.ndim(z) else float(out[0])

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


def strehl_at_focus(E0, L, lam0, zf, ff=None, n_bg=1.0, u=4, pad=1.6):
    """Strehl ratio referenced to the SAME plane: on-axis intensity of the design at its own
    focus zf divided by that of an aberration-free lens with the same aperture amplitudes
    focused at zf. By construction <= 1 (up to sampling), also for low Fresnel numbers where
    the intensity maximum is shifted away from the geometric focus."""
    ideal = ideal_aperture(E0, L, lam0, zf, n_bg=n_bg)
    ff = FocalField(E0, L, lam0, n_bg=n_bg, u=u, pad=pad) if ff is None else ff
    ref = FocalField(ideal, L, lam0, n_bg=n_bg, u=u, pad=pad)
    return float(ff.on_axis(zf) / ref.on_axis(zf))


# --------------------------------------------------------------------------- one-call designer
def design_lens(lam0, L, H, D_lens, f, shape="circle", material=None, n_pillar=2.4, M=7,
                n_sub=1.0, n_bg=1.0, sizes=None, mode="complex", lib=None):
    """Design a singlet metalens: library -> hyperbolic phase -> atom selection.

    Returns dict with: lib, ci, R, inside, D_lens (snapped to the lattice), NA, idx (atom index
    map), size_map (um, 0 outside), E0 (aperture field, physics convention), phi_target,
    offset, placements (list for GDS / full-wave bridges)."""
    lib = lib or meta_atom_library(lam0, L, H, sizes=sizes, shape=shape, material=material,
                                   n_pillar=n_pillar, M=M, n_sub=n_sub, n_bg=n_bg)
    ci, R, inside, D = lens_grid(D_lens, L)
    phi = lens_phase(R, lam0 / n_bg, f)
    idx, off = select_atoms(lib["t"], phi, inside, mode=mode)
    E0 = aperture_field(lib["t"], idx)
    size_map = np.where(idx >= 0, lib["sizes"][np.clip(idx, 0, None)], 0.0)
    ii, jj = np.nonzero(inside)
    placements = [{"x": float(ci[i]), "y": float(ci[j]), "shape": shape,
                   "params": float(size_map[i, j]), "angle": 0.0} for i, j in zip(ii, jj)]
    return {"lib": lib, "ci": ci, "R": R, "inside": inside, "D_lens": D, "f": f, "lam0": lam0,
            "L": L, "H": H, "shape": shape, "NA": float(n_bg * np.sin(np.arctan(D / 2 / f))),
            "idx": idx, "size_map": size_map, "E0": E0, "phi_target": phi, "offset": off,
            "placements": placements, "n_bg": n_bg, "n_sub": n_sub}


def analyze_focus(design, u=4, pad=1.6, nz=81):
    """Focal analysis of a design_lens() result with the standard metric set.
    Returns dict: z_focus, fwhm (um), strehl, efficiency suite, encircled-energy radius,
    sidelobe, depth of focus, plus the fields needed for plotting (I, x, zc, I_axis)."""
    lam0, f, L, n_bg = design["lam0"], design["f"], design["L"], design["n_bg"]
    ff = FocalField(design["E0"], L, lam0, n_bg=n_bg, u=u, pad=pad)
    zf, zc, Iz = ff.find_focus(f, nz=nz)
    I = ff.intensity(zf); x = ff.x
    met = metrics.focal_metrics(I, x); fwhm = met["fwhm_x"]
    es = metrics.efficiency_suite(I, x, fwhm, P_incident=ff.P_incident)
    dl = lam0 / (2 * design["NA"])
    return {"z_focus": zf, "fwhm": fwhm, "diffraction_fwhm": dl, "peak_ij": met["peak_ij"],
            "strehl": strehl_at_focus(design["E0"], L, lam0, zf, ff=ff, n_bg=n_bg, u=u, pad=pad),
            "eff_absolute": es["absolute"], "eff_relative": es["relative"], "eff_legacy": es["legacy"],
            "ee80": metrics.encircled_energy(I, x, 0.8),
            "sidelobe_dB": metrics.sidelobe_level(I[:, met["peak_ij"][1]])["dB"],
            "dof": metrics.depth_of_focus(Iz, zc),
            "fresnel_number": (design["D_lens"] / 2) ** 2 / (lam0 / n_bg * f),
            "I": I, "x": x, "zc": zc, "I_axis": Iz, "field": ff}
