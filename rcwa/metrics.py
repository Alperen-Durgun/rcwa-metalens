"""Design metrics — Item 4.
Focusing efficiency, FWHM, Strehl ratio, MTF, far-field. Metasurface/lens figures of merit.
"""
import numpy as np


def fwhm_1d(cut, coord):
    """FWHM of the MAIN LOBE of a cut (in units of coord).

    The half-maximum crossings on both sides of the peak are located by linear
    interpolation (sub-grid accuracy); sidelobes that exceed 0.5 are ignored."""
    c = np.asarray(cut, float); x = np.asarray(coord, float)
    if c.size < 2 or not np.all(np.isfinite(c)) or c.max() <= 0:
        return 0.0
    c = c / c.max(); p = int(np.argmax(c))

    def _cross(step):
        i = p
        while 0 <= i + step < c.size and c[i + step] >= 0.5:
            i += step
        j = i + step
        if j < 0 or j >= c.size:
            return x[i]                       # lobe runs off the window edge
        return x[i] + (0.5 - c[i]) * (x[j] - x[i]) / (c[j] - c[i])

    return float(abs(_cross(+1) - _cross(-1)))


def focal_metrics(I, coord):
    """2D focal intensity I(x,y), coord=1D axis. FWHM (x and y) + peak position."""
    I = np.asarray(I); j = np.unravel_index(np.argmax(I), I.shape)
    fx = fwhm_1d(I[:, j[1]], coord); fy = fwhm_1d(I[j[0], :], coord)
    return {"fwhm_x": fx, "fwhm_y": fy, "peak_ij": j, "peak": float(I.max())}


def focusing_efficiency(I, coord, fwhm, r_factor=1.5):
    """Power in the focal spot / total power in the plane (disk of radius r_factor·FWHM)."""
    j = np.unravel_index(np.argmax(I), I.shape)
    X, Y = np.meshgrid(coord, coord, indexing="ij")
    rr = np.hypot(X - coord[j[0]], Y - coord[j[1]])
    return float(I[rr <= r_factor * fwhm].sum() / I.sum())


def strehl(I_design, I_ideal):
    """Strehl ratio: design peak / ideal (aberration-free) peak (same power normalization)."""
    return float((I_design.max() / I_design.sum()) / (I_ideal.max() / I_ideal.sum()))


def mtf_1d(psf_row):
    """1D MTF: normalized |FFT| of the PSF cut (contrast vs. spatial frequency)."""
    m = np.abs(np.fft.fftshift(np.fft.fft(psf_row)))
    return m / m.max()


def efficiency_suite(I, coord, fwhm, P_incident=None, r_abs=3.0, r_leg=1.5):
    """Literature-standard efficiency definitions (all of them, to remove ambiguity):
      - absolute : power in the 3×FWHM disk / INCIDENT power (P_incident)   [standard reporting]
      - relative : power in the 3×FWHM disk / TOTAL focal-plane power
      - legacy   : power in the 1.5×FWHM disk / focal-plane total (our old definition)
    If P_incident is not given, absolute=None."""
    I = np.asarray(I); j = np.unravel_index(np.argmax(I), I.shape)
    X, Y = np.meshgrid(coord, coord, indexing="ij")
    rr = np.hypot(X - coord[j[0]], Y - coord[j[1]])
    P_focal = float(I.sum())
    P_abs = float(I[rr <= r_abs * fwhm].sum())
    P_leg = float(I[rr <= r_leg * fwhm].sum())
    return {"absolute": (P_abs / P_incident) if P_incident else None,
            "relative": P_abs / P_focal, "legacy": P_leg / P_focal,
            "P_focalplane": P_focal, "P_incident": P_incident}


def encircled_energy(I, coord, fraction=0.8):
    """Encircled energy: peak-centered radius (in units of coord) containing 'fraction'
    of the total focal-plane energy. A measure of spot concentration."""
    I = np.asarray(I); j = np.unravel_index(np.argmax(I), I.shape)
    X, Y = np.meshgrid(coord, coord, indexing="ij")
    rr = np.hypot(X - coord[j[0]], Y - coord[j[1]]).ravel()
    vals = I.ravel(); order = np.argsort(rr)
    cum = np.cumsum(vals[order]); cum /= cum[-1]
    idx = np.searchsorted(cum, fraction)
    return float(rr[order][min(idx, len(rr) - 1)])


def sidelobe_level(psf_row):
    """Ratio of the highest sidelobe to the main peak (linear and dB). From a PSF cut."""
    c = np.asarray(psf_row, float); c = c / c.max()
    p = int(np.argmax(c))
    # find the first minima of the main lobe
    lo = p
    while lo > 0 and c[lo - 1] <= c[lo]:
        lo -= 1
    hi = p
    while hi < len(c) - 1 and c[hi + 1] <= c[hi]:
        hi += 1
    side = np.concatenate([c[:lo], c[hi + 1:]])
    if side.size == 0:
        return {"ratio": 0.0, "dB": -np.inf}
    s = float(side.max())
    return {"ratio": s, "dB": float(10 * np.log10(s)) if s > 0 else -np.inf}


def depth_of_focus(I_axis, z):
    """Depth of focus: FWHM of the on-axis I(z) (in units of z)."""
    return fwhm_1d(np.asarray(I_axis, float), np.asarray(z, float))


def mtf_2d(psf2d):
    """2D MTF: radial average of the |FFT| of the PSF (normalized). Returns: (mtf_radial, )."""
    P = np.abs(np.fft.fftshift(np.fft.fft2(psf2d)))
    N = P.shape[0]; c = N // 2
    y, x = np.mgrid[0:N, 0:N]; r = np.hypot(x - c, y - c).astype(int)
    nr = np.bincount(r.ravel()); tot = np.bincount(r.ravel(), P.ravel())
    prof = tot / np.maximum(nr, 1)
    return prof / prof[0]


_TRAPZ = getattr(np, "trapezoid", getattr(np, "trapz", np.sum))  # numpy 2.x: trapz->trapezoid


def strehl_mtf(mtf_design, mtf_ideal):
    """MTF-based Strehl: area under design MTF / area under ideal MTF (alternative Strehl definition)."""
    a = float(_TRAPZ(mtf_design)); b = float(_TRAPZ(mtf_ideal))
    return a / b if b > 0 else float("nan")


def group_delay(phase, omega):
    """Group delay GD=dφ/dω and GDD=d²φ/dω² from the meta-atom phase (achromatic design criterion).
    phase, omega: from a wavelength sweep (ω=2πc/λ). Returns: (GD_center, GDD_center)."""
    phase = np.unwrap(np.asarray(phase, float)); omega = np.asarray(omega, float)
    gd = np.gradient(phase, omega); gdd = np.gradient(gd, omega)
    k = len(omega) // 2
    return float(gd[k]), float(gdd[k])


def angular_spectrum(E0, dx, lam, z):
    """Propagate the field E0(x,y) to z in free space (exact)."""
    k0 = 2 * np.pi / lam; N = E0.shape[0]
    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx); KX, KY = np.meshgrid(kx, kx, indexing="ij")
    KZ = np.sqrt((k0 ** 2 - KX ** 2 - KY ** 2).astype(complex)); KZ = np.where(np.imag(KZ) < 0, -KZ, KZ)
    return np.fft.ifft2(np.fft.fft2(E0) * np.exp(1j * KZ * z))
