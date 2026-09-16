"""Tasarım metrikleri — Madde 4.
Odaklama verimi, FWHM, Strehl oranı, MTF, uzak-alan. Metayüzey/lens başarı ölçütleri.
"""
import numpy as np


def fwhm_1d(cut, coord):
    """Bir kesitin FWHM'i (coord birimiyle)."""
    c = cut / cut.max()
    idx = coord[c >= 0.5]
    return float(idx.max() - idx.min()) if idx.size else 0.0


def focal_metrics(I, coord):
    """2B odak şiddeti I(x,y), coord=1B eksen. FWHM (x ve y) + tepe konumu."""
    I = np.asarray(I); j = np.unravel_index(np.argmax(I), I.shape)
    fx = fwhm_1d(I[:, j[1]], coord); fy = fwhm_1d(I[j[0], :], coord)
    return {"fwhm_x": fx, "fwhm_y": fy, "peak_ij": j, "peak": float(I.max())}


def focusing_efficiency(I, coord, fwhm, r_factor=1.5):
    """Odak lekesindeki güç / düzlemdeki toplam güç (r_factor·FWHM yarıçaplı disk)."""
    j = np.unravel_index(np.argmax(I), I.shape)
    X, Y = np.meshgrid(coord, coord, indexing="ij")
    rr = np.hypot(X - coord[j[0]], Y - coord[j[1]])
    return float(I[rr <= r_factor * fwhm].sum() / I.sum())


def strehl(I_design, I_ideal):
    """Strehl oranı: tasarım tepe / ideal (aberasyonsuz) tepe (aynı güç normalizasyonu)."""
    return float((I_design.max() / I_design.sum()) / (I_ideal.max() / I_ideal.sum()))


def mtf_1d(psf_row):
    """1B MTF: PSF kesitinin normalize |FFT|'i (uzaysal frekansa karşı kontrast)."""
    m = np.abs(np.fft.fftshift(np.fft.fft(psf_row)))
    return m / m.max()


def efficiency_suite(I, coord, fwhm, P_incident=None, r_abs=3.0, r_leg=1.5):
    """Literatür-standardı verim tanımları (tutarsızlığı gidermek için hepsi):
      - absolute : 3×FWHM diskindeki güç / GELEN güç (P_incident)   [standart raporlama]
      - relative : 3×FWHM diskindeki güç / odak-düzlemi TOPLAM gücü
      - legacy   : 1.5×FWHM diskindeki güç / odak-düzlemi toplam (bizim eski tanım)
    P_incident verilmezse absolute=None."""
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
    """Kuşatılmış enerji: tepe merkezli, toplam odak-düzlemi enerjisinin 'fraction'ını
    içeren yarıçap (coord birimi). Leke konsantrasyonu ölçütü."""
    I = np.asarray(I); j = np.unravel_index(np.argmax(I), I.shape)
    X, Y = np.meshgrid(coord, coord, indexing="ij")
    rr = np.hypot(X - coord[j[0]], Y - coord[j[1]]).ravel()
    vals = I.ravel(); order = np.argsort(rr)
    cum = np.cumsum(vals[order]); cum /= cum[-1]
    idx = np.searchsorted(cum, fraction)
    return float(rr[order][min(idx, len(rr) - 1)])


def sidelobe_level(psf_row):
    """En yüksek yan-lobun ana tepeye oranı (lineer ve dB). PSF kesitinden."""
    c = np.asarray(psf_row, float); c = c / c.max()
    p = int(np.argmax(c))
    # ana lobun ilk minimumlarını bul
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
    """Odak derinliği: eksen-üstü I(z)'nin FWHM'i (z birimi)."""
    return fwhm_1d(np.asarray(I_axis, float), np.asarray(z, float))


def mtf_2d(psf2d):
    """2B MTF: PSF'in |FFT|'inin radyal ortalaması (normalize). Döndürür: (mtf_radial, )."""
    P = np.abs(np.fft.fftshift(np.fft.fft2(psf2d)))
    N = P.shape[0]; c = N // 2
    y, x = np.mgrid[0:N, 0:N]; r = np.hypot(x - c, y - c).astype(int)
    nr = np.bincount(r.ravel()); tot = np.bincount(r.ravel(), P.ravel())
    prof = tot / np.maximum(nr, 1)
    return prof / prof[0]


_TRAPZ = getattr(np, "trapezoid", getattr(np, "trapz", np.sum))  # numpy 2.x: trapz->trapezoid


def strehl_mtf(mtf_design, mtf_ideal):
    """MTF-tabanlı Strehl: tasarım MTF-altı-alan / ideal MTF-altı-alan (alternatif Strehl tanımı)."""
    a = float(_TRAPZ(mtf_design)); b = float(_TRAPZ(mtf_ideal))
    return a / b if b > 0 else float("nan")


def group_delay(phase, omega):
    """Meta-atom fazından grup gecikmesi GD=dφ/dω ve GDD=d²φ/dω² (akromatik tasarım kriteri).
    phase, omega: dalga boyu taramasından (ω=2πc/λ). Döndürür: (GD_merkez, GDD_merkez)."""
    phase = np.unwrap(np.asarray(phase, float)); omega = np.asarray(omega, float)
    gd = np.gradient(phase, omega); gdd = np.gradient(gd, omega)
    k = len(omega) // 2
    return float(gd[k]), float(gdd[k])


def angular_spectrum(E0, dx, lam, z):
    """E0(x,y) alanını serbest uzayda z'ye taşı (exact)."""
    k0 = 2 * np.pi / lam; N = E0.shape[0]
    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx); KX, KY = np.meshgrid(kx, kx, indexing="ij")
    KZ = np.sqrt((k0 ** 2 - KX ** 2 - KY ** 2).astype(complex)); KZ = np.where(np.imag(KZ) < 0, -KZ, KZ)
    return np.fft.ifft2(np.fft.fft2(E0) * np.exp(1j * KZ * z))
