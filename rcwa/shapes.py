"""Keyfi meta-atom geometrileri — Madde 4.
Hem RCWA için rasterize edilmiş eps(x,y) hem de GDS için polygon köşeleri üretir;
ikisi tutarlıdır. Şekiller: square, rect, circle, ellipse, hexagon, cross.
params: şekle göre (um):
  square:  size (kenar)             rect: (wx, wy)
  circle:  size (çap)              ellipse: (ax, by) (çaplar)
  hexagon: size (köşegen çapı)     cross: (kol_uzunluğu, kol_kalınlığı)
"""
import numpy as np


def _rot(x, y, a):
    c, s = np.cos(a), np.sin(a)
    return c * x + s * y, -s * x + c * y


def rasterize(shape, params, L, N, n_pillar=2.4, n_bg=1.0, angle=0.0):
    """Birim hücrede (L×L) merkeze yerleştirilmiş şeklin eps(x,y) rasteri (N×N)."""
    x = (np.arange(N) + 0.5) / N * L - L / 2
    X, Y = np.meshgrid(x, x, indexing="ij")
    Xr, Yr = _rot(X, Y, angle)
    p = params if np.iterable(params) else (params,)
    if shape == "square":
        m = (np.abs(Xr) < p[0] / 2) & (np.abs(Yr) < p[0] / 2)
    elif shape == "rect":
        m = (np.abs(Xr) < p[0] / 2) & (np.abs(Yr) < p[1] / 2)
    elif shape == "circle":
        m = (Xr ** 2 + Yr ** 2) < (p[0] / 2) ** 2
    elif shape == "ellipse":
        m = (Xr / (p[0] / 2)) ** 2 + (Yr / (p[1] / 2)) ** 2 < 1
    elif shape == "hexagon":
        r = p[0] / 2; m = np.ones_like(Xr, bool)
        for k in range(6):
            th = np.pi / 6 + k * np.pi / 3
            m &= (Xr * np.cos(th) + Yr * np.sin(th)) < r * np.cos(np.pi / 6)
    elif shape == "cross":
        arm, th = p[0], p[1]
        m = ((np.abs(Xr) < arm / 2) & (np.abs(Yr) < th / 2)) | \
            ((np.abs(Yr) < arm / 2) & (np.abs(Xr) < th / 2))
    else:
        raise ValueError(f"bilinmeyen şekil: {shape}")
    return np.where(m, n_pillar ** 2, n_bg ** 2).astype(complex)


def polygon(shape, params, center=(0.0, 0.0), angle=0.0, n_circle=48):
    """Şeklin (cx,cy merkezli, angle döndürülmüş) köşe listesi — GDS için."""
    p = params if np.iterable(params) else (params,)
    cx, cy = center
    if shape in ("square", "rect"):
        wx = p[0]; wy = p[0] if shape == "square" else p[1]
        pts = [(-wx/2, -wy/2), (wx/2, -wy/2), (wx/2, wy/2), (-wx/2, wy/2)]
    elif shape in ("circle", "ellipse"):
        ax = p[0]/2; by = p[0]/2 if shape == "circle" else p[1]/2
        t = np.linspace(0, 2*np.pi, n_circle, endpoint=False)
        pts = list(zip(ax*np.cos(t), by*np.sin(t)))
    elif shape == "hexagon":
        r = p[0]/2; t = np.pi/6 + np.arange(6)*np.pi/3
        pts = list(zip(r*np.cos(t), r*np.sin(t)))
    elif shape == "cross":
        arm, th = p[0]/2, p[1]/2
        pts = [(-arm,-th),(-th,-th),(-th,-arm),(th,-arm),(th,-th),(arm,-th),
               (arm,th),(th,th),(th,arm),(-th,arm),(-th,th),(-arm,th)]
    else:
        raise ValueError(f"bilinmeyen şekil: {shape}")
    c, s = np.cos(angle), np.sin(angle)
    return [(cx + c*x - s*y, cy + s*x + c*y) for x, y in pts]
