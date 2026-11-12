"""Arbitrary meta-atom geometries — Item 4.
Produces both a rasterized eps(x,y) for RCWA and polygon vertices for GDS;
the two are consistent. Shapes: square, rect, circle, ellipse, hexagon, cross.
params: per shape (um):
  square:  size (side)             rect: (wx, wy)
  circle:  size (diameter)         ellipse: (ax, by) (diameters)
  hexagon: size (diagonal diameter)  cross: (arm_length, arm_thickness)
"""
import numpy as np

DEFAULT_N = 512   # default raster resolution per side used by the designers (see rasterize)


def _rot(x, y, a):
    c, s = np.cos(a), np.sin(a)
    return c * x + s * y, -s * x + c * y


def rasterize(shape, params, L, N, n_pillar=2.4, n_bg=1.0, angle=0.0, aa=1):
    """eps(x,y) raster (N×N) of the shape centered in an (L×L) unit cell.

    Resolution matters: a binary mask quantizes the feature size to the pixel pitch L/N,
    so a size sweep on a coarse grid advances in steps (N=64, L=0.35um -> 5.5nm steps, i.e.
    phase jumps of ~0.5 rad for a square pillar). Use N >= DEFAULT_N for design work — the
    solver cost does not depend on N (only an FFT does).

    aa : optional anti-aliasing. aa>1 sub-samples each pixel aa×aa and stores the
    area-weighted MEAN permittivity. It smooths a sweep on a coarse grid but slightly biases
    the result (boundary pixels would need an anisotropic average); prefer a larger N."""
    if aa and aa > 1:
        fine = rasterize(shape, params, L, N * int(aa), n_pillar=n_pillar, n_bg=n_bg, angle=angle, aa=1)
        a = int(aa)
        return fine.reshape(N, a, N, a).mean(axis=(1, 3))
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
        raise ValueError(f"unknown shape: {shape}")
    return np.where(m, n_pillar ** 2, n_bg ** 2).astype(complex)


def polygon(shape, params, center=(0.0, 0.0), angle=0.0, n_circle=48):
    """Vertex list of the shape (centered at cx,cy, rotated by angle) — for GDS."""
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
        raise ValueError(f"unknown shape: {shape}")
    c, s = np.cos(angle), np.sin(angle)
    return [(cx + c*x - s*y, cy + s*x + c*y) for x, y in pts]
