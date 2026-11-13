"""Convolution (Toeplitz) matrix — Lesson 03 & Coding Lesson 01.
Converts the Fourier representation of a periodic eps(x) profile into a matrix
that couples the diffraction orders. All of RCWA's 'material knowledge' is here.
"""
import numpy as np


def convmat1d(eps_x, P):
    """P x P convolution matrix of a 1D periodic profile.

    Parameters
    ----------
    eps_x : 1D array
        Dielectric constant eps(x) sampled uniformly over one period.
    P : int
        Number of spatial harmonics (orders) to keep — must be odd (2M+1).

    Returns
    -------
    C : (P, P) complex matrix   (C[r,c] = eps_{r-c})
    """
    if P % 2 == 0:
        raise ValueError("P must be odd (2M+1).")
    eps_x = np.asarray(eps_x)
    Nx = eps_x.shape[0]
    if Nx < 2 * P - 1:
        raise ValueError(f"eps_x has {Nx} samples; at least 2P-1 = {2 * P - 1} are needed for P = {P} orders.")
    # Fourier coefficients (fftshift puts the zero harmonic in the center)
    A = np.fft.fftshift(np.fft.fft(eps_x)) / Nx
    p0 = Nx // 2
    idx = np.arange(P)
    return A[p0 + (idx[:, None] - idx[None, :])].astype(complex)


def convmat2d(eps, Px, Py):
    """Convolution matrix of a 2D periodic eps(x,y) profile — 2D Lesson & 2D Code.

    eps : (Nx, Ny) complex array, dielectric sampled over the unit cell.
    Px, Py : number of harmonics in the x and y directions (both odd).
    Returns: (Px*Py, Px*Py) matrix. Order layout is x-outer, y-inner (meshgrid 'ij').
    """
    eps = np.asarray(eps)
    Nx, Ny = eps.shape
    if Px % 2 == 0 or Py % 2 == 0:
        raise ValueError("Px and Py must be odd (2M+1).")
    if Nx < 2 * Px - 1 or Ny < 2 * Py - 1:
        raise ValueError(f"eps grid {Nx}x{Ny} is too coarse for {Px}x{Py} orders "
                         f"(need at least {2 * Px - 1}x{2 * Py - 1} samples).")
    A = np.fft.fftshift(np.fft.fft2(eps)) / (Nx * Ny)
    p0, q0 = Nx // 2, Ny // 2
    Mx, My = Px // 2, Py // 2
    MX, MY = np.meshgrid(np.arange(-Mx, Mx + 1), np.arange(-My, My + 1), indexing='ij')
    MXf, MYf = MX.flatten(), MY.flatten()
    return A[p0 + (MXf[:, None] - MXf[None, :]), q0 + (MYf[:, None] - MYf[None, :])].astype(complex)


def convmat2d_li(eps, Px, Py):
    """Li-factorized convolution matrices of a 2D profile (Li 1997, "zigzag" rule).

    Returns (ER, ERx, ERy):
      ER  : plain Laurent matrix [[eps]]            -> used for Ez (tangential everywhere)
      ERx : inverse rule along x, Laurent along y   -> multiplies Ex (normal to x-walls)
      ERy : inverse rule along y, Laurent along x   -> multiplies Ey (normal to y-walls)
    For axis-aligned (rectangular) features this is the correct Fourier factorization and
    converges much faster than plain Laurent at high index contrast; for curved boundaries
    it is an approximation that still improves on Laurent in most dielectric cases.
    """
    eps = np.asarray(eps, dtype=complex)
    Nx, Ny = eps.shape
    ER = convmat2d(eps, Px, Py)
    inv_eps = 1.0 / eps
    ix = np.arange(Px); iy = np.arange(Py)
    dmx = ix[:, None] - ix[None, :]; dmy = iy[:, None] - iy[None, :]
    if Nx < 2 * Px - 1 or Ny < 2 * Py - 1:
        raise ValueError(f"eps grid {Nx}x{Ny} is too coarse for {Px}x{Py} orders")
    # ERx: for every y-line, the 1D inverse-rule matrix along x; then Fourier transform along y
    # (all lines at once: one FFT along x, then the Toeplitz gather; same values as convmat1d)
    Lx = np.fft.fftshift(np.fft.fft(inv_eps, axis=0), axes=0) / Nx                     # (Nx,Ny)
    Ax = np.linalg.inv(Lx[Nx // 2 + dmx].transpose(2, 0, 1))                             # (Ny,Px,Px)
    Fx = np.fft.fftshift(np.fft.fft(Ax, axis=0), axes=0) / Ny
    Gx = Fx[Ny // 2 + dmy]                                    # [n,n',m,m']
    ERx = Gx.transpose(2, 0, 3, 1).reshape(Px * Py, Px * Py)  # index (m,n),(m',n'), x-outer
    # ERy: for every x-line, the 1D inverse-rule matrix along y; then Fourier transform along x
    Ly = np.fft.fftshift(np.fft.fft(inv_eps, axis=1), axes=1) / Ny                     # (Nx,Ny)
    Ay = np.linalg.inv(Ly[:, Ny // 2 + dmy])                                             # (Nx,Py,Py)
    Fy = np.fft.fftshift(np.fft.fft(Ay, axis=0), axes=0) / Nx
    Gy = Fy[Nx // 2 + dmx]                                    # [m,m',n,n']
    ERy = Gy.transpose(0, 2, 1, 3).reshape(Px * Py, Px * Py)
    return ER, ERx, ERy
