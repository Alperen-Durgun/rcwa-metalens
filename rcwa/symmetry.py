"""Mirror-symmetry reduction of the 2D RCWA problem (roadmap N2).

At normal incidence, with a purely x- or purely y-polarized plane wave and a unit cell that
is mirror-symmetric in x AND y, the Fourier amplitudes of the field have a fixed parity:
for an Ey source, Ey is even in x and y and Ex is odd in x and y (the roles swap for an Ex
source). The whole problem - layer eigenmodes, S-matrices, incidence and exit media - can
then be solved in that parity subspace, which has about a quarter of the unknowns:
(Mx+1)(My+1) + Mx*My instead of 2(2Mx+1)(2My+1). Dense linear algebra scales with the cube,
so the eigenproblem is up to ~60x cheaper.

Everything here is exact (a change of basis), not an approximation. The solver checks the
symmetry numerically and falls back to the full problem when it does not hold.

The rasters of this package sample the cell at x_i = (i + 1/2) L/N, i.e. the mirror plane
is at x0 = (N - 1) L / (2N) in FFT coordinates. The convolution matrices are translated so
that the mirror plane sits at the origin; the zeroth order is unaffected by that translation
and the other orders are translated back.
"""
import numpy as np


def raster_is_mirror_symmetric(eps):
    """True if eps(x, y) is exactly symmetric under x -> -x and y -> -y (cell-centred)."""
    eps = np.asarray(eps)
    return bool(np.array_equal(eps, eps[::-1, :]) and np.array_equal(eps, eps[:, ::-1]))


def translation(MXf, MYf, Nx, Ny, sign):
    """Diagonal phase that moves the mirror plane of a cell-centred raster to the origin."""
    x0 = (Nx - 1) / (2.0 * Nx); y0 = (Ny - 1) / (2.0 * Ny)
    return np.exp(sign * 2j * np.pi * (MXf * x0 + MYf * y0))


def _mirror_index(MXf, MYf, fx, fy):
    """Index of the order (fx*m, fy*n) for every order (m, n)."""
    lut = {(int(m), int(n)): i for i, (m, n) in enumerate(zip(MXf, MYf))}
    return np.array([lut[(fx * int(m), fy * int(n))] for m, n in zip(MXf, MYf)])


def is_mirror_symmetric_matrix(A, MXf, MYf, tol=1e-10):
    """Checks A[(m,n),(m',n')] == A[(-m,n),(-m',n')] == A[(m,-n),(m',-n')]."""
    sx = _mirror_index(MXf, MYf, -1, 1); sy = _mirror_index(MXf, MYf, 1, -1)
    s = max(np.abs(A).max(), 1e-300)
    return (np.abs(A[np.ix_(sx, sx)] - A).max() <= tol * s
            and np.abs(A[np.ix_(sy, sy)] - A).max() <= tol * s)


def parity_basis(MXf, MYf, px, py):
    """Orthonormal real basis (P x r) of the scalar Fourier vectors with parity px in m and
    py in n (+1 even, -1 odd)."""
    P = len(MXf); idx = {(int(m), int(n)): i for i, (m, n) in enumerate(zip(MXf, MYf))}
    cols = []
    for m, n in zip(MXf, MYf):
        m, n = int(m), int(n)
        if m < 0 or n < 0 or (px < 0 and m == 0) or (py < 0 and n == 0):
            continue
        v = np.zeros(P)
        for sm in ((1, -1) if m else (1,)):
            for sn in ((1, -1) if n else (1,)):
                v[idx[(sm * m, sn * n)]] += (px if sm < 0 else 1) * (py if sn < 0 else 1)
        cols.append(v / np.linalg.norm(v))
    return np.array(cols).T if cols else np.zeros((P, 0))


def vector_basis(MXf, MYf, first, second):
    """Basis of 2P-vectors [a; b] with a in parity class `first` and b in `second`
    (each a (px, py) tuple)."""
    Ua = parity_basis(MXf, MYf, *first); Ub = parity_basis(MXf, MYf, *second)
    P = len(MXf)
    U = np.zeros((2 * P, Ua.shape[1] + Ub.shape[1]))
    U[:P, :Ua.shape[1]] = Ua; U[P:, Ua.shape[1]:] = Ub
    return U


def e_basis(MXf, MYf, y_polarized):
    """Parity basis of the tangential E vector [Ex; Ey] for an x- or y-polarized source."""
    ee, oo = (1, 1), (-1, -1)
    return vector_basis(MXf, MYf, oo, ee) if y_polarized else vector_basis(MXf, MYf, ee, oo)


def h_basis(Q, UE, MXf, MYf, tol=1e-9):
    """Parity basis of the tangential H vector: the one of the two candidates into which
    Q maps the E subspace without leakage. Returns None if neither fits."""
    QU = Q @ UE; nrm = max(np.abs(QU).max(), 1e-300)
    for cand in (vector_basis(MXf, MYf, (1, 1), (-1, -1)), vector_basis(MXf, MYf, (-1, -1), (1, 1))):
        if cand.shape[1] != UE.shape[1]:
            continue
        leak = QU - cand @ (cand.T @ QU)
        if np.abs(leak).max() <= tol * nrm:
            return cand
    return None
