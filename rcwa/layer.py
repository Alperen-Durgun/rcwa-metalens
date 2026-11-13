"""Layer solution — Lesson 04 & Coding Lesson 02.
Produces the eigenmodes (W, V, LAM) and the layer scattering matrix for
homogeneous and grating layers. Rumpf (EMPossible) formulation, with a gap medium.
"""
import numpy as np
from scipy.linalg import eig
from .convmat import convmat1d


def rayleigh_safe_scale(kx, ky, ers, tol=1e-9):
    """Scale factor (1 or 1+1e-7, ...) for the transverse wavevectors that moves every order
    off an exact Rayleigh (Wood) anomaly, where kx^2 + ky^2 = eps of the gap/incidence/exit
    medium, kz = 0 and the homogeneous-medium matrices become singular. Equivalent to
    detuning the wavelength by ~1e-7 relative — far below any physical resolution."""
    kt2 = np.asarray(kx) ** 2 + np.asarray(ky) ** 2
    s = 1.0
    for _ in range(8):
        if not any(np.any(np.abs(er - kt2 * s * s) < tol) for er in ers):
            break
        s *= 1.0 + 1e-7
    return s


def homogeneous_modes(KX, KY, er, ur=1.0):
    """Eigenmodes of a homogeneous region (W=I, V, KZ)."""
    kx, ky = np.diag(KX), np.diag(KY)                                  # KX, KY are diagonal
    D = np.diag
    Q = (1.0 / ur) * np.block([[D(kx * ky), D(ur * er - kx * kx)],
                               [D(ky * ky - ur * er), D(-(ky * kx))]])
    kz = np.conj(np.sqrt((er * ur) - kx ** 2 - ky ** 2 + 0j))
    KZ = np.diag(kz)
    W = np.eye(2 * len(kx), dtype=complex)
    V = Q / (1j * np.concatenate([kz, kz]))[None, :]        # Q @ inv(LAM), LAM diagonal
    return W, V, KZ


def grating_modes(eps_x, KX, KY, P, rule="inverse"):
    """Eigenmode solution of a grating layer: P,Q -> Omega^2 -> eig.

    rule="inverse" (default): Li's Fourier factorization for an x-periodic grating — the
    matrix multiplying Ex (normal to the walls) is inv([[1/eps]]); Ey and Ez keep the Laurent
    matrix [[eps]]. This is what makes TM converge quickly at high contrast. TE is unaffected.
    rule="laurent": the plain rule everywhere (slow TM convergence; kept for comparison)."""
    eps_x = np.asarray(eps_x, dtype=complex)
    ER = convmat1d(eps_x, P)
    ERi = np.linalg.inv(ER)
    ERx = np.linalg.inv(convmat1d(1.0 / eps_x, P)) if rule == "inverse" else ER
    I = np.eye(P, dtype=complex)
    Pmat = np.block([[KX @ ERi @ KY, I - KX @ ERi @ KX],
                     [KY @ ERi @ KY - I, -KY @ ERi @ KX]])
    Qmat = np.block([[KX @ KY, ER - KX @ KX],
                     [KY @ KY - ERx, -KY @ KX]])
    OM2 = Pmat @ Qmat
    w2, W = eig(OM2)
    lam = np.sqrt(w2.astype(complex))
    lam = np.where(np.real(lam) < 0, -lam, lam)  # Re(lam) >= 0 (stability)
    V = (Qmat @ W) / lam[None, :]                # Qmat @ W @ inv(diag(lam))
    return W, V, lam


def layer_smatrix(W, V, lam, k0, d, Wg, Vg):
    """Scattering matrix of a layer (referenced to the gap medium)."""
    WiWg = np.linalg.solve(W, Wg); ViVg = np.linalg.solve(V, Vg)
    A = WiWg + ViVg
    B = WiWg - ViVg
    x = np.exp(-lam * k0 * d)[:, None]            # diagonal propagator X as a column scale
    AiXB = np.linalg.solve(A, x * B)              # inv(A) @ X @ B
    AiXA = np.linalg.solve(A, x * A)              # inv(A) @ X @ A
    AiB = np.linalg.solve(A, B)                   # inv(A) @ B
    XB = x * B
    D = A - XB @ AiXB
    S = np.linalg.solve(D, np.hstack([XB @ AiXA - B, x * (A - B @ AiB)]))
    n = A.shape[0]
    S11 = S[:, :n]; S12 = S[:, n:]
    return (S11, S12, S12, S11)


def pq_matrices(ER, KX, KY, ERx=None, ERy=None):
    """The P and Q matrices of a layer (Omega^2 = P Q), Rumpf formulation.
    KX, KY are diagonal, so the products are row/column scalings (no dense matmul)."""
    P = KX.shape[0]
    I = np.eye(P, dtype=complex)
    ERi = np.linalg.inv(ER)
    ERx = ER if ERx is None else ERx
    ERy = ER if ERy is None else ERy
    kx, ky = np.diag(KX), np.diag(KY)
    if np.count_nonzero(KX - np.diag(kx)) or np.count_nonzero(KY - np.diag(ky)):
        Pmat = np.block([[KX @ ERi @ KY, I - KX @ ERi @ KX],
                         [KY @ ERi @ KY - I, -KY @ ERi @ KX]])
        Qmat = np.block([[KX @ KY, ERy - KX @ KX],
                         [KY @ KY - ERx, -KY @ KX]])
        return Pmat, Qmat
    sc = lambda a, M, b: a[:, None] * M * b[None, :]           # noqa: E731  diag(a) M diag(b)
    Pmat = np.block([[sc(kx, ERi, ky), I - sc(kx, ERi, kx)],
                     [sc(ky, ERi, ky) - I, -sc(ky, ERi, kx)]])
    Qmat = np.block([[np.diag(kx * ky), ERy - np.diag(kx * kx)],
                     [np.diag(ky * ky) - ERx, -np.diag(ky * kx)]])
    return Pmat, Qmat


def modes_from_PQ(Pmat, Qmat):
    """Eigenmodes (W, V, lam) from P and Q (full or symmetry-reduced)."""
    w2, W = eig(Pmat @ Qmat)
    lam = np.sqrt(w2.astype(complex))
    lam = np.where(np.real(lam) < 0, -lam, lam)
    V = (Qmat @ W) / lam[None, :]
    return W, V, lam


def modes_from_ER(ER, KX, KY, ERx=None, ERy=None):
    """Layer eigenmodes (W, V, lam) from a given convolution matrix ER.
    Works with both 1D (convmat1d) and 2D (convmat2d) ER — the core is the same.
    ERx / ERy: optional Li-factorized matrices multiplying Ex / Ey (see convmat2d_li);
    when omitted the plain Laurent ER is used for both."""
    Pmat, Qmat = pq_matrices(ER, KX, KY, ERx, ERy)
    w2, W = eig(Pmat @ Qmat)
    lam = np.sqrt(w2.astype(complex))
    lam = np.where(np.real(lam) < 0, -lam, lam)
    V = (Qmat @ W) / lam[None, :]                # Qmat @ W @ inv(diag(lam))
    return W, V, lam


_MODE_CACHE = {}

def modes_from_ER_cached(ER, KX, KY, maxsize=256):
    """In-memory cached version of modes_from_ER — B5.
    When the same (ER,KX,KY) layer recurs (e.g. a repeated layer stack), it avoids
    re-solving the eigenmodes. Key: hash of the array bytes."""
    import hashlib
    key = hashlib.md5(ER.tobytes() + np.ascontiguousarray(np.diag(KX)).tobytes()
                      + np.ascontiguousarray(np.diag(KY)).tobytes()).hexdigest()
    if key in _MODE_CACHE:
        return _MODE_CACHE[key]
    res = modes_from_ER(ER, KX, KY)
    if len(_MODE_CACHE) < maxsize:
        _MODE_CACHE[key] = res
    return res
