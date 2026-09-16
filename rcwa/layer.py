"""Katman çözümü — Ders 04 & Kod Dersi 02.
Homojen ve gratingli katmanlar için özmodları (W, V, LAM) ve katman
scattering matrisini üretir. Rumpf (EMPossible) formülasyonu, gap medium ile.
"""
import numpy as np
from scipy.linalg import eig
from .convmat import convmat1d


def homogeneous_modes(KX, KY, er, ur=1.0):
    """Homojen bir bölgenin özmodları (W=I, V, KZ)."""
    P = KX.shape[0]
    I = np.eye(P, dtype=complex)
    Z = np.zeros((P, P), dtype=complex)
    Q = (1.0 / ur) * np.block([[KX @ KY, ur * er * I - KX @ KX],
                               [KY @ KY - ur * er * I, -KY @ KX]])
    KZ = np.conj(np.sqrt((er * ur) * I - KX @ KX - KY @ KY))
    W = np.eye(2 * P, dtype=complex)
    LAM = 1j * np.block([[KZ, Z], [Z, KZ]])
    V = Q @ np.linalg.inv(LAM)
    return W, V, KZ


def grating_modes(eps_x, KX, KY, P):
    """Gratingli katmanın özmod çözümü: P,Q -> Omega^2 -> eig."""
    ER = convmat1d(eps_x, P)
    ERi = np.linalg.inv(ER)
    I = np.eye(P, dtype=complex)
    Pmat = np.block([[KX @ ERi @ KY, I - KX @ ERi @ KX],
                     [KY @ ERi @ KY - I, -KY @ ERi @ KX]])
    Qmat = np.block([[KX @ KY, ER - KX @ KX],
                     [KY @ KY - ER, -KY @ KX]])
    OM2 = Pmat @ Qmat
    w2, W = eig(OM2)
    lam = np.sqrt(w2.astype(complex))
    lam = np.where(np.real(lam) < 0, -lam, lam)  # Re(lam) >= 0 (kararlılık)
    LAM = np.diag(lam)
    V = Qmat @ W @ np.linalg.inv(LAM)
    return W, V, lam


def layer_smatrix(W, V, lam, k0, d, Wg, Vg):
    """Bir katmanın scattering matrisi (gap medium referansıyla)."""
    Wi = np.linalg.inv(W); Vi = np.linalg.inv(V)
    A = Wi @ Wg + Vi @ Vg
    B = Wi @ Wg - Vi @ Vg
    X = np.diag(np.exp(-lam * k0 * d))
    Ai = np.linalg.inv(A)
    D = A - X @ B @ Ai @ X @ B
    Di = np.linalg.inv(D)
    S11 = Di @ (X @ B @ Ai @ X @ A - B)
    S12 = Di @ X @ (A - B @ Ai @ B)
    return (S11, S12, S12, S11)


def modes_from_ER(ER, KX, KY):
    """Verilen konvolüsyon matrisi ER'den katman özmodları (W, V, lam).
    Hem 1B (convmat1d) hem 2B (convmat2d) ER ile çalışır — çekirdek aynı."""
    P = KX.shape[0]
    I = np.eye(P, dtype=complex)
    ERi = np.linalg.inv(ER)
    Pmat = np.block([[KX @ ERi @ KY, I - KX @ ERi @ KX],
                     [KY @ ERi @ KY - I, -KY @ ERi @ KX]])
    Qmat = np.block([[KX @ KY, ER - KX @ KX],
                     [KY @ KY - ER, -KY @ KX]])
    w2, W = eig(Pmat @ Qmat)
    lam = np.sqrt(w2.astype(complex))
    lam = np.where(np.real(lam) < 0, -lam, lam)
    V = Qmat @ W @ np.linalg.inv(np.diag(lam))
    return W, V, lam


_MODE_CACHE = {}

def modes_from_ER_cached(ER, KX, KY, maxsize=256):
    """modes_from_ER'in bellek-içi önbellekli sürümü — B5.
    Aynı (ER,KX,KY) katman tekrar geçtiğinde (ör. tekrarlı katman yığını) özmodları
    yeniden çözmez. Anahtar: dizi baytlarının hash'i."""
    import hashlib
    key = hashlib.md5(ER.tobytes() + np.ascontiguousarray(np.diag(KX)).tobytes()
                      + np.ascontiguousarray(np.diag(KY)).tobytes()).hexdigest()
    if key in _MODE_CACHE:
        return _MODE_CACHE[key]
    res = modes_from_ER(ER, KX, KY)
    if len(_MODE_CACHE) < maxsize:
        _MODE_CACHE[key] = res
    return res
