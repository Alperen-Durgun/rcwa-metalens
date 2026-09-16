"""Skaler 1B RCWA (TE/TM) + Li faktörizasyon kuralları — A2.

1B (planar) grating için hafif, hızlı skaler çözücü. TM polarizasyonunda iki
faktörizasyon kuralı: 'laurent' (standart) ve 'inverse' (Li 1996 ters kural).
Ters kural yüksek-kontrast/metalik yapılarda TM yakınsamasını DRAMATİK hızlandırır
(çok daha az mertebeyle doğru değer). Enerji korunumu (R+T=1) ile doğrulanmış;
inverse rule, grcwa'nın yüksek nG'de bile yaklaşamadığı değere düşük M'de ulaşır.

Kaynak: Li 1996 (inverse rule), Lalanne–Morris 1996 (TM yakınsama).
Bkz. [[📚 RCWA ve Metalens Kaynakçası]], [[Kod Dersi 05 - Doğrulama ve Yakınsama]].
"""
import numpy as np
from scipy.linalg import eig
from .convmat import convmat1d
from .redheffer import redheffer_star


def _layer_S(W, V, q, k0, d, Wg, Vg):
    Wi = np.linalg.inv(W); Vi = np.linalg.inv(V)
    A = Wi @ Wg + Vi @ Vg; B = Wi @ Wg - Vi @ Vg
    X = np.diag(np.exp(-q * k0 * d)); Ai = np.linalg.inv(A)
    D = A - X @ B @ Ai @ X @ B; Di = np.linalg.inv(D)
    S11 = Di @ (X @ B @ Ai @ X @ A - B); S12 = Di @ X @ (A - B @ Ai @ B)
    return (S11, S12, S12, S11)


def _modes(Om2, Vf):
    w2, W = eig(Om2); q = np.sqrt(w2.astype(complex)); q = np.where(np.real(q) < 0, -q, q)
    return W, Vf @ W @ np.diag(q), q


def solve_scalar_1d(lam0, theta_deg, er_ref, er_trn, period, eps_x, thickness, M,
                    pol="TE", rule="inverse", symmetric=False):
    """Skaler 1B RCWA. pol='TE'|'TM'; rule='laurent'|'inverse' (TM için).
    symmetric=True: normal geliş + ayna-simetrik hücrede çift-parite alt-uzayında
    çözer (boyut ~yarı, eig ~3-4x hızlı, sonuç birebir aynı) — B6.
    Döndürür: dict(Rtot, Ttot, R, T, orders)."""
    if symmetric:
        if abs(theta_deg) > 1e-9:
            raise ValueError("symmetric=True yalnız normal gelişte (theta=0) geçerli.")
        return _solve_scalar_1d_sym(lam0, er_ref, er_trn, period, eps_x, thickness, M, pol, rule)
    k0 = 2 * np.pi / lam0; n_ref = np.sqrt(er_ref); th = np.deg2rad(theta_deg)
    m = np.arange(-M, M + 1); P = 2 * M + 1
    kx = n_ref * np.sin(th) - m * (lam0 / period); KX = np.diag(kx.astype(complex))
    I = np.eye(P, dtype=complex)
    E = convmat1d(np.asarray(eps_x, complex), P)
    Wg, Vg, _ = _modes(KX @ KX - I, I)               # gap (serbest uzay)
    if pol == "TE":
        W, V, q = _modes(KX @ KX - E, I)
    else:
        Ei = np.linalg.inv(E); G = KX @ Ei @ KX - I
        if rule == "laurent":
            W, V, q = _modes(E @ G, Ei)
        else:                                        # Li ters kural
            A = convmat1d(1.0 / np.asarray(eps_x, complex), P); Ai = np.linalg.inv(A)
            W, V, q = _modes(Ai @ G, A)

    def homo(er):
        Wh, Vh, qh = _modes(KX @ KX - er * I, I if pol == "TE" else (1.0 / er) * I)
        return Wh, Vh
    Wref, Vref = homo(er_ref); Wtrn, Vtrn = homo(er_trn)
    Ar = np.linalg.inv(Wg) @ Wref + np.linalg.inv(Vg) @ Vref
    Br = np.linalg.inv(Wg) @ Wref - np.linalg.inv(Vg) @ Vref
    Ari = np.linalg.inv(Ar); Sref = (-Ari @ Br, 2 * Ari, 0.5 * (Ar - Br @ Ari @ Br), Br @ Ari)
    At = np.linalg.inv(Wg) @ Wtrn + np.linalg.inv(Vg) @ Vtrn
    Bt = np.linalg.inv(Wg) @ Wtrn - np.linalg.inv(Vg) @ Vtrn
    Ati = np.linalg.inv(At); Strn = (Bt @ Ati, 0.5 * (At - Bt @ Ati @ Bt), 2 * Ati, -Ati @ Bt)
    S = redheffer_star(redheffer_star(Sref, _layer_S(W, V, q, k0, thickness, Wg, Vg)), Strn)

    delta = np.zeros(P, dtype=complex); delta[M] = 1.0
    src = np.linalg.inv(Wref) @ delta
    rf = Wref @ (S[0] @ src); tf = Wtrn @ (S[2] @ src)
    kzr = np.conj(np.sqrt(er_ref - kx ** 2 + 0j)); kzt = np.conj(np.sqrt(er_trn - kx ** 2 + 0j))
    kzi = n_ref * np.cos(th)
    if pol == "TE":
        R = np.abs(rf) ** 2 * np.real(kzr / kzi); T = np.abs(tf) ** 2 * np.real(kzt / kzi)
    else:
        R = np.abs(rf) ** 2 * np.real((kzr / er_ref) / (kzi / er_ref))
        T = np.abs(tf) ** 2 * np.real((kzt / er_trn) / (kzi / er_ref))
    return dict(Rtot=float(np.real(R.sum())), Ttot=float(np.real(T.sum())),
                R=np.real(R), T=np.real(T), orders=m)


def _parity_bases(M):
    """Çift (Te: M+1 sütun) ve tek (To: M sütun) parite baz matrisleri."""
    P = 2 * M + 1
    Te = np.zeros((P, M + 1), dtype=complex); Te[M, 0] = 1.0
    To = np.zeros((P, M), dtype=complex)
    for j in range(1, M + 1):
        Te[M + j, j] = 1 / np.sqrt(2); Te[M - j, j] = 1 / np.sqrt(2)
        To[M + j, j - 1] = 1 / np.sqrt(2); To[M - j, j - 1] = -1 / np.sqrt(2)
    return Te, To


def _solve_scalar_1d_sym(lam0, er_ref, er_trn, period, eps_x, thickness, M, pol, rule):
    """B6: normal gelişte simetrik hücrede çift-parite indirgemesiyle çözüm."""
    k0 = 2 * np.pi / lam0
    m = np.arange(-M, M + 1); P = 2 * M + 1; kx = -m * (lam0 / period)
    Te, To = _parity_bases(M); D = M + 1; I = np.eye(D, dtype=complex)
    KX2e = Te.conj().T @ np.diag((kx ** 2).astype(complex)) @ Te
    E = convmat1d(np.asarray(eps_x, complex), P); Ee = Te.conj().T @ E @ Te
    C = np.zeros((M, D), dtype=complex)
    for j in range(1, M + 1):
        C[j - 1, j] = kx[M + j]
    Wg, Vg, _ = _modes(KX2e - I, I)
    if pol == "TE":
        W, V, q = _modes(KX2e - Ee, I)
    else:
        Eo = To.conj().T @ E @ To; G = C.conj().T @ np.linalg.inv(Eo) @ C - I
        if rule == "laurent":
            W, V, q = _modes(Ee @ G, np.linalg.inv(Ee))
        else:
            A = convmat1d(1.0 / np.asarray(eps_x, complex), P); Ae = Te.conj().T @ A @ Te
            W, V, q = _modes(np.linalg.inv(Ae) @ G, Ae)

    def homo(er):
        Vf = I if pol == "TE" else (1.0 / er) * I
        return _modes(KX2e - er * I, Vf)[:2]
    Wr, Vr = homo(er_ref); Wt, Vt = homo(er_trn)
    Ar = np.linalg.inv(Wg) @ Wr + np.linalg.inv(Vg) @ Vr
    Br = np.linalg.inv(Wg) @ Wr - np.linalg.inv(Vg) @ Vr; Ari = np.linalg.inv(Ar)
    Sref = (-Ari @ Br, 2 * Ari, 0.5 * (Ar - Br @ Ari @ Br), Br @ Ari)
    At = np.linalg.inv(Wg) @ Wt + np.linalg.inv(Vg) @ Vt
    Bt = np.linalg.inv(Wg) @ Wt - np.linalg.inv(Vg) @ Vt; Ati = np.linalg.inv(At)
    Strn = (Bt @ Ati, 0.5 * (At - Bt @ Ati @ Bt), 2 * Ati, -Ati @ Bt)
    S = redheffer_star(redheffer_star(Sref, _layer_S(W, V, q, k0, thickness, Wg, Vg)), Strn)
    delta = np.zeros(D, dtype=complex); delta[0] = 1.0
    src = np.linalg.inv(Wr) @ delta
    rf = Te @ (Wr @ (S[0] @ src)); tf = Te @ (Wt @ (S[2] @ src))
    kzr = np.conj(np.sqrt(er_ref - kx ** 2 + 0j)); kzt = np.conj(np.sqrt(er_trn - kx ** 2 + 0j))
    kzi = np.sqrt(er_ref)
    if pol == "TE":
        R = np.abs(rf) ** 2 * np.real(kzr / kzi); T = np.abs(tf) ** 2 * np.real(kzt / kzi)
    else:
        R = np.abs(rf) ** 2 * np.real((kzr / er_ref) / (kzi / er_ref))
        T = np.abs(tf) ** 2 * np.real((kzt / er_trn) / (kzi / er_ref))
    return dict(Rtot=float(np.real(R.sum())), Ttot=float(np.real(T.sum())),
                R=np.real(R), T=np.real(T), orders=m)
