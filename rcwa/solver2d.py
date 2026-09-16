"""Tam 2B (iki-yönlü periyodik) RCWA çözücü — 2B Ders serisi & Kod 2B.

1B çözücüyle aynı iskelet; farkı: yapı hem x hem y'de periyodik, alanlar
vektörel (TE/TM ayrışmaz, kuplajlı). Kx ve Ky iki-boyutlu mertebe ızgarasından
kurulur, malzeme convmat2d ile temsil edilir. Doğrulama: grcwa (Fan/Stanford)
ve enerji korunumu.
"""
import numpy as np
from .convmat import convmat2d
from .layer import homogeneous_modes, modes_from_ER, layer_smatrix
from .redheffer import redheffer_star


def solve_rcwa_2d(lam0, theta_deg, phi_deg, er_ref, er_trn, Lx, Ly, layers, Mx, My, pol=(0., 1.)):
    """2B RCWA çöz.

    layers : [(eps2d, kalinlik), ...]  eps2d = (Nx, Ny) birim hücre örneklemesi
    Mx, My : x/y yönünde tutulan mertebe yarı-sayısı (toplam (2Mx+1)(2My+1))
    pol    : (TM, TE) bileşenleri
    """
    k0 = 2 * np.pi / lam0
    n_ref = np.sqrt(er_ref)
    th = np.deg2rad(theta_deg); ph = np.deg2rad(phi_deg)
    MX, MY = np.meshgrid(np.arange(-Mx, Mx + 1), np.arange(-My, My + 1), indexing='ij')
    MXf, MYf = MX.flatten(), MY.flatten()
    P = (2 * Mx + 1) * (2 * My + 1)
    kx_inc = n_ref * np.sin(th) * np.cos(ph)
    ky_inc = n_ref * np.sin(th) * np.sin(ph)
    kz_inc = n_ref * np.cos(th)
    KX = np.diag((kx_inc - MXf * (lam0 / Lx)).astype(complex))
    KY = np.diag((ky_inc - MYf * (lam0 / Ly)).astype(complex))
    Px, Py = 2 * Mx + 1, 2 * My + 1

    Wg, Vg, _ = homogeneous_modes(KX, KY, 1.0, 1.0)
    Z = np.zeros((2 * P, 2 * P), complex); Ipq = np.eye(2 * P, dtype=complex)
    Sg = (Z.copy(), Ipq.copy(), Ipq.copy(), Z.copy())

    for eps2d, d in layers:
        ER = convmat2d(np.asarray(eps2d, dtype=complex), Px, Py)
        W, V, lam = modes_from_ER(ER, KX, KY)
        Sg = redheffer_star(Sg, layer_smatrix(W, V, lam, k0, d, Wg, Vg))

    Wref, Vref, KZref = homogeneous_modes(KX, KY, er_ref, 1.0)
    Aref = np.linalg.inv(Wg) @ Wref + np.linalg.inv(Vg) @ Vref
    Bref = np.linalg.inv(Wg) @ Wref - np.linalg.inv(Vg) @ Vref
    Ai = np.linalg.inv(Aref)
    Sref = (-Ai @ Bref, 2 * Ai, 0.5 * (Aref - Bref @ Ai @ Bref), Bref @ Ai)

    Wtrn, Vtrn, KZtrn = homogeneous_modes(KX, KY, er_trn, 1.0)
    Atrn = np.linalg.inv(Wg) @ Wtrn + np.linalg.inv(Vg) @ Vtrn
    Btrn = np.linalg.inv(Wg) @ Wtrn - np.linalg.inv(Vg) @ Vtrn
    Ti = np.linalg.inv(Atrn)
    Strn = (Btrn @ Ti, 0.5 * (Atrn - Btrn @ Ti @ Btrn), 2 * Ti, -Ti @ Btrn)

    S11, S12, S21, S22 = redheffer_star(redheffer_star(Sref, Sg), Strn)

    i0 = int(np.where((MXf == 0) & (MYf == 0))[0][0])
    delta = np.zeros(P, dtype=complex); delta[i0] = 1.0
    p_tm, s_te = pol
    if np.isclose(th, 0.0):
        ate = np.array([0., 1., 0.]); atm = np.array([1., 0., 0.])
    else:
        khat = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        ate = np.cross([0, 0, 1.], khat); ate /= np.linalg.norm(ate)
        atm = np.cross(ate, khat); atm /= np.linalg.norm(atm)
    Ppol = p_tm * atm + s_te * ate
    esrc = np.concatenate([Ppol[0] * delta, Ppol[1] * delta])
    csrc = np.linalg.inv(Wref) @ esrc

    cref = S11 @ csrc; ctrn = S21 @ csrc
    rt = Wref @ cref; tt = Wtrn @ ctrn
    rx, ry = rt[:P], rt[P:]; tx, ty = tt[:P], tt[P:]
    rz = -np.linalg.inv(KZref) @ (KX @ rx + KY @ ry)
    tz = -np.linalg.inv(KZtrn) @ (KX @ tx + KY @ ty)
    R = (np.abs(rx) ** 2 + np.abs(ry) ** 2 + np.abs(rz) ** 2) * np.real(np.diag(KZref) / kz_inc)
    T = (np.abs(tx) ** 2 + np.abs(ty) ** 2 + np.abs(tz) ** 2) * np.real(np.diag(KZtrn) / kz_inc)
    return dict(Rtot=float(np.real(R.sum())), Ttot=float(np.real(T.sum())),
                R=np.real(R), T=np.real(T), i0=i0, tx=tx, ty=ty, orders=(MXf, MYf))
