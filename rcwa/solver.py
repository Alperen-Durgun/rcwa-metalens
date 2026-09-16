"""Tam 1D RCWA çözücü — Ders 06 & Kod Dersi 04.
Bütün adımları birleştirir: k-vektörleri -> katman özmodları -> S-matris ->
Redheffer star -> yansıma/geçiş genlikleri -> kırınım verimleri.
"""
import numpy as np
from .layer import homogeneous_modes, grating_modes, layer_smatrix
from .redheffer import redheffer_star


def solve_rcwa_1d(lam0, theta_deg, er_ref, er_trn, period, layers, P,
                  pol=(0.0, 1.0), phi_deg=0.0):
    """Planar (1D) RCWA çöz.

    layers : [(eps_x, kalinlik), ...]  her katman bir periyot boyunca örneklenmiş eps(x)
    pol    : (TM_bileşeni, TE_bileşeni)   TE = y ekseni boyunca (düzleme dik) E
    Döndürür: R, T (mertebe başına verim), toplamlar, sıfırıncı mertebe genliği.
    """
    k0 = 2 * np.pi / lam0
    n_ref = np.sqrt(er_ref)
    theta = np.deg2rad(theta_deg); phi = np.deg2rad(phi_deg)
    M = (P - 1) // 2
    p = np.arange(-M, M + 1)
    kx_inc = n_ref * np.sin(theta) * np.cos(phi)
    ky_inc = n_ref * np.sin(theta) * np.sin(phi)
    kz_inc = n_ref * np.cos(theta)
    kx = kx_inc - p * (lam0 / period)
    KX = np.diag(kx.astype(complex))
    KY = ky_inc * np.eye(P, dtype=complex)

    # gap medium (serbest uzay) — tüm katmanlar buna göre bağlanır
    Wg, Vg, _ = homogeneous_modes(KX, KY, 1.0, 1.0)

    # global S: başlangıçta 'bağlantısız' (birim) matris
    Z = np.zeros((2 * P, 2 * P), dtype=complex)
    Ipq = np.eye(2 * P, dtype=complex)
    Sg = (Z.copy(), Ipq.copy(), Ipq.copy(), Z.copy())

    # cihaz katmanları
    for eps_x, d in layers:
        W, V, lam = grating_modes(np.asarray(eps_x, dtype=complex), KX, KY, P)
        Sl = layer_smatrix(W, V, lam, k0, d, Wg, Vg)
        Sg = redheffer_star(Sg, Sl)

    # yansıma tarafı
    Wref, Vref, KZref = homogeneous_modes(KX, KY, er_ref, 1.0)
    Aref = np.linalg.inv(Wg) @ Wref + np.linalg.inv(Vg) @ Vref
    Bref = np.linalg.inv(Wg) @ Wref - np.linalg.inv(Vg) @ Vref
    Ai = np.linalg.inv(Aref)
    Sref = (-Ai @ Bref, 2 * Ai, 0.5 * (Aref - Bref @ Ai @ Bref), Bref @ Ai)

    # geçiş tarafı
    Wtrn, Vtrn, KZtrn = homogeneous_modes(KX, KY, er_trn, 1.0)
    Atrn = np.linalg.inv(Wg) @ Wtrn + np.linalg.inv(Vg) @ Vtrn
    Btrn = np.linalg.inv(Wg) @ Wtrn - np.linalg.inv(Vg) @ Vtrn
    Ti = np.linalg.inv(Atrn)
    Strn = (Btrn @ Ti, 0.5 * (Atrn - Btrn @ Ti @ Btrn), 2 * Ti, -Ti @ Btrn)

    # global bağlantı: yansıma * cihaz * geçiş
    S11, S12, S21, S22 = redheffer_star(redheffer_star(Sref, Sg), Strn)

    # kaynak (sıfırıncı mertebede birim genlik)
    delta = np.zeros(P, dtype=complex); delta[M] = 1.0
    p_tm, s_te = pol
    if np.isclose(theta, 0.0):
        ate = np.array([0.0, 1.0, 0.0]); atm = np.array([1.0, 0.0, 0.0])
    else:
        khat = np.array([np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi), np.cos(theta)])
        ate = np.cross([0, 0, 1.0], khat); ate /= np.linalg.norm(ate)
        atm = np.cross(ate, khat); atm /= np.linalg.norm(atm)
    Ppol = p_tm * atm + s_te * ate
    esrc = np.concatenate([Ppol[0] * delta, Ppol[1] * delta])
    csrc = np.linalg.inv(Wref) @ esrc

    cref = S11 @ csrc
    ctrn = S21 @ csrc
    rt = Wref @ cref; tt = Wtrn @ ctrn
    rx, ry = rt[:P], rt[P:]
    tx, ty = tt[:P], tt[P:]
    rz = -np.linalg.inv(KZref) @ (KX @ rx + KY @ ry)
    tz = -np.linalg.inv(KZtrn) @ (KX @ tx + KY @ ty)

    R = (np.abs(rx) ** 2 + np.abs(ry) ** 2 + np.abs(rz) ** 2) * np.real(np.diag(KZref) / kz_inc)
    T = (np.abs(tx) ** 2 + np.abs(ty) ** 2 + np.abs(tz) ** 2) * np.real(np.diag(KZtrn) / kz_inc)

    return dict(orders=p, R=np.real(R), T=np.real(T),
                Rtot=float(np.real(R.sum())), Ttot=float(np.real(T.sum())),
                zeroth_index=M, tx=tx, ty=ty, rx=rx, ry=ry)
