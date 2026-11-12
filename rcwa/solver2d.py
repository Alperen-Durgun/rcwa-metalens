"""Full 2D (doubly periodic) RCWA solver — 2D Lesson series & 2D Code.

Same skeleton as the 1D solver; the difference: the structure is periodic in
both x and y, and the fields are vectorial (TE/TM do not decouple, they are
coupled). Kx and Ky are built from a two-dimensional order grid, and the
material is represented via convmat2d. Validation: grcwa (Fan/Stanford)
and energy conservation.
"""
import numpy as np
from .convmat import convmat2d, convmat2d_li
from .layer import homogeneous_modes, modes_from_ER, layer_smatrix, rayleigh_safe_scale
from .redheffer import redheffer_star


def solve_rcwa_2d(lam0, theta_deg, phi_deg, er_ref, er_trn, Lx, Ly, layers, Mx, My, pol=(0., 1.),
                  rule="li"):
    """Solve 2D RCWA.

    layers : [(eps2d, thickness), ...]  eps2d = (Nx, Ny) unit-cell sampling
    Mx, My : half-number of orders kept in x/y (total (2Mx+1)(2My+1))
    pol    : (TM, TE) components
    rule   : "li" (default; Li 1997 factorization, see convmat2d_li — far faster convergence
             at high index contrast) or "laurent" (plain rule, kept for comparison)

    Material convention: permittivities follow the usual optics convention eps = (n + ik)^2,
    i.e. Im(eps) > 0 means ABSORPTION (as in refractiveindex.info, grcwa, Tidy3D). The solver
    itself works in exp(+j*omega*t), so the inputs are conjugated internally. Output amplitudes
    are in the solver convention — see rcwa/convention.py.
    """
    er_ref = np.conj(complex(er_ref)); er_trn = np.conj(complex(er_trn))
    layers = [(np.conj(np.asarray(e, dtype=complex)), d) for e, d in layers]
    k0 = 2 * np.pi / lam0
    n_ref = np.sqrt(er_ref)
    th = np.deg2rad(theta_deg); ph = np.deg2rad(phi_deg)
    MX, MY = np.meshgrid(np.arange(-Mx, Mx + 1), np.arange(-My, My + 1), indexing='ij')
    MXf, MYf = MX.flatten(), MY.flatten()
    P = (2 * Mx + 1) * (2 * My + 1)
    kx_inc = n_ref * np.sin(th) * np.cos(ph)
    ky_inc = n_ref * np.sin(th) * np.sin(ph)
    kz_inc = n_ref * np.cos(th)
    kxv0 = kx_inc - MXf * (lam0 / Lx); kyv0 = ky_inc - MYf * (lam0 / Ly)
    s = rayleigh_safe_scale(kxv0, kyv0, (1.0, er_ref, er_trn))
    KX = np.diag((s * kxv0).astype(complex))
    KY = np.diag((s * kyv0).astype(complex))
    Px, Py = 2 * Mx + 1, 2 * My + 1

    Wg, Vg, _ = homogeneous_modes(KX, KY, 1.0, 1.0)
    Vgi = np.linalg.inv(Vg)
    WgiW = lambda Wm: Wm                      # the gap modes are W = I, so inv(Wg) @ W = W
    Z = np.zeros((2 * P, 2 * P), complex); Ipq = np.eye(2 * P, dtype=complex)
    Sg = None                                 # None = identity (no device layer yet)

    for eps2d, d in layers:
        if rule == "li":
            # NOTE: sample eps2d finely (>= ~256 per side): a coarse binary raster quantizes
            # the feature size and that, not the order count, then limits the accuracy.
            ER, ERx, ERy = convmat2d_li(np.asarray(eps2d, dtype=complex), Px, Py)
            W, V, lam = modes_from_ER(ER, KX, KY, ERx, ERy)
        else:
            ER = convmat2d(np.asarray(eps2d, dtype=complex), Px, Py)
            W, V, lam = modes_from_ER(ER, KX, KY)
        Sl = layer_smatrix(W, V, lam, k0, d, Wg, Vg)
        Sg = Sl if Sg is None else redheffer_star(Sg, Sl)
    if Sg is None:
        Sg = (Z.copy(), Ipq.copy(), Ipq.copy(), Z.copy())

    Wref, Vref, KZref = homogeneous_modes(KX, KY, er_ref, 1.0)
    Aref = WgiW(Wref) + Vgi @ Vref
    Bref = WgiW(Wref) - Vgi @ Vref
    Ai = np.linalg.inv(Aref)
    Sref = (-Ai @ Bref, 2 * Ai, 0.5 * (Aref - Bref @ Ai @ Bref), Bref @ Ai)

    Wtrn, Vtrn, KZtrn = homogeneous_modes(KX, KY, er_trn, 1.0)
    Atrn = WgiW(Wtrn) + Vgi @ Vtrn
    Btrn = WgiW(Wtrn) - Vgi @ Vtrn
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
    csrc = np.linalg.solve(Wref, esrc)

    cref = S11 @ csrc; ctrn = S21 @ csrc
    rt = Wref @ cref; tt = Wtrn @ ctrn
    rx, ry = rt[:P], rt[P:]; tx, ty = tt[:P], tt[P:]
    kxv, kyv = np.diag(KX), np.diag(KY)
    rz = -(kxv * rx + kyv * ry) / np.diag(KZref)
    tz = -(kxv * tx + kyv * ty) / np.diag(KZtrn)
    R = (np.abs(rx) ** 2 + np.abs(ry) ** 2 + np.abs(rz) ** 2) * np.real(np.diag(KZref) / kz_inc)
    T = (np.abs(tx) ** 2 + np.abs(ty) ** 2 + np.abs(tz) ** 2) * np.real(np.diag(KZtrn) / kz_inc)
    return dict(Rtot=float(np.real(R.sum())), Ttot=float(np.real(T.sum())),
                R=np.real(R), T=np.real(T), i0=i0, zeroth_index=i0,
                tx=tx, ty=ty, rx=rx, ry=ry, orders=(MXf, MYf))
