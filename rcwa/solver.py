"""Full 1D RCWA solver — Lesson 06 & Coding Lesson 04.
Combines all steps: k-vectors -> layer eigenmodes -> S-matrix ->
Redheffer star -> reflection/transmission amplitudes -> diffraction efficiencies.
"""
import numpy as np
from .layer import homogeneous_modes, grating_modes, layer_smatrix, rayleigh_safe_scale
from .redheffer import redheffer_star


def solve_rcwa_1d(lam0, theta_deg, er_ref, er_trn, period, layers, P,
                  pol=(0.0, 1.0), phi_deg=0.0, rule="inverse"):
    """Solve planar (1D) RCWA.

    layers : [(eps_x, thickness), ...]  each layer is eps(x) sampled over one period
    pol    : (TM_component, TE_component)   TE = E along the y axis (out of plane)
    rule   : "inverse" (Li factorization, fast TM convergence; default) or "laurent"
    Returns: R, T (per-order efficiency), totals, zeroth-order amplitude.

    Material convention: permittivities follow the usual optics convention eps = (n + ik)^2,
    i.e. Im(eps) > 0 means ABSORPTION (as in refractiveindex.info, grcwa, Tidy3D). The solver
    itself works in exp(+j*omega*t), so the inputs are conjugated internally. Output amplitudes
    are in the solver convention — see rcwa/convention.py.
    """
    er_ref = np.conj(complex(er_ref)); er_trn = np.conj(complex(er_trn))
    layers = [(np.conj(np.asarray(e, dtype=complex)), d) for e, d in layers]
    k0 = 2 * np.pi / lam0
    n_ref = np.sqrt(er_ref)
    theta = np.deg2rad(theta_deg); phi = np.deg2rad(phi_deg)
    M = (P - 1) // 2
    p = np.arange(-M, M + 1)
    kx_inc = n_ref * np.sin(theta) * np.cos(phi)
    ky_inc = n_ref * np.sin(theta) * np.sin(phi)
    kz_inc = n_ref * np.cos(theta)
    kx = kx_inc - p * (lam0 / period)
    s = rayleigh_safe_scale(kx, ky_inc, (1.0, er_ref, er_trn))
    KX = np.diag((s * kx).astype(complex))
    KY = (s * ky_inc) * np.eye(P, dtype=complex)

    # gap medium (free space) — all layers are connected relative to this
    Wg, Vg, _ = homogeneous_modes(KX, KY, 1.0, 1.0)
    Vgi = np.linalg.inv(Vg)
    WgiW = lambda Wm: Wm                      # the gap modes are W = I, so inv(Wg) @ W = W

    # global S: initially the 'disconnected' (identity) matrix
    Z = np.zeros((2 * P, 2 * P), dtype=complex)
    Ipq = np.eye(2 * P, dtype=complex)
    Sg = None                                 # None = identity (no device layer yet)

    # device layers
    for eps_x, d in layers:
        W, V, lam = grating_modes(np.asarray(eps_x, dtype=complex), KX, KY, P, rule=rule)
        Sl = layer_smatrix(W, V, lam, k0, d, Wg, Vg)
        Sg = Sl if Sg is None else redheffer_star(Sg, Sl)
    if Sg is None:
        Sg = (Z.copy(), Ipq.copy(), Ipq.copy(), Z.copy())

    # reflection side
    Wref, Vref, KZref = homogeneous_modes(KX, KY, er_ref, 1.0)
    Aref = WgiW(Wref) + Vgi @ Vref
    Bref = WgiW(Wref) - Vgi @ Vref
    Ai = np.linalg.inv(Aref)
    Sref = (-Ai @ Bref, 2 * Ai, 0.5 * (Aref - Bref @ Ai @ Bref), Bref @ Ai)

    # transmission side
    Wtrn, Vtrn, KZtrn = homogeneous_modes(KX, KY, er_trn, 1.0)
    Atrn = WgiW(Wtrn) + Vgi @ Vtrn
    Btrn = WgiW(Wtrn) - Vgi @ Vtrn
    Ti = np.linalg.inv(Atrn)
    Strn = (Btrn @ Ti, 0.5 * (Atrn - Btrn @ Ti @ Btrn), 2 * Ti, -Ti @ Btrn)

    # global connection: reflection * device * transmission
    S11, S12, S21, S22 = redheffer_star(redheffer_star(Sref, Sg), Strn)

    # source (unit amplitude in the zeroth order)
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
    csrc = np.linalg.solve(Wref, esrc)

    cref = S11 @ csrc
    ctrn = S21 @ csrc
    rt = Wref @ cref; tt = Wtrn @ ctrn
    rx, ry = rt[:P], rt[P:]
    tx, ty = tt[:P], tt[P:]
    kxv, kyv = np.diag(KX), np.diag(KY)
    rz = -(kxv * rx + kyv * ry) / np.diag(KZref)
    tz = -(kxv * tx + kyv * ty) / np.diag(KZtrn)

    R = (np.abs(rx) ** 2 + np.abs(ry) ** 2 + np.abs(rz) ** 2) * np.real(np.diag(KZref) / kz_inc)
    T = (np.abs(tx) ** 2 + np.abs(ty) ** 2 + np.abs(tz) ** 2) * np.real(np.diag(KZtrn) / kz_inc)

    return dict(orders=p, R=np.real(R), T=np.real(T),
                Rtot=float(np.real(R.sum())), Ttot=float(np.real(T.sum())),
                zeroth_index=M, i0=M, tx=tx, ty=ty, rx=rx, ry=ry)
