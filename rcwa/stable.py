"""Stable automatic solver — A1.

For a thick, high-index super-cell, RCWA blows up at certain order counts (M)
because the interface matrix becomes singular (cond(A)~1e16, guided-mode
resonance). This is not specific to us (grcwa blows up too). Proven fix: try a
few values of M and pick the ones that satisfy ENERGY CONSERVATION (ΣDE≈1) —
all valid solutions give the same physics.

Note: the definitive fix for singular M would be the "enhanced transmittance
matrix" (Moharam 1995b); layer splitting does NOT help (tried, worse). So the
practical and reliable route is a condition/energy-conserving automatic sweep.
"""
import numpy as np
from .solver import solve_rcwa_1d


def solve_1d_auto(lam0, theta_deg, er_ref, er_trn, period, layers, base_M,
                  pol=(0., 1.), M_list=None, tol=0.02):
    """Energy-conserving automatic-M 1D RCWA. Returns the stable solution.

    Added to the returned dict: stable_M, sigDE, n_stable, M_tried.
    Raises RuntimeError if no stable solution exists (suggests FDTD).
    """
    if M_list is None:
        cand = {base_M - 20, base_M - 10, base_M, base_M + 10, base_M + 20}
        M_list = sorted(m for m in cand if m >= 8)
    good = []
    for M in M_list:
        r = solve_rcwa_1d(lam0, theta_deg, er_ref, er_trn, period, layers, 2 * M + 1, pol=pol)
        sig = r["Rtot"] + r["Ttot"]
        if np.isfinite(sig) and abs(sig - 1) < tol:
            good.append((abs(sig - 1), M, r, sig))
    if not good:
        raise RuntimeError("No stable M found (none conserved energy). "
                           "Widen M_list or use FDTD/GPU.")
    good.sort(key=lambda t: -t[1])          # energy conservation is the gate; among the
    _, M, r, sig = good[0]                  # stable solutions the largest M is best converged
    r["stable_M"] = M; r["sigDE"] = sig; r["n_stable"] = len(good)
    r["M_tried"] = M_list
    return r


def solve_2d_auto(lam0, theta_deg, phi_deg, er_ref, er_trn, Lx, Ly, layers, base_M,
                  pol=(0., 1.), M_list=None, tol=0.02):
    """Energy-conserving automatic-M 2D RCWA (2D counterpart of A1)."""
    from .solver2d import solve_rcwa_2d
    if M_list is None:
        M_list = sorted({m for m in (base_M - 2, base_M - 1, base_M, base_M + 1) if m >= 3})
    good = []
    for M in M_list:
        r = solve_rcwa_2d(lam0, theta_deg, phi_deg, er_ref, er_trn, Lx, Ly, layers, M, M, pol=pol)
        sig = r["Rtot"] + r["Ttot"]
        if np.isfinite(sig) and abs(sig - 1) < tol:
            good.append((abs(sig - 1), M, r, sig))
    if not good:
        raise RuntimeError("No stable M found (2D); widen M_list or use FDTD.")
    good.sort(key=lambda t: -t[1]); _, M, r, sig = good[0]     # largest stable M
    r["stable_M"] = M; r["sigDE"] = sig; r["n_stable"] = len(good)
    r["M_tried"] = M_list
    return r
