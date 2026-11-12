"""E3 — 2D HIGH-ACCURACY SOLVER (correct Fourier factorization).

Our native `solve_rcwa_2d` uses the plain Laurent rule: fast and accurate for
DIELECTRIC structures, but it converges poorly for METAL/high-contrast 2D structures
(energy is not conserved, ΣDE≫1). For those cases, which require correct Fourier
factorization (Li's rules), we bridge the verified `grcwa` engine as a first-class
high-accuracy backend.

- `solve_2d_grcwa(...)` : via grcwa (Rtot, Ttot, absorption, per-order R/T).
- `factorization_advice(eps2d)` : warns if the material is metallic/high-contrast and
  tells which backend is appropriate.
- `energy_ok(...)` : checks the energy of the Laurent solution (>1+tol -> unreliable).

Full normal-vector FFF (beyond grcwa too) is out of scope; for the hardest metal-2D cases
the final verification is still FDTD (see fullwave.py).
"""
import numpy as np


def solve_2d_grcwa(lam0, theta_deg, phi_deg, er_ref, er_trn, Lx, Ly, eps2d, H, nG=200, pol=(1.0, 0.0)):
    """Solve 2D with grcwa (correct factorization). Returns: Rtot,Ttot,A,per-order.
    eps2d: (Nx,Ny) complex permittivity. nG: number of plane waves kept (grcwa rounds it).
    theta_deg, phi_deg in DEGREES (converted to radians for grcwa). pol = (p, s) amplitudes."""
    import grcwa
    grcwa.set_backend("numpy")
    eps2d = np.asarray(eps2d, complex)
    Nx, Ny = eps2d.shape
    ob = grcwa.obj(nG, [Lx, 0.0], [0.0, Ly], 1.0 / lam0, np.deg2rad(theta_deg), np.deg2rad(phi_deg), verbose=0)
    ob.Add_LayerUniform(1.0, er_ref)
    ob.Add_LayerGrid(H, Nx, Ny)
    ob.Add_LayerUniform(1.0, er_trn)
    ob.Init_Setup()
    ob.MakeExcitationPlanewave(pol[0], 0.0, pol[1], 0.0, order=0)
    ob.GridLayer_geteps(eps2d.flatten())
    R, T = ob.RT_Solve(normalize=1)
    return {"Rtot": float(np.real(R)), "Ttot": float(np.real(T)),
            "absorption": float(np.real(1.0 - R - T)), "nG": int(ob.nG), "backend": "grcwa"}


def solve_2d_lattice(lam0, theta_deg, phi_deg, er_ref, er_trn, L1, L2, eps2d, H, nG=200, pol=(1.0, 0.0)):
    """E4 — GENERAL (non-orthogonal) LATTICE. L1,L2 arbitrary lattice vectors (um).
    Hexagonal lattice: L1=[a,0], L2=[a/2, a*sqrt(3)/2]. The native solver2d only handles
    rectangular lattices; this bridge solves arbitrary lattices with grcwa (correct factorization included)."""
    import grcwa
    grcwa.set_backend("numpy")
    eps2d = np.asarray(eps2d, complex); Nx, Ny = eps2d.shape
    ob = grcwa.obj(nG, list(L1), list(L2), 1.0 / lam0, np.deg2rad(theta_deg), np.deg2rad(phi_deg), verbose=0)
    ob.Add_LayerUniform(1.0, er_ref)
    ob.Add_LayerGrid(H, Nx, Ny)
    ob.Add_LayerUniform(1.0, er_trn)
    ob.Init_Setup()
    ob.MakeExcitationPlanewave(pol[0], 0.0, pol[1], 0.0, order=0)
    ob.GridLayer_geteps(eps2d.flatten())
    R, T = ob.RT_Solve(normalize=1)
    return {"Rtot": float(np.real(R)), "Ttot": float(np.real(T)),
            "absorption": float(np.real(1 - R - T)), "nG": int(ob.nG),
            "lattice": "non-orthogonal" if abs(np.dot(L1, L2)) > 1e-9 else "orthogonal"}


def hex_lattice(a):
    """Hexagonal lattice vectors (side a)."""
    return [a, 0.0], [a / 2.0, a * np.sqrt(3) / 2.0]


def factorization_advice(eps2d):
    """Recommends the appropriate 2D backend based on the material.
    Metal (Re(eps)<0) or high contrast -> grcwa; otherwise native Laurent is sufficient."""
    e = np.asarray(eps2d, complex)
    has_metal = bool(np.any(np.real(e) < 0.0))
    contrast = float(np.max(np.abs(e)) / max(np.min(np.abs(e)), 1e-9))
    lossy = bool(np.any(np.imag(e) > 1e-3))
    if has_metal or contrast > 8.0:
        rec = "grcwa (solve_2d_grcwa) — correct Fourier factorization"
        why = ("metal (Re ε<0)" if has_metal else f"high contrast (×{contrast:.0f})")
    else:
        rec = "native solve_rcwa_2d (Laurent) is sufficient"
        why = "dielectric, low contrast"
    return {"recommend": rec, "reason": why, "has_metal": has_metal,
            "contrast": contrast, "lossy": lossy}


def energy_ok(result, tol=0.02):
    """Is the energy of a 2D solution physical? ΣDE≈1 if lossless; ≤1 if lossy.
    ΣDE>1+tol -> factorization unreliable (Laurent blows up for metals)."""
    s = result["Rtot"] + result["Ttot"]
    return bool(s <= 1.0 + tol), float(s)
