"""Measuring the locally periodic approximation in 2D (roadmap P3) — EXPERIMENTAL.

Status (13-11-2026): first supercell results disagree strongly with the LPA at N = 4
(NA 0.45): LPA (cell-wise constant field) ~0.77 vs full RCWA ~0.45-0.49 into the design order,
but the supercell RCWA is not converged (M per cell 4 -> 6 changes it by 0.035) and M per cell
8 exceeds 4 GB of memory. Confirm with an FDTD supercell before drawing conclusions.

A metagrating of N different pillars along x (one lattice period L each, phases 2*pi*k/N chosen
from a library with the robust rule) deflects normal incidence into one diffraction order at
sin(theta) = lam0 / (n_bg * N * L), i.e. it mimics the local structure of a lens at
NA = lam0 / (N * L). Two predictions of its deflection efficiency are compared:

  * LPA: every pillar keeps its periodic-library t (no coupling to different neighbours) and
    the field is constant over its cell (as in design.FocalField); efficiency into the design
    order = (n_bg/n_sub) * |(1/N) sum_k t_k exp(-i 2 pi k / N)|^2 * sinc^2(pi / N), the last
    factor being the staircase loss of a cell-wise constant field;
  * full RCWA of the N x 1 supercell (all neighbour coupling included).

The difference is the LPA error at that deflection angle. y-polarized light (along the
pillar rows), as in the libraries.
"""
import numpy as np
from . import design as dz, shapes
from .solver2d import solve_rcwa_2d


def metagrating_cells(lib, N, sigma_nm=3.0, nbin=720):
    """Library indices of the N pillars (target phases 2 pi k / N, robust rule)."""
    t = np.asarray(lib["t"], complex)
    stats = dz.size_error_stats(lib, sigma_nm) if sigma_nm else None
    idx, _ = dz.select_atoms(t, 2 * np.pi * np.arange(N) / N, mode="robust" if stats else "complex",
                             nbin=nbin, t_stats=stats)
    return idx


def lpa_deflection(lib, idx):
    """LPA deflection efficiency into the designed order: dict(cellwise, ideal_sampling).
    `cellwise` includes the sinc^2(pi/N) staircase factor of a cell-wise constant field (what
    design.FocalField assumes); `ideal_sampling` omits it."""
    t = np.asarray(lib["t"], complex)[idx]; N = len(idx)
    k = np.arange(N)
    scale = lib.get("n_bg", 1.0) / lib.get("n_sub", 1.0)
    c = np.mean(t * np.exp(-2j * np.pi * k / N))         # phase grows with k (physics convention)
    ideal = float(scale * abs(c) ** 2)
    return {"cellwise": ideal * float(np.sinc(1.0 / N)) ** 2, "ideal_sampling": ideal}


def supercell_deflection(lib, idx, M_per_cell=5, My=None, raster_per_cell=128):
    """Full RCWA of the N x 1 supercell. Returns dict(T_design, T_other, T0, T_total, R_total, ...).
    The solver labels orders by kx_m = kx_inc - m lam/Lx, so a phase that grows with x (physics
    convention, the metagrating's design) sends light into m = -1: T_design = T[m=-1]."""
    L, H = lib["L"], lib["H"]; N = len(idx); n = lib["n"]
    Nc = int(raster_per_cell)
    cells = [shapes.rasterize(lib["shape"], float(lib["sizes"][i]), L, Nc, n_pillar=n, n_bg=lib["n_bg"])
             for i in idx]
    eps = np.concatenate(cells, axis=0)                      # x along axis 0
    Mx = int(M_per_cell) * N + (N // 2); My = int(My or M_per_cell)
    r = solve_rcwa_2d(lib["lam0"], 0, 0, lib["n_sub"] ** 2, lib["n_bg"] ** 2, N * L, L,
                      [(eps, H)], Mx, My, pol=(0., 1.), symmetry=False)
    mx, my = r["orders"]
    T = r["T"]
    def order(m):
        sel = np.where((mx == m) & (my == 0))[0]
        return float(T[sel[0]]) if len(sel) else 0.0
    return {"T_design": order(-1), "T_other": order(1), "T0": order(0), "T_total": r["Ttot"],
            "R_total": r["Rtot"], "Mx": Mx, "My": My}
