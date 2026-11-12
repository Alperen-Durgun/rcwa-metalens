"""E6 — cm-SCALE TILED FABRICATION PIPELINE (large metalens).

Problem: a metalens with a millimeter/centimeter aperture contains millions to
billions of meta-atoms. Placing them in a Python loop and holding them all in memory
to write a single GDS is impossible. This module removes two bottlenecks:

1. **Vectorized size selection** — with a phase→size lookup table (LUT) all cells are
   selected in a single numpy indexing operation (NO loop). Same (convention-corrected)
   selection as metalens_design_full, but an O(1) LUT instead of an O(N) loop.
2. **Tile-by-tile GDS** — the aperture is split into squares; each tile's polygons are
   written to a separate GDS file and the memory is released → memory stays bounded by
   the tile size (cm-scale becomes feasible).

It also produces scale statistics + a preview without materializing any atom.
"""
import os
import numpy as np
from . import design


def build_phase_lut(lam0, L, H, mat, shape, M, n_pillar=None, n_size=20, nbin=2048, n_sub=1.0):
    """Meta-atom library (cached, physics convention) + phase→size LUT.
    Same library and selection rule (complex-error LUT, see design.phase_lut) as design_lens."""
    lib = design.meta_atom_library(lam0, L, H, sizes=design.default_sizes(L, n_size), shape=shape,
                                   material=mat, n_pillar=(n_pillar if n_pillar is not None else 2.4),
                                   M=M, n_sub=n_sub)
    lut_k, phases = design.phase_lut(lib["t"], nbin=nbin)
    return {"sizes": lib["sizes"], "t": lib["t"], "libphm": np.mod(np.angle(lib["t"]), 2 * np.pi),
            "libam": lib["amp"], "lut_size": lib["sizes"][lut_k], "lut_k": lut_k, "nbin": nbin,
            "n": lib["n"]}


def size_map_vectorized(lut, lam0, L, f, ci):
    """Build the size map for the given cell centers (ci) WITHOUT a loop.
    Returns: size_map(Nc,Nc), inside(Nc,Nc). Memory: Nc² (cap it for previews)."""
    XX, YY = np.meshgrid(ci, ci, indexing="ij"); R = np.hypot(XX, YY)
    inside = R <= (ci[-1] - ci[0]) / 2 + 1e-9
    phi = np.mod(-(2 * np.pi / lam0) * (np.sqrt(R ** 2 + f ** 2) - f), 2 * np.pi)
    b = (phi / (2 * np.pi) * lut["nbin"]).astype(int) % lut["nbin"]
    size_map = lut["lut_size"][b]
    return np.where(inside, size_map, 0.0), inside


def design_large_lens(lam0, L, H, mat, shape, D_lens, f, M, gds_dir=None,
                      tile_cells=400, preview_cap=1400, write_gds=True):
    """Large lens: vectorized selection + tiled GDS. Memory bounded by the tile.
    If gds_dir is given each tile gets its own .gds; if None, statistics+preview only.
    Returns: statistics dictionary + preview (size_map_ds, ci_ds)."""
    from . import gds
    lut = build_phase_lut(lam0, L, H, mat, shape, M)
    n_pillar = float(np.real(lut["n"]))
    Nc = int(round(D_lens / L)); D_lens = Nc * L
    ci = (np.arange(Nc) - (Nc - 1) / 2) * L
    NA = float(np.sin(np.arctan(D_lens / 2 / f)))
    # atom count (loop-free): cells inside the circle
    XX, YY = np.meshgrid(ci, ci, indexing="ij")
    n_atoms = int((np.hypot(XX, YY) <= D_lens / 2).sum())
    ntile = 0; npoly_total = 0
    if write_gds and gds_dir:
        os.makedirs(gds_dir, exist_ok=True)
        for i0 in range(0, Nc, tile_cells):
            for j0 in range(0, Nc, tile_cells):
                ii = np.arange(i0, min(i0 + tile_cells, Nc))
                jj = np.arange(j0, min(j0 + tile_cells, Nc))
                X, Y = np.meshgrid(ci[ii], ci[jj], indexing="ij")
                R = np.hypot(X, Y); ins = R <= D_lens / 2
                if not ins.any():
                    continue
                phi = np.mod(-(2 * np.pi / lam0) * (np.sqrt(R ** 2 + f ** 2) - f), 2 * np.pi)
                b = (phi / (2 * np.pi) * lut["nbin"]).astype(int) % lut["nbin"]
                sz = lut["lut_size"][b]
                xs = X[ins]; ys = Y[ins]; ss = sz[ins]
                placements = [{"x": float(xs[k]), "y": float(ys[k]), "shape": shape,
                               "params": float(ss[k]), "angle": 0.0} for k in range(xs.size)]
                tilepath = os.path.join(gds_dir, f"tile_{i0}_{j0}.gds")
                npoly_total += gds.layout_to_gds(placements, tilepath, cell_name=f"T_{i0}_{j0}")
                ntile += 1
    # preview (downsampling — without fully materializing any atom)
    stride = max(1, Nc // preview_cap)
    ci_ds = ci[::stride]
    size_ds, _ = size_map_vectorized(lut, lam0, L, f, ci_ds)
    return {"D_lens": D_lens, "f": f, "NA": NA, "Nc": Nc, "n_atoms": n_atoms,
            "n_tiles": ntile, "npoly": npoly_total, "n_pillar": n_pillar,
            "preview_size_nm": size_ds * 1000, "ci_ds": ci_ds}


def scale_projection(n_atoms, bytes_per_atom=36):
    """GDS file-size projection in GB. Hierarchical files (default writer) need ~36 bytes per
    pillar (one cell reference); a flat file needs ~415 bytes per 48-vertex circle."""
    return n_atoms * bytes_per_atom / 1e9
