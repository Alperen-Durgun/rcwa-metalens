"""Automated tests.
Run:  python tests/test_rcwa.py   (pytest works too)
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import solve_rcwa_1d


def _fresnel_slab_R(lam0, th, n1, d, pol):
    thr = np.deg2rad(th); s = np.sin(thr)
    c0 = np.cos(thr); c1 = np.sqrt(1 - (s / n1) ** 2 + 0j)
    e0 = c0 if pol == 'TE' else 1 / c0
    e1 = n1 * c1 if pol == 'TE' else n1 / c1
    dl = 2 * np.pi / lam0 * n1 * c1 * d
    m11 = np.cos(dl); m12 = 1j * np.sin(dl) / e1
    m21 = 1j * e1 * np.sin(dl); m22 = np.cos(dl)
    r = (e0 * (m11 + m12 * e0) - (m21 + m22 * e0)) / (e0 * (m11 + m12 * e0) + (m21 + m22 * e0))
    return abs(r) ** 2


def test_fresnel_match():
    lam0, P, n1, d = 0.55, 21, 1.5, 0.40
    for pol, pv in [('TE', (0., 1.)), ('TM', (1., 0.))]:
        for th in (0., 30., 55.):
            eps = np.full(512, n1 ** 2, complex)
            res = solve_rcwa_1d(lam0, th, 1.0, 1.0, 0.30, [(eps, d)], P, pol=pv)
            R0 = res['R'][res['zeroth_index']]
            assert abs(R0 - _fresnel_slab_R(lam0, th, n1, d, pol)) < 1e-6


def test_energy_conservation():
    lam0, period, d = 0.55, 1.20, 0.30
    x = (np.arange(1024) + 0.5) / 1024
    eps = np.where(x < 0.5, 4.0, 1.0).astype(complex)
    for pv in [(0., 1.), (1., 0.)]:
        r = solve_rcwa_1d(lam0, 15., 1.0, 2.25, period, [(eps, d)], 41, pol=pv)
        assert abs(r['Rtot'] + r['Ttot'] - 1.0) < 1e-6



def test_energy_conservation_2d():
    from rcwa import solve_rcwa_2d
    Lx = Ly = 1.0
    x = (np.arange(96) + 0.5) / 96 * Lx
    X, Y = np.meshgrid(x, x, indexing='ij')
    eps = np.where((np.abs(X - .5) < .25) & (np.abs(Y - .5) < .25), 4.0, 1.0)
    r = solve_rcwa_2d(0.55, 0, 0, 1.0, 2.25, Lx, Ly, [(eps, 0.30)], 7, 7, pol=(0., 1.))
    assert abs(r['Rtot'] + r['Ttot'] - 1.0) < 1e-6



def test_materials():
    from rcwa import materials
    import numpy as np
    assert abs(np.sqrt(materials.eps_of("SiO2", 0.633)).real - 1.457) < 0.01
    assert abs(materials.eps_of("air", 0.633) - 1.0) < 1e-9
    assert np.imag(np.sqrt(materials.eps_of("Si", 0.633))) > 0  # loss


def test_cache_computes_once():
    """Second call with the same key is served from disk; a different key recomputes.
    Uses a private temporary cache folder (never the user's real cache)."""
    from rcwa import cache
    import tempfile, shutil
    calls = {"n": 0}
    def comp():
        calls["n"] += 1; return {"x": np.arange(4.0)}
    old = cache.CACHE_DIR; cache.CACHE_DIR = tempfile.mkdtemp(prefix="rcwa_cache_ut_")
    try:
        a = cache.cached_library("unittest", {"v": 1}, comp)
        b = cache.cached_library("unittest", {"v": 1}, comp)
        assert calls["n"] == 1 and np.array_equal(a["x"], b["x"])
        cache.cached_library("unittest", {"v": 2}, comp)
        assert calls["n"] == 2 and cache.clear_cache() == 2
    finally:
        shutil.rmtree(cache.CACHE_DIR, ignore_errors=True); cache.CACHE_DIR = old


def test_solve_1d_auto_stable():
    from rcwa import solve_1d_auto
    import numpy as np
    x = (np.arange(256) + 0.5) / 256
    eps = np.where(x < 0.5, 4.0, 1.0).astype(complex)
    r = solve_1d_auto(0.55, 15.0, 1.0, 2.25, 1.2, [(eps, 0.3)], base_M=40, pol=(0., 1.))
    assert abs(r["sigDE"] - 1) < 0.02 and "stable_M" in r




def test_tm_inverse_rule():
    from rcwa import solve_scalar_1d
    import numpy as np
    x = (np.arange(1024) + 0.5) / 1024
    eps = np.where(x < 0.5, 9.0, 1.0)
    lo = solve_scalar_1d(0.55, 0, 1, 1, 1.0, eps, 0.25, 8, "TM", "laurent")
    hi = solve_scalar_1d(0.55, 0, 1, 1, 1.0, eps, 0.25, 8, "TM", "inverse")
    ref = solve_scalar_1d(0.55, 0, 1, 1, 1.0, eps, 0.25, 90, "TM", "inverse")["Ttot"]
    # energy conservation
    assert abs(hi["Rtot"] + hi["Ttot"] - 1) < 1e-6
    # the inverse rule is closer to the reference at low M
    assert abs(hi["Ttot"] - ref) < abs(lo["Ttot"] - ref)


def test_symmetry_reduction():
    from rcwa import solve_scalar_1d
    import numpy as np
    x = (np.arange(2048) + 0.5) / 2048
    eps = np.where(np.minimum(x, 1 - x) < 0.25, 9.0, 1.0)  # symmetric about x=0
    for pol in ("TE", "TM"):
        full = solve_scalar_1d(0.55, 0, 1, 1, 1.0, eps, 0.25, 40, pol, "inverse")
        sym = solve_scalar_1d(0.55, 0, 1, 1, 1.0, eps, 0.25, 40, pol, "inverse", symmetric=True)
        assert abs(sym["Rtot"] + sym["Ttot"] - 1) < 1e-6       # energy
        assert abs(full["Ttot"] - sym["Ttot"]) < 1e-4          # identical




def test_convmat_correctness():
    from rcwa import convmat1d
    import numpy as np
    x = (np.arange(1024) + 0.5) / 1024
    duty, ehi, elo = 0.5, 4.0, 1.0
    eps = np.where(x < duty, ehi, elo)
    C = convmat1d(eps, 11)
    # 0th Fourier coefficient = mean eps
    assert abs(C[5, 5] - (duty * ehi + (1 - duty) * elo)) < 1e-3
    # Toeplitz: C[i,j] depends only on i-j
    assert abs(C[0, 1] - C[5, 6]) < 1e-9


def test_convergence_energy_stable():
    from rcwa import solve_rcwa_1d
    import numpy as np
    x = (np.arange(1024) + 0.5) / 1024
    eps = np.where(x < 0.5, 4.0, 1.0).astype(complex)
    for M in (11, 21, 41):
        r = solve_rcwa_1d(0.55, 10.0, 1.0, 2.25, 1.2, [(eps, 0.3)], 2 * M + 1, pol=(0., 1.))
        assert abs(r["Rtot"] + r["Ttot"] - 1) < 1e-6


def test_jones_isotropic_diagonal():
    from rcwa import solve_rcwa_2d
    import numpy as np
    L = 0.35; x = (np.arange(64) + 0.5) / 64 * L
    X, Y = np.meshgrid(x, x, indexing="ij")
    eps = np.where((np.abs(X - L/2) < 0.09) & (np.abs(Y - L/2) < 0.09), 2.4**2, 1.0)  # square (isotropic)
    rx = solve_rcwa_2d(0.633, 0, 0, 1, 1, L, L, [(eps, 0.6)], 6, 6, pol=(1., 0.))
    ry = solve_rcwa_2d(0.633, 0, 0, 1, 1, L, L, [(eps, 0.6)], 6, 6, pol=(0., 1.))
    txy = abs(ry["tx"][ry["i0"]]); tyx = abs(rx["ty"][rx["i0"]])
    assert txy < 1e-3 and tyx < 1e-3   # cross-polarization ~0


def test_regression_binary_grating():
    from rcwa import solve_rcwa_1d
    import numpy as np
    x = (np.arange(1024) + 0.5) / 1024
    eps = np.where(x < 0.5, 4.0, 1.0).astype(complex)
    r = solve_rcwa_1d(0.55, 15.0, 1.0, 2.25, 1.2, [(eps, 0.3)], 81, pol=(0., 1.))
    # stored reference value (regression): 0.9180 (verified with grcwa)
    assert abs(r["Ttot"] - 0.9180) < 2e-3




def test_gds_roundtrip():
    from rcwa import gds
    import tempfile, os
    pl = [{"x": i*0.35, "y": j*0.35, "shape": "circle", "params": 0.2, "angle": 0} for i in range(3) for j in range(3)]
    fp = os.path.join(tempfile.gettempdir(), "rcwa_test.gds")
    n = gds.layout_to_gds(pl, fp)
    assert n == 9 and gds.read_gds_summary(fp)["polygons"] == 9


def test_gds_unicode_path():
    """Regression: gdstk cannot write to a path with non-ASCII characters on Windows;
    gds.py must still succeed. Uses non-ASCII (Turkish) letters in the path on purpose."""
    from rcwa import gds
    import tempfile, os, shutil
    base = os.path.join(tempfile.gettempdir(), "rcwa_Ki\u015fisel_\u00e7\u0131\u011f_\u0130\u015f_\u00f6\u00e7\u00fc")
    os.makedirs(base, exist_ok=True)
    fp = os.path.join(base, "maske_\u015fablon.gds")
    try:
        pl = [{"x": i*0.35, "y": j*0.35, "shape": "circle", "params": 0.2, "angle": 0} for i in range(2) for j in range(2)]
        n = gds.layout_to_gds(pl, fp)
        assert n == 4 and os.path.exists(fp)
        assert gds.read_gds_summary(fp)["polygons"] == 4
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_gdswriter_geometry():
    """Pure-Python GDSII writer: readable by gdstk + area identical to a gdstk-written file."""
    import gdstk, tempfile, os
    from rcwa import gds
    from rcwa.shapes import polygon
    pl = [{"x": i * 0.35, "y": j * 0.35, "shape": "circle", "params": 0.2, "angle": 0} for i in range(3) for j in range(3)]
    p = os.path.join(tempfile.gettempdir(), "pure_ut.gds")
    n = gds.layout_to_gds(pl, p)                          # pure-Python writer (hierarchical)
    polys = gdstk.read_gds(p).top_level()[0].get_polygons()   # gdstk reads it back, references flattened
    a_pure = sum(abs(x.area()) for x in polys)
    # reference: gdstk WRITE+READ (same nm rounding) — fair comparison
    pr = os.path.join(tempfile.gettempdir(), "ref_ut.gds")
    lib = gdstk.Library(unit=1e-6, precision=1e-9); c = lib.new_cell("R")
    for q in pl:
        c.add(gdstk.Polygon(polygon(q["shape"], q["params"], center=(q["x"], q["y"])), layer=1))
    lib.write_gds(pr)
    a_ref = sum(abs(x.area()) for x in gdstk.read_gds(pr).cells[0].polygons)
    assert n == len(polys) == 9 and abs(a_pure - a_ref) < 1e-6
    os.remove(p); os.remove(pr)


def test_materials_table():
    from rcwa import materials
    import numpy as np
    nk = materials.n_table("Au", 0.633)
    assert np.real(nk) < 1.0 and np.imag(nk) > 2.0   # metal: small n, large k
    assert "TiO2_t" in materials.list_materials()


def test_shapes_fill():
    from rcwa import shapes
    import numpy as np
    sq = (np.real(shapes.rasterize("square", 0.2, 0.35, 128)) > 1.5).mean()
    ci = (np.real(shapes.rasterize("circle", 0.2, 0.35, 128)) > 1.5).mean()
    assert abs(ci / sq - np.pi / 4) < 0.05   # circle/square ~ π/4


def test_metrics_strehl():
    from rcwa import metrics
    import numpy as np
    N = 128; x = (np.arange(N) - N/2) * 0.05; X, Y = np.meshgrid(x, x, indexing="ij")
    I = np.exp(-(X**2 + Y**2) / (2*0.3**2))
    assert abs(metrics.strehl(I, I) - 1.0) < 1e-9         # identical -> Strehl=1
    assert metrics.focal_metrics(I, x)["fwhm_x"] > 0




def test_fullwave_bridge():
    from rcwa import fullwave
    pl = [{"x": (i-1)*0.35, "y": (j-1)*0.35, "params": 0.2, "angle": 0} for i in range(3) for j in range(3)]
    design = {"lam0": 0.633, "Lcell": 0.35, "height": 0.6, "n_pillar": 2.4,
              "shape": "circle", "D_lens": 2.0, "f": 4.0, "placements": pl}
    g = fullwave.meep_geometry(design)           # pure python, always works
    assert len(g) == 9 and g[0]["kind"] == "cylinder"
    try:
        import tidy3d  # noqa
        sim = fullwave.build_tidy3d(design)
        assert len(sim.structures[0].geometry.geometries) == 9 and len(sim.monitors) == 1
    except ImportError:
        pass  # tidy3d is optional


def test_adjoint_gradient():
    """E1: the adjoint (autodiff) gradient must match finite differences."""
    from rcwa import adjoint
    cfg = dict(nG=25, Lx=0.4, Ly=0.4, Nx=6, Ny=6, lam=0.633, thick=0.3, eps_hi=4.0 + 0.2j, eps_lo=1.0)
    ga, gfd, diff = adjoint.fd_grad_check(cfg)
    assert diff < 1e-4 and abs(ga) > 0


def test_adjoint_improves():
    """E1: Adam optimization must increase the absorption."""
    from rcwa import adjoint
    cfg = dict(nG=25, Lx=0.4, Ly=0.4, Nx=6, Ny=6, lam=0.633, thick=0.3, eps_hi=4.0 + 0.2j, eps_lo=1.0)
    r = adjoint.optimize_absorber(cfg, steps=20, lr=0.08)
    assert r["A_final"] >= r["A0"]


def test_hiacc2d_dielectric_matches():
    """E3: for dielectrics the grcwa high-accuracy backend matches native Laurent."""
    import numpy as np
    from rcwa import hiacc2d
    from rcwa import solve_rcwa_2d
    L = 0.35; H = 0.6; N = 48; x = (np.arange(N) + 0.5) / N * L
    X, Y = np.meshgrid(x, x, indexing="ij")
    eps = np.where((np.abs(X - L/2) < 0.09) & (np.abs(Y - L/2) < 0.09), 2.4**2, 1.0)
    rn = solve_rcwa_2d(0.633, 0, 0, 1, 1, L, L, [(eps, H)], 7, 7, pol=(0., 1.))
    rg = hiacc2d.solve_2d_grcwa(0.633, 0, 0, 1, 1, L, L, eps, H, nG=110)
    assert abs(rn["Ttot"] - rg["Ttot"]) < 0.02


def test_hiacc2d_metal_physical_and_advice():
    """E3: for metals grcwa is physical (R+T<=1) + the advice points to grcwa."""
    import numpy as np
    from rcwa import hiacc2d
    L = 0.3; x = (np.arange(48) + 0.5) / 48 * L; X, Y = np.meshgrid(x, x, indexing="ij")
    eps = np.where((np.abs(X - L/2) < 0.09) & (np.abs(Y - L/2) < 0.09), (-9.8 + 0.31j), 1.0)
    r = hiacc2d.solve_2d_grcwa(0.5, 0, 0, 1, 1, L, L, eps, 0.1, nG=120)
    ok, s = hiacc2d.energy_ok(r)
    assert ok and hiacc2d.factorization_advice(eps)["has_metal"]


def test_hex_lattice_energy():
    """E4: hexagonal (non-orthogonal) lattice, ΣDE≈1 for a lossless dielectric."""
    import numpy as np
    from rcwa import hiacc2d
    L1, L2 = hiacc2d.hex_lattice(0.4); N = 48
    x = (np.arange(N) + 0.5) / N; X, Y = np.meshgrid(x, x, indexing="ij")
    eps = np.where(((X - .5)**2 + (Y - .5)**2) < 0.18**2, 2.4**2, 1.0)
    r = hiacc2d.solve_2d_lattice(0.633, 0, 0, 1, 1, L1, L2, eps, 0.6, nG=110)
    assert abs(r["Rtot"] + r["Ttot"] - 1.0) < 0.02 and r["lattice"] == "non-orthogonal"


def test_materials_yaml_sellmeier():
    """E7: RI.info Sellmeier YAML import (SiO2 n@633≈1.457)."""
    import tempfile, os
    from rcwa import materials as Mt
    yml = ("DATA:\n  - type: formula 1\n    coefficients: 0 0.6961663 0.0684043 "
           "0.4079426 0.1162414 0.8974794 9.896161\n")
    p = os.path.join(tempfile.gettempdir(), "sio2_test.yml"); open(p, "w").write(yml)
    Mt.register_yaml("SiO2_ut", p)
    n = (Mt.eps_of("SiO2_ut", 0.633) ** 0.5).real
    assert abs(n - 1.457) < 0.01


def test_exporting_bsdf_drc():
    """E8: BSDF order table + DRC violation detection."""
    import numpy as np
    from rcwa import exporting as X
    from rcwa import solve_rcwa_2d
    L = 0.8; N = 48; x = (np.arange(N) + 0.5) / N * L; Xg, Yg = np.meshgrid(x, x, indexing="ij")
    eps = np.where(Xg < L/2, 4.0, 1.0)
    r = solve_rcwa_2d(0.633, 0, 0, 1, 1, L, L, [(eps, 0.3)], 4, 4, pol=(0., 1.))
    rows = X.bsdf_orders(r, 0.633, L, L)
    assert len(rows) == 81 and any(d["propagating"] for d in rows)
    pl = [{"x": 0.0, "y": 0.0, "params": 0.02, "angle": 0},
          {"x": 0.10, "y": 0.0, "params": 0.2, "angle": 0}]
    d = X.drc_check(pl, min_feature=0.05, min_gap=0.04)
    assert (not d["ok"]) and d["n_feature_viol"] >= 1


def test_pipeline_tiling_zemax():
    """E5/E6: tiling preserves the pillars; Zemax phase CSV has the correct NA."""
    import tempfile, os
    from rcwa import pipeline as P
    pl = [{"x": i * 0.35, "y": j * 0.35, "params": 0.2, "angle": 0} for i in range(6) for j in range(6)]
    design = {"lam0": 0.633, "Lcell": 0.35, "height": 0.6, "n_pillar": 2.4,
              "shape": "circle", "D_lens": 2.0, "f": 4.0, "placements": pl}
    tiles = P.tile_design(design, 1.0)
    assert sum(len(t["placements"]) for t in tiles) == len(pl)
    p = os.path.join(tempfile.gettempdir(), "zemax_ut.csv")
    info = P.zemax_phase_csv(design, p)
    assert info["samples"] == 256 and 0 < info["NA"] < 1 and os.path.exists(p)


def test_metrics_professional():
    """Professional metrics: efficiency definitions, encircled energy, sidelobe, DOF, 2D MTF, group delay."""
    from rcwa import metrics
    import numpy as np
    xc = (np.arange(256) - 128) * 0.03; X, Y = np.meshgrid(xc, xc, indexing="ij"); sig = 0.3
    I = np.exp(-(X**2 + Y**2) / (2 * sig**2)); fwhm = 2.3548 * sig
    es = metrics.efficiency_suite(I, xc, fwhm, P_incident=I.sum() * 1.25)
    assert abs(es["absolute"] - 0.8) < 0.02 and 0.99 < es["relative"] <= 1.001
    assert abs(metrics.encircled_energy(I, xc, 0.8) - 1.794 * sig) < 0.05
    assert metrics.sidelobe_level(I[:, 128])["ratio"] < 0.05          # pure Gaussian -> no sidelobe
    z = np.linspace(-5, 5, 201); Iz = np.exp(-z**2 / 2)
    assert abs(metrics.depth_of_focus(Iz, z) - 2.3548) < 0.1
    mt = metrics.mtf_2d(I); assert abs(mt[0] - 1.0) < 1e-9 and abs(metrics.strehl_mtf(mt, mt) - 1.0) < 1e-9
    lam = np.linspace(0.6, 0.66, 11); om = 2 * np.pi * 3e8 / (lam * 1e-6)
    gd, gdd = metrics.group_delay(1.5 * om * 1e-15, om); assert abs(gd - 1.5e-15) < 1e-17 and abs(gdd) < 1e-30


def test_largelens_vectorized():
    """E6: the large-lens LUT applies the same rule as design.phase_lut (complex-error match)."""
    from rcwa import largelens as LL
    st = LL.design_large_lens(0.633, 0.35, 0.60, "TiO2_t", "circle", 8.0, 10.0, 7, gds_dir=None)
    assert st["n_atoms"] > 0 and st["NA"] > 0 and st["preview_size_nm"].shape[0] > 0
    lut = LL.build_phase_lut(0.633, 0.35, 0.60, "TiO2_t", "circle", 7)
    t = lut["t"]; nb = lut["nbin"]
    for b in (0, nb // 3, nb - 1):
        phi = (b + 0.5) * 2 * np.pi / nb
        k = int(np.argmin(np.abs(t / np.abs(t).max() - np.exp(1j * phi))))
        assert abs(lut["lut_size"][b] - lut["sizes"][k]) < 1e-12


def test_design_core_lens():
    """design.design_lens + analyze_focus: full 2pi library, focus near f, Strehl <= 1,
    the on-axis fast path agrees with the full propagation, selection respects the mask."""
    from rcwa import design as dz
    d = dz.design_lens(0.633, 0.35, 0.60, 8.0, 10.0, material="TiO2_t")
    a = dz.analyze_focus(d)
    lib = d["lib"]; s = lib["sizes"]
    smooth = np.ones(len(s) - 1, bool)                      # phase winds through a resonance
    for r in lib.get("resonances", []):
        smooth &= ~((s[1:] >= r["lo"] - 0.002) & (s[:-1] <= r["hi"] + 0.002))
    assert lib["coverage"] >= 0.95 and lib["refined"] and np.all(np.diff(lib["phase"])[smooth] > -0.5)
    assert (d["idx"][~d["inside"]] == -1).all() and (d["idx"][d["inside"]] >= 0).all()
    assert len(d["placements"]) == int(d["inside"].sum())
    assert 0.7 * 10.0 < a["z_focus"] <= 10.2                 # focal shift toward the lens (low Fresnel number)
    assert 0.6 < a["strehl"] <= 1.005 and 0.5 < a["eff_absolute"] <= 1.0
    assert 0.6 < a["fwhm"] / a["diffraction_fwhm"] < 1.3
    ff = a["field"]; I = ff.intensity(a["z_focus"])
    assert abs(ff.on_axis(a["z_focus"]) / I.max() - 1) < 0.08


def test_select_atoms_prefers_bright():
    """Complex-error selection avoids a dark atom that has exactly the right phase."""
    from rcwa import design as dz
    t = np.array([1.0 * np.exp(1j * 0.0), 0.15 * np.exp(1j * 1.0), 1.0 * np.exp(1j * 1.35), 1.0 * np.exp(1j * 3.0)])
    idx_c, _ = dz.select_atoms(t, np.array([1.0]), mode="complex", n_offset=1)
    idx_p, _ = dz.select_atoms(t, np.array([1.0]), mode="phase", n_offset=1)
    assert idx_p[0] == 1 and idx_c[0] == 2


def test_cli_parameter_introspection():
    """Every experiment type exposes its parameters to the CLI help; typos are rejected."""
    from rcwa import cli, experiments as R
    for name in R.RUNNERS:
        assert len(cli.experiment_params(name)) >= 1, name
    keys = dict(cli.experiment_params("metalens_design_full"))
    assert {"lam0", "D_lens", "f", "material", "shape", "n_sub"} <= set(keys) and keys["D_lens"] == "16.0"
    assert "angle_max" in dict(cli.experiment_params("fov_analysis")) and "D_lens" in dict(cli.experiment_params("fov_analysis"))
    assert cli.main(["metalens_design_full", "D_len=5"]) == 2 and cli.main(["no_such_type"]) == 1


def test_backends_available():
    """E2: the backend picker runs without crashing and reports autograd."""
    from rcwa import backends
    av = backends.available()
    assert av["autograd"] is True
    gb, gpu = backends.pick_grcwa_backend()
    assert gb in ("jax", "autograd")


def test_phase_convention_slab():
    """Solver amplitudes are exp(+jwt): the transmission phase DECREASES with optical
    thickness; physical_phase flips it (grows with delay). Locks the convention."""
    from rcwa import solve_rcwa_2d, physical_phase
    lam = 0.633; out = {}
    for d in (0.02, 0.04):
        r1 = solve_rcwa_1d(lam, 0, 1, 1, 0.3, [(np.full(64, 2.25, complex), d)], 5, pol=(0., 1.))
        r2 = solve_rcwa_2d(lam, 0, 0, 1, 1, 0.3, 0.3, [(np.full((16, 16), 2.25), d)], 1, 1, pol=(0., 1.))
        out[d] = (r1["ty"][r1["zeroth_index"]], r2["ty"][r2["i0"]])
    for k in (0, 1):
        assert np.angle(out[0.04][k]) < np.angle(out[0.02][k]) < 0
        assert physical_phase(out[0.04][k]) > physical_phase(out[0.02][k]) > 0
    assert abs(out[0.04][0] - out[0.04][1]) < 1e-6      # 1D and 2D share one convention


def test_lens_1d_physical_focus():
    """Full super-cell solve of a 1D metalens: the PHYSICAL-convention (negated) library
    focuses near f; using the raw solver phase gives a diverging lens (no focus)."""
    from rcwa import solve_1d_auto
    from rcwa.convention import propagate_orders_1d
    lam, Lc, H, n = 0.633, 0.35, 0.6, 2.4
    Dl, f, guard = 3.5, 5.0, 3.0
    N = int(round(Dl / Lc)); Dl = N * Lc; D = Dl + 2 * guard; X0 = D / 2
    ws = np.linspace(0.04, Lc - 0.04, 60); ph = []
    for w in ws:
        x = (np.arange(256) + 0.5) / 256 * Lc
        e = np.where(np.abs(x - Lc / 2) < w / 2, n * n, 1.0).astype(complex)
        r = solve_rcwa_1d(lam, 0, 1, 1, Lc, [(e, H)], 21, pol=(0., 1.))
        ph.append(np.angle(r["ty"][r["zeroth_index"]]))
    ph = np.unwrap(np.array(ph)); ph -= ph.min()
    xc = (np.arange(N) + 0.5) * Lc - Dl / 2
    phi = np.mod(-(2 * np.pi / lam) * (np.sqrt(xc ** 2 + f ** 2) - f), 2 * np.pi)
    xg = (np.arange(4096) + 0.5) / 4096 * D; zc = np.linspace(2.5, 8.0, 56); peak = {}
    for sgn in (-1, +1):
        phm = np.mod(sgn * ph, 2 * np.pi)
        ch = np.array([ws[np.argmin(np.abs(np.angle(np.exp(1j * (phm - p)))))] for p in phi])
        col = np.ones(4096)
        for i in range(N):
            col[np.abs(xg - (X0 + xc[i])) < ch[i] / 2] = n * n
        res = solve_1d_auto(lam, 0, 1, 1, D, [(col.astype(complex), H)], 60, M_list=[60, 70])
        Iax = np.array([abs(propagate_orders_1d(res, lam, D, X0, z)[0]) ** 2 for z in zc])
        peak[sgn] = (Iax.max(), zc[int(np.argmax(Iax))])
    assert peak[-1][0] > 3.5 and abs(peak[-1][1] - f) < 1.0     # focuses near the design focal length
    assert peak[+1][0] < 0.6 * peak[-1][0]                      # raw solver phase: no comparable focus


def test_fwhm_interpolated():
    """fwhm_1d is sub-grid accurate and ignores sidelobes."""
    from rcwa import metrics
    x = np.linspace(-3, 3, 41); g = np.exp(-x ** 2 / (2 * 0.5 ** 2))       # coarse grid (dx=0.15)
    assert abs(metrics.fwhm_1d(g, x) - 2.3548 * 0.5) < 0.02
    g2 = g + 0.7 * np.exp(-(x - 2.4) ** 2 / (2 * 0.1 ** 2))                 # strong sidelobe
    assert abs(metrics.fwhm_1d(g2, x) - 2.3548 * 0.5) < 0.05


def test_hiacc2d_oblique_angle_units():
    """E3: solve_2d_grcwa takes DEGREES — an oblique uniform slab matches the native solver."""
    from rcwa import hiacc2d, solve_rcwa_2d
    eps = np.full((16, 16), 2.25); L = 0.3
    rn = solve_rcwa_2d(0.633, 30.0, 0, 1, 1, L, L, [(eps, 0.25)], 1, 1, pol=(0., 1.))     # TE
    rg = hiacc2d.solve_2d_grcwa(0.633, 30.0, 0, 1, 1, L, L, eps, 0.25, nG=9, pol=(0.0, 1.0))  # s
    assert abs(rn["Rtot"] - rg["Rtot"]) < 2e-3 and abs(rn["Ttot"] - rg["Ttot"]) < 2e-3


def test_result_keys_consistent():
    """1D and 2D results expose the same zeroth-order keys."""
    from rcwa import solve_rcwa_2d
    r1 = solve_rcwa_1d(0.55, 0, 1, 1, 0.3, [(np.full(32, 2.25, complex), 0.1)], 3)
    r2 = solve_rcwa_2d(0.55, 0, 0, 1, 1, 0.3, 0.3, [(np.full((8, 8), 2.25), 0.1)], 1, 1)
    for r in (r1, r2):
        assert r["i0"] == r["zeroth_index"] and all(k in r for k in ("tx", "ty", "rx", "ry"))


def test_convmat_vectorized_matches_definition():
    """convmat1d/2d equal the element-wise definition C[a,b] = eps_{a-b}; coarse grids are rejected."""
    from rcwa import convmat1d, convmat2d
    rng = np.random.default_rng(0)
    e1 = rng.uniform(1, 4, 32); P = 7; A = np.fft.fftshift(np.fft.fft(e1)) / 32
    C = convmat1d(e1, P)
    assert all(abs(C[r, c] - A[16 + r - c]) < 1e-14 for r in range(P) for c in range(P))
    e2 = rng.uniform(1, 4, (16, 12)); A2 = np.fft.fftshift(np.fft.fft2(e2)) / (16 * 12)
    C2 = convmat2d(e2, 3, 5)
    idx = [(m, n) for m in range(-1, 2) for n in range(-2, 3)]
    assert all(abs(C2[a, b] - A2[8 + idx[a][0] - idx[b][0], 6 + idx[a][1] - idx[b][1]]) < 1e-14
               for a in range(15) for b in range(15))
    for bad in (lambda: convmat1d(e1, 21), lambda: convmat2d(e2, 9, 9)):
        try:
            bad(); raise AssertionError("coarse grid must raise")
        except ValueError:
            pass


def test_vector_1d_tm_uses_inverse_rule():
    """The full-vector 1D solver applies Li's rule: TM at high contrast converges far faster
    than plain Laurent, and TE is untouched."""
    from rcwa import solve_scalar_1d
    x = (np.arange(2048) + 0.5) / 2048; eps = np.where(x < 0.5, 9.0, 1.0)
    ref = solve_scalar_1d(0.55, 0, 1, 1, 1.0, eps, 0.25, 100, "TM", "inverse")["Ttot"]
    li = solve_rcwa_1d(0.55, 0, 1, 1, 1.0, [(eps, 0.25)], 21, pol=(1., 0.))["Ttot"]
    la = solve_rcwa_1d(0.55, 0, 1, 1, 1.0, [(eps, 0.25)], 21, pol=(1., 0.), rule="laurent")["Ttot"]
    assert abs(li - ref) < 1e-3 and abs(li - ref) < 0.3 * abs(la - ref)
    te = [solve_rcwa_1d(0.55, 0, 1, 1, 1.0, [(eps, 0.25)], 21, pol=(0., 1.), rule=r)["Ttot"]
          for r in ("inverse", "laurent")]
    assert abs(te[0] - te[1]) < 1e-12


def test_2d_li_factorization_converges_faster():
    """2D: with Li's factorization a low order count is already close to the converged phase;
    plain Laurent at the same order count is several times further away. Energy is conserved."""
    from rcwa import solve_rcwa_2d, shapes
    e = shapes.rasterize("square", 0.20, 0.35, 256, n_pillar=2.52)
    def t0(M, rule):
        r = solve_rcwa_2d(0.633, 0, 0, 1, 1, 0.35, 0.35, [(e, 0.6)], M, M, pol=(0., 1.), rule=rule)
        assert abs(r["Rtot"] + r["Ttot"] - 1) < 1e-6
        return r["ty"][r["i0"]]
    ref = t0(7, "li"); li = t0(3, "li"); la = t0(3, "laurent")
    assert abs(np.angle(li / ref)) < 0.05 and abs(np.angle(la / ref)) > 2 * abs(np.angle(li / ref))


def test_rasterize_resolution_and_fill():
    """Fine rasters reproduce the analytic fill factor; aa returns area-weighted values."""
    from rcwa import shapes
    L, s = 0.35, 0.2
    fill = (np.real(shapes.rasterize("circle", s, L, shapes.DEFAULT_N)) > 1.5).mean()
    assert abs(fill - np.pi * (s / 2) ** 2 / L ** 2) < 2e-3
    aa = np.real(shapes.rasterize("circle", s, L, 32, n_pillar=2.0, aa=8))
    assert aa.shape == (32, 32) and abs((aa.mean() - 1.0) / 3.0 - np.pi * (s / 2) ** 2 / L ** 2) < 2e-3


def test_lossy_material_absorbs():
    """Im(eps) > 0 is ABSORPTION in every solver (never gain); a lossy slab matches the
    analytic absorbing-film result and the grcwa reference."""
    from rcwa import solve_rcwa_2d, solve_scalar_1d, hiacc2d
    lam, d, nk = 0.55, 0.3, 2.0 + 0.12j; eps = nk ** 2
    k0 = 2 * np.pi / lam; dl = k0 * nk * d                     # analytic film (exp(-iwt): n+ik decays)
    m11 = np.cos(dl); m12 = -1j * np.sin(dl) / nk; m21 = -1j * nk * np.sin(dl)
    t = 2 / (2 * m11 + m12 + m21); r = (m12 - m21) / (2 * m11 + m12 + m21)
    Ra, Ta = abs(r) ** 2, abs(t) ** 2
    assert Ra + Ta < 0.95
    r1 = solve_rcwa_1d(lam, 0, 1, 1, 0.3, [(np.full(64, eps, complex), d)], 5)
    r2 = solve_rcwa_2d(lam, 0, 0, 1, 1, 0.3, 0.3, [(np.full((16, 16), eps), d)], 1, 1)
    r3 = solve_scalar_1d(lam, 0, 1, 1, 0.3, np.full(64, eps, complex), d, 2, "TM")
    for rr in (r1, r2, r3):
        assert abs(rr["Rtot"] - Ra) < 1e-6 and abs(rr["Ttot"] - Ta) < 1e-6
    L = 0.35; x = (np.arange(128) + 0.5) / 128 * L; X, Y = np.meshgrid(x, x, indexing="ij")
    ep = np.where((np.abs(X - L / 2) < 0.1) & (np.abs(Y - L / 2) < 0.1), eps, 1.0)
    rn = solve_rcwa_2d(0.633, 0, 0, 1, 1, L, L, [(ep, 0.4)], 6, 6, pol=(0., 1.))
    rg = hiacc2d.solve_2d_grcwa(0.633, 0, 0, 1, 1, L, L, ep, 0.4, nG=150, pol=(0., 1.))
    assert rn["Rtot"] + rn["Ttot"] < 0.99
    assert abs(rn["Ttot"] - rg["Ttot"]) < 0.02 and abs(rn["Rtot"] - rg["Rtot"]) < 0.02


def test_rayleigh_anomaly_is_finite():
    """period = integer * wavelength puts an order exactly at grazing (kz = 0): the solvers
    must stay finite and conserve energy there."""
    from rcwa import solve_rcwa_2d, solve_scalar_1d
    x = (np.arange(1024) + 0.5) / 1024; eps = np.where(x < 0.5, 4.0, 1.0)
    r1 = solve_rcwa_1d(0.55, 0, 1, 1, 1.1, [(eps.astype(complex), 0.3)], 41)
    r3 = solve_scalar_1d(0.55, 0, 1, 1, 1.1, eps, 0.3, 20, "TE")
    e2 = np.where(np.add.outer(np.abs(x[::8] - 0.5), np.abs(x[::8] - 0.5) * 0) < 0.25, 4.0, 1.0)
    r2 = solve_rcwa_2d(0.55, 0, 0, 1, 1, 1.1, 1.1, [(e2, 0.3)], 3, 3)
    for r in (r1, r2, r3):
        assert np.isfinite(r["Rtot"]) and abs(r["Rtot"] + r["Ttot"] - 1) < 1e-3


def test_gds_hierarchical_equals_flat():
    """Hierarchical (cell-reference) and flat GDS describe the same geometry; the hierarchical
    file is several times smaller. Mixed shapes and rotations included."""
    import gdstk, tempfile, os
    from rcwa import gds
    sizes = [0.10, 0.15, 0.20]
    pl = [{"x": i * 0.35, "y": j * 0.35, "shape": "circle", "params": sizes[(i + j) % 3], "angle": 0.0}
          for i in range(20) for j in range(20)]
    pl += [{"x": -1.0, "y": -1.0, "shape": "rect", "params": (0.3, 0.1), "angle": 0.5},
           {"x": -2.0, "y": -1.0, "shape": "rect", "params": (0.3, 0.1), "angle": 0.5},
           {"x": -3.0, "y": -1.0, "shape": "cross", "params": (0.3, 0.1), "angle": 0.0}]
    tmp = tempfile.gettempdir(); ph = os.path.join(tmp, "h_ut.gds"); pf = os.path.join(tmp, "f_ut.gds")
    assert gds.layout_to_gds(pl, ph) == len(pl) and gds.layout_to_gds(pl, pf, hierarchical=False) == len(pl)
    sh, sf = gds.read_gds_summary(ph), gds.read_gds_summary(pf)
    assert sh["polygons"] == sf["polygons"] == len(pl)
    assert sh["references"] == len(pl) and sh["cells"] == 3 + 2 + 1 and sf["references"] == 0
    def geom(path):
        polys = gdstk.read_gds(path).top_level()[0].get_polygons()
        c = np.array([q.points.mean(axis=0) for q in polys]); a = np.array([abs(q.area()) for q in polys])
        o = np.lexsort((np.round(c[:, 1], 3), np.round(c[:, 0], 3)))
        return c[o], a[o]
    (ch_, ah), (cf, af) = geom(ph), geom(pf)
    assert np.abs(ch_ - cf).max() < 2e-3 and np.abs(ah - af).max() < 1e-4
    assert os.path.getsize(ph) * 5 < os.path.getsize(pf)
    os.remove(ph); os.remove(pf)


def test_drc_scales_and_detects():
    """k-d tree DRC: clean lattice passes; an oversized pillar and a tiny one are both found."""
    from rcwa import exporting as X
    pl = [{"x": i * 0.35, "y": j * 0.35, "params": 0.2, "angle": 0} for i in range(120) for j in range(120)]
    d = X.drc_check(pl, min_feature=0.05, min_gap=0.04)
    assert d["ok"] and d["n"] == 14400 and abs(d["worst_gap"] - 0.15) < 1e-9
    pl[777]["params"] = 0.48; pl[5000]["params"] = 0.03
    d = X.drc_check(pl, min_feature=0.05, min_gap=0.04)
    assert not d["ok"] and d["n_feature_viol"] == 1 and d["n_gap_viol"] == 5      # the pillar + its 4 neighbours


def test_materials_provenance_and_range():
    """Built-in tables are flagged approximate, refuse to extrapolate, and typos are explained."""
    from rcwa import materials as Mt
    i = Mt.info("TiO2_t")
    assert i["approximate"] and i["kind"] == "table" and i["range_um"] == (0.45, 0.70)
    assert not Mt.info("SiO2")["approximate"]
    for bad in (lambda: Mt.eps_of("TiO2_t", 1.55), lambda: Mt.eps_of("TiO2t", 0.55)):
        try:
            bad(); raise AssertionError("must raise")
        except ValueError as e:
            assert "TiO2" in str(e)


def test_substrate_energy_bookkeeping():
    """n_sub: with a glass substrate the library still covers 2pi and the absolute efficiency
    accounts for the index of the incidence medium (never exceeds 1)."""
    from rcwa import design as dz
    d = dz.design_lens(0.633, 0.35, 0.60, 6.0, 8.0, material="TiO2_t", n_sub=1.46, sizes=dz.default_sizes(0.35, 12))
    a = dz.analyze_focus(d)
    assert d["lib"]["coverage"] > 0.9 and 0.4 < a["eff_absolute"] < 1.0 and a["strehl"] <= 1.005



def _synthetic_resonant_library():
    """Dense synthetic library: full 2pi phase ramp with a narrow, deep |t| dip at 0.5 um."""
    s = np.linspace(0.0, 1.0, 1001)
    a = 1 - 0.99 * np.exp(-((s - 0.5) / 0.004) ** 2)
    return {"sizes": s, "t": a * np.exp(2j * np.pi * s), "refined": True}


def test_size_error_stats_analytic():
    """B3: for a linear t(s) the Gauss-Hermite statistics are exact: E[t] = t(s + bias),
    Var[t] = |dt/ds|^2 sigma^2."""
    from rcwa import design as dz
    s = np.linspace(0.1, 0.3, 41); c0, c1 = 1 + 0.5j, 2 - 1j
    lib = {"sizes": s, "t": c0 + c1 * s}
    mean, var = dz.size_error_stats(lib, sigma_nm=3.0, bias_nm=1.0)
    k = slice(5, -5)                                           # away from the clamped ends
    assert np.allclose(mean[k], c0 + c1 * (s[k] + 1e-3), atol=1e-12)
    assert np.allclose(var[k], abs(c1) ** 2 * (3e-3) ** 2, rtol=1e-9)
    m0, v0 = dz.size_error_stats(lib, sigma_nm=0.0)
    assert np.allclose(m0, lib["t"]) and np.allclose(v0, 0)


def test_resonance_guard_and_robust_selection():
    """B1-B3 on a synthetic library: the dip is detected, the guard band excludes it, and the
    robust rule keeps a larger distance from the resonance than the nominal complex rule."""
    from rcwa import design as dz
    lib = _synthetic_resonant_library()
    res = dz.find_resonances(lib["sizes"], lib["t"])
    assert len(res) == 1 and abs(res[0]["center"] - 0.5) < 1e-9 and res[0]["t_min"] < 0.02
    assert 0.004 < res[0]["width"] < 0.012
    lib["resonances"] = res
    ok = dz.resonance_mask(lib, guard_nm=3)
    assert not ok[np.abs(lib["sizes"] - 0.5) < 0.004].any() and ok[np.abs(lib["sizes"] - 0.5) > 0.02].all()
    lut_g, _ = dz.phase_lut(lib["t"], allowed=ok)
    assert ok[lut_g].all()
    phi = np.array([2 * np.pi * 0.508])                        # target just beside the dip
    ic, _ = dz.select_atoms(lib["t"], phi, mode="complex", n_offset=1)
    ir, _ = dz.select_atoms(lib["t"], phi, mode="robust", n_offset=1,
                            t_stats=dz.size_error_stats(lib, sigma_nm=4))
    dist = lambda i: abs(lib["sizes"][i[0]] - 0.5)              # noqa: E731
    assert dist(ir) > dist(ic) + 0.002
    d = dz.design_lens(0.633, 0.35, 0.60, 4.0, 6.0, lib=lib, mode="robust", sigma_nm=3, guard_nm=2)
    used = d["idx"][d["inside"]]
    assert d["allowed"][used].all() and d["mode"] == "robust"
    try:
        dz.design_lens(0.633, 0.35, 0.60, 4.0, 6.0, lib=lib, mode="robust", sigma_nm=None)
        raise AssertionError("robust mode without sigma_nm must fail")
    except ValueError:
        pass


def test_refine_library_finds_resonance():
    """B1: a 12.6 nm sweep of the default TiO2 library (633 nm, L=350, H=600) steps over the
    guided-mode resonance near 221 nm; adaptive refinement finds it (M=4 to keep it cheap)."""
    from rcwa import design as dz
    s = np.linspace(0.186, 0.249, 6)
    lib = dz.meta_atom_library(0.633, 0.35, 0.60, sizes=s, material="TiO2_t", M=4)
    assert dz.find_resonances(lib["sizes"], lib["t"]) == [] and lib["amp"].min() > 0.9
    ref = dz.refine_library(lib)
    assert ref["refined"] and ref["refine_converged"] and len(ref["resonances"]) == 1
    r = ref["resonances"][0]
    assert 0.219 < r["center"] < 0.223 and r["t_min"] < 0.3 and r["width"] < 0.004
    assert np.all(np.diff(ref["sizes"]) > 0) and set(np.round(s, 9)) <= set(np.round(ref["sizes"], 9))


def test_lens_tolerance_consistency():
    """B4: zero error reproduces analyze_focus exactly (interpolation is exact at library
    nodes). Physics check: on a unit-amplitude library whose phase is linear in size, a uniform
    bias is only a global phase, so the focus must not change; random error does change it and
    is reproducible with the same seed."""
    from rcwa import design as dz
    lib = _synthetic_resonant_library()
    lib["resonances"] = dz.find_resonances(lib["sizes"], lib["t"])
    d = dz.design_lens(0.633, 0.35, 0.60, 4.0, 6.0, lib=lib)
    E0, oor = dz.perturbed_field(d, lib, 0.0)
    assert np.allclose(E0, d["E0"]) and oor == 0.0
    a = dz.analyze_focus(d, nz=41)
    r = dz.lens_tolerance(d, lib, bias_nm=(0, 30), sigma_nm=0)
    assert r["n_trial"] == 1 and abs(r["eff"][0, 0] - a["eff_absolute"]) < 1e-12
    assert abs(r["strehl"][0, 0] - a["strehl"]) < 1e-12 and abs(r["eff"][1, 0] - r["eff"][0, 0]) < 1e-9
    r1 = dz.lens_tolerance(d, lib, bias_nm=(0,), sigma_nm=20, n_trial=3, seed=5)
    r2 = dz.lens_tolerance(d, lib, bias_nm=(0,), sigma_nm=20, n_trial=3, seed=5)
    assert np.array_equal(r1["eff"], r2["eff"]) and r1["eff_stats"]["std"][0] > 0



def test_parallel_library_matches_serial():
    """N3: the library computed by worker processes equals the serial one to machine precision
    (not bit for bit: the workers run single-threaded BLAS, which changes summation order)."""
    from rcwa import design as dz
    s = [0.10, 0.20, 0.28]
    kw = dict(sizes=s, material="TiO2_t", M=3, use_cache=False)
    a = dz.meta_atom_library(0.633, 0.35, 0.60, n_jobs=1, **kw)["t"]
    b = dz.meta_atom_library(0.633, 0.35, 0.60, n_jobs=2, **kw)["t"]
    assert np.allclose(a, b, rtol=0, atol=1e-9)
    assert dz.resolve_jobs(3) == 3 and dz.resolve_jobs(0) >= 1


def test_certify_library_convergence():
    """N1: an index-matched pillar (no contrast) is exactly converged at any M; a real TiO2
    pillar at M=2 is not, and the certificate says so; subsets and resonance bands work."""
    from rcwa import design as dz
    flat = dz.meta_atom_library(0.633, 0.35, 0.60, sizes=[0.1, 0.2], n_pillar=1.0, M=2, use_cache=False)
    c0 = dz.certify_library(flat, use_cache=False)["convergence"]
    assert c0["passed"] and c0["max_d_phase"] < 1e-9 and c0["M_check"] == 4
    lib = dz.meta_atom_library(0.633, 0.35, 0.60, sizes=[0.12, 0.20, 0.27], material="TiO2_t",
                               M=2, use_cache=False)
    c = dz.certify_library(lib, tol_phase=1e-4, use_cache=False)["convergence"]
    assert not c["passed"] and c["max_d_phase"] > 1e-4 and len(c["d_phase"]) == 3
    lib["resonances"] = [{"lo": 0.19, "hi": 0.21}]
    c2 = dz.certify_library(lib, sizes=[0.20], tol_phase=1.0, tol_amp=1.0, tol_res=1e-6,
                            use_cache=False)["convergence"]
    assert c2["in_resonance"].tolist() == [True] and not c2["passed"]
    try:
        dz.certify_library(lib, sizes=[0.15])
        raise AssertionError("a size outside the library must be rejected")
    except ValueError:
        pass


def test_symmetry_reduction_matches_full():
    """N2: the mirror-parity reduced solve gives the full solve's amplitudes in every order
    (x and y polarization, substrate, absorbing pillar, a stack with a uniform spacer)."""
    from rcwa import shapes
    from rcwa.solver2d import solve_rcwa_2d
    N = 128
    cases = [("circle", 0.20, 3, (0., 1.), 1.0, False), ("circle", 0.20, 3, (1., 0.), 1.46, False),
             ("square", 0.25, 4, (0., 1.), 1.46, False), ("circle", 0.22, 3, (0., 1.), 1.0, True)]
    for shape, size, M, pol, n_sub, stack in cases:
        eps = shapes.rasterize(shape, size, 0.35, N, n_pillar=complex(2.5, 0.02))
        lay = [(eps, 0.6)] if not stack else [(eps, 0.3), (np.full((N, N), 1.45 ** 2 + 0j), 0.1),
                                              (shapes.rasterize(shape, 0.8 * size, 0.35, N, n_pillar=2.5), 0.25)]
        a = solve_rcwa_2d(0.633, 0, 0, n_sub ** 2, 1.0, 0.35, 0.35, lay, M, M, pol=pol, symmetry=False)
        b = solve_rcwa_2d(0.633, 0, 0, n_sub ** 2, 1.0, 0.35, 0.35, lay, M, M, pol=pol)
        assert b["symmetry"] and not a["symmetry"]
        for k in ("tx", "ty", "rx", "ry"):
            assert np.abs(a[k] - b[k]).max() < 1e-10, (shape, k)
        assert abs(a["Ttot"] - b["Ttot"]) < 1e-12 and 0.5 < b["Rtot"] + b["Ttot"] <= 1 + 1e-9      # absorbing pillar


def test_symmetry_falls_back_when_not_applicable():
    """N2: oblique incidence, mixed polarization or an asymmetric cell use the full solve."""
    from rcwa import shapes
    from rcwa.solver2d import solve_rcwa_2d
    eps = shapes.rasterize("circle", 0.2, 0.35, 128, n_pillar=2.5)
    off = np.roll(eps, 5, axis=0)                                 # pillar off the cell centre
    kw = dict(Mx=3, My=3)
    assert not solve_rcwa_2d(0.633, 10, 0, 1, 1, 0.35, 0.35, [(eps, 0.6)], pol=(0., 1.), **kw)["symmetry"]
    assert not solve_rcwa_2d(0.633, 0, 0, 1, 1, 0.35, 0.35, [(eps, 0.6)], pol=(0.6, 0.8), **kw)["symmetry"]
    r = solve_rcwa_2d(0.633, 0, 0, 1, 1, 0.35, 0.35, [(off, 0.6)], pol=(0., 1.), **kw)
    f = solve_rcwa_2d(0.633, 0, 0, 1, 1, 0.35, 0.35, [(off, 0.6)], pol=(0., 1.), symmetry=False, **kw)
    assert not r["symmetry"] and np.array_equal(r["ty"], f["ty"])


def test_design_lens_uses_only_certified_sizes():
    """N1 in the design: with certify=True the atoms whose phase changes more than the
    tolerance between M and M+2 are excluded from the selection."""
    from rcwa import design as dz
    lib = dz.meta_atom_library(0.633, 0.35, 0.60, sizes=np.linspace(0.08, 0.29, 8),
                               material="TiO2_t", M=2, use_cache=False)
    cl = dz.certify_library(lib, tol_phase=0.03, tol_amp=0.05, use_cache=False)
    ok = cl["convergence"]["ok"]
    assert ok.dtype == bool and len(ok) == 8 and ok.any() and not ok.all()
    d = dz.design_lens(0.633, 0.35, 0.60, 4.0, 6.0, lib=cl, certify=True, refine=False, mode="complex")
    assert ok[d["idx"][d["inside"]]].all()

if __name__ == "__main__":
    from rcwa.cli import utf8_stdio
    utf8_stdio()
    _tests = [(k, v) for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for _name, _fn in _tests:
        _fn(); print(f"{_name}: PASSED")
    print(f"All {len(_tests)} tests passed ✅")
