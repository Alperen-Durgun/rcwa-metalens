"""Otomatik testler — Kod Dersi 05.
Çalıştır:  python tests/test_rcwa.py   (pytest de çalışır)
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
    assert np.imag(np.sqrt(materials.eps_of("Si", 0.633))) > 0  # kayıp


def test_cache_computes_once():
    from rcwa import cache
    import numpy as np
    calls = {"n": 0}
    def comp():
        calls["n"] += 1; return {"x": np.arange(4.0)}
    k = {"unittest": 1, "v": np.random.randint(1_000_000_000)}
    cache.cached_library("unittest", k, comp)
    cache.cached_library("unittest", k, comp)
    assert calls["n"] == 1


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
    # enerji korunumu
    assert abs(hi["Rtot"] + hi["Ttot"] - 1) < 1e-6
    # inverse rule düşük M'de referansa daha yakın
    assert abs(hi["Ttot"] - ref) < abs(lo["Ttot"] - ref)


def test_symmetry_reduction():
    from rcwa import solve_scalar_1d
    import numpy as np
    x = (np.arange(2048) + 0.5) / 2048
    eps = np.where(np.minimum(x, 1 - x) < 0.25, 9.0, 1.0)  # x=0'da simetrik
    for pol in ("TE", "TM"):
        full = solve_scalar_1d(0.55, 0, 1, 1, 1.0, eps, 0.25, 40, pol, "inverse")
        sym = solve_scalar_1d(0.55, 0, 1, 1, 1.0, eps, 0.25, 40, pol, "inverse", symmetric=True)
        assert abs(sym["Rtot"] + sym["Ttot"] - 1) < 1e-6       # enerji
        assert abs(full["Ttot"] - sym["Ttot"]) < 1e-4          # birebir aynı




def test_convmat_correctness():
    from rcwa import convmat1d
    import numpy as np
    x = (np.arange(1024) + 0.5) / 1024
    duty, ehi, elo = 0.5, 4.0, 1.0
    eps = np.where(x < duty, ehi, elo)
    C = convmat1d(eps, 11)
    # 0. Fourier katsayisi = ortalama eps
    assert abs(C[5, 5] - (duty * ehi + (1 - duty) * elo)) < 1e-3
    # Toeplitz: C[i,j] sadece i-j'ye bagli
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
    eps = np.where((np.abs(X - L/2) < 0.09) & (np.abs(Y - L/2) < 0.09), 2.4**2, 1.0)  # kare (izotropik)
    rx = solve_rcwa_2d(0.633, 0, 0, 1, 1, L, L, [(eps, 0.6)], 6, 6, pol=(1., 0.))
    ry = solve_rcwa_2d(0.633, 0, 0, 1, 1, L, L, [(eps, 0.6)], 6, 6, pol=(0., 1.))
    txy = abs(ry["tx"][ry["i0"]]); tyx = abs(rx["ty"][rx["i0"]])
    assert txy < 1e-3 and tyx < 1e-3   # capraz kutuplanma ~0


def test_regression_binary_grating():
    from rcwa import solve_rcwa_1d
    import numpy as np
    x = (np.arange(1024) + 0.5) / 1024
    eps = np.where(x < 0.5, 4.0, 1.0).astype(complex)
    r = solve_rcwa_1d(0.55, 15.0, 1.0, 2.25, 1.2, [(eps, 0.3)], 81, pol=(0., 1.))
    # kayitli referans deger (regresyon): 0.9180 (grcwa ile dogrulanmis)
    assert abs(r["Ttot"] - 0.9180) < 2e-3




def test_gds_roundtrip():
    from rcwa import gds
    import tempfile, os
    pl = [{"x": i*0.35, "y": j*0.35, "shape": "circle", "params": 0.2, "angle": 0} for i in range(3) for j in range(3)]
    fp = os.path.join(tempfile.gettempdir(), "rcwa_test.gds")
    n = gds.layout_to_gds(pl, fp)
    assert n == 9 and gds.read_gds_summary(fp)["polygons"] == 9


def test_gds_unicode_path():
    """Regresyon: gdstk Windows'ta Türkçe karakterli yola yazamaz; gds.py fallback
    ile geçici dosyaya yazıp taşımalı. Yolda 'ş,ğ,ı,İ,ö,ç,ü' kullan."""
    from rcwa import gds
    import tempfile, os, shutil
    base = os.path.join(tempfile.gettempdir(), "rcwa_Kişisel_çığ_İş_öçü")
    os.makedirs(base, exist_ok=True)
    fp = os.path.join(base, "maske_şablon.gds")
    try:
        pl = [{"x": i*0.35, "y": j*0.35, "shape": "circle", "params": 0.2, "angle": 0} for i in range(2) for j in range(2)]
        n = gds.layout_to_gds(pl, fp)
        assert n == 4 and os.path.exists(fp)
        assert gds.read_gds_summary(fp)["polygons"] == 4
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_gdswriter_geometry():
    """Saf-Python GDSII yazıcı: gdstk ile geri okunur + gdstk yazımıyla alan birebir."""
    import gdstk, tempfile, os
    from rcwa import gds
    from rcwa.shapes import polygon
    pl = [{"x": i * 0.35, "y": j * 0.35, "shape": "circle", "params": 0.2, "angle": 0} for i in range(3) for j in range(3)]
    p = os.path.join(tempfile.gettempdir(), "pure_ut.gds")
    n = gds.layout_to_gds(pl, p)                          # saf-Python yazıcı
    polys = gdstk.read_gds(p).cells[0].polygons           # gdstk geri okur
    a_pure = sum(abs(x.area()) for x in polys)
    # referans: gdstk YAZ+OKU (aynı nm-yuvarlama) — adil kıyas
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
    assert np.real(nk) < 1.0 and np.imag(nk) > 2.0   # metal: küçük n, büyük k
    assert "TiO2_t" in materials.list_materials()


def test_shapes_fill():
    from rcwa import shapes
    import numpy as np
    sq = (np.real(shapes.rasterize("square", 0.2, 0.35, 128)) > 1.5).mean()
    ci = (np.real(shapes.rasterize("circle", 0.2, 0.35, 128)) > 1.5).mean()
    assert abs(ci / sq - np.pi / 4) < 0.05   # daire/kare ~ π/4


def test_metrics_strehl():
    from rcwa import metrics
    import numpy as np
    N = 128; x = (np.arange(N) - N/2) * 0.05; X, Y = np.meshgrid(x, x, indexing="ij")
    I = np.exp(-(X**2 + Y**2) / (2*0.3**2))
    assert abs(metrics.strehl(I, I) - 1.0) < 1e-9         # aynı -> Strehl=1
    assert metrics.focal_metrics(I, x)["fwhm_x"] > 0




def test_fullwave_bridge():
    from rcwa import fullwave
    pl = [{"x": (i-1)*0.35, "y": (j-1)*0.35, "params": 0.2, "angle": 0} for i in range(3) for j in range(3)]
    design = {"lam0": 0.633, "Lcell": 0.35, "height": 0.6, "n_pillar": 2.4,
              "shape": "circle", "D_lens": 2.0, "f": 4.0, "placements": pl}
    g = fullwave.meep_geometry(design)           # saf python, her zaman çalışır
    assert len(g) == 9 and g[0]["kind"] == "cylinder"
    try:
        import tidy3d  # noqa
        sim = fullwave.build_tidy3d(design)
        assert len(sim.structures[0].geometry.geometries) == 9 and len(sim.monitors) == 1
    except ImportError:
        pass  # tidy3d opsiyonel


def test_adjoint_gradient():
    """E1: adjoint (autodiff) gradyanı sonlu-farkla örtüşmeli."""
    from rcwa import adjoint
    cfg = dict(nG=25, Lx=0.4, Ly=0.4, Nx=6, Ny=6, lam=0.633, thick=0.3, eps_hi=4.0 + 0.2j, eps_lo=1.0)
    ga, gfd, diff = adjoint.fd_grad_check(cfg)
    assert diff < 1e-4 and abs(ga) > 0


def test_adjoint_improves():
    """E1: Adam optimizasyonu absorpsiyonu artırmalı."""
    from rcwa import adjoint
    cfg = dict(nG=25, Lx=0.4, Ly=0.4, Nx=6, Ny=6, lam=0.633, thick=0.3, eps_hi=4.0 + 0.2j, eps_lo=1.0)
    r = adjoint.optimize_absorber(cfg, steps=20, lr=0.08)
    assert r["A_final"] >= r["A0"]


def test_hiacc2d_dielectric_matches():
    """E3: dielektrikte grcwa yüksek-doğruluk backend'i native Laurent ile örtüşür."""
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
    """E3: metalde grcwa fiziksel (R+T<=1) + öneri grcwa'yı işaret eder."""
    import numpy as np
    from rcwa import hiacc2d
    L = 0.3; x = (np.arange(48) + 0.5) / 48 * L; X, Y = np.meshgrid(x, x, indexing="ij")
    eps = np.where((np.abs(X - L/2) < 0.09) & (np.abs(Y - L/2) < 0.09), (-9.8 + 0.31j), 1.0)
    r = hiacc2d.solve_2d_grcwa(0.5, 0, 0, 1, 1, L, L, eps, 0.1, nG=120)
    ok, s = hiacc2d.energy_ok(r)
    assert ok and hiacc2d.factorization_advice(eps)["has_metal"]


def test_hex_lattice_energy():
    """E4: altıgen (non-ortogonal) kafes, kayıpsız dielektrikte ΣDE≈1."""
    import numpy as np
    from rcwa import hiacc2d
    L1, L2 = hiacc2d.hex_lattice(0.4); N = 48
    x = (np.arange(N) + 0.5) / N; X, Y = np.meshgrid(x, x, indexing="ij")
    eps = np.where(((X - .5)**2 + (Y - .5)**2) < 0.18**2, 2.4**2, 1.0)
    r = hiacc2d.solve_2d_lattice(0.633, 0, 0, 1, 1, L1, L2, eps, 0.6, nG=110)
    assert abs(r["Rtot"] + r["Ttot"] - 1.0) < 0.02 and r["lattice"] == "non-ortogonal"


def test_materials_yaml_sellmeier():
    """E7: RI.info Sellmeier YAML içe aktarımı (SiO2 n@633≈1.457)."""
    import tempfile, os
    from rcwa import materials as Mt
    yml = ("DATA:\n  - type: formula 1\n    coefficients: 0 0.6961663 0.0684043 "
           "0.4079426 0.1162414 0.8974794 9.896161\n")
    p = os.path.join(tempfile.gettempdir(), "sio2_test.yml"); open(p, "w").write(yml)
    Mt.register_yaml("SiO2_ut", p)
    n = (Mt.eps_of("SiO2_ut", 0.633) ** 0.5).real
    assert abs(n - 1.457) < 0.01


def test_exporting_bsdf_drc():
    """E8: BSDF mertebe tablosu + DRC ihlal yakalama."""
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
    """E5/E6: kutucuklama sütun korur; Zemax faz CSV NA doğru."""
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
    """Profesyonel metrikler: verim tanımları, kuşatılmış enerji, yan-lob, DOF, 2B MTF, grup gecikmesi."""
    from rcwa import metrics
    import numpy as np
    xc = (np.arange(256) - 128) * 0.03; X, Y = np.meshgrid(xc, xc, indexing="ij"); sig = 0.3
    I = np.exp(-(X**2 + Y**2) / (2 * sig**2)); fwhm = 2.3548 * sig
    es = metrics.efficiency_suite(I, xc, fwhm, P_incident=I.sum() * 1.25)
    assert abs(es["absolute"] - 0.8) < 0.02 and 0.99 < es["relative"] <= 1.001
    assert abs(metrics.encircled_energy(I, xc, 0.8) - 1.794 * sig) < 0.05
    assert metrics.sidelobe_level(I[:, 128])["ratio"] < 0.05          # saf Gauss -> yan-lob yok
    z = np.linspace(-5, 5, 201); Iz = np.exp(-z**2 / 2)
    assert abs(metrics.depth_of_focus(Iz, z) - 2.3548) < 0.1
    mt = metrics.mtf_2d(I); assert abs(mt[0] - 1.0) < 1e-9 and abs(metrics.strehl_mtf(mt, mt) - 1.0) < 1e-9
    lam = np.linspace(0.6, 0.66, 11); om = 2 * np.pi * 3e8 / (lam * 1e-6)
    gd, gdd = metrics.group_delay(1.5 * om * 1e-15, om); assert abs(gd - 1.5e-15) < 1e-17 and abs(gdd) < 1e-30


def test_largelens_vectorized():
    """E6: büyük-lens vektörize seçim + LUT, argmin seçimiyle birebir örtüşür."""
    from rcwa import largelens as LL
    import numpy as np
    st = LL.design_large_lens(0.633, 0.35, 0.60, "TiO2_t", "circle", 8.0, 10.0, 7, gds_dir=None)
    assert st["n_atoms"] > 0 and st["NA"] > 0 and st["preview_size_nm"].shape[0] > 0
    lut = LL.build_phase_lut(0.633, 0.35, 0.60, "TiO2_t", "circle", 7, st["n_pillar"])
    r = 1.0; phi = np.mod(-(2 * np.pi / 0.633) * (np.sqrt(r ** 2 + 10 ** 2) - 10), 2 * np.pi)
    b = int(phi / (2 * np.pi) * lut["nbin"]) % lut["nbin"]
    k = int(np.argmin(np.abs(np.angle(np.exp(1j * (lut["libphm"] - phi))))))
    assert abs(lut["lut_size"][b] - lut["sizes"][k]) < 1e-9


def test_backends_available():
    """E2: backend seçici çökme olmadan çalışır ve autograd'ı bildirir."""
    from rcwa import backends
    av = backends.available()
    assert av["autograd"] is True
    gb, gpu = backends.pick_grcwa_backend()
    assert gb in ("jax", "autograd")


if __name__ == "__main__":
    test_fresnel_match(); print("test_fresnel_match: GEÇTİ")
    test_energy_conservation(); print("test_energy_conservation: GEÇTİ")
    test_energy_conservation_2d(); print("test_energy_conservation_2d: GEÇTİ")
    test_materials(); print("test_materials: GEÇTİ")
    test_cache_computes_once(); print("test_cache_computes_once: GEÇTİ")
    test_solve_1d_auto_stable(); print("test_solve_1d_auto_stable: GEÇTİ")
    test_tm_inverse_rule(); print("test_tm_inverse_rule: GEÇTİ")
    test_symmetry_reduction(); print("test_symmetry_reduction: GEÇTİ")
    test_convmat_correctness(); print("test_convmat_correctness: GEÇTİ")
    test_convergence_energy_stable(); print("test_convergence_energy_stable: GEÇTİ")
    test_jones_isotropic_diagonal(); print("test_jones_isotropic_diagonal: GEÇTİ")
    test_regression_binary_grating(); print("test_regression_binary_grating: GEÇTİ")
    test_gds_roundtrip(); print("test_gds_roundtrip: GEÇTİ")
    test_gds_unicode_path(); print("test_gds_unicode_path: GEÇTİ")
    test_gdswriter_geometry(); print("test_gdswriter_geometry: GEÇTİ")
    test_materials_table(); print("test_materials_table: GEÇTİ")
    test_shapes_fill(); print("test_shapes_fill: GEÇTİ")
    test_metrics_strehl(); print("test_metrics_strehl: GEÇTİ")
    test_fullwave_bridge(); print("test_fullwave_bridge: GEÇTİ")
    test_adjoint_gradient(); print("test_adjoint_gradient: GEÇTİ")
    test_adjoint_improves(); print("test_adjoint_improves: GEÇTİ")
    test_hiacc2d_dielectric_matches(); print("test_hiacc2d_dielectric_matches: GEÇTİ")
    test_hiacc2d_metal_physical_and_advice(); print("test_hiacc2d_metal_physical_and_advice: GEÇTİ")
    test_hex_lattice_energy(); print("test_hex_lattice_energy: GEÇTİ")
    test_materials_yaml_sellmeier(); print("test_materials_yaml_sellmeier: GEÇTİ")
    test_exporting_bsdf_drc(); print("test_exporting_bsdf_drc: GEÇTİ")
    test_pipeline_tiling_zemax(); print("test_pipeline_tiling_zemax: GEÇTİ")
    test_metrics_professional(); print("test_metrics_professional: GEÇTİ")
    test_largelens_vectorized(); print("test_largelens_vectorized: GEÇTİ")
    test_backends_available(); print("test_backends_available: GEÇTİ")
    print("Tüm testler geçti ✅")

