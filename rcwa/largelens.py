"""E6 — cm-ÖLÇEK KUTUCUKLAMALI ÜRETİM HATTI (büyük metalens).

Sorun: milimetre/santimetre açıklıklı bir metalens milyonlarca–milyarlarca
meta-atom içerir. Python döngüsüyle yerleştirmek ve hepsini bellekte tutup tek
GDS yazmak imkânsız. Bu modül iki darboğazı çözer:

1. **Vektörize boyut seçimi** — faz→boyut arama tablosu (LUT) ile tüm hücreler
   tek numpy indekslemesinde seçilir (döngü YOK). metalens_design_full ile aynı
   (konvansiyon-düzeltmeli) seçim, ama O(N) döngü yerine O(1) LUT.
2. **Kutucuk-kutucuk GDS** — açıklık karelere bölünür; her kutucuğun poligonları
   ayrı bir GDS dosyasına yazılıp bellek serbest bırakılır → bellek kutucuk
   boyutuyla sınırlı kalır (cm-ölçek mümkün).

Ayrıca hiçbir atomu maddileştirmeden ölçek istatistiği + önizleme üretir.
"""
import os
import numpy as np
from . import materials, shapes, cache
from .solver2d import solve_rcwa_2d


def build_phase_lut(lam0, L, H, mat, shape, M, n_pillar, n_size=20, nbin=2048):
    """Meta-atom kütüphanesi (önbellekli) + faz→boyut LUT. Konvansiyon düzeltmeli."""
    sizes = np.linspace(0.06, L - 0.05, n_size)
    def _lib():
        lp = []; la = []
        for s in sizes:
            eps = shapes.rasterize(shape, s, L, 64, n_pillar=n_pillar)
            r = solve_rcwa_2d(lam0, 0, 0, 1, 1, L, L, [(eps, H)], M, M, pol=(0., 1.))
            t0 = r["ty"][r["i0"]]; lp.append(np.angle(t0)); la.append(abs(t0))
        lp = np.unwrap(np.array(lp)); lp -= lp.min()
        return {"phm": np.mod(lp, 2 * np.pi), "am": np.array(la)}
    c = cache.cached_library("libdesign", {"lam0": lam0, "L": L, "H": H, "mat": mat,
                             "shape": shape, "M": M, "n": round(n_pillar, 3)}, _lib)
    libphm = np.mod(-c["phm"], 2 * np.pi)                       # fiziksel/FDTD konvansiyonu
    phases = np.linspace(0, 2 * np.pi, nbin, endpoint=False)
    # her faz-kutusu için en yakın atom (dairesel mesafe) — vektörize
    d = np.abs(np.angle(np.exp(1j * (libphm[None, :] - phases[:, None]))))
    lut_k = np.argmin(d, axis=1)                                # nbin -> boyut indeksi
    return {"sizes": sizes, "libphm": libphm, "libam": c["am"],
            "lut_size": sizes[lut_k], "lut_k": lut_k, "nbin": nbin}


def size_map_vectorized(lut, lam0, L, f, ci):
    """Verilen hücre-merkezleri (ci) için boyut haritasını DÖNGÜSÜZ üret.
    Döndürür: size_map(Nc,Nc), inside(Nc,Nc). Bellek: Nc² (önizleme için sınırla)."""
    XX, YY = np.meshgrid(ci, ci, indexing="ij"); R = np.hypot(XX, YY)
    inside = R <= (ci[-1] - ci[0]) / 2 + 1e-9
    phi = np.mod(-(2 * np.pi / lam0) * (np.sqrt(R ** 2 + f ** 2) - f), 2 * np.pi)
    b = (phi / (2 * np.pi) * lut["nbin"]).astype(int) % lut["nbin"]
    size_map = lut["lut_size"][b]
    return np.where(inside, size_map, 0.0), inside


def design_large_lens(lam0, L, H, mat, shape, D_lens, f, M, gds_dir=None,
                      tile_cells=400, preview_cap=1400, write_gds=True):
    """Büyük lens: vektörize seçim + kutucuklu GDS. Bellek kutucukla sınırlı.
    gds_dir verilirse her kutucuk ayrı .gds; None ise yalnız istatistik+önizleme.
    Döndürür: istatistik sözlüğü + preview (size_map_ds, ci_ds)."""
    from . import gds
    n_pillar = float(np.real(np.sqrt(materials.eps_of(mat, lam0)))) if mat else 2.4
    lut = build_phase_lut(lam0, L, H, mat, shape, M, n_pillar)
    Nc = int(round(D_lens / L)); D_lens = Nc * L
    ci = (np.arange(Nc) - (Nc - 1) / 2) * L
    NA = float(np.sin(np.arctan(D_lens / 2 / f)))
    # atom sayısı (döngüsüz): daire içindeki hücreler
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
    # önizleme (altörnekleme — hiçbir atomu tam maddileştirmeden)
    stride = max(1, Nc // preview_cap)
    ci_ds = ci[::stride]
    size_ds, _ = size_map_vectorized(lut, lam0, L, f, ci_ds)
    return {"D_lens": D_lens, "f": f, "NA": NA, "Nc": Nc, "n_atoms": n_atoms,
            "n_tiles": ntile, "npoly": npoly_total, "n_pillar": n_pillar,
            "preview_size_nm": size_ds * 1000, "ci_ds": ci_ds}


def scale_projection(n_atoms, npoly_bytes=415):
    """GDS dosya boyutu projeksiyonu. ~415 byte/poligon (dairesel sütun ~48 köşe,
    ölçüldü: D=150µm → 144k atom / 60MB). Kare şekiller için daha düşük."""
    return n_atoms * npoly_bytes / 1e9   # GB
