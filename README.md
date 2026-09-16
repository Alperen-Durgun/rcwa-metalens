# rcwa_py — RCWA & Metalens Tasarım Kütüphanesi

> A from-scratch, tested **RCWA (Rigorous Coupled-Wave Analysis)** solver in Python
> for **metalens design and simulation** — 1D & 2D solvers, meta-atom libraries,
> GDS mask export, professional optical metrics, and FDTD cross-validation
> (Meep / Tidy3D). Educational yet FDTD-validated.

Sıfırdan yazılmış, **test edilmiş** bir RCWA çözücü ve metalens tasarım hattı.
1B/2B çözücüler, meta-atom faz kütüphaneleri, üretilebilir **GDS-II** maske çıktısı,
profesyonel optik metrikler ve **FDTD ile çapraz-doğrulama** (Meep + Tidy3D) içerir.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

---

## ✨ Öne çıkanlar

- **Çekirdek RCWA** — 1B (TE/TM, Li ters-kural, simetri indirgeme) ve tam-vektörel 2B çözücü; enerji-korumalı otomatik mertebe seçimi.
- **Kararlılık** — Redheffer star-product + S-matris; enerji korunumu birim testlerle doğrulandı.
- **Malzemeler** — dispersif n(λ)+kayıp (SiO₂, TiO₂, Si, Ag, SiN); ölçülü n,k tabloları + `refractiveindex.info` YAML içe aktarma.
- **Metalens hattı** — meta-atom kütüphanesi → faz profili → yerleşim → odak; yerel-periyodik yaklaşım.
- **Üretim** — saf-Python, Unicode-yol-güvenli **GDS-II** yazıcı; keyfi geometri (daire/kare/altıgen/haç/elips); cm-ölçek kutucuklamalı büyük lens.
- **Profesyonel metrikler** — FWHM, odaklama verimi (mutlak/bağıl), Strehl, MTF, encircled energy, yan-lob, DOF, kromatik kayma, grup gecikmesi.
- **FDTD köprüleri** — yerel **Meep** (çevrimdışı) ve bulut **Tidy3D** ile bağımsız doğrulama.
- **Otomasyon** — `run_experiment.py` ile **23 deney tipi**; `verify_all.py` ile tek komutta uçtan uca doğrulama.

## 📦 Kurulum

```bash
cd rcwa_py
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Zorunlu: `numpy scipy matplotlib grcwa autograd gdstk pyyaml`.
Opsiyonel: `tidy3d` (bulut FDTD), `meep` (yerel FDTD, conda ile), `jax`/`cupy` (GPU).

## 🚀 Hızlı başlangıç

```python
import numpy as np
from rcwa import solve_rcwa_1d

x = (np.arange(512) + 0.5) / 512
eps = np.where(x < 0.5, 4.0, 1.0)          # ikili grating: n=2 / hava
res = solve_rcwa_1d(lam0=0.55, theta_deg=15, er_ref=1.0, er_trn=2.25,
                    period=1.2, layers=[(eps, 0.30)], P=41, pol=(0, 1))  # TE
print(res['Rtot'], res['Ttot'], res['Rtot'] + res['Ttot'])  # ~ enerji korunur
```

Tam metalens tasarımı (tasarım + metrik + GDS):

```bash
python metadesign.py metalens_design_full D_lens=16 f=20 material=TiO2_t lam0=0.633
```

## ✅ Doğrulama

```bash
python verify_all.py            # birim testler + metrikler + 23 üreteç + konvansiyon denetimi
python verify_all.py --report   # + Markdown rapor
python tests/test_rcwa.py       # yalnız birim testler (30 test)
```

Çözücü, birim-hücre düzeyinde Tidy3D FDTD ile **%0.04** farkla; tam-lens düzeyinde
Meep/Tidy3D 3B FDTD ile **difraksiyon-sınırlı odak** üzerinden doğrulanmıştır.

## 🗂️ Yapı

```
rcwa_py/
├── rcwa/                    # çekirdek paket
│   ├── solver.py           # 1B RCWA çözücü
│   ├── solver1d_scalar.py  # skaler 1B (Li ters-kural + simetri)
│   ├── solver2d.py         # tam-vektörel 2B çözücü
│   ├── convmat.py          # konvolüsyon matrisi (1B+2B)
│   ├── layer.py            # özmodlar + katman S-matrisi
│   ├── redheffer.py        # star-product
│   ├── stable.py           # enerji-korumalı otomatik-M seçimi
│   ├── materials.py        # dispersif n(λ), ölçülü n,k
│   ├── metrics.py          # FWHM / verim / Strehl / MTF / DOF / grup gecikmesi
│   ├── shapes.py           # keyfi geometri + GDS poligon
│   ├── gds.py / gdswriter.py  # saf-Python, Unicode-güvenli GDS-II yazıcı
│   ├── largelens.py        # cm-ölçek kutucuklamalı büyük lens
│   ├── fullwave.py         # Meep + Tidy3D FDTD köprüsü
│   ├── adjoint.py          # adjoint / ters tasarım
│   ├── hiacc2d.py          # metal-2B + altıgen kafes
│   ├── exporting.py        # BSDF / uzak-alan / DRC
│   ├── pipeline.py         # Zemax faz-CSV + kutucuklama
│   ├── backends.py         # GPU arka uçları (jax/cupy)
│   ├── cache.py            # kütüphane önbelleği
│   └── materials_data/     # ölçülü n,k tabloları
├── examples/               # 24 örnek (RCWA + Meep + Tidy3D)
├── tests/test_rcwa.py      # 30 birim test
├── run_experiment.py       # 23 deney tipi otomasyonu
├── metadesign.py           # komut satırı arayüzü
├── verify_all.py           # uçtan uca doğrulama
├── check_env.py            # ortam kontrolü
└── requirements.txt
```

## 🔬 Deney tipleri (özet)

`grating`, `metalens_library`, `angle_sweep`, `grating2d`, `metalens_pillar_2d`,
`metalens_layout_2d`, `param_map_2d`, `lens_fullwave_1d`, `wavelength_sweep`,
`birefringent_library`, `jones_matrix`, `pb_phase_library`, `inverse_design_deflector`,
`achromatic_analysis`, `achromatic_design`, `vortex_beam`, `hologram`,
`polarization_multiplexed`, `metalens_design_full`, `tolerance_analysis`,
`fov_analysis`, `index_robustness`, `metalens_large_tiled`.

Her tipin parametreleri için `metadesign.py <tip>` çalıştırın.

## 🔑 Tidy3D (opsiyonel)

API anahtarını **koda yazmayın**. Bir kez yapılandırın:

```bash
tidy3d configure      # anahtar ~/.tidy3d/config'e yazılır (repo dışı, .gitignore'lu)
```

## 📄 Lisans

[MIT](LICENSE) © 2026 Alperen Durgun
