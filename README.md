# rcwa_py — RCWA & Metalens Design Library

> A from-scratch, tested **RCWA (Rigorous Coupled-Wave Analysis)** solver in Python
> for **metalens design and simulation** — 1D & 2D solvers, meta-atom libraries,
> GDS mask export, professional optical metrics, and FDTD cross-validation
> (Meep / Tidy3D). Educational yet FDTD-validated.

A from-scratch, **tested** RCWA solver and metalens design pipeline. It includes
1D/2D solvers, meta-atom phase libraries, manufacturable **GDS-II** mask output,
professional optical metrics, and **FDTD cross-validation** (Meep + Tidy3D).

[![CI](https://github.com/Alperen-Durgun/rcwa-metalens/actions/workflows/ci.yml/badge.svg)](https://github.com/Alperen-Durgun/rcwa-metalens/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

---

## ✨ Highlights

- **Core RCWA** — 1D (TE/TM, Li inverse rule, symmetry reduction) and fully vectorial 2D solver; energy-conserving automatic order selection.
- **Stability** — Redheffer star-product + S-matrix; energy conservation verified by unit tests.
- **Materials** — dispersive n(λ)+loss (SiO₂, TiO₂, Si, Ag, SiN); measured n,k tables + `refractiveindex.info` YAML import.
- **Metalens pipeline** — meta-atom library → phase profile → layout → focus; local-periodic approximation.
- **Fabrication** — pure-Python, Unicode-path-safe **GDS-II** writer; arbitrary geometry (circle/square/hexagon/cross/ellipse); cm-scale tiled large lenses.
- **Professional metrics** — FWHM, focusing efficiency (absolute/relative), Strehl, MTF, encircled energy, sidelobe, DOF, chromatic shift, group delay.
- **FDTD bridges** — local **Meep** (offline) and cloud **Tidy3D** for independent validation.
- **Automation** — `run_experiment.py` with **23 experiment types**; `verify_all.py` for one-command end-to-end validation.

## 📦 Installation

```bash
cd rcwa_py
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Required: `numpy scipy matplotlib grcwa autograd gdstk pyyaml`.
Optional: `tidy3d` (cloud FDTD), `meep` (local FDTD, via conda), `jax`/`cupy` (GPU).

## 🚀 Quick start

```python
import numpy as np
from rcwa import solve_rcwa_1d

x = (np.arange(512) + 0.5) / 512
eps = np.where(x < 0.5, 4.0, 1.0)          # binary grating: n=2 / air
res = solve_rcwa_1d(lam0=0.55, theta_deg=15, er_ref=1.0, er_trn=2.25,
                    period=1.2, layers=[(eps, 0.30)], P=41, pol=(0, 1))  # TE
print(res['Rtot'], res['Ttot'], res['Rtot'] + res['Ttot'])  # ~ energy conserved
```

Full metalens design (design + metrics + GDS):

```bash
python metadesign.py metalens_design_full D_lens=16 f=20 material=TiO2_t lam0=0.633
```

## ✅ Validation

```bash
python verify_all.py            # unit tests + metrics + 23 generators + convention audit
python verify_all.py --report   # + Markdown report
python tests/test_rcwa.py       # unit tests only (30 tests)
```

The solver is validated against Tidy3D FDTD at the unit-cell level to within **0.04%**,
and at the full-lens level against Meep/Tidy3D 3D FDTD via **diffraction-limited focus**.

## 🗂️ Structure

```
rcwa_py/
├── rcwa/                    # core package
│   ├── solver.py           # 1D RCWA solver
│   ├── solver1d_scalar.py  # scalar 1D (Li inverse rule + symmetry)
│   ├── solver2d.py         # fully vectorial 2D solver
│   ├── convmat.py          # convolution matrix (1D+2D)
│   ├── layer.py            # eigenmodes + layer S-matrix
│   ├── redheffer.py        # star-product
│   ├── stable.py           # energy-conserving automatic-M selection
│   ├── materials.py        # dispersive n(λ), measured n,k
│   ├── metrics.py          # FWHM / efficiency / Strehl / MTF / DOF / group delay
│   ├── shapes.py           # arbitrary geometry + GDS polygons
│   ├── gds.py / gdswriter.py  # pure-Python, Unicode-safe GDS-II writer
│   ├── largelens.py        # cm-scale tiled large lens
│   ├── fullwave.py         # Meep + Tidy3D FDTD bridge
│   ├── adjoint.py          # adjoint / inverse design
│   ├── hiacc2d.py          # metallic-2D + hexagonal lattice
│   ├── exporting.py        # BSDF / far-field / DRC
│   ├── pipeline.py         # Zemax phase-CSV + tiling
│   ├── backends.py         # GPU backends (jax/cupy)
│   ├── cache.py            # library cache
│   └── materials_data/     # measured n,k tables
├── examples/               # 24 examples (RCWA + Meep + Tidy3D)
├── tests/test_rcwa.py      # 30 unit tests
├── run_experiment.py       # automation for 23 experiment types
├── metadesign.py           # command-line interface
├── verify_all.py           # end-to-end validation
├── check_env.py            # environment check
└── requirements.txt
```

## 🔬 Experiment types (summary)

`grating`, `metalens_library`, `angle_sweep`, `grating2d`, `metalens_pillar_2d`,
`metalens_layout_2d`, `param_map_2d`, `lens_fullwave_1d`, `wavelength_sweep`,
`birefringent_library`, `jones_matrix`, `pb_phase_library`, `inverse_design_deflector`,
`achromatic_analysis`, `achromatic_design`, `vortex_beam`, `hologram`,
`polarization_multiplexed`, `metalens_design_full`, `tolerance_analysis`,
`fov_analysis`, `index_robustness`, `metalens_large_tiled`.

Run `metadesign.py <type>` to see each type's parameters.

## 🔑 Tidy3D (optional)

Do **not** hardcode the API key. Configure it once:

```bash
tidy3d configure      # the key is written to ~/.tidy3d/config (outside the repo, gitignored)
```

## 📄 License

[MIT](LICENSE) © 2026 Alperen Durgun
