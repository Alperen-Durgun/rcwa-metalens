# rcwa-metalens

**A from-scratch RCWA solver and metalens design pipeline in Python** — from a single
meta-atom to a fabrication-ready GDS mask, with the optical metrics and the FDTD bridges
needed to check the result.

[![CI](https://github.com/Alperen-Durgun/rcwa-metalens/actions/workflows/ci.yml/badge.svg)](https://github.com/Alperen-Durgun/rcwa-metalens/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

```
meta-atom library ──► lens phase ──► atom selection ──► focal analysis ──► GDS + DRC
   (RCWA, 2D)        (hyperbolic)    (complex match)    (Strehl, MTF …)    (hierarchical)
                                                              │
                                              design JSON ──► Meep / Tidy3D (FDTD check)
```

## What is inside

| Area | What you get |
|---|---|
| **Solvers** | Full-vector 1D and 2D RCWA (S-matrix, Redheffer star product) with Li's Fourier factorization; scalar 1D solver with mirror-symmetry reduction; energy-gated automatic order selection |
| **Materials** | Dispersive, absorbing materials (`n + ik`); `refractiveindex.info` YAML import; provenance and valid range for every dataset |
| **Design core** | `rcwa.design`: library → hyperbolic phase → vectorized atom selection → angular-spectrum focal analysis, all in one convention |
| **Metrics** | FWHM, absolute/relative focusing efficiency, Strehl (same-plane reference), MTF, encircled energy, sidelobe level, depth of focus, group delay |
| **Fabrication** | Pure-Python GDS-II writer (works with non-ASCII Windows paths), hierarchical output (~36 bytes per pillar), tiled cm-scale lenses, design-rule check |
| **Verification** | Bridges to Meep (local FDTD) and Tidy3D (cloud FDTD); grcwa cross-checks |
| **Automation** | 24 experiment generators behind one command line; optional Markdown-note workflow (Obsidian friendly) |

## Install

```bash
git clone https://github.com/Alperen-Durgun/rcwa-metalens
cd rcwa-metalens
python -m venv .venv
# Windows:    .venv\Scripts\activate
# Linux/Mac:  source .venv/bin/activate
pip install -e .            # or:  pip install -r requirements.txt  (run from the clone, no install)
```

Required: `numpy scipy matplotlib grcwa autograd gdstk pyyaml`.
Optional: `tidy3d` (cloud FDTD, `pip install -e ".[fdtd]"`), `meep` (local FDTD, via conda),
`jax` / `cupy` (GPU). `python check_env.py` reports what is installed and what each package enables.

On Windows, `setup_env.bat` creates the environment, `run_tests.bat` runs the tests.

## Quick start

### Command line

```bash
rcwa-metalens list                                   # the 24 experiment types
rcwa-metalens help metalens_design_full              # its parameters and defaults
rcwa-metalens metalens_design_full D_lens=16 f=20 material=TiO2_t lam0=0.633
```

From a clone without installing, use `python metadesign.py …` with the same arguments.
Results (figure, GDS, design JSON) go to `./workspace/outputs`; choose another place with
`--out DIR` or the `RCWA_PROJECT_DIR` environment variable. Lengths are in micrometres,
angles in degrees. A mistyped parameter is rejected with a suggestion instead of being ignored.

### Python — design a lens

```python
from rcwa import design, gds

d = design.design_lens(lam0=0.633, L=0.35, H=0.60, D_lens=16, f=20,
                       shape="circle", material="TiO2_t")
a = design.analyze_focus(d)
print(f"focus {a['z_focus']:.2f} um, FWHM {a['fwhm']*1000:.0f} nm, "
      f"Strehl {a['strehl']:.2f}, efficiency {a['eff_absolute']:.0%}")

gds.layout_to_gds(d["placements"], "lens.gds")       # fabrication mask

# resonance-aware, tolerance-robust selection (see docs/conventions.md)
lib = design.refine_library(d["lib"])                 # finds narrow resonances, lib["resonances"]
dr = design.design_lens(lam0=0.633, L=0.35, H=0.60, D_lens=16, f=20, material="TiO2_t",
                        lib=lib, mode="robust", sigma_nm=3)
tol = design.lens_tolerance(dr, lib, bias_nm=(-4, 0, 4), sigma_nm=2)
```

### Python — the solver itself

```python
import numpy as np
from rcwa import solve_rcwa_1d

x = (np.arange(512) + 0.5) / 512
eps = np.where(x < 0.5, 4.0, 1.0)                    # binary grating: n = 2 / air
res = solve_rcwa_1d(lam0=0.55, theta_deg=15, er_ref=1.0, er_trn=2.25,
                    period=1.2, layers=[(eps, 0.30)], P=41, pol=(0, 1))   # TE
print(res["Rtot"], res["Ttot"], res["Rtot"] + res["Ttot"])               # energy is conserved
```

## Conventions you must know

* **Phase / time convention.** The solvers return complex amplitudes in the engineering
  convention `exp(+jωt)`; lens formulas, the angular-spectrum propagator and FDTD tools use
  `exp(−iωt)`. They differ by complex conjugation. Use `rcwa.to_physical` /
  `rcwa.physical_phase`, or simply work through `rcwa.design`, which converts once. Using a
  raw solver phase in a lens layout produces a *diverging* lens.
* **Materials.** `eps = (n + ik)²`, `Im(eps) > 0` is absorption.
* **Units.** Micrometres and degrees everywhere.

Details, definitions of every metric and accuracy guidance: [docs/conventions.md](docs/conventions.md).

## Validation

```bash
pytest                              # 52 unit tests + 31 end-to-end tests
python verify_all.py                # human-readable pass/fail table of everything
```

What the tests establish:

* analytic checks — Fresnel slab (to 1e-6), absorbing film, energy conservation in 1D/2D,
  behaviour exactly at a Rayleigh anomaly;
* cross-checks against the independent solver **grcwa** — dielectric, lossy and metallic
  cells, oblique incidence, hexagonal lattice;
* the phase convention — a 1D metalens solved as one full super-cell focuses at the design
  focal length with the converted library and does not focus with the raw solver phase;
* GDS — files are read back with gdstk; hierarchical and flat output describe identical geometry;
* every experiment generator, the command line and the note workflow run end to end.

**FDTD — unit cell (V1, passed).** The meta-atom library was compared with Tidy3D FDTD
(`examples/17_unitcell_validation.py`, record `validation/V1_unitcell_2026-11-13.json`):
TiO₂ circle, λ = 633 nm, Λ = 350 nm, H = 600 nm, normal incidence, RCWA M = 9, FDTD 20 steps
per wavelength, zeroth-order t from the cell-averaged field over an air reference. Both sides in
the physics convention; one constant reference-plane phase is fitted and removed.

| Pillar diameter | \|t\| RCWA | \|t\| FDTD | Δ\|t\| | Δphase (rad) |
|---|---|---|---|---|
| 100 nm | 0.9998 | 1.0000 | −0.0002 | +0.028 |
| 150 nm | 0.9865 | 0.9848 | +0.0017 | +0.009 |
| 190 nm | 0.9934 | 0.9962 | −0.0028 | −0.013 |
| 250 nm | 0.9339 | 0.9331 | +0.0008 | +0.012 |
| 290 nm | 0.9949 | 0.9983 | −0.0034 | −0.034 |
| 215 nm (6 nm from the 221 nm resonance; not judged) | 0.9140 | 0.9021 | +0.0119 | +0.003 |

Acceptance was |Δ|t|| < 0.02 and |Δphase| < 0.1 rad; the largest deviations are 0.0034 and
0.034 rad. Refining the FDTD grid (20 → 30 steps per wavelength) moved the 250 nm result by
0.0007 in |t| and 0.011 rad, the same order as the RCWA–FDTD differences. Scope: one material,
one wavelength, normal incidence, no substrate. **Not yet repeated for this version:** focusing
of a small 3D lens (V2, `examples/16_lens_focus_fdtd.py`).

## Limits — read before sending a mask to a foundry

* Lens designs use the **locally periodic approximation**: each pillar is simulated in an
  infinite array of identical neighbours. It degrades at high NA and for strongly coupled
  pillars; confirm a design with FDTD on a representative section.
* The built-in material tables are **approximate** (a few points across 0.45–0.70 µm). Thin-film
  indices depend on the deposition process — load measured data with
  `materials.register_yaml(...)`. Reports print where the material data came from.
* The full-wave 1D super-cell generator needs roughly `M ≳ 1.5·n·D/λ` orders; the report says
  whether a run was resolved.
* Substrates (`n_sub`) are supported by the RCWA library and passed to the FDTD bridges, but
  the FDTD bridges with a substrate have not been run against a real FDTD install yet.
* GPU back-ends are detected and selected, but have not been exercised on GPU hardware.
* The default uniform size sweep can **step over narrow resonances** (one sits at ≈ 221 nm in
  the default 633 nm TiO₂ library). Use `refine_library` and `mode="robust"` for anything you
  intend to fabricate; `lens_tolerance` reports the size-error sensitivity.

## Experiment types

`grating`, `metalens_library`, `angle_sweep`, `grating2d`, `metalens_pillar_2d`,
`lens_fullwave_1d`, `metalens_layout_2d`, `wavelength_sweep`, `birefringent_library`,
`jones_matrix`, `pb_phase_library`, `param_map_2d`, `inverse_design_deflector`,
`achromatic_analysis`, `achromatic_design`, `vortex_beam`, `hologram`,
`polarization_multiplexed`, `metalens_design_full`, `tolerance_analysis`, `lens_tolerance`,
`fov_analysis`,
`index_robustness`, `metalens_large_tiled`.

`rcwa-metalens list` prints a one-line description of each.

## Repository layout

```
rcwa/                     the package
  solver.py  solver2d.py  solver1d_scalar.py  stable.py      RCWA solvers
  convmat.py  layer.py  redheffer.py                          solver building blocks
  convention.py                                               phase/time convention helpers
  materials.py  materials_data/                               dispersive materials
  shapes.py                                                   meta-atom geometries (raster + polygons)
  design.py                                                   metalens design core
  metrics.py                                                  optical figures of merit
  gds.py  gdswriter.py  largelens.py  exporting.py            GDS, tiled large lenses, DRC, BSDF
  fullwave.py  pipeline.py                                    Meep / Tidy3D / Zemax bridges
  adjoint.py  hiacc2d.py  backends.py                         adjoint design, grcwa bridge, GPU detection
  experiments.py  cli.py                                      24 generators + command line
examples/                 19 Python scripts + 1 shell script (validation, libraries, Meep and Tidy3D runs)
tests/                    test_rcwa.py (unit) · test_experiments.py (end to end)
docs/conventions.md       conventions, metric definitions, accuracy guidance
metadesign.py  run_experiment.py  verify_all.py  check_env.py launchers for use from a clone
```

### Note workflow (optional)

Experiments can also be described as Markdown notes with front matter and run in batch —
handy inside an Obsidian vault:

```markdown
---
experiment_type: metalens_design_full
status: pending
D_lens: 16
f: 20
---
```

Put such notes in `workspace/experiments/`. `rcwa-metalens run --all` runs every pending
note, writes a report note (results table + figure) to `workspace/reports/` and marks the
note done. The original Turkish keys (`deney_tipi`, `durum: bekliyor`) are accepted too,
and inside an Obsidian vault the vault's own folders are used.

## Tidy3D key

Never put the API key in a file of this repository. Configure it once with `tidy3d configure`;
it is stored in `~/.tidy3d/config`.

## Citing, contributing, license

See [CITATION.cff](CITATION.cff), [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).
[MIT](LICENSE) © 2026 Alperen Durgun
