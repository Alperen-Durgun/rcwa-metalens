# Reference recipes

Unit-cell recipes found with `rcwa-metalens recipe_search` (roadmap U1). Each `.md` file is the
generator's report (candidate table, limits, recommendation rule); the `.png` is the height ×
period map of the robust figure of merit. Regenerate with the parameters at the top of each report.

Common limits: circular pillars, fused-silica substrate (Malitson), air above, min feature
60 nm, min gap 60 nm, aspect ratio ≤ 15 (typical e-beam/ALD values, **not** a foundry rule set),
size error σ = 3 nm, RCWA M = 9, locally periodic approximation at normal incidence.

| Platform | Material data | Recommended | Robust FOM | Status |
|---|---|---|---|---|
| 633 nm | ALD TiO₂ (`TiO2_Siefke`) | H = 600 nm, L = 375 nm, sizes 60–307 nm except 225 nm (certificate) | 0.948 | **Unit cell validated with FDTD** (`validation/V1r_TiO2_Siefke_633_600_375_*`: 0.0022 in \|t\|, 0.0094 rad). Lens-level FDTD (V2) not yet run. |
| 532 nm | LPCVD Si₃N₄ (`Si3N4_Luke`) | H = 800 nm, L = 300 nm, sizes 60–240 nm | 0.947 | **Unit cell validated with FDTD** (`validation/V1r_Si3N4_Luke_532_800_300_*`: 0.0025 in \|t\|, 0.030 rad). Phase gap 0.56 rad (within the π/4 rule). Alternative: H = 900 nm, L = 300 nm (gap 0.31 rad, FOM 0.945, aspect ratio 15 = at the limit). |
| 940 nm | amorphous Si (`aSi_Pierce`) | H = 420 nm, L = 350 nm, sizes 60–290 nm | 0.880 | **Unit cell validated with FDTD** (`validation/V1r_aSi_Pierce_940_420_350_*`: 0.0035 in \|t\|, 0.028 rad, absorbing pillar). Absorption lowers the FOM by ~5–7 points against the dielectric platforms. **Material uncertainty is high:** a single 1972 dataset of an evaporated 60 nm film (46 points); a-Si(:H) films differ strongly — measure your film. High index contrast: certify M before use. |

## Convergence certificate and index sensitivity of the recommended recipes

Measured after the search (`design.certify_library`, M = 9 vs 11; `recipe.index_sensitivity`,
atoms fixed at the nominal index, used sizes re-solved).

| Recipe | Certificate | Sizes to avoid | FOM at Δn (nominal → shifted) | Dataset spread at λ |
|---|---|---|---|---|
| 633 nm TiO₂, 600/375 | 11 of 61 sizes fail, all next to the 225 nm and 311 nm resonances (up to 0.83 rad at 310 nm) | **225 nm and 308–315 nm** — the robust selection never uses them (FOM unchanged with them excluded) | 0.961 → −0.05: 0.953, −0.02: 0.960, +0.02: **0.932**, +0.05: 0.946 (not monotonic: a resonance moves onto a used atom) | 0.021 between the two ALD films (Siefke 2.389, Jolivet 2.410); Sarkar film 2.130 |
| 532 nm Si₃N₄, 800/300 | passes (0.003 rad) | none | 0.965 → ±0.02: 0.961–0.962, ±0.05: 0.946–0.947 (smooth) | 0.029 (Luke 2.056, Philipp 2.027) |
| 940 nm a-Si, 420/350 | passes (0.006 rad) | none | 0.900 → ±0.1: 0.874–0.875, −0.2: 0.808, +0.2: 0.765 | one dataset only |

Practical reading: the Si₃N₄ recipe is the most tolerant one; the TiO₂ recipe loses ~3 FOM
points if the film is 0.02 higher than the Siefke data, which is exactly the difference to the
Jolivet ALD film — measure your film before you fabricate. Use `certify=1 index_dn=0.02` in
`recipe_search` to get both checks in the report.

## Process sensitivity (633 nm TiO₂ recipe)

Atoms chosen for vertical walls and no residual film, used sizes re-solved with the defect:

| Defect | FOM | Designed for the defect instead |
|---|---|---|
| none | 0.961 | — |
| sidewall 88° | 0.912 | 0.965 (`sidewall_deg=88`) |
| sidewall 86° | 0.916 | — |
| TiO₂ residual film 10 nm | 0.941 | — |
| TiO₂ residual film 20 nm | 0.889 | 0.921 (`residual=("TiO2_Siefke", 0.02)`) |

A 2° taper costs ~5 FOM points — more than a ±4 nm size bias — but almost nothing if the design
uses the measured angle. Characterize the process first (sidewall angle, residual, film index),
then design with those numbers.

What the recommendation means: among candidates that close the phase circle (largest phase gap
≤ π/4) and whose FOM is within 0.01 of the best, the one with the fewest deep resonances
(|t| < 0.5), then the highest NA the period can sample (λ/2L). See `docs/conventions.md`,
"Recipe search".

Before fabrication, in this order: certify the library (`design.certify_library`), validate the
unit cell with FDTD (`examples/18`; done for these three), use measured n, k of your own film (`materials.register_yaml`),
and run `lens_tolerance` for the lens you intend to make.
