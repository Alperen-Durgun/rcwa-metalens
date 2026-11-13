# Reference recipes

Unit-cell recipes found with `rcwa-metalens recipe_search` (roadmap U1). Each `.md` file is the
generator's report (candidate table, limits, recommendation rule); the `.png` is the height ×
period map of the robust figure of merit. Regenerate with the parameters at the top of each report.

Common limits: circular pillars, fused-silica substrate (Malitson), air above, min feature
60 nm, min gap 60 nm, aspect ratio ≤ 15 (typical e-beam/ALD values, **not** a foundry rule set),
size error σ = 3 nm, RCWA M = 9, locally periodic approximation at normal incidence.

| Platform | Material data | Recommended | Robust FOM | Status |
|---|---|---|---|---|
| 633 nm | ALD TiO₂ (`TiO2_Siefke`) | H = 600 nm, L = 375 nm, sizes 60–315 nm | 0.948 | **Unit cell validated with FDTD** (`validation/V1r_TiO2_Siefke_633_600_375_*`: 0.0022 in \|t\|, 0.0094 rad). Lens-level FDTD (V2) not yet run. |
| 532 nm | LPCVD Si₃N₄ (`Si3N4_Luke`) | H = 800 nm, L = 300 nm, sizes 60–240 nm | 0.947 | RCWA only. Phase gap 0.56 rad (within the π/4 rule). Alternative: H = 900 nm, L = 300 nm (gap 0.31 rad, FOM 0.945, aspect ratio 15 = at the limit). Not FDTD-validated, no convergence certificate yet. |
| 940 nm | amorphous Si (`aSi_Pierce`) | H = 420 nm, L = 350 nm, sizes 60–290 nm | 0.880 | RCWA only; absorption lowers the FOM by ~5–7 points against the dielectric platforms. **Material uncertainty is high:** a single 1972 dataset of an evaporated 60 nm film (46 points); a-Si(:H) films differ strongly — measure your film. High index contrast: certify M before use. |

What the recommendation means: among candidates that close the phase circle (largest phase gap
≤ π/4) and whose FOM is within 0.01 of the best, the one with the fewest deep resonances
(|t| < 0.5), then the highest NA the period can sample (λ/2L). See `docs/conventions.md`,
"Recipe search".

Before fabrication, in this order: certify the library (`design.certify_library`), validate the
unit cell with FDTD (`examples/18`), use measured n, k of your own film (`materials.register_yaml`),
and run `lens_tolerance` for the lens you intend to make.
