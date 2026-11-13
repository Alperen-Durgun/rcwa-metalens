# recipe_search — recipe_TiO2-ALD_633nm

Parameters: {'material': 'TiO2_Siefke', 'lam0': '0.633', 'H_min': '0.55', 'H_max': '0.80', 'n_H': '6', 'L_min': '0.30', 'L_max': '0.40', 'n_L': '5'}

- **Platform:** TiO2_Siefke (circle) at λ = 633 nm, substrate n = 1.457 (SiO2_Malitson), exit medium air
- **Limits:** min feature 60 nm, min gap 60 nm, aspect ratio ≤ 15; size error σ = 3 nm; RCWA M = 9
- **Recommended:** H = 600 nm, L = 375 nm, sizes 60–315 nm — robust FOM 0.948, phase gap 0.30 rad, 0 deep resonance(s), sampling allows NA ≤ 0.84 (rule: within 0.01 of the best FOM, fewest deep resonances, then highest NA)

| H (nm) | L (nm) | sizes (nm) | aspect | coverage (2π) | max phase gap (rad) | min \|t\| | resonances nm (depth \|t\|) | max NA (λ/2L) | FOM nominal | FOM robust |
|---|---|---|---|---|---|---|---|---|---|---|
| 650 | 400 | 60–340 | 10.8 | 1.29 | 0.22 | 0.01 | 215 (0.23), 278 (0.28), 334 (0.01) | 0.79 | 0.974 | **0.955** |
| 700 | 400 | 60–340 | 11.7 | 1.34 | 0.22 | 0.04 | 213 (0.12), 261 (0.03), 306 (0.51), 331 (0.12) | 0.79 | 0.975 | **0.951** |
| 650 | 375 | 60–315 | 10.8 | 1.12 | 0.32 | 0.17 | 222 (0.28), 292 (0.14) | 0.84 | 0.970 | **0.949** |
| 600 | 375 | 60–315 | 10.0 | 1.05 | 0.30 | 0.61 | 225 (0.72), 311 (0.50) | 0.84 | 0.966 | **0.948** |
| 550 | 375 | 60–315 | 9.2 | 1.04 | 0.27 | 0.28 | 231 (0.68), 314 (0.23) | 0.84 | 0.964 | **0.946** |
| 550 | 400 | 60–340 | 9.2 | 1.22 | 0.25 | 0.08 | 222 (0.08), 308 (0.56), 331 (0.06) | 0.79 | 0.966 | **0.946** |
| 600 | 400 | 60–340 | 10.0 | 1.20 | 0.23 | 0.13 | 298 (0.30), 312 (0.13), 338 (0.11) | 0.79 | 0.970 | **0.945** |
| 650 | 350 | 60–290 | 10.8 | 0.97 | 0.30 | 0.05 | 232 (0.04) | 0.90 | 0.968 | **0.944** |
- FOM = focal intensity of an ideal lens from this library relative to a perfect phase plate; locally periodic approximation, normal incidence. Verify the chosen recipe with FDTD (examples/17).

**Summary:** recommended H=600nm L=375nm FOM 0.948
