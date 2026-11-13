# recipe_search — recipe_Si3N4_532nm

Parameters: {'material': 'Si3N4_Luke', 'lam0': '0.532', 'H_min': '0.60', 'H_max': '1.20', 'n_H': '7', 'L_min': '0.24', 'L_max': '0.36', 'n_L': '5'}

- **Platform:** Si3N4_Luke (circle) at λ = 532 nm, substrate n = 1.461 (SiO2_Malitson), exit medium air
- **Limits:** min feature 60 nm, min gap 60 nm, aspect ratio ≤ 15; size error σ = 3 nm; RCWA M = 9
- **Recommended:** H = 800 nm, L = 300 nm, sizes 60–240 nm — robust FOM 0.947, phase gap 0.56 rad, 0 deep resonance(s), sampling allows NA ≤ 0.89 (rule: within 0.01 of the best FOM, fewest deep resonances, then highest NA)

| H (nm) | L (nm) | sizes (nm) | aspect | coverage (2π) | max phase gap (rad) | min \|t\| | resonances nm (depth \|t\|) | max NA (λ/2L) | FOM nominal | FOM robust |
|---|---|---|---|---|---|---|---|---|---|---|
| 900 | 330 | 60–270 | 15.0 | 1.22 | 0.28 | 0.04 | 246 (0.03) | 0.81 | 0.976 | **0.956** |
| 800 | 300 | 60–240 | 13.3 | 0.91 | 0.56 | 1.18 | — | 0.89 | 0.965 | **0.947** |
| 900 | 300 | 60–240 | 15.0 | 1.06 | 0.31 | 1.06 | — | 0.89 | 0.968 | **0.945** |
| 900 | 270 | 60–210 | 15.0 | 0.88 | 0.75 | 1.18 | — | 0.99 | 0.966 | **0.944** |
| 700 | 360 | 60–300 | 11.7 | 1.13 | 0.28 | 0.05 | 208 (0.23), 240 (0.04), 290 (0.09) | 0.74 | 0.963 | **0.942** |
| 700 | 300 | 60–240 | 11.7 | 0.81 | 1.19 | 1.19 | — | 0.89 | 0.951 | **0.935** |
| 600 | 360 | 60–300 | 10.0 | 1.00 | 0.65 | 0.20 | 210 (0.45), 260 (0.17), 293 (0.26) | 0.74 | 0.933 | **0.931** |
| 800 | 330 | 60–270 | 13.3 | 1.17 | 0.31 | 0.05 | 222 (0.52), 254 (0.04) | 0.81 | 0.951 | **0.922** |

- Infeasible candidates: 15 (aspect ratio 16.7 > 15.0; aspect ratio 18.3 > 15.0; aspect ratio 20.0 > 15.0)
- FOM = focal intensity of an ideal lens from this library relative to a perfect phase plate; locally periodic approximation, normal incidence. Verify the chosen recipe with FDTD (examples/17).

**Summary:** recommended H=800nm L=300nm FOM 0.947
