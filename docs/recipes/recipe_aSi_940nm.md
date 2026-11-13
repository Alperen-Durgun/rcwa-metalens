# recipe_search — recipe_aSi_940nm

Parameters: {'material': 'aSi_Pierce', 'lam0': '0.940', 'H_min': '0.30', 'H_max': '0.60', 'n_H': '6', 'L_min': '0.30', 'L_max': '0.45', 'n_L': '4'}

- **Platform:** aSi_Pierce (circle) at λ = 940 nm, substrate n = 1.451 (SiO2_Malitson), exit medium air
- **Limits:** min feature 60 nm, min gap 60 nm, aspect ratio ≤ 15; size error σ = 3 nm; RCWA M = 9
- **Recommended:** H = 420 nm, L = 350 nm, sizes 60–290 nm — robust FOM 0.880, phase gap 0.37 rad, 0 deep resonance(s), sampling allows NA ≤ 1.00 (rule: within 0.01 of the best FOM, fewest deep resonances, then highest NA)

| H (nm) | L (nm) | sizes (nm) | aspect | coverage (2π) | max phase gap (rad) | min \|t\| | resonances nm (depth \|t\|) | max NA (λ/2L) | FOM nominal | FOM robust |
|---|---|---|---|---|---|---|---|---|---|---|
| 480 | 400 | 60–340 | 8.0 | 1.10 | 0.25 | 0.52 | 274 (0.44), 318 (0.59) | 1.00 | 0.920 | **0.890** |
| 420 | 350 | 60–290 | 7.0 | 0.94 | 0.37 | 1.12 | — | 1.00 | 0.900 | **0.880** |
| 480 | 350 | 60–290 | 8.0 | 0.99 | 0.27 | 1.08 | — | 1.00 | 0.893 | **0.873** |
| 540 | 400 | 60–340 | 9.0 | 1.18 | 0.31 | 0.56 | 316 (0.47) | 1.00 | 0.897 | **0.869** |
| 600 | 300 | 60–240 | 10.0 | 1.05 | 0.30 | 0.76 | — | 1.00 | 0.876 | **0.855** |
| 540 | 300 | 60–240 | 9.0 | 0.94 | 0.38 | 1.04 | — | 1.00 | 0.875 | **0.854** |
| 600 | 450 | 60–390 | 10.0 | 1.58 | 0.27 | 0.10 | 312 (0.27), 343 (0.09) | 1.00 | 0.891 | **0.853** |
| 540 | 350 | 60–290 | 9.0 | 1.05 | 0.29 | 0.31 | 253 (0.26) | 1.00 | 0.874 | **0.852** |
- FOM = focal intensity of an ideal lens from this library relative to a perfect phase plate; locally periodic approximation, normal incidence. Verify the chosen recipe with FDTD (examples/17).

**Summary:** recommended H=420nm L=350nm FOM 0.880
