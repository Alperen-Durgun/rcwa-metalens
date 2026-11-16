# Solver cross-checks

Comparisons of the RCWA solver with another RCWA code on the same unit cell. They are not
full-wave (FDTD) validations and are not read by the recipe pipeline (only `V1*` / `V2*`
records in `validation/` count for the trust level).

* `planopsim_TiO2_Siefke_633_600_375_2026-11-16.json` - PlanOpSim cloud RCWA (trial, Python
  SDK 5.1.0), TiO2 circular pillars H 600 / L 375 nm on fused silica, 633 nm, diameters
  100-310 nm. Same material data on both sides. 100-300 nm: |t| agrees to 0.003 and, after
  a constant phase of 0.05 rad, the phase to 0.021 rad at PlanOpSim accuracy 12 (0.0044 and
  0.031 rad at accuracy 8). PlanOpSim moves towards our values as its accuracy rises (8 to 12:
  up to 0.032 rad), ours is converged (M 9 to 13: < 0.005 rad). At 310 nm both codes are
  unconverged on a sharp resonance; our certificate rejects that size.
