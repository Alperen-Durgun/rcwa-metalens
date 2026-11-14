# Superseded validation records

Kept for traceability; the recipe pipeline does not read this folder.

* `V2_lens_D6_f6_TiO2_Siefke_2026-11-13_run1.json` - first V2 run (15 FDTD steps per
  wavelength, no near-field monitor). Its LPA numbers come from the focal analysis before
  the window fix (FocalField FFT window 1.6 x the aperture), which biased the focus by ~2 %.
* `V2_lens_D6_f6_TiO2_Siefke_2026-11-13_steps15.json` - the same lens at 15 steps with the
  near-field monitor, analysed before the fix as well. Its FDTD values stay valid
  (focus 5.006 um at 15 steps); the current record `../V2_lens_D6_f6_TiO2_Siefke_2026-11-13.json`
  is the 25-step run analysed with the corrected focal analysis.

`lam0` (0.633 um) was added to both afterwards: example 20 did not store it, and every V2
run used its default wavelength.

* `V2_lens_D6_f6_TiO2_Siefke_2026-11-13.json` and `V2_lens_D10_f10_TiO2_Siefke_2026-11-13.json` -
  superseded by the 2026-11-14 records: the same FDTD data (re-analysed from the saved
  simulation files, no new run) with the power-flow relative efficiency and the FWHM along the
  polarization. Their relative efficiency compared |E|^2 over different windows (LPA: 8 x the
  aperture, FDTD: the 8 / 12 um monitor) and is not a valid comparison.

* `V1r_TiO2_Siefke_633_600_375_square_2026-11-14.json` (FAILED) and
  `V1r_TiO2_Siefke_633_600_375_square_conv40_2026-11-14.json` (passed, two sizes) - diagnostic
  runs of square pillars. At 20 FDTD steps per wavelength the gap resonance near 280 nm sat
  about one FDTD cell (~13 nm) lower than in RCWA (FDTD |t| 0.61 vs 0.98 at 272.5 nm); its
  outlier also shifted the constant-phase fit (that analysis was then made robust). RCWA is
  converged there (M 9-17), and at 40 steps FDTD agrees. The validation record is
  `../V1r_TiO2_Siefke_633_600_375_square_s40_2026-11-14.json` (same six sizes, 40 steps, passed).
