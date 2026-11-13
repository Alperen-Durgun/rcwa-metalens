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
