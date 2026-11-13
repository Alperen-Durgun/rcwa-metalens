# Changelog

## Unreleased

### Added — one-command recipe with manifest and trust level (roadmap S1-S3)

* `rcwa-metalens recipe spec.yaml [--out DIR] [--jobs N]` (`rcwa/recipe_run.py`,
  `docs/recipe_pipeline.md`, example `docs/recipe_spec_example.yaml`): specification -> recipe
  folder with `report.md` (plan sections R1-R10, trust level first), `recipe.json` (all
  numbers), `lens.gds` and `manifest.json` (code version and commit, normalized specification
  and hash, material data hash, validation record hashes, output hashes).
* Steps: validated specification (unknown keys refused); sourced platform data; unit cell
  fixed or searched under the DRC profile, the sub-wavelength condition and the lens NA
  (L <= lam0 / 2NA); refined library with convergence certificate, certified sizes only;
  robust design; focal analysis; process window (size bias, random size error, film index
  shift with RCWA re-solve of the used sizes); matching validation records; reproducible GDS
  read back; DRC with aspect ratio. An infeasible specification gives a report with the
  reason (exit code 3).
* Trust level T0-T2 from an explicit checklist (plan section 7); T3 needs fabrication data
  and is never assigned. T2 also needs an error budget (roadmap T2, next), so recipes are T1
  for now.
* Reproducibility checked: byte-identical `report.md`, `recipe.json`, `lens.gds` with a warm
  and an empty library cache and with 1 and 2 worker processes. `gds.layout_to_gds` and
  `gdswriter.write_gds` take `timestamp=` (default unchanged: now).
* Checked against V2: the specification of the V2 lens (D 6 / f 6 um) reproduces its LPA
  numbers exactly (focus 5.167 um, FWHM 646 nm, Strehl 0.914). A 2 x 2 search
  (H 0.55/0.60, L 0.35/0.375) picks 600/375 with robust FOM 0.948, as `recipe_search` did.
* Validation records: `lam0` (0.633 um, the example-20 default every V2 run used) added to the
  V2 records and written by example 20 from now on; the two V2 records analysed before the
  focal-analysis fix moved to `validation/superseded/` (README there).
* New tests `tests/test_recipe_run.py` (24), run in CI. `sigma_nm: 0` gives a nominal
  (complex-match) selection and, correctly, trust level T0.

### Fixed — focal analysis window bias (found by V2)

* `design.FocalField`: on-axis values (`on_axis`, `find_focus`, `strehl_at_focus`, depth of
  focus) now use the Rayleigh-Sommerfeld integral of the aperture samples - exact, no window.
  The FFT angular spectrum is periodic and its window was only 1.6 x the aperture, so light
  leaving at steep angles re-entered from the periodic images: the focus of NA 0.45 lenses
  moved by 2-5 % and zig-zagged with the window size (D 6 um: 5.28 / 5.38 / 5.07 / 5.20 um at
  pad 1.6 / 2 / 3 / 4; converged 5.17 um). A band-limited angular spectrum does not cure it.
  Plane fields (FWHM, efficiencies) keep the FFT, now padded 8 x by default (`FFT_PAD`, capped
  at `FFT_MAX_N` = 2048 samples with a warning); FWHM converges to < 1 % there (was up to 4 %
  off). The FFT is computed lazily, only when a plane field is requested.
* Changed results: every lens metric from `analyze_focus` and the generators that use it.
  Benchmark `lens_16um_default`: focus 19.50 -> 19.18 um, Strehl 0.989 -> 0.965, FWHM
  0.847 -> 0.842 um, absolute efficiency 0.830 -> 0.829 (baseline values updated; timings kept).
  Library and recipe figures of merit (`recipe.library_fom`) do not propagate and are unchanged.
* The achromatic-lens experiment uses `FocalField` instead of its own 1.6 x FFT window.
* New test `test_focal_field_on_axis_is_window_independent`.

### V2 lens validation (3D FDTD)

* `examples/20_lens_fdtd_aperture.py`: 633 nm recipe (TiO2_Siefke 600/375 on silica), PEC
  aperture, NA 0.45. Corrected focal analysis vs Tidy3D:
  - D 6 / f 6 um: focus 5.167 vs 5.006 um at 15 steps/wavelength (-3.1 %), 5.060 um at 25 steps
    (-2.1 %, passes 3 %); a second-order extrapolation gives ~5.09 um (~-1.5 %).
  - D 10 / f 10 um (15 steps only; 25 steps would cost ~6 FC): focus 9.382 vs 9.094 um (-3.1 %,
    misses 3 % by 0.1 point; at 15 steps D 6 also gave -3.1 % and improved to -2.1 % at 25).
  - FWHM in the FDTD monitor plane: 654 vs 637 / 631 nm (D 6, 15 / 25 steps), 704 vs 693 nm
    (D 10); relative efficiency 0.928 vs 0.905 (D 6), 0.929 vs 0.902 (D 10).
  The earlier report of this entry (focus -5.2 / -6.6 %, "unresolved sub-cell content") was
  the window bias above, not a limit of the locally periodic approximation; the "LPA" side of
  the near-field diagnostics used the same biased propagator. Remaining gap ~2 % (about half
  FDTD grid error at 15 steps). The per-cell field in FDTD matches the library within 0.15 rad
  rms (rim -0.12 rad, -17 % amplitude); light outside the cells carries 4.8 % of the power.
* The example now compares FWHM and efficiency in the FDTD monitor plane and computes its
  near-field diagnostics with the Rayleigh-Sommerfeld sum. Re-run on the saved data: the FDTD
  near field propagated gives 5.103 / 9.270 um (direct FDTD 5.060 / 9.094); the propagating
  FDTD field averaged per cell, through the LPA propagator, gives 5.043 / 9.192 um. So with
  the per-cell values taken from FDTD the LPA reproduces the FDTD focus; the remaining ~2 %
  comes from the per-cell values (neighbour coupling, phase rms 0.15-0.17 rad).
* Cost (estimates): 2 x 0.258 + 0.896 + ~1.8 FC.

### Changed defaults — the lens generators use the validated recipe

* `metalens_design_full`, `fov_analysis`, `index_robustness`, `lens_tolerance` and
  `metalens_large_tiled` now default to the 633 nm reference recipe (`docs/recipes`): material
  `TiO2_Siefke` (measured ALD film data) instead of the approximate `TiO2_t` table, period
  375 nm instead of 350 nm, fused-silica substrate (`substrate=SiO2_Malitson`; `substrate=air`
  or `n_sub=` to override) instead of none. `experiments.RECIPE` holds the values; the large
  tiled lens now passes the substrate to its look-up table. The CLI follows helper functions
  recursively when it lists a generator's parameters. Old behaviour:
  `material=TiO2_t period=0.35 substrate=air`.

### Validation — the LPA in 2D, metagratings against FDTD (roadmap P3)

* `examples/19_metagrating_lpa_fdtd.py`, `rcwa/lpa.py`: N-pillar metagratings of the 633 nm
  recipe (design order at sin θ = λ/NL), LPA vs supercell RCWA vs Tidy3D (DiffractionMonitor).

  | N | sin θ | LPA | supercell RCWA | FDTD |
  |---|---|---|---|---|
  | 4 | 0.42 | 0.787 | 0.828 | 0.820 |
  | 6 | 0.28 | 0.881 | 0.895 | 0.896 |
  | 8 | 0.21 | 0.918 | 0.909 | 0.902 |

  Supercell RCWA matches FDTD within 0.008; the LPA is within 3.5 points up to sin θ = 0.42 and
  pessimistic at the largest angle (as in the 1D study). Estimated maximum cost 0.109 FlexCredit.

### Your own process data — measured films and DRC profiles (roadmap M3, F1)

* `materials.register_cauchy`, `register_sellmeier` (ellipsometry fits, range-checked, labelled
  user-supplied) and `register_measured_csv` with a format template `docs/material_template.csv`
  (its numbers are marked as format examples, not data). Check: the Sellmeier form reproduces
  Malitson's fused silica to 1e-12. Tauc–Lorentz is not included (no reference implementation
  to verify it against yet).
* `exporting.drc_check(..., profile=, height=, max_aspect=)`: aspect ratio H / min(width, gap)
  per pillar; profiles by name (`DRC_PROFILES`: this package's own defaults, explicitly not
  foundry rules) or from a YAML file (`docs/drc_profile_template.yaml`, all values null — an
  unfilled profile is refused rather than guessed).

### Fabrication geometry — tapered sidewalls and residual film (roadmap P5, M4)

* `meta_atom_library(..., sidewall_deg=, n_steps=, residual=(material, thickness))`, also in
  `design_lens` and `recipe.evaluate_cell`; `design.pillar_layers` builds the stack (residual film
  on the substrate, then the pillar in n_steps slices whose width changes linearly; the size is
  the mid-height width). The default geometry keeps its cache keys.
* Checks: a residual film with the substrate's index adds exactly exp(+i k0 n_sub t); a vertical
  pillar cut into four slices equals the single layer (1e-9); a taper is widest at the bottom.
* **Process sensitivity of the 633 nm TiO₂ recipe** (atoms chosen for vertical walls and no
  residual, the used sizes re-solved with the defect, FOM of that fixed selection): nominal
  0.961; sidewall 88°: 0.912, 86°: 0.916; residual TiO₂ film 10 nm: 0.941, 20 nm: 0.889.
  A 2° taper costs more than a ±4 nm size bias.
* **Designing for the known process recovers it:** the same recipe designed and evaluated with an
  88° library gives FOM (nominal rule) 0.965 — the loss is gone; with a 20 nm residual film 0.921
  (the film's own reflection remains). Measure sidewall angle and residual thickness (SEM, cross
  section) and pass them to the design: `design_lens(..., sidewall_deg=, residual=)`.

### Recipes — certificate and material-index sensitivity (roadmap N1 in the search, M2)

* `recipe.evaluate_cell(..., certify=True)` / `recipe_search certify=1`: the library is
  certified (M vs M+2) and only certified sizes enter the FOM and the phase gap.
* `recipe.index_sensitivity` / `recipe_search index_dn=…`: atoms chosen at the nominal index,
  the used sizes re-solved at n ± Δn, FOM of that fixed selection. `materials.index_spread`
  compares datasets of one material at a wavelength.
* Findings for the reference recipes (details in `docs/recipes/README.md`): the 633 nm TiO₂
  recipe fails the certificate next to its resonances (11 of 61 sizes, up to 0.83 rad at 310 nm)
  but the robust selection never uses those sizes (FOM unchanged when they are excluded); its
  FOM drops from 0.961 to 0.932 for Δn = +0.02, the difference between the two ALD datasets.
  Si₃N₄ (532 nm) and a-Si (940 nm) pass the certificate; Si₃N₄ is the most index-tolerant
  (±0.02: −0.3 points), a-Si the least (±0.2: −9 to −13 points).

### Validation — all three reference recipes against FDTD (unit cell)

| Recipe | max \|Δ\|t\|\| | max \|Δphase\| | fitted constant phase | est. max cost |
|---|---|---|---|---|
| 633 nm ALD TiO₂, H 600 / L 375 | 0.0022 | 0.0094 rad | −0.027 rad | 0.175 + 0.075 FC |
| 532 nm Si₃N₄, H 800 / L 300 | 0.0025 | 0.030 rad | −0.023 rad | 0.150 FC |
| 940 nm a-Si (absorbing), H 420 / L 350 | 0.0035 | 0.028 rad | −0.031 rad | 0.150 FC |

Five sizes each, fused-silica substrate, bare-interface reference, Tidy3D 2.12.2, RCWA M = 11;
records in `validation/V1r_*`. All fields decayed below the shutoff except the two TiO₂ tasks
next to its 311 nm resonance (re-run, see below).

### Validation — the recommended 633 nm recipe against FDTD

* Recipe: ALD TiO₂ (Siefke data), circles, H = 600 nm, L = 375 nm, on fused silica, light from
  the substrate. `examples/18` picked five sizes spread over the phase circle (145–298 nm) and
  one 4 nm from the 311 nm resonance band (reported only); RCWA M = 11, Tidy3D 2.12.2, 20 steps
  per wavelength. Records: `validation/V1r_TiO2_Siefke_633_600_375_*.json`.
* **Passed:** largest |Δ|t|| 0.0022 and |Δphase| 0.0094 rad (limits 0.02 / 0.1). The fitted
  constant phase is −0.027 rad: with the bare-interface reference it should be ~0, an
  independent confirmation of the phase convention.
* Two tasks next to the resonance had not decayed to the 1e-5 shutoff (4.6e-4, 2.1e-4). Re-run
  with 6 ps: 298 nm still passes (|Δphase| 0.012 rad); 306 nm differs by 0.020 in |t| and 0.089
  rad — the locally periodic library is least reliable next to a resonance, which is what the
  guard/robust selection avoids.
* Cost: estimated maximum 0.175 + 0.075 FlexCredit.
* Examples 17/18 switch the console to UTF-8 (Tidy3D's progress arrows crashed a redirected
  cp1252 console after the batch had finished); example 18 gained `--load` (analyse a finished
  batch from disk), `--sizes`, `--run-time`, `--offset`, `--tag`.

### Recipes — unit-cell search (roadmap U1, U2)

* `rcwa.recipe` and the `recipe_search` generator: height × period scan for a material and
  wavelength under fabrication limits (min feature, min gap, aspect ratio) and the
  sub-wavelength condition; each candidate library is refined and scored by a lens figure of
  merit (focal efficiency of an ideal lens built from it with the robust rule; analytic check
  cos²(π/72) for 36 ideal phases). `recipe.recommend` picks, among candidates that close the
  phase circle (gap ≤ π/4) and are within 0.01 of the best FOM, the one with the fewest deep
  resonances, then the highest NA the period can sample (λ/2L).
* `examples/18_recipe_unitcell_validation.py`: FDTD check of any recipe (substrate incidence,
  bare-interface reference, test sizes picked away from resonances).
* **Reference recipes** (`docs/recipes/`, reports + maps, same limits, fused-silica substrate):
  633 nm ALD TiO₂ → H = 600, L = 375 nm (FOM 0.948, unit cell FDTD-validated); 532 nm LPCVD
  Si₃N₄ → H = 800, L = 300 nm (FOM 0.947, aspect 13.3; RCWA only); 940 nm amorphous Si →
  H = 420, L = 350 nm (FOM 0.880, absorbing; RCWA only, single old dataset — measure the film).
  This meets the 0.4 acceptance item "a recipe for each platform or a proof that none exists";
  the SiN and a-Si recipes still need certification and FDTD.
* Fixed: the aspect-ratio limit rejected H = 900 nm at 60 nm features (0.9/0.06 =
  15.000000000000002 > 15). Fixed: the library cache key used unrounded H, L, λ, so a grid from
  `np.linspace` (0.6000000000000001) missed results computed at 0.6.
* CLI: `--name N` names the figure and also writes the report as `N.md`.
* **633 nm, ALD TiO₂ (Siefke), fused-silica substrate, limits 60 nm / 60 nm / aspect ≤ 15,
  σ = 3 nm** (30 candidates, H 550–800 nm × L 300–400 nm): the current recipe H = 600 nm,
  L = 350 nm does **not** close the phase circle with the measured film index (0.88 × 2π, gap
  0.76 rad). Widening the period to 375 nm does (H = 600 nm: 1.05 × 2π, gap 0.30 rad, FOM 0.948,
  two shallow resonances: the library never drops below |t| = 0.61). The highest FOM, 0.955 at H = 650 / L = 400 nm, comes
  with three resonances, the deepest down to |t| = 0.006. The seven best candidates are within one FOM point.
* Docs: restored the "Focal analysis and metric definitions" heading in `docs/conventions.md`
  (lost in an earlier edit); new "Recipe search" section.

### Materials — sourced datasets (roadmap M1)

* 15 refractiveindex.info files shipped unchanged in `rcwa/materials_data/rii/` (CC0 1.0,
  references and DOIs in `SOURCES.md`): TiO₂ (Siefke ALD, Jolivet amorphous ALD, Sarkar film,
  Devore rutile), Si₃N₄ (Luke, Philipp), SiO₂ (Malitson), Al₂O₃ (Malitson, Boidin, Kischkat),
  GaN (Kawashima, Barker), amorphous Si (Pierce), crystalline Si (Green 2008, Schinke).
  `materials.info(name)` returns the reference and license.
* New full reader `materials.load_rii` / `register_rii` (also behind `register_yaml`):
  formulas 1–5, tabulated n / k / nk blocks merged, validity range = intersection, and no
  extrapolation for formulas either. **Fixed:** the old reader used only the first data block
  (a separate k block was ignored) and evaluated any formula as a Sellmeier formula
  (formula 4, e.g. Devore's rutile, gave wrong values); formulas had no range check.
* Checks: Malitson SiO₂ equals the in-code formula to 2e-16 and gives n_d = 1.4585; Devore
  (formula 4) equals its closed form exactly.
* **Finding for the recipe work (0.4):** the default `TiO2_t` table has n = 2.50 at 633 nm; the
  ALD TiO₂ films are 2.39 (Siefke) and 2.41 (Jolivet), another film 2.13 (Sarkar). The default
  recipe therefore assumes a ~0.1 higher index than measured ALD films. Not changed yet: a lower
  index needs a re-optimized height/period (U1).

### Tooling — benchmark suite (roadmap N5)

* `benchmarks/run.py`: four fixed problems (1D slab against Fresnel, 2D pillar at M = 9 with
  and without the symmetry reduction, the default 20-size library, the default 16 µm lens).
  `--check` fails when a value drifts from `benchmarks/baseline.json`; times are only reported.
  Runs in CI. Baseline from the reference laptop (Windows, i5-5200U): the M = 9 pillar takes
  0.64 s reduced vs 5.33 s full (8.3×); the whole default lens 40 s. Linux vs Windows values
  agree to 1e-12.

### Accuracy — Li factorization for circles (roadmap N4)

* Measured, no code change: with ≥ 60 nm between pillars the Li (zigzag) factorization
  converges geometrically for circles (phase error at M = 7 against M = 15: 0.005 rad at
  150 nm, 0.007 rad at 250 nm, 0.018 rad at 290 nm). The slow convergence at 300 nm came from a
  gap resonance, not from the factorization (see the defaults entry). A normal-vector
  factorization is therefore not needed now; revisit for metals or touching pillars.

### Changed defaults — results change (roadmap U4); re-run designs

Measured for the 16 µm TiO₂ test lens (λ = 633 nm, Λ = 350 nm, H = 600 nm, f = 20 µm) on the
reference laptop. Every change can be undone with a parameter.

| Default | 0.2.1 | now | Why |
|---|---|---|---|
| `M` (design, library) | 7 | 9 | passes the convergence certificate (M→M+2: 0.009 rad, M→M+4: 0.014 rad, all sizes); with N2 an M = 9 solve costs less than an M = 7 solve did |
| size range | 60–300 nm (gap 50 nm) | 60–290 nm (gap 60 nm) | a gap resonance at ~302 nm made the 300 nm pillar converge slowly: 0.16 rad error at M = 7 against M = 15, 0.04 rad still at M = 11 |
| selection | `complex` | `robust`, σ = 3 nm | |
| library | uniform 20 sizes | refined (resonances located) | |
| test lens: Strehl / abs. efficiency | 0.963 / 0.816 | 0.989 / 0.830 | |
| same lens, bias −4 / 0 / +4 nm + random σ = 2 nm (mean efficiency) | 0.798 / 0.810 / 0.802 (complex rule, new range) | 0.813 / 0.820 / 0.811 | |

* Old behaviour: `design_lens(..., mode="complex", refine=False, M=7,
  sizes=default_sizes(L, margin=0.05))`. The generators `metalens_design_full`,
  `metalens_layout_2d`, `fov_analysis`, `index_robustness` and `lens_tolerance` follow the new
  defaults (`selection=complex` restores the old rule); `metalens_large_tiled` keeps its own
  complex-rule LUT.
* `design_lens(..., certify=True)` uses only sizes that pass `certify_library`.

### Performance — mirror-symmetry reduction of the 2D solver (roadmap N2)

* `solve_rcwa_2d(..., symmetry="auto")` (default): at normal incidence, with a purely x- or
  y-polarized source and layer rasters that are mirror-symmetric in x and y, the problem is
  solved in the field-parity subspace ((Mx+1)(My+1) + Mx·My unknowns instead of
  2(2Mx+1)(2My+1)); `rcwa/symmetry.py`. The conditions are checked numerically (raster, the
  convolution matrices, the H-field subspace) and the full problem is used when any fails;
  `res["symmetry"]` says which path ran. `symmetry=False` forces the full problem.
* Same result as the full solve: largest difference in any order 1e-12 (tests cover x/y
  polarization, substrate, absorbing pillar, a three-layer stack; fall-back for oblique
  incidence, mixed polarization and an off-centre pillar).
* Speed (TiO₂ circle, one layer, 2-core VM): M = 5: 4.8×, M = 7: 7.3×, M = 9: 10× (6.2 s →
  0.62 s). The remaining time is mostly the (now small) eigenproblem.
* The full path was also made cheaper without changing a bit of its output: diagonal KX/KY
  products as scalings, and the Li matrices from one FFT per axis instead of one per line.

### Accuracy — convergence certificate (roadmap N1)

* `design.certify_library(lib, dM=2)`: re-solves every size at M + 2 and records the phase and
  |t| change per size (`lib["convergence"]`); default limits 0.02 rad and 0.01 outside
  resonance bands, 0.05 rad inside.
* **Finding:** the default library (TiO₂, 633 nm, Λ = 350 nm, H = 600 nm, M = 7, refined to 59
  sizes) does **not** pass: the largest pillar (300 nm, 50 nm gap to the neighbour) changes by
  0.045 rad from M = 7 to M = 9; the resonance shoulders (218–222 nm) by 0.012–0.022 rad. Large
  fill factors need more orders; with N2 that is now affordable.

### Performance — parallel library (roadmap N3)

* `meta_atom_library(..., n_jobs=)` and `refine_library(..., n_jobs=)`; default from the
  `RCWA_N_JOBS` environment variable, else 1 (serial, unchanged). `0` = all cores.
* `parallel_map` now starts workers with "spawn" and limits each worker to one BLAS thread.
  Without the limit the workers competed for the cores: 2 workers were 2x slower than serial
  on a 2-core machine.
* Measured on the reference laptop (i5-5200U, 2 cores / 4 threads, Windows, 8 sizes, M = 7):
  serial 14.6 s, 2 workers 14.2 s, 4 workers 11.3 s (1.3x). The serial solve already uses
  multi-threaded BLAS and spawn start-up costs ~1 s per worker, so the gain is small on this
  hardware; the roadmap target "≥ 3x on 4 cores" is not reachable on 2 physical cores. Results
  agree with the serial run to 1e-11 (not bit for bit: single-threaded BLAS sums in another
  order).

### Validation — unit cell against FDTD (roadmap V1)

* New `examples/17_unitcell_validation.py`: zeroth-order t from the cell-averaged field over
  an air reference, both sides in the physics convention (no sign guessing), six sizes plus a
  finer-grid FDTD check, cost estimate before any run, JSON record in `validation/`.
* Result (Tidy3D 2.12.2, record `validation/V1_unitcell_2026-11-13.json`): passed. Five
  off-resonance sizes agree within 0.0034 in |t| and 0.034 rad in phase (limits 0.02 and 0.1);
  FDTD grid refinement changes the result by 0.0007 / 0.011 rad. Estimated maximum cost of the
  9-task batch: 0.225 FlexCredit.

### Added — lens tolerance analysis (roadmap B4)

* `design.lens_tolerance` and `design.perturbed_field`: Strehl / absolute-efficiency
  distribution of a lens under a uniform size bias plus a random per-pillar error N(0, σ²).
  The size response is interpolated on a refined library (no extra solves). Against re-solved
  RCWA for uniform bias (same lens as below, ±8 nm, three selection rules, 18 cases) the
  interpolated efficiency agreed within 0.7 points, slightly pessimistic; one Strehl value
  differed by 0.017.
* New generator `lens_tolerance` (24th): default vs robust selection side by side. The existing
  `tolerance_analysis` (1D deflector) is unchanged.
* Measured with the generator defaults (D = 16 µm, f = 20 µm, σ = 2 nm random, 10 trials, bias
  −6…+6 nm): mean efficiency 0.749–0.798 (default selection) vs 0.797–0.821 (robust, σ_design =
  3 nm); Strehl 0.880–0.932 vs 0.944–0.958. Error-free values: 0.816 / 0.963 and 0.831 / 0.987.

### Added — resonance-aware, tolerance-robust atom selection (roadmap B1–B3)

* `design.refine_library`: adaptive refinement of a meta-atom library. Each interval is checked
  at its midpoint against complex linear interpolation and split until it agrees within `tol`.
  `design.find_resonances` records the transmission dips (`lib["resonances"]`: centre,
  half-depth band, width, minimum |t|).
* `design.resonance_mask` and `design_lens(guard_nm=...)`: exclude atoms near a resonance.
* `design.size_error_stats` and `mode="robust"` with `sigma_nm`: atoms are chosen by the expected
  squared error under a Gaussian size error, `|E[t]/A − e^{iφ}|² + Var[t]/A²`, from the refined
  library (no extra solves).
* `design.max_phase_gap`; `design_lens` warns when the usable atoms leave a phase gap > π/4.
* Defaults are unchanged (`mode="complex"`, no refinement, no guard). The default selection
  arithmetic is the same as in 0.2.1; the default lens above gives the 0.2.1 numbers.

Measured (TiO₂ circle, λ = 633 nm, Λ = 350 nm, H = 600 nm, M = 7, D = 16 µm, f = 20 µm; uniform
size bias, the shifted sizes re-solved with RCWA; LPA, no FDTD). Absolute efficiency:

| Selection | 0 | −3 nm | −4 nm | −8 nm | +3 nm | +4 nm |
|---|---|---|---|---|---|---|
| default (20 sizes) | 0.816 | 0.723 | 0.721 | 0.770 | 0.803 | 0.803 |
| refined + `guard_nm=4` | 0.831 | 0.830 | 0.824 | 0.768 | 0.816 | 0.821 |
| refined + `mode="robust", sigma_nm=3` | 0.831 | 0.832 | 0.832 | 0.807 | 0.825 | 0.822 |

The refinement found the guided-mode resonance at 221.05 nm with 39 extra solves (59 sizes in
total). The largest drop within ±4 nm goes from 9.5 points to 0.9 points (robust). The
near-resonance atoms are also where the LPA is least reliable, so avoiding them helps for a
second reason that this table does not measure.

## 0.2.1

* Fixed: on Windows the command line crashed with `UnicodeEncodeError` when its output was
  redirected or the console was not UTF-8 (reports contain λ, Σ, θ, ✅). All entry points now
  switch stdout/stderr to UTF-8. Found by the Windows CI job; a regression test was added.
* Fixed: `inverse_design_deflector` reseeded NumPy's global random generator as a side effect;
  it and `hologram` (new `seed` parameter) now use local generators, so results are reproducible
  without touching global state.
* Tests: the cache test uses a private temporary folder (it failed on a second run because of
  the reseeding above).

## 0.2.0

A correctness and structure release. Several results change with respect to 0.1.0 — in every
case because 0.1.0 was wrong or less accurate. Re-run designs made with 0.1.0.

### Fixed — physics

* **1D lens designers produced diverging lenses.** `lens_fullwave_1d`, `achromatic_analysis`
  and examples 08–11 used the raw solver phase (engineering convention `exp(+jωt)`) in a
  physics-convention lens profile; a matching sign error in the propagation hid it. Both are
  corrected, and `rcwa.convention` now defines the rule in one place. The 2D designers were
  already correct.
* **Absorbing materials produced gain** in the native solvers (`Im(eps) > 0` was treated with
  the wrong sign; a lossy film gave R + T ≈ 3). Inputs are now conjugated internally;
  results match the analytic absorbing film and grcwa. Lens designers now include
  absorption instead of silently dropping it.
* **`hiacc2d` passed angles in degrees to grcwa**, which expects radians (wrong results at
  oblique incidence). A polarization argument was added.
* **NaN at an exact Rayleigh anomaly** (period an integer multiple of the wavelength).
* **Strehl ratio could exceed 1** for small lenses; it is now referenced to an ideal lens
  focused at the same plane.
* **FWHM** was quantized to the grid and counted sidelobes above half maximum; it is now the
  interpolated main-lobe width.
* **DRC** inspected only one nearest neighbour and could miss a violation on a square
  lattice; it now checks the eight nearest and scales to millions of pillars.
* `run_fullwave(backend="meep", until=...)` raised `TypeError`.
* Material tables silently clamped outside their wavelength range; they now raise an error.
* `lens_fullwave_1d`: the focus is computed from the exit field windowed to the lens
  aperture (the plane wave leaking through the guard gaps and the periodic replicas used to
  corrupt the result); the largest energy-conserving order count is used.

### Changed — accuracy and speed

* 2D solver: **Li's Fourier factorization is the default** (`rule="li"`). At the same order
  count the library phase is several times closer to the converged value than with the
  plain Laurent rule (`rule="laurent"` is still available).
* Full-vector 1D solver: Li's inverse rule for TM (`rule="inverse"`, default).
* Designers rasterize meta-atoms at 512 samples per period instead of 48–100.
* Solver core: redundant matrix inversions removed, convolution matrices vectorized
  (about one third faster, identical results).
* Atom selection minimizes the complex error and optimizes the global phase offset
  (`selection=phase` restores nearest-phase matching).
* `solve_1d_auto` / `solve_2d_auto` return the largest stable order count.

### Added

* `rcwa.design` — shared design core (library, selection, `FocalField`, `analyze_focus`).
* `rcwa.convention` — `to_physical`, `physical_phase`, `propagate_orders_1d`.
* Command line `rcwa-metalens` (`list`, `help <type>`, `<type> key=value …`, `run`, `--out`);
  parameter help is generated from the generators themselves; unknown parameters are rejected.
* Hierarchical GDS output (default): about 36 bytes per pillar instead of about 415.
* Substrate index `n_sub` in the design generators and in the FDTD bridges.
* `materials.info()` — provenance and valid range; helpful errors for unknown names.
* End-to-end tests for all 23 generators, the CLI and the note workflow; lint in CI; CI
  installs the package and runs the installed command; Windows job.
* Note workflow accepts English front matter (`experiment_type`, `status: pending`).

### Moved / renamed

* `run_experiment.py` → `rcwa/experiments.py`, `metadesign.py` → `rcwa/cli.py` (thin launchers
  with the old names remain in the repository root).
* Outputs of a plain clone go to `./workspace/{experiments,reports,outputs}` (previously two
  folders above the repository). Inside an Obsidian vault the vault folders are still used.
  The cache moved to `~/.cache/rcwa-metalens` (`RCWA_CACHE_DIR`).
* `.bat` launchers: `setup_env.bat`, `run_experiments.bat`, `run_tests.bat`, `create_shortcuts.bat`.
* `examples/05_grcwa_karsilastirma.py` → `05_grcwa_comparison.py`; examples write to `examples/output/`.
* Result keys: `M_denenen` → `M_tried`; DRC violations `tip`/`deger` → `type`/`value`;
  `fullwave` `not` → `note`; lattice label `non-ortogonal` → `non-orthogonal`.
* Library phases in reports are given in the physics convention (increasing with delay).
* Code, comments and reports translated to English.

### Not re-validated in this release

* The FDTD comparisons (Tidy3D / Meep) were made with 0.1.0 and have not been repeated.
* FDTD bridges with a substrate and the GPU back-ends have not been run on real installations.

## 0.1.0

Initial public version.
