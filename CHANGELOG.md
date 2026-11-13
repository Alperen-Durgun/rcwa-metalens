# Changelog

## Unreleased

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
