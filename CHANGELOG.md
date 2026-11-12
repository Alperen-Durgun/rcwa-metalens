# Changelog

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
