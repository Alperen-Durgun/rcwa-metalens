# Conventions, definitions and accuracy guidance

## Units and geometry

* Lengths are in **micrometres**, angles in **degrees** (the only exception: `rcwa.adjoint`
  passes `theta`/`phi` straight to grcwa, in radians).
* The structure is periodic in x (1D) or x and y (2D); light travels toward **+z**, from the
  incidence medium (`er_ref`, the substrate side in the design tools) into the exit medium (`er_trn`).
* Permittivity profiles are sampled uniformly over one period: `eps_x[i]` at
  `x = (i + 0.5)/N · period`; 2D arrays are indexed `[ix, iy]`.
* `pol = (TM, TE)` amplitudes. TE is the field along y for incidence in the x–z plane.
* Diffraction order `m` has the transverse wavevector `kx_m = kx_inc − m·λ/Λ`
  (normalized to `k0`). Results list the orders in `res["orders"]`; the zeroth order is at
  `res["i0"]` (alias `res["zeroth_index"]`).

## Time convention — the one that bites

The solvers work in the engineering convention **`exp(+jωt)`**: a wave travelling toward +z is
`exp(−jkz)` and the transmission phase becomes *more negative* as the optical delay grows.

Lens formulas, the angular-spectrum propagator `exp(+i·kz·z)` and FDTD tools (Tidy3D, Meep)
use the physics convention **`exp(−iωt)`**. The two are complex conjugates of each other.
Efficiencies (`R`, `T`) are identical in both.

| You have | You want | Do |
|---|---|---|
| solver amplitude `t` | physics-convention amplitude | `rcwa.to_physical(t)` (= `conj(t)`) |
| solver amplitude `t` | phase that grows with delay | `rcwa.physical_phase(t)` (= `−angle(t)`) |
| a 1D solve | the field behind the grating | `rcwa.convention.propagate_orders_1d(...)` |

`rcwa.design` converts once, right after the solve; everything it returns (`lib["t"]`,
`E0`, phases) is in the physics convention. A lens laid out from raw solver phases
**diverges** — the test `test_lens_1d_physical_focus` demonstrates this with a full
super-cell solve.

## Materials

* `eps = (n + ik)²`; **`Im(eps) > 0` is absorption** (as in refractiveindex.info, grcwa,
  Tidy3D). The solvers conjugate the inputs internally, so pass physical values.
* `materials.info(name)` gives the source, the tabulated range and whether the data is
  approximate. Tables do not extrapolate — a wavelength outside the range raises an error.
* The built-in tables are coarse. For anything you intend to fabricate, register measured
  data: `materials.register_yaml("MyTiO2", "path/to/refractiveindex.yml")`.

## Meta-atom library and atom selection

* `design.meta_atom_library` solves one pillar per size in a square lattice (period `L`,
  height `H`), incident from the substrate (`n_sub`), and stores the zeroth-order
  co-polarized transmission coefficient `t`. `lib["coverage"] ≥ 1` means the sizes span 2π.
* `select_atoms(mode="complex")` picks, for each target phase φ, the atom minimizing
  `|t/|t|max − exp(iφ)|`. Compared with nearest-phase matching (`mode="phase"`) it avoids
  atoms that hit the right phase at a transmission dip. The lens's free global phase offset
  is optimized as well (24 trial offsets; `largelens` uses offset 0).
* The whole approach is the **locally periodic approximation**: neighbour coupling between
  *different* pillars is ignored. It is the standard first-pass design method and it is an
  approximation.

### Resonance-aware selection

A uniform size sweep can step over a narrow guided-mode resonance. The default TiO₂ library at
633 nm has one at ≈ 221 nm that is about 2 nm wide (|t| → 0.01), between two samples that both
look bright. Fabrication errors of a few nm then push atoms onto it.

* `refine_library(lib)` splits every interval whose midpoint deviates from complex linear
  interpolation by more than `tol` (default 0.05 of max |t|), down to `min_step` (0.5 nm), with
  at most `max_solves` extra solves. A resonance much narrower than an interval that leaves no
  trace at the interval's ends or midpoint can still be missed.
* Resonances are detected as |t| dips (prominence ≥ 0.2). For a lossless sub-wavelength lattice
  with one propagating order, a resonance drives |t| toward zero, so a deep dip is the relevant
  signature. Lossy or multi-order cases are not covered by this argument.
* `guard_nm` excludes atoms within that distance of a resonance band. `mode="robust"` with
  `sigma_nm` minimizes the expected error over N(0, σ²) size errors instead. It needs a refined
  library, because the size response is interpolated between samples.
* `lens_tolerance(design, lib, bias_nm, sigma_nm, n_trial)` evaluates a lens with every pillar
  at `s + bias + N(0, σ²)`, using the same interpolation. Measured against re-solved RCWA for
  uniform bias: efficiency within 0.7 points (slightly pessimistic). Sizes pushed beyond the
  library range are clamped to the end sizes; the fraction is reported (`out_of_range`).


`design.analyze_focus` propagates the cell-sampled aperture field with the angular-spectrum
method (4 sub-samples per cell, zero-padded window) and reports:

| Key | Definition |
|---|---|
| `z_focus` | position of the on-axis intensity maximum within 0.4 f … 1.8 f, parabolically refined |
| `fwhm` | full width at half maximum of the **main lobe**, half-maximum crossings interpolated |
| `diffraction_fwhm` | λ / (2·NA), the reference value printed next to it |
| `strehl` | on-axis intensity at `z_focus` divided by that of an aberration-free lens with the *same aperture amplitudes* focused at the *same plane*. ≤ 1 by construction |
| `eff_absolute` | power inside a disk of radius 3·FWHM / power **incident** on the aperture (accounts for `n_sub`) |
| `eff_relative` | same disk / total power in the focal plane (i.e. normalized to what is transmitted) |
| `eff_legacy` | disk of radius 1.5·FWHM / focal-plane power (older reports used this) |
| `ee80` | radius of the peak-centred circle containing 80 % of the focal-plane energy |
| `sidelobe_dB` | highest sidelobe relative to the peak along the x cut |
| `dof` | FWHM of the on-axis intensity I(z) |
| `fresnel_number` | a² / (λ f) with a = D/2 |

**Focal shift.** For a small Fresnel number (N ≲ 5) the intensity maximum of *any* lens lies
closer to the lens than the geometric focus. Small test lenses therefore report
`z_focus < f`; this is diffraction, not a design error, and is why the Strehl ratio is
referenced to the plane of the actual maximum.

## Accuracy guidance

* **Orders (2D).** With the default Li factorization, `M = 5…7` (i.e. 11×11 … 15×15 orders)
  gives a library phase within a few hundredths of a radian of the converged value for
  TiO₂-like pillars. Plain Laurent (`rule="laurent"`) needs far more and is kept only for comparison.
* **Orders (1D super-cell).** A super-cell of width D with index n needs about
  `M ≳ 1.5·n·D/λ`. Energy conservation (ΣDE ≈ 1) is necessary but not sufficient: an
  under-resolved solve can conserve energy and still put the focus in the wrong place.
* **Certificate.** `design.certify_library` re-solves at M + 2 and reports the phase change
  per size. Pillars that nearly touch their neighbours (gap ≲ 60 nm at Λ = 350 nm) converge
  slowest; the default TiO₂ library fails the 0.02 rad limit at M = 7 only there.
* **Symmetry.** Normal incidence + pure x/y polarization + a mirror-symmetric cell is solved
  in the parity subspace automatically (`res["symmetry"]`); identical results, 7–10× faster
  at M = 7–9.
* **Circles.** Li's zigzag rule is exact only for axis-aligned walls, but for circles with
  ≥ 60 nm gaps it still converges geometrically (≤ 0.018 rad at M = 7 against M = 15).
* **Raster.** A binary pixel mask quantizes the pillar size to `L/N`. The designers use
  `shapes.DEFAULT_N = 512` samples per period (0.7 nm at L = 0.35 µm). The solver cost does
  not depend on N. Coarse grids produce stair-stepped phase-versus-size curves.
* **Ill-conditioning.** Thick, high-index layers make RCWA ill-conditioned at particular
  order counts. `solve_1d_auto` / `solve_2d_auto` try several and keep the largest one that
  conserves energy.
* **Rayleigh anomalies.** When an order is exactly at grazing (period = integer × λ/n) the
  transverse wavevectors are detuned by 1e-7 relative to avoid a singular matrix.
* **Metals in 2D.** Li's zigzag factorization is exact only for axis-aligned walls. For
  metallic or curved high-contrast structures compare with `hiacc2d.solve_2d_grcwa` and, for
  final numbers, FDTD.

## GDS output

* Coordinates in micrometres, database unit 1 nm, circles as 48-gons.
* Default output is **hierarchical**: one cell per distinct (shape, size, rotation), the lens
  is a list of references. `hierarchical=False` writes flat polygons.
* `exporting.drc_check` tests the minimum feature and the smallest edge-to-edge gap to the
  eight nearest neighbours. The gap uses the first size parameter as the pillar extent:
  exact for circles and axis-aligned squares, approximate for other shapes.
