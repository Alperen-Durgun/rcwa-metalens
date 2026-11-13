# Recipe pipeline

```
rcwa-metalens recipe spec.yaml [--out DIR] [--jobs N]
```

turns a specification (example: [`recipe_spec_example.yaml`](recipe_spec_example.yaml)) into a
recipe folder (default `recipes/<name>/`):

| File | Content |
|---|---|
| `report.md` | the recipe report, sections R1-R10 of the plan, trust level first |
| `recipe.json` | every number in the report, plus the library table and all unit-cell candidates |
| `lens.gds` | the mask layout (hierarchical GDSII, layer 1/0, fixed date) |
| `manifest.json` | traceability: code version and commit, normalized specification and its hash, material data hash, validation record hashes, hashes of the files above |

Exit code 0: recipe written; 3: the specification cannot be met (the report says why); 2: the
specification is invalid (unknown key, missing value, wrong type or range - nothing is guessed).

## Steps

| Step | What happens | Plan |
|---|---|---|
| 1 Specification | all keys checked, defaults filled in | - |
| 2 Platform | material provenance (reference, licence, data file hash), indices at the wavelength; a wavelength outside the data range is refused | R1 |
| 3 Unit cell | the fixed cell, or every cell of the search, is checked against the DRC profile (min feature, min gap, aspect ratio), the sub-wavelength condition and the lens NA (L <= lam0 / 2NA); feasible cells get a refined, certified library and a robust figure of merit; `recipe.recommend` chooses | R2 |
| 4 Library | refined adaptively (resonances located), convergence certificate M vs M+2 on every size; sizes that fail are not used | R3 |
| 5 Lens design | hyperbolic phase, robust atom selection for the stated size error, from certified sizes only | R4 |
| 6 Performance | focal analysis (`design.analyze_focus`): focus, FWHM, Strehl, absolute and relative efficiency, each with an error budget (`rcwa/budget.py`) | R5 |
| 7 Process window | the budget scenarios in table form: uniform size bias (+-bias, +-bias/2), random size error (Monte Carlo), film index +-dn (used sizes re-solved with RCWA) | R6 |
| 8 Validation | full-wave records in `validation/` that match material, wavelength, H, L and substrate | R8 |
| 9 Fabrication | GDS written and read back, DRC against the named profile including aspect ratio | R7 |
| 10 Report | trust level with the checklist behind it; manifest | R9, R10 |

## Trust level

| Level | Requires (all) |
|---|---|
| T0 | - |
| T1 | material data from a cited source (shipped dataset or your registered measurement; the approximate built-in tables do not count); every size the lens uses passes the convergence certificate; refined library; robust selection; process window evaluated; DRC clean |
| T2 | T1 + a passed unit-cell FDTD record (V1) and a passed small-lens FDTD record (V2) for this platform + a complete error budget (all terms evaluated, model term measured for focus, FWHM and relative efficiency at an NA at least the lens NA) |
| T3 | T2 + calibration with fabrication data (SEM sizes, measured optics; roadmap F4) - never assigned by the pipeline |

The report lists every requirement as met or not and names what the next level needs.

## Off-axis performance (optional, roadmap P2)

With `field_of_view: {angles_deg: [...], pol: TE}` in the specification, every pillar size the
lens uses is solved again with RCWA at each field angle (`rcwa/angular.py`; the angle is in air
before the substrate, the plane of incidence is x-z), and the report adds the off-axis spot on
a flat sensor at the normal-incidence focus for two models: the angle-dependent atom response
and the former fixed response (normal-incidence library plus tilt phase). The two agree as long
as only the zeroth order propagates; for a period L a first diffraction order opens in the
substrate above asin(lam0/L - n_sub) (13.4 deg for 633 nm, L = 375 nm, fused silica) and the fixed
response becomes optimistic: for a D 10 / f 20 um lens of the 633 nm recipe the relative peak at
20 deg is 0.786 instead of 0.893. The oblique solver is checked against an analytic thin-film
transfer matrix (TE and TM, 0-35 deg, < 1e-10) and against grcwa (pillar, 20 deg). The off-axis
numbers have no error budget: the full-wave model check (V2) was done at normal incidence.

## Error budget (roadmap T2)

For every metric the budget lists six terms; each is the largest deviation from the nominal
prediction over its scenarios:

| Term | Scenarios |
|---|---|
| RCWA M+2 | the sizes the lens uses re-solved with M + 2 Fourier orders |
| FFT window | focal-plane metrics with an FFT window twice as large (on-axis values are exact: 0) |
| film index | pillar index n0 +- dn, used sizes re-solved with RCWA |
| size bias | uniform size bias +-b and +-b/2 |
| random size | random size error sigma per pillar: mean shift + 1 standard deviation over the trials |
| model (V2) | the largest FDTD - LPA difference in the platform's small-lens records whose NA is at least the lens NA |

Terms are half-widths of ranges, not standard deviations. The total is their root sum of
squares (independent terms); the worst case is their sum. The small-lens records measured
focus, FWHM and relative efficiency; Strehl and absolute efficiency have no model term, and
their totals are flagged as lower bounds. If no record covers the lens NA (the LPA error grows
with NA) the model term is unknown, all totals are lower bounds, and the recipe stays at T1.
The model term has a sign: for the 633 nm TiO2 platform the FDTD focus is 2-3 % shorter and
the relative efficiency 2-3 points lower than predicted.

## What the numbers mean

* Performance and process-window numbers come from the locally periodic approximation (each
  pillar simulated in an infinite array of identical neighbours) with scalar propagation. For
  the 633 nm TiO2 recipe the small-lens FDTD comparison (V2) found the focus 2-3 % beyond the
  FDTD focus and the relative efficiency 2.3-2.7 points above it; see the R8 section of a report.
* The validation records validate the *platform* (material, wavelength, H, L, substrate), not
  the particular lens of a recipe. They exist for vertical pillars without a residual film; a
  recipe with a tapered sidewall or a residual film has no applicable record.
* The convergence certificate compares M with M+2; it estimates the truncation error, it is not
  a bound.
* Size errors in the process window use the interpolated refined library (no extra solves);
  pillars pushed outside the library range get the response of the library end, and the report
  gives their fraction.
* Reproducibility: the same specification, code commit and environment give byte-identical
  `report.md`, `recipe.json` and `lens.gds` (checked with a warm and an empty library cache and
  with 1 and 2 worker processes). `manifest.json` differs only in `created_utc`.
