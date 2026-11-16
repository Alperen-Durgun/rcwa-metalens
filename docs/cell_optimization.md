# Parametrized meta-cell optimization

`rcwa/cellopt.py` searches the free parameters of a unit cell (rectangle `p0, p1`, ellipse,
cross, rotation `angle`, height `H`) for a target zeroth-order Jones response, solving each
proposed cell with the full 2D RCWA. It is meant for shapes with two or more parameters,
where a full grid sweep is expensive.

```python
from rcwa import cellopt as co
T = [co.Target("txx/tyy", "phase", 3.14159, weight=2),   # retardance pi (half-wave plate)
     co.Target("txx", "power", 1.0), co.Target("tyy", "power", 1.0)]
r = co.optimize_cell("rect", {"p0": (0.06, 0.315), "p1": (0.06, 0.315)}, T,
                     lam0=0.633, L=0.375, H=0.6, n_pillar=2.3893, n_sub=1.457,
                     optimizer="bayesian", max_evals=60, stop=1e-3, seed=0)
r["best"], r["cost"], r["outputs"]
```

Note workflow: experiment type `cell_optimize` (`shape`, `vars: p0=0.06:0.315, p1=0.06:0.315`,
`targets: txx/tyy phase 180 2; txx power 1; tyy power 1` with phases in degrees, `fix_<name>`
for fixed parameters, `optimizer`, `max_evals`, `stop`, `seed`, `M`, `N`, plus the usual
`lam0, period, height, material, substrate`).

## Cost

Per target: `abs`, `re`, `im` - |representation - value|; `power` - |P - value| with
P = |t|^2 n_exit / n_inc; `phase` - wrapped |angle - value| / pi (0..1). Outputs are `txx`,
`tyy`, `txy`, `tyx` or a ratio `a/b`. The total is the weighted mean.

## Optimizers

`grid` (reference), `bayesian` (Gaussian process, Matern 5/2, expected improvement),
`cmaes`, `differential_evolution` (rand/1/bin), `genetic`, `particle_swarm`,
`simulated_annealing`, `gradient_descent` (finite differences, step towards cost 0). All are
bounded, reproducible with `seed`, stop at `max_evals` or when the cost reaches `stop`, and
keep every evaluation in `history`. The algorithms are the standard published ones,
implemented here from their definitions; they are tested on analytic functions (bowl,
Rosenbrock) and on a one-parameter cell against a direct sweep.

## Benchmark: half-wave plate cell, 633 nm TiO2 600/375

Rectangle on fused silica, both sides 60-315 nm (min gap 60 nm), targets retardance pi
(weight 2) and power 1 in both polarizations, M = 7. Evaluations needed to reach the best cost
of a 21 x 21 grid (441 cells, cost 0.0989, 78 s):

| Optimizer | Cells to reach the grid optimum | Best cost (120 cells max) |
|---|---|---|
| bayesian | 9 | 0.0989 |
| genetic | 17 | 0.0980 |
| particle_swarm | 53 | 0.0980 |
| cmaes | 67 | 0.0989 |
| simulated_annealing | 78 | 0.0986 |
| differential_evolution | not within 120 (mirror solution, 0.114) | 0.114 |
| gradient_descent | not reached (started on the symmetric diagonal, stopped at 0.504) | 0.504 |

All the successful runs find the same cell (about 112 x 315 nm or its mirror): the optimum
lies on the 315 nm bound, and the retardance there is about 214 deg rather than 180 deg. At
H = 600 nm with a 60 nm minimum gap a full half-wave plate is not reachable; it needs a taller
pillar or a smaller gap. Use `gradient_descent` only from a non-symmetric start (`x0`) or
after a global optimizer.
