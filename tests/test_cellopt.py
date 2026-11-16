"""Parametrized meta-cell optimization (rcwa/cellopt.py)."""
import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import cellopt as co, design

OPT = [o for o in co.OPTIMIZERS if o != "grid"]


def _sphere(x):
    return float((x[0] - 0.3) ** 2 + (x[1] + 0.4) ** 2)


@pytest.mark.parametrize("opt", OPT)
def test_optimizers_find_the_minimum_of_a_bowl(opt):
    r = co.minimize(_sphere, [(-1, 1), (-1, 1)], opt, max_evals=200, stop=1e-3, seed=1)
    assert r["cost"] <= 1e-3, (opt, r["cost"], r["x"])
    assert r["n_evals"] <= 200 and len(r["history"]) == r["n_evals"]
    assert all(-1 <= v <= 1 for x, _ in r["history"] for v in x)          # bounds respected


@pytest.mark.parametrize("opt", ["cmaes", "differential_evolution"])
def test_rosenbrock_valley(opt):
    f = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2          # noqa: E731
    r = co.minimize(f, [(-2, 2), (-1, 3)], opt, max_evals=600, stop=1e-3, seed=0)
    assert r["cost"] < 0.05, (opt, r["cost"], r["x"])


def test_reproducible_and_stop_condition():
    a = co.minimize(_sphere, [(-1, 1), (-1, 1)], "differential_evolution", 80, seed=3)
    b = co.minimize(_sphere, [(-1, 1), (-1, 1)], "differential_evolution", 80, seed=3)
    assert a["history"] == b["history"]
    c = co.minimize(_sphere, [(-1, 1), (-1, 1)], "differential_evolution", 500, stop=1e-2, seed=3)
    assert c["cost"] <= 1e-2 and c["n_evals"] < 500


def test_grid_is_a_full_factorial_sweep():
    r = co.minimize(_sphere, [(-1, 1), (-1, 1)], "grid", max_evals=25, n=5)
    assert r["n_evals"] == 25 and sorted({round(x[0], 9) for x, _ in r["history"]}) == [-1, -0.5, 0, 0.5, 1]


def test_cost_definitions():
    o = {"txx": 0.8 * np.exp(1j * 3.0), "tyy": 0.9 + 0j}
    assert co.target_cost(o, co.Target("txx", "abs", 1.0)) == pytest.approx(0.2)
    assert co.target_cost(o, co.Target("txx", "phase", -3.0)) == pytest.approx((2 * np.pi - 6.0) / np.pi)  # wrapped
    assert co.target_cost(o, co.Target("txx/tyy", "phase", 3.0)) == pytest.approx(0.0, abs=1e-12)
    assert co.target_cost(o, co.Target("tyy", "power", 0.81), power_factor=1 / 1.5) == pytest.approx(abs(0.81 / 1.5 - 0.81))
    with pytest.raises(ValueError):
        co.Target("txx", "amplitude", 1)


def test_cell_jones_matches_the_library_and_symmetry():
    """Circle: tyy equals the library's t (same solver, same convention), txx = tyy, no cross terms;
    a rectangle rotated by 90 degrees swaps txx and tyy."""
    kw = dict(lam0=0.633, L=0.375, H=0.6, n_pillar=2.3893, n_sub=1.457, n_exit=1.0, M=5, N=256)
    J = co.cell_jones("circle", (0.2,), **kw)
    t = design.meta_atom_library(0.633, 0.375, 0.6, sizes=[0.2], n_pillar=2.3893, M=5, n_sub=1.457,
                                 raster_n=256, use_cache=False, n_jobs=1)["t"][0]
    assert abs(J[1, 1] - t) < 1e-10 and abs(J[0, 0] - J[1, 1]) < 1e-3 and abs(J[0, 1]) < 1e-6
    A = co.cell_jones("rect", (0.25, 0.1), **kw)
    B = co.cell_jones("rect", (0.25, 0.1), angle=np.pi / 2, **kw)
    assert abs(A[0, 0] - B[1, 1]) < 1e-6 and abs(A[1, 1] - B[0, 0]) < 1e-6


def test_optimize_circle_to_a_target_phase():
    """One-parameter check against a direct sweep: find the diameter whose phase relative to the
    smallest pillar is 2 rad."""
    kw = dict(lam0=0.633, L=0.375, H=0.6, n_pillar=2.3893, n_sub=1.457, n_exit=1.0, M=5, N=256)
    t0 = co.cell_jones("circle", (0.08,), **kw)[1, 1]
    target = np.angle(t0) + 2.0
    r = co.optimize_cell("circle", {"p0": (0.08, 0.30)}, [co.Target("tyy", "phase", target)],
                         optimizer="bayesian", max_evals=25, stop=2e-3, seed=0, **kw)
    d = np.linspace(0.08, 0.30, 89)
    ph = np.array([np.angle(co.cell_jones("circle", (s,), **kw)[1, 1] / t0) for s in d])
    best_sweep = d[np.argmin(np.abs(np.angle(np.exp(1j * (ph - 2.0)))))]
    assert r["cost"] < 2e-3 and r["n_evals"] <= 25
    assert abs(r["best"]["p0"] - best_sweep) < 0.006                          # within the sweep step
