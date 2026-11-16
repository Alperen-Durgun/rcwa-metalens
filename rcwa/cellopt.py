"""Parametrized meta-cell optimization.

A unit cell is described by a shape with free parameters (e.g. a rectangle wx, wy, a cross
arm, width, a rotation angle), each with a lower and an upper bound. A cost function compares
the zeroth-order Jones response of the cell (full RCWA, rcwa.solver2d) with targets, and an
optimizer proposes the next parameter set from the costs seen so far, so that far fewer cells
are solved than in a full grid sweep.

Targets: Target(output, representation, value, weight)
  output          "txx", "tyy", "txy", "tyx" (zeroth-order transmitted Jones elements, physics
                  convention: the phase grows with the optical delay) or a ratio "a/b"
                  (e.g. "txx/tyy": its phase is the retardance)
  representation  "abs", "re", "im"   cost |repr(value_calc) - value|
                  "power"             cost |P - value|, P = |t|^2 n_exit / n_inc (power
                                      transmission into the order; for a ratio |a/b|^2)
                  "phase"             cost = wrapped |angle - value| / pi  (value in rad, cost 0..1)
The total cost is sum(weight * cost) / sum(weight).

Optimizers (bounded; all work on the unit cube internally, seed -> reproducible):
  "grid"                    full factorial sweep (reference; n points per axis)
  "bayesian"                Gaussian-process surrogate (Matern 5/2) + expected improvement
  "cmaes"                   covariance matrix adaptation evolution strategy (Hansen's (mu/mu_w, lambda))
  "differential_evolution"  DE/rand/1/bin
  "genetic"                 elitist genetic algorithm (tournament, uniform crossover, mutation)
  "particle_swarm"          global-best particle swarm with stagnation stop
  "simulated_annealing"     random-step annealing, exponential cooling
  "gradient_descent"        finite-difference gradient, step towards cost 0, backtracking
They are standard published algorithms implemented here from their textbook definitions.
Every run stops at max_evals or when the cost falls below stop; all evaluations are kept in the
history (the cost function is memoized, so a revisited point costs nothing).
"""
import numpy as np

from . import shapes
from .solver2d import solve_rcwa_2d
from .convention import to_physical

OPTIMIZERS = ("grid", "bayesian", "cmaes", "differential_evolution", "genetic", "particle_swarm",
              "simulated_annealing", "gradient_descent")
REPRESENTATIONS = ("abs", "re", "im", "power", "phase")


class _Stop(Exception):
    pass


# ----------------------------------------------------------------------------- the cell
def cell_jones(shape, params, lam0, L, H, n_pillar, n_sub=1.0, n_exit=1.0, M=9, N=None, angle=0.0):
    """Zeroth-order transmitted Jones matrix [[txx, txy], [tyx, tyy]] of one pillar (shape and
    params as rcwa.shapes.rasterize, angle in rad) on a square lattice L, height H, light from
    the substrate (n_sub) into n_exit, normal incidence. Physics convention."""
    eps = shapes.rasterize(shape, params, L, int(N or shapes.DEFAULT_N), n_pillar=n_pillar, n_bg=n_exit, angle=angle)
    lay = [(eps, H)]
    rx = solve_rcwa_2d(lam0, 0, 0, n_sub ** 2, n_exit ** 2, L, L, lay, M, M, pol=(1.0, 0.0))
    ry = solve_rcwa_2d(lam0, 0, 0, n_sub ** 2, n_exit ** 2, L, L, lay, M, M, pol=(0.0, 1.0))
    J = np.array([[rx["tx"][rx["i0"]], ry["tx"][ry["i0"]]], [rx["ty"][rx["i0"]], ry["ty"][ry["i0"]]]], complex)
    return to_physical(J)


def jones_outputs(J):
    return {"txx": complex(J[0, 0]), "txy": complex(J[0, 1]), "tyx": complex(J[1, 0]), "tyy": complex(J[1, 1])}


# ----------------------------------------------------------------------------- targets / cost
class Target:
    def __init__(self, output, representation, value, weight=1.0):
        if representation not in REPRESENTATIONS:
            raise ValueError(f"representation must be one of {REPRESENTATIONS}")
        self.output, self.representation, self.value, self.weight = output, representation, float(value), float(weight)

    def __repr__(self):
        return f"Target({self.output!r}, {self.representation!r}, {self.value:g}, w={self.weight:g})"


def _output(outs, name):
    if "/" in name:
        a, b = name.split("/", 1)
        return outs[a.strip()] / outs[b.strip()], True
    return outs[name], False


def target_cost(outs, t, power_factor=1.0):
    v, ratio = _output(outs, t.output)
    r = t.representation
    if r == "abs":
        return abs(abs(v) - t.value)
    if r == "re":
        return abs(v.real - t.value)
    if r == "im":
        return abs(v.imag - t.value)
    if r == "power":
        return abs(abs(v) ** 2 * (1.0 if ratio else power_factor) - t.value)
    d = np.angle(v) - t.value                                   # phase
    return abs(np.angle(np.exp(1j * d))) / np.pi


def total_cost(outs, targets, power_factor=1.0):
    w = sum(t.weight for t in targets)
    return float(sum(t.weight * target_cost(outs, t, power_factor) for t in targets) / w)


# ----------------------------------------------------------------------------- evaluator
class _Eval:
    """Maps unit-cube points to the user's bounds, memoizes, counts, records, stops."""

    def __init__(self, f, lo, hi, max_evals, stop):
        self.f, self.lo, self.hi = f, np.asarray(lo, float), np.asarray(hi, float)
        self.max_evals, self.stop = int(max_evals), float(stop)
        self.memo = {}; self.hist = []; self.best = (np.inf, None)

    def x(self, u):
        return self.lo + np.clip(u, 0.0, 1.0) * (self.hi - self.lo)

    def __call__(self, u):
        u = np.clip(np.asarray(u, float), 0.0, 1.0)
        key = tuple(np.round(u, 12))
        if key in self.memo:
            return self.memo[key]
        if len(self.hist) >= self.max_evals:
            raise _Stop
        c = float(self.f(self.x(u)))
        if not np.isfinite(c):
            c = 1e9
        self.memo[key] = c
        self.hist.append((self.x(u).tolist(), c))
        if c < self.best[0]:
            self.best = (c, u.copy())
        if c <= self.stop:
            raise _Stop
        return c


# ----------------------------------------------------------------------------- optimizers
def _grid(E, d, rng, n=None, **_):
    n = int(n or max(2, round(E.max_evals ** (1.0 / d))))
    axes = [np.linspace(0, 1, n)] * d
    for p in np.array(np.meshgrid(*axes, indexing="ij")).reshape(d, -1).T:
        E(p)


def _bayesian(E, d, rng, initial_points=None, n_candidates=2000, xi=0.01, **_):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    from scipy.stats import norm
    n0 = int(initial_points or max(5, 4 * d))
    X = list(_lhs(n0, d, rng)); Y = [E(x) for x in X]
    kern = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.full(d, 0.3), length_scale_bounds=(1e-3, 10.0), nu=2.5) \
        + WhiteKernel(1e-6, (1e-9, 1e-2))
    while True:
        gp = GaussianProcessRegressor(kern, normalize_y=True, random_state=int(rng.integers(1 << 31)), n_restarts_optimizer=0)
        gp.fit(np.array(X), np.array(Y))
        xb = np.array(X)[int(np.argmin(Y))]
        cand = np.vstack([rng.random((n_candidates, d)),
                          np.clip(xb + 0.05 * rng.standard_normal((n_candidates // 2, d)), 0, 1)])
        mu, sd = gp.predict(cand, return_std=True)
        imp = min(Y) - mu - xi
        z = imp / np.maximum(sd, 1e-12)
        ei = imp * norm.cdf(z) + sd * norm.pdf(z)
        x = cand[int(np.argmax(ei))]
        X.append(x); Y.append(E(x))


def _cmaes(E, d, rng, sigma=0.3, x0=None, **_):
    lam = 4 + int(3 * np.log(d)); mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)); w /= w.sum(); mueff = 1 / np.sum(w ** 2)
    cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d); cs = (mueff + 2) / (d + mueff + 5)
    c1 = 2 / ((d + 1.3) ** 2 + mueff); cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + cs
    chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d * d))
    m = np.full(d, 0.5) if x0 is None else np.asarray(x0, float)
    s = float(sigma); C = np.eye(d); pc = np.zeros(d); ps = np.zeros(d); g = 0
    while True:
        g += 1
        Dg, B = np.linalg.eigh(C); Dg = np.sqrt(np.maximum(Dg, 1e-20))
        z = rng.standard_normal((lam, d)); y = z @ (B * Dg).T; xs = m + s * y
        f = np.array([E(np.clip(x, 0, 1)) + 1e3 * np.sum((x - np.clip(x, 0, 1)) ** 2) for x in xs])
        idx = np.argsort(f)[:mu]
        yw = w @ y[idx]; m = m + s * yw
        Cinvsqrt = B @ np.diag(1 / Dg) @ B.T
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * Cinvsqrt @ yw
        hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * g)) / chiN < 1.4 + 2 / (d + 1)
        pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mueff) * yw
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hs) * cc * (2 - cc) * C) \
            + cmu * (y[idx].T * w) @ y[idx]
        s *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        if s < 1e-9:
            raise _Stop


def _de(E, d, rng, population_size=None, mutation=0.7, crossover=0.9, **_):
    n = int(population_size or max(5 * d, 6))
    P = _lhs(n, d, rng); F = np.array([E(x) for x in P])
    while True:
        for i in range(n):
            a, b, c = P[rng.choice([j for j in range(n) if j != i], 3, replace=False)]
            v = np.clip(a + mutation * (b - c), 0, 1)
            mask = rng.random(d) < crossover; mask[rng.integers(d)] = True
            u = np.where(mask, v, P[i]); fu = E(u)
            if fu <= F[i]:
                P[i], F[i] = u, fu


def _genetic(E, d, rng, population_size=None, n_parents=None, mutation_rate=0.2, sigma=0.1, **_):
    n = int(population_size or max(6 * d, 8)); k = int(n_parents or max(2 * d, 2))
    if n <= k:
        raise ValueError("population_size must be larger than n_parents")
    P = _lhs(n, d, rng); F = np.array([E(x) for x in P])
    while True:
        o = np.argsort(F); par, fpar = P[o[:k]], F[o[:k]]
        kids = []
        while len(kids) < n - k:
            i, j = [min(rng.choice(k, 2, replace=False), key=lambda q: fpar[q]) for _ in range(2)]
            child = np.where(rng.random(d) < 0.5, par[i], par[j])
            mut = rng.random(d) < mutation_rate
            child = np.clip(child + mut * sigma * rng.standard_normal(d), 0, 1)
            kids.append(child)
        kids = np.array(kids)
        P = np.vstack([par, kids]); F = np.concatenate([fpar, [E(x) for x in kids]])


def _pso(E, d, rng, population_size=None, omega=0.5, phip=0.5, phig=0.5, stagnation=1e-8, patience=8, **_):
    n = int(population_size or max(5 * d, 6))
    X = _lhs(n, d, rng); V = 0.1 * (rng.random((n, d)) - 0.5)
    F = np.array([E(x) for x in X]); Pb, Fb = X.copy(), F.copy()
    g = Pb[np.argmin(Fb)].copy(); fg = Fb.min(); still = 0
    while True:
        rp, rg = rng.random((n, d)), rng.random((n, d))
        V = omega * V + phip * rp * (Pb - X) + phig * rg * (g - X)
        X = np.clip(X + V, 0, 1)
        F = np.array([E(x) for x in X])
        better = F < Fb; Pb[better], Fb[better] = X[better], F[better]
        if Fb.min() < fg - stagnation:
            g, fg, still = Pb[np.argmin(Fb)].copy(), Fb.min(), 0
        else:
            still += 1
            if still >= patience:
                raise _Stop


def _annealing(E, d, rng, step=0.1, t_start=1.0, t_end=1e-3, **_):
    x = rng.random(d); fx = E(x)
    n = max(E.max_evals - 1, 1); k = 0
    while True:
        T = t_start * (t_end / t_start) ** min(k / n, 1.0); k += 1
        y = np.clip(x + step * rng.standard_normal(d), 0, 1); fy = E(y)
        if fy < fx or rng.random() < np.exp(-(fy - fx) / max(T, 1e-12)):
            x, fx = y, fy


def _gradient(E, d, rng, h=0.01, multiplier=0.5, x0=None, **_):
    x = np.full(d, 0.5) if x0 is None else np.asarray(x0, float); fx = E(x)
    while True:
        g = np.array([(E(np.clip(x + h * e, 0, 1)) - fx) / h for e in np.eye(d)])
        gg = float(g @ g)
        if gg < 1e-16:
            raise _Stop
        step = multiplier * fx / gg                  # Newton step towards cost 0, damped
        for _ in range(12):
            y = np.clip(x - step * g, 0, 1); fy = E(y)
            if fy < fx:
                x, fx = y, fy; break
            step /= 2
        else:
            raise _Stop


_ALG = {"grid": _grid, "bayesian": _bayesian, "cmaes": _cmaes, "differential_evolution": _de,
        "genetic": _genetic, "particle_swarm": _pso, "simulated_annealing": _annealing,
        "gradient_descent": _gradient}


def _lhs(n, d, rng):
    """Latin hypercube sample of n points in [0, 1]^d."""
    u = (np.argsort(rng.random((n, d)), axis=0) + rng.random((n, d))) / n
    return u


# ----------------------------------------------------------------------------- public API
def minimize(f, bounds, optimizer="bayesian", max_evals=100, stop=0.0, seed=0, **options):
    """Minimize f(x) over the box bounds [(lo, hi), ...]. Returns dict(x, cost, n_evals,
    history [(x, cost), ...], optimizer, options)."""
    if optimizer not in _ALG:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
    lo, hi = np.array([b[0] for b in bounds], float), np.array([b[1] for b in bounds], float)
    if np.any(hi <= lo):
        raise ValueError("every bound needs lo < hi")
    E = _Eval(f, lo, hi, max_evals, stop)
    rng = np.random.default_rng(seed)
    try:
        _ALG[optimizer](E, len(bounds), rng, **options)
    except _Stop:
        pass
    c, u = E.best
    return {"x": None if u is None else E.x(u).tolist(), "cost": float(c), "n_evals": len(E.hist),
            "history": E.hist, "optimizer": optimizer, "options": options, "seed": int(seed)}


def optimize_cell(shape, variables, targets, lam0, L, H, n_pillar, n_sub=1.0, n_exit=1.0, M=9, N=None,
                  fixed=None, optimizer="bayesian", max_evals=60, stop=1e-3, seed=0, **options):
    """Optimize a meta-cell. variables: {name: (lo, hi)} with names from the shape parameters
    ("p0", "p1", ... in rcwa.shapes.rasterize order), "angle" (rad), "H" (height um);
    fixed: values for the parameters not optimized (same names). Returns the minimize() result
    plus the best parameters by name and the Jones outputs there."""
    names = list(variables)
    fixed = dict(fixed or {})
    nshape = {"square": 1, "circle": 1, "hexagon": 1, "rect": 2, "ellipse": 2, "cross": 2}[shape]
    pf = n_exit / n_sub

    def build(x):
        v = {**fixed, **dict(zip(names, x))}
        params = tuple(float(v[f"p{i}"]) for i in range(nshape))
        return params, float(v.get("angle", 0.0)), float(v.get("H", H))

    def f(x):
        params, ang, h = build(x)
        if min(params) <= 0:
            return 1e9
        return total_cost(jones_outputs(cell_jones(shape, params, lam0, L, h, n_pillar, n_sub, n_exit, M, N, ang)),
                          targets, pf)

    res = minimize(f, [variables[k] for k in names], optimizer, max_evals, stop, seed, **options)
    params, ang, h = build(res["x"])
    outs = jones_outputs(cell_jones(shape, params, lam0, L, h, n_pillar, n_sub, n_exit, M, N, ang))
    res.update({"variables": names, "best": dict(zip(names, res["x"])), "outputs": outs,
                "power_factor": pf, "targets": [repr(t) for t in targets],
                "cell": {"shape": shape, "params": list(params), "angle": ang, "H": h, "L": L, "lam0": lam0,
                         "n_pillar": [complex(n_pillar).real, complex(n_pillar).imag], "n_sub": n_sub,
                         "n_exit": n_exit, "M": M}})
    return res
