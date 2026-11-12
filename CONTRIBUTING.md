# Contributing

## Set up

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Before you open a pull request

```bash
ruff check .                       # pyflakes-level lint (must be clean)
pytest tests/test_rcwa.py          # solvers, design core, GDS, materials
pytest tests/test_experiments.py   # all generators, CLI, note workflow
```

Both test files also run without pytest (`python tests/test_rcwa.py`).

## Rules that keep the physics right

1. **One convention per layer.** Solver amplitudes are `exp(+jωt)`; everything in
   `rcwa.design` and above is `exp(−iωt)`. Convert with `rcwa.to_physical` immediately after
   a solve and never mix raw solver phases into a lens layout. See `docs/conventions.md`.
2. **A physics change needs a physics test** — an analytic result, an independent solver
   (grcwa) or a conservation law, not only a regression number.
3. **Energy conservation is necessary, not sufficient.** Check convergence in the order
   count as well.
4. **Do not add material data without a source.** Tables carry their provenance; approximate
   data must say so.
5. New experiment generators go into `rcwa/experiments.py`, read parameters with
   `num(fm, "name", default)` / `fm.get("name", default)` (the CLI help is generated from
   those calls), start with a one-line docstring, and get an entry in `FAST` in
   `tests/test_experiments.py`.

## Secrets

Never commit API keys. Tidy3D keys live in `~/.tidy3d/config`, outside the repository.
