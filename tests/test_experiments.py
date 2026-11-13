"""End-to-end smoke tests of the experiment generators, the CLI and the note workflow.
Run:  python tests/test_experiments.py     (pytest works too)

Everything is written to a temporary project folder; nothing touches the repository.
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TMP = tempfile.mkdtemp(prefix="rcwa_test_")
os.environ["RCWA_PROJECT_DIR"] = _TMP              # must be set before rcwa.experiments is imported
import matplotlib
matplotlib.use("Agg")
from rcwa import experiments as R, cli

# small/fast parameter sets — every generator must run end to end with these
FAST = {
    "grating": {}, "metalens_library": {"n_width": 6}, "angle_sweep": {"n_angle": 5},
    "grating2d": {"M": 4}, "metalens_pillar_2d": {"n_side": 4, "M": 4},
    "lens_fullwave_1d": {"D_lens": 3.5, "f": 5.0, "M": 60},
    "metalens_layout_2d": {"D_lens": 4.0, "f": 6.0, "M": 4},
    "wavelength_sweep": {"n_lam": 3, "M": 4}, "birefringent_library": {"n_w": 2, "M": 4},
    "jones_matrix": {"M": 4}, "pb_phase_library": {"n_angle": 3, "M": 4},
    "param_map_2d": {"n_h": 2, "n_s": 2, "M": 4},
    "inverse_design_deflector": {"n_seg": 6, "maxiter": 2},
    "achromatic_analysis": {"D_lens": 4.0, "n_lam": 3},
    "vortex_beam": {}, "hologram": {"iters": 5}, "polarization_multiplexed": {},
    "metalens_design_full": {"D_lens": 4.0, "f": 6.0, "M": 4},
    "tolerance_analysis": {"n_trial": 4, "n_seg": 6},
    "recipe_search": {"n_H": 2, "n_L": 1, "H_min": 0.5, "H_max": 0.6, "L_min": 0.35, "L_max": 0.35,
                      "M": 3, "n_sizes": 6},
    "lens_tolerance": {"D_lens": 4.0, "f": 6.0, "M": 3, "n_bias": 3, "n_trial": 2,
                       "refine_max_solves": 10},
    "fov_analysis": {"D_lens": 4.0, "f": 6.0, "n_angle": 3, "angle_max": 10, "M": 4},
    "index_robustness": {"D_lens": 4.0, "f": 6.0, "M": 4, "n_dn": 3, "dn_max": 0.1},
    "metalens_large_tiled": {"D_lens": 15.0, "f": 20.0, "M": 4, "tile_cells": 30},
    "achromatic_design": {"D_lens": 4.0, "f": 8.0, "band": 0.06, "n_size": 3, "n_h": 2, "M": 4},
}


def test_fast_set_covers_every_generator():
    assert set(FAST) == set(R.RUNNERS)


def _run(name):
    fig, body, summary = R.RUNNERS[name](dict(FAST[name]), "T_" + name)
    assert os.path.isfile(fig) and fig.startswith(_TMP), fig
    assert isinstance(body, str) and body.strip() and isinstance(summary, str) and summary.strip()
    assert "nan" not in summary.lower(), summary
    return body, summary


def _make(name):
    def test():
        _run(name)
    test.__name__ = "test_generator_" + name
    return test


for _n in FAST:
    globals()["test_generator_" + _n] = _make(_n)


def test_design_full_outputs():
    """The flagship generator writes a readable GDS + design JSON and reports DRC/material notes."""
    import json
    from rcwa import gds
    body, _ = _run("metalens_design_full")
    g = os.path.join(R.SIM, "T_metalens_design_full.gds")
    d = json.load(open(os.path.join(R.SIM, "T_metalens_design_full_design.json")))
    s = gds.read_gds_summary(g)
    assert s["polygons"] == len(d["placements"]) > 50 and s["references"] == s["polygons"]
    assert "DRC" in body and "Material data" in body and "Strehl" in body


def test_large_tiled_writes_tiles():
    from rcwa import gds
    body, _ = _run("metalens_large_tiled")
    tiles = os.listdir(os.path.join(R.SIM, "T_metalens_large_tiled_gds"))
    n = sum(gds.read_gds_summary(os.path.join(R.SIM, "T_metalens_large_tiled_gds", t))["polygons"] for t in tiles)
    assert len(tiles) >= 2 and n == 1457


def test_cli_runs_and_reports():
    assert cli.main(["list"]) == 0 and cli.main(["help", "grating"]) == 0
    assert cli.main(["grating", "period=1.3"]) == 0
    assert os.path.isfile(os.path.join(R.SIM, "CLI_grating.png"))
    assert cli.main(["grating", "perod=1.3"]) == 2          # typo is rejected, not ignored


def test_console_output_survives_cp1252():
    """Windows regression: with a cp1252 (non-UTF-8) stdout — the default when output is
    redirected — the command line must still print its Greek letters and check marks."""
    import subprocess
    env = dict(os.environ, PYTHONIOENCODING="cp1252", PYTHONUTF8="0")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for args in (["list"], ["grating", "period=1.3"], ["help", "fov_analysis"]):
        p = subprocess.run([sys.executable, os.path.join(root, "metadesign.py")] + args,
                           env=env, cwd=root, capture_output=True)
        assert p.returncode == 0, p.stderr.decode("utf-8", "replace")[-400:]
    assert "polarization_multiplexed" in subprocess.run(
        [sys.executable, os.path.join(root, "metadesign.py"), "list"], env=env, cwd=root,
        capture_output=True).stdout.decode("utf-8")


def test_note_workflow_roundtrip():
    """An experiment note with status 'bekliyor' is run, a report is written, the note is marked done."""
    os.makedirs(R.DENEYLER, exist_ok=True)
    note = os.path.join(R.DENEYLER, "demo grating.md")
    with open(note, "w", encoding="utf-8") as f:
        f.write("---\ndeney_tipi: grating\ndurum: 🔬 bekliyor\nperiod: 1.1\n---\n# demo\n")
    R.main(["run_experiment", "--all"])
    txt = open(note, encoding="utf-8").read()
    assert "tamamlandi" in txt and "son_rapor:" in txt
    reports = [r for r in os.listdir(R.RAPORLAR) if "demo grating" in r]
    assert len(reports) == 1
    rep = open(os.path.join(R.RAPORLAR, reports[0]), encoding="utf-8").read()
    assert "ΣDE = 1.0000" in rep and "| period | 1.1 |" in rep and "deney_tipi" not in rep.split("---")[2]


def test_note_workflow_english_keys():
    """English front matter (experiment_type / status: pending) works the same way; a note
    without a type or with a mistyped one is skipped with a message, not run as something else."""
    os.makedirs(R.DENEYLER, exist_ok=True)
    note = os.path.join(R.DENEYLER, "english demo.md")
    with open(note, "w", encoding="utf-8", newline="") as f:
        f.write("---\r\nexperiment_type: grating\r\nstatus: pending\r\nduty: 0.4\r\n---\r\n# demo\r\n")
    bad = os.path.join(R.DENEYLER, "typo.md")
    with open(bad, "w", encoding="utf-8") as f:
        f.write("---\nexperiment_type: gratin\nstatus: pending\n---\n")
    R.main(["run_experiment", "--all"])
    txt = open(note, encoding="utf-8").read()
    assert "status: 🟢 done" in txt and "last_report:" in txt and "durum" not in txt
    reports = [r for r in os.listdir(R.RAPORLAR) if r.startswith("Report - english demo")]
    assert len(reports) == 1 and "kind: report" in open(os.path.join(R.RAPORLAR, reports[0]), encoding="utf-8").read()
    assert "pending" in open(bad, encoding="utf-8").read()            # untouched
    assert not [r for r in os.listdir(R.RAPORLAR) if "typo" in r]


if __name__ == "__main__":
    from rcwa.cli import utf8_stdio
    utf8_stdio()
    import time
    _tests = [(k, v) for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for _name, _fn in _tests:
        _t = time.time(); _fn(); print(f"{_name}: PASSED  [{time.time() - _t:.1f}s]")
    print(f"All {len(_tests)} tests passed ✅")
