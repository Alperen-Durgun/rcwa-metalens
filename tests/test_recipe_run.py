"""Recipe pipeline (roadmap S1-S3): specification checks, trust-level rules, validation-record
matching, reproducible GDS, and one end-to-end run checked for byte-identical output."""
import copy
import datetime
import hashlib
import json
import os

import numpy as np
import pytest

from rcwa import recipe_run as rr

BASE = {"name": "t", "wavelength_um": 0.633, "material": "TiO2_Siefke",
        "lens": {"diameter_um": 4.0, "focal_um": 6.0},
        "unit_cell": {"height_um": 0.60, "period_um": 0.375}}


def _spec(**changes):
    s = copy.deepcopy(BASE)
    for k, v in changes.items():
        sec, _, key = k.partition("__")
        if key:
            s.setdefault(sec, {})[key] = v
        elif v is None:
            s.pop(sec, None)
        else:
            s[sec] = v
    return s


def test_spec_defaults_and_normalization():
    s = rr.load_spec(BASE)
    assert s["substrate"] == "SiO2_Malitson" and s["unit_cell"]["shape"] == "circle"
    assert s["fabrication"]["drc_profile"] == "recipe_search_default" and s["fabrication"]["sidewall_deg"] == 90.0
    assert s["tolerance"] == {"sigma_nm": 3.0, "bias_nm": 4.0, "index_dn": 0.02, "n_trial": 10, "seed": 0}
    assert s["numerics"] == {"M": 9, "n_sizes": 20}
    assert isinstance(s["lens"]["diameter_um"], float)
    assert BASE["lens"] == {"diameter_um": 4.0, "focal_um": 6.0}           # input not modified


@pytest.mark.parametrize("changes, words", [
    ({"wavelength_um": None}, "missing required key wavelength_um"),
    ({"lens__focal": 6.0}, "unknown key(s) lens.focal"),
    ({"colour": "red"}, "unknown key(s) colour"),
    ({"wavelength_um": "0.633"}, "expected a number"),
    ({"wavelength_um": True}, "expected a number"),
    ({"tolerance": {"n_trial": 2.5}}, "expected an integer"),
    ({"wavelength_um": -1.0}, "must be > 0"),
    ({"tolerance": {"sigma_nm": -1.0}}, "sigma_nm must be >= 0"),
    ({"fabrication": {"sidewall_deg": 30.0}}, "sidewall_deg"),
    ({"unit_cell": {"height_um": 0.6}}, "needs both height_um and period_um"),
    ({"unit_cell": None}, "no unit cell"),
    ({"search": {"height_um": [0.6], "period_um": [0.35]}}, "either unit_cell"),
    ({"unit_cell": {"shape": "star"}}, "unit_cell.shape"),
    ({"fabrication": {"residual": ["TiO2_Siefke"]}}, "residual must be"),
])
def test_spec_refuses_bad_input(changes, words):
    with pytest.raises(rr.SpecError) as e:
        rr.load_spec(_spec(**changes))
    assert words in str(e.value)


def test_spec_from_yaml(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("name: y\nwavelength_um: 0.633\nmaterial: TiO2_Siefke\nlens: {diameter_um: 4, focal_um: 6}\n"
                 "search: {height_um: [0.5, 0.6], period_um: [0.35]}\n", encoding="utf-8")
    s = rr.load_spec(str(p))
    assert s["search"]["height_um"] == [0.5, 0.6] and s["unit_cell"]["height_um"] is None


def test_platform_wavelength_range_and_substrate():
    p = rr.platform_info("TiO2_Siefke", "SiO2_Malitson", 0.633)
    assert p["sourced"] and not p["approximate"] and len(p["data_sha256"]) == 64
    assert abs(p["n_sub"] - 1.4570) < 1e-3 and p["n_bg"] == 1.0
    assert rr.platform_info("TiO2_Siefke", "air", 0.633)["n_sub"] == 1.0
    assert not rr.platform_info("TiO2_t", "air", 0.633)["sourced"]       # built-in approximate table
    with pytest.raises(rr.SpecError):
        rr.platform_info("TiO2_t", "air", 1.5)                            # table covers 0.45-0.70 um


def test_trust_level_rules():
    t1 = ["material_sourced", "library_certified", "library_refined", "robust_selection",
          "process_window", "drc_clean"]
    t2 = ["unit_cell_fullwave", "lens_fullwave", "error_budget"]
    all_ok = {k: True for k in t1 + t2}
    assert rr.trust_level(all_ok)[0] == "T2"                 # T3 needs fabrication data: never assigned
    assert "fabrication_calibration" in rr.trust_level(all_ok)[1][0]
    for k in t1:
        lvl, miss = rr.trust_level({**all_ok, k: False})
        assert lvl == "T0" and miss == [k]
    for k in t2:
        lvl, miss = rr.trust_level({**all_ok, k: False})
        assert lvl == "T1" and miss == [k]
    assert set(rr.CHECK_TEXT) == set(t1 + t2)


def test_validation_record_matching(tmp_path):
    plat = {"material": "TiO2_Siefke", "n_sub": 1.457}
    v1 = {"passed": True, "material": "TiO2_Siefke", "lam0": 0.633, "H": 0.6, "L": 0.375, "n_sub": 1.457,
          "d_amp": [0.001, -0.05], "d_phase_rad": [0.01, 0.3], "judged": [True, False], "fdtd_steps": 20}
    recs = {"V1r_a.json": v1,
            "V1r_wrong_lam.json": {**v1, "lam0": 0.532},
            "V1r_wrong_sub.json": {**v1, "n_sub": 1.0},
            "V1r_no_lam.json": {k: v for k, v in v1.items() if k != "lam0"},
            "V2_a.json": {"passed": False, "material": "TiO2_Siefke", "lam0": 0.633, "H": 0.6, "L": 0.375,
                          "n_sub": 1.457, "D": 6.0, "f": 6.0, "steps": 15,
                          "lpa": {"z_focus": 5.17, "eff_relative": 0.93}, "fdtd": {"z_focus": 5.0, "eff_relative": 0.9}},
            "notes.json": {"passed": True}}
    for name, rec in recs.items():
        (tmp_path / name).write_text(json.dumps(rec), encoding="utf-8")
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    out = rr.find_validation_records(str(tmp_path), plat, 0.633, 0.6, 0.375)
    assert [r["file"] for r in out["V1"]] == ["V1r_a.json"] and [r["file"] for r in out["V2"]] == ["V2_a.json"]
    v = out["V1"][0]
    assert v["max_d_amp"] == pytest.approx(0.001) and v["max_d_phase_rad"] == pytest.approx(0.01)   # judged only
    assert v["n_judged"] == 1 and v["n_sizes"] == 2 and len(v["sha256"]) == 64
    assert out["V2"][0]["passed"] is False and out["V2"][0]["fdtd_focus"] == 5.0
    assert rr.find_validation_records(str(tmp_path), plat, 0.633, 0.6, 0.35)["V1"] == []
    tap = rr.find_validation_records(str(tmp_path), plat, 0.633, 0.6, 0.375, sidewall_deg=88.0)
    assert tap["skipped_geometry"] and tap["V1"] == [] and tap["V2"] == []
    assert rr.find_validation_records(None, plat, 0.633, 0.6, 0.375)["V1"] == []


def test_repository_validation_records_have_the_match_keys():
    vdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "validation")
    for fn in sorted(os.listdir(vdir)):
        if fn.endswith(".json") and fn.startswith(("V1r", "V2")):
            rec = json.load(open(os.path.join(vdir, fn), encoding="utf-8"))
            assert all(k in rec for k in ("material", "lam0", "H", "L", "n_sub", "passed")), fn


def test_gds_fixed_timestamp_is_byte_identical(tmp_path):
    from rcwa.gds import layout_to_gds
    pl = [{"x": 0.0, "y": 0.0, "shape": "circle", "params": 0.1, "angle": 0.0},
          {"x": 0.35, "y": 0.0, "shape": "circle", "params": 0.2, "angle": 0.0}]
    t = datetime.datetime(2000, 1, 1)
    a, b = tmp_path / "a.gds", tmp_path / "b.gds"
    layout_to_gds(pl, str(a), timestamp=t); layout_to_gds(pl, str(b), timestamp=t)
    assert a.read_bytes() == b.read_bytes()
    c = tmp_path / "c.gds"; layout_to_gds(pl, str(c), timestamp=datetime.datetime(2001, 2, 3))
    assert c.read_bytes() != a.read_bytes() and len(c.read_bytes()) == len(a.read_bytes())


def test_infeasible_spec_writes_a_report(tmp_path):
    s = _spec(lens={"diameter_um": 20.0, "focal_um": 2.0})            # NA 0.98: L = 0.375 cannot sample it
    res = rr.run_recipe(s, str(tmp_path / "r"), validation_dir=None, log=lambda *a: None)
    assert res["status"] == "infeasible" and "cannot sample NA" in res["reason"]
    rep = (tmp_path / "r" / "report.md").read_text(encoding="utf-8")
    assert "INFEASIBLE" in rep and "cannot sample NA" in rep
    man = json.loads((tmp_path / "r" / "manifest.json").read_text(encoding="utf-8"))
    assert man["status"] == "infeasible" and man["trust_level"] is None and "report.md" in man["outputs"]


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_recipe_end_to_end_is_reproducible(tmp_path):
    """Small lens of the 633 nm reference recipe; no validation records (so T1 at most)."""
    logs = []
    a = rr.run_recipe(BASE, str(tmp_path / "a"), validation_dir=None, log=logs.append)
    assert any(m.startswith("[10/10]") for m in logs)
    assert a["status"] == "ok" and a["trust"]["level"] == "T1"
    ch = a["trust"]["checks"]
    assert all(ch[k] for k in ("material_sourced", "library_certified", "library_refined",
                               "robust_selection", "process_window", "drc_clean"))
    assert not ch["unit_cell_fullwave"] and not ch["lens_fullwave"] and not ch["error_budget"]
    assert a["library"]["certificate"]["used_sizes_certified"]
    assert a["library"]["certificate"]["used_max_d_phase_rad"] < 0.02
    lo, hi = a["library"]["size_range_um"]
    assert abs(lo - 0.06) < 1e-12 and abs(hi - 0.315) < 1e-12
    assert min(a["design"]["sizes_used_um"]) >= 0.06 - 1e-12
    fb = a["fabrication"]
    assert fb["drc"]["ok"] and fb["n_pillars_written"] == a["design"]["n_pillars"]
    if fb["gds_readback"] is not None:
        assert fb["gds_readback"]["matches"]
    pw = a["process_window"]
    assert [r["bias_nm"] for r in pw["size_bias"]] == [-4.0, -2.0, 0.0, 2.0, 4.0]
    zero = pw["size_bias"][2]                                     # unbiased = nominal, same focus scan
    assert zero["strehl"] == pytest.approx(a["performance"]["strehl"], abs=1e-12)
    assert zero["eff_absolute"] == pytest.approx(a["performance"]["eff_absolute"], abs=1e-12)
    assert [r["dn"] for r in pw["index_shift"]] == [-0.02, 0.02]
    p = a["performance"]
    assert 0.6 * 6.0 < p["z_focus_um"] < 6.2 and 0.5 < p["strehl"] <= 1.0 and 0.5 < p["eff_absolute"] < 1.0
    rep = (tmp_path / "a" / "report.md").read_text(encoding="utf-8")
    assert "Trust level: **T1**" in rep and "not computed" in rep and "No full-wave record" in rep
    man = json.loads((tmp_path / "a" / "manifest.json").read_text(encoding="utf-8"))
    assert man["schema"] == rr.SCHEMA and man["spec_sha256"] == rr._sha256_json(rr.load_spec(BASE))
    assert set(man["outputs"]) == {"lens.gds", "recipe.json", "report.md"}
    for fn, h in man["outputs"].items():
        assert _sha(tmp_path / "a" / fn) == h
    rr.run_recipe(BASE, str(tmp_path / "b"), validation_dir=None, log=lambda *x: None)
    man_b = json.loads((tmp_path / "b" / "manifest.json").read_text(encoding="utf-8"))
    assert man_b["outputs"] == man["outputs"]                     # byte-identical outputs
    assert np.isfinite(json.loads((tmp_path / "a" / "recipe.json").read_text())["performance"]["fwhm_um"])


def test_recipe_without_size_error_is_nominal_and_t0(tmp_path):
    """sigma_nm = 0 is allowed: nominal selection, no random-error run, trust level T0 because the
    robust selection requirement of T1 is not met."""
    s = _spec(tolerance={"sigma_nm": 0.0})
    r = rr.run_recipe(s, str(tmp_path / "n"), validation_dir=None, log=lambda *a: None)
    assert r["status"] == "ok" and r["design"]["selection"].startswith("nominal")
    assert r["process_window"]["random_size"] is None
    assert r["trust"]["level"] == "T0" and r["trust"]["next_level_needs"] == ["robust_selection"]
    assert "nominal (complex match" in (tmp_path / "n" / "report.md").read_text(encoding="utf-8")
