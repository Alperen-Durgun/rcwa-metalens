"""One-command recipe (roadmap S1-S3): specification -> recipe folder with a trust level.

    rcwa-metalens recipe spec.yaml [--out DIR] [--jobs N]

The pipeline follows the 10 steps of the plan (docs/recipe_pipeline.md):

  1 specification      validated YAML / dict (unknown keys are refused, nothing is guessed)
  2 platform           sourced material data, substrate, data file hash          (R1)
  3 unit cell          fixed (H, L) or a search, fabrication + sampling limits    (R2)
  4 library            refined, resonance map, convergence certificate            (R3)
  5 lens design        robust selection from certified sizes only                 (R4)
  6 performance        focal analysis (error budget: roadmap T2)                  (R5)
  7 process window     size bias, random size error, film index shift            (R6)
  8 validation         full-wave records in validation/ that match the platform   (R8)
  9 fabrication        GDS (reproducible), DRC against the named profile          (R7)
 10 report             trust level T0-T3 with the checklist behind it             (R9)
    + manifest.json    code version, inputs, data and output hashes              (R10)

If step 3 finds no feasible unit cell the pipeline stops and the report says why.
All performance numbers come from the locally periodic approximation (LPA) unless a
validation record says otherwise; the report states which.
"""
import copy
import datetime
import hashlib
import json
import os
import platform as _platform

import numpy as np

from . import design as dz
from . import materials, recipe
from .exporting import drc_check, load_drc_profile

SCHEMA = "rcwa-metalens-recipe/1"
GDS_DATE = datetime.datetime(2000, 1, 1)        # fixed GDS date: same input -> same bytes
NZ = 81                                          # focus scan points, the same for every metric

# key: (type, default, required). Nested dicts are sections; defaults of None mean "not set".
SPEC_FIELDS = {
    "name": (str, None, True),
    "wavelength_um": (float, None, True),
    "material": (str, None, True),
    "substrate": (str, "SiO2_Malitson", False),
    "lens": {"diameter_um": (float, None, True), "focal_um": (float, None, True)},
    "unit_cell": {"shape": (str, "circle", False), "height_um": (float, None, False),
                  "period_um": (float, None, False)},
    "search": {"height_um": (list, None, False), "period_um": (list, None, False)},
    "fabrication": {"drc_profile": (str, "recipe_search_default", False),
                    "sidewall_deg": (float, 90.0, False), "residual": (list, None, False)},
    "tolerance": {"sigma_nm": (float, 3.0, False), "bias_nm": (float, 4.0, False),
                  "index_dn": (float, 0.02, False), "n_trial": (int, 10, False), "seed": (int, 0, False)},
    "numerics": {"M": (int, 9, False), "n_sizes": (int, 20, False)},
}
SHAPES = ("circle",)                             # roadmap U3 adds more


class SpecError(ValueError):
    pass


def _coerce(key, typ, val):
    if typ is float:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise SpecError(f"{key}: expected a number, got {val!r}")
        return float(val)
    if typ is int:
        if isinstance(val, bool) or not isinstance(val, int):
            raise SpecError(f"{key}: expected an integer, got {val!r}")
        return int(val)
    if typ is str:
        if not isinstance(val, str) or not val.strip():
            raise SpecError(f"{key}: expected a non-empty string, got {val!r}")
        return val.strip()
    if typ is list:
        if not isinstance(val, (list, tuple)) or not val:
            raise SpecError(f"{key}: expected a non-empty list, got {val!r}")
        return list(val)
    raise SpecError(f"{key}: unsupported type")


def _fill(fields, given, prefix=""):
    if not isinstance(given, dict):
        raise SpecError(f"{prefix or 'spec'}: expected a mapping")
    unknown = sorted(set(given) - set(fields))
    if unknown:
        raise SpecError(f"unknown key(s) {', '.join(prefix + k for k in unknown)}")
    out = {}
    for k, f in fields.items():
        if isinstance(f, dict):
            out[k] = _fill(f, given.get(k) or {}, prefix + k + ".")
            continue
        typ, default, required = f
        if given.get(k) is None:
            if required:
                raise SpecError(f"missing required key {prefix + k}")
            out[k] = copy.deepcopy(default)
        else:
            out[k] = _coerce(prefix + k, typ, given[k])
    return out


def load_spec(spec):
    """A recipe specification from a YAML/JSON path or a dict -> normalized dict.
    Every key is checked; unknown keys and out-of-range values raise SpecError."""
    if isinstance(spec, (str, os.PathLike)):
        import yaml
        with open(spec, encoding="utf-8") as f:
            spec = yaml.safe_load(f) or {}
    s = _fill(SPEC_FIELDS, spec)
    pos = [("wavelength_um", s["wavelength_um"]), ("lens.diameter_um", s["lens"]["diameter_um"]),
           ("lens.focal_um", s["lens"]["focal_um"]), ("tolerance.index_dn", s["tolerance"]["index_dn"]),
           ("tolerance.bias_nm", s["tolerance"]["bias_nm"]), ("tolerance.n_trial", s["tolerance"]["n_trial"]),
           ("numerics.M", s["numerics"]["M"]), ("numerics.n_sizes", s["numerics"]["n_sizes"])]
    for key, v in pos:
        if not v > 0:
            raise SpecError(f"{key} must be > 0 (got {v})")
    if s["tolerance"]["sigma_nm"] < 0:
        raise SpecError("tolerance.sigma_nm must be >= 0")
    if not 45.0 < s["fabrication"]["sidewall_deg"] <= 90.0:
        raise SpecError("fabrication.sidewall_deg must be in (45, 90]")
    if s["unit_cell"]["shape"] not in SHAPES:
        raise SpecError(f"unit_cell.shape must be one of {SHAPES}")
    uc, se = s["unit_cell"], s["search"]
    fixed = uc["height_um"] is not None or uc["period_um"] is not None
    if fixed and (uc["height_um"] is None or uc["period_um"] is None):
        raise SpecError("unit_cell needs both height_um and period_um (or neither, with a search)")
    if fixed and (se["height_um"] or se["period_um"]):
        raise SpecError("give either unit_cell.height_um/period_um or search.*, not both")
    if not fixed and not (se["height_um"] and se["period_um"]):
        raise SpecError("no unit cell: give unit_cell.height_um and period_um, or search.height_um and search.period_um")
    for key in ("height_um", "period_um"):
        if se[key]:
            se[key] = [_coerce(f"search.{key}[]", float, v) for v in se[key]]
            if min(se[key]) <= 0:
                raise SpecError(f"search.{key} values must be > 0")
    res = s["fabrication"]["residual"]
    if res is not None:
        if len(res) != 2 or not isinstance(res[0], str) or isinstance(res[1], bool) \
                or not isinstance(res[1], (int, float)) or res[1] < 0:
            raise SpecError("fabrication.residual must be [material name, thickness_um >= 0]")
        s["fabrication"]["residual"] = [res[0], float(res[1])]
    return s


# --------------------------------------------------------------------------- helpers
def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (complex, np.complexfloating)):
        return {"re": float(np.real(x)), "im": float(np.imag(x))}
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    return x


def _git_state(root):
    """(commit, dirty) of the checkout that holds the package, or (None, None)."""
    import subprocess
    try:
        c = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if c.returncode != 0:
            return None, None
        d = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, timeout=10)
        return c.stdout.strip(), bool(d.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


def platform_info(material, substrate, lam0):
    """Step 2 (R1): material provenance and indices. Raises SpecError for a wavelength outside
    the data range. 'sourced' is True for shipped datasets and user-registered measured data."""
    inf = materials.info(material)
    rng = inf.get("range_um")
    if rng and not (rng[0] <= lam0 <= rng[1]):
        raise SpecError(f"wavelength {lam0} um is outside the data range {rng} of {material}")
    n = dz.pillar_index(material, lam0)
    sourced = material in getattr(materials, "_SOURCES", {})
    data_file = None; data_sha = None
    if inf.get("source_file"):
        p = os.path.join(os.path.dirname(materials.__file__), "materials_data", "rii", inf["source_file"])
        if os.path.isfile(p):
            data_file = inf["source_file"]; data_sha = _sha256_file(p)
    if substrate in ("air", "none"):
        n_sub = 1.0; sub_inf = {"name": "air"}
    else:
        sub_inf = materials.info(substrate)
        n_sub_c = complex(np.sqrt(complex(materials.eps_of(substrate, lam0))))
        if abs(n_sub_c.imag) > 1e-6:
            raise SpecError(f"substrate {substrate} absorbs at {lam0} um (k = {n_sub_c.imag:.2e}); not supported")
        n_sub = float(n_sub_c.real)
    return {"material": material, "n": n, "sourced": bool(sourced), "approximate": bool(inf.get("approximate")),
            "references": inf.get("references") or inf.get("note"), "license": inf.get("license"),
            "range_um": rng, "data_file": data_file, "data_sha256": data_sha,
            "substrate": substrate, "n_sub": n_sub, "substrate_references": sub_inf.get("references"),
            "n_bg": 1.0, "medium": "air"}


def lens_na(D, f, n_bg=1.0):
    return float(n_bg * np.sin(np.arctan(D / 2.0 / f)))


def choose_unit_cell(spec, plat, drc, n_jobs=None, log=print):
    """Step 3 (R2): evaluate the fixed cell or every searched cell under the fabrication limits
    and the sampling limit of the lens NA (L <= lam0 / (2 NA)). Returns (rows, chosen or None)."""
    lam0 = spec["wavelength_um"]; NA = lens_na(spec["lens"]["diameter_um"], spec["lens"]["focal_um"])
    uc, se = spec["unit_cell"], spec["search"]
    if uc["height_um"] is not None:
        cells = [(uc["height_um"], uc["period_um"])]
    else:
        cells = [(H, L) for H in se["height_um"] for L in se["period_um"]]
    fab = spec["fabrication"]; tol = spec["tolerance"]; num = spec["numerics"]
    rows = []
    for H, L in cells:
        if L > lam0 / (2 * NA) * (1 + 1e-12):
            rows.append({"H": H, "L": L, "feasible": False,
                         "reason": f"period {L} um cannot sample NA {NA:.3f} (needs L <= {lam0 / (2 * NA):.4f} um)"})
            continue
        log(f"  unit cell H = {H} um, L = {L} um ...")
        r = recipe.evaluate_cell(lam0, H, L, material=plat["material"], shape=uc["shape"],
                                 min_feature=drc["min_feature_um"], min_gap=drc["min_gap_um"],
                                 max_aspect=drc["max_aspect"] if drc["max_aspect"] is not None else np.inf,
                                 M=num["M"], n_sizes=num["n_sizes"], sigma_nm=tol["sigma_nm"],
                                 n_sub=plat["n_sub"], n_bg=plat["n_bg"], refine=True, n_jobs=n_jobs,
                                 certify=True, sidewall_deg=fab["sidewall_deg"],
                                 residual=tuple(fab["residual"]) if fab["residual"] else None)
        rows.append(r)
    chosen = recipe.recommend(rows)
    return rows, chosen


def index_shift_metrics(d, dn, n_jobs=None):
    """Lens metrics when the pillar film index is n0 + dn for every used size (selection fixed,
    sizes re-solved with RCWA). Returns {dn: analyze_focus subset}."""
    lib = d["lib"]; ins = d["inside"]
    used = np.unique(np.round(d["size_map"][ins], 12))
    out = {}
    for s in dn:
        t = dz.meta_atom_library(lib["lam0"], lib["L"], lib["H"], sizes=used, shape=lib["shape"],
                                 material=None, n_pillar=complex(lib["n"]) + s, M=lib["M"],
                                 n_sub=lib["n_sub"], n_bg=lib["n_bg"], raster_n=lib.get("raster_n"),
                                 n_jobs=n_jobs, sidewall_deg=lib.get("sidewall_deg", 90.0),
                                 n_steps=lib.get("n_steps"), residual=lib.get("residual"))["t"]
        k = np.searchsorted(used, np.round(d["size_map"], 12))
        E0 = np.where(ins, np.asarray(t)[np.clip(k, 0, len(used) - 1)], 0.0)
        dd = dict(d); dd["E0"] = E0
        a = dz.analyze_focus(dd, nz=NZ)
        out[float(s)] = {k_: float(a[k_]) for k_ in ("z_focus", "fwhm", "strehl", "eff_absolute", "eff_relative")}
    return out


def find_validation_records(vdir, plat, lam0, H, L, sidewall_deg=90.0, residual=None):
    """Step 8 (R8): full-wave records in vdir that apply to this platform. A record applies when
    material, wavelength, pillar height, period and substrate index match; records do not store
    the pillar profile, and all current ones are for vertical pillars without a residual film,
    so a tapered or residual-film recipe has no applicable record. Returns {"V1": [...],
    "V2": [...], "skipped_geometry": bool}."""
    out = {"V1": [], "V2": [], "skipped_geometry": False}
    if vdir is None or not os.path.isdir(vdir):
        return out
    if sidewall_deg < 90.0 or residual:
        out["skipped_geometry"] = True
        return out
    for fn in sorted(os.listdir(vdir)):
        if not fn.endswith(".json"):
            continue
        kind = "V1" if fn.startswith("V1") else "V2" if fn.startswith("V2") else None
        if kind is None:
            continue
        try:
            with open(os.path.join(vdir, fn), encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, ValueError):
            continue
        need = ("material", "lam0", "H", "L", "n_sub")
        if any(k not in rec for k in need):
            continue
        if not (rec["material"] == plat["material"] and abs(rec["lam0"] - lam0) < 1e-9
                and abs(rec["H"] - H) < 1e-9 and abs(rec["L"] - L) < 1e-9
                and abs(rec["n_sub"] - plat["n_sub"]) < 1e-6):
            continue
        item = {"file": fn, "sha256": _sha256_file(os.path.join(vdir, fn)), "passed": bool(rec.get("passed"))}
        if kind == "V1":
            da = np.abs(np.asarray(rec.get("d_amp", []), float)); dp = np.abs(np.asarray(rec.get("d_phase_rad", []), float))
            jm = np.asarray(rec.get("judged", [True] * len(da)), bool)
            item.update({"max_d_amp": float(da[jm].max()) if jm.any() else None,     # acceptance is on judged sizes
                         "max_d_phase_rad": float(dp[jm].max()) if jm.any() else None,
                         "n_judged": int(jm.sum()), "n_sizes": int(len(jm)), "fdtd_steps": rec.get("fdtd_steps")})
        else:
            lp, fd = rec.get("lpa", {}), rec.get("fdtd", {})
            item.update({"D": rec.get("D"), "f": rec.get("f"), "steps": rec.get("steps"),
                         "lpa_focus": lp.get("z_focus"), "fdtd_focus": fd.get("z_focus"),
                         "lpa_eff_relative": lp.get("eff_relative_in_fdtd_plane", lp.get("eff_relative")),
                         "fdtd_eff_relative": fd.get("eff_relative")})
        out[kind].append(item)
    return out


def trust_level(checks):
    """Step 10 (R9, plan section 7). checks: dict of booleans. Returns (level, missing) where
    missing lists the first unmet requirements of the next level. T3 needs fabrication data
    (roadmap F4) and is never assigned by this pipeline."""
    t1 = ["material_sourced", "library_certified", "library_refined", "robust_selection",
          "process_window", "drc_clean"]
    t2 = ["unit_cell_fullwave", "lens_fullwave", "error_budget"]
    miss1 = [k for k in t1 if not checks.get(k)]
    if miss1:
        return "T0", miss1
    miss2 = [k for k in t2 if not checks.get(k)]
    if miss2:
        return "T1", miss2
    return "T2", ["fabrication_calibration (roadmap F4: measured SEM sizes and optics)"]


CHECK_TEXT = {
    "material_sourced": "material data from a cited source (shipped dataset or your measured data)",
    "library_certified": "every pillar size used by the lens passes the M vs M+2 convergence certificate",
    "library_refined": "library refined adaptively, resonances located",
    "robust_selection": "atoms chosen by the robust rule for the stated size error",
    "process_window": "size bias, random size error and film-index shift evaluated",
    "drc_clean": "design-rule check clean against the named profile (incl. aspect ratio)",
    "unit_cell_fullwave": "unit-cell FDTD record for this platform passed (V1)",
    "lens_fullwave": "small-lens FDTD record for this platform passed (V2)",
    "error_budget": "error budget on the performance numbers (roadmap T2)",
}


# --------------------------------------------------------------------------- the pipeline
def run_recipe(spec, out_dir, validation_dir="auto", n_jobs=None, log=print):
    """Run the pipeline and write the recipe folder. Returns the result dict (also written as
    recipe.json). validation_dir="auto" uses the repository's validation/ folder if present."""
    spec = load_spec(spec)
    os.makedirs(out_dir, exist_ok=True)
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if validation_dir == "auto":
        cand = os.path.join(pkg_root, "validation")
        validation_dir = cand if os.path.isdir(cand) else None
    lam0 = spec["wavelength_um"]; D = spec["lens"]["diameter_um"]; f = spec["lens"]["focal_um"]
    fab, tol, num = spec["fabrication"], spec["tolerance"], spec["numerics"]
    residual = tuple(fab["residual"]) if fab["residual"] else None
    res = {"schema": SCHEMA, "name": spec["name"], "spec": spec, "status": "running"}

    log("[1/10] specification ok"); log("[2/10] platform")
    plat = platform_info(spec["material"], spec["substrate"], lam0)
    res["platform"] = plat
    drc = load_drc_profile(fab["drc_profile"])
    res["drc_profile"] = {"name": fab["drc_profile"], **drc}

    log("[3/10] unit cell")
    rows, cell = choose_unit_cell(spec, plat, drc, n_jobs=n_jobs, log=log)
    res["unit_cell_candidates"] = rows
    if cell is None:
        res["status"] = "infeasible"
        res["reason"] = "; ".join(sorted({r.get("reason") or "phase circle not closed (gap > pi/4)" for r in rows}))
        return _finish(res, spec, out_dir, validation_dir, pkg_root, log)
    H, L = cell["H"], cell["L"]
    res["unit_cell"] = cell

    log(f"[4-5/10] library and lens design (H = {H} um, L = {L} um)")
    sizes = np.linspace(drc["min_feature_um"], L - drc["min_gap_um"], num["n_sizes"])
    robust = tol["sigma_nm"] > 0                 # sigma 0: nominal (complex-match) selection
    d = dz.design_lens(lam0, L, H, D, f, shape=spec["unit_cell"]["shape"], material=plat["material"],
                       M=num["M"], n_sub=plat["n_sub"], n_bg=plat["n_bg"], sizes=sizes,
                       mode="robust" if robust else "complex", refine=True,
                       sigma_nm=tol["sigma_nm"] if robust else None, certify=True,
                       sidewall_deg=fab["sidewall_deg"], residual=residual)
    lib = d["lib"]; conv = lib["convergence"]; ins = d["inside"]
    used = np.unique(np.round(d["size_map"][ins], 12))
    ok_sizes = np.round(np.asarray(conv["sizes"])[np.asarray(conv["ok"], bool)], 12)
    used_cert = bool(np.all(np.isin(used, ok_sizes)))
    cs = np.round(np.asarray(conv["sizes"]), 12); um = np.isin(cs, used)
    used_dphi = float(np.asarray(conv["d_phase"])[um].max()) if um.any() else None
    used_damp = float(np.asarray(conv["d_amp"])[um].max()) if um.any() else None
    res["library"] = {"M": lib["M"], "M_check": conv["M_check"], "n_sizes": len(lib["sizes"]),
                      "size_range_um": [float(lib["sizes"][0]), float(lib["sizes"][-1])],
                      "coverage_2pi": float(lib["coverage"]), "phase_gap_rad": float(d["phase_gap"]),
                      "resonances": [{k: float(r[k]) for k in ("center", "lo", "hi", "t_min")} for r in lib.get("resonances", [])],
                      "certificate": {"passed_all_sizes": bool(conv["passed"]),
                                      "used_max_d_phase_rad": used_dphi, "used_max_d_amp": used_damp,
                                      "max_d_phase_rad": conv["max_d_phase"], "max_d_amp": conv["max_d_amp"],
                                      "max_d_phase_resonance_rad": conv["max_d_phase_resonance"],
                                      "tol_phase": conv["tol_phase"], "tol_amp": conv["tol_amp"],
                                      "uncertified_um": [float(s) for s, o in zip(conv["sizes"], conv["ok"]) if not o],
                                      "used_sizes_certified": used_cert},
                      "refined": bool(lib.get("refined")),
                      "table": [{"size_um": float(s), "t": complex(t), "certified": bool(o)}
                                for s, t, o in zip(lib["sizes"], lib["t"], conv["ok"])]}
    res["design"] = {"D_um": float(d["D_lens"]), "f_um": f, "NA": float(d["NA"]), "n_pillars": int(ins.sum()),
                     "selection": "robust" if robust else "nominal (complex match, no size error)",
                     "sigma_nm": tol["sigma_nm"], "phase_offset_rad": float(d["offset"]),
                     "sizes_used_um": [float(s) for s in used]}

    log("[6/10] performance (LPA)")
    a = dz.analyze_focus(d, nz=NZ)
    res["performance"] = {"model": "LPA + scalar Rayleigh-Sommerfeld / angular spectrum",
                          "z_focus_um": float(a["z_focus"]), "fwhm_um": float(a["fwhm"]),
                          "diffraction_fwhm_um": float(a["diffraction_fwhm"]), "strehl": float(a["strehl"]),
                          "eff_absolute": float(a["eff_absolute"]), "eff_relative": float(a["eff_relative"]),
                          "fresnel_number": float(a["fresnel_number"]), "error_budget": None}

    log("[7/10] process window")
    b = tol["bias_nm"]
    bias = dz.lens_tolerance(d, bias_nm=(-b, -b / 2, 0.0, b / 2, b), sigma_nm=0.0, nz=NZ)
    rand = dz.lens_tolerance(d, bias_nm=(0.0,), sigma_nm=tol["sigma_nm"], n_trial=tol["n_trial"], seed=tol["seed"],
                             nz=NZ) if tol["sigma_nm"] > 0 else None
    idx = index_shift_metrics(d, (-tol["index_dn"], tol["index_dn"]), n_jobs=n_jobs)
    res["process_window"] = {
        "size_bias": [{"bias_nm": float(x), "strehl": float(s), "eff_absolute": float(e)}
                      for x, s, e in zip(bias["bias_nm"], bias["strehl"][:, 0], bias["eff"][:, 0])],
        "size_bias_out_of_range": float(bias["out_of_range"]),
        "random_size": None if rand is None else {
            "sigma_nm": rand["sigma_nm"], "n_trial": rand["n_trial"], "seed": tol["seed"],
            "strehl_mean": float(rand["strehl_stats"]["mean"][0]), "strehl_p10": float(rand["strehl_stats"]["p10"][0]),
            "eff_mean": float(rand["eff_stats"]["mean"][0]), "eff_p10": float(rand["eff_stats"]["p10"][0])},
        "index_shift": [{"dn": k, **v} for k, v in idx.items()],
        "model": ("interpolated refined library for size errors; RCWA re-solve of the used sizes for the "
                  "index (pillar index only; a residual film keeps its index)")}

    log("[8/10] validation records")
    res["validation"] = find_validation_records(validation_dir, plat, lam0, H, L, fab["sidewall_deg"], residual)
    res["validation"]["dir"] = os.path.relpath(validation_dir, pkg_root) if validation_dir else None

    log("[9/10] fabrication files")
    gds_path = os.path.join(out_dir, "lens.gds")
    from .gds import layout_to_gds
    n_written = layout_to_gds(d["placements"], gds_path, cell_name=_cell_name(spec["name"]), timestamp=GDS_DATE)
    rule = drc_check(d["placements"], profile=drc, height=H)
    res["fabrication"] = {"gds": "lens.gds", "n_pillars_written": int(n_written), "layer": 1, "datatype": 0,
                          "layer_table": [{"layer": 1, "datatype": 0, "content": f"pillars, {plat['material']}, height {H} um"}],
                          "sidewall_deg": fab["sidewall_deg"], "residual": fab["residual"],
                          "drc": {k: rule[k] for k in ("ok", "n", "n_feature_viol", "n_gap_viol", "n_aspect_viol",
                                                        "worst_gap", "min_size", "worst_aspect")},
                          "drc_violations": rule["violations"][:20]}
    try:
        from .gds import read_gds_summary
        rb = read_gds_summary(gds_path)
        res["fabrication"]["gds_readback"] = {**rb, "matches": rb["polygons"] == n_written}
    except ImportError:
        res["fabrication"]["gds_readback"] = None

    checks = {"material_sourced": plat["sourced"] and not plat["approximate"],
              "library_certified": used_cert, "library_refined": bool(lib.get("refined")),
              "robust_selection": robust,
              "process_window": True, "drc_clean": bool(rule["ok"]),
              "unit_cell_fullwave": any(r["passed"] for r in res["validation"]["V1"]),
              "lens_fullwave": any(r["passed"] for r in res["validation"]["V2"]),
              "error_budget": res["performance"]["error_budget"] is not None}
    level, missing = trust_level(checks)
    res["trust"] = {"level": level, "checks": checks, "next_level_needs": missing}
    res["status"] = "ok"
    return _finish(res, spec, out_dir, validation_dir, pkg_root, log)


def _cell_name(name):
    import re
    s = re.sub(r"[^A-Za-z0-9_]", "_", name).upper()[:24]
    return s or "METALENS"


def _finish(res, spec, out_dir, validation_dir, pkg_root, log):
    from . import __version__
    log("[10/10] report and manifest")
    with open(os.path.join(out_dir, "recipe.json"), "w", encoding="utf-8", newline="\n") as fjs:
        json.dump(_jsonable(res), fjs, indent=1, sort_keys=True)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8", newline="\n") as fmd:
        fmd.write(render_report(res))
    outputs = {}
    for fn in sorted(os.listdir(out_dir)):
        if fn != "manifest.json" and os.path.isfile(os.path.join(out_dir, fn)):
            outputs[fn] = _sha256_file(os.path.join(out_dir, fn))
    commit, dirty = _git_state(pkg_root)
    import scipy
    plat = res.get("platform", {})
    manifest = {"schema": SCHEMA, "name": spec["name"], "status": res["status"],
                "trust_level": res.get("trust", {}).get("level"),
                "code": {"package_version": __version__, "git_commit": commit, "git_dirty": dirty},
                "environment": {"python": _platform.python_version(), "numpy": np.__version__,
                                "scipy": scipy.__version__, "platform": _platform.platform()},
                "spec": spec, "spec_sha256": _sha256_json(spec),
                "material_data": {"material": plat.get("material"), "file": plat.get("data_file"),
                                  "sha256": plat.get("data_sha256"), "license": plat.get("license")},
                "validation_records": [{"file": r["file"], "sha256": r["sha256"]}
                                       for k in ("V1", "V2") for r in res.get("validation", {}).get(k, [])],
                "outputs": outputs,
                "reproducibility": ("outputs are byte-identical for the same spec, code commit and "
                                    "environment (checked with a warm and an empty library cache and "
                                    "with 1 and 2 worker processes); created_utc is the only field "
                                    "that changes between runs"),
                "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8", newline="\n") as fm:
        json.dump(_jsonable(manifest), fm, indent=1, sort_keys=True)
    log(f"recipe written to {out_dir} (status {res['status']}, trust {manifest['trust_level']})")
    return res


# --------------------------------------------------------------------------- report
def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _candidate_table(rows, chosen):
    out = ["| H (um) | L (um) | meets limits | coverage (x 2pi) | phase gap (rad) | deep resonances | robust FOM | note |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r.get("feasible"):
            note = r.get("reason", "")
        elif r["max_phase_gap"] > np.pi / 4:
            note = "phase circle not closed (gap > pi/4)"
        else:
            note = "**chosen**" if chosen is not None and (r["H"], r["L"]) == (chosen["H"], chosen["L"]) else ""
        out.append(f"| {r['H']} | {r['L']} | {'yes' if r.get('feasible') else 'no'} | {_f(r.get('coverage_2pi'), 2)} | "
                   f"{_f(r.get('max_phase_gap'))} | {r.get('n_deep_resonances', 'n/a')} | {_f(r.get('fom_robust'))} | {note} |")
    return "\n".join(out)


def render_report(res):
    """Markdown report. Section numbers follow the plan's R1-R10; the trust level comes first."""
    s = res["spec"]; L_ = []
    w = L_.append
    w(f"# Recipe: {s['name']}\n")
    if res["status"] == "infeasible":
        w("**Status: INFEASIBLE** - no unit cell meets the specification.\n")
        w(f"Reason: {res['reason']}\n")
        w("\n## R2 Unit-cell candidates\n")
        w(_candidate_table(res["unit_cell_candidates"], None))
        w("")
        return "\n".join(L_) + "\n"
    t = res["trust"]; p = res["platform"]; u = res["unit_cell"]; lb = res["library"]; dg = res["design"]
    pf = res["performance"]; pw = res["process_window"]; fb = res["fabrication"]; va = res["validation"]
    w(f"## R9 Trust level: **{t['level']}**\n")
    w("| Requirement | Met |\n|---|---|")
    for k, v in t["checks"].items():
        w(f"| {CHECK_TEXT[k]} | {'yes' if v else '**no**'} |")
    w(f"\nNext level needs: {', '.join(t['next_level_needs'])}.\n")
    w("Levels (plan section 7): T0 concept; T1 numerically verified; T2 full-wave verified with an "
      "error budget; T3 calibrated with fabrication data (never assigned by this pipeline).\n")
    w("## Specification\n")
    w(f"Wavelength {s['wavelength_um']} um; lens D = {s['lens']['diameter_um']} um, f = {s['lens']['focal_um']} um; "
      f"material {s['material']} on {s['substrate']}; DRC profile `{s['fabrication']['drc_profile']}`.\n")
    w("## R1 Platform\n")
    w(f"* Pillar: {p['material']}, n = {p['n'].real:.4f} + {p['n'].imag:.2e}i at {s['wavelength_um']} um "
      f"({'sourced' if p['sourced'] else 'NOT sourced'}{', approximate table' if p['approximate'] else ''}).")
    if p["references"]:
        w(f"* Reference: {str(p['references'])[:300]}")
    if p["license"]:
        w(f"* Data license: {p['license']}; file {p['data_file']} (sha256 {str(p['data_sha256'])[:16]}...)")
    w(f"* Substrate: {p['substrate']}, n = {p['n_sub']:.4f}; exit medium {p['medium']} (n = 1).\n")
    w("## R2 Unit cell\n")
    w(f"* Shape {s['unit_cell']['shape']}, height H = {u['H']} um, period L = {u['L']} um, "
      f"sidewall {s['fabrication']['sidewall_deg']} deg, residual film {s['fabrication']['residual'] or 'none'}.")
    w(f"* Size range {lb['size_range_um'][0]*1000:.0f}-{lb['size_range_um'][1]*1000:.0f} nm "
      f"(min feature {res['drc_profile']['min_feature_um']*1000:.0f} nm, min gap {res['drc_profile']['min_gap_um']*1000:.0f} nm); "
      f"aspect ratio {u['aspect']:.1f} (limit {res['drc_profile']['max_aspect']}).")
    w(f"* Robust FOM {u['fom_robust']:.3f} (nominal {u['fom_nominal']:.3f}); highest NA the period samples "
      f"{u['na_max_sampling']:.3f}; lens NA {dg['NA']:.3f}.")
    if len(res["unit_cell_candidates"]) > 1:
        w(f"* Chosen among {len(res['unit_cell_candidates'])} candidates by `recipe.recommend`: phase gap <= pi/4, "
          "robust FOM within 0.01 of the best, then fewest deep resonances (|t| < 0.5), then highest samplable NA.\n")
        w(_candidate_table(res["unit_cell_candidates"], u))
    w("")
    c = lb["certificate"]
    w("## R3 Library\n")
    w(f"* {lb['n_sizes']} sizes (refined: {lb['refined']}), RCWA M = {lb['M']}, phase coverage "
      f"{lb['coverage_2pi']:.2f} x 2pi, largest phase gap {lb['phase_gap_rad']:.3f} rad among usable sizes.")
    w(f"* Convergence certificate (M = {lb['M']} vs {lb['M_check']}, tolerances {c['tol_phase']} rad and "
      f"{c['tol_amp']} in |t|): on the sizes the lens uses the largest change is {_f(c['used_max_d_phase_rad'], 4)} rad "
      f"and {_f(c['used_max_d_amp'], 4)} in |t|; over the whole library outside resonances {c['max_d_phase_rad']:.4f} rad and "
      f"{c['max_d_amp']:.4f} (sizes that fail are excluded below). The M vs M+2 change estimates the "
      "truncation error; it is not a bound.")
    if c["uncertified_um"]:
        w(f"* Not certified, excluded from the design: {', '.join(f'{x*1000:.1f}' for x in c['uncertified_um'])} nm.")
    if lb["resonances"]:
        w("* Resonances: " + "; ".join(f"{r['center']*1000:.1f} nm (|t| min {r['t_min']:.2f})" for r in lb["resonances"]))
    w(f"* All sizes used by the lens are certified: {'yes' if c['used_sizes_certified'] else '**no**'}.\n")
    w("## R4 Design\n")
    w(f"* D = {dg['D_um']:.3f} um (snapped to the lattice), f = {dg['f_um']} um, NA = {dg['NA']:.3f}, "
      f"{dg['n_pillars']} pillars, {len(dg['sizes_used_um'])} distinct sizes.")
    w(f"* Selection: {'robust rule for a ' + str(dg['sigma_nm']) + ' nm size error' if dg['selection'] == 'robust' else dg['selection']}; "
      "hyperbolic phase.\n")
    w("## R5 Performance (locally periodic approximation)\n")
    w("| Focus (um) | FWHM (nm) | diffraction FWHM (nm) | Strehl | abs. efficiency | rel. efficiency |\n|---|---|---|---|---|---|")
    w(f"| {pf['z_focus_um']:.3f} | {pf['fwhm_um']*1000:.0f} | {pf['diffraction_fwhm_um']*1000:.0f} | "
      f"{pf['strehl']:.3f} | {pf['eff_absolute']:.3f} | {pf['eff_relative']:.3f} |")
    w(f"\nFresnel number {pf['fresnel_number']:.2f}" + (" (small: the intensity maximum lies before f)" if pf["fresnel_number"] < 5 else "") + ".")
    if pf["error_budget"] is None:
        w("Error budget: **not computed** (roadmap T2); the numbers above carry no error bars yet.\n")
    w("## R6 Process window\n")
    w("| size bias (nm) | Strehl | abs. efficiency |\n|---|---|---|")
    for r in pw["size_bias"]:
        w(f"| {r['bias_nm']:+.1f} | {r['strehl']:.3f} | {r['eff_absolute']:.3f} |")
    if pw["random_size"]:
        rr = pw["random_size"]
        w(f"\nRandom size error sigma = {rr['sigma_nm']} nm ({rr['n_trial']} trials, seed {rr['seed']}): Strehl mean "
          f"{rr['strehl_mean']:.3f} (p10 {rr['strehl_p10']:.3f}), efficiency mean {rr['eff_mean']:.3f} (p10 {rr['eff_p10']:.3f}).")
    w("\n| film index shift dn | focus (um) | Strehl | abs. efficiency |\n|---|---|---|---|")
    for r in pw["index_shift"]:
        w(f"| {r['dn']:+.3f} | {r['z_focus']:.3f} | {r['strehl']:.3f} | {r['eff_absolute']:.3f} |")
    if pw["size_bias_out_of_range"] > 0:
        w(f"\n{pw['size_bias_out_of_range']*100:.1f} % of the pillars leave the library range at the largest bias "
          "(their response is clamped to the end of the library).")
    w(f"\nModel: {pw['model']}.\n")
    w("## R7 Fabrication files\n")
    dr = fb["drc"]
    w(f"* `lens.gds`: {fb['n_pillars_written']} pillars, layer {fb['layer']}/{fb['datatype']} "
      f"({fb['layer_table'][0]['content']}); hierarchical, fixed date for reproducibility.")
    if fb.get("gds_readback"):
        w(f"* Read back: {fb['gds_readback']['polygons']} polygons (matches: {fb['gds_readback']['matches']}).")
    w(f"* DRC (`{s['fabrication']['drc_profile']}`): {'clean' if dr['ok'] else '**violations**'} - smallest pillar "
      f"{dr['min_size']*1000:.1f} nm, smallest gap {dr['worst_gap']*1000:.1f} nm, largest aspect ratio {dr['worst_aspect']:.1f}; "
      f"{dr['n_feature_viol']} feature, {dr['n_gap_viol']} gap, {dr['n_aspect_viol']} aspect violations.\n")
    w("## R8 Validation records\n")
    if va["skipped_geometry"]:
        w("No record applies: the existing records are for vertical pillars without a residual film.\n")
    elif not va["V1"] and not va["V2"]:
        w(f"No full-wave record matches this platform (searched `{va['dir']}`).\n")
    for r in va["V1"]:
        w(f"* V1 unit cell `{r['file']}`: {'passed' if r['passed'] else 'FAILED'}; on the {r['n_judged']} judged of "
          f"{r['n_sizes']} sizes max |dt| {_f(r['max_d_amp'], 4)}, max dphase {_f(r['max_d_phase_rad'], 3)} rad "
          f"({r['fdtd_steps']} FDTD steps/wavelength).")
    for r in va["V2"]:
        w(f"* V2 lens `{r['file']}` (D {r['D']:.2f} um, f {r['f']} um, {r['steps']} steps/wavelength): "
          f"{'passed' if r['passed'] else 'not passed'}; focus LPA {_f(r['lpa_focus'])} vs FDTD {_f(r['fdtd_focus'])} um, "
          f"relative efficiency {_f(r['lpa_eff_relative'])} vs {_f(r['fdtd_eff_relative'])}.")
    w("\nThe records validate the platform (material, wavelength, H, L, substrate), not this particular lens.\n")
    w("## R10 Traceability\n")
    w("`manifest.json` holds the code version and commit, the normalized specification and its hash, the "
      "material data hash, the validation record hashes and the hashes of every output file.\n")
    return "\n".join(L_) + "\n"
