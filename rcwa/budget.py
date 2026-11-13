"""Error budget of the predicted lens performance (roadmap T2).

For each metric - focus position, FWHM, Strehl ratio, absolute and relative efficiency - the
budget lists independent terms, each the largest deviation from the nominal prediction over
a stated scenario set:

  numerical_rcwa     the used pillar sizes re-solved with M + 2 Fourier orders
  numerical_prop     focal-plane metrics with an FFT window twice as large (on-axis values,
                     i.e. focus and Strehl, are exact Rayleigh-Sommerfeld sums: 0)
  material           pillar film index n0 +- dn (used sizes re-solved with RCWA)
  fabrication_bias   uniform size bias +-b and +-b/2 (interpolated refined library)
  fabrication_random random size error N(0, sigma^2) per pillar: |mean shift| + 1 std over
                     the Monte Carlo trials
  model              locally periodic approximation + scalar propagation against FDTD, from
                     the platform's small-lens records (V2): the largest |FDTD - LPA| seen.
                     Only focus, FWHM and relative efficiency were measured; Strehl and
                     absolute efficiency have no model term (None, reported as not measured).

Terms are half-widths of ranges, not standard deviations. `rss` combines them in quadrature
(assuming independence); `linear` is their sum (worst case). The model term also carries its
sign: in V2 the FDTD focus was shorter than the LPA focus.
"""
import numpy as np

from . import design as dz

METRICS = ("z_focus", "fwhm", "strehl", "eff_absolute", "eff_relative")
RELATIVE = ("z_focus", "fwhm")            # quoted relative to the value; others in absolute points
MODEL_MEASURED = ("z_focus", "fwhm", "eff_relative")
NZ = 81


def lens_metrics(d, E0=None, pad=None):
    """The five metrics of a design (optionally with another aperture field or FFT window)."""
    dd = d if E0 is None else {**d, "E0": E0}
    a = dz.analyze_focus(dd, pad=pad, nz=NZ)
    return {k: float(a[k]) for k in METRICS}


def _resolved_field(d, n_pillar=None, M=None, n_jobs=None):
    """Aperture field with every used size re-solved by RCWA (other index or order count)."""
    lib = d["lib"]; ins = d["inside"]
    used = np.unique(np.round(d["size_map"][ins], 12))
    t = dz.meta_atom_library(lib["lam0"], lib["L"], lib["H"], sizes=used, shape=lib["shape"], material=None,
                             n_pillar=complex(lib["n"]) if n_pillar is None else n_pillar,
                             M=lib["M"] if M is None else M, n_sub=lib["n_sub"], n_bg=lib["n_bg"],
                             raster_n=lib.get("raster_n"), n_jobs=n_jobs,
                             sidewall_deg=lib.get("sidewall_deg", 90.0), n_steps=lib.get("n_steps"),
                             residual=lib.get("residual"))["t"]
    k = np.searchsorted(used, np.round(d["size_map"], 12))
    return np.where(ins, np.asarray(t)[np.clip(k, 0, len(used) - 1)], 0.0)


def _dev(nom, others):
    return {k: float(max(abs(o[k] - nom[k]) for o in others)) for k in METRICS}


def model_term(records, NA):
    """Model (LPA) deviations from the applicable V2 records: dict metric -> (abs relative or
    absolute deviation, signed mean, files) for the measured metrics; None if no record
    applies. A record applies when it is for this platform (the caller filters that) and its
    lens NA is at least the recipe's NA (the LPA error grows with NA)."""
    use = [r for r in records if r.get("NA") is not None and NA <= r["NA"] + 1e-3]
    if not use:
        return None
    out = {"files": [r["file"] for r in use], "record_NA": max(r["NA"] for r in use), "metrics": {}}
    pairs = {"z_focus": ("lpa_focus", "fdtd_focus", True), "fwhm": ("lpa_fwhm", "fdtd_fwhm", True),
             "eff_relative": ("lpa_eff_relative", "fdtd_eff_relative", False)}
    for m, (kl, kf, rel) in pairs.items():
        dv = [(r[kf] - r[kl]) / r[kl] if rel else (r[kf] - r[kl]) for r in use
              if r.get(kl) is not None and r.get(kf) is not None]
        out["metrics"][m] = None if not dv else {"max_abs": float(max(abs(v) for v in dv)),
                                                 "mean_signed": float(np.mean(dv)), "relative": rel, "n": len(dv)}
    return out


def error_budget(d, v2_records=(), dn=0.02, bias_nm=4.0, sigma_nm=3.0, n_trial=10, seed=0, n_jobs=None):
    """Error budget of design d (see module docstring). v2_records: the matching V2 items of
    recipe_run.find_validation_records. Returns dict(nominal, scenarios, terms, model, total,
    complete, gaps). `scenarios` holds every evaluated case (also used for the process window)."""
    nom = lens_metrics(d)
    sc = {}
    # numerical: RCWA order count and FFT window
    M2 = int(d["lib"]["M"]) + 2
    sc["rcwa_M+2"] = lens_metrics(d, _resolved_field(d, M=M2, n_jobs=n_jobs))
    ff = dz.FocalField(d["E0"], d["L"], d["lam0"], n_bg=d["n_bg"])
    big_pad = 2 * ff.pad
    prop_ok = int(ff._Ncf * big_pad) <= 2 * dz.FFT_MAX_N
    sc["fft_window_x2"] = lens_metrics(d, pad=big_pad) if prop_ok else None
    # material
    n0 = complex(d["lib"]["n"])
    for s in (-dn, dn):
        sc[f"index_{s:+.3f}"] = lens_metrics(d, _resolved_field(d, n_pillar=n0 + s, n_jobs=n_jobs))
    # fabrication: uniform bias, then random size error
    lib = d["lib"] if d["lib"].get("refined") else dz.refine_library(d["lib"])
    bias = [-bias_nm, -bias_nm / 2, bias_nm / 2, bias_nm]
    oor = 0.0
    for b in bias:
        E0, o = dz.perturbed_field(d, lib, b * 1e-3); oor = max(oor, o)
        sc[f"bias_{b:+.1f}nm"] = lens_metrics(d, E0)
    rand = []
    if sigma_nm > 0:
        rng = np.random.default_rng(seed)
        for _ in range(int(n_trial)):
            E0, o = dz.perturbed_field(d, lib, rng.normal(0.0, sigma_nm * 1e-3, d["inside"].shape)); oor = max(oor, o)
            rand.append(lens_metrics(d, E0))
    terms = {"numerical_rcwa": _dev(nom, [sc["rcwa_M+2"]]),
             "numerical_prop": None if sc["fft_window_x2"] is None else
             {**_dev(nom, [sc["fft_window_x2"]]), "z_focus": 0.0, "strehl": 0.0},
             "material": _dev(nom, [sc[f"index_{s:+.3f}"] for s in (-dn, dn)]),
             "fabrication_bias": _dev(nom, [sc[f"bias_{b:+.1f}nm"] for b in bias]),
             "fabrication_random": None}
    if rand:
        A = {k: np.array([r[k] for r in rand]) for k in METRICS}
        terms["fabrication_random"] = {k: float(abs(A[k].mean() - nom[k]) + A[k].std()) for k in METRICS}
        sc["random"] = {"n_trial": int(n_trial), "seed": int(seed), "sigma_nm": float(sigma_nm),
                        "mean": {k: float(A[k].mean()) for k in METRICS},
                        "std": {k: float(A[k].std()) for k in METRICS},
                        "p10": {k: float(np.percentile(A[k], 10)) for k in METRICS}}
    NA = float(d["NA"])
    mt = model_term(list(v2_records), NA)
    model = {}
    for k in METRICS:
        m = None if mt is None else mt["metrics"].get(k)
        model[k] = None if m is None else (m["max_abs"] * nom[k] if m["relative"] else m["max_abs"])
    terms["model"] = model
    total = {}
    for k in METRICS:
        vals = [t[k] for t in terms.values() if t is not None and t.get(k) is not None]
        total[k] = {"rss": float(np.sqrt(np.sum(np.square(vals)))), "linear": float(np.sum(vals)),
                    "n_terms": len(vals)}
    gaps = []
    if sc["fft_window_x2"] is None:
        gaps.append("FFT window term not evaluated (lens too large for a doubled window)")
    if sigma_nm <= 0:
        gaps.append("no random size error term (sigma_nm = 0)")
    if mt is None:
        gaps.append(f"no V2 record of this platform covers NA {NA:.3f}: model term unknown")
    gaps_model_unmeasured = [k for k in METRICS if model[k] is None and (mt is not None)]
    complete = (mt is not None and all(model[k] is not None for k in MODEL_MEASURED)
                and sc["fft_window_x2"] is not None and sigma_nm > 0)
    return {"nominal": nom, "scenarios": sc, "terms": terms, "model": mt, "total": total,
            "out_of_range": float(oor), "complete": bool(complete), "gaps": gaps,
            "model_not_measured": gaps_model_unmeasured,
            "settings": {"dn": dn, "bias_nm": bias_nm, "sigma_nm": sigma_nm, "n_trial": int(n_trial),
                         "seed": int(seed), "M_check": M2, "fft_pad_check": big_pad}}
