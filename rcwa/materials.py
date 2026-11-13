"""Dispersive material library — A3.

Wavelength-dependent refractive index n(λ) and absorption k (loss). eps = (n + ik)^2,
Im(eps) > 0 = absorption. λ is in microns (um).

DATA QUALITY — read before fabricating anything:
the built-in models and tables are APPROXIMATE, coarse literature values (a handful of
points across the visible). They are fine for learning, method development and relative
comparisons, not for a mask you will pay for. The index of deposited films (TiO2, SiN,
a-Si ...) depends strongly on the process: measure your film (ellipsometry) or take a
dataset from refractiveindex.info and load it with ``register_yaml`` / ``register_table``.
``info(name)`` tells where a material's data comes from and its valid wavelength range;
tables refuse to extrapolate outside that range.
"""
import numpy as np

def _sellmeier(lam, B, C):
    l2 = lam ** 2
    n2 = 1.0 + sum(b * l2 / (l2 - c) for b, c in zip(B, C))
    return np.sqrt(n2)

# Approximate models (visible region)
def n_SiO2(lam):   # fused silica (Malitson)
    return _sellmeier(lam, [0.6961663, 0.4079426, 0.8974794],
                      [0.0684043**2, 0.1162414**2, 9.896161**2])

def n_TiO2(lam):   # TiO2 (approx., visible) — n~2.4 range
    return np.sqrt(5.913 + 0.2441 / (lam ** 2 - 0.0803))

def n_Si(lam):     # crystalline Si (approx.); absorbing in the visible
    n = 3.42 + 0.09 / (lam ** 2)
    k = np.where(lam < 0.5, 0.2, 0.02)   # coarse loss
    return n + 1j * k

def n_Ag(lam):     # silver (coarse Drude approximation, metal — high loss)
    # qualitative only; use the Johnson & Christy table for precise work
    eps = 5.0 - (9.2 ** 2) / ((1.2398 / lam) * ((1.2398 / lam) + 1j * 0.02))
    return np.sqrt(eps + 0j)

MATERIALS = {"SiO2": n_SiO2, "TiO2": n_TiO2, "Si": n_Si, "Ag": n_Ag, "air": lambda lam: 1.0 + 0j}

def eps_of(material, lam):
    """Complex eps of the material at λ. material: a name or a constant n."""
    if isinstance(material, (int, float, complex)):
        return complex(material) ** 2
    n = MATERIALS[material](lam)
    return np.asarray(n) ** 2


# --- Measured n,k table support (Item 3) ---
import os as _os
_DATA_DIR = _os.path.join(_os.path.dirname(__file__), "materials_data")
_TABLES = {}

def load_nk_csv(path):
    """refractiveindex.info-like CSV: rows 'wavelength_um,n,k'. (# comment)."""
    wl, n, k = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(";", ",").split(",")
            if not parts[0].replace(".", "", 1).replace("e-", "", 1).replace("-", "", 1).isdigit():
                continue                                   # header line
            wl.append(float(parts[0])); n.append(float(parts[1]))
            k.append(float(parts[2]) if len(parts) > 2 else 0.0)
    return np.array(wl), np.array(n), np.array(k)

_NOTES = {"SiO2": "Sellmeier fit (Malitson 1965), fused silica — reliable 0.21-3.7 um",
          "TiO2": "approximate single-pole dispersion formula, visible only",
          "Si": "approximate formula with a coarse two-level loss, visible only",
          "Ag": "coarse Drude model — qualitative only",
          "air": "n = 1"}


def register_table(name, wl, n, k, note="user-registered table"):
    """Register tabulated n,k data (wl in um). The table is used by eps_of / n_table."""
    idx = np.argsort(wl); _TABLES[name] = (np.array(wl, float)[idx], np.array(n, float)[idx], np.array(k, float)[idx])
    _NOTES[name] = note


def _first_comment(path):
    with open(path) as f:
        line = f.readline().strip()
    if not line.startswith("#"):
        return ""
    line = line.lstrip("# ").strip()
    if "(" in line and line.endswith(")"):               # "wavelength_um,n,k  (source ...)" -> source
        line = line[line.index("(") + 1:-1]
    return line.replace(" - approx.", "").replace("approx., ", "").strip()

def _load_builtin_tables():
    if not _os.path.isdir(_DATA_DIR):
        return
    for f in _os.listdir(_DATA_DIR):
        if f.endswith(".csv"):
            name = f[:-4]
            try:
                p = _os.path.join(_DATA_DIR, f)
                register_table(name, *load_nk_csv(p),
                               note="built-in APPROXIMATE table — " + _first_comment(p))
            except Exception:
                pass
_load_builtin_tables()

def n_table(name, lam):
    """Complex refractive index (n+ik) at λ from the table, linear interpolation."""
    wl, n, k = _TABLES[name]
    la = np.asarray(lam, float)
    if np.any(la < wl[0] - 1e-9) or np.any(la > wl[-1] + 1e-9):
        raise ValueError(f"material '{name}': wavelength {float(np.min(la)):.4g}-{float(np.max(la)):.4g} um is outside "
                         f"the tabulated range {wl[0]:.4g}-{wl[-1]:.4g} um (no extrapolation). "
                         f"Register a wider dataset with materials.register_yaml / register_table.")
    nn = np.interp(lam, wl, n); kk = np.interp(lam, wl, k)
    return nn + 1j * kk

# extend eps_of to also recognize table names
_eps_of_base = eps_of
def eps_of(material, lam):  # noqa: F811
    """Complex permittivity (n + ik)^2 at wavelength lam (um). material: a registered name
    (table or formula) or a number (constant refractive index)."""
    if isinstance(material, str) and material in _TABLES:
        return n_table(material, lam) ** 2
    if isinstance(material, str) and material not in MATERIALS:
        import difflib
        near = difflib.get_close_matches(material, list_materials(), n=3, cutoff=0.4)
        raise ValueError(f"unknown material '{material}'" + (f" (did you mean {', '.join(near)}?)" if near else "")
                         + f". Available: {', '.join(list_materials())}")
    return _eps_of_base(material, lam)


def list_materials():
    return sorted(set(MATERIALS.keys()) | set(_TABLES.keys()))


def info(name):
    """Provenance of a material -> {name, kind, range_um, points, note, approximate}."""
    if name in _TABLES:
        wl = _TABLES[name][0]; note = _NOTES.get(name, "table")
        return {"name": name, "kind": "table", "range_um": (float(wl[0]), float(wl[-1])),
                "points": int(len(wl)), "note": note, "approximate": "APPROXIMATE" in note}
    if name in MATERIALS:
        note = _NOTES.get(name, "user-registered formula")
        return {"name": name, "kind": "formula", "range_um": None, "points": None, "note": note,
                "approximate": "approx" in note.lower() or "coarse" in note.lower()}
    raise ValueError(f"unknown material '{name}'")


# --- E7: refractiveindex.info YAML import + broadband + anisotropic ---
def sellmeier_formula(lam, coeffs, formula=1):
    """refractiveindex.info Sellmeier (formula 1/2). coeffs: [C1,C2,...] (RI.info order).
    formula 1: n^2 = 1 + Σ Ci λ^2/(λ^2 - C(i+1)^2). λ in microns."""
    l2 = lam ** 2; n2 = 1.0 + coeffs[0]
    it = iter(coeffs[1:])
    for b in it:
        c = next(it)
        if formula == 2:
            n2 += b * l2 / (l2 - c)
        else:
            n2 += b * l2 / (l2 - c ** 2)
    return np.sqrt(n2)


def broadband_fit(wl, n, k, deg=6):
    """Polynomial broadband model over measured (wl,n,k) (MCM-like).
    Returns: fn(lam)->n+ik. For fast, smooth dispersion in the solver."""
    cn = np.polyfit(wl, n, deg); ck = np.polyfit(wl, k, deg)
    return lambda lam, cn=cn, ck=ck: np.polyval(cn, lam) + 1j * np.polyval(ck, lam)


def eps_tensor(material, lam):
    """Anisotropic diagonal tensor eps = diag(exx,eyy,ezz).
    material: scalar name/number (isotropic) OR {'xx':..,'yy':..,'zz':..} (per-component materials).
    Note: our native RCWA solver uses an isotropic ER; birefringence is achieved GEOMETRICALLY
    (see birefringent_library). This function is for tensor data representation + future use."""
    if isinstance(material, dict):
        return np.diag([complex(eps_of(material[a], lam)) for a in ("xx", "yy", "zz")])
    e = complex(eps_of(material, lam))
    return np.diag([e, e, e])


# --- sourced datasets (roadmap M1): full refractiveindex.info reader + shipped files ---
def rii_formula(lam, formula, c):
    """refractiveindex.info dispersion formulas 1-5 (lam in um), returns real n.
    1 Sellmeier:   n^2 - 1 = C1 + sum Ci lam^2 / (lam^2 - C(i+1)^2)
    2 Sellmeier-2: n^2 - 1 = C1 + sum Ci lam^2 / (lam^2 - C(i+1))
    3 Polynomial:  n^2     = C1 + sum Ci lam^C(i+1)
    4 RI.info:     n^2     = C1 + C2 lam^C3/(lam^2 - C4^C5) + C6 lam^C7/(lam^2 - C8^C9)
                             + sum_{i>=10} Ci lam^C(i+1)
    5 Cauchy:      n       = C1 + sum Ci lam^C(i+1)"""
    lam = np.asarray(lam, float); c = list(c)
    pairs = lambda start: zip(c[start::2], c[start + 1::2])   # noqa: E731
    if formula in (1, 2):
        n2 = 1.0 + c[0]
        for b, cc in pairs(1):
            n2 = n2 + b * lam ** 2 / (lam ** 2 - (cc ** 2 if formula == 1 else cc))
        return np.sqrt(n2)
    if formula == 3:
        return np.sqrt(c[0] + sum(b * lam ** e for b, e in pairs(1)))
    if formula == 4:
        c = c + [0.0] * max(0, 9 - len(c))
        n2 = c[0] + c[1] * lam ** c[2] / (lam ** 2 - c[3] ** c[4]) + c[5] * lam ** c[6] / (lam ** 2 - c[7] ** c[8])
        for b, e in pairs(9):
            n2 = n2 + b * lam ** e
        return np.sqrt(n2)
    if formula == 5:
        return c[0] + sum(b * lam ** e for b, e in pairs(1))
    raise ValueError(f"refractiveindex.info formula {formula} is not supported (1-5 are)")


def _plain(text):
    """REFERENCES / COMMENTS text without HTML or Markdown markup, one line."""
    import re
    t = re.sub(r"<[^>]+>", "", str(text or "")).replace("**", "").replace("*", "")
    return " ".join(t.split())


def load_rii(path):
    """Read a refractiveindex.info YAML file completely: every DATA block (tabulated n, k, nk
    and formulas 1-5) is combined into one complex index n + ik with its validity range (the
    intersection of the blocks' ranges). Returns dict(fn, range_um, kind, references, notes,
    file). fn raises ValueError outside the range (no extrapolation)."""
    import yaml
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    n_part = k_part = None
    for blk in doc.get("DATA", []):
        t = blk.get("type", "").strip()
        if t.startswith("tabulated"):
            arr = np.array([r.split() for r in blk["data"].strip().splitlines()], float)
            arr = arr[np.argsort(arr[:, 0])]
            what = t.split()[1]
            if what in ("n", "nk"):
                n_part = ("tab", arr[:, 0], arr[:, 1])
            if what == "nk":
                k_part = ("tab", arr[:, 0], arr[:, 2])
            if what == "k":
                k_part = ("tab", arr[:, 0], arr[:, 1])
        elif t.startswith("formula"):
            fnum = int(t.split()[1]); coeffs = [float(v) for v in str(blk["coefficients"]).split()]
            wr = blk.get("wavelength_range")          # absent in hand-written files: unbounded
            lo, hi = (float(v) for v in str(wr).split()) if wr else (0.0, float("inf"))
            n_part = ("formula", fnum, coeffs, lo, hi)
    if n_part is None:
        raise ValueError(f"{path}: no refractive-index data block")

    def rng(p):
        return (p[1][0], p[1][-1]) if p[0] == "tab" else (p[3], p[4])
    lo, hi = rng(n_part)
    if k_part is not None:
        lo, hi = max(lo, k_part[1][0]), min(hi, k_part[1][-1])
    lo, hi = float(lo), float(hi)
    name = _os.path.basename(path)

    def fn(lam, n_part=n_part, k_part=k_part, lo=lo, hi=hi):
        la = np.asarray(lam, float)
        if np.any(la < lo - 1e-9) or np.any(la > hi + 1e-9):
            raise ValueError(f"{name}: wavelength {float(np.min(la)):.4g}-{float(np.max(la)):.4g} um is outside "
                             f"the data range {lo:.4g}-{hi:.4g} um (no extrapolation)")
        n = (np.interp(la, n_part[1], n_part[2]) if n_part[0] == "tab"
             else rii_formula(la, n_part[1], n_part[2]))
        k = np.interp(la, k_part[1], k_part[2]) if k_part is not None else 0.0
        return n + 1j * k
    kind = "table" if n_part[0] == "tab" else f"formula {n_part[1]}"
    return {"fn": fn, "range_um": (lo, hi), "kind": kind, "file": name,
            "references": _plain(doc.get("REFERENCES")),
            "notes": _plain(doc.get("NOTES") or doc.get("COMMENTS"))}


_SOURCES = {}     # name -> load_rii() record (provenance of sourced materials)


def register_rii(name, path, note=None):
    """Register a refractiveindex.info YAML file as material `name` (range-checked)."""
    rec = load_rii(path)
    MATERIALS[name] = rec["fn"]
    _SOURCES[name] = rec
    _NOTES[name] = note or f"refractiveindex.info: {rec['references'][:120]}"
    return rec["kind"]


def register_yaml(name, path):
    """Register a refractiveindex.info YAML as a material (all formats of load_rii)."""
    return register_rii(name, path)


def load_nk_yaml(path):
    """Backward-compatible reader: ('table', wl, n, k) for a single tabulated block,
    otherwise ('formula', fn) with fn = load_rii(path)['fn'] (range-checked)."""
    rec = load_rii(path)
    if rec["kind"] == "table":
        import yaml
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        blocks = doc.get("DATA", [])
        if len(blocks) == 1 and blocks[0]["type"].strip() in ("tabulated nk", "tabulated n"):
            arr = np.array([r.split() for r in blocks[0]["data"].strip().splitlines()], float)
            k = arr[:, 2] if arr.shape[1] > 2 else np.zeros(len(arr))
            return ("table", arr[:, 0], arr[:, 1], k)
    return ("formula", rec["fn"])


# Sourced datasets shipped with the package (refractiveindex.info, CC0 1.0 public domain;
# files unchanged, see materials_data/rii/SOURCES.md). name -> file
SOURCED = {
    "TiO2_Siefke": "TiO2_Siefke.yml",            # ALD film, 0.12-125 um
    "TiO2_Jolivet_amorphous": "TiO2_Jolivet-amorphous.yml",   # ALD 200 C, 0.207-0.827 um
    "TiO2_Sarkar": "TiO2_Sarkar.yml",            # 200 nm film on glass, 0.30-1.69 um
    "TiO2_Devore_o": "TiO2_Devore-o.yml",        # rutile crystal, ordinary ray
    "Si3N4_Luke": "Si3N4_Luke.yml",              # LPCVD film, 0.31-5.5 um
    "Si3N4_Philipp": "Si3N4_Philipp.yml",
    "SiO2_Malitson": "SiO2_Malitson.yml",        # fused silica
    "Al2O3_Malitson_o": "Al2O3_Malitson.yml",    # sapphire, ordinary ray
    "Al2O3_Boidin": "Al2O3_Boidin.yml",          # PLD amorphous film, n only
    "Al2O3_Kischkat": "Al2O3_Kischkat.yml",      # sputtered film, 1.54-14.3 um
    "GaN_Kawashima_o": "GaN_Kawashima.yml",      # wurtzite film, ordinary ray
    "GaN_Barker_o": "GaN_Barker-o.yml",
    "aSi_Pierce": "Si_Pierce.yml",               # amorphous Si, 60 nm film, coarse (46 points)
    "Si_Green2008": "Si_Green-2008.yml",         # crystalline Si, 300 K
    "Si_Schinke": "Si_Schinke.yml",              # crystalline Si, 300 K, with uncertainties
}
_RII_DIR = _os.path.join(_DATA_DIR, "rii")


def _load_sourced():
    for name, fname in SOURCED.items():
        p = _os.path.join(_RII_DIR, fname)
        if _os.path.isfile(p):
            register_rii(name, p)
_load_sourced()


_info_base = info
def info(name):  # noqa: F811
    """Provenance of a material -> {name, kind, range_um, points, note, approximate,
    references, source_file, license} (the last three for sourced datasets)."""
    if name in _SOURCES:
        r = _SOURCES[name]
        return {"name": name, "kind": r["kind"], "range_um": r["range_um"], "points": None,
                "note": f"{r['notes']} — {r['references'][:160]}", "approximate": False,
                "references": r["references"], "source_file": r["file"],
                "license": "refractiveindex.info database, CC0 1.0 (public domain)"}
    return _info_base(name)
