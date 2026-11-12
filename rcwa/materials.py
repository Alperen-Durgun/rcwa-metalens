"""Dispersive material library — A3.

Wavelength-dependent refractive index n(λ) and absorption k (loss). eps = (n + ik)^2.
λ is in microns (um). Simple Sellmeier/tables; extend as needed.
Source values are approximate literature; replace with measured tables for precise work.
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
            wl.append(float(parts[0])); n.append(float(parts[1]))
            k.append(float(parts[2]) if len(parts) > 2 else 0.0)
    return np.array(wl), np.array(n), np.array(k)

def register_table(name, wl, n, k):
    idx = np.argsort(wl); _TABLES[name] = (np.array(wl)[idx], np.array(n)[idx], np.array(k)[idx])

def _load_builtin_tables():
    if not _os.path.isdir(_DATA_DIR):
        return
    for f in _os.listdir(_DATA_DIR):
        if f.endswith(".csv"):
            name = f[:-4]
            try:
                register_table(name, *load_nk_csv(_os.path.join(_DATA_DIR, f)))
            except Exception:
                pass
_load_builtin_tables()

def n_table(name, lam):
    """Complex refractive index (n+ik) at λ from the table, linear interpolation."""
    wl, n, k = _TABLES[name]
    nn = np.interp(lam, wl, n); kk = np.interp(lam, wl, k)
    return nn + 1j * kk

# extend eps_of to also recognize table names
_eps_of_base = eps_of
def eps_of(material, lam):  # noqa: F811
    if isinstance(material, str) and material in _TABLES:
        return n_table(material, lam) ** 2
    return _eps_of_base(material, lam)

def list_materials():
    return sorted(list(MATERIALS.keys()) + list(_TABLES.keys()))


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


def load_nk_yaml(path):
    """Import a refractiveindex.info .yml file. Supports 'tabulated nk/n' or
    'formula' (Sellmeier). Returns: ('table',wl,n,k) or ('formula',fn)."""
    import yaml
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    for blk in doc.get("DATA", []):
        t = blk.get("type", "")
        if t.startswith("tabulated"):
            rows = [r.split() for r in blk["data"].strip().splitlines()]
            arr = np.array(rows, float)
            wl = arr[:, 0]; n = arr[:, 1]
            k = arr[:, 2] if arr.shape[1] > 2 else np.zeros_like(n)
            return ("table", wl, n, k)
        if t.startswith("formula"):
            formula = int(t.split()[1]); coeffs = [float(v) for v in blk["coefficients"].split()]
            return ("formula", lambda lam, c=coeffs, fm=formula: sellmeier_formula(lam, c, fm))
    raise ValueError("no tabulated/formula DATA found in YAML")


def register_yaml(name, path):
    """Register a refractiveindex.info YAML as a material (table or formula)."""
    kind, *rest = load_nk_yaml(path)
    if kind == "table":
        register_table(name, *rest)
    else:
        MATERIALS[name] = lambda lam, fn=rest[0]: fn(lam)
    return kind


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
