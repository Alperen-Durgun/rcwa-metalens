"""Dispersif malzeme kütüphanesi — A3.

Dalga boyuna bağlı kırılma indisi n(λ) ve soğurma k (kayıp). eps = (n + ik)^2.
λ mikron (um) cinsinden. Basit Sellmeier/tablolar; gerektikçe genişlet.
Kaynak değerleri yaklaşık literatür; kesin iş için ölçülü tablolarla değiştir.
"""
import numpy as np

def _sellmeier(lam, B, C):
    l2 = lam ** 2
    n2 = 1.0 + sum(b * l2 / (l2 - c) for b, c in zip(B, C))
    return np.sqrt(n2)

# Yaklaşık modeller (görünür bölge)
def n_SiO2(lam):   # erimiş silika (Malitson)
    return _sellmeier(lam, [0.6961663, 0.4079426, 0.8974794],
                      [0.0684043**2, 0.1162414**2, 9.896161**2])

def n_TiO2(lam):   # TiO2 (yaklaşık, görünür) — n~2.4 civarı
    return np.sqrt(5.913 + 0.2441 / (lam ** 2 - 0.0803))

def n_Si(lam):     # kristal Si (yaklaşık); görünürde soğurmalı
    n = 3.42 + 0.09 / (lam ** 2)
    k = np.where(lam < 0.5, 0.2, 0.02)   # kaba kayıp
    return n + 1j * k

def n_Ag(lam):     # gümüş (kaba Drude yaklaşımı, metal — yüksek kayıp)
    # sadece niteliksel; kesin iş için Johnson&Christy tablosu kullan
    eps = 5.0 - (9.2 ** 2) / ((1.2398 / lam) * ((1.2398 / lam) + 1j * 0.02))
    return np.sqrt(eps + 0j)

MATERIALS = {"SiO2": n_SiO2, "TiO2": n_TiO2, "Si": n_Si, "Ag": n_Ag, "air": lambda lam: 1.0 + 0j}

def eps_of(material, lam):
    """Malzemenin λ'daki karmaşık eps'i. material: isim ya da sabit n."""
    if isinstance(material, (int, float, complex)):
        return complex(material) ** 2
    n = MATERIALS[material](lam)
    return np.asarray(n) ** 2


# --- Ölçülü n,k tablo desteği (Madde 3) ---
import os as _os
_DATA_DIR = _os.path.join(_os.path.dirname(__file__), "materials_data")
_TABLES = {}

def load_nk_csv(path):
    """refractiveindex.info benzeri CSV: satırlar 'wavelength_um,n,k'. (# yorum)."""
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
    """Tablodan λ'da karmaşık kırılma indisi (n+ik), lineer interpolasyon."""
    wl, n, k = _TABLES[name]
    nn = np.interp(lam, wl, n); kk = np.interp(lam, wl, k)
    return nn + 1j * kk

# eps_of'u tablo isimlerini de tanıyacak şekilde genişlet
_eps_of_base = eps_of
def eps_of(material, lam):  # noqa: F811
    if isinstance(material, str) and material in _TABLES:
        return n_table(material, lam) ** 2
    return _eps_of_base(material, lam)

def list_materials():
    return sorted(list(MATERIALS.keys()) + list(_TABLES.keys()))


# --- E7: refractiveindex.info YAML içe aktarma + geniş-bant + anizotropik ---
def sellmeier_formula(lam, coeffs, formula=1):
    """refractiveindex.info Sellmeier (formula 1/2). coeffs: [C1,C2,...] (RI.info sırası).
    formula 1: n^2 = 1 + Σ Ci λ^2/(λ^2 - C(i+1)^2). λ mikron."""
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
    """refractiveindex.info .yml dosyasını içe aktar. 'tabulated nk/n' veya
    'formula' (Sellmeier) destekler. Döndürür: ('table',wl,n,k) ya da ('formula',fn)."""
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
    raise ValueError("YAML içinde tabulated/formula DATA bulunamadı")


def register_yaml(name, path):
    """Bir refractiveindex.info YAML'ını malzeme olarak kaydet (tablo ya da formül)."""
    kind, *rest = load_nk_yaml(path)
    if kind == "table":
        register_table(name, *rest)
    else:
        MATERIALS[name] = lambda lam, fn=rest[0]: fn(lam)
    return kind


def broadband_fit(wl, n, k, deg=6):
    """Ölçülü (wl,n,k) üstüne polinom geniş-bant modeli (MCM-benzeri).
    Döndürür: fn(lam)->n+ik. Solver'da hızlı, düzgün dispersiyon için."""
    cn = np.polyfit(wl, n, deg); ck = np.polyfit(wl, k, deg)
    return lambda lam, cn=cn, ck=ck: np.polyval(cn, lam) + 1j * np.polyval(ck, lam)


def eps_tensor(material, lam):
    """Anizotropik diagonal tensör eps = diag(exx,eyy,ezz).
    material: skaler isim/sayı (izotropik) VEYA {'xx':..,'yy':..,'zz':..} (bileşen malzemeler).
    Not: native RCWA çözücümüz izotropik ER kullanır; çift-kırılım GEOMETRİK sağlanır
    (bkz. birefringent_library). Bu fonksiyon tensör veri temsili + ileride kullanım içindir."""
    if isinstance(material, dict):
        return np.diag([complex(eps_of(material[a], lam)) for a in ("xx", "yy", "zz")])
    e = complex(eps_of(material, lam))
    return np.diag([e, e, e])
