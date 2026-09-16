"""E3 — 2B YÜKSEK-DOĞRULUK ÇÖZÜCÜ (doğru Fourier faktörizasyonu).

Bizim native `solve_rcwa_2d` düz Laurent kuralı kullanır: DİELEKTRİK yapılarda
hızlı ve doğru, ama METAL/yüksek-kontrast 2B yapılarda yakınsaması kötüdür
(enerji korunmaz, ΣDE≫1). Doğru Fourier faktörizasyonu (Li kuralları) gerektiren
bu durumlar için, doğrulanmış `grcwa` motorunu birinci-sınıf yüksek-doğruluk
backend'i olarak köprüleriz.

- `solve_2d_grcwa(...)` : grcwa ile (Rtot, Ttot, absorpsiyon, per-order R/T).
- `factorization_advice(eps2d)` : malzeme metalik/yüksek-kontrastsa uyarır ve
  hangi backend'in uygun olduğunu söyler.
- `energy_ok(...)` : Laurent çözümünün enerjisini kontrol eder (>1+tol -> güvenilmez).

Tam normal-vektör FFF (grcwa'nın da ötesi) kapsam dışı; en zorlu metal-2B için
nihai doğrulama yine FDTD (bkz. fullwave.py).
"""
import numpy as np


def solve_2d_grcwa(lam0, theta_deg, phi_deg, er_ref, er_trn, Lx, Ly, eps2d, H, nG=200):
    """grcwa ile 2B çöz (doğru faktörizasyon). Döndürür: Rtot,Ttot,A,per-order.
    eps2d: (Nx,Ny) karmaşık permittivite. nG: tutulan düzlem-dalga sayısı (grcwa yuvarlar)."""
    import grcwa
    grcwa.set_backend("numpy")
    eps2d = np.asarray(eps2d, complex)
    Nx, Ny = eps2d.shape
    ob = grcwa.obj(nG, [Lx, 0.0], [0.0, Ly], 1.0 / lam0, theta_deg, phi_deg, verbose=0)
    ob.Add_LayerUniform(1.0, er_ref)
    ob.Add_LayerGrid(H, Nx, Ny)
    ob.Add_LayerUniform(1.0, er_trn)
    ob.Init_Setup()
    ob.MakeExcitationPlanewave(1.0, 0.0, 0.0, 0.0, order=0)
    ob.GridLayer_geteps(eps2d.flatten())
    R, T = ob.RT_Solve(normalize=1)
    return {"Rtot": float(np.real(R)), "Ttot": float(np.real(T)),
            "absorption": float(np.real(1.0 - R - T)), "nG": int(ob.nG), "backend": "grcwa"}


def solve_2d_lattice(lam0, theta_deg, phi_deg, er_ref, er_trn, L1, L2, eps2d, H, nG=200):
    """E4 — GENEL (non-ortogonal) KAFES. L1,L2 keyfi kafes vektörleri (um).
    Altıgen örgü: L1=[a,0], L2=[a/2, a*sqrt(3)/2]. Native solver2d yalnız dikdörtgen
    yapabilir; bu köprü grcwa ile keyfi kafesi çözer (doğru faktörizasyon dahil)."""
    import grcwa
    grcwa.set_backend("numpy")
    eps2d = np.asarray(eps2d, complex); Nx, Ny = eps2d.shape
    ob = grcwa.obj(nG, list(L1), list(L2), 1.0 / lam0, theta_deg, phi_deg, verbose=0)
    ob.Add_LayerUniform(1.0, er_ref)
    ob.Add_LayerGrid(H, Nx, Ny)
    ob.Add_LayerUniform(1.0, er_trn)
    ob.Init_Setup()
    ob.MakeExcitationPlanewave(1.0, 0.0, 0.0, 0.0, order=0)
    ob.GridLayer_geteps(eps2d.flatten())
    R, T = ob.RT_Solve(normalize=1)
    return {"Rtot": float(np.real(R)), "Ttot": float(np.real(T)),
            "absorption": float(np.real(1 - R - T)), "nG": int(ob.nG),
            "lattice": "non-ortogonal" if abs(np.dot(L1, L2)) > 1e-9 else "ortogonal"}


def hex_lattice(a):
    """Altıgen örgü kafes vektörleri (kenar a)."""
    return [a, 0.0], [a / 2.0, a * np.sqrt(3) / 2.0]


def factorization_advice(eps2d):
    """Malzemeye bakıp uygun 2B backend'i önerir.
    Metal (Re(eps)<0) veya yüksek kontrast -> grcwa; aksi halde native Laurent yeterli."""
    e = np.asarray(eps2d, complex)
    has_metal = bool(np.any(np.real(e) < 0.0))
    contrast = float(np.max(np.abs(e)) / max(np.min(np.abs(e)), 1e-9))
    lossy = bool(np.any(np.imag(e) > 1e-3))
    if has_metal or contrast > 8.0:
        rec = "grcwa (solve_2d_grcwa) — doğru Fourier faktörizasyonu"
        why = ("metal (Re ε<0)" if has_metal else f"yüksek kontrast (×{contrast:.0f})")
    else:
        rec = "native solve_rcwa_2d (Laurent) yeterli"
        why = "dielektrik, düşük kontrast"
    return {"recommend": rec, "reason": why, "has_metal": has_metal,
            "contrast": contrast, "lossy": lossy}


def energy_ok(result, tol=0.02):
    """Bir 2B çözümün enerjisi fiziksel mi? Kayıpsızsa ΣDE≈1; kayıplıysa ≤1.
    ΣDE>1+tol -> faktörizasyon güvenilmez (Laurent metalde patlar)."""
    s = result["Rtot"] + result["Ttot"]
    return bool(s <= 1.0 + tol), float(s)
