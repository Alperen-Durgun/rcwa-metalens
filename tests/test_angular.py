"""Angle-dependent library (roadmap P2): angle mapping, diffraction orders, oblique complex
transmission against an analytic thin-film transfer matrix, theta = 0 against the normal
library, a cross-solver check with grcwa, and the off-axis comparison bookkeeping."""
import numpy as np
import pytest

from rcwa import angular, design

LAM, L, NSUB = 0.633, 0.375, 1.457


def tmm_tangential(n_in, layers, n_out, lam, kx, pol):
    """Analytic transmission of the tangential E field (out / in) of a thin-film stack,
    physics convention exp(-i w t) with waves exp(+i kz z). Admittance Y = kz (TE), n^2 k0^2 / kz (TM)."""
    k0 = 2 * np.pi / lam
    kz = lambda n: np.sqrt((n * k0) ** 2 - kx ** 2 + 0j)                                   # noqa: E731
    Y = lambda n: kz(n) if pol == "TE" else (n * k0) ** 2 / kz(n)                          # noqa: E731
    Mt = np.eye(2, dtype=complex)
    for n, d in layers:
        q = kz(n) * d; y = Y(n)
        Mt = Mt @ np.array([[np.cos(q), -1j * np.sin(q) / y], [-1j * y * np.sin(q), np.cos(q)]])
    B, C = Mt @ np.array([1.0, Y(n_out)])
    return 2 * Y(n_in) / (Y(n_in) * B + C)


def test_tmm_reference_is_physical():
    """The reference itself: an empty layer gives the propagation phase exp(+i kz d)."""
    kx = 0.3 * 2 * np.pi / LAM; d = 0.4
    kz = np.sqrt((2 * np.pi / LAM) ** 2 - kx ** 2)
    for pol in ("TE", "TM"):
        assert abs(tmm_tangential(1.0, [(1.0, d)], 1.0, LAM, kx, pol) - np.exp(1j * kz * d)) < 1e-14


def test_substrate_angle_and_orders():
    assert angular.substrate_angle(0.0, NSUB) == 0.0
    assert angular.substrate_angle(30.0, NSUB) == pytest.approx(np.rad2deg(np.arcsin(0.5 / NSUB)))
    a_sub = angular.first_order_angle(LAM, L, NSUB); a_out = angular.first_order_angle(LAM, L, 1.0)
    assert a_sub == pytest.approx(np.rad2deg(np.arcsin(LAM / L - NSUB))) and 13.0 < a_sub < 14.0
    assert a_out == pytest.approx(np.rad2deg(np.arcsin(LAM / L - 1.0))) and 43.0 < a_out < 44.0
    assert angular.orders_at(10.0, LAM, L, NSUB) == {"substrate": [], "exit": []}
    o = angular.orders_at(20.0, LAM, L, NSUB)
    assert o["substrate"] == [(1, 0)] and o["exit"] == []
    assert angular.first_order_angle(LAM, 0.2, NSUB) is None           # lam0/L - n_sub > 1: never


@pytest.mark.parametrize("pol", ["TE", "TM"])
@pytest.mark.parametrize("theta", [0.0, 20.0, 35.0])
def test_oblique_slab_matches_transfer_matrix(pol, theta):
    """A square pillar of side L fills the cell: a uniform film. The oblique complex t of the
    library worker (angle mapping, polarization normalisation, convention) equals the
    analytic thin-film value."""
    nf, d = 2.0, 0.3
    th_sub = angular.substrate_angle(theta, NSUB)
    job = (LAM, L, d, "square", L, 64, complex(nf), 1.0, NSUB, 1, (90.0, 1, None), th_sub, pol)
    t_raw, T, R = angular._atom_oblique(job)
    from rcwa.convention import to_physical
    kx = 2 * np.pi / LAM * np.sin(np.deg2rad(theta))
    ta = tmm_tangential(NSUB, [(nf, d)], 1.0, LAM, kx, pol)
    assert abs(to_physical(t_raw) - ta) < 1e-10
    assert abs(T + R - 1) < 1e-10                                       # lossless film


def test_angle_library_at_normal_incidence_equals_the_library():
    sizes = np.array([0.10, 0.18, 0.25])
    lib = design.meta_atom_library(LAM, L, 0.6, sizes=sizes, material="TiO2_Siefke", M=5, n_sub=NSUB)
    a = angular.angle_library(lib, [0.0], pol="TE")
    assert np.max(np.abs(a["t"][0] - lib["t"])) < 1e-12
    b = angular.angle_library(lib, [0.0], pol="TM")                     # circle: x and y equivalent
    assert np.max(np.abs(b["t"][0] - lib["t"])) < 1e-9
    assert a["t"].shape == (1, 3) and a["pol"] == "TE" and a["M"] == 5
    with pytest.raises(ValueError):
        angular.angle_library(lib, [0.0], pol="X")


def test_oblique_pillar_matches_grcwa_power():
    """Cross-solver check at 20 deg (TE): total transmission of a TiO2 pillar array on silica
    agrees with grcwa (independent RCWA code) within the difference both show at 0 deg."""
    grcwa = pytest.importorskip("grcwa")
    n = design.pillar_index("TiO2_Siefke", LAM); s = 0.25; H = 0.6; N = 256
    for theta in (0.0, 20.0):
        th_sub = angular.substrate_angle(theta, NSUB)
        _, T, R = angular._atom_oblique((LAM, L, H, "circle", s, N, n, 1.0, NSUB, 9, (90.0, 1, None), th_sub, "TE"))
        lay = design.pillar_layers("circle", s, L, H, N, n, 1.0)
        ob = grcwa.obj(361, [L, 0], [0, L], 1 / LAM, np.deg2rad(th_sub), 0.0, verbose=0)
        ob.Add_LayerUniform(1.0, NSUB ** 2); ob.Add_LayerGrid(H, N, N); ob.Add_LayerUniform(1.0, 1.0)
        ob.Init_Setup(); ob.MakeExcitationPlanewave(0.0, 0.0, 1.0, 0.0, order=0)
        ob.GridLayer_geteps(lay[0][0].flatten())
        Rg, Tg = ob.RT_Solve(normalize=1)
        assert abs(T - float(np.real(Tg))) < 0.012, (theta, T, Tg)
        assert abs(T + R - 1) < 1e-6


def test_fov_compare_bookkeeping():
    """With an angle library whose response does not change with angle, the angle-dependent
    model equals the former fixed-response model exactly; the normal-incidence peak is 1."""
    sizes = np.linspace(0.06, 0.315, 8)
    lib = design.meta_atom_library(LAM, L, 0.6, sizes=sizes, material="TiO2_Siefke", M=3, n_sub=NSUB)
    d = design.design_lens(LAM, L, 0.6, 3.0, 6.0, lib=lib, refine=False, mode="complex", n_sub=NSUB)
    used = np.unique(np.round(d["size_map"][d["inside"]], 12))
    angles = np.array([0.0, 5.0, 10.0])
    k = [int(np.argmin(np.abs(lib["sizes"] - u))) for u in used]           # exact library values
    t_used = lib["t"][k]
    alib = {"sizes": used, "t": np.tile(t_used, (3, 1)), "T": np.ones((3, len(used))), "angles_deg": angles}
    r = angular.fov_compare(d, angles, alib=alib)
    assert r["atoms"]["rel_peak"][0] == pytest.approx(1.0, abs=1e-12)
    for key in ("rel_peak", "x_peak", "fwhm"):
        assert np.allclose(r["atoms"][key], r["fixed"][key], atol=1e-12, rtol=0)
    assert r["atoms"]["x_peak"][2] > r["atoms"]["x_peak"][1] > 0        # the spot moves with the angle
    bad = {**alib, "sizes": used[:-1], "t": alib["t"][:, :-1]}
    with pytest.raises(ValueError):
        angular.fov_compare(d, angles, alib=bad)
