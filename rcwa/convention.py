"""Time / phase convention — READ THIS before using a solver phase in a design.

The RCWA solvers in this package (``solve_rcwa_1d``, ``solve_rcwa_2d``,
``solve_scalar_1d``) follow the *engineering* time convention ``exp(+j*omega*t)``
(Rumpf's formulation). In that convention a wave travelling toward +z is
``exp(-j*k*z)``, so the transmission phase of a meta-atom becomes MORE NEGATIVE
as its optical delay grows.

Almost everything a metalens designer combines the solver with uses the
*physics* convention ``exp(-i*omega*t)`` instead:

* the hyperbolic lens profile  phi(r) = -(2*pi/lam) * (sqrt(r^2 + f^2) - f),
* the angular-spectrum propagator  exp(+i*kz*z)  (``metrics.angular_spectrum``),
* FDTD tools (Tidy3D, Meep).

The two conventions are related by complex conjugation. Reflectance,
transmittance and diffraction efficiencies are identical in both; only complex
amplitudes / phases differ. Using a solver phase directly in a physics-convention
design yields a DIVERGING lens (verified against FDTD in 2D and against a
full super-cell RCWA solve in 1D — see ``tests/test_rcwa.py``).

Rule: convert once, right after the solve, with ``to_physical`` /
``physical_phase``, and do all design work in the physics convention.
"""
import numpy as np

TIME_CONVENTION = "exp(+j*omega*t)"   # convention of the raw solver amplitudes


def to_physical(t):
    """Solver complex amplitude(s) -> physics convention exp(-i*omega*t)."""
    return np.conj(t)


def physical_phase(t):
    """Phase of solver amplitude(s) in the physics convention (grows with optical delay)."""
    return -np.angle(t)


def propagate_orders_1d(res, lam0, period, x, z, kx_inc=0.0, n_trn=1.0, component="ty"):
    """Physical-convention field of the transmitted orders of a 1D solve.

    res      : result dict of ``solve_rcwa_1d`` (uses res["orders"], res[component])
    x, z     : lateral positions (array, um) and distance behind the structure (um)
    kx_inc   : normalized incident kx (n_ref*sin(theta)*cos(phi)); 0 at normal incidence
    Returns  : complex field E(x) at distance z (evanescent orders decay).
    """
    m = np.asarray(res["orders"])
    t = np.conj(np.asarray(res[component]))
    kx = kx_inc - m * (lam0 / period)
    kz = np.sqrt((n_trn ** 2 - kx ** 2).astype(complex))
    kz = np.where(np.imag(kz) < 0, -kz, kz)
    k0 = 2 * np.pi / lam0
    x = np.atleast_1d(np.asarray(x, float))
    return np.exp(1j * k0 * (np.outer(x, kx) + z * kz[None, :])) @ t
