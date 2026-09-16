"""E1 — ADJOINT / AUTODIFF TERS TASARIM (türevlenebilir RCWA).

Modern ters tasarımın standardı: hedef fonksiyonun tüm tasarım parametrelerine
gradyanını, adjoint (otomatik türev) ile ~tek çözümde hesaplayıp gradyan-tabanlı
optimize etmek. Burada doğrulanmış `grcwa` motoru autograd backend'iyle kullanılır;
gradyan sonlu-fark ile bire bir örtüşür (test_adjoint).

Aynı çerçeve deflektör, ışın-ayırıcı, absorber, metalens-hücresi vb. her
grcwa-ifade-edilebilir hedefe uygulanır. GPU için bkz. backends.py (torch/jax).

Bağımlılık: grcwa, autograd (ikisi de kurulu).
"""
import numpy as _np


def _agnp(prefer_gpu=True):
    import grcwa
    try:
        from .backends import pick_grcwa_backend
        bk, _gpu = pick_grcwa_backend(prefer_gpu=prefer_gpu)
    except Exception:
        bk = "autograd"
    grcwa.set_backend(bk)   # 'jax' (GPU+autodiff) varsa onu, yoksa 'autograd' (CPU)
    if bk == "jax":
        import jax.numpy as anp
    else:
        import autograd.numpy as anp
    return anp


def rcwa_TR(dof, cfg, anp=None):
    """Desenli tek katman için (R, T) döndürür — türevlenebilir.
    dof: [0,1] yoğunluk (Nx*Ny), eps = eps_lo + dof*(eps_hi-eps_lo).
    cfg: {nG, Lx, Ly, Nx, Ny, lam, thick, eps_hi, eps_lo, eps_sup, eps_sub, theta, phi}"""
    import grcwa
    if anp is None:
        anp = _agnp()
    eps = cfg["eps_lo"] + dof * (cfg["eps_hi"] - cfg["eps_lo"])
    obj = grcwa.obj(cfg["nG"], [cfg["Lx"], 0.0], [0.0, cfg["Ly"]],
                    1.0 / cfg["lam"], cfg.get("theta", 0.0), cfg.get("phi", 0.0), verbose=0)
    obj.Add_LayerUniform(1.0, cfg.get("eps_sup", 1.0))
    obj.Add_LayerGrid(cfg["thick"], cfg["Nx"], cfg["Ny"])
    obj.Add_LayerUniform(1.0, cfg.get("eps_sub", 1.0))
    obj.Init_Setup()
    obj.MakeExcitationPlanewave(cfg.get("p_amp", 1.0), 0.0, cfg.get("s_amp", 0.0), 0.0, order=0)
    obj.GridLayer_geteps(eps.flatten())
    R, T = obj.RT_Solve(normalize=1)
    return R, T


def absorption(dof, cfg, anp=None):
    """Absorpsiyon A = 1 - R - T (kayıplı eps_hi ile). Türevlenebilir hedef."""
    R, T = rcwa_TR(dof, cfg, anp)
    return 1.0 - R - T


def adam(fun, x0, steps=60, lr=0.05, maximize=True, clip=(0.0, 1.0), seed=0):
    """Basit Adam optimizeri (autograd.grad ile). fun: skaler hedef.
    Döndürür: (x_final, history[list]). clip ile [0,1] projeksiyonu."""
    from autograd import grad
    g = grad(fun)
    x = _np.array(x0, float)
    m = _np.zeros_like(x); v = _np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    sgn = 1.0 if maximize else -1.0
    hist = []
    for t in range(1, steps + 1):
        val = float(fun(x)); hist.append(val)
        gr = _np.array(g(x)) * sgn
        m = b1 * m + (1 - b1) * gr
        v = b2 * v + (1 - b2) * gr * gr
        mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
        x = x + lr * mh / (_np.sqrt(vh) + eps)
        if clip is not None:
            x = _np.clip(x, clip[0], clip[1])
    hist.append(float(fun(x)))
    return x, hist


def optimize_absorber(cfg, steps=60, lr=0.05, seed=0):
    """Serbest-biçim metasurface absorber topoloji optimizasyonu (adjoint).
    Döndürür: {pattern(Nx,Ny), A0, A_final, history}."""
    anp = _agnp()
    rng = _np.random.default_rng(seed)
    x0 = rng.uniform(0.35, 0.65, cfg["Nx"] * cfg["Ny"])
    fun = lambda d: absorption(d, cfg, anp)
    A0 = float(fun(x0))
    xf, hist = adam(fun, x0, steps=steps, lr=lr, maximize=True)
    return {"pattern": xf.reshape(cfg["Nx"], cfg["Ny"]), "A0": A0,
            "A_final": float(fun(xf)), "history": hist}


def fd_grad_check(cfg, seed=1, h=1e-4):
    """Gradyan doğrulaması: adjoint vs sonlu-fark (tek DOF). test için."""
    from autograd import grad
    anp = _agnp()
    rng = _np.random.default_rng(seed)
    x = rng.uniform(0.3, 0.7, cfg["Nx"] * cfg["Ny"])
    fun = lambda d: absorption(d, cfg, anp)
    ga = _np.array(grad(fun)(x))
    x2 = x.copy(); x2[0] += h
    gfd = (fun(x2) - fun(x)) / h
    return float(ga[0]), float(gfd), float(abs(ga[0] - gfd))
