"""E2 — GPU-READY BACKEND selection.

Our RCWA core is numpy/CPU. For GPU + automatic differentiation there are two real paths:
  1) grcwa 'jax' backend  -> if jax-GPU is installed, gradients run on GPU (adjoint inverse design).
  2) cupy                  -> numpy-compatible; forward solve on GPU (no autodiff).

This module reports which accelerators are available and selects a backend suitable for adjoint.
If there is no hardware/installation it falls back automatically to autograd/CPU (code-ready, no crash).

Installation (user, optional):
  pip install "jax[cuda12]"   # NVIDIA GPU + autodiff (grcwa jax backend)
  pip install cupy-cuda12x    # numpy-compatible GPU arrays
"""

def available():
    """Which accelerators are installed? -> dict."""
    out = {"autograd": False, "jax": False, "jax_gpu": False, "cupy": False, "torch": False, "torch_gpu": False}
    try:
        import autograd; out["autograd"] = True  # noqa: F401  (availability probe)
    except Exception: pass
    try:
        import jax; out["jax"] = True
        out["jax_gpu"] = any(d.platform == "gpu" for d in jax.devices())
    except Exception: pass
    try:
        import cupy; out["cupy"] = cupy.cuda.runtime.getDeviceCount() > 0
    except Exception: pass
    try:
        import torch; out["torch"] = True; out["torch_gpu"] = torch.cuda.is_available()
    except Exception: pass
    return out


def pick_grcwa_backend(prefer_gpu=True):
    """Select a grcwa backend for adjoint: jax (if GPU) else autograd.
    Returns: (backend_name, is_gpu)."""
    av = available()
    if prefer_gpu and av["jax"]:
        return "jax", av["jax_gpu"]
    return "autograd", False


def get_xp(prefer_gpu=True):
    """Array module for forward (gradient-free) computation: cupy (GPU) else numpy.
    Usage: xp = get_xp(); a = xp.zeros(...)  -> code is identical to numpy."""
    if prefer_gpu:
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() > 0:
                return cp, "cupy-gpu"
        except Exception:
            pass
    import numpy as np
    return np, "numpy-cpu"


def summary():
    av = available()
    xp, xpname = get_xp()
    gb, gpu = pick_grcwa_backend()
    return (f"Accelerators: {av}\n"
            f"Forward array module: {xpname}\n"
            f"Adjoint backend: {gb} (GPU={gpu})")


if __name__ == "__main__":
    print(summary())
