"""E2 — GPU-HAZIR BACKEND seçimi.

RCWA çekirdeğimiz numpy/CPU. GPU + otomatik türev için iki gerçek yol:
  1) grcwa 'jax' backend'i  -> jax-GPU kuruluysa gradyanlar GPU'da (adjoint ters tasarım).
  2) cupy                    -> numpy uyumlu; ileri çözüm GPU'da (autodiff yok).

Bu modül hangi hızlandırıcının mevcut olduğunu bildirir ve adjoint'e uygun backend'i seçtirir.
Donanım/kurulum yoksa otomatik olarak autograd/CPU'ya düşer (kod-hazır, kilitlenmesiz).

Kurulum (kullanıcı, isteğe bağlı):
  pip install "jax[cuda12]"   # NVIDIA GPU + autodiff (grcwa jax backend)
  pip install cupy-cuda12x    # numpy-uyumlu GPU dizilimi
"""

def available():
    """Hangi hızlandırıcılar kurulu? -> sözlük."""
    out = {"autograd": False, "jax": False, "jax_gpu": False, "cupy": False, "torch": False, "torch_gpu": False}
    try:
        import autograd; out["autograd"] = True
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
    """adjoint için grcwa backend'i seç: jax(GPU varsa) yoksa autograd.
    Döndürür: (backend_adı, gpu_mu)."""
    av = available()
    if prefer_gpu and av["jax"]:
        return "jax", av["jax_gpu"]
    return "autograd", False


def get_xp(prefer_gpu=True):
    """İleri (gradyansız) hesap için dizilim modülü: cupy(GPU) yoksa numpy.
    Kullanım: xp = get_xp(); a = xp.zeros(...)  -> kod numpy ile aynı."""
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
    return (f"Hızlandırıcılar: {av}\n"
            f"İleri hesap dizilimi: {xpname}\n"
            f"Adjoint backend: {gb} (GPU={gpu})")


if __name__ == "__main__":
    print(summary())
