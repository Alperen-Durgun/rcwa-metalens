"""Örnek 12 — Li ters kural (inverse rule): TM yakınsama hızlandırma (A2).

Yüksek-kontrastlı grating'te TM geçişini mertebe sayısına (M) karşı çizer:
'laurent' (standart) yavaş, 'inverse' (Li 1996) çok hızlı yakınsar — ikisi de
aynı doğru değere. Enerji korunumu R+T=1 ile doğrulanmıştır.
Çalıştır:  python examples/12_tm_inverse_rule.py
"""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import solve_scalar_1d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

lam0, period, d, eps_hi = 0.55, 1.0, 0.25, 9.0
x = (np.arange(2048) + 0.5) / 2048; eps = np.where(x < 0.5, eps_hi, 1.0)
Ms = [6, 10, 16, 24, 40, 60, 90]
Tl = [solve_scalar_1d(lam0, 0, 1, 1, period, eps, d, M, "TM", "laurent")["Ttot"] for M in Ms]
Ti = [solve_scalar_1d(lam0, 0, 1, 1, period, eps, d, M, "TM", "inverse")["Ttot"] for M in Ms]
ref = Ti[-1]
print(f"Yakınsak TM T ≈ {ref:.4f}")
print("M    laurent   inverse")
for M, a, b in zip(Ms, Tl, Ti):
    print(f"{M:3d}  {a:.4f}    {b:.4f}")

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(Ms, Tl, "o-", label="Laurent (standart)")
ax.plot(Ms, Ti, "s-", label="Inverse rule (Li 1996)")
ax.axhline(ref, color="gray", ls=":", label="yakınsak değer")
ax.set_xlabel("Mertebe sayısı M"); ax.set_ylabel("TM geçiş T")
ax.set_title(f"TM yakınsama — yüksek kontrast (ε={eps_hi:.0f}) grating"); ax.legend(); ax.grid(alpha=.3)
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "Simulasyonlar", "tm_inverse_rule.png"))
fig.tight_layout(); fig.savefig(out, dpi=130); print("Grafik ->", out)
