"""Meep köprüsü mini-testi — fullwave.build_meep gerçek Meep ile çalışıyor mu?
Küçük yapay tasarım (birkaç sütun), düşük çözünürlük, kısa koşu. Sadece köprüyü doğrular.

Çalıştır (WSL, mp ortamı):
  $HOME/miniforge3/envs/mp/bin/python examples/meep_bridge_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from rcwa import fullwave

# minik yapay tasarım: 3x3 dielektrik sütun
L = 0.35
pl = [{"x": (i - 1) * L, "y": (j - 1) * L, "params": 0.15, "angle": 0.0}
      for i in range(3) for j in range(3)]
design = {"lam0": 0.633, "Lcell": L, "height": 0.40, "n_pillar": 2.4,
          "shape": "circle", "D_lens": 1.5, "f": 2.0, "placements": pl}

print("Meep köprüsü testi: %d sütun, çözünürlük=10, kısa koşu..." % len(pl))
import meep as mp
sim = fullwave.build_meep(design, resolution=10)
print("  build_meep OK — hücre boyutu kuruldu.")
sim.run(until=12)
val = sim.get_field_point(mp.Ey, mp.Vector3(0, 0, 0))
print("  koşu bitti. merkez |Ey| = %.4e" % abs(val))
print("✅ MEEP KÖPRÜSÜ ÇALIŞIYOR — yerel/çevrimdışı 3B FDTD hazır.")
