"""Meep bridge mini-test — does fullwave.build_meep work with a real Meep install?
Small synthetic design (a few pillars), low resolution, short run. It only validates the bridge.

Run (WSL, mp environment):
  $HOME/miniforge3/envs/mp/bin/python examples/meep_bridge_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import fullwave

# tiny synthetic design: 3x3 dielectric pillars
L = 0.35
pl = [{"x": (i - 1) * L, "y": (j - 1) * L, "params": 0.15, "angle": 0.0}
      for i in range(3) for j in range(3)]
design = {"lam0": 0.633, "Lcell": L, "height": 0.40, "n_pillar": 2.4,
          "shape": "circle", "D_lens": 1.5, "f": 2.0, "placements": pl}

print("Meep bridge test: %d pillars, resolution=10, short run..." % len(pl))
import meep as mp
sim = fullwave.build_meep(design, resolution=10)
print("  build_meep OK — cell built.")
sim.run(until=12)
val = sim.get_field_point(mp.Ey, mp.Vector3(0, 0, 0))
print("  run finished. center |Ey| = %.4e" % abs(val))
print("✅ MEEP BRIDGE WORKS — local/offline 3D FDTD is ready.")
