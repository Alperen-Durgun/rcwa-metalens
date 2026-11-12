"""Example 13 — Verify the design with 3D FULL-WAVE (CHOOSE the backend) — Item 1.

Takes the `*_design.json` produced by metalens_design_full and sets up a 3D FDTD
verification with the backend of your choice. Two options:
  BACKEND = "tidy3d"  -> GPU cloud (requires an account/API key + internet; very fast)
  BACKEND = "meep"    -> local/offline FDTD (installed via conda; heavy)

Install:
  Tidy3D: pip install tidy3d   (then set the API key with: tidy3d configure)
  Meep:   conda install -c conda-forge pymeep

Run:  python examples/13_fullwave_verify.py  <design.json>
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcwa import fullwave

BACKEND = "tidy3d"   # "tidy3d" or "meep" — YOUR CHOICE

def main(path):
    design = json.load(open(path))
    print(f"Design: {design['shape']} / {design.get('material','?')}, "
          f"{len(design['placements'])} pillars, D={design['D_lens']}um f={design['f']}um")
    print(f"Backend: {BACKEND}")
    if BACKEND == "tidy3d":
        from tidy3d import web
        import numpy as np
        sim = fullwave.build_tidy3d(design, run_time=2.5e-13)  # long enough for the focus
        ng = len(sim.structures[0].geometry.geometries)
        print(f"Tidy3D Simulation built: {ng} pillars, size={tuple(round(x,1) for x in sim.size)}um.")
        job = web.Job(simulation=sim, task_name="metalens_verify")
        cost = job.estimate_cost()   # free
        print(f"*** Estimated cost: {cost:.2f} FlexCredit ***  (check that you have enough credits!)")
        if len(sys.argv) > 2 and sys.argv[2] == "run":
            data = job.run()          # SPENDS credits
            fd = data["focus"]
            I = (np.abs(fd.Ex.values) ** 2 + np.abs(fd.Ey.values) ** 2 + np.abs(fd.Ez.values) ** 2).squeeze()
            xs = np.asarray(fd.Ex.coords["x"].values)
            j = np.unravel_index(np.argmax(I), I.shape)
            cut = I[:, j[1]] / I.max()
            hw = xs[cut >= 0.5]; fwhm = (hw.max() - hw.min()) * 1000
            print(f"\nFDTD focal plane (z=f={design['f']}um):")
            print(f"  Peak position x={xs[j[0]]:.2f}um  |  FWHM={fwhm:.0f}nm")
            NA = np.sin(np.arctan((design['D_lens'])/2/design['f']))
            print(f"  Diffraction limit λ/(2NA)={design['lam0']/(2*NA)*1000:.0f}nm")
            print("  (RCWA prediction: focus z=24.8um, FWHM=788nm) -> compare")
        else:
            print("To run (spends credits): python examples/13_fullwave_verify.py <json> run")
    else:
        fullwave.run_fullwave(design, backend="meep", resolution=20, until=150)
        print("Meep FDTD finished; post-process the field to extract the focus (see examples/11).")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/13_fullwave_verify.py <design.json>"); sys.exit(1)
    main(sys.argv[1])
