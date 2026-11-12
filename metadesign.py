"""E9 — LIGHTWEIGHT CLI LAUNCHER (outside Obsidian / quick use).

Instead of a GUI: run the automation generators from the terminal. The Obsidian
automation (form→run→report) is the main interface; this is for quick/scripted use.

Usage (inside the .venv, from the repository root):
  python metadesign.py --list                       # list the experiment types
  python metadesign.py metalens_design_full D_lens=6 f=6 material=TiO2_t
  python metadesign.py grating period=1.2 duty=0.5
  python metadesign.py --menu                        # interactive menu
"""
import sys, os
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _parse_kv(args):
    fm = {}
    for a in args:
        if "=" in a:
            k, v = a.split("=", 1)
            fm[k] = v   # run_experiment.num() converts to numbers; string fields stay as-is
    return fm


def main(argv):
    import run_experiment as R
    names = list(R.RUNNERS.keys())
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); print("Types:", ", ".join(names)); return 0
    if argv[0] == "--list":
        print("Available experiment types (%d):" % len(names))
        for n in names:
            print("  -", n)
        return 0
    if argv[0] == "--menu":
        for i, n in enumerate(names):
            print(f"[{i:2d}] {n}")
        try:
            sel = int(input("Select (number): ").strip()); name = names[sel]
        except Exception:
            print("Invalid selection."); return 1
        raw = input("Parameters (key=val separated by spaces, empty=defaults): ").strip()
        fm = _parse_kv(raw.split()) if raw else {}
    else:
        name = argv[0]
        if name not in R.RUNNERS:
            print(f"Unknown type: {name}\nTypes: {', '.join(names)}"); return 1
        fm = _parse_kv(argv[1:])
    slug = "CLI_" + name
    print(f"Running: {name}  parameters={fm or 'defaults'}")
    fig, body, summary = R.RUNNERS[name](fm, slug)
    print("\n=== RESULT ===\n" + body)
    print(f"\nSummary: {summary}")
    print(f"Figure: {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
