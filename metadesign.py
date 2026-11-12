"""Launcher for the command-line interface when working from a clone (no install needed).
Same arguments as the installed ``rcwa-metalens`` command — see rcwa/cli.py.

    python metadesign.py list
    python metadesign.py help metalens_design_full
    python metadesign.py metalens_design_full D_lens=16 f=20 material=TiO2_t
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcwa.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
