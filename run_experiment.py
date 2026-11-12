"""Launcher for the experiment-note workflow when working from a clone.
The implementation lives in rcwa/experiments.py.

    python run_experiment.py --all                 # run all pending experiment notes
    python run_experiment.py "path/to/note.md"     # a single note
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcwa.experiments import *          # noqa: F401,F403  (kept for scripts that import this module)
from rcwa.experiments import RUNNERS, PROJECT, DENEYLER, RAPORLAR, SIM, main, num  # noqa: F401

if __name__ == "__main__":
    main(sys.argv)
