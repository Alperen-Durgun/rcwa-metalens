#!/bin/bash
# Meep memory-limit sweep: raise the resolution step by step and stop at the first
# out-of-memory/error. The ceiling of this machine = the last resolution that finished.
#
# Usage (WSL / Linux, from the repository root, conda env with pymeep):
#   bash examples/meep_limit_sweep.sh path/to/<name>_design.json [python-executable]
# The design JSON is written by:  rcwa-metalens metalens_design_full ...
D="${1:?usage: meep_limit_sweep.sh <design.json> [python]}"
PY="${2:-python}"
for R in 12 14 16 18 20 22; do
  echo "===================== res=$R STARTING ====================="
  "$PY" -u examples/meep_lens_focus.py "$D" "$R"
  if [ $? -ne 0 ]; then
    echo "!!! res=$R OOM/ERROR -> MEEP CEILING = previous res !!!"
    break
  fi
  echo "===================== res=$R OK ====================="
done
echo "SWEEP-DONE"
