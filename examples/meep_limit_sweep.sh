#!/bin/bash
# Meep OOM sweep: increase resolution, stop on OOM/error. Ceiling = last OK res.
# Run (WSL, from the rcwa_py folder):  wsl bash examples/meep_limit_sweep.sh
PY=$HOME/miniforge3/envs/mp/bin/python
D="../../Simulasyonlar/Gercek_Tasarim_2_HighNA_TiO2_633nm_D10f8_design.json"
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
