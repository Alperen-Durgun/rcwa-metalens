#!/bin/bash
# Meep OOM taraması: çözünürlüğü artır, OOM/hata olunca dur. Tavan = son OK res.
# Çalıştır (WSL, rcwa_py klasöründen):  wsl bash examples/meep_limit_sweep.sh
PY=$HOME/miniforge3/envs/mp/bin/python
D="../../Simulasyonlar/Gercek_Tasarim_2_HighNA_TiO2_633nm_D10f8_design.json"
for R in 12 14 16 18 20 22; do
  echo "===================== res=$R BASLIYOR ====================="
  "$PY" -u examples/meep_lens_focus.py "$D" "$R"
  if [ $? -ne 0 ]; then
    echo "!!! res=$R OOM/HATA -> MEEP TAVANI = bir onceki res !!!"
    break
  fi
  echo "===================== res=$R OK ====================="
done
echo "TARAMA-BITTI"
