#!/bin/bash
# Run stream E, after stream C: Gallm at 1.0 px (the uncorrected, no error field, re-rendered, warped).
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
PY="C:/Users/zcloc/.conda/envs/refract/python.exe"
while ! grep -q "stream C done" queue_c_console.txt 2>/dev/null; do sleep 30; done
$PY sruns.py Gallm_r2_tol1.0 Gallm0_r2_tol1.0 Gallmc_r2_tol1.0 Gallmw_r2_tol1.0 --threads 12 > sruns_gallm_e_console.txt 2>&1
echo "stream E done"
