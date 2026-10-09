#!/bin/bash
# Run stream A (two streams of 12 threads share the cores): waits for the 4-camera corrected run, then V3's finals.
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
PY="C:/Users/zcloc/.conda/envs/refract/python.exe"
while ! grep -q "exit" sruns_Gcal4c4_n8_r4_tol1.0_out.txt 2>/dev/null; do sleep 30; done
$PY t1_runs.py final --tag m15vc --pre data/T1pre_s4_m15vc --tol2d 1.5 --threads 12 > t1_runs_final_m15vc_console.txt 2>&1
$PY t1_runs.py final --tag m15vc10 --pre data/T1pre_s4_m15vc --tol2d 1.0 --threads 12 > t1_runs_final_m15vc10_console.txt 2>&1
echo "stream A done"
