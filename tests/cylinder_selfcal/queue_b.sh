#!/bin/bash
# Run stream B (two streams of 12 threads share the cores): waits for the 3-camera corrected run, then V3's controls,
# the 2-camera corrected run and the Gallm runs (the latter once synth_sc.py has rendered Gallmw).
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
PY="C:/Users/zcloc/.conda/envs/refract/python.exe"
while ! grep -q "exit" sruns_Gcal4c3_n8_r3_tol1.0_out.txt 2>/dev/null; do sleep 30; done
$PY t1_runs.py control --tag m15vc --pre data/T1pre_s4_m15vc --tol2d 1.5 --threads 12 > t1_runs_control_m15vc_console.txt 2>&1
$PY t1_runs.py control --tag m15vc10 --pre data/T1pre_s4_m15vc --tol2d 1.0 --threads 12 > t1_runs_control_m15vc10_console.txt 2>&1
$PY sruns.py Gcal4c2_n8_r2_tol1.0 --threads 12 > sruns_Gcal4c2_n8_r2_tol1.0_console.txt 2>&1
while ! grep -q "noise measured after the warp" synth_sc_warped_Gallmw_out.txt 2>/dev/null; do sleep 30; done
$PY sruns.py Gallm0_r2_tol1.5 Gallmw_r2_tol1.5 Gallmc_r2_tol1.5 --threads 12 > sruns_gallm_b1_console.txt 2>&1
echo "stream B done"
