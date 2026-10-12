#!/bin/bash
# Stream 1 as it was run: L2 (T1 with the library correction = the self-calibration's two-pass field, 1.0 px; final
# and control), then L3 (the synthetic sets, 2/3/4 cameras) and its scores. 12 threads.
cd "$(dirname "$0")"
PY="C:/Users/zcloc/.conda/envs/refract/python.exe"
export PYTHONDONTWRITEBYTECODE=1
CASE=D:/Refraction/pyOpenLPT/cases/cyl_20261007/data/runs
$PY t1_lib_runs.py final --tag lc2 --base $CASE/s3_final_m15t10/config_final_m15t10.json \
    --corr data/fields/t1_vc2_total_cam1.txt data/fields/t1_vc2_total_cam2.txt --tol2d 1.0 > l2_final_console.txt 2>&1
$PY t1_lib_runs.py control --tag lc2 --base $CASE/s3_final_m15t10/config_final_m15t10.json \
    --corr data/fields/t1_vc2_total_cam1.txt data/fields/t1_vc2_total_cam2.txt --tol2d 1.0 > l2_control_console.txt 2>&1
for n in 2 3 4; do $PY l3_synthetic.py run $n > l3_run_r${n}_console.txt 2>&1; done
$PY l3_synthetic.py score > l3_score_console.txt 2>&1
echo "stream 1 done"
