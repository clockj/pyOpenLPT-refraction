#!/bin/bash
# T1's second pass, the preparation: after the first pass's final run (s3_final_m15vc), gather its tracks on the warped
# frames, fit the field left, compose it with the first pass's, and warp the 15 px frames by the total (v3_warp.py
# vc2 -> data/T1pre_s4_m15vc2).
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
PY="C:/Users/zcloc/.conda/envs/refract/python.exe"
P=../../cases/cyl_20261007/data
while ! grep -q "exit" t1_runs_final_m15vc_out.txt 2>/dev/null; do sleep 30; done
$PY gather.py t1_vc --run $P/runs/s3_final_m15vc --lists data/T1pre_s4_m15vc/cam1ImageNames.txt \
    data/T1pre_s4_m15vc/cam2ImageNames.txt --noise 7.281137583191247 5.35252574528454 --cams $P/cams_T0/cam1.txt \
    $P/cams_T0/cam2.txt --every 5 --depths --procs 6 > gather_t1_vc_console.txt 2>&1
$PY fit_t1.py t1_vc data/fields/t1_vc_pass2.npz > fit_t1_t1_vc_console.txt 2>&1
$PY compose_fields.py data/fields/t1_vc2_total.npz data/fields/t1_m15_all.npz data/fields/t1_vc_pass2.npz \
    > compose_t1_console.txt 2>&1
$PY v3_warp.py vc2 data/fields/t1_vc2_total.npz --procs 6 > v3_warp_vc2_console.txt 2>&1
echo "prep vc2 done"
