#!/bin/bash
# Run stream C, after stream B: Gallm at 1.0 px; the second pass (VSC's iteration) on Gallm (warped) and on the
# 2-camera set (re-rendered): gather the first pass's run, fit the field left, compose it with the first, apply the
# total to the original images, track.
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
PY="C:/Users/zcloc/.conda/envs/refract/python.exe"
while ! grep -q "stream B done" queue_b_console.txt 2>/dev/null; do sleep 30; done
# the second pass, 2 cameras (re-rendered)
L=data/cal4/imgLists/Gcal4c2_n8
$PY gather.py cal4_c2 --run runs/Gcal4c2_n8_r2_tol1.0 --lists $L/cam1ImageNames.txt $L/cam2ImageNames.txt \
    --noise 8 8 --cams data/cal4/camFile/cam1.txt data/cal4/camFile/cam2.txt --otf data/cal4/otf_r2.txt --every 1 \
    --procs 8 > gather_cal4_c2_console.txt 2>&1
$PY v2_fit.py cal4_c2 --gather cal4_c2 --case cal4 --set Gcal4c2_n8 > v2_fit_cal4_c2_console.txt 2>&1
$PY compose_fields.py data/fields/cal4_r2_total2.npz data/fields/cal4_r2.npz data/fields/cal4_c2.npz \
    > compose_cal4_console.txt 2>&1
$PY synth_sc.py corrected cal4 Gcal4_n8 Gcal4c2b_n8 data/fields/cal4_r2_total2.npz > synth_sc_corrected_Gcal4c2b_console.txt 2>&1
$PY sruns.py Gcal4c2b_n8_r2_tol1.0 --threads 12 > sruns_Gcal4c2b_n8_r2_tol1.0_console.txt 2>&1
# the second pass, Gallm (warped)
L=data/gallm/imgLists/Gallmw
$PY gather.py gallm_w1 --run runs/Gallmw_r2_tol1.5 --lists $L/cam1ImageNames.txt $L/cam2ImageNames.txt \
    --noise 8.452 6.189 --cams data/gallm/camFile/cam1.txt data/gallm/camFile/cam2.txt --every 1 --procs 8 \
    > gather_gallm_w1_console.txt 2>&1
$PY v2_fit.py gallm_w1 --gather gallm_w1 --case gallm --set Gallmw > v2_fit_gallm_w1_console.txt 2>&1
$PY compose_fields.py data/fields/gallm_total2.npz data/fields/gallm_r2.npz data/fields/gallm_w1.npz \
    > compose_gallm_console.txt 2>&1
$PY synth_sc.py warped gallm Gallm Gallmw2 data/fields/gallm_total2.npz > synth_sc_warped_Gallmw2_console.txt 2>&1
$PY sruns.py Gallmw2_r2_tol1.5 Gallmw2_r2_tol1.0 --threads 12 > sruns_gallmw2_console.txt 2>&1
echo "stream C done"
