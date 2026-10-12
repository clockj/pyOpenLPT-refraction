#!/bin/bash
# V3's last step, after the second pass's runs: gather the second pass's final run on its frames, compare the five T1
# runs (v3_compare.py -> v3_compare_out.txt), and regenerate every table (summary_sc.py).
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
PY="C:/Users/zcloc/.conda/envs/refract/python.exe"
P=../../cases/cyl_20261007/data
while ! grep -q "stream A2 done" queue_a2_console.txt 2>/dev/null; do sleep 30; done
NOISE=$($PY -c "import json; m = json.load(open('data/T1pre_s4_m15vc2/manifest.json')); print(*[repr(m['cameras'][c]['noise_8bit_median']) for c in ('cam1', 'cam2')])")
$PY gather.py t1_vc2 --run $P/runs/s3_final_m15vc2 --lists data/T1pre_s4_m15vc2/cam1ImageNames.txt \
    data/T1pre_s4_m15vc2/cam2ImageNames.txt --noise $NOISE --cams $P/cams_T0/cam1.txt $P/cams_T0/cam2.txt \
    --every 5 --depths --procs 6 > gather_t1_vc2_console.txt 2>&1
$PY v3_compare.py m15=s3_final_m15,s3_control_m15,t1_m15 t10=s3_final_m15t10,s3_control_m15t10,t1_t10 \
    vc=s3_final_m15vc,s3_control_m15vc,t1_vc vc10=s3_final_m15vc10,s3_control_m15vc10,t1_vc10 \
    vc2=s3_final_m15vc2,s3_control_m15vc2,t1_vc2 --ref m15 --name v3_compare > v3_compare_console.txt 2>&1
$PY summary_sc.py > summary_sc_console.txt 2>&1
echo "final v3 done"
