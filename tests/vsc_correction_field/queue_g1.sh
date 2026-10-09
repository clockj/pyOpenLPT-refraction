#!/bin/bash
# Stream 2 as it was run: G1 end to end on projects/t1_gui (its pass 1 by g1_pass.py --via service already done):
# track at 1.0 px with pass 1's corrections (lp1), the run's tracks into the project, pass 2 through the GUI's service
# (against the self-calibration's two-pass field), the guard (a third pass on the same tracks must be refused), then
# track with pass 2's corrections (lp2: final and control). 12 threads; starts when L0 is done.
cd "$(dirname "$0")"
PY="C:/Users/zcloc/.conda/envs/refract/python.exe"
export PYTHONDONTWRITEBYTECODE=1
while [ ! -f l0_off_compare_out.txt ]; do sleep 30; done
$PY t1_lib_runs.py final --tag lp1 --base projects/t1_gui/config.json --tol2d 1.0 > g1_lp1_final_console.txt 2>&1
rm -rf projects/t1_gui/Results
cp -rp D:/Refraction/pyOpenLPT/cases/cyl_20261007/data/runs/s3_final_lp1/Results projects/t1_gui/Results
$PY g1_pass.py t1_gui --via service --compare t1_vc2_total > g1_pass2_console.txt 2>&1
$PY g1_pass.py t1_gui --via service --expect-refusal > g1_guard_console.txt 2>&1
$PY t1_lib_runs.py final --tag lp2 --base projects/t1_gui/config.json --tol2d 1.0 > g1_lp2_final_console.txt 2>&1
$PY t1_lib_runs.py control --tag lp2 --base projects/t1_gui/config.json --tol2d 1.0 > g1_lp2_control_console.txt 2>&1
echo "stream 2 done"
