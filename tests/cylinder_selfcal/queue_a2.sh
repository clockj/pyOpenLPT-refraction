#!/bin/bash
# Run stream A2, after stream A: the 15 px frames uncorrected at 1.0 px (tag m15t10: the tolerance's effect alone, so
# that V3's gain at 1.0 px can be told from it), then T1's second pass (VSC's iteration, data/T1pre_s4_m15vc2) if
# data/work/pass2_tol.txt says so (its tolerance; written after Gallm's second pass is judged).
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
PY="C:/Users/zcloc/.conda/envs/refract/python.exe"
P=../../cases/cyl_20261007/data
while ! grep -q "stream A done" queue_a_console.txt 2>/dev/null; do sleep 30; done
$PY t1_runs.py final --tag m15t10 --pre $P/T1pre_s4_m15 --tol2d 1.0 --threads 12 > t1_runs_final_m15t10_console.txt 2>&1
$PY t1_runs.py control --tag m15t10 --pre $P/T1pre_s4_m15 --tol2d 1.0 --threads 12 > t1_runs_control_m15t10_console.txt 2>&1
while [ ! -f data/work/pass2_tol.txt ]; do sleep 30; done
TOL=$(cat data/work/pass2_tol.txt)
if [ "$TOL" != "skip" ]; then
  $PY t1_runs.py final --tag m15vc2 --pre data/T1pre_s4_m15vc2 --tol2d $TOL --threads 12 > t1_runs_final_m15vc2_console.txt 2>&1
  $PY t1_runs.py control --tag m15vc2 --pre data/T1pre_s4_m15vc2 --tol2d $TOL --threads 12 > t1_runs_control_m15vc2_console.txt 2>&1
fi
echo "stream A2 done"
