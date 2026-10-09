#!/bin/sh
# I1's checks of the install, one at a time (OMP_NUM_THREADS = 12 each): three T1 runs, the comparison, pytest.
cd "$(dirname "$0")" || exit 1
PY="C:/Users/zcloc/.conda/envs/refract/python.exe"
for name in t1_gui t1c_gui t1c_vcf; do
  "$PY" i1_install.py run "$name" > "i1_queue_${name}_console.txt" 2>&1
done
"$PY" i1_install.py compare > i1_queue_compare_console.txt 2>&1
"$PY" i1_install.py pytest > i1_queue_pytest_console.txt 2>&1
echo "queue done" > i1_queue_done.txt
