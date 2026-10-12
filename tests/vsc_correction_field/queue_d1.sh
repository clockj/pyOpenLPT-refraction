#!/bin/sh
# Follow-up 2's checks, one at a time: D1, G3 on the installed module (a fresh T1 project), pytest with it.
cd "$(dirname "$0")" || exit 1
PY="C:/Users/zcloc/.conda/envs/refract/python.exe"
OMP_NUM_THREADS=12 "$PY" d1_detection.py > d1_detection_console.txt 2>&1
OMP_NUM_THREADS=12 "$PY" g3_gui.py t1_g3gui2 gui > g3_gui_gui_console.txt 2>&1
"$PY" i1_install.py pytest d0_pytest > d0_queue_pytest_console.txt 2>&1
echo "queue done" > d0_queue_done.txt
