#!/usr/bin/env bash
# Queue 4 of the stb_improve campaign (G-S50): your rebuilt build_local.py build (build/Release/OpenLPT.exe, which the
# GUI runs, and build-py/Release's module) against the campaign's final build on frames 0-50.
# Run from the repository root:  bash tests/stb_improve/queue4.sh   (log: results/queue4.log)
set -u
PY="/c/Users/zcloc/.conda/envs/refract/python.exe"
RUN="$PY tests/stb_improve/stbrun.py"
CMP="$PY tests/stb_improve/compare_runs.py"
LOG=tests/stb_improve/results/queue4.log
mkdir -p tests/stb_improve/results
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
step() { say "start $*"; "$@" >> "$LOG" 2>&1; say "exit $? $*"; }

say "queue4 begins"
step $RUN g_s50_user_exe --build user --exe --frames 0 50
step $RUN g_s50_user_py --build user --frames 0 50
step $CMP identical s50_final3 g_s50_user_exe --out g_s50_exe_vs_final3
step $CMP identical s50_final3 g_s50_user_py --out g_s50_py_vs_final3
say "queue4 ends"
