#!/usr/bin/env bash
# Queue 5 of the stb_improve campaign (G5, 6 Oct): your exe and module rebuilt with E4's origin-free predictor
# (build_local.py: build/Release/OpenLPT.exe, which the GUI runs, and build-py/Release's module) against the tested
# `md` build on frames 0-50, directly and through the GUI's own path.
# Run from the repository root:  bash tests/stb_improve/queue5.sh   (log: results/queue5.log)
set -u
PY="/c/Users/zcloc/.conda/envs/refract/python.exe"
RUN="$PY tests/stb_improve/stbrun.py"
CMP="$PY tests/stb_improve/compare_runs.py"
LOG=tests/stb_improve/results/queue5.log
mkdir -p tests/stb_improve/results
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
step() { say "start $*"; "$@" >> "$LOG" 2>&1; say "exit $? $*"; }

say "queue5 begins"
step $RUN s50_md --build md --frames 0 50 --force
step $RUN g5_s50_user_exe --build user --exe --frames 0 50 --force
step $RUN g5_s50_user_py --build user --frames 0 50 --force
step $PY tests/stb_improve/g_gui_e2e.py e2e --frames 0 50 --name g5_e2e_gui
step $CMP identical s50_md g5_s50_user_exe --out g5_s50_exe_vs_md
step $CMP identical s50_md g5_s50_user_py --out g5_s50_py_vs_md
step $CMP identical s50_md g5_e2e_gui --out g5_e2e_gui_vs_md
say "queue5 ends"
