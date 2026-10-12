#!/usr/bin/env bash
# Queue 3 of the stb_improve campaign: the shipped configuration (Matrix inline size 3, the default since 5 Oct
# evening) under the index checks and AddressSanitizer, and the unit tests with explicit sizes; after queue 2.
# Run from the repository root:  bash tests/stb_improve/queue3.sh   (log: results/queue3.log)
set -u
PY="/c/Users/zcloc/.conda/envs/refract/python.exe"
RUN="$PY tests/stb_improve/stbrun.py"
LOG=tests/stb_improve/results/queue3.log
mkdir -p tests/stb_improve/results
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
step() { say "start $*"; "$@" >> "$LOG" 2>&1; say "exit $? $*"; }

say "queue3 begins"
step $PY tests/stb_improve/build_variant.py final3-check --def OPENLPT_MATRIX_CHECKS --fresh
step $PY tests/stb_improve/build_variant.py asan-final3 --mode tests --asan --fresh
step $PY tests/stb_improve/cpp/run_unit_tests.py
step $RUN u1_check_final3_f10 --build final3-check --frames 0 10
step $RUN u2_asan_final3_f5 --build asan-final3 --exe --frames 0 5 --asan-options malloc_context_size=0
say "queue3 ends"
