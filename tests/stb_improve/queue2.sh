#!/usr/bin/env bash
# Queue 2 of the stb_improve campaign: the final build (A1 + A3 + A4 + B0-B4 + the IPR log fix), after queue 1.
# Run from the repository root:  bash tests/stb_improve/queue2.sh   (log: results/queue2.log)
set -u
PY="/c/Users/zcloc/.conda/envs/refract/python.exe"
RUN="$PY tests/stb_improve/stbrun.py"
LOG=tests/stb_improve/results/queue2.log
mkdir -p tests/stb_improve/results
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
step() { say "start $*"; "$@" >> "$LOG" 2>&1; say "exit $? $*"; }

say "queue2 begins"
# index checks on the final code (frames 0-10)
step $RUN u1_check_final_f10 --build final-check --frames 0 10
# which part removed the growth: B1 (the tables, caching the lines of sight) alone, without B3 and B4
step $RUN b300_b1 --build b1 --frames 0 300 --config-extra "profile = 1"
# the final code with the inline sizes 9 and 3, built from the same tree (A1's choice is made on the final code)
step $PY tests/stb_improve/build_variant.py final --fresh
step $PY tests/stb_improve/build_variant.py final3 --def OPENLPT_MATRIX_INLINE=3 --fresh
# identical outputs: the final build against b2 (they differ only in the IPR log line's order), and size 3 against 9
step $RUN s50_final --build final --frames 0 50
step $RUN s50_final3 --build final3 --frames 0 50
# A1's inline size on the final code: 9 against 3, frames 0-300
step $RUN b300_final --build final --frames 0 300
step $RUN b300_final3 --build final3 --frames 0 300
# A3: saves every 200 frames on the final build, against b300_final (saves only at the end)
step $RUN b300_final_si200 --build final --frames 0 300 --config-extra "save_interval = 200"
# R300 on the final build
step $RUN r200_final --build final --frames 0 200
step $RUN r300_final --build final --frames 0 300 --load 200 --load-from r200_final
# AddressSanitizer on the final code (frames 0-5)
step $RUN u2_asan_final_f5 --build asan-final --exe --frames 0 5 --asan-options malloc_context_size=0
say "queue2 ends"
