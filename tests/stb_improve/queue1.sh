#!/usr/bin/env bash
# Queue 1 of the stb_improve campaign: every run of Phase A and the identical-output check of Phase B, one after
# another on a quiet machine.  Each line is one stbrun.py run (runs/<name>/); this script's own log is
# results/queue1.log.  Run from the repository root:  bash tests/stb_improve/queue1.sh
set -u
PY="/c/Users/zcloc/.conda/envs/refract/python.exe"
RUN="$PY tests/stb_improve/stbrun.py"
LOG=tests/stb_improve/results/queue1.log
mkdir -p tests/stb_improve/results
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
step() { say "start $*"; "$@" >> "$LOG" 2>&1; say "exit $? $*"; }

say "queue1 begins"
# --- memory checks after A1: index checks (frames 0-10) ---
step $RUN u1_check_a1_f10 --build a1-9-check --frames 0 10
# --- S1 identical outputs, frames 0-50: the baseline twice (determinism), A1 at both sizes, all of B0/B1/B3/B4 ---
step $RUN s50_base --build base --frames 0 50
step $RUN s50_base_2 --build base --frames 0 50
step $RUN s50_a1_9 --build a1-9 --frames 0 50
step $RUN s50_a1_3 --build a1-3 --frames 0 50
step $RUN s50_b134 --build b134 --frames 0 50 --config-extra "profile = 1"
# B2 changes results: compared with s50_b134 (matched objects per round, tracks)
step $RUN s50_b2 --build b2 --frames 0 50 --config-extra "profile = 1"
# --- C1 end to end: the same run from config.txt and from config.json; P2 logic: saves at 20, 40 and 45 ---
step $RUN s20_text --build b134 --frames 0 20
step $RUN s20_json --build b134 --frames 0 20 --json
step $RUN p2_si20 --build b134 --frames 0 45 --config-extra "save_interval = 20"
# --- P1 benchmarks, frames 0-300 from scratch ---
step $RUN b300_base --build base --frames 0 300
step $RUN b300_a1_9 --build a1-9 --frames 0 300
step $RUN b300_b134 --build b134 --frames 0 300 --config-extra "profile = 1"
step $RUN b300_b2 --build b2 --frames 0 300 --config-extra "profile = 1"
step $RUN b300_a1_3 --build a1-3 --frames 0 300
# --- R300 restart tests: tracks saved at frame 200 by a 0-200 run, then frames 201-300 in a fresh process ---
step $RUN r200_base --build base --frames 0 200
step $RUN r300_base --build base --frames 0 300 --load 200 --load-from r200_base
step $RUN r200_a1_9 --build a1-9 --frames 0 200
step $RUN r300_a1_9 --build a1-9 --frames 0 300 --load 200 --load-from r200_a1_9
step $RUN r200_a1_3 --build a1-3 --frames 0 200
step $RUN r300_a1_3 --build a1-3 --frames 0 300 --load 200 --load-from r200_a1_3
# --- memory check after A1 under AddressSanitizer (frames 0-5) ---
step $RUN u2_asan_a1_f5 --build asan-a1 --exe --frames 0 5 --asan-options malloc_context_size=0
say "queue1 ends"
