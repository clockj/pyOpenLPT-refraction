#!/bin/bash
# G2's 2-camera GUI path over two passes as it was run (after stream 1, so that at most two tracking runs share the
# machine): a project from the 2-camera run V2 fitted, pass 1 through the GUI's service, tracking at 1.0 px with it,
# pass 2, tracking with the total, then every 2-camera run scored against the truth. 12 threads.
cd "$(dirname "$0")"
PY="C:/Users/zcloc/.conda/envs/refract/python.exe"
export PYTHONDONTWRITEBYTECODE=1
while [ ! -f l3_synthetic_score_out.txt ]; do sleep 30; done
$PY g2_synthetic.py prepare 2 syn2_gui > g2_syn2_prepare_console.txt 2>&1
$PY g2_synthetic.py vsc syn2_gui --via service > g2_syn2_pass1_console.txt 2>&1
$PY g2_synthetic.py track syn2_gui g2_lp1_r2 > g2_syn2_track1_console.txt 2>&1
$PY g2_synthetic.py vsc syn2_gui --via service > g2_syn2_pass2_console.txt 2>&1
$PY g2_synthetic.py track syn2_gui g2_lp2_r2 > g2_syn2_track2_console.txt 2>&1
$PY g2_synthetic.py score > g2_score_console.txt 2>&1
echo "G2 queue done"
