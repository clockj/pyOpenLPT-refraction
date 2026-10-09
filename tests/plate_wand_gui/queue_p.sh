#!/bin/sh
# The flat-window GUI options' tests on the final code, one at a time; then the wand tab's regression (W11, tag p4pwg).
cd "$(dirname "$0")" || exit 1
PY="C:/Users/zcloc/.conda/envs/refract/python.exe"
for t in p3_panel p0_unchanged p1_real_case p2_options p2b_thickness p3b_screens; do
  "$PY" "$t.py" > "${t}_console.txt" 2>&1
done
(cd ../wand_gui_fixes && "$PY" w11_regression.py --tag p4pwg > ../plate_wand_gui/p4_w11_console.txt 2>&1)
echo "queue done" > queue_p_done.txt
