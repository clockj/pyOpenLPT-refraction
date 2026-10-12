#!/bin/bash
# L0 as it was run: two streams of 12 threads, then the verdict.
cd "$(dirname "$0")"
PY="C:/Users/zcloc/.conda/envs/refract/python.exe"
export PYTHONDONTWRITEBYTECODE=1
(
  $PY l0_off.py run t1_p4c_a > l0_streamA_console.txt 2>&1
  $PY l0_off.py run syn_p4c >> l0_streamA_console.txt 2>&1
  $PY l0_off.py pytest p4c >> l0_streamA_console.txt 2>&1
) &
(
  $PY l0_off.py run t1_vcf > l0_streamB_console.txt 2>&1
  $PY l0_off.py run syn_vcf >> l0_streamB_console.txt 2>&1
  $PY l0_off.py pytest vcf >> l0_streamB_console.txt 2>&1
) &
wait
$PY l0_off.py run t1_p4c_b > l0_streamC_console.txt 2>&1
$PY l0_off.py compare > l0_compare_console.txt 2>&1
echo "L0 queue done"
