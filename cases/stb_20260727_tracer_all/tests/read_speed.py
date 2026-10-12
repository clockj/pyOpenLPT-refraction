"""R1: how fast the masked images come from X:, read one by one as STB reads them, against a parallel copy to D:.

  - sequential: cv2.imread of N frames x 4 cameras straight from X: (STB loads the four images of a frame one after
    another, outside its frame timer);
  - robocopy /MT:16 of the next N frames x 4 cameras to a scratch folder on D:, then the same reads from D:.
The two runs of the batch differ by about 1 s per frame between wall time and the frame timer's sum, which is where
the loading goes.

    python tests/read_speed.py [RUN] [--frames N] [--start F]
Writes tests/read_speed.txt.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import case  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="10Hz_r_b_2")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--start", type=int, default=7000)
    args = ap.parse_args()
    cfg = case.load_config()
    n, f0 = args.frames, args.start
    lines = [f"R1 image read speed, run {args.run}", ""]

    t = time.perf_counter()
    nbytes = 0
    for fr in range(f0, f0 + n):
        for cam in range(4):
            p = case.image_path(cfg, args.run, cam, fr)
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            nbytes += os.path.getsize(p)
            assert img is not None
    dt = time.perf_counter() - t
    lines.append(f"sequential reads from X:: {n} frames x 4 cameras, {nbytes / 1e6:.0f} MB in {dt:.1f} s: "
                 f"{dt / n:.2f} s per frame, {nbytes / 1e6 / dt:.1f} MB/s")

    scratch = os.path.join(cfg["staging_root"], "smoke", "r1_read_speed")
    shutil.rmtree(scratch, ignore_errors=True)
    f1 = f0 + n
    t = time.perf_counter()
    for cam in range(4):
        src = os.path.join(case.case_dir(cfg, args.run), cfg["image_folder"], f"cam{cam}")
        names = [cfg["image_name"].format(fr) for fr in range(f1, f1 + n)]
        subprocess.call(["robocopy", src, os.path.join(scratch, f"cam{cam}"), *names, "/MT:16", "/NP", "/NFL",
                         "/NDL", "/NJH", "/NJS"], stdout=subprocess.DEVNULL)
    dt = time.perf_counter() - t
    copied = sum(os.path.getsize(os.path.join(dp, f)) for dp, _d, fn in os.walk(scratch) for f in fn)
    lines.append(f"robocopy /MT:16 to D: (one call per camera): {n} frames x 4 cameras, {copied / 1e6:.0f} MB in "
                 f"{dt:.1f} s: {dt / n:.2f} s per frame, {copied / 1e6 / dt:.1f} MB/s")

    t = time.perf_counter()
    for fr in range(f1, f1 + n):
        for cam in range(4):
            img = cv2.imread(os.path.join(scratch, f"cam{cam}", cfg["image_name"].format(fr)), cv2.IMREAD_UNCHANGED)
            assert img is not None
    dt = time.perf_counter() - t
    lines.append(f"sequential reads from D: (the copies): {dt / n:.3f} s per frame")
    shutil.rmtree(scratch, ignore_errors=True)
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(case.TESTS, "read_speed.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


if __name__ == "__main__":
    main()
