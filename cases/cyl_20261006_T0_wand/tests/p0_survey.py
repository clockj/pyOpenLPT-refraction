"""P0 (before the plan): how many frames of run 20261006/T0 are usable, and whether the GUI's "ellipse" estimator
measures this wand with rod settings sized for it.  Read-only on the raw images.

    python tests/p0_survey.py [STEP] [OPEN_PX] [MASK_PX]      (defaults: every 100th frame, 95 px, 50 px)

For every STEP-th frame and both cameras: ball_ellipses.detect_wand_balls (unchanged; the code behind the GUI's
"ellipse" mode), dark balls, radius 40-120 px.  A camera-frame is "clean" when it has two balls, the overlap guard
passed, and both fits have an edge RMS <= 1 px and an aspect ratio 1.15-1.45 (the tube stretches a ball's image
1.26-1.34 : 1).  Also: an opening scan on five frames (how many blobs > 500 px^2 survive openings of 21-141 px),
which brackets the opening that removes the rod but keeps the small ball.
Writes p0_survey_out.txt and p0_survey.json beside itself.
"""
import importlib.util
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

RAW = r"X:\Shijie Zhong\Refraction\Raw\20261006\T0"
BE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "modules", "camera_calibration",
                  "curved_calibration", "ball_ellipses.py")
HERE = os.path.dirname(os.path.abspath(__file__))
CAMS = ("cam1", "cam2")
LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def _be():
    spec = importlib.util.spec_from_file_location("ball_ellipses", os.path.normpath(BE))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def work(args):
    f, cam, open_px, mask_px = args
    cv2.setNumThreads(1)
    be = _be()
    img = cv2.imread(os.path.join(RAW, cam, f"Img{f:06d}.tif"), cv2.IMREAD_UNCHANGED)
    rows = be.detect_wand_balls(img, "dark", min_r_px=40.0, max_r_px=120.0, open_px=open_px, rod_mask_px=mask_px,
                                full=True)
    return f, cam, rows


def clean(rows):
    if len(rows) != 2 or any(r[7] for r in rows):
        return False
    for r in rows:
        rms = 1.0 / r[3] - 1.0
        ab = max(r[4], r[5]) / min(r[4], r[5])
        if rms > 1.0 or not 1.15 <= ab <= 1.45:
            return False
    return True


def opening_scan(frames):
    be = _be()
    det = be.settings_for(120.0)
    say("Opening scan: blobs > 500 px^2 left after an opening of k px (1 = rod kept or a ball lost, 2 = separated)")
    for f in frames:
        for cam in CAMS:
            img = cv2.imread(os.path.join(RAW, cam, f"Img{f:06d}.tif"), cv2.IMREAD_UNCHANGED)
            N, _bg, seg = be.normalised_darkness(img.astype(np.float32), det)
            m = (N > seg).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            parts = []
            for k in (21, 41, 61, 71, 81, 91, 101, 111, 121, 141):
                mo = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
                n, _lab, st, _c = cv2.connectedComponentsWithStats(mo, connectivity=8)
                parts.append(f"{k}:{sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 500)}")
            say(f"  frame {f:5d} {cam}: " + " ".join(parts))


def main():
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    open_px = int(sys.argv[2]) if len(sys.argv) > 2 else 95
    mask_px = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0
    say("P0: usable frames and the estimator on run 20261006/T0 (read-only)")
    opening_scan([0, 2500, 4000, 8500, 10000])
    frames = list(range(0, 10640, step))
    tasks = [(f, c, open_px, mask_px) for f in frames for c in CAMS]
    res = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for f, cam, rows in ex.map(work, tasks, chunksize=4):
            res[(f, cam)] = rows
    say(f"\nSurvey: every {step}th frame ({len(frames)} frames), rod opening {open_px} px, rod mask {mask_px:g} px")
    both = 0
    for cam in CAMS:
        ok = [f for f in frames if clean(res[(f, cam)])]
        two = sum(1 for f in frames if len(res[(f, cam)]) == 2 and not any(r[7] for r in res[(f, cam)]))
        s = np.array([[r[4], r[5]] for f in ok for r in sorted(res[(f, cam)], key=lambda r: r[2])]).reshape(-1, 2, 2)
        rms = [1.0 / r[3] - 1.0 for f in ok for r in res[(f, cam)]]
        if len(s):
            med = np.median(s, axis=0)
            say(f"  {cam}: two balls apart in {two}, clean in {len(ok)}; median semi-axes (a, b): small "
                f"{med[0, 0]:.1f}, {med[0, 1]:.1f} px; large {med[1, 0]:.1f}, {med[1, 1]:.1f} px; edge RMS median "
                f"{np.median(rms):.3f} px, 90th percentile {np.percentile(rms, 90):.3f} px")
        else:
            say(f"  {cam}: two balls apart in {two}, clean in 0")
    both = sum(1 for f in frames if clean(res[(f, "cam1")]) and clean(res[(f, "cam2")]))
    say(f"  clean in both cameras: {both} of {len(frames)} frames ({100.0 * both / len(frames):.0f} %)")
    with open(os.path.join(HERE, "p0_survey.json"), "w") as fh:
        json.dump({"step": step, "open_px": open_px, "rod_mask_px": mask_px,
                   "rows": {f"{f}_{c}": res[(f, c)] for (f, c) in res}}, fh)
    with open(os.path.join(HERE, "p0_survey_out.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
