"""I5: a finder threshold per dense run that gives the same number of detections as a tested setting.

For every run and camera, the finder's detections on frames 0, 3000, 6000 and 9000 (the rule of threshold_scan.py) at
every integer threshold from 20 to 160.  For a target (a run and threshold of T1 that tracked well), each other run gets
the threshold whose mean detections per camera are closest to the target's.  Read only.

    python tests/propose_thresholds.py --like 35Hz_r_b_2:80 [--like 35Hz_r_b_1:40]
Writes tests/propose_thresholds.txt and tests/propose_thresholds.json.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import case  # noqa: E402
from threshold_scan import finder_peaks  # noqa: E402

FRAMES = (0, 3000, 6000, 9000)
TH = np.arange(20, 161)


def counts(cfg, run, cam):
    """Mean detections over FRAMES at every threshold in TH."""
    tot = np.zeros(len(TH))
    for fr in FRAMES:
        pk = finder_peaks(cv2.imread(case.image_path(cfg, run, cam, fr), cv2.IMREAD_UNCHANGED))
        hist = np.bincount(pk.astype(np.int64), minlength=256)
        above = np.cumsum(hist[::-1])[::-1]          # above[t] = peaks >= t
        tot += above[TH]
    return tot / len(FRAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", action="append", required=True, help="RUN:THRESHOLD of a tested setting")
    args = ap.parse_args()
    cfg = case.load_config()
    runs = cfg["cases"]
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(lambda rc: counts(cfg, *rc), [(r, c) for r in runs for c in range(4)]))
    per_run = {r: np.array(res[4 * i:4 * i + 4]) for i, r in enumerate(runs)}   # (4 cams, len(TH))
    mean = {r: per_run[r].mean(axis=0) for r in runs}
    out = {"frames": FRAMES, "thresholds": TH.tolist(),
           "mean_per_camera": {r: mean[r].tolist() for r in runs},
           "per_camera": {r: per_run[r].tolist() for r in runs}, "proposals": {}}
    lines = ["I5 finder thresholds that match a tested setting's detections (mean of frames "
             f"{', '.join(map(str, FRAMES))}; the finder's rule)", ""]
    for like in args.like:
        ref_run, ref_th = like.split(":")
        ref_th = int(ref_th)
        target = mean[ref_run][TH == ref_th][0]
        lines.append(f"target: {ref_run} at {ref_th}: {target:,.0f} detections per camera (mean)")
        lines.append(f"{'run':11s} {'threshold':>9s} {'mean/cam':>9s}   per camera")
        prop = {}
        for r in runs:
            k = int(np.argmin(np.abs(mean[r] - target)))
            th = int(TH[k])
            prop[r] = th
            cams = " ".join(f"{per_run[r][c, k]:8,.0f}" for c in range(4))
            note = "  (below 20 needed: kept at 20)" if k == 0 and mean[r][0] < target else ""
            lines.append(f"{r:11s} {th:9d} {mean[r][k]:9,.0f}   {cams}{note}")
        out["proposals"][like] = prop
        lines.append("")
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(HERE, "propose_thresholds.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    case.save_json(os.path.join(HERE, "propose_thresholds.json"), out)


if __name__ == "__main__":
    main()
