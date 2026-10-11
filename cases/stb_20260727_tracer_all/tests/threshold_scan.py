"""I4: how many particle images the tracer finder detects per camera at a range of intensity thresholds, frame 0 of
every run.  Read only.

The count follows ObjectFinder2D::findTracer2D: a pixel at or above the threshold that is a local maximum over its
four neighbours (myMATH::isLocalMax: not below the upper and left ones, above the lower and right ones; the border
rows and columns skipped).  It is checked against the detections the smoke runs logged at threshold 20.

    python tests/threshold_scan.py
Writes tests/threshold_scan.txt and tests/threshold_scan.json.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import case  # noqa: E402

THRESHOLDS = [20, 25, 30, 40, 50, 60, 80, 100]


def finder_peaks(img):
    """Peak values of the finder's local maxima (interior pixels)."""
    c = img[1:-1, 1:-1].astype(np.int16)
    up, left = img[:-2, 1:-1], img[1:-1, :-2]
    down, right = img[2:, 1:-1], img[1:-1, 2:]
    m = (c >= up) & (c >= left) & (c > down) & (c > right)
    return c[m]


def scan(cfg, run, cam):
    img = cv2.imread(case.image_path(cfg, run, cam, 0), cv2.IMREAD_UNCHANGED)
    pk = finder_peaks(img)
    return {"run": run, "cam": cam, "counts": {t: int((pk >= t).sum()) for t in THRESHOLDS},
            "peak_median_above20": float(np.median(pk[pk >= 20])) if (pk >= 20).any() else None}


def main():
    cfg = case.load_config()
    runs = cfg["cases"]
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(lambda rc: scan(cfg, *rc), [(r, c) for r in runs for c in range(4)]))
    ref = {r["cam"]: r["counts"][20] for r in rows if r["run"] == runs[0]}
    lines = ["I4 finder detections per camera, frame 0, by intensity threshold (the finder's rule; tracerConfig: 20)",
             "", f"{'run':11s} {'cam':>3s} " + " ".join(f"{'>=' + str(t):>8s}" for t in THRESHOLDS) +
             f" {'median peak':>12s}"]
    for r in rows:
        lines.append(f"{r['run']:11s} {r['cam']:3d} " + " ".join(f"{r['counts'][t]:8d}" for t in THRESHOLDS) +
                     f" {r['peak_median_above20'] or 0:12.0f}")
    lines += ["", "Mean over the four cameras, as a multiple of 10Hz_r_b_1's count at threshold 20:"]
    tot_ref = sum(ref.values())
    for run in runs:
        rr = [r for r in rows if r["run"] == run]
        ratio = {t: sum(r["counts"][t] for r in rr) / tot_ref for t in THRESHOLDS}
        lines.append(f"{run:11s} " + " ".join(f"{ratio[t]:8.2f}" for t in THRESHOLDS))
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(case.TESTS, "threshold_scan.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    case.save_json(os.path.join(case.TESTS, "threshold_scan.json"), {"thresholds": THRESHOLDS, "rows": rows})


if __name__ == "__main__":
    main()
