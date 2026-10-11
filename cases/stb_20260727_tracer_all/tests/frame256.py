"""D1: what happens to the images at frame 256, where the long tracks of all three T3 runs drop (read only).

At frame 256 the 10 Hz batch lost 1.4 % of its long tracks (recovered two frames later), and the two 35 Hz runs of T3
lost 34 % and 40 %, then more over the next two frames.  No other frame of the 10 Hz run's first 9,980 lost more
than 0.8 % (frames 4-5 aside, the start-up).  This looks at each camera's images around frame 256, in imgFile_mask
(what STB reads), imgFile (the videos minus the background) and imgFile_bb (the bubble masks), for a
duplicated, skipped or changed frame:
  - mean: the image's mean grey level;
  - peaks: the finder's detections (myMATH::isLocalMax rule) at the run's threshold;
  - r1, r2: correlation of the frame with the frame 1 and 2 before it (same camera).  A duplicate gives r1 = 1; a
    skipped frame gives an r1 as low as the usual r2;
  - same: the file is byte-identical to the frame before it;
  - mtime: the file's modification time.
Images are read one at a time (no parallel reads beside the batch).

    python tests/frame256.py [--frames 252 260] [--runs RUN:THR ...]
Writes tests/frame256.txt and frame256.json.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import case  # noqa: E402
from threshold_scan import finder_peaks  # noqa: E402

RUNS = ["10Hz_r_b_1:20", "35Hz_r_b_2:80", "35Hz_r_b_1:30"]
FOLDERS = ["imgFile_mask", "imgFile"]


def corr(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def path_of(cfg, run, folder, cam, frame):
    return os.path.join(case.case_dir(cfg, run), folder, f"cam{cam}", cfg["image_name"].format(frame))


def look(cfg, run, thr, folder, cam, frames):
    rows, imgs, digests = [], {}, {}
    for k in range(frames[0] - 2, frames[1] + 1):
        p = path_of(cfg, run, folder, cam, k)
        if not os.path.exists(p):
            return [{"run": run, "folder": folder, "cam": cam, "missing": p}]
        with open(p, "rb") as f:
            raw = f.read()
        digests[k] = hashlib.md5(raw).hexdigest()
        imgs[k] = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if k < frames[0]:
            continue
        img = imgs[k]
        rows.append({
            "run": run, "folder": folder, "cam": cam, "frame": k,
            "mean": float(img.mean()),
            "peaks": int((finder_peaks(img) >= thr).sum()),
            "r1": corr(img, imgs[k - 1]),
            "r2": corr(img, imgs[k - 2]),
            "same": digests[k] == digests[k - 1],
            "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds"),
            "shape": list(img.shape), "dtype": str(img.dtype),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs=2, default=[252, 260])
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--folders", nargs="+", default=FOLDERS)
    ap.add_argument("--out", default="frame256")
    args = ap.parse_args()
    cfg = case.load_config()
    rows = []
    lines = [f"D1 images of each camera, frames {args.frames[0]}-{args.frames[1]} (r1, r2: correlation with the frame "
             "1 and 2 before; same: byte-identical to the frame before)", ""]
    for spec in args.runs:
        run, thr = spec.split(":")
        for folder in args.folders:
            for cam in range(4):
                rs = look(cfg, run, int(thr), folder, cam, args.frames)
                rows += rs
                if "missing" in rs[0]:
                    lines.append(f"{run} {folder} cam{cam}: missing {rs[0]['missing']}")
                    continue
                lines.append(f"{run} {folder} cam{cam} (threshold {thr}, {rs[0]['shape']} {rs[0]['dtype']})")
                lines.append(f"  {'frame':>5s} {'mean':>7s} {'peaks':>7s} {'r1':>6s} {'r2':>6s} {'same':>5s}  mtime")
                for r in rs:
                    lines.append(f"  {r['frame']:5d} {r['mean']:7.2f} {r['peaks']:7d} {r['r1']:6.3f} {r['r2']:6.3f} "
                                 f"{'yes' if r['same'] else '':>5s}  {r['mtime']}")
                print("\n".join(lines[-len(rs) - 2:]), flush=True)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(HERE, args.out + ".txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    with open(os.path.join(HERE, args.out + ".json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
