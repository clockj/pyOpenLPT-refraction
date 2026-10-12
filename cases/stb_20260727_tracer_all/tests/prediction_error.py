"""T4: how far STB's prediction misses in tracks of the 10 Hz and 35 Hz runs (offline, from saved tracks).

STB predicts the next point of a long track with the mean step of the last m = 5 frames (predictNext, the stb_improve
campaign's E4): p = y_last + (y_last - y_(last-5)) / 5.  For every point of the saved tracks with at least six earlier
points, the miss |p - y| is measured.  The shake then searches around the prediction with a width of 0.25 voxel
(0.032 mm), halved over its 4 loops; a miss well beyond that is hard to recover.  Read only.

    python tests/prediction_error.py FILE[::LABEL] ...
Writes tests/prediction_error.txt.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import track_motion  # noqa: E402

SHAKE_MM = 0.25 * 0.129


def misses(tracks, m=5):
    out = []
    for t in tracks:
        if len(t) < m + 2:
            continue
        for k in range(m, len(t) - 1):
            p = t[k] + (t[k] - t[k - m]) / m
            out.append(np.linalg.norm(t[k + 1] - p))
    return np.array(out)


def main():
    lines = ["T4 misses of STB's prediction (mean step of 5 frames) in saved long tracks, mm", "",
             f"{'label':34s} {'tracks':>7s} {'points':>9s} {'median':>8s} {'p90':>8s} {'p99':>8s} "
             f"{'> 0.032':>8s} {'> 0.1':>7s}"]
    for arg in sys.argv[1:]:
        path, _, label = arg.partition("::")
        tracks = track_motion.load_tracks(path)
        e = misses(tracks)
        n_tr = sum(1 for t in tracks if len(t) >= 7)
        lines.append(f"{(label or os.path.basename(path)):34s} {n_tr:7d} {len(e):9d} {np.median(e):8.4f} "
                     f"{np.percentile(e, 90):8.4f} {np.percentile(e, 99):8.4f} {100 * np.mean(e > SHAKE_MM):7.1f}% "
                     f"{100 * np.mean(e > 0.1):6.1f}%")
    lines += ["", f"> 0.032: beyond the shake's starting width (0.25 voxel = {SHAKE_MM:.4f} mm); > 0.1: far beyond it. "
                  "These are tracks that survived, so the misses of the tracks that were lost are larger."]
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(HERE, "prediction_error.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


if __name__ == "__main__":
    main()
