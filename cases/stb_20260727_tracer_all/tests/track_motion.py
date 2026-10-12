"""T2: how fast the particles move and how straight their tracks are, from the tracks STB saved in test runs.

For each named smoke run: per long track (4+ points) of InitialTrack\\LongTrackActive_3.csv (the initial phase's
tracks, frames 0-3) and of ConvergeTrack\\LongTrackActive_<last>.csv:
  - step length |p(k+1) - p(k)| (mm per frame) and the change of step |d2p| (mm per frame^2);
  - STB's linear-fit residual of the last four points, with its weights (-0.2, 0.1, 0.4, 0.7) on the deviation from
    the least-squares line through them (the limit is 0.05 mm), as the stb_improve campaign ported it.
Read only.

    python tests/track_motion.py NAME ...
Writes tests/track_motion.txt.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402


def load_tracks(path):
    a = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 4), ndmin=2)
    if a.size == 0:
        return []
    out = []
    ids = a[:, 0]
    cut = np.flatnonzero(np.diff(ids)) + 1
    for blk in np.split(a, cut):
        out.append(blk[np.argsort(blk[:, 1]), 2:5])
    return out


def lf_residual(p4):
    """Weighted deviation of the last of four points from the least-squares line through them, as STB's
    linear-fit test (weights -0.2, 0.1, 0.4, 0.7 on t = 0..3): the predicted last point is sum w_j p_j."""
    w = np.array([-0.2, 0.1, 0.4, 0.7])
    pred = (w[:, None] * p4).sum(axis=0)
    return float(np.linalg.norm(p4[-1] - pred))


def stats(tracks):
    steps, acc, res = [], [], []
    for t in tracks:
        if len(t) < 4:
            continue
        d = np.diff(t, axis=0)
        steps += list(np.linalg.norm(d, axis=1))
        acc += list(np.linalg.norm(np.diff(d, axis=0), axis=1))
        res.append(lf_residual(t[-4:]))
    q = lambda x, p: float(np.percentile(x, p)) if len(x) else float("nan")  # noqa: E731
    return {"tracks": sum(1 for t in tracks if len(t) >= 4), "step_med": q(steps, 50), "step_p90": q(steps, 90),
            "acc_med": q(acc, 50), "acc_p90": q(acc, 90), "lf_med": q(res, 50), "lf_p90": q(res, 90),
            "lf_over": float(np.mean(np.array(res) > 0.05)) if res else float("nan")}


def main():
    cfg = case.load_config()
    lines = ["T2 particle motion in the saved tracks (mm per frame; linear-fit residual of the last 4 points, "
             "limit 0.05 mm)", "",
             f"{'run':28s} {'file':26s} {'tracks':>7s} {'step med':>9s} {'p90':>6s} {'d2 med':>7s} {'p90':>6s} "
             f"{'LF med':>7s} {'p90':>6s} {'> 0.05':>7s}"]
    for name in sys.argv[1:]:
        root = os.path.join(cfg["staging_root"], "smoke", name, cfg["output_folder"])
        files = [os.path.join(root, "InitialTrack", "LongTrackActive_3.csv")]
        conv = sorted(glob.glob(os.path.join(root, "ConvergeTrack", "LongTrackActive_*.csv")))
        files += conv[-1:]
        for f in files:
            if not os.path.exists(f):
                continue
            s = stats(load_tracks(f))
            lines.append(f"{name:28s} {os.path.relpath(f, root):26s} {s['tracks']:7d} {s['step_med']:9.3f} "
                         f"{s['step_p90']:6.3f} {s['acc_med']:7.4f} {s['acc_p90']:6.4f} {s['lf_med']:7.4f} "
                         f"{s['lf_p90']:6.4f} {100 * s['lf_over']:6.1f}%")
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(HERE, "track_motion.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


if __name__ == "__main__":
    main()
