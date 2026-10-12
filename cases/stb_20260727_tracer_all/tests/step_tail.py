"""U1-U6, an added check (not in the plan, added after U2): does a smaller linking spacing lose fast particles?

With spacing 5 voxels a short track links an IPR particle within 0.65 mm of its prediction (widening to 0.94 mm),
against 3.9-5.7 mm with 30 voxels, and a short track without long-track neighbours predicts no motion.  So
particles that move much faster than the median 0.32 mm per frame could be lost.  For each run and test: the steps
of the long tracks alive at the end of frame 200 (rebuilt as in track_quality.py), their tail percentiles and
maximum, and the tracks counted by their fastest step.  Read only.

    python tests/step_tail.py        writes tests/u_step_tail.txt
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import track_quality as tq  # noqa: E402
import tune_tests as tt  # noqa: E402

BINS = [0.4, 0.5, 0.6, 0.8, 1.0]  # mm per frame: tracks whose fastest step is above each


def tail(conv_dir, max_save=None):
    tracks, _files = tq.alive_at(conv_dir, tt.Q_FRAME, max_save)
    tracks = [(fr, p) for fr, p in tracks if len(fr) >= 4]
    steps = [np.linalg.norm(np.diff(p, axis=0), axis=1) for _fr, p in tracks]
    flat = np.concatenate(steps)
    fastest = np.array([s.max() for s in steps])
    return {"tracks": len(tracks), "p50": np.percentile(flat, 50), "p90": np.percentile(flat, 90),
            "p99": np.percentile(flat, 99), "p999": np.percentile(flat, 99.9), "max": flat.max(),
            "over": [int((fastest > b).sum()) for b in BINS]}


def main():
    cfg = tt.case.load_config()
    lines = ["Added check: steps of the long tracks alive at the end of frame 200 (rebuilt), mm per frame; "
             "'fastest > x': tracks whose fastest step exceeds x", "",
             f"{'run':22s} {'tracks':>7s} {'p50':>6s} {'p90':>6s} {'p99':>6s} {'p99.9':>6s} {'max':>6s} " +
             " ".join(f"{'>' + format(b, 'g'):>7s}" for b in BINS)]
    for run in ("35Hz_r_b_1", "35Hz_r_b_2", "10Hz_r_b_1"):
        label, bdir, ms = tt.baseline(cfg, run)
        cands = [(label, bdir, ms)] + [(tt.name_of(t, run), tt.test_dir(cfg, t, run), None)
                                        for t in ("U1", "U2", "U3", "U4", "U5", "U6")]
        for name, d, ms in cands:
            if name != label and not os.path.exists(os.path.join(d, "run_info.json")):
                continue
            r = tail(os.path.join(d, cfg["output_folder"], "ConvergeTrack"), ms)
            lines.append(f"{name:22s} {r['tracks']:7d} {r['p50']:6.3f} {r['p90']:6.3f} {r['p99']:6.3f} "
                         f"{r['p999']:6.3f} {r['max']:6.3f} " + " ".join(f"{n:7d}" for n in r["over"]))
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(HERE, "u_step_tail.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


if __name__ == "__main__":
    main()
