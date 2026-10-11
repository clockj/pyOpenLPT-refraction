"""Quality of the long tracks of a run at one frame (plan 2026-10-11-plan-tuning.md, section 3). Read only.

The tracks alive at the end of frame F, rebuilt from the run's saved track files, cut at F:
  - STB keeps only its latest LongTrackActive_<s>.csv, but a long track that ends goes to LongTrackInactive_<s>.csv or
    ExitTrack_<s>.csv at the next save s and is written nowhere else.
  - So the tracks alive at F are those of the files saved after F that have a point at F (a track removed in frame
    F is in the files of the save at F, and is not alive at its end).
  - Tracks that end with fewer than 7 points are never saved (LEN_LONG_TRACK), in any run alike.
Measures on those tracks, up to F:
  - step |p(k+1) - p(k)|, median and 90th percentile (mm per frame), as T2 (track_motion.py);
  - STB's linear-fit residual of the last four points (limit 0.05 mm), median and 90th percentile, as T2;
  - neighbour check: velocity v = (p(F) - p(F-2)) / 2 against the median velocity of the other tracks within R of
    p(F) (each axis's median), for tracks with at least 3 such neighbours: |v - v_med| / |v_med|, median, 90th
    percentile and the share above 0.5; and the share of tracks that have 3 neighbours (coverage).

    python tests/track_quality.py RUN_DIR F [--max-save S] [--radius R]
RUN_DIR holds ConvergeTrack\\ (a run's output folder); --max-save skips files saved after S (long runs).
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

LF_W = np.array([-0.2, 0.1, 0.4, 0.7])  # least-squares line through t = 0..3, evaluated at t = 3 (STB's test)
RADIUS_MM = 8.0  # the plan said 4 mm; at 4 mm only 23-57 % of the dense runs' tracks have 3 neighbours, at 8 mm
#                  89-97 %, with the same median (T3, frame 200)
MIN_NB = 3
RE_FILE = re.compile(r"^(LongTrackActive|LongTrackInactive|ExitTrack)_(\d+)\.csv$")


def track_files(conv_dir, after, max_save=None):
    """The saved long-track files of saves s > after (s <= max_save if given): [(kind, s, path)]."""
    out = []
    for p in glob.glob(os.path.join(conv_dir, "*.csv")):
        m = RE_FILE.match(os.path.basename(p))
        if not m:
            continue
        s = int(m.group(2))
        if s > after and (max_save is None or s <= max_save):
            out.append((m.group(1), s, p))
    return sorted(out, key=lambda x: (x[1], x[0]))


def read_tracks(path):
    """[(frames, xyz)] of one track file, each track sorted by frame."""
    df = pd.read_csv(path, usecols=[0, 1, 2, 3, 4], header=0, names=["id", "f", "x", "y", "z"],
                     dtype={"id": np.int64, "f": np.int64, "x": np.float64, "y": np.float64, "z": np.float64})
    if df.empty:
        return []
    a = df.to_numpy()
    ids = a[:, 0]
    cut = np.flatnonzero(np.diff(ids)) + 1
    out = []
    for blk in np.split(a, cut):
        blk = blk[np.argsort(blk[:, 1])]
        out.append((blk[:, 1].astype(np.int64), blk[:, 2:5].astype(np.float64)))
    return out


def alive_at(conv_dir, frame, max_save=None):
    """The tracks alive at the end of `frame`, cut at it: [(frames, xyz)], with the files read.  `frame` must be a
    save frame: a track that fails the line test in frame F has a point at F and goes to the files of the next save,
    so only when that save is F itself does it stay out."""
    if not track_files(conv_dir, frame - 1, frame):
        raise ValueError(f"{conv_dir}: no save at frame {frame}")
    files = track_files(conv_dir, frame, max_save)
    tracks = []
    for _kind, _s, p in files:
        for fr, xyz in read_tracks(p):
            if fr[0] <= frame <= fr[-1]:
                keep = fr <= frame
                tracks.append((fr[keep], xyz[keep]))
    return tracks, [os.path.basename(p) for _k, _s, p in files]


def active_file(conv_dir, frame):
    """The tracks of LongTrackActive_<frame>.csv (alive at the end of that frame, as saved)."""
    p = os.path.join(conv_dir, f"LongTrackActive_{frame}.csv")
    return read_tracks(p) if os.path.exists(p) else None


def measures(tracks, radius=RADIUS_MM):
    """The quality measures of tracks that all end at the same frame."""
    long_tracks = [(fr, p) for fr, p in tracks if len(fr) >= 4]
    q = lambda x, pc: float(np.percentile(x, pc)) if len(x) else float("nan")  # noqa: E731
    steps, res = [], []
    for fr, p in long_tracks:
        steps.append(np.linalg.norm(np.diff(p, axis=0), axis=1))
        res.append(float(np.linalg.norm(p[-1] - (LF_W[:, None] * p[-4:]).sum(axis=0))))
    steps = np.concatenate(steps) if steps else np.array([])
    # neighbour check at the last frame: tracks with points at F-2, F-1, F (tracks have no gaps)
    pos = np.array([p[-1] for fr, p in long_tracks if len(fr) >= 3 and fr[-1] - fr[-3] == 2])
    vel = np.array([(p[-1] - p[-3]) / 2.0 for fr, p in long_tracks if len(fr) >= 3 and fr[-1] - fr[-3] == 2])
    rel, n_cov = [], 0
    if len(pos) > MIN_NB:
        tree = cKDTree(pos)
        for i, nb in enumerate(tree.query_ball_point(pos, r=radius)):
            nb = [j for j in nb if j != i]
            if len(nb) < MIN_NB:
                continue
            n_cov += 1
            vm = np.median(vel[nb], axis=0)
            nv = np.linalg.norm(vm)
            if nv > 0:
                rel.append(float(np.linalg.norm(vel[i] - vm) / nv))
    rel = np.array(rel)
    return {"tracks": len(long_tracks), "step_med": q(steps, 50), "step_p90": q(steps, 90),
            "lf_med": q(res, 50), "lf_p90": q(res, 90),
            "nb_radius_mm": radius, "nb_coverage": n_cov / len(pos) if len(pos) else float("nan"),
            "nb_med": q(rel, 50), "nb_p90": q(rel, 90),
            "nb_over_half": float(np.mean(rel > 0.5)) if len(rel) else float("nan")}


def quality_at(run_dir, frame, max_save=None, radius=RADIUS_MM, use_active=False):
    """Measures of the tracks alive at the end of `frame` in a run's output folder (use_active: read
    LongTrackActive_<frame>.csv instead of rebuilding)."""
    conv = os.path.join(run_dir, "ConvergeTrack")
    if use_active:
        tracks = active_file(conv, frame)
        files = [f"LongTrackActive_{frame}.csv"]
        if tracks is None:
            return None
    else:
        tracks, files = alive_at(conv, frame, max_save)
    m = measures(tracks, radius)
    m["frame"] = frame
    m["files"] = files
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("frame", type=int)
    ap.add_argument("--max-save", type=int, default=None)
    ap.add_argument("--radius", type=float, default=RADIUS_MM)
    ap.add_argument("--active", action="store_true", help="read LongTrackActive_<frame>.csv instead of rebuilding")
    args = ap.parse_args()
    m = quality_at(args.run_dir, args.frame, args.max_save, args.radius, args.active)
    for k, v in (m or {}).items():
        print(f"{k:14s} {v}")


if __name__ == "__main__":
    main()
