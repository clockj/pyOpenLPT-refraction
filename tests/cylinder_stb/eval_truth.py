"""Score an STB run of a synthetic case against its truth (plan: Phase 5 metrics).

    python eval_truth.py RUN_DIR CASE_NAME [--r-match 0.2] [--first 4]
    evaluate(run_dir, case_dir, r_match, first) -> dict

Truth: data/CASE/truth.csv (synth.py / synth_lines.py).  The run's tracks: Results/ConvergeTrack/*Track*_<last>.csv
(long active, long inactive, exit and short active).  A particle is trackable in frame t when both cameras see it on
their sensors.  Per frame t >= FIRST:
  - reconstructed: the points of every track in that frame; matched: within R_MATCH mm of a trackable particle (each
    particle used once, nearest first); duplicates: within R_MATCH of a particle already matched; ghosts: the rest
    (near no particle);
  - recall: matched / trackable; position error of the matched (RMS per axis).
Per run: long tracks (the long and exit files): count, mean length, purity (share of a track's points, in the scored
frames, whose nearest particle within R_MATCH is the track's majority particle), and track recall (trackable particle-frames covered by a long track whose majority particle it is), plus
the per-frame time from the log.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

import cstb


def read_truth(case_dir):
    with open(os.path.join(case_dir, "truth.csv"), newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    n_cam = sum(1 for k in rows[0] if re.fullmatch(r"u\d+", k))
    T = defaultdict(dict)
    for r in rows:
        t, i = int(r["frame"]), int(r["id"])
        X = np.array([float(r["X"]), float(r["Y"]), float(r["Z"])])
        seen = all(r[f"u{c + 1}"] != "" for c in range(n_cam))
        T[t][i] = (X, seen)
    return T


def read_tracks(run_dir):
    files = glob.glob(os.path.join(run_dir, "Results", "ConvergeTrack", "*.csv"))
    if not files:
        return {}, {}
    last = max(int(re.search(r"_(\d+)\.csv$", f).group(1)) for f in files)
    tracks = {}
    kinds = {}
    for kind in ("LongTrackActive", "LongTrackInactive", "ExitTrack", "ShortTrackActive"):
        p = os.path.join(run_dir, "Results", "ConvergeTrack", f"{kind}_{last}.csv")
        if not os.path.exists(p):
            continue
        with open(p, newline="", encoding="utf-8") as fh:
            rd = csv.reader(fh)
            next(rd, None)
            for row in rd:
                if not row:
                    continue
                key = (kind, int(row[0]))
                tracks.setdefault(key, []).append((int(row[1]), float(row[2]), float(row[3]), float(row[4])))
                kinds[key] = kind
    return tracks, kinds


def frame_times(run_dir):
    log = os.path.join(run_dir, "log.txt")
    if not os.path.exists(log):
        return []
    with open(log, encoding="utf-8", errors="replace") as fh:
        return [float(m.group(1)) for m in re.finditer(r"Total time for frame \d+: ([0-9.]+)", fh.read())]


def evaluate(run_dir, case_dir, r_match=0.2, first=4):
    T = read_truth(case_dir)
    tracks, kinds = read_tracks(run_dir)
    by_frame = defaultdict(list)                # t -> [(track key, X)]
    for key, pts in tracks.items():
        for (t, x, y, z) in pts:
            by_frame[t].append((key, np.array([x, y, z])))
    frames = sorted(t for t in T if t >= first)
    per_frame = []
    assign = {}                                  # (track key, t) -> particle id
    errs = []
    for t in frames:
        ids = [i for i, (X, s) in T[t].items() if s]
        if not ids:
            continue
        P = np.array([T[t][i][0] for i in ids])
        tree = cKDTree(P)
        rec = by_frame.get(t, [])
        cand = []
        for k, (key, X) in enumerate(rec):
            d, j = tree.query(X)
            if d <= r_match:
                cand.append((d, k, j))
                assign[(key, t)] = ids[j]             # the nearest particle (for the tracks' purity)
        cand.sort()
        used_p, used_r = set(), set()
        n_dup = 0
        for d, k, j in cand:
            if j in used_p:
                n_dup += 1                            # a second reconstruction of a particle already matched
                continue
            used_r.add(k)
            used_p.add(j)
            errs.append(rec[k][1] - P[j])
        per_frame.append({"t": t, "trackable": len(ids), "reconstructed": len(rec), "matched": len(used_r),
                          "duplicates": n_dup, "ghosts": len(rec) - len(cand)})
    errs = np.array(errs) if errs else np.zeros((0, 3))
    # long tracks
    long_keys = [k for k in tracks if kinds[k] != "ShortTrackActive"]
    purity, lengths = [], []
    covered_pairs = set()                         # (particle, frame) covered by a long track whose majority it is
    scored = set(frames)
    for key in long_keys:
        pts = [p for p in tracks[key] if p[0] in scored]     # purity over the scored frames only
        if not pts:
            continue
        lengths.append(len(tracks[key]))
        labels = [assign.get((key, t)) for (t, _x, _y, _z) in pts]
        c = Counter(l for l in labels if l is not None)
        if c:
            pid, n = c.most_common(1)[0]
            purity.append(n / len(pts))
            covered_pairs.update((pid, t) for (t, _x, _y, _z), l in zip(pts, labels) if l == pid)
        else:
            purity.append(0.0)
    trackable_total = sum(r["trackable"] for r in per_frame)
    ft = frame_times(run_dir)
    tot = lambda k: sum(r[k] for r in per_frame)
    return {
        "frames_scored": len(per_frame), "trackable": trackable_total,
        "reconstructed": tot("reconstructed"), "matched": tot("matched"), "duplicates": tot("duplicates"),
        "ghosts": tot("ghosts"),
        "recall": tot("matched") / max(1, trackable_total),
        "ghost_share": tot("ghosts") / max(1, tot("reconstructed")),
        "pos_rms_mm": np.sqrt(np.mean(errs ** 2, axis=0)).tolist() if len(errs) else None,
        "long_tracks": len(lengths), "long_track_mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "long_track_purity_mean": float(np.mean(purity)) if purity else 0.0,
        "long_track_purity_ge_0.9": float(np.mean(np.array(purity) >= 0.9)) if purity else 0.0,
        "long_tracks_ghost": int(np.sum(np.array(purity) == 0.0)) if purity else 0,
        "track_recall": len(covered_pairs) / max(1, trackable_total),
        "frame_time_s_mean": float(np.mean(ft)) if ft else None,
        "per_frame": per_frame,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("case")
    ap.add_argument("--r-match", type=float, default=0.2)
    ap.add_argument("--first", type=int, default=4)
    args = ap.parse_args()
    res = evaluate(args.run_dir, os.path.join(cstb.DATA, args.case), args.r_match, args.first)
    print(json.dumps({k: v for k, v in res.items() if k != "per_frame"}, indent=1))


if __name__ == "__main__":
    main()
