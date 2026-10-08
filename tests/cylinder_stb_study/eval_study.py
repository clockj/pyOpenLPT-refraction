"""Score an STB run of a study case against its truth: tests/cylinder_stb/eval_truth.py extended to n cameras, layers
and a view volume (plan: 2026-10-08-plan.md, Part S's scoring).

    evaluate(run_dir, case_dir, set_name, cams, r_match=0.2, first=4, frames=None) -> dict

Truth (synth_study.py): data/<case>/arrays: X (F, N, 3), layer (N,), the rendered pixels uv_<variant>_c<k> (F, N, 2)
of the set's variant; sets.json: the set's layers and variant.  A particle takes part in frame t when its layer is
rendered in the set; n_seen(t, i) = how many of the run's cameras (cams, e.g. [0, 1, 2]) have its image on their
sensors.  Every rig tracks the same view volume (T1's).
Per frame t >= FIRST (and in FRAMES if given):
  - reconstructed: the points of every track in frame t (the run's ConvergeTrack files);
  - matched: within R_MATCH mm of a particle seen by >= 2 cameras, nearest first, each particle once; duplicates: within
    R_MATCH of a particle already matched; ghosts: within R_MATCH of no particle seen by >= 2 cameras;
  - recall, over the particles inside the view volume: matched / present, for those seen by all of the run's cameras
    ("all"), by at least n - 1 ("n-1", 3 and 4 cameras) and by at least 2 ("ge2"), separately for the central layer
    ('in', r < 30 mm) and the rest;
  - position error of the matched (RMS per axis).
Per run: long tracks (the long and exit files): count, mean length, purity (share of a track's points, in the scored
frames, whose nearest particle within R_MATCH is the track's majority particle), ghost tracks (purity 0), track recall
(particle-frames seen by >= 2 cameras inside the volume and covered by a long track whose majority particle it is,
over all such particle-frames), and the time per frame from the log.
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

import study
from eval_truth import frame_times, read_tracks

LAYERS = ["in", "outA", "outB"]


def truth(case_dir, set_name, cams):
    sets = study.load_json(os.path.join(case_dir, "sets.json"))
    s = sets[set_name]
    a = lambda n: np.load(os.path.join(case_dir, "arrays", n + ".npy"), mmap_mode="r")
    X = np.asarray(a("X"))
    layer = np.asarray(a("layer"))
    member = np.isin(layer, [LAYERS.index(ly) for ly in s["layers"]])
    n_seen = np.zeros(X.shape[:2], np.int16)
    for k in cams:
        uv = np.asarray(a(f"uv_{s['variant']}_c{k}"))
        on = np.isfinite(uv[..., 0]) & (uv[..., 0] >= 0) & (uv[..., 0] < 1280) & (uv[..., 1] >= 0) & (uv[..., 1] < 800)
        n_seen += on
    n_seen[:, ~member] = 0
    return X, layer, n_seen


def evaluate(run_dir, case_dir, set_name, cams, r_match=0.2, first=4, frames=None, frame_offset=0):
    """frame_offset: the run's frame t is the truth's frame t + frame_offset (cam1's frames, for G3's control)."""
    X, layer, n_seen = truth(case_dir, set_name, cams)
    vol = study.view_volume()
    n = len(cams)
    tracks, kinds = read_tracks(run_dir)
    by_frame = defaultdict(list)
    for key, pts in tracks.items():
        for (t, x, y, z) in pts:
            by_frame[t].append((key, np.array([x, y, z])))
    F = X.shape[0]
    run_frames = sorted(t for t in range(F - frame_offset) if t >= first and (frames is None or t in frames))
    cats = {"all": lambda k: k >= n, "ge2": lambda k: k >= 2}
    if n >= 3:
        cats["n-1"] = lambda k: k >= n - 1
    tally = {f"{c}_{g}": [0, 0] for c in cats for g in ("in", "out")}
    per_frame, errs, assign = [], [], {}
    covered = set()
    for t in run_frames:
        tt = t + frame_offset
        ok2 = n_seen[tt] >= 2
        ids = np.flatnonzero(ok2)
        rec = by_frame.get(t, [])
        matched_ids = set()
        n_dup = n_ghost = n_match = 0
        if len(ids):
            tree = cKDTree(X[tt, ids])
            cand = []
            for kk, (key, Xr) in enumerate(rec):
                d, j = tree.query(Xr)
                if d <= r_match:
                    cand.append((d, kk, j))
                    assign[(key, t)] = int(ids[j])
                else:
                    n_ghost += 1
            cand.sort()
            used_p = set()
            for d, kk, j in cand:
                if j in used_p:
                    n_dup += 1
                    continue
                used_p.add(j)
                n_match += 1
                matched_ids.add(int(ids[j]))
                errs.append(rec[kk][1] - X[tt, ids[j]])
        else:
            n_ghost = len(rec)
        inv = study.in_volume(X[tt], vol)
        for c, f in cats.items():
            for g, gm in (("in", layer == 0), ("out", layer != 0)):
                sel = np.flatnonzero(inv & gm & f(n_seen[tt]))
                tally[f"{c}_{g}"][0] += sum(1 for i in sel if int(i) in matched_ids)
                tally[f"{c}_{g}"][1] += len(sel)
        per_frame.append({"t": t, "present_ge2_in_volume": int(np.sum(inv & ok2)), "reconstructed": len(rec),
                          "matched": n_match, "duplicates": n_dup, "ghosts": n_ghost})
    errs = np.array(errs) if errs else np.zeros((0, 3))
    # long tracks
    scored = set(run_frames)
    long_keys = [k for k in tracks if kinds[k] != "ShortTrackActive"]
    purity, lengths = [], []
    for key in long_keys:
        pts = [p for p in tracks[key] if p[0] in scored]
        if not pts:
            continue
        lengths.append(len(tracks[key]))
        labels = [assign.get((key, t)) for (t, _x, _y, _z) in pts]
        cnt = Counter(lb for lb in labels if lb is not None)
        if cnt:
            pid, m = cnt.most_common(1)[0]
            purity.append(m / len(pts))
            covered.update((pid, t) for (t, _x, _y, _z), lb in zip(pts, labels) if lb == pid)
        else:
            purity.append(0.0)
    track_den = 0
    track_num = 0
    for t in run_frames:
        tt = t + frame_offset
        sel = np.flatnonzero(study.in_volume(X[tt], vol) & (n_seen[tt] >= 2))
        track_den += len(sel)
        track_num += sum(1 for i in sel if (int(i), t) in covered)
    tot = lambda k: sum(r[k] for r in per_frame)
    nf = max(1, len(per_frame))
    ft = frame_times(run_dir)
    out = {
        "frames_scored": len(per_frame), "cameras": n,
        "reconstructed_per_frame": tot("reconstructed") / nf, "matched_per_frame": tot("matched") / nf,
        "ghosts_per_frame": tot("ghosts") / nf, "duplicates_per_frame": tot("duplicates") / nf,
        "ghost_share": tot("ghosts") / max(1, tot("reconstructed")),
        "present_ge2_in_volume_per_frame": tot("present_ge2_in_volume") / nf,
        "recall": {k: (v[0] / v[1] if v[1] else None) for k, v in tally.items()},
        "present_per_frame": {k: v[1] / nf for k, v in tally.items()},
        "pos_rms_um": (1e3 * np.sqrt(np.mean(errs ** 2, axis=0))).tolist() if len(errs) else None,
        "long_tracks": len(lengths), "long_track_mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "long_track_purity_mean": float(np.mean(purity)) if purity else 0.0,
        "long_track_purity_ge_0.9": float(np.mean(np.array(purity) >= 0.9)) if purity else 0.0,
        "long_tracks_ghost": int(np.sum(np.array(purity) == 0.0)) if purity else 0,
        "track_recall": track_num / max(1, track_den),
        "frame_time_s_mean": float(np.mean(ft)) if ft else None,
        "per_frame": per_frame,
    }
    return out


def track_labels(run_dir, case_dir, set_name, cams, r_match=0.2, frame_offset=0):
    """Every track of a run with its points' particles: {key: [(t, particle id or -1), ...]}, the kinds, and the
    tracks' points; a point's particle is the nearest one seen by >= 2 of the cameras within R_MATCH mm."""
    X, _layer, n_seen = truth(case_dir, set_name, cams)
    tracks, kinds = read_tracks(run_dir)
    trees = {}
    out = {}
    for key, pts in tracks.items():
        lab = []
        for (t, x, y, z) in pts:
            tt = t + frame_offset
            if tt not in trees:
                ids = np.flatnonzero(n_seen[tt] >= 2)
                trees[tt] = (cKDTree(X[tt, ids]), ids) if len(ids) else (None, ids)
            tree, ids = trees[tt]
            pid = -1
            if tree is not None:
                d, j = tree.query([x, y, z])
                if d <= r_match:
                    pid = int(ids[j])
            lab.append((t, pid))
        out[key] = lab
    return out, kinds, tracks


def line(name, r):
    """One summary line of a scored run."""
    rc = r["recall"]
    parts = [f"recall (central, seen by all) {rc['all_in']:.3f}" if rc.get("all_in") is not None else "recall -"]
    if "n-1_in" in rc and rc["n-1_in"] is not None:
        parts.append(f">= n-1 {rc['n-1_in']:.3f}")
    if rc.get("ge2_in") is not None:
        parts.append(f">= 2 {rc['ge2_in']:.3f}")
    if rc.get("ge2_out") is not None and r["present_per_frame"]["ge2_out"] > 0:
        parts.append(f"outside the central region, >= 2: {rc['ge2_out']:.3f} of "
                     f"{r['present_per_frame']['ge2_out']:.0f}")
    pos = r["pos_rms_um"]
    return (f"{name}: {', '.join(parts)}; reconstructed {r['reconstructed_per_frame']:.0f} per frame, ghosts "
            f"{r['ghosts_per_frame']:.0f} ({100 * r['ghost_share']:.1f} %), duplicates {r['duplicates_per_frame']:.1f}; "
            f"position RMS " + (", ".join(f"{v:.1f}" for v in pos) + " um" if pos else "-")
            + f"; long tracks {r['long_tracks']} (ghost {r['long_tracks_ghost']}), purity "
            f"{r['long_track_purity_mean']:.3f}, track recall {r['track_recall']:.3f}; "
            + (f"{r['frame_time_s_mean']:.2f} s per frame" if r["frame_time_s_mean"] else "-"))
