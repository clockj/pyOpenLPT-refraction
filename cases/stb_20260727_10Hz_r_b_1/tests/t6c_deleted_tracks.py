"""T6c: are the tracks that the linear-fit test deletes unreliable?  (asked on 5 Oct, from T4's saved tracks)

STB deletes a long track when the last of its last four points lies more than 0.05 mm from the straight line through
them (STB::checkLinearFit).  From the tracks T4 (new camFiles) saved at frame 500, for every long track that ended
inactive (>= 7 points; the exit tracks left the view volume instead):
  - glitch or curvature: the last point's error against a constant-velocity and a constant-acceleration prediction from
    the 6 points before it, in units of the track's own noise; and the acceleration a quadratic fit finds over the 7
    points before the end;
  - other rules on the same points: the share of deleted tracks that a looser limit (0.075, 0.1, 0.15 mm), a quadratic
    check (5 points), or a noise-relative limit would have kept at that point;
  - continuation: whether another saved track starts within 10 frames where the deleted track's particle should be
    (extrapolated from its last good points), moving the same way; if so the deletion only cut a real trajectory;
  - the short tracks still active at frame 500 (1-3 points): their displacement per frame against the tracked
    particles' 0.08-0.12 mm/frame, to see how many are not real particles;
  - how robust the continuation result is: radius 0.2-1 mm, any velocity, windows of 10 and 30 frames, short tracks
    included as candidates;
  - where the deleted tracks end: how many cameras see their last good point (against their middle point and the
    exit tracks' last point), and the deletions per tracked point over the view volume (6 x 6 bins in each plane).

Writes t6c_deleted_tracks.json and t6c_deleted_tracks_out.txt beside this script.
    python tests/t6c_deleted_tracks.py [new|old]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import case  # noqa: E402

RUNS = {"new": "new_cams", "old": "old_cams"}
LIMIT = 0.05
R_CONT = 0.2      # mm: a continuation's first point within this of the extrapolated position
V_CONT = 0.03     # mm/frame: and its velocity within this of the deleted track's
_lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def read(path):
    d = pd.read_csv(path, usecols=[0, 1, 2, 3, 4])
    d.columns = ["tid", "frame", "x", "y", "z"]
    return d


def tracks(d):
    """{tid: (frames, P)} for a track file."""
    out = {}
    P = d[["x", "y", "z"]].to_numpy(float)
    F = d["frame"].to_numpy(int)
    for tid, ii in d.groupby("tid", sort=False).indices.items():
        out[tid] = (F[ii], P[ii])
    return out


def poly_predict(P, deg, t_pred):
    """LS polynomial (degree deg) through points P at t = 0..k-1, evaluated at t_pred, per axis."""
    t = np.arange(len(P), dtype=float)
    V = np.vander(t, deg + 1)
    coef, *_ = np.linalg.lstsq(V, P, rcond=None)
    return np.vander(np.array([t_pred], float), deg + 1) @ coef, coef


def lf_resid(P4):
    w = np.array([-0.2, 0.1, 0.4, 0.7])
    return float(np.linalg.norm(P4[-1] - w @ P4))


def quad_resid5(P5):
    """Residual of the last point from a LS quadratic through the last 5 points."""
    est, _ = poly_predict(P5, 2, len(P5) - 1)
    return float(np.linalg.norm(P5[-1] - est[0]))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "new"
    cfg = case.load_config()
    conv = os.path.join(cfg["out_dir"], RUNS[which], "Results", "ConvergeTrack")
    save = 500
    inact = tracks(read(os.path.join(conv, f"LongTrackInactive_{save}.csv")))
    others = {}
    for kind in ("ExitTrack", "LongTrackInactive", "LongTrackActive"):
        p = os.path.join(conv, f"{kind}_{save}.csv")
        for tid, v in tracks(read(p)).items():
            others[(kind, tid)] = v
    log(f"T6c ({which} camFiles, tracks saved at frame {save}): {len(inact)} inactive long tracks; "
        f"{len(others)} saved tracks to search for continuations")

    # continuation candidates indexed by start frame
    starts = {}
    for key, (F, P) in others.items():
        if len(P) < 4:
            continue
        v0 = (P[3] - P[0]) / 3.0
        starts.setdefault(int(F[0]), []).append((key, P[0], v0))
    trees = {f: (cKDTree(np.array([s[1] for s in lst])), lst) for f, lst in starts.items()}

    rows = []
    for tid, (F, P) in inact.items():
        n = len(P)
        if n < 9:
            continue
        good = P[:-1]
        d2 = good[2:] - 2 * good[1:-1] + good[:-2]
        sig = float(np.median(1.4826 * np.median(np.abs(d2 - np.median(d2, 0)), 0)) / np.sqrt(6))
        sig = max(sig, 1e-4)
        e_lin = float(np.linalg.norm(P[-1] - poly_predict(P[-7:-1], 1, 6)[0][0]))
        e_quad = float(np.linalg.norm(P[-1] - poly_predict(P[-7:-1], 2, 6)[0][0]))
        _, coef = poly_predict(P[-8:-1], 2, 0)
        acc = float(np.linalg.norm(2 * coef[0]))                # mm / frame^2
        r_lin = lf_resid(P[-4:])
        r_q5 = quad_resid5(P[-5:])
        # continuation: extrapolate from the last good point with the velocity of the 5 points before it
        v = poly_predict(P[-6:-1], 1, 0)[1][0]
        p_last_good, f_last_good = P[-2], F[-2]
        cont, gap = False, None
        for s in range(int(F[-1]), int(F[-1]) + 11):
            if s not in trees:
                continue
            tree, lst = trees[s]
            pred = p_last_good + v * (s - f_last_good)
            for j in tree.query_ball_point(pred, R_CONT):
                key, q0, v0 = lst[j]
                if key == ("LongTrackInactive", tid):
                    continue
                if np.linalg.norm(v0 - v) < V_CONT:
                    cont, gap = True, s - int(F[-1])
                    break
            if cont:
                break
        rows.append({"tid": tid, "n": n, "sigma": sig, "e_lin": e_lin, "e_quad": e_quad, "acc": acc, "r_lin": r_lin,
                     "r_q5": r_q5, "speed": float(np.linalg.norm(v)), "cont": cont, "gap": gap})
    df = pd.DataFrame(rows)
    lf = df[df["r_lin"] > LIMIT]
    log(f"  inactive long tracks with >= 9 points: {len(df)}; of them ended by the linear-fit test (last point > "
        f"{LIMIT} mm): {len(lf)}")
    q = lambda s, p: float(np.percentile(s, p))  # noqa: E731
    log(f"  their own noise per axis before the end: median {lf['sigma'].median():.4f} mm")
    log(f"  last point against a prediction from the 6 points before it: constant velocity median "
        f"{lf['e_lin'].median():.4f} mm ({(lf['e_lin'] / lf['sigma']).median():.0f} sigma), constant acceleration "
        f"median {lf['e_quad'].median():.4f} mm ({(lf['e_quad'] / lf['sigma']).median():.0f} sigma)")
    log(f"  acceleration fitted over the 7 points before the end: median {lf['acc'].median():.5f}, p90 "
        f"{q(lf['acc'], 90):.5f} mm/frame^2 (a 0.05 mm linear-fit residual needs 0.1 mm/frame^2)")
    res = {"which": which, "n_inactive_ge9": int(len(df)), "n_ended_by_lf": int(len(lf)),
           "sigma_median": float(lf["sigma"].median()), "e_lin_median": float(lf["e_lin"].median()),
           "e_quad_median": float(lf["e_quad"].median()), "acc_median": float(lf["acc"].median()),
           "acc_p90": q(lf["acc"], 90)}
    kept = {}
    for lim in (0.075, 0.1, 0.15):
        kept[f"linear {lim} mm"] = float(np.mean(lf["r_lin"] <= lim))
    kept["quadratic 5 points, 0.05 mm"] = float(np.mean(lf["r_q5"] <= LIMIT))
    for k in (6, 8, 10):
        kept[f"linear, k={k} x own sigma (min 0.05)"] = float(np.mean(lf["r_lin"] <= np.maximum(LIMIT, k * lf["sigma"])))
    log("  share of these deletions that another rule would have avoided at that point: "
        + "; ".join(f"{k} {v:.2f}" for k, v in kept.items()))
    res["kept_by_other_rules"] = kept
    c = lf["cont"]
    gaps = lf.loc[c, "gap"]
    log(f"  continued by another saved track within 10 frames (start within {R_CONT} mm of the extrapolated "
        f"position, velocity within {V_CONT} mm/frame): {c.mean():.2f} ({c.sum()} of {len(lf)}); gap median "
        f"{gaps.median() if len(gaps) else float('nan'):.0f} frames (p10 {q(gaps, 10) if len(gaps) else float('nan'):.0f},"
        f" p90 {q(gaps, 90) if len(gaps) else float('nan'):.0f})")
    res["continued_share"] = float(c.mean())
    res["gap_median"] = float(gaps.median()) if len(gaps) else None
    res["gap_hist"] = {int(k): int(v) for k, v in gaps.value_counts().sort_index().items()}

    # the short tracks still active at frame 500
    sh = tracks(read(os.path.join(conv, f"ShortTrackActive_{save}.csv")))
    disp = {2: [], 3: []}
    for tid, (F, P) in sh.items():
        if len(P) in disp:
            disp[len(P)].append(float(np.median(np.linalg.norm(np.diff(P, axis=0), axis=1))))
    long_speed = df["speed"]
    log(f"  short tracks active at frame {save}: {len(sh)} ({sum(len(v) for v in disp.values())} with 2-3 points); "
        f"tracked long tracks move {q(long_speed, 10):.3f}-{q(long_speed, 90):.3f} mm/frame (p10-p90)")
    for k, v in disp.items():
        v = np.array(v)
        if len(v):
            log(f"    {k} points: displacement per frame median {np.median(v):.3f} mm; within 0.06-0.16 mm/frame "
                f"(as the tracked particles): {np.mean((v > 0.06) & (v < 0.16)):.2f}; over 0.5 mm/frame: "
                f"{np.mean(v > 0.5):.2f}")
            res[f"short_{k}pt"] = {"n": int(len(v)), "median_disp": float(np.median(v)),
                                   "share_like_tracked": float(np.mean((v > 0.06) & (v < 0.16))),
                                   "share_over_0.5": float(np.mean(v > 0.5))}
    # robustness of the continuation result (short tracks included as candidates)
    for key, v in tracks(read(os.path.join(conv, f"ShortTrackActive_{save}.csv"))).items():
        F, P = v
        k = min(3, len(P) - 1)
        starts.setdefault(int(F[0]), []).append((("ShortTrackActive", key), P[0], (P[k] - P[0]) / max(1, k)))
    trees = {f: (cKDTree(np.array([s[1] for s in lst])), lst) for f, lst in starts.items()}
    ends = []
    for tid, (F, P) in inact.items():
        if len(P) >= 9 and lf_resid(P[-4:]) > LIMIT:
            ends.append((tid, F[-2], P[-2], poly_predict(P[-6:-1], 1, 0)[1][0], F[-1]))
    sens = []
    for R, V, W in ((0.2, 0.03, 10), (0.5, 0.05, 10), (1.0, 0.08, 10), (0.5, 0.05, 30), (1.0, np.inf, 10)):
        n = 0
        for tid, fg, pg, v, fe in ends:
            found = False
            for s in range(int(fe), int(fe) + W + 1):
                if s in trees:
                    tree, lst = trees[s]
                    for j in tree.query_ball_point(pg + v * (s - fg), R):
                        k2, _q0, v0 = lst[j]
                        if k2 != ("LongTrackInactive", tid) and np.linalg.norm(v0 - v) < V:
                            found = True
                            break
                if found:
                    break
            n += found
        sens.append({"R_mm": R, "V_mm_per_frame": None if np.isinf(V) else V, "window": W, "continued": n,
                     "share": n / max(1, len(ends))})
        log(f"  continuation, radius {R} mm, velocity within {V} mm/frame, {W} frames, short tracks included: "
            f"{n}/{len(ends)} = {n / max(1, len(ends)):.3f}")
    res["continuation_sensitivity"] = sens

    # where the deleted tracks end: cameras seeing the point, and the deletions over the view volume
    cams = case.load_cameras(case.camfile_paths(cfg, which))

    def n_cams(X):
        k = np.zeros(len(X), int)
        for cm in cams:
            uv, ok = case.project(cm, X)
            k += ok & case.in_image(uv, cm)
        return k
    end_good = np.array([e[2] for e in ends])
    mids = np.array([P[len(P) // 2] for tid, (F, P) in inact.items() if len(P) >= 9 and lf_resid(P[-4:]) > LIMIT])
    ex = tracks(read(os.path.join(conv, f"ExitTrack_{save}.csv")))
    ex_end = np.array([P[-1] for F, P in ex.values()])
    res["cameras_seeing"] = {}
    for name, X in (("deleted, last good point", end_good), ("deleted, middle point", mids),
                    ("exit tracks, last point", ex_end)):
        k = n_cams(X)
        share = {str(m): float(np.mean(k == m)) for m in (4, 3, 2)} | {"<=1": float(np.mean(k <= 1))}
        res["cameras_seeing"][name] = share
        log(f"  cameras seeing the {name} ({len(X)}): 4 {share['4']:.2f}, 3 {share['3']:.2f}, 2 {share['2']:.2f}, "
            f"<=1 {share['<=1']:.2f}")
    if which == "new":
        with open(os.path.join(case.TESTS, "t0_frame_map.json"), encoding="utf-8") as f:
            vv = json.load(f)["new_view_volume"]
    else:
        vv = cfg["old_view_volume"]
    pts = []
    for kind in ("LongTrackInactive", "LongTrackActive", "ExitTrack"):
        for F, P in tracks(read(os.path.join(conv, f"{kind}_{save}.csv"))).values():
            pts.append(P[::5])
    pts = np.concatenate(pts)
    res["deletion_rate_bins"] = {}
    for a1, a2 in ((0, 2), (0, 1), (2, 1)):
        e1 = np.linspace(vv[2 * a1], vv[2 * a1 + 1], 7)
        e2 = np.linspace(vv[2 * a2], vv[2 * a2 + 1], 7)
        He, _, _ = np.histogram2d(end_good[:, a1], end_good[:, a2], bins=[e1, e2])
        Hp, _, _ = np.histogram2d(pts[:, a1], pts[:, a2], bins=[e1, e2])
        rate = np.where(Hp > 200, He / np.maximum(Hp, 1) * (len(pts) / max(1, len(end_good))), np.nan)
        name = "xyz"[a1] + "-" + "xyz"[a2]
        res["deletion_rate_bins"][name] = {"rows": "xyz"[a1], "cols": "xyz"[a2], "edges_rows": e1, "edges_cols": e2,
                                           "relative_rate": rate}
        log(f"  deletions per tracked point relative to the mean, {'xyz'[a1]} rows (low to high) x {'xyz'[a2]} "
            f"columns: p10 {np.nanpercentile(rate, 10):.2f}, p90 {np.nanpercentile(rate, 90):.2f}")
        for r in rate:
            log("     " + " ".join(f"{v:5.2f}" if np.isfinite(v) else "    -" for v in r))

    out_json = os.path.join(case.TESTS, f"t6c_deleted_tracks_{which}.json")
    case.save_json(out_json, res)
    with open(os.path.join(case.TESTS, f"t6c_deleted_tracks_{which}_out.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
