"""The neighbour test for ghost tracks (paper Section 9.3, Fig. 17(c); asked by the author 2026-10-08: "remove the
unreasonable tracks, like the tracks crossing the others ... use surrounding tracks to identify those ghost tracks").

Each long track is judged by the other tracks around it, in the same frame:
  - the tracks are smoothed with the group's Gaussian filter (width W steps; experiment_numbers.smooth), and only
    their smoothed positions take part;
  - at each smoothed position of a track, the neighbours are the other tracks' positions in the same frame within RHO
    mm; with N_MIN or more of them, the point's normalized residual is
        |v - v_med| / (median_j |v_j - v_med| + EPS),
    v_med being the neighbours' component-wise median velocity: the normalized median test of PIV (Westerweel and
    Scarano), written for scattered tracks;
  - a track is dropped when more than half of its judged points exceed R_TH; a track with no judged point is kept.
Nothing assumes the swirl: a track is judged only against the tracks beside it.

    python ghost_filter.py check        the test on the study's synthetic recording matched to T1 (tests/
                                        cylinder_stb_study, run Gallm_r2), where every track's truth is known, over a
                                        small grid of RHO, R_TH and EPS; then on T1 with every setting
    python ghost_filter.py check --set Gall12m15 --run Gall12m15_r2 --case gall12 ...   another synthetic case

Writes ghost_filter_check_out.txt and .json beside itself (with --run: ghost_filter_check_<run>_*).
"""
import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment_numbers as en  # noqa: E402

STUDY = en.REPO / "tests" / "cylinder_stb_study"
T1_TRACKS = en.PROC / "T1" / "Results" / "ConvergeTrack"
LONG_KINDS = ("LongTrackActive", "LongTrackInactive", "ExitTrack")

# The setting used. R_TH = 2 is the normalized median test's usual threshold; RHO and EPS (the velocity noise floor)
# were chosen on the synthetic case matched to T1 (`check`): the most ghosts dropped while dropping under 5 % of the
# real tracks (Gallm_r2: 42 % of the ghosts, 3.7 % of the real tracks; checked on Gall12m15_r2 and Gall_r2: 38 / 2.2 %
# and 28 / 1.3 %).
RHO, N_MIN, R_TH, EPS = 4.0, 3, 2.0, 2.0


def read_long_tracks(folder):
    """{key: rows [frame, X, Y, Z]} of the long tracks (active, inactive, exited) at the last frame."""
    files = glob.glob(str(Path(folder) / "*.csv"))
    last = max(int(Path(f).stem.rsplit("_", 1)[1]) for f in files)
    tracks = {}
    for kind in LONG_KINDS:
        p = Path(folder) / f"{kind}_{last}.csv"
        if not p.exists():
            continue
        with open(p, newline="", encoding="utf-8") as fh:
            rd = csv.reader(fh)
            next(rd, None)
            for row in rd:
                if row:
                    tracks.setdefault((kind, int(row[0])), []).append([float(x) for x in row[1:5]])
    return {k: np.array(sorted(v)) for k, v in tracks.items()}


def smoothed(tracks, w=en.W_FILTER, dt=en.STEP_S):
    """{key: (frames, positions, velocities)} of the tracks long enough for the filter (the ends dropped)."""
    fw = en.gaussian_kernels(w)[0]
    out = {}
    for key, t in tracks.items():
        if len(t) >= 2 * fw + 1:
            xs, vs = en.smooth(t[:, 1:4], w, dt)
            out[key] = (t[fw:len(t) - fw, 0].astype(int), xs, vs)
    return out


def judge(sm, rho=RHO, n_min=N_MIN, r_th=R_TH, eps=EPS):
    """{key: (points judged, points above r_th)} by the normalized median test against the same frame's neighbours."""
    keys = list(sm)
    by_frame = {}
    for i, k in enumerate(keys):
        fr, xs, vs = sm[k]
        for j, f in enumerate(fr):
            by_frame.setdefault(int(f), []).append((i, xs[j], vs[j]))
    res = {k: [0, 0] for k in keys}
    for f, rows in by_frame.items():
        own = np.array([r[0] for r in rows])
        X = np.array([r[1] for r in rows])
        V = np.array([r[2] for r in rows])
        tree = cKDTree(X)
        for a, nb in enumerate(tree.query_ball_point(X, rho)):
            nb = [b for b in nb if own[b] != own[a]]
            if len(nb) < n_min:
                continue
            vm = np.median(V[nb], axis=0)
            rm = np.median(np.linalg.norm(V[nb] - vm, axis=1))
            r = np.linalg.norm(V[a] - vm) / (rm + eps)
            st = res[keys[own[a]]]
            st[0] += 1
            st[1] += int(r > r_th)
    return res


def dropped(res):
    """The keys of the tracks with more than half of their judged points above the threshold."""
    return {k for k, (n, bad) in res.items() if n > 0 and bad > 0.5 * n}


# The swirl test of the case's T1-e (cases/cyl_20261007/tests/t1e_ghost_control.swirl): a track's mean |v_r| / |v|
# over its points at least SWIRL_R_MIN mm from the axis; above SWIRL_MAX it does not follow the swirl. Here on the
# smoothed tracks. A track that never reaches SWIRL_R_MIN (the core, where the vortex's centre is off the axis) gets the
# same test about the core's own centre (core_centre), over its points at least CORE_R_MIN from it. Unlike the
# neighbour test it assumes the flow: Fig. 17(c) uses it to choose the tracks it draws (the author asked for the
# swirling tracks only), and nothing else does.
SWIRL_R_MIN, SWIRL_MAX, CORE_R_MIN = 8.0, 0.3, 1.5


def core_centre(sm, r_core=SWIRL_R_MIN):
    """(X, Z) in the run's frame of the point to which the smoothed velocities of all points within r_core of the axis
    are most nearly perpendicular (least squares: sum (v . (p - c))^2 over the plane across the axis)."""
    P = np.concatenate([xs for _f, xs, _v in sm.values()])
    W = np.concatenate([vs for _f, _x, vs in sm.values()])
    m = np.hypot(P[:, 0], P[:, 2]) < r_core
    p, v = P[m][:, [0, 2]], W[m][:, [0, 2]]
    return np.linalg.solve(v.T @ v, v.T @ np.sum(v * p, axis=1))


def swirl_ratio(sm, centre=(0.0, 0.0), r_min=SWIRL_R_MIN):
    """{key: mean |v_r| / |v| over the track's smoothed points >= r_min from the line along the axis through centre
    ((X, Z) of the run's frame; the axis itself by default), or NaN if it has none}."""
    out = {}
    for k, (_fr, xs, vs) in sm.items():
        px, pz = xs[:, 0] - centre[0], xs[:, 2] - centre[1]
        r = np.hypot(px, pz)
        sp = np.linalg.norm(vs, axis=1)
        ok = (r >= r_min) & (sp > 0)
        vr = (px * vs[:, 0] + pz * vs[:, 2]) / np.maximum(r, 1e-9)
        out[k] = float(np.mean(np.abs(vr[ok]) / sp[ok])) if ok.any() else float("nan")
    return out


def off_swirl(sm, r_max=SWIRL_MAX):
    """The keys of the tracks that do not follow the swirl: by the test about the axis, or, for a track that never
    reaches SWIRL_R_MIN from it, about the core's centre."""
    ax = swirl_ratio(sm)
    co = swirl_ratio(sm, core_centre(sm), CORE_R_MIN)
    return {k for k in sm if ax[k] > r_max or (np.isnan(ax[k]) and co[k] > r_max)}


def labels_synthetic(run, case, set_name):
    """{key: True if a ghost} for the long tracks of a study run: no true particle holds the majority of its points
    (eval_study.track_labels, as the study's G3)."""
    sys.path.insert(0, str(STUDY))
    import eval_study  # noqa: E402
    lab, kinds, _tracks = eval_study.track_labels(str(STUDY / "runs" / run), str(STUDY / "data" / case), set_name, [0, 1])
    out = {}
    for key, pts in lab.items():
        if kinds[key] not in LONG_KINDS:
            continue
        ids = [p for _t, p in pts if p >= 0]
        top = int(max(np.unique(ids, return_counts=True)[1])) if ids else 0
        out[key] = bool(top < 0.5 * len(pts))
    return out


def check(args):
    stem = "ghost_filter_check" if args.run == "Gallm_r2" else f"ghost_filter_check_{args.run}"
    lines, out = [], {"run": args.run, "set": args.set, "case": args.case}

    def log(s):
        print(s)
        lines.append(s)

    syn = read_long_tracks(STUDY / "runs" / args.run / "Results" / "ConvergeTrack")
    truth = labels_synthetic(args.run, args.case, args.set)
    sm = smoothed(syn)
    keys = [k for k in sm if k in truth]
    g = int(sum(truth[k] for k in keys))
    log(f"synthetic {args.run}: {len(syn)} long tracks, {len(sm)} long enough for the filter (>= 19 positions), "
        f"{g} of them ghosts ({100 * g / max(len(keys), 1):.1f} %)")
    t1 = smoothed(read_long_tracks(T1_TRACKS))
    for name, s in (("synthetic", sm), ("T1", t1)):
        sp = np.concatenate([np.linalg.norm(v[2], axis=1) for v in s.values()])
        nb = []
        allp = {}
        for k, (fr, xs, _vs) in s.items():
            for f, x in zip(fr, xs):
                allp.setdefault(int(f), []).append(x)
        for f, xs in allp.items():
            tr = cKDTree(np.array(xs))
            nb.extend(len(n) - 1 for n in tr.query_ball_point(np.array(xs), RHO))
        log(f"  {name}: smoothed speed median {np.median(sp):.1f} mm/s; neighbours within {RHO:.0f} mm in the same "
            f"frame: median {np.median(nb):.0f} (10-90 %: {np.percentile(nb, 10):.0f}-{np.percentile(nb, 90):.0f})")
    t1_len = {k: len(v[0]) + 18 for k, v in t1.items()}
    log(f"T1: {len(t1)} long tracks long enough for the filter; {sum(n >= 100 for n in t1_len.values())} of >= 100 "
        f"positions (Fig. 17(c))")
    log("  rho  r_th  eps | synthetic: ghosts dropped, real dropped, ghost share kept (before -> after) | "
        "T1: dropped, of >= 100 positions")
    out["grid"] = []
    for rho in (3.0, 4.0, 5.0):
        for r_th in (2.0, 3.0):
            for eps in (2.0, 5.0, 10.0):
                d = dropped(judge(sm, rho, N_MIN, r_th, eps))
                gd = sum(1 for k in keys if truth[k] and k in d)
                rd = sum(1 for k in keys if not truth[k] and k in d)
                kept = [k for k in keys if k not in d]
                gk = int(sum(truth[k] for k in kept))
                dt1 = dropped(judge(t1, rho, N_MIN, r_th, eps))
                long_t1 = [k for k in t1 if t1_len[k] >= 100]
                dl = sum(1 for k in long_t1 if k in dt1)
                row = {"rho": rho, "r_th": r_th, "eps": eps, "ghosts_dropped": gd, "ghosts": g,
                       "real_dropped": rd, "real": len(keys) - g, "ghost_share_before": g / len(keys),
                       "ghost_share_after": gk / max(len(kept), 1), "t1_dropped": len(dt1), "t1_tracks": len(t1),
                       "t1_long_dropped": dl, "t1_long": len(long_t1)}
                out["grid"].append(row)
                log(f"  {rho:3.0f}  {r_th:4.1f} {eps:4.0f} | {gd:3d}/{g:3d} ({100 * gd / max(g, 1):4.1f} %), "
                    f"{rd:3d}/{len(keys) - g:3d} ({100 * rd / max(len(keys) - g, 1):4.1f} %), "
                    f"{100 * g / len(keys):4.1f} -> {100 * gk / max(len(kept), 1):4.1f} % | "
                    f"{len(dt1):4d}/{len(t1)} ({100 * len(dt1) / len(t1):4.1f} %), {dl}/{len(long_t1)}")
    # the track rule at the chosen setting: more than half of the judged points flagged (used), or 30 % (stricter)
    rs, r1 = judge(sm), judge(t1)
    long_t1 = [k for k in t1 if t1_len[k] >= 100]
    out["rules"] = {}
    for name, share in (("majority (used)", 0.5), ("30 %", 0.3)):
        d = {k for k in keys if rs[k][0] > 0 and rs[k][1] > share * rs[k][0]}
        gd = sum(1 for k in d if truth[k])
        rd = len(d) - gd
        dl = sum(1 for k in long_t1 if r1[k][0] > 0 and r1[k][1] > share * r1[k][0])
        out["rules"][name] = {"ghosts_dropped": gd, "ghosts": g, "real_dropped": rd, "real": len(keys) - g,
                              "t1_long_dropped": dl, "t1_long": len(long_t1)}
        log(f"rule '{name}' at rho {RHO:g}, r_th {R_TH:g}, eps {EPS:g}: synthetic ghosts {gd}/{g} "
            f"({100 * gd / max(g, 1):.1f} %), real {rd}/{len(keys) - g} ({100 * rd / max(len(keys) - g, 1):.1f} %); "
            f"T1 long tracks dropped {dl}/{len(long_t1)}")
    # the swirl test (display only), alone and after the neighbour test
    ds, d1s = off_swirl(sm), off_swirl(t1)
    dn, d1n = dropped(rs), dropped(r1)
    out["swirl"] = {}
    for name, dsyn, dt1 in (("swirl test alone", ds, d1s), ("neighbour test, then the swirl test", dn | ds, d1n | d1s)):
        gd = sum(1 for k in keys if k in dsyn and truth[k])
        rd = sum(1 for k in keys if k in dsyn and not truth[k])
        kept = [k for k in keys if k not in dsyn]
        gk = sum(truth[k] for k in kept)
        dl = sum(1 for k in long_t1 if k in dt1)
        out["swirl"][name] = {"ghosts_dropped": gd, "ghosts": g, "real_dropped": rd, "real": len(keys) - g,
                              "ghost_share_after": gk / max(len(kept), 1), "t1_long_dropped": dl,
                              "t1_long": len(long_t1)}
        log(f"{name} (mean |v_r|/|v| > {SWIRL_MAX:g} at r >= {SWIRL_R_MIN:g} mm): synthetic ghosts {gd}/{g} "
            f"({100 * gd / max(g, 1):.1f} %), real {rd}/{len(keys) - g} ({100 * rd / max(len(keys) - g, 1):.1f} %), "
            f"ghost share {100 * g / len(keys):.1f} -> {100 * gk / max(len(kept), 1):.1f} %; T1 long tracks dropped "
            f"{dl}/{len(long_t1)}")
    with open(HERE / f"{stem}_out.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(HERE / f"{stem}.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["check"])
    ap.add_argument("--run", default="Gallm_r2")
    ap.add_argument("--set", default="Gallm")
    ap.add_argument("--case", default="gallm")
    args = ap.parse_args()
    check(args)


if __name__ == "__main__":
    main()
