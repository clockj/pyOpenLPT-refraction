"""G3 (plan: 2026-10-08-plan.md, Part G): does the time-shifted control estimate the share of ghosts, where the truth
is known?

    python g3_control.py [--set Gall12]          (another G-all set: runs <set>_r2 and <set>_r2_control)
    python g3_control.py --set Gall12_m15 --run Gall12m15_r2      (runs RUN and RUN_control; outputs g3_control_Gall12m15)

On G-all (data/gall): Gall_r2 (the run) and Gall_r2_control (cam2's images 40 frames later, so every track is a
ghost), both from g2_realism.py.  T1-e's estimators (cases/cyl_20261007/tests/t1e_ghost_control.py: its stats and
swirl functions are imported), applied as on T1:
  - counts: the control's active long tracks over the run's at the same frame (the logs); T1-e quoted the last frame's
    ratio, an upper bound;
  - swirl: per long track at the last frame (>= 4 frames), the mean |v_r| / |v| over its steps >= 8 mm from the axis;
    g = (the run's share above 0.3) / (the control's), assuming real tracks stay below 0.3.
The truth (eval_study.track_labels): each of the run's long tracks (the long and exit files, >= 4 frames) labelled by
its points' nearest particles within 0.2 mm; a ghost track when no particle holds the majority of its points
(purity < 0.5).  Per frame t, the tracks that cover t with >= 4 points up to t (the active long tracks, as near as
the final files show them) and the share of them that are ghosts, beside the counts estimator at t.
Writes g3_control_out.txt, .json and g3_control.png.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np

import study
import eval_study

sys.path.insert(0, os.path.join(study.CASE_T1, "tests"))
from s3_track import parse_log  # noqa: E402
from t1e_ghost_control import swirl  # noqa: E402



def as_arrays(tracks, kinds, keep=("LongTrackActive", "LongTrackInactive", "ExitTrack")):
    return {k: np.array(sorted(v, key=lambda r: r[0])) for k, v in tracks.items() if kinds[k] in keep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="Gall")
    ap.add_argument("--case", default="gall")
    ap.add_argument("--run", default=None, help="the run's name, if not <set>_r2 (its control: <run>_control)")
    args = ap.parse_args()
    CASE = os.path.join(study.DATA, args.case)
    run = args.run or f"{args.set}_r2"
    RUN = os.path.join(study.RUNS, run)
    CONTROL = os.path.join(study.RUNS, f"{run}_control")
    stem = "g3_control" if run == "Gall_r2" else f"g3_control_{run[:-3] if run.endswith('_r2') else run}"
    log = study.Log(stem)
    out = {"set": args.set}
    lr, lc = parse_log(os.path.join(RUN, "log.txt")), parse_log(os.path.join(CONTROL, "log.txt"))
    nr, nc = lr["active_long_tracks"], lc["active_long_tracks"]
    n = min(len(nr), len(nc))
    log(f"G3: the control's ghost estimate on G-all; run {os.path.basename(RUN)} ({len(nr)} frames logged), control "
        f"{os.path.basename(CONTROL)} ({len(nc)})")
    labels, kinds, tracks = eval_study.track_labels(RUN, CASE, args.set, [0, 1])
    # the truth per frame: long tracks covering t with >= 4 points up to t, and the ghosts among them
    pur = {}
    for key, lab in labels.items():
        if kinds[key] == "ShortTrackActive":
            continue
        c = Counter(p for _t, p in lab if p >= 0)
        pid, m = c.most_common(1)[0] if c else (-1, 0)
        pur[key] = (pid, m / len(lab))
    t_first = min(t for lab in labels.values() for t, _p in lab)
    last = max(t for lab in labels.values() for t, _p in lab)
    per = []
    for t in range(t_first, last + 1):
        act = [k for k, lab in labels.items() if k in pur and lab[0][0] <= t <= lab[-1][0]
               and sum(1 for tt, _p in lab if tt <= t) >= 4]
        gh = sum(1 for k in act if pur[k][1] < 0.5)
        per.append({"t": t, "active_long": len(act), "ghosts": gh, "ghost_share": gh / max(1, len(act))})
    out["truth_per_frame"] = per
    marks = sorted({k for k in (9, 19, 29, n - 1) if 0 <= k < n})
    rows = []
    for k in marks:
        tr = next((r for r in per if r["t"] == k), None)
        rows.append({"log_entry": k + 1, "run": nr[k], "control": nc[k], "counts_estimate": nc[k] / max(1, nr[k]),
                     "truth_ghost_share": tr["ghost_share"] if tr else None,
                     "truth_active_long": tr["active_long"] if tr else None})
        log(f"  log entry {k + 1}: active long tracks run {nr[k]}, control {nc[k]}: counts estimate "
            f"{nc[k] / max(1, nr[k]):.3f}; truth (frame {k}): {tr['ghosts'] if tr else '-'} ghosts of "
            f"{tr['active_long'] if tr else '-'} long tracks ({100 * tr['ghost_share']:.1f} %)" if tr else "")
    out["marks"] = rows
    out["counts_estimate_last"] = nc[n - 1] / max(1, nr[n - 1])
    # the swirl estimator at the control's last frame
    T_run = as_arrays(tracks, kinds)
    tracks_c, kinds_c = eval_study.read_tracks(CONTROL)
    T_ctl = as_arrays(tracks_c, kinds_c)
    last_c = max(int(t) for v in T_ctl.values() for t in v[:, 0])
    wr, _ = swirl(T_run, last_c)
    wc, _ = swirl(T_ctl, last_c)
    sw = {}
    for nm, w in (("run", wr), ("control", wc)):
        w = w[np.isfinite(w)]
        sw[nm] = {"tracks": int(len(w)), "pcts": np.percentile(w, [10, 50, 90]).tolist() if len(w) else None,
                  "share_above_0.3": float(np.mean(w > 0.3)) if len(w) else None}
        log(f"  swirl, {nm}: {len(w)} long tracks with steps >= 8 mm from the axis; mean |v_r| / |v| per track "
            f"10/50/90 % {np.round(sw[nm]['pcts'], 3).tolist() if len(w) else '-'}; above 0.3: "
            f"{100 * sw[nm]['share_above_0.3']:.1f} %")
    g = sw["run"]["share_above_0.3"] / max(sw["control"]["share_above_0.3"], 1e-9)
    sw["ghost_share_estimate"] = g
    out["swirl"] = sw
    # the truth at the control's last frame, as the swirl estimator counts them (>= 4 frames, steps >= 8 mm)
    keys = [k for k, v in T_run.items() if len(v[v[:, 0] <= last_c]) >= 4]
    gh = [pur[k][1] < 0.5 for k in keys if k in pur]
    out["truth_ghost_share_tracks_last"] = float(np.mean(gh)) if gh else None
    # the tracks' own swirl, real against ghost (does the swirl test separate them on G-all?)
    real_w, ghost_w = [], []
    for k in keys:
        w, _ = swirl({k: T_run[k]}, last_c)
        if len(w) and np.isfinite(w[0]):
            (ghost_w if pur[k][1] < 0.5 else real_w).append(float(w[0]))
    out["swirl_by_truth"] = {"real_share_above_0.3": float(np.mean(np.array(real_w) > 0.3)) if real_w else None,
                             "ghost_share_above_0.3": float(np.mean(np.array(ghost_w) > 0.3)) if ghost_w else None,
                             "real": len(real_w), "ghost": len(ghost_w)}
    log(f"  ghosts among the run's long tracks: from the swirl {100 * g:.0f} %, from the counts (last frame) "
        f"{100 * out['counts_estimate_last']:.0f} %; the truth (long tracks at frame {last_c}, >= 4 frames, purity < 0.5)"
        f" {100 * (out['truth_ghost_share_tracks_last'] or 0):.0f} %")
    sb = out["swirl_by_truth"]
    log(f"  the swirl test by the truth: real tracks above 0.3: "
        f"{100 * (sb['real_share_above_0.3'] or 0):.1f} % of {sb['real']}; ghost tracks above 0.3: "
        f"{100 * (sb['ghost_share_above_0.3'] or 0):.1f} % of {sb['ghost']}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(np.arange(1, len(nr) + 1), nr, label="run (G-all)")
    ax[0].plot(np.arange(1, len(nc) + 1), nc, label="control (all ghosts)")
    ax[0].plot([r["t"] + 1 for r in per], [r["active_long"] for r in per], ":", label="run, long tracks (files)")
    ax[0].plot([r["t"] + 1 for r in per], [r["ghosts"] for r in per], "--", label="run, of them ghosts (truth)")
    ax[0].set_xlabel("frame (log entry)")
    ax[0].set_ylabel("active long tracks")
    ax[0].legend(fontsize=8)
    ax[1].plot([r["t"] + 1 for r in per], [r["ghost_share"] for r in per], label="truth")
    ax[1].plot(np.arange(1, n + 1), [nc[k] / max(1, nr[k]) for k in range(n)], label="counts estimate")
    ax[1].axhline(g, ls="--", c="k", lw=0.8, label="swirl estimate")
    ax[1].set_xlabel("frame (log entry)")
    ax[1].set_ylabel("share of ghosts among the long tracks")
    ax[1].set_ylim(0, 1)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(study.HERE, f"{stem}.png"), dpi=110)
    plt.close(fig)
    log.save(out)


if __name__ == "__main__":
    main()
