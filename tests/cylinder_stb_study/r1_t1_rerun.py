"""R1 (plan: 2026-10-08-plan.md, end of section 7, "decision D3's rerun of T1"): does a 15 px median help T1's tracking?

    python r1_t1_rerun.py

Three runs of T1 (cases/cyl_20261007, build p4c; final: frames 0-250; control: 0-100 with cam2's images 100 steps
later), alike except for the preprocessing's median window and the line detector's threshold:
  base   S1's frames (a 31 px median), k_detect 6: the case's s3_final and s3_control;
  m15    a 15 px median (s1_preprocess.py --median 15), k_detect 6: s3_final_m15 and s3_control_m15;
  m15k5  a 15 px median and k_detect 5: s3_final_m15k5 and s3_control_m15k5.
T1 has no truth, so each run is judged by:
  - the detections per frame (IPR's first loop) and the objects IPR finds per frame (the logs);
  - T1-e's counts: the run's active long tracks per frame against its control's, whose tracks are all ghosts;
  - T1-e's swirl test: the share of the long tracks at frame 100 whose radial velocity is above 0.3 of their speed,
    against the control's.  Both estimates of the ghost share read 1.5-3.5x the truth on the study's synthetic cases
    (G3), so they are compared between runs, not taken at face value;
  - the real tracks they leave: the long tracks at frame 100 (at least 4 frames) x (1 - the ghost share);
  - recurrence: a long track of run A recurs in run B if at least half of its points have a long-track point of B
    within 0.1 mm (about 3 px) at the same frame.  A ghost is a chance pairing of two detections, which a change of
    preprocessing reshuffles; a real particle is there in every run.  So the tracks of A that recur in B are mostly
    real, and the share of each run's tracks that recur in the others shows how many of them are;
  - T1-a to T1-d (t1_checks.py): residuals, shapes, flow.
Reads the runs (cases/cyl_20261007/data/runs) and the checks' outputs beside the case's scripts; writes
r1_t1_rerun_out.txt, .json and r1_t1_rerun.png beside itself.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
from scipy.spatial import cKDTree

import study

CASE_TESTS = os.path.join(study.CASE_T1, "tests")
sys.path.insert(0, study.CASE_T1)
sys.path.insert(0, CASE_TESTS)
from t1_checks import read_long_tracks  # noqa: E402
from t1e_ghost_control import swirl  # noqa: E402

RUNS_DIR = os.path.join(study.CASE_T1, "data", "runs")
VARIANTS = [("base", "31 px median, k 6", "s3_final", "s3_control"),
            ("m15", "15 px median, k 6", "s3_final_m15", "s3_control_m15"),
            ("m15k5", "15 px median, k 5", "s3_final_m15k5", "s3_control_m15k5")]
R_RECUR = 0.1           # mm
CUT = 100               # the controls' last frame


def frame_log(path):
    """Per frame of the log: the first IPR loop's detections (cam1, cam2), the objects IPR found, the active long
    tracks, the time."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    out = []
    for block in re.split(r"(?=(?:Initial|Convergence) phase at frame \d+)", text)[1:]:
        fr = int(re.match(r"\w+ phase at frame (\d+)", block).group(1))
        d = re.search(r"2D detections per active camera: (\d+)\s+(\d+)", block)
        o = re.search(r"IPR FINISH! FOUND (\d+) OBJECTS", block)
        lt = re.search(r"No\. of active long tracks: \d+ \+ \d+ - \d+ = (\d+)", block)
        t = re.search(r"Total time for frame \d+: ([0-9.]+)", block)
        if t is None:
            continue
        out.append({"frame": fr, "det": [int(d.group(1)), int(d.group(2))] if d else None,
                    "objects": int(o.group(1)) if o else None, "long": int(lt.group(1)) if lt else None,
                    "seconds": float(t.group(1))})
    return out


def cut(tracks, last, min_len=4):
    """The tracks cut at frame last, kept if they span at least min_len consecutive frames (T1-e's rule)."""
    out = {}
    for k, tr in tracks.items():
        tr = tr[tr[:, 0] <= last]
        if len(tr) >= min_len and not np.any(np.diff(tr[:, 0]) != 1):
            out[k] = tr
    return out


def recurrence(ta, tb, r=R_RECUR):
    """Per track of ta: does it recur in tb (at least half of its points within r of one of tb's points at the same
    frame)?"""
    pts = {}
    for tr in tb.values():
        for row in tr:
            pts.setdefault(int(row[0]), []).append(row[1:4])
    trees = {f: cKDTree(np.array(p)) for f, p in pts.items()}
    out = {}
    for k, tr in ta.items():
        hit = 0
        for row in tr:
            t = trees.get(int(row[0]))
            if t is not None and t.query(row[1:4])[0] <= r:
                hit += 1
        out[k] = hit >= 0.5 * len(tr)
    return out


def main():
    log = study.Log("r1_t1_rerun")
    res = {"variants": {}}
    runs = {}
    for name, label, rf, rc in VARIANTS:
        v = {"label": label, "final": rf, "control": rc}
        fl, cl = frame_log(os.path.join(RUNS_DIR, rf, "log.txt")), frame_log(os.path.join(RUNS_DIR, rc, "log.txt"))
        with open(os.path.join(RUNS_DIR, rf, f"config_{rf[3:]}.json"), encoding="utf-8") as fh:   # s3_X: config_X
            tracer = json.load(fh)["objects"][0]["tracer"]
        v["lines"] = tracer["lines"]
        v["frames_final"], v["frames_control"] = len(fl), len(cl)
        det = np.array([f["det"] for f in fl if f["det"]], float)
        v["detections_per_frame"] = det.mean(axis=0).round(1).tolist()
        v["ipr_objects_per_frame"] = float(np.mean([f["objects"] for f in fl if f["objects"] is not None]))
        v["seconds_per_frame"] = float(np.mean([f["seconds"] for f in fl]))
        v["long_final"] = {f["frame"]: f["long"] for f in fl if f["long"] is not None}
        v["long_control"] = {f["frame"]: f["long"] for f in cl if f["long"] is not None}
        # T1-e's counts, at the frames T1-e reports (log entries 20 ... 97: frames 23 ... 100)
        ratio = {}
        for fr in (23, 43, 63, 83, CUT):
            if fr in v["long_final"] and fr in v["long_control"]:
                ratio[fr] = v["long_control"][fr] / max(v["long_final"][fr], 1)
        v["counts_ratio"] = ratio
        tf, last_f = read_long_tracks(os.path.join(RUNS_DIR, rf))
        tc, last_c = read_long_tracks(os.path.join(RUNS_DIR, rc))
        v["last_frame_final"], v["last_frame_control"] = last_f, last_c
        tf_cut, tc_cut = cut(tf, last_c), cut(tc, last_c)
        wf, _ = swirl(tf, last_c)
        wc, _ = swirl(tc, last_c)
        wf_all, _ = swirl(tf, last_f)
        sh = lambda w: float(np.mean(w[np.isfinite(w)] > 0.3))  # noqa: E731
        g_counts = ratio.get(CUT)
        g_swirl = sh(wf) / max(sh(wc), 1e-9)
        n100 = len(tf_cut)
        v.update({
            "long_tracks_at_100": n100, "control_long_tracks_at_100": len(tc_cut),
            "swirl_above_0.3_final": sh(wf), "swirl_above_0.3_control": sh(wc),
            "swirl_above_0.3_final_at_end": sh(wf_all),
            "ghost_share_counts": g_counts, "ghost_share_swirl": g_swirl,
            "real_at_100_counts": n100 * (1 - g_counts) if g_counts is not None else None,
            "real_at_100_swirl": n100 * (1 - g_swirl),
            "long_tracks_total": len(tf),
            "long_track_points_total": int(sum(len(t) for t in tf.values())),
            "length_pcts": np.percentile([len(t) for t in tf.values()], [10, 50, 90]).tolist(),
            "swirl_final_values": [float(x) for x in wf[np.isfinite(wf)]],
            "swirl_control_values": [float(x) for x in wc[np.isfinite(wc)]],
        })
        # T1-a to T1-d
        sfx = "" if rf == "s3_final" else f"_{rf}"
        p = os.path.join(CASE_TESTS, f"t1_checks{sfx}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                ck = json.load(fh)
            v["checks"] = {k: ck[k] for k in ("T1a_cam1", "T1a_cam2", "T1a_epipolar", "T1b_sync", "T1c_cam1",
                                               "T1c_cam2") if k in ck}
            v["checks"]["T1d"] = {k: ck["T1d"][k] for k in ("long_tracks", "length_pcts", "mean_length",
                                                             "speed_mm_s_median", "accel_rms_over_speed_rms")}
        runs[name] = {"all": tf, "cut": tf_cut, "swirl_cut": dict(zip(tf_cut.keys(), wf))}
        res["variants"][name] = v

    for name, v in res["variants"].items():
        log(f"{name} ({v['label']}; detector noise {np.round(v['lines']['noise'], 2).tolist()}"
            f"{', k ' + str(v['lines']['k_detect']) if 'k_detect' in v['lines'] else ''}): "
            f"{v['frames_final']} frames, {v['seconds_per_frame']:.1f} s per frame")
        log(f"  detections per frame {v['detections_per_frame']}; IPR objects per frame {v['ipr_objects_per_frame']:.0f}")
        log(f"  active long tracks, final / control: " + "; ".join(
            f"frame {fr} {v['long_final'][fr]} / {v['long_control'][fr]} = {r:.3f}" for fr, r in v["counts_ratio"].items()))
        log(f"  long tracks at frame {CUT} (>= 4 frames): final {v['long_tracks_at_100']}, control "
            f"{v['control_long_tracks_at_100']}; swirl above 0.3: final {v['swirl_above_0.3_final'] * 100:.1f} %, "
            f"control {v['swirl_above_0.3_control'] * 100:.1f} % (final at frame {v['last_frame_final']}: "
            f"{v['swirl_above_0.3_final_at_end'] * 100:.1f} %)")
        gc = v["ghost_share_counts"]
        log(f"  ghost share: counts {gc * 100:.0f} %, swirl {v['ghost_share_swirl'] * 100:.0f} %; real long tracks at "
            f"frame {CUT}: {v['real_at_100_counts']:.0f} (counts), {v['real_at_100_swirl']:.0f} (swirl)")
        log(f"  the whole run: {v['long_tracks_total']} long tracks, {v['long_track_points_total']} points; length "
            f"10/50/90 % {v['length_pcts']}")
        if "checks" in v:
            c = v["checks"]
            log(f"  T1-a: RMS du, dv cam1 {c['T1a_cam1']['rms_du']:.2f}, {c['T1a_cam1']['rms_dv']:.2f}; cam2 "
                f"{c['T1a_cam2']['rms_du']:.2f}, {c['T1a_cam2']['rms_dv']:.2f} px; epipolar mean "
                f"{c['T1a_epipolar']['mean']:+.3f} +- {c['T1a_epipolar']['sem']:.3f}, RMS {c['T1a_epipolar']['rms']:.2f} px "
                f"(n {c['T1a_epipolar']['n']})")
            log(f"  T1-c: |width error| 68 % cam1 {c['T1c_cam1']['wh_q68']:.2f} / {c['T1c_cam1']['wv_q68']:.2f}, cam2 "
                f"{c['T1c_cam2']['wh_q68']:.2f} / {c['T1c_cam2']['wv_q68']:.2f} px (horizontal / vertical)")
            d = c["T1d"]
            log(f"  T1-d: {d['long_tracks']} long tracks, length 10/50/90 % {d['length_pcts']}; speed median "
                f"{d['speed_mm_s_median']:.1f} mm/s; second difference / step {d['accel_rms_over_speed_rms']:.3f}")

    # recurrence between the runs, on the tracks cut at frame 100 and on the whole runs
    rec = {}
    names = [n for n, *_ in VARIANTS]
    for key, which in (("at_100", "cut"), ("whole", "all")):
        rec[key] = {}
        for a in names:
            for b in names:
                if a == b:
                    continue
                r = recurrence(runs[a][which], runs[b][which])
                vals = np.array(list(r.values()))
                entry = {"tracks": int(len(vals)), "recur": int(vals.sum()), "share": float(vals.mean())}
                if which == "cut":
                    w = runs[a]["swirl_cut"]
                    lo = np.array([r[k] for k in r if np.isfinite(w.get(k, np.nan)) and w[k] <= 0.3])
                    hi = np.array([r[k] for k in r if np.isfinite(w.get(k, np.nan)) and w[k] > 0.3])
                    entry.update({"share_swirl_le_0.3": float(lo.mean()) if len(lo) else None, "n_le": int(len(lo)),
                                  "share_swirl_gt_0.3": float(hi.mean()) if len(hi) else None, "n_gt": int(len(hi))})
                rec[key][f"{a}_in_{b}"] = entry
    res["recurrence"] = rec
    log(f"recurrence (a track of A recurs in B: at least half its points within {R_RECUR} mm of B's at the same frame):")
    for key in ("at_100", "whole"):
        for ab, e in rec[key].items():
            a, b = ab.split("_in_")
            extra = ""
            if "share_swirl_le_0.3" in e and e["share_swirl_le_0.3"] is not None:
                extra = (f"; swirl <= 0.3: {e['share_swirl_le_0.3'] * 100:.0f} % of {e['n_le']}, > 0.3: "
                         f"{e['share_swirl_gt_0.3'] * 100:.0f} % of {e['n_gt']}")
            log(f"  {'frames 0-100' if key == 'at_100' else 'whole run'}: {a}'s {e['tracks']} tracks in {b}: "
                f"{e['recur']} recur ({e['share'] * 100:.0f} %){extra}")

    # the figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"base": "#4c72b0", "m15": "#dd8452", "m15k5": "#55a868"}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for name, v in res["variants"].items():
        f = sorted(v["long_final"])
        ax[0].plot(f, [v["long_final"][k] for k in f], color=col[name], label=f"{name}: {v['label']}")
        c = sorted(v["long_control"])
        ax[0].plot(c, [v["long_control"][k] for k in c], color=col[name], ls="--", lw=1)
    ax[0].set_xlabel("frame (every 4th raw frame)")
    ax[0].set_ylabel("active long tracks")
    ax[0].set_title("runs (solid) and their controls, all ghosts (dashed)", fontsize=9)
    ax[0].legend(fontsize=8)
    x = np.arange(len(names))
    vs = [res["variants"][n] for n in names]
    ax[1].bar(x - 0.27, [v["long_tracks_at_100"] for v in vs], 0.27, color="#bbbbbb", label="long tracks")
    ax[1].bar(x, [v["real_at_100_counts"] for v in vs], 0.27, color="#8172b3", label="real, by the counts")
    ax[1].bar(x + 0.27, [v["real_at_100_swirl"] for v in vs], 0.27, color="#64b5cd", label="real, by the swirl")
    ax[1].set_xticks(x, names)
    ax[1].set_ylabel(f"tracks at frame {CUT}")
    ax[1].set_title("long tracks and the real ones the two ghost estimates leave", fontsize=9)
    ax[1].legend(fontsize=8)
    bins = np.linspace(0, 1, 26)
    for name, v in res["variants"].items():
        ax[2].hist(v["swirl_final_values"], bins=bins, histtype="step", density=True, color=col[name], label=name)
        ax[2].hist(v["swirl_control_values"], bins=bins, histtype="step", density=True, color=col[name], ls="--",
                   lw=0.8)
    ax[2].axvline(0.3, color="k", lw=0.6)
    ax[2].set_xlabel("per track: mean |v_r| / |v| (r >= 8 mm), frames 0-100")
    ax[2].set_title("swirl test: runs (solid), controls (dashed)", fontsize=9)
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(study.HERE, "r1_t1_rerun.png"), dpi=110)
    plt.close(fig)
    for v in res["variants"].values():
        v["long_final"] = {str(k): x for k, x in v["long_final"].items()}
        v["long_control"] = {str(k): x for k, x in v["long_control"].items()}
        v["counts_ratio"] = {str(k): x for k, x in v["counts_ratio"].items()}
    log.save(res)


if __name__ == "__main__":
    main()
