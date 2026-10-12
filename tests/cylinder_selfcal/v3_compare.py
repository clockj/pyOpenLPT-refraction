"""V3 and M2 (plan, section 3): T1's runs compared end to end, as the study's r1_t1_rerun.py compares them.

    python v3_compare.py LABEL=FINAL,CONTROL,GATHER [...] [--ref m15] [--name v3_compare]

Each variant: its final run (frames 0-250) and control (0-100, cam2 100 steps later: every track a ghost) in the case's
data/runs/, and a gather.py of its final run on its own frames (every 5th frame, depths).  T1 has no truth, so:
  - the logs: detections per frame (IPR's first loop), active long tracks per frame in the final and the control;
  - T1-e's counts estimate of the ghost share at frame 100 (the control's active long tracks over the final's), and its
    swirl estimate (the share of long tracks at frame 100 whose mean |v_r| / |v| exceeds 0.3, over the control's);
    both read 2-3x the truth on the study's synthetic cases, so they are compared between runs; the counts estimate
    follows the true gain there (the study's section 7);
  - the real long tracks at frame 100 (>= 4 frames) these leave, and the whole run's long tracks and points;
  - recurrence against the reference run: a track recurs when at least half its points have one of the other run's
    long-track points within 0.1 mm at the same frame (ghosts are reshuffled by any change, real particles are not);
  - the tracks' quality on the gathered frames (the detections within 3 px of the tracks' projections, both cameras):
    the residuals du, dv (robust SD and RMS), the offset across the epipolar curve (median, robust SD, RMS), the share
    of the tracks' points with both detections; the widths against S2's model (|measured - predicted| 68 %); the flow
    (speed, the second difference over the step, lengths).
Writes <name>_out.txt, .json and <name>.png.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import selfcal as sc

sys.path.insert(0, sc.STUDY)
from r1_t1_rerun import cut, frame_log, recurrence  # noqa: E402  (the study's functions)
from t1e_ghost_control import swirl  # noqa: E402  (the case's)

CUT = 100
STEP_S = 4 / 4000.0


def track_quality(gname, model):
    recs = sc.load_gather(gname)
    fd = sc.fitdata(recs, sc.R_PAIR)
    n_all = sum(len(g["X"]) for g in recs)
    m = [((g["pair"] >= 0) & (g["pair_dist"] <= sc.R_PAIR)).all(axis=1) & np.all(np.isfinite(g["J"].reshape(len(g["X"]), -1)), axis=1)
         for g in recs if len(g["X"])]
    e12 = np.concatenate([g["epi12"][mm] for g, mm in zip([g for g in recs if len(g["X"])], m)])
    rsd = lambda x: float(1.4826 * np.median(np.abs(x - np.median(x))))  # noqa: E731
    out = {"points": int(n_all), "paired_both": int(len(fd)), "paired_share": len(fd) / max(n_all, 1)}
    for c in (0, 1):
        for k, ax in ((0, "du"), (1, "dv")):
            r = fd.r[:, c, k]
            out[f"cam{c + 1}_{ax}"] = {"robust_sd": rsd(r), "rms": float(np.sqrt(np.mean(r ** 2))),
                                       "median": float(np.median(r))}
    out["epi"] = {"median": float(np.nanmedian(e12)), "robust_sd": rsd(e12[np.isfinite(e12)]),
                  "rms": float(np.sqrt(np.nanmean(e12 ** 2)))}
    # the widths against S2's model
    for c in (0, 1):
        z = np.concatenate([g[f"z_track{c}"][mm] for g, mm in zip([g for g in recs if len(g["X"])], m)])
        d = np.concatenate([g[f"det{c}"][g["pair"][mm, c]] for g, mm in zip([g for g in recs if len(g["X"])], m)])
        cm = model[c]
        ok = np.all(np.isfinite(z), axis=1)
        Lh = cm["A_mm"] * cm["f_px"] * np.abs(1 / (z[ok, 0] + cm["delta_mm"]) - 1 / cm["z_f_mm"])
        Lv = cm["A_mm"] * cm["f_px"] * np.abs(1 / (z[ok, 1] + cm["delta_mm"]) - 1 / cm["z_f_mm"])
        s0 = cm["sigma0_px"]
        wh = np.sqrt(d[ok, 2] ** 2 / 12 + d[ok, 4] ** 2) - np.sqrt(Lh ** 2 / 16 + s0 ** 2)
        wv = np.sqrt(d[ok, 3] ** 2 / 12 + d[ok, 5] ** 2) - np.sqrt(Lv ** 2 / 16 + s0 ** 2)
        out[f"cam{c + 1}_width_q68"] = [float(np.percentile(np.abs(wh), 68)), float(np.percentile(np.abs(wv), 68))]
    return out


def flow(tracks):
    lens, rough, speed = [], [], []
    for tr in tracks.values():
        if len(tr) < 5 or np.any(np.diff(tr[:, 0]) != 1):
            continue
        X = tr[:, 1:4]
        v = np.diff(X, axis=0)
        a = np.diff(X, 2, axis=0)
        lens.append(len(tr))
        speed.append(np.linalg.norm(v, axis=1))
        rough.append(np.sqrt((a ** 2).sum(axis=1).mean()) / max(np.sqrt((v ** 2).sum(axis=1).mean()), 1e-12))
    sp = np.concatenate(speed)
    return {"length_pcts": np.percentile(lens, [10, 50, 90]).tolist(), "speed_median_mm_s": float(np.median(sp) / STEP_S),
            "roughness_median": float(np.median(rough))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--ref", default=None, help="the label recurrence is measured against (default: the first)")
    ap.add_argument("--name", default="v3_compare")
    a = ap.parse_args()
    log = sc.Log(a.name)
    model = sc.load_json(os.path.join(sc.CASE_DATA, "T1_shape.json"))["cameras"]
    var = []
    for v in a.variants:
        label, rest = v.split("=")
        rf, rc, gname = rest.split(",")
        var.append((label, rf, rc, gname))
    ref = a.ref or var[0][0]
    res, runs = {}, {}
    for label, rf, rc, gname in var:
        df, dc = os.path.join(sc.CASE_RUNS, rf), os.path.join(sc.CASE_RUNS, rc)
        fl, cl = frame_log(os.path.join(df, "log.txt")), frame_log(os.path.join(dc, "log.txt"))
        lf = {f["frame"]: f["long"] for f in fl if f["long"] is not None}
        lc = {f["frame"]: f["long"] for f in cl if f["long"] is not None}
        det = np.array([f["det"] for f in fl if f["det"]], float)
        tf, last_f = sc.read_tracks(df)
        tc, last_c = sc.read_tracks(dc)
        tf_cut, tc_cut = cut(tf, last_c), cut(tc, last_c)
        wf, _ = swirl(tf, last_c)
        wc, _ = swirl(tc, last_c)
        sh = lambda w: float(np.mean(w[np.isfinite(w)] > 0.3))  # noqa: E731
        g_counts = lc.get(CUT, np.nan) / max(lf.get(CUT, 1), 1)
        g_swirl = sh(wf) / max(sh(wc), 1e-9)
        n100 = len(tf_cut)
        v = {"final": rf, "control": rc, "gather": gname, "frames": len(fl), "seconds_per_frame":
             float(np.mean([f["seconds"] for f in fl])), "detections_per_frame": det.mean(axis=0).round(1).tolist(),
             "active_long_100": [lf.get(CUT), lc.get(CUT)], "ghost_counts": g_counts, "ghost_swirl": g_swirl,
             "long_at_100": n100, "control_long_at_100": len(tc_cut),
             "real_at_100_counts": n100 * (1 - g_counts), "real_at_100_swirl": n100 * (1 - g_swirl),
             "long_tracks_total": len(tf), "points_total": int(sum(len(t) for t in tf.values())),
             "flow": flow(tf), "quality": track_quality(gname, model),
             "long_final_series": {str(k): x for k, x in lf.items()}, "long_control_series": {str(k): x for k, x in lc.items()}}
        res[label] = v
        runs[label] = {"cut": tf_cut, "all": tf}
        q = v["quality"]
        log(f"{label}: {rf} / {rc}; {v['frames']} frames, {v['seconds_per_frame']:.1f} s per frame; detections "
            f"{v['detections_per_frame']}")
        log(f"  active long tracks at frame {CUT}: {lf.get(CUT)} (control {lc.get(CUT)}): ghosts by the counts "
            f"{100 * g_counts:.0f} %, by the swirl {100 * g_swirl:.0f} %; long tracks at {CUT} {n100}: real "
            f"{v['real_at_100_counts']:.0f} (counts), {v['real_at_100_swirl']:.0f} (swirl); whole run {len(tf)} long "
            f"tracks, {v['points_total']} points")
        log(f"  quality: {q['paired_both']} of {q['points']} track points with both detections "
            f"({100 * q['paired_share']:.0f} %); robust SD du, dv cam1 {q['cam1_du']['robust_sd']:.3f}, "
            f"{q['cam1_dv']['robust_sd']:.3f}, cam2 {q['cam2_du']['robust_sd']:.3f}, {q['cam2_dv']['robust_sd']:.3f} px "
            f"(RMS {q['cam1_du']['rms']:.2f}, {q['cam1_dv']['rms']:.2f}, {q['cam2_du']['rms']:.2f}, "
            f"{q['cam2_dv']['rms']:.2f}); epipolar offset median {q['epi']['median']:+.3f}, robust SD "
            f"{q['epi']['robust_sd']:.3f}, RMS {q['epi']['rms']:.3f} px; widths |error| 68 % cam1 "
            f"{q['cam1_width_q68'][0]:.2f}/{q['cam1_width_q68'][1]:.2f}, cam2 {q['cam2_width_q68'][0]:.2f}/"
            f"{q['cam2_width_q68'][1]:.2f} px; flow: length 10/50/90 % {v['flow']['length_pcts']}, speed "
            f"{v['flow']['speed_median_mm_s']:.1f} mm/s, roughness {v['flow']['roughness_median']:.3f}")
    rec = {}
    for lab in res:
        if lab == ref:
            continue
        for key, which in (("at_100", "cut"), ("whole", "all")):
            r_ab = recurrence(runs[lab][which], runs[ref][which])
            r_ba = recurrence(runs[ref][which], runs[lab][which])
            va, vb = np.array(list(r_ab.values())), np.array(list(r_ba.values()))
            rec[f"{lab}_{key}"] = {f"{lab}_in_{ref}": [int(va.sum()), int(len(va)), float(va.mean())],
                                   f"{ref}_in_{lab}": [int(vb.sum()), int(len(vb)), float(vb.mean())]}
            log(f"  recurrence {'frames 0-100' if key == 'at_100' else 'whole run'}: {lab}'s {len(va)} tracks in {ref}: "
                f"{int(va.sum())} ({100 * va.mean():.0f} %); {ref}'s {len(vb)} in {lab}: {int(vb.sum())} "
                f"({100 * vb.mean():.0f} %)")
    out = {"variants": res, "recurrence": rec, "ref": ref}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for lab, v in res.items():
        f = sorted(int(k) for k in v["long_final_series"])
        ax[0].plot(f, [v["long_final_series"][str(k)] for k in f], label=lab)
        c = sorted(int(k) for k in v["long_control_series"])
        ax[0].plot(c, [v["long_control_series"][str(k)] for k in c], ls="--", lw=1, color=ax[0].lines[-1].get_color())
    ax[0].set_xlabel("frame")
    ax[0].set_ylabel("active long tracks (dashed: control, all ghosts)")
    ax[0].legend(fontsize=8)
    labs = list(res)
    x = np.arange(len(labs))
    ax[1].bar(x - 0.27, [res[l]["long_at_100"] for l in labs], 0.27, color="#bbbbbb", label="long tracks at 100")
    ax[1].bar(x, [res[l]["real_at_100_counts"] for l in labs], 0.27, color="#8172b3", label="real, by the counts")
    ax[1].bar(x + 0.27, [res[l]["real_at_100_swirl"] for l in labs], 0.27, color="#64b5cd", label="real, by the swirl")
    ax[1].set_xticks(x, labs)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(sc.HERE, f"{a.name}.png"), dpi=100)
    plt.close(fig)
    log.save(out)


if __name__ == "__main__":
    main()
