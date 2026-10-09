"""Every table of the report (2026-10-08-report.md), from the saved results of this folder and the study's runs.

    python summary_sc.py

Reads v0_pattern.json, v0b_truthfree.json, v1_benefit.json, v2_fit_*.json, sruns_*.json (and the study's s_runs.json,
g2_realism.json for the references), v3_compare*.json, m01_shape.json.  Writes summary_sc_out.txt and
summary_sc.png (V2: before / after / clean with 2, 3 and 4 cameras).
"""
from __future__ import annotations

import glob
import os

import numpy as np

import selfcal as sc


def load(p):
    p = os.path.join(sc.HERE, p)
    return sc.load_json(p) if os.path.exists(p) else None


def runs_mine():
    out = {}
    for p in glob.glob(os.path.join(sc.HERE, "sruns_*.json")):
        d = sc.load_json(p)
        for k, v in d.get("runs", {}).items():
            out[k] = v
    return out


def study_runs():
    s = sc.load_json(os.path.join(sc.STUDY, "s_runs.json"))["runs"]
    g = sc.load_json(os.path.join(sc.STUDY, "g2_realism.json"))["runs"]
    return {**{k: v for k, v in s.items()}, **{k: v for k, v in g.items()}}


def row(name, r):
    if r is None or "score" not in r:
        return f"| {name} | – | – | – | – | – | – | – |"
    s = r["score"]
    rc = s["recall"]
    pos = s["pos_rms_um"]
    n1 = rc.get("n-1_in")
    return (f"| {name} | {rc['all_in']:.3f} | {'–' if n1 is None else f'{n1:.3f}'} | {rc['ge2_in']:.3f} | "
            f"{100 * s['ghost_share']:.1f} % | {pos[0]:.0f} / {pos[1]:.0f} / {pos[2]:.0f} | "
            f"{s['long_track_purity_mean']:.3f} | {s['long_tracks']} |")


def main():
    log = sc.Log("summary_sc")
    mine, study = runs_mine(), study_runs()
    log("## V0")
    v0 = load("v0_pattern.json")
    if v0:
        m = v0["maps"]["epi12_by_cam2"]
        log(f"pattern (epi12 by cam2 cell, noise off) {m['stats']['rms_noise_corrected']:.3f} px; halves corr "
            f"{m['halves_corr']:.2f}; truth-free (ML, 320 px) corr {v0['truth_free']['corr_with_tracks']:.2f}; field "
            f"offset RMS {v0['field']['offset_rms_px']:.3f} px; points' SD {v0['field']['points_offset_sd_before_after']}")
    log("## V1")
    v1 = load("v1_benefit.json")
    if v1:
        log("| field | within 1 px | true at 0.5 | 0.75 | 1.0 | 1.5 px | random at 1.0 | 1.5 px |")
        log("|---|---|---|---|---|---|---|---|")
        for k, v in v1["variants"].items():
            t = v["tol"]
            log(f"| {k} | {100 * v['profile']['1.00']:.0f} % | {t['0.5']['true_per_frame']:.0f} | "
                f"{t['0.75']['true_per_frame']:.0f} | {t['1.0']['true_per_frame']:.0f} | {t['1.5']['true_per_frame']:.0f}"
                f" | {t['1.0']['random_per_frame']:.0f} | {t['1.5']['random_per_frame']:.0f} |")
    log("## V2, 2-4 cameras: the field's recovery")
    log("| cameras | track points | observable truth RMS | fit's error | fit / truth | vs truth + the tracked points' "
        "deviation: error, slope | the deviation's RMS |")
    log("|---|---|---|---|---|---|---|")
    for n in (2, 3, 4):
        f = load(f"v2_fit_cal4_r{n}.json")
        if not f:
            continue
        o = f["tracks"]["observable"]
        b = f["tracks"].get("vs_truth_plus_bias", {})
        log(f"| {n} | {f['points']} | {o['rms_true']:.3f} | {o['rms_error']:.3f} | {o['slope']:.2f} | "
            f"{b.get('rms_error', float('nan')):.3f}, {b.get('slope', float('nan')):.2f} | "
            f"{b.get('bias_rms', float('nan')):.3f} |")
    f = load("v2_fit_gallm_r2.json")
    if f:
        o = f["tracks"]["observable"]
        b = f["tracks"]["vs_truth_plus_bias"]
        log(f"| Gallm (2) | {f['points']} | {o['rms_true']:.3f} | {o['rms_error']:.3f} | {o['slope']:.2f} | "
            f"{b['rms_error']:.3f}, {b['slope']:.2f} | {b['bias_rms']:.3f} |")
    log("## V2, 2-4 cameras: tracking (recall of the central particles; ghosts; position RMS x/y/z um)")
    log("| run | recall, seen by all | >= n-1 | >= 2 | ghosts | position RMS | purity | long tracks |")
    log("|---|---|---|---|---|---|---|---|")
    table = [("2 cameras, clean (exact cameras, 1.0 px)", study.get("S_n8_r2")),
             ("2 cameras, error field, 1.0 px", study.get("Gcal_n8_r2_tol1.0")),
             ("2 cameras, error field, 1.5 px", study.get("Gcal_n8_r2_tol1.5")),
             ("2 cameras, corrected (1 pass), 1.0 px", mine.get("Gcal4c2_n8_r2_tol1.0")),
             ("2 cameras, corrected (2 passes), 1.0 px", mine.get("Gcal4c2b_n8_r2_tol1.0")),
             ("3 cameras, clean", study.get("S_n8_r3")),
             ("3 cameras, error field, 1.5 px", mine.get("Gcal4_n8_r3_tol1.5")),
             ("3 cameras, corrected, 1.0 px", mine.get("Gcal4c3_n8_r3_tol1.0")),
             ("4 cameras, clean", study.get("S_n8_r4")),
             ("4 cameras, error field, 1.5 px", mine.get("Gcal4_n8_r4_tol1.5")),
             ("4 cameras, corrected, 1.0 px", mine.get("Gcal4c4_n8_r4_tol1.0"))]
    for name, r in table:
        log(row(name, r))
    log("## V2, Gallm (G-all matched to T1, 2 cameras)")
    log("| run | recall, seen by both | – | >= 2 | ghosts | position RMS | purity | long tracks |")
    log("|---|---|---|---|---|---|---|---|")
    for name, r in (("Gallm, 1.5 px (the study's)", study.get("Gallm_r2")), ("Gallm, 1.0 px", mine.get("Gallm_r2_tol1.0")),
                    ("no error field (a), 1.5 px", mine.get("Gallm0_r2_tol1.5")),
                    ("no error field (a), 1.0 px", mine.get("Gallm0_r2_tol1.0")),
                    ("re-rendered at -c (c'), 1.5 px", mine.get("Gallmc_r2_tol1.5")),
                    ("re-rendered at -c (c'), 1.0 px", mine.get("Gallmc_r2_tol1.0")),
                    ("warped by c (c), 1.5 px", mine.get("Gallmw_r2_tol1.5")),
                    ("warped by c (c), 1.0 px", mine.get("Gallmw_r2_tol1.0")),
                    ("warped, 2 passes, 1.5 px", mine.get("Gallmw2_r2_tol1.5")),
                    ("warped, 2 passes, 1.0 px", mine.get("Gallmw2_r2_tol1.0"))):
        log(row(name, r))
    for p in sorted(glob.glob(os.path.join(sc.HERE, "v3_compare*.json"))):
        d = sc.load_json(p)
        log(f"## {os.path.basename(p)}")
        log("| variant | detections | active long at 100: run / control | long at 100 (>= 4 frames): run / control | "
            "ghosts: counts / swirl | real at 100: counts / swirl | whole run: tracks, points | epipolar robust SD | "
            "paired share |")
        log("|---|---|---|---|---|---|---|---|---|")
        for lab, v in d["variants"].items():
            q = v["quality"]
            log(f"| {lab} | {v['detections_per_frame']} | {v['active_long_100'][0]} / {v['active_long_100'][1]} | "
                f"{v['long_at_100']} / {v['control_long_at_100']} | "
                f"{100 * v['ghost_counts']:.0f} / {100 * v['ghost_swirl']:.0f} % | {v['real_at_100_counts']:.0f} / "
                f"{v['real_at_100_swirl']:.0f} | {v['long_tracks_total']}, {v['points_total']} | "
                f"{q['epi']['robust_sd']:.3f} | {100 * q['paired_share']:.0f} % |")
        for k, r in d["recurrence"].items():
            log(f"  recurrence {k}: {r}")
    m = load("m01_shape.json")
    if m:
        log("## M")
        log(f"S2's cue on the held-out frames: true {100 * m['M1']['s2_model_cue'][0]:.1f} %, random "
            f"{100 * m['M1']['s2_model_cue'][2]:.2f} %; the refits' best gain at that random rate "
            f"{100 * m['M1']['roc_gain_at_s2_random']:+.1f} points")
        for k, v in m["M1"]["roc"].items():
            log(f"  ROC {k}: {v['true_at']}")
        for k, v in m["M1"].get("roc_maps", {}).items():
            log(f"  ROC {k}: {v['true_at']}")
    # the figure: V2 with 2, 3 and 4 cameras
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = {2: [("clean", study.get("S_n8_r2")), ("error, 1.5 px", study.get("Gcal_n8_r2_tol1.5")),
                  ("corrected, 1 pass", mine.get("Gcal4c2_n8_r2_tol1.0")),
                  ("corrected, 2 passes", mine.get("Gcal4c2b_n8_r2_tol1.0"))],
              3: [("clean", study.get("S_n8_r3")), ("error, 1.5 px", mine.get("Gcal4_n8_r3_tol1.5")),
                  ("corrected, 1 pass", mine.get("Gcal4c3_n8_r3_tol1.0"))],
              4: [("clean", study.get("S_n8_r4")), ("error, 1.5 px", mine.get("Gcal4_n8_r4_tol1.5")),
                  ("corrected, 1 pass", mine.get("Gcal4c4_n8_r4_tol1.0"))]}
    colours = {"clean": "#4c72b0", "error, 1.5 px": "#c44e52", "corrected, 1 pass": "#55a868",
               "corrected, 2 passes": "#8172b3"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for gi, (n, items) in enumerate(groups.items()):
        for k, (lab, r) in enumerate(items):
            if r is None or "score" not in r:
                continue
            s = r["score"]
            x = gi + (k - 1.5) * 0.2
            kw = {"width": 0.19, "color": colours[lab], "label": lab if gi == 0 or lab == "corrected, 2 passes" else None}
            ax[0].bar(x, s["recall"]["all_in"], **kw)
            ax[1].bar(x, 100 * s["ghost_share"], **kw)
            ax[2].bar(x, float(np.mean(s["pos_rms_um"])), **kw)
    for a_, yl in zip(ax, ("recall (central, seen by all)", "ghosts [%]", "position RMS, mean of x, y, z [um]")):
        a_.set_xticks(range(3), ["2 cameras", "3 cameras", "4 cameras"])
        a_.set_ylabel(yl)
    ax[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("V2: a 0.7 px error field in every camera; clean = exact cameras", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(sc.HERE, "summary_sc.png"), dpi=110)
    plt.close(fig)
    log.save()


if __name__ == "__main__":
    main()
