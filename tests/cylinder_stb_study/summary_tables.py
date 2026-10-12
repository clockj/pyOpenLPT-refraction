"""The study's results as markdown tables, from the saved outputs (for the report; every number comes from here).

    python summary_tables.py

Reads s_runs.json, g2_realism.json, g3_control.json, g1_gap.json, d1_detect_t1.json, d2_detect_synth.json,
make_rigs.json and data/*/synth.json; writes summary_tables_out.txt beside itself.
"""
from __future__ import annotations

import os

import study

L = []


def out(s=""):
    print(s)
    L.append(s)


def J(name):
    p = os.path.join(study.HERE, name)
    return study.load_json(p) if os.path.exists(p) else None


def f2(v):
    return "–" if v is None else f"{v:.2f}"


def f3(v):
    return "–" if v is None else f"{v:.3f}"


def pc(v, d=1):
    return "–" if v is None else f"{100 * v:.{d}f} %"


def runrow(name, sc, extra=""):
    rc = sc["recall"]
    pos = sc["pos_rms_um"]
    pos_s = "/".join(f"{v:.0f}" for v in pos) if pos else "–"
    t_s = f"{sc['frame_time_s_mean']:.1f}" if sc["frame_time_s_mean"] else "–"
    return (f"| {name}{extra} | {f3(rc.get('all_in'))} | {f3(rc.get('n-1_in'))} | {f3(rc.get('ge2_in'))} | "
            f"{sc['present_per_frame'].get('all_in', 0):.0f} | {sc['reconstructed_per_frame']:.0f} | "
            f"{pc(sc['ghost_share'])} | {pos_s} | {sc['long_track_purity_mean']:.3f} | {sc['track_recall']:.3f} | "
            f"{t_s} |")


HEAD = ("| run | recall, seen by all | ≥ n−1 | ≥ 2 | particles seen by all, per frame | reconstructed per frame | "
        "ghosts | position RMS x/y/z µm | track purity | track recall | s per frame |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|")


def main():
    s = J("s_runs.json") or {"runs": {}}
    R = s["runs"]
    out("## Parts S and O (this work's settings) and today's code")
    out()
    out(HEAD)
    order = [f"{st}_r{n}" for st in ("S_n4", "S_n8", "S_n16", "O05_n8", "O10_n8") for n in (2, 3, 4)] + \
            [f"today_S_n8_r{n}" for n in (2, 3, 4)]
    for name in order:
        if name in R and "score" in R[name]:
            out(runrow(name, R[name]["score"], f" ({R[name]['threads']} threads)"))
    out()
    g2 = J("g2_realism.json") or {"runs": {}}
    out("## G2 (2 cameras)")
    out()
    out(HEAD)
    for ref in ("S_n8_r2", "O10_n8_r2"):
        if ref in R and "score" in R[ref]:
            out(runrow(ref, R[ref]["score"], " (G-base)" if ref == "S_n8_r2" else " (G-tube)"))
    for name, r in g2["runs"].items():
        if "score" in r:
            out(runrow(name, r["score"], f" (2-D tol {r['tol_2d_px']} px)"))
    out()
    out("## G3: the ghost estimators against the truth (T1-e's methods on the G-all cases)")
    out()
    out("| case | log entry | run's active long tracks | control's | counts estimate | truth (long tracks, purity < 0.5) |")
    out("|---|---|---|---|---|---|")
    summ = []
    for case, name in (("G-all, ±25 %", "g3_control.json"), ("G-all, ±12.5 %", "g3_control_Gall12.json"),
                       ("G-all matched to T1", "g3_control_Gallm.json"),
                       ("G-all, ±12.5 %, measured noise, 31 px", "g3_control_Gall12mn.json"),
                       ("G-all, ±12.5 %, 15 px median", "g3_control_Gall12m15.json"),
                       ("G-all, ±12.5 %, 15 px median, k 5", "g3_control_Gall12m15k5.json")):
        g3 = J(name)
        if not g3:
            continue
        for m in g3["marks"]:
            out(f"| {case} | {m['log_entry']} | {m['run']} | {m['control']} | {pc(m['counts_estimate'])} | "
                f"{pc(m['truth_ghost_share'])} of {m['truth_active_long']} |")
        sw = g3["swirl"]
        summ.append(f"| {case} | {pc(g3['counts_estimate_last'])} | {pc(sw['ghost_share_estimate'])} | "
                    f"{pc(g3['truth_ghost_share_tracks_last'])} | {pc(g3['swirl_by_truth']['real_share_above_0.3'])} of "
                    f"{g3['swirl_by_truth']['real']} | {pc(g3['swirl_by_truth']['ghost_share_above_0.3'])} of "
                    f"{g3['swirl_by_truth']['ghost']} |")
    out()
    out("| case | counts estimate (last frame) | swirl estimate | truth | real tracks above 0.3 | ghost tracks above 0.3 |")
    out("|---|---|---|---|---|---|")
    for r in summ:
        out(r)
    out("| T1 (T1-e) | 51 % (entry 97); 39 % (entry 40) | 44 % | – | – | – |")
    out()
    g1 = J("g1_gap.json")
    if g1:
        out("## G1: the gap table")
        out()
        sets = list(g1["sets"])
        out("| statistic | " + " | ".join(sets) + " |")
        out("|---|" + "---|" * len(sets))
        E = {k: v["estimate"] for k, v in g1["sets"].items()}
        T = {k: v.get("truth") for k, v in g1["sets"].items()}
        rows = [
            ("detections per frame, cam1 / cam2", lambda k: f"{E[k]['detections_per_frame'][0]:.0f} / {E[k]['detections_per_frame'][1]:.0f}"),
            ("true pairs per frame, estimated (± SE)", lambda k: f"{E[k]['true_pairs_per_frame']:.0f} ± {E[k]['true_pairs_se_per_frame']:.0f}"),
            ("share of the signal band that is true, estimated", lambda k: pc(E[k]["true_share_of_signal"])),
            ("detections with a partner, cam1 / cam2, estimated", lambda k: f"{pc(E[k]['detections_with_partner'][0], 0)} / {pc(E[k]['detections_with_partner'][1], 0)}"),
            ("true pairs within 1 px of the curve, estimated", lambda k: pc(E[k]["epipolar_true_within"].get("1.00"), 0)),
            ("depth cue keeps true / random, estimated", lambda k: f"{pc(E[k]['shape_true_kept'], 0)} / {pc(E[k]['shape_random_kept'])}"),
            ("detection recall, cam1 / cam2 (truth)", lambda k: f"{pc(T[k]['cam1']['recall_both_in_volume'], 0)} / {pc(T[k]['cam2']['recall_both_in_volume'], 0)}" if T[k] else "–"),
            ("false detections per frame, cam1 / cam2 (truth)", lambda k: f"{T[k]['cam1']['false_per_frame']:.0f} / {T[k]['cam2']['false_per_frame']:.0f}" if T[k] else "–"),
            ("detections with a partner (truth)", lambda k: f"{pc(T[k]['detections_with_partner'][0], 0)} / {pc(T[k]['detections_with_partner'][1], 0)}" if T[k] else "–"),
            ("true pairs in the signal band per frame (truth)", lambda k: f"{T[k]['true_pairs_in_signal_band_per_frame']:.0f}" if T[k] else "–"),
            ("true pairs within 0.5 / 1 / 2 px (truth)", lambda k: " / ".join(pc(T[k]['true_epipolar_within'][x], 0) for x in ("0.50", "1.00", "2.00")) if T[k] else "–"),
            ("depth cue keeps true / random (truth)", lambda k: f"{pc(T[k]['shape_true_kept'], 0)} / {pc(T[k]['shape_random_kept'])}" if T[k] else "–"),
            ("centre error 50 / 90 %, cam1 (truth)", lambda k: "/".join(f"{x:.2f}" for x in T[k]['cam1']['centre_err_px_pcts'][:2]) + " px" if T[k] else "–"),
        ]
        for lab, f in rows:
            vals = []
            for k in sets:
                try:
                    vals.append(f(k))
                except Exception:
                    vals.append("–")
            out(f"| {lab} | " + " | ".join(vals) + " |")
        out()
    d1 = J("d1_detect_t1.json")
    if d1:
        out("## D1: the detector's settings on T1 (10 frames)")
        out()
        out("| setting | detections cam1 / cam2 | true pairs per frame (± SE) | share | with the depth cue (± SE) | share |")
        out("|---|---|---|---|---|---|")
        B = d1["results"]["base"]
        for k, r in d1["results"].items():
            ch = "" if k == "base" else f" ({100 * (r['true_pairs_per_frame'] / B['true_pairs_per_frame'] - 1):+.0f} %)"
            chc = "" if k == "base" else f" ({100 * (r['shape_true_pairs_per_frame'] / B['shape_true_pairs_per_frame'] - 1):+.0f} %)"
            out(f"| {k} | {r['detections_per_frame'][0]:.0f} / {r['detections_per_frame'][1]:.0f} | "
                f"{r['true_pairs_per_frame']:.0f} ± {r['true_pairs_se_per_frame']:.0f}{ch} | {pc(r['true_share_of_signal'])} | "
                f"{r['shape_true_pairs_per_frame']:.0f} ± {r['shape_true_pairs_se_per_frame']:.0f}{chc} | "
                f"{pc(r['shape_true_share_of_signal'])} |")
        out()
    d2 = {"sets": {}}
    for tag in ("S_n8", "Gall"):                      # d2_detect_synth.py --tag (one process per set)
        part = J(f"d2_detect_synth_{tag}.json")
        if part:
            d2["sets"].update(part["sets"])
    if d2["sets"]:
        out("## D2: the same settings on synthetic sets (10 frames, truth)")
        out()
        for sn, sets in d2["sets"].items():
            out(f"{sn}:")
            out()
            out("| setting | detection recall cam1 / cam2 | false per frame | true pairs in the band (truth) | share (truth) | estimated true pairs |")
            out("|---|---|---|---|---|---|")
            B = sets.get("base")
            for k, r in sets.items():
                t, e = r["truth"], r["estimate"]
                ch = "" if (k == "base" or not B) else f" ({100 * (t['true_pairs_in_signal_band_per_frame'] / B['truth']['true_pairs_in_signal_band_per_frame'] - 1):+.0f} %)"
                che = "" if (k == "base" or not B) else f" ({100 * (e['true_pairs_per_frame'] / B['estimate']['true_pairs_per_frame'] - 1):+.0f} %)"
                out(f"| {k} | {pc(t['cam1']['recall_both_in_volume'], 0)} / {pc(t['cam2']['recall_both_in_volume'], 0)} | "
                    f"{t['cam1']['false_per_frame']:.0f} / {t['cam2']['false_per_frame']:.0f} | "
                    f"{t['true_pairs_in_signal_band_per_frame']:.0f}{ch} | {pc(t['signal_band_true_share'])} | "
                    f"{e['true_pairs_per_frame']:.0f}{che} |")
            out()
    # T1's rerun with a 15 px median (r1_t1_rerun.py), and the same change on G-all at ±12.5 % with the truth (G3)
    g = [(lab, J(f"g3_control_{n}.json")) for lab, n in (("31 px", "Gall12mn"), ("15 px", "Gall12m15"),
                                                         ("15 px, k 5", "Gall12m15k5"))]
    if all(x for _, x in g):
        out("## The median change on G-all at ±12.5 % (the detector told the measured noise): estimates and truth")
        out()
        out("| preprocessing | active long tracks (last entry) | control's | counts estimate | swirl estimate | truth | "
            "real tracks, truth | real tracks, counts |")
        out("|---|---|---|---|---|---|---|---|")
        for lab, x in g:
            m = x["marks"][-1]
            out(f"| {lab} | {m['run']} | {m['control']} | {pc(m['counts_estimate'])} | "
                f"{pc(x['swirl']['ghost_share_estimate'])} | {pc(m['truth_ghost_share'])} | "
                f"{m['truth_active_long'] * (1 - m['truth_ghost_share']):.0f} | "
                f"{m['run'] * (1 - m['counts_estimate']):.0f} |")
        out()
    r1 = J("r1_t1_rerun.json")
    if r1:
        V = r1["variants"]
        out("## T1 rerun (r1_t1_rerun.py): frames 0-250, controls 0-100")
        out()
        out("| run | detections per frame | long tracks at frame 100 | control's | counts estimate | swirl estimate | "
            "real at 100, counts | real at 100, swirl | long tracks, whole run | their points | s per frame |")
        out("|---|---|---|---|---|---|---|---|---|---|---|")
        for k, v in V.items():
            out(f"| {k}: {v['label']} | {v['detections_per_frame'][0]:.0f} / {v['detections_per_frame'][1]:.0f} | "
                f"{v['long_final'].get('100')} | {v['long_control'].get('100')} | {pc(v['ghost_share_counts'])} | "
                f"{pc(v['ghost_share_swirl'])} | {v['real_at_100_counts']:.0f} | {v['real_at_100_swirl']:.0f} | "
                f"{v['long_tracks_total']} | {v['long_track_points_total']} | {v['seconds_per_frame']:.1f} |")
        out()
        out("| tracks of | recur in | frames 0-100 | whole run | frames 0-100, swirl <= 0.3 | frames 0-100, swirl > 0.3 |")
        out("|---|---|---|---|---|---|")
        for ab, e in r1["recurrence"]["at_100"].items():
            w = r1["recurrence"]["whole"][ab]
            a, b = ab.split("_in_")
            out(f"| {a} | {b} | {e['recur']} of {e['tracks']} ({pc(e['share'], 0)}) | {w['recur']} of {w['tracks']} "
                f"({pc(w['share'], 0)}) | {pc(e['share_swirl_le_0.3'], 0)} of {e['n_le']} | "
                f"{pc(e['share_swirl_gt_0.3'], 0)} of {e['n_gt']} |")
        out()
        out("| run | T1-a RMS du, dv cam1 / cam2 [px] | epipolar mean ± s.e., RMS [px] | T1-c 68 % width error cam1 / cam2 "
            "[px] | T1-d length 10/50/90 % | speed median [mm/s] | second difference / step |")
        out("|---|---|---|---|---|---|---|")
        for k, v in V.items():
            c = v.get("checks")
            if not c:
                continue
            a1, a2, ep, d = c["T1a_cam1"], c["T1a_cam2"], c["T1a_epipolar"], c["T1d"]
            out(f"| {k} | {a1['rms_du']:.2f}, {a1['rms_dv']:.2f} / {a2['rms_du']:.2f}, {a2['rms_dv']:.2f} | "
                f"{ep['mean']:+.3f} ± {ep['sem']:.3f}, {ep['rms']:.2f} | "
                f"{c['T1c_cam1']['wh_q68']:.2f}, {c['T1c_cam1']['wv_q68']:.2f} / {c['T1c_cam2']['wh_q68']:.2f}, "
                f"{c['T1c_cam2']['wv_q68']:.2f} | {' / '.join(f'{x:.0f}' for x in d['length_pcts'])} | "
                f"{d['speed_mm_s_median']:.1f} | "
                f"{d['accel_rms_over_speed_rms']:.3f} |")
        out()
    with open(os.path.join(study.HERE, "summary_tables_out.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
