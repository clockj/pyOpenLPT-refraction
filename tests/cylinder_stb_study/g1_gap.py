"""G1 (plan: 2026-10-08-plan.md, Part G): the statistics T1 can only estimate, on the synthetic sets with the truth,
beside T1's: the gap table.

    python g1_gap.py [--only SET ...]

Sets (cam1 and cam2), 10 frames each (0, 4, ..., 36): data/main's S_n8 (G-base), Gcal_n8 (G-cal), Gshape_nT1
(G-shape), O05_n8 and O10_n8 (G-tube: particles outside the central region), and data/gall's Gall (G-all).  T1: D1's
"base" (the same code on 10 T1 frames; d1_detect_t1.json).  Per set, the tracker's detector (LineFinder::detect,
defaults, the set's noise per camera) and pairstats.py (S2's method):
  - the truth-free figures, exactly as on T1: detections per frame, estimated true pairs per frame and their share
    of the signal band (d <= 1.5 px), the share of detections with a partner, the true pairs' distance profile
    across the curved epipolar lines, the depth cue's true and random pairs kept;
  - the same counted with the truth (detections assigned to images within 2.5 px): true pairs found (any distance)
    and in the signal band, the share of the signal band that is true, the profile of the true pairs' distances, the
    depth cue's true and random pairs kept; and the detection recall of the particles inside the view volume seen by
    both cameras, false detections (assigned to no image), centre errors.
The depth cue's model: the set's own (clean: data/main/synth.json; T1's for Gshape_nT1 and Gall), tolerances 0.8 px /
0.2.  Writes g1_gap_out.txt and g1_gap.json.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

import study
import pairstats as ps

SETS = {"S_n8": "main", "Gcal_n8": "main", "Gshape_nT1": "main", "O05_n8": "main", "O10_n8": "main",
        "Gall": "gall", "Gshape0_nT1": "main", "Gshape12_nT1": "main", "Gall12": "gall", "Gallm": "gallm"}
FRAMES = list(range(0, 40, 4))


def models_for(lpt, case_dir, variant, camfiles):
    if variant.startswith("t1") or variant.startswith("all"):  # t1, t1x, t1h, all, all12, allm: T1's model
        m = study.load_json(os.path.join(study.CASE_T1, "data", "T1_shape.json"))["cameras"]
        return [ps.Model(lpt, f, c["A_mm"], c["z_f_mm"], c["sigma0_px"], c["delta_mm"]) for f, c in zip(camfiles, m)]
    p = study.load_json(os.path.join(case_dir, "synth.json"))["params_clean"]
    return [ps.Model(lpt, f, c["A"], c["z_f"], c["sigma0"]) for f, c in zip(camfiles, p)]


def run_set(lpt, name, log):
    case_dir = os.path.join(study.DATA, SETS[name])
    s = study.load_json(os.path.join(case_dir, "sets.json"))[name]
    camfiles = [os.path.join(case_dir, "camFile", f"cam{k + 1}.txt") for k in range(2)]
    cams, sm_, vol = ps.matcher(lpt, camfiles, os.path.join(study.DATA, "g1", name))
    models = models_for(lpt, case_dir, s["variant"], camfiles)
    P = ps.line_param(lpt)
    pairs, n_det, recs, t0 = [], [], [], time.time()
    for t in FRAMES:
        dets = []
        for k in range(2):
            img = ps.read8(os.path.join(case_dir, "img", name, f"cam{k + 1}", f"img{t:06d}.tif"))
            dets.append(ps.detect(lpt, img, s["noise"][k], P))
        fp = ps.frame_pairs(cams, sm_, dets, vol, models)
        imgs, X, layer = ps.truth_frame(case_dir, name, t)
        recs.append(ps.truth_figures(dets, imgs, X, layer, fp, vol))
        pairs.append(fp)
        n_det.append([len(dets[0]), len(dets[1])])
    est = ps.summarize(pairs, n_det)
    tru = ps.truth_summary(recs, pairs)
    return {"estimate": est, "truth": tru, "seconds": time.time() - t0}


def pct_list(d, keys=("0.50", "1.00", "1.50", "2.00")):
    return ", ".join("%.0f" % (100 * d[k]) if d.get(k) is not None else "-" for k in keys)


def show(log, name, r):
    e, t = r["estimate"], r.get("truth")
    e_epi = pct_list(e["epipolar_true_within"])
    t_epi = pct_list(t["true_epipolar_within"]) if t else ""
    log(f"  {name}:")
    log(f"    estimated as on T1: detections {e['detections_per_frame'][0]:.0f} / {e['detections_per_frame'][1]:.0f}; "
        f"true pairs {e['true_pairs_per_frame']:.0f} +- {e['true_pairs_se_per_frame']:.0f} per frame "
        f"({100 * e['true_share_of_signal']:.1f} % of the signal band); detections with a partner "
        f"{100 * e['detections_with_partner'][0]:.0f} / {100 * e['detections_with_partner'][1]:.0f} %; true within "
        f"0.5 / 1 / 1.5 / 2 px {e_epi} %; "
        f"depth cue keeps {100 * (e['shape_true_kept'] or 0):.0f} % of true and {100 * (e['shape_random_kept'] or 0):.1f} % "
        f"of random pairs")
    if t:
        log(f"    with the truth: detection recall (inside the volume, seen by both) "
            f"{100 * t['cam1']['recall_both_in_volume']:.1f} / {100 * t['cam2']['recall_both_in_volume']:.1f} % of "
            f"{t['cam1']['present_per_frame']:.0f}; false detections {t['cam1']['false_per_frame']:.0f} / "
            f"{t['cam2']['false_per_frame']:.0f} per frame; centre error 50/90 % "
            f"{'/'.join(f'{x:.2f}' for x in t['cam1']['centre_err_px_pcts'][:2])}, "
            f"{'/'.join(f'{x:.2f}' for x in t['cam2']['centre_err_px_pcts'][:2])} px")
        log(f"      particles detected in both {t['paired_particles_per_frame']:.0f} per frame "
            f"({t['paired_in_volume_per_frame']:.0f} inside the volume); detections with a partner "
            f"{100 * t['detections_with_partner'][0]:.0f} / {100 * t['detections_with_partner'][1]:.0f} %; true pairs "
            f"among the candidates {t['true_pairs_found_per_frame']:.0f} per frame, in the signal band "
            f"{t['true_pairs_in_signal_band_per_frame']:.0f} ({100 * (t['signal_band_true_share'] or 0):.1f} % of it); "
            f"true within 0.5 / 1 / 1.5 / 2 px "
            f"{t_epi} %; "
            f"depth cue keeps {100 * (t['shape_true_kept'] or 0):.0f} % of true and "
            f"{100 * (t['shape_random_kept'] or 0):.1f} % of random pairs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None)
    args = ap.parse_args()
    lpt, build = study.lib()
    log = study.Log("g1_gap" if not args.only else "g1_gap_" + "_".join(args.only))
    out_path = os.path.join(study.HERE, "g1_gap.json")
    out = study.load_json(out_path) if os.path.exists(out_path) else {"sets": {}}
    out["build"] = build
    log(f"G1: the gap table; frames {FRAMES} of each synthetic set, T1 from D1's base; build {build['name']}")
    d1 = os.path.join(study.HERE, "d1_detect_t1.json")
    if os.path.exists(d1):
        out["sets"]["T1"] = {"estimate": study.load_json(d1)["results"]["base"]}
        show(log, "T1 (D1's base)", out["sets"]["T1"])
    for name in args.only or list(SETS):
        out["sets"][name] = run_set(lpt, name, log)
        show(log, name, out["sets"][name])
        study.save_json(out_path, out)
    log.save(out)


if __name__ == "__main__":
    main()
