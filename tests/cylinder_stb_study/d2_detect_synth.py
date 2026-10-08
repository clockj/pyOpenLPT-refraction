"""D2 (plan: 2026-10-08-plan.md, Part D): D1's detector settings on synthetic sets, where the truth is known.

    python d2_detect_synth.py [--sets S_n8 Gall] [--only NAME ...] [--tag T]
    (with --tag the outputs are d2_detect_synth_T_out.txt and .json, so that sets can run in parallel processes)

Sets (cam1 and cam2, frames 0, 4, ..., 36): data/main's S_n8 (the clean case) and data/gall's Gall (every realism
factor: the most T1-like).  Settings: D1's (d1_detect_t1.SETTINGS) and D1's "best" if it ran; for the median windows
the set's frames are rendered again before the preprocessing (synth_study.float_image) and preprocessed with that
window (synth_study.preprocess).  Per setting, with pairstats.py:
  - the truth: the detection recall of the particles inside the view volume seen by both cameras, false detections
    (assigned to no image within 2.5 px), centre errors, true pairs in the signal band and their share of it;
  - the truth-free estimate D1 scores T1 by (true pairs per frame, share), to see whether its ranking of the settings
    holds where the truth is known.
Judged (plan): D1's ranking holds where the truth is known.  Writes d2_detect_synth_out.txt and .json.
"""
from __future__ import annotations

import argparse
import os
import time

import study
import pairstats as ps
import d1_detect_t1 as d1
import g1_gap as g1
import synth_study


def image(case_dir, set_name, s, k, t, median):
    if median == 31:
        return ps.read8(os.path.join(case_dir, "img", set_name, f"cam{k + 1}", f"img{t:06d}.tif"))
    seed = study.load_json(os.path.join(case_dir, "synth.json"))["noise_seed"]
    return synth_study.preprocess(synth_study.float_image(case_dir, set_name, k, t, seed), s["noise"][k], median)


def run(lpt, set_name, over, log, median=31):
    case_dir = os.path.join(study.DATA, g1.SETS[set_name])
    s = study.load_json(os.path.join(case_dir, "sets.json"))[set_name]
    camfiles = [os.path.join(case_dir, "camFile", f"cam{k + 1}.txt") for k in range(2)]
    key = ("m", set_name)
    if key not in run.cache:
        run.cache[key] = (ps.matcher(lpt, camfiles, os.path.join(study.DATA, "d2", set_name)),
                          g1.models_for(lpt, case_dir, s["variant"], camfiles))
    (cams, sm_, vol), models = run.cache[key]
    P = ps.line_param(lpt, **over)
    pairs, n_det, recs, t0 = [], [], [], time.time()
    for t in g1.FRAMES:
        dets = [ps.detect(lpt, image(case_dir, set_name, s, k, t, median), s["noise"][k], P) for k in range(2)]
        fp = ps.frame_pairs(cams, sm_, dets, vol, models)
        imgs, X, layer = ps.truth_frame(case_dir, set_name, t)
        recs.append(ps.truth_figures(dets, imgs, X, layer, fp, vol))
        pairs.append(fp)
        n_det.append([len(dets[0]), len(dets[1])])
    return {"estimate": ps.summarize(pairs, n_det), "truth": ps.truth_summary(recs, pairs),
            "seconds": time.time() - t0}


run.cache = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["S_n8", "Gall"])
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    lpt, build = study.lib()
    stem = "d2_detect_synth" + (f"_{args.tag}" if args.tag else "")
    log = study.Log(stem)
    settings = dict(d1.SETTINGS)
    d1j = os.path.join(study.HERE, "d1_detect_t1.json")
    if os.path.exists(d1j):
        best = study.load_json(d1j).get("best")
        if best:
            settings["best"] = (best["param"], best["median_px"])
    names = args.only or list(settings)
    out = {"build": build, "frames": g1.FRAMES, "settings": {k: {"param": v[0], "median_px": v[1]}
                                                             for k, v in settings.items()}, "sets": {}}
    log(f"D2: D1's settings on synthetic sets {args.sets}, frames {g1.FRAMES}; build {build['name']}")
    for sn in args.sets:
        out["sets"][sn] = {}
        log(f"  {sn}:")
        for name in names:
            r = run(lpt, sn, settings[name][0], log, settings[name][1])
            out["sets"][sn][name] = r
            e, t = r["estimate"], r["truth"]
            log(f"    {name:9s}: recall {100 * t['cam1']['recall_both_in_volume']:.1f} / "
                f"{100 * t['cam2']['recall_both_in_volume']:.1f} %, false {t['cam1']['false_per_frame']:.0f} / "
                f"{t['cam2']['false_per_frame']:.0f} per frame, centre error median "
                f"{t['cam1']['centre_err_px_pcts'][0]:.2f} / {t['cam2']['centre_err_px_pcts'][0]:.2f} px; true pairs "
                f"in the signal band {t['true_pairs_in_signal_band_per_frame']:.0f} per frame "
                f"({100 * (t['signal_band_true_share'] or 0):.1f} %), estimated {e['true_pairs_per_frame']:.0f} +- "
                f"{e['true_pairs_se_per_frame']:.0f} ({100 * e['true_share_of_signal']:.1f} %); after the depth cue "
                f"(truth) {100 * (t['shape_true_kept'] or 0):.0f} % of true, {100 * (t['shape_random_kept'] or 0):.1f} % "
                f"of random; {r['seconds']:.0f} s")
            study.save_json(os.path.join(study.HERE, stem + ".json"), out)
        base = out["sets"][sn].get("base")
        if base:
            log("    against base (truth: true pairs in the signal band; estimate as D1):")
            for name, r in out["sets"][sn].items():
                if name == "base":
                    continue
                tt = r["truth"]["true_pairs_in_signal_band_per_frame"] / max(1e-9, base["truth"][
                    "true_pairs_in_signal_band_per_frame"]) - 1
                ee = r["estimate"]["true_pairs_per_frame"] / max(1e-9, base["estimate"]["true_pairs_per_frame"]) - 1
                log(f"      {name:9s}: truth {100 * tt:+.1f} %, estimate {100 * ee:+.1f} %")
    log.save(out)


if __name__ == "__main__":
    main()
