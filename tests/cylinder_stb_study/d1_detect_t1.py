"""D1 (plan: 2026-10-08-plan.md, Part D): a truth-free score of the line detector's settings on T1.

    python d1_detect_t1.py [--only NAME ...] [--no-best]

10 of T1's frames (S1's numbering i = 0, 25, ..., 225: raw frames 4 i), both cameras.  Per setting, the detector the
tracker runs (LineFinder::detect, build p4c) on each frame with that frame's noise, then pairstats.py (S2's method):
the candidate pairs along the curved epipolar lines, the estimated true pairs in the signal band (d <= 1.5 px) per
frame and their share of it, the same after the depth cue (T1's shape model, S2: data/T1_shape.json), and the true
pairs' distance profile.
Settings, one change at a time from the tracker's defaults, then the qualifying ones together:
  base        LineFinderParam's defaults (sigma0 1.0, k_detect 6, 2 deblending passes, second pass, validation) on S1's
              frames (a 31 px median);
  sigma1.4    the templates' blur at T1's measured 1.48 / 1.40 px (L2), sigma0 1.4;
  k5, k7      the detection threshold k_detect;
  deblend3    3 deblending passes;
  nosecond    no second pass on the residual;
  med15, med51  S1's preprocessing with a 15 or 51 px median, from the raw frames (data/T1raw) and S1's background and
              gain; the noise of each frame as S1 computes it;
  best        every change that qualifies (below), together (skipped if none, or with --no-best);
  med15_k5    added after the first pass: med15 and k5, the two largest gains, together (k5 alone missed the rule
              by a share 0.4 points lower, within its standard error).
Judged (plan): a setting qualifies if it raises the estimated true pairs per frame by at least 10 % and by more than
two standard errors, without lowering their share of the signal band; the same is reported after the depth cue.
Writes d1_detect_t1_out.txt and .json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

import study
import pairstats as ps

sys.path.insert(0, study.CASE_T1)
import lines  # noqa: E402  (the case's preprocessing)
import t1  # noqa: E402

FRAMES = list(range(0, 226, 25))
SETTINGS = {
    "base": ({}, 31),
    "sigma1.4": ({"sigma0": 1.4}, 31),
    "k5": ({"k_detect": 5.0}, 31),
    "k7": ({"k_detect": 7.0}, 31),
    "deblend3": ({"deblend_passes": 3}, 31),
    "nosecond": ({"second_pass": False}, 31),
    "med15": ({}, 15),
    "med51": ({}, 51),
    "med15_k5": ({"k_detect": 5.0}, 15),     # added after the first pass: the two largest gains together
}
PRE = os.path.join(study.CASE_T1, "data", "T1pre_s4")


def t1_models(lpt, camfiles):
    m = study.load_json(os.path.join(study.CASE_T1, "data", "T1_shape.json"))
    return [ps.Model(lpt, f, c["A_mm"], c["z_f_mm"], c["sigma0_px"], c["delta_mm"])
            for f, c in zip(camfiles, m["cameras"])]


def frame_images(i, median, cache):
    """Both cameras' 8-bit frames and noises for S1's frame i, with S1's preprocessing at the given median."""
    k = 4 * i
    man = study.load_json(os.path.join(PRE, "manifest.json"))
    pos = man["frames"].index(k)
    out = []
    for cam in ("cam1", "cam2"):
        if median == 31:
            img = ps.read8(os.path.join(PRE, cam, f"frame{k:06d}.tif"))
            noise = man["cameras"][cam]["noise_8bit"][pos]
        else:
            key = ("bg", cam)
            if key not in cache:
                cache[key] = np.load(os.path.join(study.CASE_T1, "data", "T1bg", f"{cam}_p90.npy"))
            gain = float(man["cameras"][cam]["gain"])
            hp, img = lines.preprocess(t1.read(cam, k), cache[key], gain, median)
            noise = float(gain * lines.noise_left(hp))
        out.append((img, noise))
    return out


def score(lpt, cams, sm_, vol, models, over, median, cache, log, name):
    P = ps.line_param(lpt, **over)
    pairs, n_det, t0 = [], [], time.time()
    per_frame = []
    for i in FRAMES:
        (im1, n1), (im2, n2) = frame_images(i, median, cache)
        dets = [ps.detect(lpt, im1, n1, P), ps.detect(lpt, im2, n2, P)]
        fp = ps.frame_pairs(cams, sm_, dets, vol, models)
        pairs.append(fp)
        n_det.append([len(dets[0]), len(dets[1])])
        t, se, s, b = ps.band_estimate(fp)
        per_frame.append({"i": i, "detections": [len(dets[0]), len(dets[1])], "true_est": t, "signal": s,
                          "background": b})
    res = ps.summarize(pairs, n_det)
    res["per_frame"] = per_frame
    res["seconds"] = time.time() - t0
    log(f"  {name:9s}: detections {res['detections_per_frame'][0]:.0f} / {res['detections_per_frame'][1]:.0f} per "
        f"frame; true pairs {res['true_pairs_per_frame']:.0f} +- {res['true_pairs_se_per_frame']:.0f} per frame "
        f"({100 * res['true_share_of_signal']:.1f} % of the {res['signal_pairs_per_frame']:.0f} signal-band pairs); "
        f"with the depth cue {res['shape_true_pairs_per_frame']:.0f} +- {res['shape_true_pairs_se_per_frame']:.0f} "
        f"({100 * (res['shape_true_share_of_signal'] or 0):.1f} % of {res['shape_signal_pairs_per_frame']:.0f}); "
        f"true within 1 px {100 * res['epipolar_true_within'].get('1.00', float('nan')):.0f} %; "
        f"{res['seconds']:.0f} s")
    return res


def qualifies(r, base, key="true_pairs_per_frame", se="true_pairs_se_per_frame", share="true_share_of_signal"):
    gain = r[key] - base[key]
    sd = math.hypot(r[se], base[se])
    return gain >= 0.10 * base[key] and gain > 2 * sd and r[share] >= base[share]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--no-best", action="store_true")
    args = ap.parse_args()
    lpt, build = study.lib()
    log = study.Log("d1_detect_t1")
    cams, sm_, vol = ps.matcher(lpt, study.CAMS_T0, os.path.join(study.DATA, "d1"))
    models = t1_models(lpt, study.CAMS_T0)
    log(f"D1: the line detector's settings on T1, frames i = {FRAMES} (raw 4 i); build {build['name']}; pairs within "
        f"{ps.BAND} px, signal band <= {ps.SIG} px, background {ps.BG0}-{ps.BAND} px (k = {ps.K_BG:.3f}); depth cue "
        f"T1's model, tolerances {ps.TOL}")
    out_path = os.path.join(study.HERE, "d1_detect_t1.json")
    prev = study.load_json(out_path) if (args.only and os.path.exists(out_path)) else {}
    out = {"build": build, "frames": FRAMES, "settings": {k: {"param": v[0], "median_px": v[1]}
                                                           for k, v in SETTINGS.items()},
           "results": prev.get("results", {})}          # --only adds to the results already there
    for k in ("best", "qualifying", "qualifying_with_depth_cue"):
        if k in prev:
            out[k] = prev[k]
    cache = {}
    names = args.only or list(SETTINGS)
    for name in names:
        over, med = SETTINGS[name]
        out["results"][name] = score(lpt, cams, sm_, vol, models, over, med, cache, log, name)
        study.save_json(os.path.join(study.HERE, "d1_detect_t1.json"), out)
    if "base" in out["results"]:
        base = out["results"]["base"]
        good, good_sc = [], []
        log("  against base:")
        for name, r in out["results"].items():
            if name == "base":
                continue
            q, q_sc = qualifies(r, base), qualifies(r, base, "shape_true_pairs_per_frame",
                                                    "shape_true_pairs_se_per_frame", "shape_true_share_of_signal")
            log(f"    {name:9s}: true pairs {100 * (r['true_pairs_per_frame'] / base['true_pairs_per_frame'] - 1):+.1f} %, "
                f"share {100 * (r['true_share_of_signal'] - base['true_share_of_signal']):+.1f} points; with the depth "
                f"cue {100 * (r['shape_true_pairs_per_frame'] / base['shape_true_pairs_per_frame'] - 1):+.1f} %, share "
                f"{100 * (r['shape_true_share_of_signal'] - base['shape_true_share_of_signal']):+.1f} points; "
                f"qualifies: {'yes' if q else 'no'} (with the depth cue: {'yes' if q_sc else 'no'})")
            if q and name in SETTINGS:
                good.append(name)
            if q_sc and name in SETTINGS:
                good_sc.append(name)
        out["qualifying"] = good
        out["qualifying_with_depth_cue"] = good_sc
        sel = sorted(set(good) | set(good_sc))
        meds = {SETTINGS[n][1] for n in sel if SETTINGS[n][1] != 31}
        if sel and not args.no_best and len(meds) <= 1:
            over = {}
            for n in sel:
                over.update(SETTINGS[n][0])
            med = meds.pop() if meds else 31
            log(f"  best: {sel} together ({over}, median {med} px)")
            out["best"] = {"from": sel, "param": over, "median_px": med}
            out["results"]["best"] = score(lpt, cams, sm_, vol, models, over, med, cache, log, "best")
            r = out["results"]["best"]
            log(f"    best: true pairs {100 * (r['true_pairs_per_frame'] / base['true_pairs_per_frame'] - 1):+.1f} %, "
                f"with the depth cue {100 * (r['shape_true_pairs_per_frame'] / base['shape_true_pairs_per_frame'] - 1):+.1f} %")
        elif not sel:
            log("  no setting qualifies: no combination run")
    log.save(out)


if __name__ == "__main__":
    main()
