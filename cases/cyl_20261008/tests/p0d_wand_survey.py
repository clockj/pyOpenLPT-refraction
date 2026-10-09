"""P0d (plan: Phase 0): the wand recording T0 of run 20261008, read-only (the 7 Oct case's p0d, adapted to three
cameras of two kinds).

    python tests/p0d_wand_survey.py scan             # the opening scan and the balls' sizes, per camera
    python tests/p0d_wand_survey.py survey [STEP]    # every STEP-th image number (default 50) with config.detect

scan:  on 8 image numbers spread over the part every camera saved, per camera: the blobs > a minimum area left after
       openings of 7-101 px (the opening must separate the balls from the rod: 2 blobs), and the two largest blobs'
       area-equivalent radii at the smallest opening that separates them.  Gives each camera's ball radii and the
       opening range, and their ratio between cam3 (v2640, 13.5 um pixels) and cams 1-2 (VEO 710L, 20 um).
survey: every STEP-th image number of every camera through the detector (case.det_settings: config.detect with the
       camera's overrides) and the acceptance rules (case.classify): the camera-frames kept, why the others were
       rejected, the balls' sizes, edge RMS and aspect ratios, where they lie on the sensor, and the frames kept in
       two and in three cameras.
Writes p0d_wand_survey_<stage>_out.txt and .json beside itself.
"""
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
CASE_DIR = os.path.dirname(HERE)
sys.path.insert(0, CASE_DIR)
import case  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

OPENINGS = (7, 11, 15, 21, 25, 31, 41, 51, 61, 71, 81, 101)


def scan_task(task):
    f, cam, path = task
    cv2.setNumThreads(1)
    be = case.ball_ellipses()
    img = case.read_image(path)
    det = be.settings_for(150.0 if cam == 3 else 100.0)
    N, _bg, seg = be.normalised_darkness(img.astype(np.float32), det)
    m = (N > seg).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    amin = 100.0 if cam != 3 else 220.0
    rows = []
    for k in OPENINGS:
        mo = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, _lab, st, _c = cv2.connectedComponentsWithStats(mo, connectivity=8)
        areas = sorted((int(st[i, cv2.CC_STAT_AREA]) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > amin),
                       reverse=True)
        rows.append((k, len(areas), [math.sqrt(a / math.pi) for a in areas[:3]]))
    # the rod's width: the dark mask's thinnest part between the balls is not measured here; the opening that
    # first separates two blobs bounds it from above
    return f, cam, rows, float(np.median(img)), float(seg)


def scan(log):
    lo = max(case.first_image(CFG, c) for c in case.cam_ids(CFG))
    hi = int(CFG["frames"]["last"])
    frames = [int(round(x)) for x in np.linspace(lo + 300, hi - 300, 8)]
    tasks = [(f, c, case.image_path(CFG, c, f)) for f in frames for c in case.cam_ids(CFG)]
    log(f"Opening scan: image numbers {frames}; blobs larger than 100 px^2 (cam3: 220) left after an opening of k px; "
        "the radii (sqrt(area/pi)) of the largest three")
    out = {}
    with ProcessPoolExecutor(max_workers=12) as ex:
        res = list(ex.map(scan_task, tasks))
    sep = {c: [] for c in case.cam_ids(CFG)}
    for f, cam, rows, med, seg in res:
        txt = " ".join(f"{k}:{n}" for k, n, _r in rows)
        two = [(k, r) for k, n, r in rows if n == 2]
        first = two[0] if two else None
        log(f"  image {f:5d} cam{cam}: median grey {med:.0f}, seg {seg:.3f}; {txt}"
            + (f"; first 2 at k={first[0]}: r = {first[1][0]:.1f}, {first[1][1]:.1f} px" if first else ""))
        if first:
            sep[cam].append({"frame": f, "k_first_two": first[0], "k_two": [k for k, _r in two],
                             "r": first[1][:2]})
        out[f"{f}/cam{cam}"] = {"rows": rows, "median_grey": med, "seg": seg}
    log("\nPer camera: the openings that leave exactly two blobs, and the balls' radii there")
    summ = {}
    for cam, L in sep.items():
        if not L:
            log(f"  cam{cam}: no frame separated into two blobs")
            continue
        ks = sorted({k for d in L for k in d["k_two"]})
        rA = [min(d["r"]) for d in L]
        rB = [max(d["r"]) for d in L]
        kmin = max(d["k_first_two"] for d in L)
        kmax = min(max(d["k_two"]) for d in L)
        summ[cam] = {"n": len(L), "r_small_median": float(np.median(rA)), "r_large_median": float(np.median(rB)),
                     "r_range": [float(min(rA)), float(max(rB))], "k_separating_all": [kmin, kmax], "k_seen": ks}
        log(f"  cam{cam}: {len(L)} of 8 frames separate; first separating opening up to {kmin} px, still two up to "
            f"{kmax} px in every one; radii: small ball median {np.median(rA):.1f} px, large {np.median(rB):.1f} px, "
            f"range {min(rA):.1f}-{max(rB):.1f} px")
    if 1 in summ and 3 in summ:
        for key in ("r_small_median", "r_large_median"):
            log(f"  cam3 / cam1 {key}: {summ[3][key] / summ[1][key]:.3f}")
    if 2 in summ and 3 in summ:
        for key in ("r_small_median", "r_large_median"):
            log(f"  cam3 / cam2 {key}: {summ[3][key] / summ[2][key]:.3f}")
    log(f"  pixel-size ratio (20 / 13.5 um): {0.020 / 0.0135:.3f}")
    return {"frames": frames, "per_image": out, "per_camera": summ}


def survey(log, step):
    cams = case.cam_ids(CFG)
    frames = case.all_frames(CFG)[::step]
    tasks = []
    for f in frames:
        for c in cams:
            p = case.image_path(CFG, c, f)
            if p is not None:
                tasks.append((f, c, p, case.det_settings(CFG, c)))
    log(f"Survey: every {step}th image number ({len(frames)}), {len(tasks)} camera-frames")
    for c in cams:
        log(f"  cam{c} settings: {case.det_settings(CFG, c)}")
    dets = {}
    with ProcessPoolExecutor(max_workers=16) as ex:
        for f, c, rows in ex.map(case.detect_task, tasks, chunksize=4):
            dets[(f, c)] = rows
    kept, reasons, medians = case.classify(dets, CFG)
    out = {"step": step, "cameras": {}}
    for c in cams:
        keys = [k for k in dets if k[1] == c]
        why = Counter(reasons[k] for k in keys if k in reasons)
        nk = sum(1 for k in keys if k in kept)
        ks = [k for k in keys if k in kept]
        erms = [case.edge_rms(r) for k in ks for r in kept[k]]
        asp = [case.aspect(r) for k in ks for r in kept[k]]
        u = [r[0] for k in ks for r in kept[k]]
        v = [r[1] for k in ks for r in kept[k]]
        ma, mb = medians[c]
        # the shape-rule passes' sizes before the size rule, to see the spread
        allr = [sorted(rows, key=lambda r: r[2]) for k, rows in dets.items() if k[1] == c and rows and len(rows) == 2]
        out["cameras"][f"cam{c}"] = {"camera_frames": len(keys), "kept": nk, "rejected": dict(why),
                                     "median_r_px": [ma, mb],
                                     "edge_rms_median": float(np.median(erms)) if erms else None,
                                     "edge_rms_p90": float(np.percentile(erms, 90)) if erms else None,
                                     "aspect_median": float(np.median(asp)) if asp else None,
                                     "aspect_p5_p95": [float(np.percentile(asp, 5)), float(np.percentile(asp, 95))]
                                     if asp else None,
                                     "u_range": [float(min(u)), float(max(u))] if u else None,
                                     "v_range": [float(min(v)), float(max(v))] if v else None,
                                     "two_ball_aspects_p5_p95": [float(np.percentile([case.aspect(r) for p in allr for r in p], q))
                                                                 for q in (5, 50, 95)] if allr else None}
        d = out["cameras"][f"cam{c}"]
        log(f"  cam{c}: kept {nk} of {len(keys)}; rejected: " + ", ".join(f"{n} {r}" for r, n in why.most_common())
            + f"; median r (A, B) {ma:.1f}, {mb:.1f} px; edge RMS median {d['edge_rms_median']}, p90 {d['edge_rms_p90']};"
            f" aspect median {d['aspect_median']}, p5-p95 {d['aspect_p5_p95']} (all two-ball pairs p5/50/95 "
            f"{d['two_ball_aspects_p5_p95']}); centres u {d['u_range']}, v {d['v_range']}")
    n2 = case.frames_in_at_least(kept, cams, 2)
    n3 = case.frames_in_all(kept, cams)
    pairs = {f"{a}{b}": len(case.frames_in_all(kept, [a, b])) for i, a in enumerate(cams) for b in cams[i + 1:]}
    log(f"  frames kept in >= 2 cameras: {len(n2)} of {len(frames)}; in all 3: {len(n3)}; per pair: {pairs}")
    log(f"  over the whole recording (x {step}): about {len(n2) * step} frames in >= 2 cameras, {len(n3) * step} in all 3")
    per_k = Counter((f // 1000) for f in n3)
    log("  kept in all three, per 1000 image numbers: " + " ".join(f"{k}k {per_k.get(k, 0)}" for k in range(14)))
    out.update({"frames_ge2": len(n2), "frames_all3": len(n3), "pairs": pairs})
    return out, dets


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "survey"
    log = case.Log(os.path.join(HERE, f"p0d_wand_survey_{stage}"))
    log(f"P0d {stage}: run 20261008/T0 ({CFG['raw_dir']}), read-only")
    if stage == "scan":
        out = scan(log)
    else:
        step = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        out, _dets = survey(log, step)
    log.save(out)


CFG = case.load_config()

if __name__ == "__main__":
    main()
