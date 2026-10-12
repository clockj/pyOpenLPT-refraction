"""V3 (plan, section 3): T1's 15 px frames warped by a fitted field, so that each camera's images move onto its model
(no library change).

    python v3_warp.py TAG FIELD.npz [--procs 8]
    python v3_warp.py --check            # the pipeline without a warp reproduces the case's 15 px frames

S1's preprocessing (cases/cyl_20261007/tests/s1_preprocess.py --median 15: lines.preprocess), with the warp put in
before the rounding: per camera and every 4th raw frame (0, 4, ..., 1000), from the local raw copy (data/T1raw) and
the 90th-percentile background (data/T1bg):
    hp   = lines.highpass(raw, bg, 15)            (background - frame, minus its 15 px median, recentred)
    hp_w = W(hp), W(q) = hp(q + c(q))            (selfcal.warp: cubic spline at the exact positions; the field's dense
                                                  form, which holds its edge values beyond the data)
    hp_w -= lines.noise_level(hp_w)               (recentred again: the warp changes the noise's spread, not its level)
    out  = clip(round(gain x hp_w), 0, 255)       (L1's gains: cam1 6, cam2 4)
and the noise the detector is told: gain x lines.noise_left(hp_w), the median over the frames (as S1 records it).
Writes data/T1pre_s4_m15<TAG>/cam1, cam2 (git-ignored), their image lists and manifest.json, and v3_warp_<TAG>_out.txt,
.json here.  The case's frames are read, never written.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import selfcal as sc

CAMS = ("cam1", "cam2")
_C = {}


def _init(field_path):
    sc.one_thread_env()
    _C["field"] = sc.Field.load(field_path) if field_path else None
    _C["maps"] = {}


def _maps(k):
    if k not in _C["maps"]:
        _C["maps"][k] = sc.warp_maps(_C["field"], k)
    return _C["maps"][k]


def work(job):
    import cv2
    import lines
    import t1
    cam_i, k, out_dir, gain = job
    cam = CAMS[cam_i]
    bg = _C.setdefault(("bg", cam), np.load(os.path.join(sc.CASE_DATA, "T1bg", f"{cam}_p90.npy")))
    hp = lines.highpass(t1.read(cam, k), bg, 15)
    if _C["field"] is not None:
        hp = sc.warp(hp, _maps(cam_i))
        hp -= np.float32(lines.noise_level(hp))
    img8 = lines.to_8bit(hp, gain)
    p = os.path.join(out_dir, cam, t1.name(k))
    cv2.imwrite(p, img8)
    return cam_i, k, float(gain * lines.noise_left(hp)), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?")
    ap.add_argument("field", nargs="?")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    import t1
    cfg = t1.load_config()
    gains = {c: float(sc.load_json(os.path.join(sc.CASE, "tests", "l1_preprocess.json"))["T1"][c]["gain"]) for c in CAMS}
    frames = t1.frames(cfg)[::4]
    if a.check:
        log = sc.Log("v3_warp_check")
        out_dir = os.path.join(sc.DATA, "work", "warp_check")
        for c in CAMS:
            os.makedirs(os.path.join(out_dir, c), exist_ok=True)
        sel = frames[::25]
        jobs = [(i, k, out_dir, gains[c]) for i, c in enumerate(CAMS) for k in sel]
        with Pool(min(a.procs, sc.THREADS_MAX), initializer=_init, initargs=(None,)) as pool:
            res = pool.map(work, jobs)
        import cv2
        worst = 0
        for cam_i, k, _n, p in res:
            ref = cv2.imread(os.path.join(sc.PRE_M15, CAMS[cam_i], t1.name(k)), cv2.IMREAD_UNCHANGED)
            new = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            worst = max(worst, int(np.abs(ref.astype(int) - new.astype(int)).max()))
        log(f"V3 warp check: the pipeline without a warp on frames {sel} of both cameras against the case's 15 px "
            f"frames: max pixel difference {worst}")
        log.save({"frames": sel, "max_pixel_difference": worst})
        return
    log = sc.Log(f"v3_warp_{a.tag}")
    out_dir = os.path.join(sc.DATA, f"T1pre_s4_m15{a.tag}")
    for c in CAMS:
        os.makedirs(os.path.join(out_dir, c), exist_ok=True)
    f = sc.Field.load(a.field)
    for k in range(2):
        g = f.dense[k]
        log(f"  {CAMS[k]}: the field's du RMS {np.sqrt(np.mean(g[0] ** 2)):.3f}, dv RMS {np.sqrt(np.mean(g[1] ** 2)):.3f} px "
            f"over the sensor (max |c| {np.sqrt(g[0] ** 2 + g[1] ** 2).max():.2f}); covered {100 * f.covered[k].mean():.0f} %")
    jobs = [(i, k, out_dir, gains[c]) for i, c in enumerate(CAMS) for k in frames]
    t0 = time.time()
    noise = {c: {} for c in CAMS}
    paths = {c: {} for c in CAMS}
    with Pool(min(a.procs, sc.THREADS_MAX), initializer=_init, initargs=(os.path.abspath(a.field),)) as pool:
        for cam_i, k, nz, p in pool.imap_unordered(work, jobs, chunksize=4):
            noise[CAMS[cam_i]][k] = nz
            paths[CAMS[cam_i]][k] = p
    man0 = sc.load_json(os.path.join(sc.PRE_M15, "manifest.json"))
    man = {"step": 4, "frames": frames, "median_px": 15, "warped_by": os.path.abspath(a.field),
           "method": "lines.highpass(raw, bg, 15), warped by the field (selfcal.warp), recentred, x gain, clipped",
           "cameras": {}}
    for c in CAMS:
        with open(os.path.join(out_dir, f"{c}ImageNames.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(paths[c][k] for k in frames) + "\n")
        nz = [noise[c][k] for k in frames]
        man["cameras"][c] = {"gain": gains[c], "noise_8bit_median": float(np.median(nz)),
                             "noise_8bit_range": [float(min(nz)), float(max(nz))], "noise_8bit": nz}
        log(f"  {c}: noise in the 8-bit counts after the warp: median {np.median(nz):.3f} (the 15 px frames' "
            f"{man0['cameras'][c]['noise_8bit_median']:.3f})")
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    log(f"V3 warp {a.tag}: {len(jobs)} frames into {out_dir} in {time.time() - t0:.0f} s")
    log.save({"tag": a.tag, "field": a.field, "out_dir": out_dir,
              "noise_median": {c: man["cameras"][c]["noise_8bit_median"] for c in CAMS}})


if __name__ == "__main__":
    main()
