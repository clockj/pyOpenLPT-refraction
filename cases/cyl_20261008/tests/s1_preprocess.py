"""S1 (plan: 2026-10-08-plan.md, Phase T): T1's tracked frames, preprocessed as on 7 Oct (lines.preprocess), for three
cameras.

    python tests/s1_preprocess.py [--step 1] [--to-x]

From the local copy of the raw frames (data/T1raw: frames 0-249, the user's first 250), every STEP-th frame of each
camera: (the 90th-percentile background of frames 0-249, data/T1bg/<cam>_p90.npy, written by P0c) - frame, minus
its median over config.tracers.preprocess.median_px (7 Oct's adopted 15 px; cam3 23 px, its pixels being 1.48x
smaller), times a gain, clipped to 8 bits.  The gain is chosen as L1 chose it on 7 Oct: the largest integer for which
the 99.9th percentile of the particles' peak hp (local maxima above 5 noise after a 1 px Gaussian; cam3 1.5 px) on
frames 0, 100, 200 stays at or below 230 counts, at least 2.
Writes data/T1pre/<cam>/frame%06d.tif (git-ignored, for the runs) and, with --to-x, <tracers.out_dir>\\images\\<cam>\\
(the deliverable), with a manifest (frames, median, gain, the noise of each frame in 8-bit counts = gain x
lines.noise_left(hp)) and image lists for OpenLPT (one per camera, local and X:).
Writes s1_preprocess_out.txt and s1_preprocess.json beside itself.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import case  # noqa: E402
import lines  # noqa: E402
import t1  # noqa: E402

GAIN_FRAMES = (0, 100, 200)


def peaks(hp, noise, smooth):
    sm = cv2.GaussianBlur(hp, (0, 0), smooth)
    k = 2 * int(round(2 * smooth)) + 1
    pk = (sm == cv2.dilate(sm, np.ones((k, k), np.uint8))) & (sm > 5 * noise)
    return hp[pk]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--to-x", action="store_true")
    args = ap.parse_args()
    cfg = t1.load_config()
    med = cfg["tracers"]["preprocess"]["median_px"]
    log = case.Log(os.path.join(HERE, "s1_preprocess"))
    frames = t1.track_frames(cfg)[::args.step]
    local = os.path.join(t1.DATA, "T1pre")
    xdir = os.path.join(cfg["tracers"]["out_dir"], "images")
    log(f"S1: T1 preprocessed, every {args.step}th frame of {t1.track_frames(cfg)[0]}-{t1.track_frames(cfg)[-1]} "
        f"({len(frames)} frames per camera), medians {med}, into {local}" + (f" and {xdir}" if args.to_x else ""))
    man = {"step": args.step, "frames": frames, "median_px": med, "cameras": {}, "method":
           "lines.preprocess: (p90 background of frames 0-249 - frame) - median, minus the noise level, x gain, "
           "clipped to 0..255"}
    t0 = time.time()
    for cam in t1.CAMS:
        size = int(med[cam])
        bg = np.load(os.path.join(t1.DATA, "T1bg", f"{cam}_p90.npy"))
        smooth = 1.5 if cam == "cam3" else 1.0
        pk = []
        for k in GAIN_FRAMES:
            hp = lines.highpass(t1.read(cam, k), bg, size)
            pk.append(peaks(hp, lines.noise_left(hp), smooth))
        pk = np.concatenate(pk)
        p999 = float(np.percentile(pk, 99.9))
        gain = max(2, int(math.floor(230.0 / p999)))
        os.makedirs(os.path.join(local, cam), exist_ok=True)
        if args.to_x:
            os.makedirs(os.path.join(xdir, cam), exist_ok=True)
        noises, sat, paths_local, paths_x = [], [], [], []
        for k in frames:
            hp, img8 = lines.preprocess(t1.read(cam, k), bg, gain, size=size)
            noises.append(float(gain * lines.noise_left(hp)))
            sat.append(float(np.mean(img8 == 255)))
            name = t1.name(k, cfg)
            p = os.path.join(local, cam, name)
            cv2.imwrite(p, img8)
            paths_local.append(p)
            if args.to_x:
                px = os.path.join(xdir, cam, name)
                shutil.copy2(p, px)
                paths_x.append(px)
        for paths, root in ((paths_local, local), (paths_x, xdir if args.to_x else None)):
            if root:
                with open(os.path.join(root, f"{cam}ImageNames.txt"), "w", encoding="utf-8") as fh:
                    fh.write("\n".join(paths) + "\n")
        man["cameras"][cam] = {"gain": gain, "peak_hp_p999": p999, "n_peaks": int(len(pk)), "median_px": size,
                               "background": f"data/T1bg/{cam}_p90.npy",
                               "noise_8bit_median": float(np.median(noises)),
                               "noise_8bit_range": [float(min(noises)), float(max(noises))],
                               "saturated_fraction_max": float(max(sat)), "noise_8bit": noises}
        log(f"  {cam}: median {size} px; {len(pk)} peaks on frames {GAIN_FRAMES}, 99.9 % {p999:.1f} counts -> gain "
            f"{gain}; noise in the 8-bit counts median {np.median(noises):.2f} ({min(noises):.2f}-{max(noises):.2f}); "
            f"saturated pixels at most {100 * max(sat):.4f} %; {time.time() - t0:.0f} s")
    for root in (local, xdir if args.to_x else None):
        if root:
            with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as fh:
                json.dump(man, fh, indent=1)
    log.save({k: v for k, v in man.items()})


if __name__ == "__main__":
    main()
