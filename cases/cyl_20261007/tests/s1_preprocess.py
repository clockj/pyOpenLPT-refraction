"""S1 (plan, Phase 6): T1's frames for tracking, preprocessed as L1 does (lines.preprocess).

    python tests/s1_preprocess.py [--step 4] [--to-x]
    python tests/s1_preprocess.py --median 15         # the study's D1 setting (tests/cylinder_stb_study): local only

From the local copy of the raw frames (data/T1raw), every STEP-th frame (P0c: 4, so that particles move 1-3 px per
step) of both cameras: (the 90th-percentile background of all 1,001 frames, data/T1bg/<cam>_p90.npy) - frame, minus
its 31 px median (--median), times L1's gain (cam1 6, cam2 4: l1_preprocess.json), clipped to 8 bits.  Writes them to
data/T1pre_s<STEP>/<cam>/frame%06d.tif (git-ignored, for the runs) and, with --to-x, to
<tracers.out_dir>\\images\\<cam>\\ (the deliverable), with a manifest (frames, gain, noise of each frame in the 8-bit
counts = gain x lines.noise_left(hp)) and image lists for OpenLPT (one per camera, local and X:).
Writes s1_preprocess_out.txt and s1_preprocess.json beside itself.  With another median M: data/T1pre_s<STEP>_m<M>
and s1_preprocess_m<M>_out.txt, .json (the runs read it with CYL_T1_PRE=T1pre_s<STEP>_m<M>: t1_setup.py).
"""
import argparse
import json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--to-x", action="store_true")
    ap.add_argument("--median", type=int, default=31, help="the high-pass's median window [px]")
    args = ap.parse_args()
    sfx = "" if args.median == 31 else f"_m{args.median}"
    if sfx and args.to_x:
        raise SystemExit("--to-x writes the deliverable (X:\\...\\T1\\images, the 31 px median); not with --median")
    cfg = t1.load_config()
    log = case.Log(os.path.join(HERE, f"s1_preprocess{sfx}"))
    with open(os.path.join(HERE, "l1_preprocess.json"), encoding="utf-8") as fh:
        l1 = json.load(fh)["T1"]
    frames = t1.frames(cfg)[::args.step]
    local = os.path.join(t1.DATA, f"T1pre_s{args.step}{sfx}")
    xdir = os.path.join(cfg["tracers"]["out_dir"], "images")
    log(f"S1: T1 preprocessed, every {args.step}th frame ({len(frames)} frames per camera: {frames[0]}..{frames[-1]}),"
        f" median {args.median} px, into {local}" + (f" and {xdir}" if args.to_x else ""))
    man = {"step": args.step, "frames": frames, "median_px": args.median, "cameras": {}, "method":
           f"lines.preprocess: (p90 background of all 1,001 frames - frame) - median_{args.median}, minus the noise "
           "level, x gain, clipped to 0..255"}
    t0 = time.time()
    for cam in t1.CAMS:
        gain = float(l1[cam]["gain"])
        bg = np.load(os.path.join(t1.DATA, "T1bg", f"{cam}_p90.npy"))
        os.makedirs(os.path.join(local, cam), exist_ok=True)
        if args.to_x:
            os.makedirs(os.path.join(xdir, cam), exist_ok=True)
        noises, paths_local, paths_x = [], [], []
        for k in frames:
            hp, img8 = lines.preprocess(t1.read(cam, k), bg, gain, size=args.median)
            noises.append(float(gain * lines.noise_left(hp)))
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
        man["cameras"][cam] = {"gain": gain, "background": f"data/T1bg/{cam}_p90.npy",
                               "noise_8bit_median": float(np.median(noises)),
                               "noise_8bit_range": [float(min(noises)), float(max(noises))], "noise_8bit": noises}
        log(f"  {cam}: gain {gain:g}; noise in the 8-bit counts median {np.median(noises):.2f} "
            f"({min(noises):.2f}-{max(noises):.2f}); {time.time() - t0:.0f} s")
    for root in (local, xdir if args.to_x else None):
        if root:
            with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as fh:
                json.dump(man, fh, indent=1)
    log.save({k: v for k, v in man.items()})


if __name__ == "__main__":
    main()
