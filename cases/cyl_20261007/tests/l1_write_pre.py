"""L1 (plan: Phase 4): write preprocessed T1 frames for development, from the local copy, into
data/T1pre/<cam>/frame%06d.tif (8-bit TIFF, as OpenLPT reads them; git-ignored).

    python tests/l1_write_pre.py [--first 0] [--last 1000] [--step 1]

lines.preprocess with P0b's background (data/T1bg/<cam>_p90.npy, the 90th percentile of all 1,001 frames) and L1's
gain (l1_preprocess.json).  Writes data/T1pre/manifest.json (gain, background file, frames, the noise of every frame
in the 8-bit counts) and l1_write_pre_out.txt beside itself.  The full-run preprocessing to X: is Phase 6's.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import lines  # noqa: E402
import t1  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=1000)
    ap.add_argument("--step", type=int, default=1)
    args = ap.parse_args()
    with open(os.path.join(HERE, "l1_preprocess.json"), encoding="utf-8") as fh:
        l1 = json.load(fh)["T1"]
    frames = list(range(args.first, args.last + 1, args.step))
    man = {"frames": frames, "cameras": {}, "method": "lines.preprocess: (p90 background - frame) - median_31, x gain, "
                                                      "clipped to 0..255"}
    t0 = time.time()
    lines_out = []
    for cam in t1.CAMS:
        gain = float(l1[cam]["gain"])
        bgf = os.path.join(t1.DATA, "T1bg", f"{cam}_p90.npy")
        bg = np.load(bgf)
        os.makedirs(os.path.join(t1.LOCAL_PRE, cam), exist_ok=True)
        noise = {}
        for k in frames:
            hp, img8 = lines.preprocess(t1.read(cam, k), bg, gain)
            noise[k] = gain * lines.noise_left(hp)
            cv2.imwrite(os.path.join(t1.LOCAL_PRE, cam, t1.name(k)), img8)
        man["cameras"][cam] = {"gain": gain, "background": bgf, "noise_out_counts": noise}
        v = np.array(list(noise.values()))
        msg = (f"{cam}: {len(frames)} frames, gain {gain:g}, noise in the 8-bit image {np.median(v):.2f} counts "
               f"({v.min():.2f}-{v.max():.2f})")
        print(msg, flush=True)
        lines_out.append(msg)
    with open(os.path.join(t1.LOCAL_PRE, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    lines_out.append(f"written {t1.LOCAL_PRE} in {time.time() - t0:.0f} s")
    print(lines_out[-1])
    with open(os.path.join(HERE, "l1_write_pre_out.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines_out) + "\n")


if __name__ == "__main__":
    main()
