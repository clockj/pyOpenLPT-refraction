"""D5 (diagnosis): column-wise structure of the images (sensor fixed-pattern noise, lighting).

A column-block gain or offset step would nudge any dot edge that sits on it, giving a centre bias fixed in the
image and varying with u.  Measures, over the background (board) pixels of all 20 images:
  * the column profile of the background divided by its smooth trend (a step at a block boundary shows here);
  * its spectrum, for a period (e.g. 16, 32, 64 columns);
  * the same along rows, for comparison.
Writes d5_out.txt beside this script.
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import case                                   # noqa: E402
import dots as D                              # noqa: E402

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def main():
    cfg = case.load_config()
    det = cfg["detect"]
    col_profiles, row_profiles = [], []
    for z, path in case.plane_files(cfg):
        img = case.read_image(path)
        N, bg, seg = D.normalised_darkness(img, det)
        board = np.abs(N) < 0.05                     # background pixels only
        ratio = np.where(board, img / np.maximum(bg, 1.0), np.nan)
        col_profiles.append(np.nanmean(ratio, axis=0))
        row_profiles.append(np.nanmean(ratio, axis=1))
    cp = np.nanmean(np.array(col_profiles), axis=0)
    rp = np.nanmean(np.array(row_profiles), axis=0)
    for name, p in (("columns", cp), ("rows", rp)):
        p = p - np.nanmean(p)
        p = np.nan_to_num(p)
        trend = cv2.GaussianBlur(p.reshape(1, -1).astype(np.float64), (0, 0), 8).ravel()
        hf = p - trend
        say(f"{name}: profile of image / background over board pixels: std {np.std(p) * 100:.3f} %, "
            f"high-pass (removing an 8-px smooth trend) std {np.std(hf) * 100:.3f} %")
        spec = np.abs(np.fft.rfft(hf)) ** 2
        freqs = np.fft.rfftfreq(len(hf))
        top = np.argsort(spec[1:])[::-1][:6] + 1
        say("  strongest periods (px): " + ", ".join(f"{1 / freqs[t]:.1f} ({spec[t] / spec[1:].sum() * 100:.1f} %)"
                                                    for t in top))
        if name == "columns":
            # mean of the high-pass profile at each column modulo 8, 16, 32, 64
            for per in (8, 16, 32, 64):
                ph = np.array([hf[k::per].mean() for k in range(per)])
                say(f"  period {per:2d}: phase-averaged amplitude (max - min) {(ph.max() - ph.min()) * 100:.3f} %")
    with open(os.path.join(HERE, "d5_out.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
