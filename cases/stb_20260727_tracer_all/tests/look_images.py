"""I2: what the masked images of each run look like (frame 0, every camera): a full frame and a 300 x 300 px crop at
the same place, shown on the same grey scale, and the intensity histogram.  Read only.

    python tests/look_images.py [RUN ...] [--frame F] [--out NAME]

Writes tests/<NAME>.png (default look_images.png).
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import case  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--vmax", type=float, default=60.0)
    ap.add_argument("--out", default="look_images")
    args = ap.parse_args()
    cfg = case.load_config()
    runs = args.runs or ["10Hz_r_b_1", "10Hz_r_b_2", "15Hz_r_b_1", "35Hz_r_b_2"]
    fig, axes = plt.subplots(len(runs), 3, figsize=(13, 3.6 * len(runs)))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(runs):
        img = cv2.imread(case.image_path(cfg, r, args.cam, args.frame), cv2.IMREAD_UNCHANGED)
        h, w = img.shape
        r0, c0 = h // 2 - 150, w // 2 - 150
        ax = axes[i, 0]
        ax.imshow(img, cmap="gray", vmin=0, vmax=args.vmax)
        ax.add_patch(plt.Rectangle((c0, r0), 300, 300, fill=False, ec="r", lw=1))
        ax.set_title(f"{r} cam{args.cam} frame {args.frame} (0-{args.vmax:g})", fontsize=9)
        ax.axis("off")
        ax = axes[i, 1]
        ax.imshow(img[r0:r0 + 300, c0:c0 + 300], cmap="gray", vmin=0, vmax=args.vmax, interpolation="nearest")
        ax.set_title("centre crop 300 x 300 px", fontsize=9)
        ax.axis("off")
        ax = axes[i, 2]
        hist = np.bincount(img.ravel(), minlength=256)
        ax.semilogy(np.arange(256), np.maximum(hist, 0.5), lw=1)
        ax.axvline(20, color="r", lw=0.8, ls="--")
        ax.set_xlim(0, 255)
        ax.set_title(f"histogram (mean {img.mean():.1f}; red: finder threshold 20)", fontsize=9)
        ax.set_xlabel("intensity")
    fig.tight_layout()
    path = os.path.join(case.TESTS, args.out + ".png")
    fig.savefig(path, dpi=80)
    print(path)


if __name__ == "__main__":
    main()
