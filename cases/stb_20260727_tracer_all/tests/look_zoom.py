"""I2b: a close look (100 x 100 px, full grey scale) at frame 0 of camera 0 in imgFile and imgFile_mask of a few runs,
to tell particle images from background.  Read only.

    python tests/look_zoom.py [RUN ...] [--cam K] [--out NAME]

Writes tests/<NAME>.png (default look_zoom.png).
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
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default="look_zoom")
    args = ap.parse_args()
    cfg = case.load_config()
    runs = args.runs or ["10Hz_r_b_1", "10Hz_r_b_2", "15Hz_r_b_1", "35Hz_r_b_2"]
    folders = ["imgFile", "imgFile_mask"]
    fig, axes = plt.subplots(len(runs), 2 * len(folders), figsize=(16, 3.9 * len(runs)))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(runs):
        for j, fold in enumerate(folders):
            p = os.path.join(case.case_dir(cfg, r), fold, f"cam{args.cam}", cfg["image_name"].format(args.frame))
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            for k, (size, vmax) in enumerate(((300, 255), (100, 255))):
                ax = axes[i, 2 * j + k]
                ax.axis("off")
                if img is None:
                    ax.set_title(f"{r} {fold}: unreadable", fontsize=8)
                    continue
                h, w = img.shape[:2]
                r0, c0 = h // 2 - size // 2 + 200, w // 2 - size // 2 + 200
                crop = img[r0:r0 + size, c0:c0 + size]
                ax.imshow(crop, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
                ax.set_title(f"{r} {fold} {img.dtype} {size}px (mean {img.mean():.1f}, "
                             f"crop max {crop.max()})", fontsize=8)
    fig.tight_layout()
    path = os.path.join(case.TESTS, args.out + ".png")
    fig.savefig(path, dpi=75)
    print(path)


if __name__ == "__main__":
    main()
