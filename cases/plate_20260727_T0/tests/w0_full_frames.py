"""W0 (2026-10-03-walls-plan.md, section 2): can the images locate the walls?  Draws one full frame of each camera and
its per-pixel minimum over 31 frames spread over the recording (the minimum shows everywhere the wand and its hanging
nut went), downsampled 4x, to see whether a wall's edge or the channel's corner is visible.

    python tests/w0_full_frames.py
Writes w0_full_frames.png beside this script.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

FRAMES = list(range(200, 12600, 400))


def main():
    cfg = case.load_config()
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for c in range(4):
        imgs = []
        for fid in FRAMES:
            try:
                imgs.append(case.read_image(case.image_path(cfg, c, fid))[::4, ::4])
            except IOError:
                pass
        stack = np.stack(imgs)
        axes[0, c].imshow(stack[len(stack) // 2], cmap="gray")
        axes[0, c].set_title(f"C{c + 1} one frame ({stack.shape[1] * 4}x{stack.shape[2] * 4})")
        axes[1, c].imshow(stack.min(axis=0), cmap="gray")
        axes[1, c].set_title(f"C{c + 1} minimum over {len(stack)} frames")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "w0_full_frames.png"), dpi=60)


if __name__ == "__main__":
    main()
