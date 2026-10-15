"""P0e (Phase 0): what the detector's settings do, drawn on T0's images, read-only.

    python tests/p0e_wand_views.py [FRAME]      # default image 2185

p0e_wand_views.png, one row per camera, with the camera's settings (case.det_settings: config.detect and its
per-camera overrides):
  (a) the whole frame, the wand boxed;
  (b) the wand: the fitted ellipses (green), the strips whose edge points the rod mask leaves out (orange), and
      circles of the Radius Range's ends (dashed: R min about the small ball, R max about the large one);
  (c) the same crop: the dark mask (grey) and what is left after the rod opening (blue): the two balls alone.
p0e_failures.png: the frames of the fine opening scan that no opening separated on cams 1 and 2 (whole frames, the
mask after the camera's opening in blue).
Writes p0e_wand_views_out.txt and .json beside itself.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASE_DIR = os.path.dirname(HERE)
sys.path.insert(0, CASE_DIR)
import case  # noqa: E402

import cv2  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

FAILURES = {1: [659, 5746, 8289], 2: [150, 1167, 4220]}   # p0d_wand_survey_scanfine_out.txt


def masks(img, det):
    """The detector's dark mask (after its 5 px clean-up) and the blobs left after the rod opening, as in
    ball_ellipses.detect_balls."""
    be = case.ball_ellipses()
    g = img.astype(np.float32)
    if det["polarity"] == "bright":
        g = 255.0 - g
    N, _bg, seg = be.normalised_darkness(g, be.settings_for(float(det["max_r_px"])))
    m = (N > seg).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    k = int(det["open_px"]) | 1
    mo = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return m, mo


def strip(ax, q, o, w, length):
    """The strip of half-width w from ball centre q towards the other ball o (the rod mask's region)."""
    q, o = np.asarray(q, float), np.asarray(o, float)
    e = (o - q) / max(np.linalg.norm(o - q), 1e-9)
    n = np.array([-e[1], e[0]])
    pts = [q + n * w, q + e * length + n * w, q + e * length - n * w, q - n * w]
    ax.add_patch(Polygon(pts, closed=True, fc=(1.0, 0.55, 0.0, 0.28), ec=(1.0, 0.45, 0.0, 0.9), lw=0.8))


def main():
    frame = int(sys.argv[1]) if len(sys.argv) > 1 else 2185
    log = case.Log(os.path.join(HERE, "p0e_wand_views"))
    cams = case.cam_ids(CFG)
    log(f"P0e: case {CFG['case']}, T0 image {frame}: the detector's settings drawn on each camera (read-only)")
    fig, axes = plt.subplots(len(cams), 3, figsize=(15, 4.6 * len(cams)))
    out = {"frame": frame, "cameras": {}}
    for row, c in enumerate(cams):
        det = case.det_settings(CFG, c)
        img = case.read_image(case.image_path(CFG, c, frame))
        rows = case.detect_image(img, det)
        m, mo = masks(img, det)
        H, W = img.shape
        out["cameras"][f"cam{c}"] = {"settings": det, "rows": rows}
        log(f"  cam{c} ({W} x {H}): settings {det}")
        for r in rows:
            log(f"      ball: u {r[0]:.2f}, v {r[1]:.2f}, r {r[2]:.2f} px (a {r[4]:.2f}, b {r[5]:.2f}, aspect "
                f"{max(r[4], r[5]) / min(r[4], r[5]):.3f}), edge RMS {case.edge_rms(r):.3f} px, dropped {int(r[7])}")
        ax0, ax1, ax2 = axes[row]
        ax0.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax0.set_title(f"cam{c}: image {frame} ({W} x {H} px)", fontsize=10)
        if len(rows) >= 2:
            a, b = sorted(rows[:2], key=lambda r: r[2])
            cu, cv = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
            half = max(3.0 * b[2], 0.5 * math.hypot(b[0] - a[0], b[1] - a[1]) + 2.2 * b[2])
        elif rows:
            a = b = rows[0]
            cu, cv, half = a[0], a[1], 3.0 * a[2]
        else:
            a = b = None
            cu, cv, half = W / 2, H / 2, 0.25 * min(W, H)
        x0, x1, y0, y1 = cu - half, cu + half, cv - half, cv + half
        ax0.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="tab:green", lw=1.2))
        for ax in (ax1, ax2):
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
        ax1.imshow(img, cmap="gray", vmin=0, vmax=255)
        for r in rows:
            ax1.add_patch(Ellipse((r[0], r[1]), 2 * r[4], 2 * r[5], angle=math.degrees(r[6]), fill=False,
                                  ec="lime", lw=1.3))
            ax1.plot(r[0], r[1], "+", color="lime", ms=8)
        if a is not None and b is not a:
            strip(ax1, a[:2], b[:2], float(det["rod_mask_px"]), 1.3 * max(a[4], a[5]))
            strip(ax1, b[:2], a[:2], float(det["rod_mask_px"]), 1.3 * max(b[4], b[5]))
            ax1.add_patch(Circle((a[0], a[1]), float(det["min_r_px"]), fill=False, ec="cyan", ls="--", lw=1.0))
            ax1.add_patch(Circle((b[0], b[1]), float(det["max_r_px"]), fill=False, ec="cyan", ls="--", lw=1.0))
        ax1.set_title(f"balls r = {', '.join(f'{r[2]:.1f}' for r in rows)} px; R {det['min_r_px']:g}-"
                      f"{det['max_r_px']:g} px (dashed); rod mask {det['rod_mask_px']:g} px (orange)", fontsize=9)
        rgb = np.zeros((H, W, 3), np.float32) + 1.0
        rgb[m > 0] = (0.72, 0.72, 0.72)
        rgb[mo > 0] = (0.2, 0.45, 0.85)
        ax2.imshow(rgb)
        n_blobs = cv2.connectedComponents(mo, connectivity=8)[0] - 1
        ax2.set_title(f"dark mask (grey); after the {det['open_px']} px opening (blue): {n_blobs} blob(s)",
                      fontsize=9)
        for ax in (ax0, ax1, ax2):
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    path = os.path.join(HERE, "p0e_wand_views.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    log(f"  figure: {os.path.basename(path)}")

    # the frames no opening separated (fine scan), cams 1-2
    n = sum(len(v) for v in FAILURES.values())
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.6))
    out["failures"] = {}
    for i, (c, frames) in enumerate(FAILURES.items()):
        det = case.det_settings(CFG, c)
        for j, f in enumerate(frames):
            img = case.read_image(case.image_path(CFG, c, f))
            m, mo = masks(img, det)
            rows = case.detect_image(img, det)
            ax = axes[i][j]
            rgb = np.dstack([img / 255.0] * 3).astype(np.float32)
            rgb[m > 0] = 0.5 * rgb[m > 0] + 0.5 * np.array((1.0, 0.6, 0.1))
            rgb[mo > 0] = 0.4 * rgb[mo > 0] + 0.6 * np.array((0.2, 0.45, 0.85))
            ax.imshow(rgb)
            for r in rows:
                ax.add_patch(Ellipse((r[0], r[1]), 2 * r[4], 2 * r[5], angle=math.degrees(r[6]), fill=False,
                                     ec="lime", lw=1.2))
            nb = cv2.connectedComponents(mo, connectivity=8)[0] - 1
            ax.set_title(f"cam{c} image {f}: {nb} blob(s) after {det['open_px']} px; detector: {len(rows)} ball(s)",
                         fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            out["failures"][f"cam{c}/{f}"] = {"blobs_after_opening": nb, "rows": rows}
            log(f"  failure frame cam{c} image {f}: {nb} blobs after the {det['open_px']} px opening; detector rows "
                + "; ".join(f"u {r[0]:.1f} v {r[1]:.1f} r {r[2]:.1f}" for r in rows))
    fig.tight_layout()
    path = os.path.join(HERE, "p0e_failures.png")
    fig.savefig(path, dpi=90)
    plt.close(fig)
    log(f"  figure: {os.path.basename(path)} ({n} frames; dark mask orange, after the opening blue)")
    log.save(out)


CFG = case.load_config()

if __name__ == "__main__":
    main()
