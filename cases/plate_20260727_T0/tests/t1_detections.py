"""T1: the wand detections (the circle finder's table, config detections "all") before any calibration.

  1. counts: frames, and how many cameras see both ends;
  2. ends A and B: the blob radii of the small (A) and large (B) ball per camera, and frames whose labels look swapped
     by the frame-to-frame tracks (consecutive frames where A <-> B swapped moves the balls much less);
  3. the coordinate convention: a sub-pixel centroid of each ball measured on the image, compared with the table's
     centre, on frames spread over the recording (a mean offset of +1 px would mean 1-based coordinates);
  4. overlays: the detections drawn on crops of those frames (t1_overlay_cam*.png).
Writes t1_out.txt and t1_detections.json beside this script.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

N_SAMPLE = 6          # frames per camera for the centroid check and the overlays


def swaps(uv_c: np.ndarray, fids: np.ndarray):
    """Frames whose A/B labels look swapped: for consecutive frame numbers, the motion with the labels as given
    against with A and B exchanged; swapped when exchanged is 3x smaller and the given motion exceeds 20 px."""
    out = []
    d_keep = np.full(len(fids), np.nan)
    for k in range(1, len(fids)):
        if fids[k] != fids[k - 1] + 1:
            continue
        a0, b0, a1, b1 = uv_c[k - 1, 0], uv_c[k - 1, 1], uv_c[k, 0], uv_c[k, 1]
        keep = np.linalg.norm(a1 - a0) + np.linalg.norm(b1 - b0)
        swap = np.linalg.norm(b1 - a0) + np.linalg.norm(a1 - b0)
        d_keep[k] = keep
        if keep > 20.0 and swap < keep / 3.0:
            out.append(int(fids[k]))
    return out, d_keep


def centroid(img: np.ndarray, u: float, v: float, r: float):
    """Sub-pixel centroid of the ball at table centre (u, v), radius r: |image - local background| weighted inside
    1.3 r, the background the median of the ring 1.6-2.2 r.  Returns (u, v) in 0-based pixel coordinates (pixel
    centres at integers) and whether the ball is brighter than its background."""
    h, w = img.shape
    R = int(np.ceil(2.3 * r))
    u0, v0 = int(round(u)), int(round(v))
    x0, x1, y0, y1 = max(u0 - R, 0), min(u0 + R + 1, w), max(v0 - R, 0), min(v0 + R + 1, h)
    sub = img[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - u, yy - v)
    ring = (rr > 1.6 * r) & (rr < 2.2 * r)
    bg = float(np.median(sub[ring]))
    inside = rr < 1.3 * r
    dif = sub - bg
    bright = float(np.mean(dif[rr < 0.8 * r])) > 0
    wgt = np.where(inside, np.clip(dif if bright else -dif, 0, None), 0.0)
    s = wgt.sum()
    return (float((wgt * xx).sum() / s), float((wgt * yy).sum() / s)), bright


def main():
    cfg = case.load_config()
    cd = case.read_detections(cfg, "all")
    uv, rad, fids = cd.uv, cd.radii, cd.frame_ids
    F, Nc = uv.shape[:2]
    out, res = [], {"frames": int(F), "frame_ids": [int(fids.min()), int(fids.max())], "cameras": []}
    log = case.Tee(os.path.join(HERE, "t1_out.txt"))
    log(f"T1: detections {case.detections_path(cfg, 'all')}")
    seen = np.isfinite(uv[..., 0])
    both = seen.all(axis=2)
    log(f"  {F} frames (numbers {fids.min()}-{fids.max()}; the recording has 12,605); cameras seeing both ends per "
        f"frame: {np.bincount(both.sum(axis=1), minlength=Nc + 1).tolist()} (0..{Nc})")
    sample = np.linspace(0, F - 1, N_SAMPLE + 2).round().astype(int)[1:-1]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for c in range(Nc):
        rA, rB = rad[:, c, 0], rad[:, c, 1]
        sw, d_keep = swaps(uv[:, c], fids)
        info = {"camera": c, "radius_A_px": [float(np.nanpercentile(rA, q)) for q in (1, 50, 99)],
                "radius_B_px": [float(np.nanpercentile(rB, q)) for q in (1, 50, 99)],
                "frames_rA_ge_rB": int(np.sum(rA >= rB)), "swapped_frames": sw[:50], "n_swapped": len(sw),
                "motion_px_per_frame": [float(np.nanpercentile(d_keep, q)) for q in (50, 99)]}
        log(f"  cam{c}: radius A {info['radius_A_px'][1]:.1f} px (1-99 %: {info['radius_A_px'][0]:.1f}-"
            f"{info['radius_A_px'][2]:.1f}), B {info['radius_B_px'][1]:.1f} ({info['radius_B_px'][0]:.1f}-"
            f"{info['radius_B_px'][2]:.1f}); frames with r_A >= r_B: {info['frames_rA_ge_rB']}; A/B swapped by the "
            f"tracks: {len(sw)} frames{(' (' + ', '.join(map(str, sw[:12])) + (' ...' if len(sw) > 12 else '') + ')') if sw else ''}; "
            f"motion between frames median {info['motion_px_per_frame'][0]:.1f} px (both ends)")
        # centroid check and overlays
        fig, axs = plt.subplots(1, N_SAMPLE, figsize=(3.0 * N_SAMPLE, 3.4))
        diffs, kinds = [], []
        for j, k in enumerate(sample):
            img = case.read_image(case.image_path(cfg, c, int(fids[k])))
            for e in range(2):
                (cu, cv_), bright = centroid(img, uv[k, c, e, 0], uv[k, c, e, 1], rad[k, c, e])
                diffs.append((uv[k, c, e, 0] - cu, uv[k, c, e, 1] - cv_))
                kinds.append(bright)
            mid = uv[k, c].mean(axis=0)
            half = int(max(np.linalg.norm(uv[k, c, 1] - uv[k, c, 0]), 60) / 2 + 2.2 * np.nanmax(rad[k, c]))
            x0, y0 = int(mid[0]) - half, int(mid[1]) - half
            crop = img[max(y0, 0):y0 + 2 * half, max(x0, 0):x0 + 2 * half]
            lo, hi = np.percentile(crop, [1, 99.5])
            ax = axs[j]
            ax.imshow(np.clip((crop - lo) / max(hi - lo, 1), 0, 1), cmap="gray",
                      extent=(max(x0, 0) - 0.5, max(x0, 0) + crop.shape[1] - 0.5, max(y0, 0) + crop.shape[0] - 0.5,
                              max(y0, 0) - 0.5))
            for e, col in ((0, "tab:orange"), (1, "tab:cyan")):
                ax.add_patch(plt.Circle((uv[k, c, e, 0], uv[k, c, e, 1]), rad[k, c, e], fill=False, color=col, lw=1))
                ax.plot(uv[k, c, e, 0], uv[k, c, e, 1], "+", color=col, ms=8)
                ax.text(uv[k, c, e, 0], uv[k, c, e, 1] - rad[k, c, e] - 4, "AB"[e], color=col, ha="center",
                        va="bottom", fontsize=9)
            ax.set_title(f"frame {int(fids[k])}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"cam{c} ({cfg['rig']['cameras'][c]['photo']}): table detections (A orange, B cyan)",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, f"t1_overlay_cam{c}.png"), dpi=90)
        plt.close(fig)
        diffs = np.asarray(diffs)
        info["table_minus_centroid_px"] = {"mean": diffs.mean(axis=0).tolist(), "sd": diffs.std(axis=0).tolist(),
                                           "max_abs": np.abs(diffs).max(axis=0).tolist()}
        info["balls_bright"] = bool(np.mean(kinds) > 0.5)
        log(f"         table centre - image centroid (0-based), {len(diffs)} balls on {N_SAMPLE} frames: mean "
            f"({diffs[:, 0].mean():+.2f}, {diffs[:, 1].mean():+.2f}) px, SD ({diffs[:, 0].std():.2f}, "
            f"{diffs[:, 1].std():.2f}); balls {'brighter' if info['balls_bright'] else 'darker'} than the background")
        res["cameras"].append(info)
    with open(os.path.join(HERE, "t1_detections.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
