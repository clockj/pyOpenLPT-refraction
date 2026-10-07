"""T1: the detection check (plan: 2026-10-06-plan.md, section 2).

    python tests/t1_detection.py

After S1 (detect.py).  T0's parts a and c (b and d measure the wand and its rod settings, which are T0's; not rerun):
  a. overlays: 20 kept camera-frames per camera (crops around both balls at full resolution, the ellipse and its
     centre drawn) and 10 rejected per camera (whole frames, with the reason), as contact sheets in
     <out_dir>/detection/t1_*.png for you and me to look at.  About a third of each is drawn from before the light
     changed (frames below LIGHT_STEP; P0: the background brightens between frames 2,000 and 3,000);
  c. balls A and B followed from frame to frame (consecutive kept frames): a swap would show as A closer to the
     previous B than to the previous A.
Judged by (plan): no wrong ellipse among those looked at; A and B never swap.
Writes t1_detection_out.txt and t1_detection.json beside this script.
"""
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

SEED = 20261006
TILE = 256
LIGHT_STEP = 2500          # P0: the light's step lies between frames 2,000 and 3,000


def draw(vis, row, col, label=None):
    u, v, _r, _m, a, b, th = row[:7]
    cv2.ellipse(vis, ((float(u), float(v)), (2 * float(a), 2 * float(b)), math.degrees(float(th))), col, 1,
                cv2.LINE_AA)
    cv2.drawMarker(vis, (int(round(u)), int(round(v))), col, cv2.MARKER_CROSS, 14, 1)
    if label:
        cv2.putText(vis, label, (int(u + 0.7 * a), int(v - 0.7 * b)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)


def crop(img, u, v, half):
    H, W = img.shape[:2]
    x0, y0 = int(round(u)) - half, int(round(v)) - half
    out = np.zeros((2 * half, 2 * half, 3), np.uint8)
    xs, ys = max(x0, 0), max(y0, 0)
    xe, ye = min(x0 + 2 * half, W), min(y0 + 2 * half, H)
    out[ys - y0:ye - y0, xs - x0:xe - x0] = img[ys:ye, xs:xe]
    return out


def sheet(tiles, cols, path):
    rows = int(math.ceil(len(tiles) / cols))
    h, w = tiles[0].shape[:2]
    out = np.full((rows * h, cols * w, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        out[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = t
    cv2.imwrite(path, out)


def sample_split(rng, items, n, key):
    """n items, about a third with key(item) < LIGHT_STEP (before the light changed), the rest after."""
    before = [x for x in items if key(x) < LIGHT_STEP]
    after = [x for x in items if key(x) >= LIGHT_STEP]
    nb = min(len(before), int(round(n / 3)))
    na = min(len(after), n - nb)
    return rng.sample(before, nb) + rng.sample(after, na)


def part_a(cfg, dets, kept, reasons, frames, log):
    rng = random.Random(SEED)
    outdir = os.path.join(cfg["out_dir"], "detection")
    os.makedirs(outdir, exist_ok=True)
    shown = {"kept": [], "rejected": []}
    for c in case.cam_ids(cfg):
        pick = sorted(sample_split(rng, frames, 20, key=lambda f: f))
        tiles = []
        for f in pick:
            img = case.read_image(case.image_path(cfg, c, f))
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            a, b = kept[(f, c)]
            for row, lab in ((a, "A"), (b, "B")):
                draw(vis, row, (0, 200, 0))
            for row, lab in ((a, "A"), (b, "B")):
                half = int(math.ceil(1.35 * max(row[4], row[5])))
                t = crop(vis, row[0], row[1], half)
                t = cv2.resize(t, (TILE, TILE), interpolation=cv2.INTER_AREA)
                cv2.putText(t, f"cam{c} {f} {lab} rms {case.edge_rms(row):.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 0, 255), 1, cv2.LINE_AA)
                tiles.append(t)
            shown["kept"].append([c, f])
        for k in range(0, len(tiles), 20):
            sheet(tiles[k:k + 20], 5, os.path.join(outdir, f"t1_kept_cam{c}_{k // 20 + 1}.png"))
        rej = [k for k in dets if k[1] == c and k in reasons and reasons[k] not in ("no ball", "image unreadable")]
        by = {}
        for k in rej:
            by.setdefault(reasons[k], []).append(k)
        pickr = []
        for reason in sorted(by):
            pickr += sample_split(rng, by[reason], min(len(by[reason]), max(1, 10 // len(by))), key=lambda k: k[0])
        pickr = sorted(pickr)[:10]
        tiles = []
        for (f, cc) in pickr:
            img = case.read_image(case.image_path(cfg, cc, f))
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for row in dets[(f, cc)]:
                draw(vis, row, (0, 0, 255))
            t = cv2.resize(vis, (640, 400), interpolation=cv2.INTER_AREA)
            cv2.putText(t, f"cam{cc} frame {f}: {reasons[(f, cc)]}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2, cv2.LINE_AA)
            tiles.append(t)
            shown["rejected"].append([cc, f, reasons[(f, cc)]])
        if tiles:
            sheet(tiles, 2, os.path.join(outdir, f"t1_rejected_cam{c}.png"))
        nb = sum(1 for f in pick if f < LIGHT_STEP)
        nbr = sum(1 for f, _cc in pickr if f < LIGHT_STEP)
        log(f"a. cam{c}: {len(pick)} kept camera-frames drawn ({nb} before frame {LIGHT_STEP}), {len(pickr)} rejected "
            f"({nbr} before)")
    log(f"   overlays written to {outdir} (t1_kept_cam*_*.png: both balls of each kept camera-frame; "
        f"t1_rejected_cam*.png: whole frames with the reason)")
    return shown


def part_c(cfg, kept, log):
    out = {}
    for c in case.cam_ids(cfg):
        fr = sorted(k[0] for k in kept if k[1] == c)
        swaps, steps = 0, []
        for f0, f1 in zip(fr[:-1], fr[1:]):
            if f1 != f0 + 1:
                continue
            a0, b0 = kept[(f0, c)]
            a1, b1 = kept[(f1, c)]
            dAA = math.hypot(a1[0] - a0[0], a1[1] - a0[1])
            dAB = math.hypot(a1[0] - b0[0], a1[1] - b0[1])
            dBB = math.hypot(b1[0] - b0[0], b1[1] - b0[1])
            dBA = math.hypot(b1[0] - a0[0], b1[1] - a0[1])
            if dAA + dBB > dAB + dBA:
                swaps += 1
            steps.append(max(dAA, dBB))
        steps = np.array(steps)
        out[f"cam{c}"] = {"pairs": int(len(steps)), "swaps": swaps, "step_median_px": float(np.median(steps)),
                          "step_p99_px": float(np.percentile(steps, 99)), "step_max_px": float(steps.max())}
        log(f"c. cam{c}: {len(steps)} consecutive kept pairs, {swaps} A/B swaps; frame-to-frame motion of a ball: "
            f"median {np.median(steps):.2f} px, 99th percentile {np.percentile(steps, 99):.2f} px, largest "
            f"{steps.max():.2f} px")
    return out


def main():
    cfg = case.load_config()
    log = case.Log(os.path.join(HERE, "t1_detection"))
    dets, kept, reasons, medians, frames = case.load_selection(cfg)
    log(f"T1: the detection check; {len(frames)} frames kept in both cameras")
    data = {"a": part_a(cfg, dets, kept, reasons, frames, log)}
    data["c"] = part_c(cfg, kept, log)
    log.save(data)


if __name__ == "__main__":
    main()
