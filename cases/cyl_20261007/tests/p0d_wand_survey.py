"""P0d (plan: Phase 0): the wand recording T0 of run 20261007, read-only, with the 6 Oct T2 case's detector and
acceptance settings (config.json; T2's p0_survey.py, adapted).

    python tests/p0d_wand_survey.py [STEP]      (default: every 50th frame)

1. The background's grey level along the recording (every 500th frame).
2. The opening scan on six frames: blobs > 500 px^2 left after openings of 21-141 px (the opening must separate the
   balls from the rod: 2 blobs).
3. Every STEP-th frame of both cameras through the detector and acceptance rules (case.py): the camera-frames kept,
   why the others were rejected, the balls' sizes and edge RMS, and where along the recording the frames kept in
   both cameras lie.
4. The one-ball camera-frames: does the ball found touch the image's edge, and is it the large or the small one?
   p0d_one_ball.jpg shows six of each camera's with the found ball's ellipse.
The headers are P0a's.  Writes p0d_wand_survey_out.txt and p0d_wand_survey.json beside itself.
"""
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
CASE_DIR = os.path.dirname(HERE)
sys.path.insert(0, CASE_DIR)
import case  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

CFG_PATH = os.path.join(CASE_DIR, "config.json")


def opening_scan(cfg, frames, log):
    be = case.ball_ellipses()
    det = be.settings_for(120.0)
    log("Opening scan: blobs > 500 px^2 left after an opening of k px (1 = rod kept or a ball lost, 2 = separated)")
    for f in frames:
        for cam in case.cam_ids(cfg):
            img = case.read_image(case.image_path(cfg, cam, f))
            N, _bg, seg = be.normalised_darkness(img.astype(np.float32), det)
            m = (N > seg).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            parts = []
            for k in (21, 41, 61, 71, 81, 91, 101, 111, 121, 141):
                mo = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
                n, _lab, st, _c = cv2.connectedComponentsWithStats(mo, connectivity=8)
                parts.append(f"{k}:{sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 500)}")
            log(f"  frame {f:5d} cam{cam}: " + " ".join(parts))


def one_ball_sheet(cfg, dets, reasons, log):
    W, H = int(cfg["image"]["width"]), int(cfg["image"]["height"])
    kinds, inside = Counter(), {1: [], 2: []}
    for (f, c), why in sorted(reasons.items()):
        if why != "one ball":
            continue
        r = dets[(f, c)][0]
        margin = min(r[0], r[1], W - 1 - r[0], H - 1 - r[1])
        edge = margin < 1.05 * max(r[4], r[5])
        kinds[(c, "touches the image's edge" if edge else "inside the image",
               "large ball" if r[2] > 75 else "small ball")] += 1
        if not edge:
            inside[c].append((f, r))
    log("One-ball camera-frames, by where the ball found is and its size (r > 75 px: the large ball):")
    for k, n in sorted(kinds.items()):
        log(f"  cam{k[0]}: {k[1]}, {k[2]}: {n}")
    tiles = []
    for c in (1, 2):
        for f, r in inside[c][:6]:
            col = cv2.cvtColor(case.read_image(case.image_path(cfg, c, f)), cv2.COLOR_GRAY2BGR)
            cv2.ellipse(col, ((r[0], r[1]), (2 * r[4], 2 * r[5]), math.degrees(r[6])), (0, 0, 255), 2)
            cv2.putText(col, f"cam{c} frame {f}: one ball, r {r[2]:.0f} px", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (0, 0, 255), 3)
            tiles.append(cv2.resize(col, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA))
    if tiles:
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
        cv2.imwrite(os.path.join(HERE, "p0d_one_ball.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
        log("  examples inside the image: p0d_one_ball.jpg")
    return {f"cam{k[0]}: {k[1]}, {k[2]}": n for k, n in kinds.items()}


def main():
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    cfg = case.load_config(CFG_PATH)
    log = case.Log(os.path.join(HERE, "p0d_wand_survey"))
    cams = case.cam_ids(cfg)
    frames_all = case.all_frames(cfg)
    log(f"P0d: run 20261007/T0 ({cfg['raw_dir']}), read-only, with the 6 Oct T2 case's settings")
    out = {"step": step}

    log("\nBackground grey level (the image's median) along the recording")
    grey = {}
    for cam in cams:
        vals = []
        for f in frames_all[::500]:
            vals.append(float(np.median(case.read_image(case.image_path(cfg, cam, f)))))
        grey[f"cam{cam}"] = vals
        log(f"  cam{cam}, every 500th frame: " + " ".join(f"{v:.0f}" for v in vals))
    out["grey_every_500"] = grey

    log("")
    opening_scan(cfg, [0, 3000, 6000, 9000, 12000, 15000], log)

    frames = frames_all[::step]
    det = cfg["detect"]
    tasks = [(f, c, case.image_path(cfg, c, f), det) for f in frames for c in cams]
    dets = {}
    with ProcessPoolExecutor(max_workers=int(det["workers"])) as ex:
        for f, c, rows in ex.map(case.detect_task, tasks, chunksize=2):
            dets[(f, c)] = [[float(x) for x in r] for r in rows] if rows is not None else None
    kept, reasons, medians = case.classify(dets, cfg)
    both = case.frames_in_all(kept, cams)
    log(f"\nSurvey: every {step}th frame ({len(frames)} frames), opening {det['open_px']} px, mask {det['rod_mask_px']:g} px")
    for c in cams:
        why = Counter(reasons[k] for k in reasons if k[1] == c)
        nk = sum(1 for k in kept if k[1] == c)
        rms = [case.edge_rms(r) for k, v in kept.items() if k[1] == c for r in v]
        log(f"  cam{c}: kept {nk}; rejected: " + ", ".join(f"{n} {r}" for r, n in why.most_common())
            + f"; median r (A, B) {medians[c][0]:.1f}, {medians[c][1]:.1f} px; edge RMS median {np.median(rms):.3f} px,"
            f" 90th percentile {np.percentile(rms, 90):.3f} px")
    log(f"  kept in both cameras: {len(both)} of {len(frames)} frames ({100.0 * len(both) / len(frames):.0f} %), "
        f"about {round(len(both) * len(frames_all) / len(frames), -2):.0f} over the whole recording")
    bins = np.arange(0, len(frames_all) + 1000, 1000)
    hist = np.histogram(both, bins=bins)[0]
    tot = np.histogram(frames, bins=bins)[0]
    log("  kept in both, per 1000 frames: " + " ".join(f"{int(b // 1000)}k {h}/{t}" for b, h, t in zip(bins, hist, tot)))
    for c in cams:
        uv = np.array([r[:2] for f in both for r in kept[(f, c)]])
        log(f"  cam{c}, the kept balls' centres: u {uv[:, 0].min():.0f}-{uv[:, 0].max():.0f}, "
            f"v {uv[:, 1].min():.0f}-{uv[:, 1].max():.0f} px")
    log("")
    out["one_ball"] = one_ball_sheet(cfg, dets, reasons, log)
    out.update({"open_px": det["open_px"], "rod_mask_px": det["rod_mask_px"], "frames_kept_in_both": both,
                "medians_r": {f"cam{c}": medians[c] for c in cams},
                "dets": {f"{f}_{c}": v for (f, c), v in dets.items()},
                "reasons": {f"{f}_{c}": v for (f, c), v in reasons.items()}})
    log.save(out)


if __name__ == "__main__":
    main()
