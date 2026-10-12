"""T1: the detection check (plan: 2026-10-06-plan.md, section 2).

    python tests/t1_detection.py

After S1 (detect.py).  Four parts:
  a. overlays: 20 kept camera-frames per camera (crops around both balls at full resolution, the ellipse and its
     centre drawn) and 10 rejected per camera (whole frames, with the reason), as contact sheets in
     <out_dir>/detection/t1_*.png for you and me to look at;
  b. the rod settings: 100 frames kept in both cameras detected again with the opening at 91 / 95 / 101 px and the
     mask at 40 / 50 / 60 px; per setting, how far the centres move from S1's (95 / 50) and how much the A-B image
     distance changes (what would bias the scale), and the camera-frames lost;
  c. balls A and B followed from frame to frame (consecutive kept frames): a swap would show as A closer to the
     previous B than to the previous A;
  d. the rod's width and the edge blur measured on 100 kept camera-frames per camera: darkness profiles across the
     rod at 35 %, 50 % and 65 % of the gap between the balls (median of the three, so that the string's knot does not
     count); width at half the rod's darkness and the 10-90 % edge width (blur sigma = width / 2.563).  T13's rule
     for the mask is the rod's half-width plus twice its blur sigma.
Judged by (plan): no wrong ellipse among those looked at; the centres move by less than about 0.05 px with the
settings; A and B never swap.  Writes t1_detection_out.txt and t1_detection.json beside this script.
"""
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402

SEED = 20261006
TILE = 256


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


def part_a(cfg, dets, kept, reasons, frames, log):
    rng = random.Random(SEED)
    outdir = os.path.join(cfg["out_dir"], "detection")
    shown = {"kept": [], "rejected": []}
    for c in case.cam_ids(cfg):
        pick = sorted(rng.sample(frames, 20))
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
            pickr += rng.sample(by[reason], min(len(by[reason]), max(1, 10 // len(by))))
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
    log(f"a. overlays written to {outdir} (t1_kept_cam*_*.png: 20 kept camera-frames per camera, both balls; "
        f"t1_rejected_cam*.png: up to 10 rejected per camera)")
    return shown


def part_b(cfg, kept, frames, log):
    rng = random.Random(SEED + 1)
    pick = sorted(rng.sample(frames, min(100, len(frames))))
    cams = case.cam_ids(cfg)
    settings = [(o, m) for o in (91, 95, 101) for m in (40.0, 50.0, 60.0)]
    tasks, meta = [], []
    for (o, m) in settings:
        det = dict(cfg["detect"], open_px=o, rod_mask_px=m)
        for f in pick:
            for c in cams:
                tasks.append((f, c, case.image_path(cfg, c, f), det))
                meta.append((o, m))
    with ProcessPoolExecutor(max_workers=int(cfg["detect"].get("workers", 8))) as ex:
        res = list(ex.map(case.detect_task, tasks, chunksize=4))
    log(f"b. the rod settings on {len(pick)} frames kept in both cameras ({len(pick) * len(cams)} camera-frames): "
        "centre shifts from S1 (opening 95, mask 50), px")
    log("   opening  mask   lost   A: RMS    max     B: RMS    max    A-B distance: mean change  RMS change")
    out = {}
    for (o, m) in settings:
        dA, dB, dD, lost = [], [], [], 0
        for (f, c, rows), mm in zip(res, meta):
            if mm != (o, m):
                continue
            a0, b0 = kept[(f, c)]
            if rows is None or len(rows) != 2 or any(r[7] for r in rows):
                lost += 1
                continue
            a1, b1 = sorted(rows, key=lambda r: r[2])
            dA.append(math.hypot(a1[0] - a0[0], a1[1] - a0[1]))
            dB.append(math.hypot(b1[0] - b0[0], b1[1] - b0[1]))
            dD.append(math.hypot(b1[0] - a1[0], b1[1] - a1[1]) - math.hypot(b0[0] - a0[0], b0[1] - a0[1]))
        dA, dB, dD = np.array(dA), np.array(dB), np.array(dD)
        rec = {"lost": lost, "A_rms": float(np.sqrt(np.mean(dA ** 2))), "A_max": float(dA.max()),
               "B_rms": float(np.sqrt(np.mean(dB ** 2))), "B_max": float(dB.max()),
               "AB_mean": float(dD.mean()), "AB_rms": float(np.sqrt(np.mean(dD ** 2)))}
        out[f"{o}_{int(m)}"] = rec
        log(f"   {o:5d}  {m:5.0f}  {lost:5d}   {rec['A_rms']:7.4f} {rec['A_max']:7.4f}   {rec['B_rms']:7.4f} "
            f"{rec['B_max']:7.4f}   {rec['AB_mean']:+12.4f} {rec['AB_rms']:11.4f}")
    return out


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


def profile_task(task):
    path, a, b, det = task
    be = case.ball_ellipses()
    img = case.read_image(path).astype(np.float32)
    N, _bg, _seg = be.normalised_darkness(img, be.settings_for(float(det["max_r_px"])))
    pa, pb = np.array(a[:2]), np.array(b[:2])
    e = (pb - pa) / np.linalg.norm(pb - pa)
    n = np.array([-e[1], e[0]])
    # the gap between the balls along the A-B line (each ball's extent along e from its ellipse)
    def extent(row, ang):
        _u, _v, _r, _m, ax, bx, th = row[:7]
        return 1.0 / math.sqrt((math.cos(ang - th) / ax) ** 2 + (math.sin(ang - th) / bx) ** 2)
    ang = math.atan2(e[1], e[0])
    s0 = extent(a, ang)
    s1 = np.linalg.norm(pb - pa) - extent(b, ang + math.pi)
    if s1 - s0 < 20:
        return None
    t = np.arange(-130.0, 130.01, 0.5)
    res = []
    for frac in (0.35, 0.5, 0.65):
        p = pa + (s0 + frac * (s1 - s0)) * e
        xs, ys = p[0] + t * n[0], p[1] + t * n[1]
        prof = map_coordinates(N, [ys, xs], order=1, mode="nearest")
        core = prof[np.abs(t) <= 10]
        if not len(core):
            continue
        peak = float(np.median(core))
        if peak < 0.2:
            continue
        def cross(level, side):
            idx = np.arange(len(t) // 2, len(t)) if side > 0 else np.arange(len(t) // 2, -1, -1)
            for i0, i1 in zip(idx[:-1], idx[1:]):
                if prof[i0] >= level > prof[i1]:
                    return t[i0] + (t[i1] - t[i0]) * (prof[i0] - level) / (prof[i0] - prof[i1])
            return None
        h1, h0 = cross(0.5 * peak, +1), cross(0.5 * peak, -1)
        p9a, p1a = cross(0.9 * peak, +1), cross(0.1 * peak, +1)
        p9b, p1b = cross(0.9 * peak, -1), cross(0.1 * peak, -1)
        if None in (h1, h0, p9a, p1a, p9b, p1b):
            continue
        res.append((h1 - h0, 0.5 * ((p1a - p9a) + (p9b - p1b))))
    if not res:
        return None
    w = float(np.median([r[0] for r in res]))
    edge = float(np.median([r[1] for r in res]))
    return w, edge, math.degrees(ang)


def part_d(cfg, kept, frames, log):
    rng = random.Random(SEED + 2)
    out = {}
    for c in case.cam_ids(cfg):
        pick = sorted(rng.sample(frames, min(100, len(frames))))
        tasks = [(case.image_path(cfg, c, f), kept[(f, c)][0], kept[(f, c)][1], cfg["detect"]) for f in pick]
        with ProcessPoolExecutor(max_workers=int(cfg["detect"].get("workers", 8))) as ex:
            res = [r for r in ex.map(profile_task, tasks, chunksize=2) if r is not None]
        w = np.array([r[0] for r in res])
        edge = np.array([r[1] for r in res])
        ang = np.abs(((np.array([r[2] for r in res]) + 90.0) % 180.0) - 90.0)   # 0 = rod along the rows
        sig = edge / 2.563
        rule = 0.5 * w + 2.0 * sig
        rec = {"profiles": int(len(res)), "width_median_px": float(np.median(w)),
               "width_p90_px": float(np.percentile(w, 90)), "width_max_px": float(w.max()),
               "edge_10_90_median_px": float(np.median(edge)), "blur_sigma_median_px": float(np.median(sig)),
               "blur_sigma_p90_px": float(np.percentile(sig, 90)), "rule_median_px": float(np.median(rule)),
               "rule_p90_px": float(np.percentile(rule, 90)), "rule_max_px": float(rule.max())}
        for lo, hi in ((0, 30), (30, 60), (60, 90.01)):
            sel = (ang >= lo) & (ang < hi)
            if sel.any():
                rec[f"width_median_px_rod_{lo}_{int(hi)}deg"] = float(np.median(w[sel]))
                rec[f"n_rod_{lo}_{int(hi)}deg"] = int(sel.sum())
        out[f"cam{c}"] = rec
        log(f"d. cam{c}: {len(res)} frames measured; rod width (half darkness) median {rec['width_median_px']:.1f} px, "
            f"90th percentile {rec['width_p90_px']:.1f}, largest {rec['width_max_px']:.1f}; edge 10-90 % "
            f"{rec['edge_10_90_median_px']:.1f} px (blur sigma {rec['blur_sigma_median_px']:.1f} px, 90th "
            f"{rec['blur_sigma_p90_px']:.1f}); T13's rule (half-width + 2 sigma): median {rec['rule_median_px']:.1f} px, "
            f"90th percentile {rec['rule_p90_px']:.1f}, largest {rec['rule_max_px']:.1f}")
        log("   rod width by the rod's angle from the image rows: "
            + ", ".join(f"{k.split('rod_')[1]}: {v:.1f} px (n {rec['n_rod_' + k.split('rod_')[1]]})"
                        for k, v in rec.items() if k.startswith("width_median_px_rod")))
    return out


def main():
    cfg = case.load_config()
    log = case.Log(os.path.join(HERE, "t1_detection"))
    dets, kept, reasons, medians, frames = case.load_selection(cfg)
    log(f"T1: the detection check; {len(frames)} frames kept in both cameras")
    data = {"a": part_a(cfg, dets, kept, reasons, frames, log)}
    data["b"] = part_b(cfg, kept, frames, log)
    data["c"] = part_c(cfg, kept, log)
    data["d"] = part_d(cfg, kept, frames, log)
    log.save(data)


if __name__ == "__main__":
    main()
