"""P0c (plan: Phase 0): how far the tracer images move between frames of T1, to choose the frame step.

    python tests/p0c_t1_motion.py        (after p0b_t1_survey.py, which saves the backgrounds)

Per camera, on hp images (lines.highpass with P0b's background): windows of 64 x 64 px on a 64 px grid, each
correlated (OpenCV's normalised cross-correlation) with the frame k later, k = 1, 2, 4, 8, 16; the peak refined to
sub-pixel by a parabola in u and in v; a window counts when its peak correlation is at least 0.3.  The search is
guided: k = 1 searches +-4 px; each larger k searches +-(3 + k / 4) px around twice the same window's displacement
at k / 2 (the images are short horizontal and vertical lines, so an unguided search at k = 8, 16 locks onto other
lines: it gave du 5/95 % of -7/+8 px at k = 8).  Start frames 0, 200, 400, 600, 800 (k = 16 from 0, 200, 400, 600,
784).
Reports per k: the windows used, the median correlation, the displacement (du, dv) percentiles, the speed |d| / k
(px per frame), and per start frame the median displacement (is the flow steady?).
Figure p0c_<cam>_k8.png: the displacement field at k = 8 from frame 400 (arrows x 8).
Check: consistency: each k's median du, dv against k/4 times k = 4's (more than 30 % off: flagged unreliable; at
k = 1 the 3-point peak fit is known to lock sub-pixel shifts towards whole pixels, at k = 16 the images have
changed too much).
Recommends the frame step: the smallest k whose median displacement is at least 1 px and whose 95th percentile is
at most 3 px; if none, the k closest to a 2 px median.
Writes p0c_t1_motion_out.txt and p0c_t1_motion.json beside itself.
"""
import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import lines  # noqa: E402
import t1  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOG = []
W = 64
KS = (1, 2, 4, 8, 16)
STARTS = (0, 200, 400, 600, 800)


def log(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def subpix(r, x, y):
    def para(a, b, c):
        d = a - 2 * b + c
        return 0.0 if d >= 0 else 0.5 * (a - c) / d
    dx = para(r[y, x - 1], r[y, x], r[y, x + 1]) if 0 < x < r.shape[1] - 1 else 0.0
    dy = para(r[y - 1, x], r[y, x], r[y + 1, x]) if 0 < y < r.shape[0] - 1 else 0.0
    return x + dx, y + dy


def displacements(a, b, k, pred=None):
    """Window displacements from a to b; pred: {(x, y): (du, dv)} predicted shifts (search +-(3 + k/4) px around
    them), else +-4 px around 0."""
    m = int(math.ceil(3 + k / 4)) if pred is not None else 4
    out = []
    H, Wd = a.shape
    edge = 24
    for y in range(edge, H - W - edge, W):
        for x in range(edge, Wd - W - edge, W):
            tpl = a[y:y + W, x:x + W]
            if tpl.std() < 1e-3:
                continue
            if pred is not None:
                if (x, y) not in pred:
                    continue
                pu, pv = pred[(x, y)]
            else:
                pu, pv = 0.0, 0.0
            cx, cy = x + int(round(pu)), y + int(round(pv))
            if cy - m < 0 or cx - m < 0 or cy + W + m > H or cx + W + m > Wd:
                continue
            srch = b[cy - m:cy + W + m, cx - m:cx + W + m]
            r = cv2.matchTemplate(srch, tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if mx < 0.3:
                continue
            px, py = subpix(r, loc[0], loc[1])
            out.append((x, y, px - m + (cx - x), py - m + (cy - y), mx))
    return np.array(out)


def main():
    log("P0c: T1 image motion by window cross-correlation (hp images, 64 px windows)")
    res = {"cameras": {}}
    rec = {}
    for cam in t1.CAMS:
        bg = np.load(os.path.join(t1.DATA, "T1bg", f"{cam}_p90.npy"))
        cache = {}

        def hp(k):
            if k not in cache:
                cache[k] = lines.highpass(t1.read(cam, k), bg).astype(np.float32)
            return cache[k]

        out = {}
        log(f"\n{cam}")
        prev = {}
        for k in KS:
            all_d, per_start = [], []
            for f0 in STARTS:
                f0 = min(f0, 1000 - k)
                pr = None
                if k > 1:
                    src = prev.get((f0, k // 2))
                    pr = {(int(r[0]), int(r[1])): (2 * r[2], 2 * r[3]) for r in src} if src is not None else {}
                d = displacements(hp(f0), hp(f0 + k), k, pr)
                prev[(f0, k)] = d
                if len(d) == 0:
                    continue
                all_d.append(d)
                per_start.append({"start": f0, "windows": int(len(d)), "du_median": float(np.median(d[:, 2])),
                                  "dv_median": float(np.median(d[:, 3]))})
                if k == 8 and f0 == 400:
                    rec[cam] = d
            D = np.vstack(all_d)
            D = D.copy()
            D[:, 0] += W / 2
            D[:, 1] += W / 2
            sp = np.hypot(D[:, 2], D[:, 3])
            o = {"windows": int(len(D)), "corr_median": float(np.median(D[:, 4])),
                 "du_p5_50_95": np.percentile(D[:, 2], [5, 50, 95]).tolist(),
                 "dv_p5_50_95": np.percentile(D[:, 3], [5, 50, 95]).tolist(),
                 "disp_p50_90_95_99": np.percentile(sp, [50, 90, 95, 99]).tolist(),
                 "speed_px_per_frame_median": float(np.median(sp) / k), "per_start": per_start}
            out[str(k)] = o
            log(f"  k = {k:2d}: {o['windows']} windows, correlation median {o['corr_median']:.2f}; du 5/50/95 % "
                f"{o['du_p5_50_95'][0]:+.2f}/{o['du_p5_50_95'][1]:+.2f}/{o['du_p5_50_95'][2]:+.2f}, dv "
                f"{o['dv_p5_50_95'][0]:+.2f}/{o['dv_p5_50_95'][1]:+.2f}/{o['dv_p5_50_95'][2]:+.2f} px; |d| 50/90/95/99 % "
                + "/".join(f"{x:.2f}" for x in o["disp_p50_90_95_99"])
                + f" px; speed {o['speed_px_per_frame_median']:.3f} px/frame")
            log("        per start frame (du, dv medians): " + "; ".join(
                f"{p['start']}: {p['du_median']:+.2f}, {p['dv_median']:+.2f}" for p in per_start))
        k4 = out["4"]
        for k in KS:
            o = out[str(k)]
            eu, ev = k4["du_p5_50_95"][1] * k / 4, k4["dv_p5_50_95"][1] * k / 4
            off = math.hypot(o["du_p5_50_95"][1] - eu, o["dv_p5_50_95"][1] - ev) / max(math.hypot(eu, ev), 1e-6)
            o["consistent_with_k4"] = bool(off <= 0.3)
            o["median_vs_k4_scaled"] = {"expected": [eu, ev], "relative_difference": off}
        log("  consistency with k = 4 (median displacement against k/4 x k = 4's): " + "; ".join(
            f"k {k}: {'ok' if out[str(k)]['consistent_with_k4'] else 'UNRELIABLE'} "
            f"({100 * out[str(k)]['median_vs_k4_scaled']['relative_difference']:.0f} % off)" for k in KS))
        res["cameras"][cam] = out
    # recommendation
    rec_k = {}
    for cam in t1.CAMS:
        ok = [k for k in KS if res["cameras"][cam][str(k)]["disp_p50_90_95_99"][0] >= 1.0
              and res["cameras"][cam][str(k)]["disp_p50_90_95_99"][2] <= 3.0]
        if ok:
            rec_k[cam] = ok[0]
        else:
            rec_k[cam] = min(KS, key=lambda k: abs(res["cameras"][cam][str(k)]["disp_p50_90_95_99"][0] - 2.0))
    res["recommended_step_per_camera"] = rec_k
    step = max(rec_k.values())
    for k in sorted(KS):
        if all(res["cameras"][c][str(k)]["disp_p50_90_95_99"][2] <= 3.0 for c in t1.CAMS) and \
           any(res["cameras"][c][str(k)]["disp_p50_90_95_99"][0] >= 1.0 for c in t1.CAMS):
            step = k
    res["recommended_step"] = step
    log(f"\nrecommended frame step: {step} (per camera: {rec_k}); at 4,000 frames/s that is {4000 / step:.0f} "
        f"steps/s, {1000 // step + 1} processed frames of the 1,001")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for j, cam in enumerate(t1.CAMS):
        d = rec.get(cam)
        if d is not None:
            ax[j].quiver(d[:, 0], d[:, 1], 8 * d[:, 2], 8 * d[:, 3], np.hypot(d[:, 2], d[:, 3]), angles="xy",
                         scale_units="xy", scale=1, cmap="viridis")
        ax[j].set_xlim(0, 1280)
        ax[j].set_ylim(800, 0)
        ax[j].set_aspect("equal")
        ax[j].set_title(f"{cam}: displacement over 8 frames from frame 400 (arrows x 8)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "p0c_k8.png"), dpi=110)
    with open(os.path.join(HERE, "p0c_t1_motion.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(HERE, "p0c_t1_motion_out.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
