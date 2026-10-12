"""L2 on T1 (plan: Phase 4): the line detector on real frames, against today's C++ tracer finder.

    python tests/l2_t1.py [STEP]        (default every 100th frame; after p0b_t1_survey.py and l1_preprocess.py)

Per camera, on every STEP-th frame of T1 (local copy): L1's preprocessing (P0b's background, L1's gain), the noise
in the 8-bit image (gain * lines.noise_left(hp)), then
  - lines.detect: count per frame and by class (H, V, ellipse, round); time per frame;
  - the C++ finder (ObjectFinder2D, radius 2 px) at min_obj_int = 3 and 4 noise sigmas: count per frame;
  - shapes of the detected objects with score >= 10: fitted RMS widths wx, wy -> the aperture-image semi-axes
    a = 2 sqrt(wx^2 - s0^2), b = 2 sqrt(wy^2 - s0^2), with s0 the median across-width of long lines; the
    astigmatic model a = alpha |t - tx|, b = beta |t - ty| puts the objects between the two focal depths on the line
    a / a_H + b / b_V = 1 (a_H: the H lines' semi-length, b_V: the V lines'), those beyond on a / a_H - b / b_V = +-1;
    reported: a_H, b_V, the distribution of q = a / a_H + b / b_V and r = |a / a_H - b / b_V| for the ellipses;
  - field dependence: a_H and b_V in a 3 x 3 grid of the image;
  - flux against the semi-axes (is the flux conserved as the image stretches?).
Figures: l2_t1_<cam>_overlay.png (a 320 x 200 px crop of frame 500, x 3, new detections as boxes coloured by class,
C++ detections as dots), l2_t1_shapes.png (a-b plane, histograms of lengths, flux).
Writes l2_t1_out.txt and l2_t1.json beside itself and the detections to data/T1det/<cam>_<frame>.npy.
"""
import json
import math
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
sys.path.insert(0, os.path.join(CASE, "data", "pyd_lines"))
import pyopenlpt as lpt  # noqa: E402
import lines  # noqa: E402
import t1  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOG = []
DET_DIR = os.path.join(t1.DATA, "T1det")
COL = {"H": (0, 140, 255), "V": (255, 120, 0), "ellipse": (0, 170, 0), "round": (200, 0, 200)}   # BGR: orange, blue


def log(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def cpp_find(img8, thr):
    cfg = lpt.TracerConfig()
    cfg._radius_obj = 2.0
    cfg._min_obj_int = int(round(thr))
    objs = lpt.ObjectFinder2D().findObject2D(lpt.Image.from_numpy(img8), cfg)
    return np.array([[o._pt_center[0], o._pt_center[1]] for o in objs]) if objs else np.zeros((0, 2))


def overlay(img8, d, cpp, path, r0=300, c0=480, h=200, w=320, z=3):
    crop = img8[r0:r0 + h, c0:c0 + w]
    vis = cv2.resize(255 - np.clip(crop.astype(np.int32) * 2, 0, 255).astype(np.uint8), None, fx=z, fy=z,
                     interpolation=cv2.INTER_NEAREST)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for o in d:
        u, v = (o["u"] - c0 + 0.5) * z, (o["v"] - r0 + 0.5) * z
        if not (-50 < u < w * z + 50 and -50 < v < h * z + 50):
            continue
        hx = max(o["Lx"] / 2 + o["sx"], 1.0) * z
        hy = max(o["Ly"] / 2 + o["sy"], 1.0) * z
        cv2.rectangle(vis, (int(u - hx), int(v - hy)), (int(u + hx), int(v + hy)), COL.get(str(o["cls"]), (0, 0, 0)), 1)
        cv2.drawMarker(vis, (int(u), int(v)), COL.get(str(o["cls"]), (0, 0, 0)), cv2.MARKER_CROSS, 6, 1)
    for (u, v) in cpp:
        uu, vv = (u - c0 + 0.5) * z, (v - r0 + 0.5) * z
        if 0 <= uu < w * z and 0 <= vv < h * z:
            cv2.circle(vis, (int(uu), int(vv)), 2, (0, 0, 220), -1)
    cv2.putText(vis, "boxes: lines.detect (H orange, V blue, ellipse green, round purple); red dots: C++ finder 4 sigma",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.imwrite(path, vis)


def main():
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    os.makedirs(DET_DIR, exist_ok=True)
    with open(os.path.join(HERE, "l1_preprocess.json"), encoding="utf-8") as fh:
        l1 = json.load(fh)["T1"]
    P = lines.det_params()
    frames = t1.frames()[::step]
    out = {"step": step, "frames": frames, "params": P, "cameras": {}}
    log(f"L2 on T1: every {step}th frame ({len(frames)} frames per camera), lines.detect with {P['k_detect']} sigma")
    allshape = {}
    for cam in t1.CAMS:
        gain = float(l1[cam]["gain"])
        bg = np.load(os.path.join(t1.DATA, "T1bg", f"{cam}_p90.npy"))
        rows, times, cnt3, cnt4, cls_counts = [], [], [], [], Counter()
        dets = []
        for k in frames:
            hp, img8 = lines.preprocess(t1.read(cam, k), bg, gain)
            noise = gain * lines.noise_left(hp)
            tm = {}
            d = lines.detect(img8, noise, P, timing=tm)
            np.save(os.path.join(DET_DIR, f"{cam}_{k:06d}.npy"), d)
            times.append(tm["total_s"])
            c3 = cpp_find(img8, 3 * noise)
            c4 = cpp_find(img8, 4 * noise)
            cnt3.append(len(c3))
            cnt4.append(len(c4))
            cc = Counter(d["cls"].tolist())
            cls_counts.update(cc)
            rows.append({"frame": k, "noise_out": noise, "n": int(len(d)), "classes": dict(cc), "cpp_3sigma": len(c3),
                         "cpp_4sigma": len(c4), "seconds": tm["total_s"], "second_pass": int((d["pass_"] == 1).sum())})
            dets.append(d)
            log(f"  {cam} frame {k:4d}: noise {noise:.1f} counts; lines.detect {len(d)} objects ("
                + ", ".join(f"{c} {cc.get(c, 0)}" for c in ("H", "V", "ellipse", "round"))
                + f"; {int((d['pass_'] == 1).sum())} from the second pass) in {tm['total_s']:.1f} s; C++ finder "
                  f"{len(c3)} at 3 sigma, {len(c4)} at 4 sigma")
            if k == 500:
                overlay(img8, d, c4, os.path.join(HERE, f"l2_t1_{cam}_overlay.png"))
        D = np.concatenate(dets)
        good = D[D["score"] >= 10]
        Hl = good[good["cls"] == "H"]
        Vl = good[good["cls"] == "V"]
        s0 = float(np.median(np.concatenate([Hl["wy"][Hl["Lx"] >= 12], Vl["wx"][Vl["Ly"] >= 8]])))
        a = 2 * np.sqrt(np.maximum(good["wx"] ** 2 - s0 ** 2, 0))
        b = 2 * np.sqrt(np.maximum(good["wy"] ** 2 - s0 ** 2, 0))
        aH = float(np.median(a[good["cls"] == "H"]))
        bV = float(np.median(b[good["cls"] == "V"]))
        el = good["cls"] == "ellipse"
        q = a[el] / aH + b[el] / bV
        r = np.abs(a[el] / aH - b[el] / bV)
        res = {"frames": rows, "gain": gain,
               "objects_per_frame_median": float(np.median([x["n"] for x in rows])),
               "class_fractions": {c: cls_counts.get(c, 0) / max(sum(cls_counts.values()), 1)
                                   for c in ("H", "V", "ellipse", "round")},
               "cpp_3sigma_per_frame_median": float(np.median(cnt3)), "cpp_4sigma_per_frame_median": float(np.median(cnt4)),
               "seconds_per_frame_median": float(np.median(times)),
               "shapes_score_ge10": {"n": int(len(good)), "s0_px": s0, "a_H_median_px": aH, "b_V_median_px": bV,
                                     "a_H_p10_p90": np.percentile(a[good["cls"] == "H"], [10, 90]).tolist(),
                                     "b_V_p10_p90": np.percentile(b[good["cls"] == "V"], [10, 90]).tolist(),
                                     "H_full_length_px_p10_50_90": (np.percentile(Hl["Lx"], [10, 50, 90]).tolist()),
                                     "V_full_length_px_p10_50_90": (np.percentile(Vl["Ly"], [10, 50, 90]).tolist()),
                                     "ellipse_q_p10_50_90": np.percentile(q, [10, 50, 90]).tolist() if len(q) else None,
                                     "ellipse_r_p10_50_90": np.percentile(r, [10, 50, 90]).tolist() if len(r) else None,
                                     "ellipse_frac_q_0p7_1p3": float(((q > 0.7) & (q < 1.3)).mean()) if len(q) else None}}
        # field dependence
        grid = []
        for gy in range(3):
            for gx in range(3):
                s = (good["u"] >= gx * 1280 / 3) & (good["u"] < (gx + 1) * 1280 / 3) & \
                    (good["v"] >= gy * 800 / 3) & (good["v"] < (gy + 1) * 800 / 3)
                hs, vs = s & (good["cls"] == "H"), s & (good["cls"] == "V")
                grid.append({"cell": [gx, gy], "n": int(s.sum()), "frac_H": float(hs.sum() / max(s.sum(), 1)),
                             "frac_V": float(vs.sum() / max(s.sum(), 1)),
                             "a_H_median": float(np.median(a[hs])) if hs.sum() >= 5 else None,
                             "b_V_median": float(np.median(b[vs])) if vs.sum() >= 5 else None})
        res["field_3x3"] = grid
        # flux against size
        size = np.maximum(a, b)
        fb = []
        for lo, hi in ((0, 2), (2, 4), (4, 7), (7, 11), (11, 16), (16, 30)):
            s = (size >= lo) & (size < hi)
            if s.sum() >= 10:
                fb.append({"semi_axis_px": [lo, hi], "n": int(s.sum()), "flux_median": float(np.median(good["flux"][s])),
                           "peak_median": float(np.median(good["peak"][s]))})
        res["flux_vs_size"] = fb
        out["cameras"][cam] = res
        allshape[cam] = (good, a, b, aH, bV)
        sh = res["shapes_score_ge10"]
        log(f"  {cam}: median {res['objects_per_frame_median']:.0f} objects per frame (C++ finder: "
            f"{res['cpp_3sigma_per_frame_median']:.0f} at 3 sigma, {res['cpp_4sigma_per_frame_median']:.0f} at 4 sigma); "
            "classes " + ", ".join(f"{c} {100 * v:.0f} %" for c, v in res["class_fractions"].items())
            + f"; {res['seconds_per_frame_median']:.1f} s per frame")
        log(f"    shapes (score >= 10, n {sh['n']}): blur s0 {s0:.2f} px; H lines full length (fitted box) 10/50/90 % "
            + "/".join(f"{x:.1f}" for x in sh["H_full_length_px_p10_50_90"]) + " px, V lines "
            + "/".join(f"{x:.1f}" for x in sh["V_full_length_px_p10_50_90"]) + f" px; semi-axes a_H {aH:.1f} px "
            f"(10-90 % {sh['a_H_p10_p90'][0]:.1f}-{sh['a_H_p10_p90'][1]:.1f}), b_V {bV:.1f} px "
            f"({sh['b_V_p10_p90'][0]:.1f}-{sh['b_V_p10_p90'][1]:.1f})")
        if sh["ellipse_q_p10_50_90"]:
            log(f"    ellipses: q = a/a_H + b/b_V 10/50/90 % " + "/".join(f"{x:.2f}" for x in sh["ellipse_q_p10_50_90"])
                + f" ({100 * sh['ellipse_frac_q_0p7_1p3']:.0f} % in 0.7-1.3); r = |a/a_H - b/b_V| "
                + "/".join(f"{x:.2f}" for x in sh["ellipse_r_p10_50_90"]))
        log("    field (3 x 3 cells, left to right, top to bottom): " + "; ".join(
            f"({g['cell'][0]},{g['cell'][1]}) n {g['n']} H {100 * g['frac_H']:.0f} % V {100 * g['frac_V']:.0f} % a_H "
            f"{g['a_H_median'] if g['a_H_median'] is None else round(g['a_H_median'], 1)} b_V "
            f"{g['b_V_median'] if g['b_V_median'] is None else round(g['b_V_median'], 1)}" for g in grid))
        log("    flux against the larger semi-axis: " + "; ".join(
            f"{f['semi_axis_px'][0]}-{f['semi_axis_px'][1]} px: n {f['n']}, flux {f['flux_median']:.0f}, peak "
            f"{f['peak_median']:.1f}" for f in fb))
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for j, cam in enumerate(t1.CAMS):
        good, a, b, aH, bV = allshape[cam]
        ax[j, 0].hist2d(a, b, bins=[np.linspace(0, 30, 61), np.linspace(0, 15, 61)], cmap="Greys", cmin=1)
        tt = np.linspace(0, 1, 50)
        ax[j, 0].plot(aH * tt, bV * (1 - tt), "r-", lw=1, label="a/a_H + b/b_V = 1")
        ax[j, 0].plot(aH * (1 + tt), bV * tt, "b--", lw=1, label="|a/a_H - b/b_V| = 1")
        ax[j, 0].plot(aH * tt, bV * (1 + tt), "b--", lw=1)
        ax[j, 0].set_xlabel("a (px, horizontal semi-axis)")
        ax[j, 0].set_ylabel("b (px, vertical semi-axis)")
        ax[j, 0].set_title(f"{cam}: objects with score >= 10")
        ax[j, 0].legend(fontsize=7)
        ax[j, 1].hist([good["Lx"][good["cls"] == "H"], good["Ly"][good["cls"] == "V"]], bins=np.arange(0, 61, 2),
                      histtype="step", label=["H: Lx", "V: Ly"])
        ax[j, 1].set_xlabel("fitted box length (px)")
        ax[j, 1].legend()
        ax[j, 2].scatter(np.maximum(a, b), good["flux"], s=2, alpha=0.3)
        ax[j, 2].set_xlabel("larger semi-axis (px)")
        ax[j, 2].set_ylabel("fitted flux (8-bit counts x px)")
        ax[j, 2].set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "l2_t1_shapes.png"), dpi=100)
    with open(os.path.join(HERE, "l2_t1.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    with open(os.path.join(HERE, "l2_t1_out.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
