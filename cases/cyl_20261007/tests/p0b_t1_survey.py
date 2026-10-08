"""P0b (plan: Phase 0): the tracer images of T1, read-only, from the local copy (tests/p0b_copy_t1.py).

    python tests/p0b_t1_survey.py

Per camera:
1. Background: the per-pixel 90th percentile over all 1,001 frames.  Does that suffice?  Against it, the same from
   every 10th frame (101), every 40th (26), and from each half of the recording (frames 0-500, 500-1000): the
   differences (median and 99th percentile of |diff|, counts).  The grey level along the recording (frame medians).
2. Every 50th frame (21 frames): hp = (background - frame) - median_31, recentred (lines.highpass); its noise (the
   negative tail, lines.noise_left; and the MAD); the noise's correlation between neighbouring pixels (|hp| < 3);
   the temporal noise (frames k and k + 1 differ: MAD / sqrt 2); the background's flatness: the level of hp in
   64 x 64 tiles (lines.noise_level, the negative tail, blind to the positive objects), its scatter over the tiles,
   against the same on the signal before the high-pass (background - frame: the clouds).
3. Objects: connected components (8-connected) of hp > k * noise, k = 4 and 6, at least 6 px, on every 50th frame:
   count per frame; classes by the bounding box (w, h) and the mass in the box's corners:
     H: w >= 2 h and w >= h + 3;  V: h >= 2 w and h >= w + 3;  cross: w, h >= 7 and less than 15 % of the signal
     outside the central row and column bands (|dx| > 2 and |dy| > 2 px from the component's centroid);
     round: max(w, h) <= 5 (else) ellipse;
   lengths (the long side) and thickness (the short side) of H and V; peak and summed hp (over the component dilated
   by 2 px) against length; the orientation of long H and V components from their intensity moments (the camera's
   roll against the tube's axis).
4. Figures: p0b_<cam>_hp.png (frame 500's hp, 2x crop), p0b_shapes.png (length histograms, contrast and flux
   against length).
Writes p0b_t1_survey_out.txt and p0b_t1_survey.json beside itself, and the backgrounds (float32 .npy, all frames)
to data/T1bg/<cam>_p90.npy for L1.
"""
import json
import math
import os
import sys
import time

import cv2
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import lines  # noqa: E402
import t1  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOG = []
BG_DIR = os.path.join(t1.DATA, "T1bg")


def log(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def percentile_stack(stack, q=90.0, rows_per_block=50):
    """Per-pixel q-th percentile (linear interpolation) of a uint8 stack (n, H, W), block by block."""
    n, H, W = stack.shape
    out = np.empty((H, W), np.float32)
    pos = q / 100.0 * (n - 1)
    k0 = int(math.floor(pos))
    fr = pos - k0
    for r0 in range(0, H, rows_per_block):
        blk = stack[:, r0:r0 + rows_per_block, :]
        part = np.partition(blk, (k0, min(k0 + 1, n - 1)), axis=0)
        lo = part[k0].astype(np.float32)
        hi = part[min(k0 + 1, n - 1)].astype(np.float32)
        out[r0:r0 + rows_per_block] = lo + fr * (hi - lo)
    return out


def diff_stats(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64)).ravel()
    return {"median": float(np.median(d)), "p99": float(np.percentile(d, 99)), "max": float(d.max()),
            "mean_signed": float((a.astype(np.float64) - b).mean())}


def components(hp, thr, min_area=6):
    mask = (hp > thr).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    dil = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        sl = (slice(y, y + h), slice(x, x + w))
        m = lab[sl] == i
        z = np.where(m, hp[sl], 0.0)
        tot = float(z.sum())
        rr, cc = np.mgrid[y:y + h, x:x + w]
        cy, cx = (z * rr).sum() / tot, (z * cc).sum() / tot
        dyy = ((z * (rr - cy) ** 2).sum() / tot)
        dxx = ((z * (cc - cx) ** 2).sum() / tot)
        dxy = ((z * (rr - cy) * (cc - cx)).sum() / tot)
        ang = 0.5 * math.degrees(math.atan2(2 * dxy, dxx - dyy))   # of the major axis, from +x towards +y (rows)
        corner = float(z[(np.abs(cc - cx) > 2) & (np.abs(rr - cy) > 2)].sum() / tot)
        # summed signal over the component dilated by 2 px (catches the faint ends)
        y0, y1 = max(y - 2, 0), min(y + h + 2, hp.shape[0])
        x0, x1 = max(x - 2, 0), min(x + w + 2, hp.shape[1])
        md = cv2.dilate((lab[y0:y1, x0:x1] == i).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        flux = float(hp[y0:y1, x0:x1][md].sum())
        if w >= 2 * h and w >= h + 3:
            cls = "H"
        elif h >= 2 * w and h >= w + 3:
            cls = "V"
        elif w >= 7 and h >= 7 and corner < 0.15:
            cls = "cross"
        elif max(w, h) <= 5:
            cls = "round"
        else:
            cls = "ellipse"
        out.append({"x": float(cx), "y": float(cy), "w": int(w), "h": int(h), "area": int(area), "cls": cls,
                    "peak": float(z.max()), "flux": flux, "angle_deg": ang, "corner": corner,
                    "rms_x": math.sqrt(max(dxx, 0)), "rms_y": math.sqrt(max(dyy, 0))})
    return out


def summarise(objs, frames_n, label):
    cls = [o["cls"] for o in objs]
    res = {"objects_per_frame": len(objs) / frames_n}
    for c in ("H", "V", "ellipse", "round", "cross"):
        res[f"frac_{c}"] = cls.count(c) / max(len(objs), 1)
    for c in ("H", "V"):
        L = np.array([o["w"] if c == "H" else o["h"] for o in objs if o["cls"] == c])
        T = np.array([o["h"] if c == "H" else o["w"] for o in objs if o["cls"] == c])
        if len(L):
            res[f"{c}_length_px_p10_50_90_99"] = np.percentile(L, [10, 50, 90, 99]).tolist()
            res[f"{c}_thickness_px_p10_50_90"] = np.percentile(T, [10, 50, 90]).tolist()
    return res


def main():
    log("P0b: T1 tracer images, from the local copy " + t1.LOCAL_RAW)
    os.makedirs(BG_DIR, exist_ok=True)
    out = {"cameras": {}}
    frames_all = t1.frames()
    sample = frames_all[::50]
    shapes_all = {}
    for cam in t1.CAMS:
        t0 = time.time()
        res = {}
        stack = np.stack([t1.read(cam, k) for k in frames_all])           # uint8 (1001, 800, 1280)
        log(f"\n{cam}: {stack.shape[0]} frames read in {time.time() - t0:.0f} s")
        med = np.median(stack.reshape(stack.shape[0], -1)[:, ::97], axis=1)
        res["grey_median_every_100"] = med[::100].tolist()
        log("  grey level (frame median) every 100th frame: " + " ".join(f"{m:.0f}" for m in med[::100]))
        bg = percentile_stack(stack, 90)
        np.save(os.path.join(BG_DIR, f"{cam}_p90.npy"), bg)
        bg10 = percentile_stack(stack[::10], 90)
        bg40 = percentile_stack(stack[::40], 90)
        bgA = percentile_stack(stack[:501], 90)
        bgB = percentile_stack(stack[500:], 90)
        res["bg_p90_median"] = float(np.median(bg))
        res["bg_every10_vs_all"] = diff_stats(bg10, bg)
        res["bg_every40_vs_all"] = diff_stats(bg40, bg)
        res["bg_first_vs_second_half"] = diff_stats(bgA, bgB)
        log(f"  background (90th percentile, 1,001 frames): median {res['bg_p90_median']:.1f}")
        for k in ("bg_every10_vs_all", "bg_every40_vs_all", "bg_first_vs_second_half"):
            s = res[k]
            log(f"    {k}: |diff| median {s['median']:.2f}, 99th percentile {s['p99']:.2f}, max {s['max']:.1f}; "
                f"mean signed {s['mean_signed']:+.2f} counts")
        del stack
        # every 50th frame
        noises, mads, flat, objs4, objs6 = [], [], [], [], []
        per_frame4, per_frame6 = [], []
        for k in sample:
            img = t1.read(cam, k)
            hp = lines.highpass(img, bg)
            sn = lines.noise_left(hp)
            noises.append(sn)
            mads.append(lines.robust_sigma(hp))
            o4 = components(hp, 4 * sn)
            o6 = components(hp, 6 * sn)
            for o in o4:
                o["frame"] = k
            objs4 += o4
            objs6 += o6
            per_frame4.append(len(o4))
            per_frame6.append(len(o6))
            # flatness: the level of hp (negative tail) in 64 x 64 tiles, and of the signal before the high-pass
            sig = bg - img.astype(np.float32)
            lv, lv0 = [], []
            for r0 in range(0, 800 - 63, 64):
                for c0 in range(0, 1280 - 63, 64):
                    lv.append(lines.noise_level(hp[r0:r0 + 64, c0:c0 + 64]))
                    lv0.append(lines.noise_level(sig[r0:r0 + 64, c0:c0 + 64]))
            lv, lv0 = np.array(lv), np.array(lv0)
            m = np.abs(hp) < 3
            ca = np.corrcoef(hp[:, :-1][m[:, :-1] & m[:, 1:]], hp[:, 1:][m[:, :-1] & m[:, 1:]])[0, 1]
            cb = np.corrcoef(hp[:-1, :][m[:-1, :] & m[1:, :]], hp[1:, :][m[:-1, :] & m[1:, :]])[0, 1]
            dt = (t1.read(cam, min(k + 1, 1000)).astype(np.float64) - t1.read(cam, min(k + 1, 1000) - 1)) / math.sqrt(2)
            flat.append({"tile_level_std": float(lv.std()), "tile_level_mean": float(lv.mean()),
                         "before_highpass_tile_level_std": float(lv0.std()), "corr_h": float(ca), "corr_v": float(cb),
                         "temporal_noise_mad": lines.robust_sigma(dt)})
            if k == 500:
                vis = np.clip(255 - hp * (255.0 / (10 * sn)), 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(HERE, f"p0b_{cam}_hp.png"),
                            cv2.resize(vis[300:500, 500:820], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
        res["noise_counts"] = {"median": float(np.median(noises)), "min": float(np.min(noises)),
                               "max": float(np.max(noises)), "mad_median": float(np.median(mads))}
        res["flatness"] = {"tile_level_std_median": float(np.median([f["tile_level_std"] for f in flat])),
                           "tile_level_mean_median": float(np.median([f["tile_level_mean"] for f in flat])),
                           "before_highpass_tile_level_std_median": float(np.median([f["before_highpass_tile_level_std"]
                                                                                    for f in flat]))}
        res["noise_correlation"] = {"horizontal": float(np.median([f["corr_h"] for f in flat])),
                                    "vertical": float(np.median([f["corr_v"] for f in flat]))}
        res["temporal_noise_counts"] = float(np.median([f["temporal_noise_mad"] for f in flat]))
        log(f"  noise of hp (negative tail): median {res['noise_counts']['median']:.2f} counts "
            f"({res['noise_counts']['min']:.2f}-{res['noise_counts']['max']:.2f} over the 21 frames; MAD "
            f"{res['noise_counts']['mad_median']:.2f})")
        log(f"  noise correlation between neighbouring pixels: {res['noise_correlation']['horizontal']:.2f} "
            f"(horizontal), {res['noise_correlation']['vertical']:.2f} (vertical); temporal noise (frame differences) "
            f"{res['temporal_noise_counts']:.2f} counts: the rest of hp's noise is a texture of faint, blurred "
            "particles, which moves with the flow (0.25-0.5 px per frame) and so cancels in consecutive frames' "
            "difference")
        log(f"  flatness: level of hp in 64 px tiles (negative tail) {res['flatness']['tile_level_mean_median']:+.3f} "
            f"counts, scatter {res['flatness']['tile_level_std_median']:.3f} over the tiles; before the high-pass "
            f"(background - frame) the scatter is {res['flatness']['before_highpass_tile_level_std_median']:.3f}")
        res["objects_per_frame_4sigma"] = {"median": float(np.median(per_frame4)), "min": int(min(per_frame4)),
                                           "max": int(max(per_frame4))}
        res["objects_per_frame_6sigma"] = {"median": float(np.median(per_frame6)), "min": int(min(per_frame6)),
                                           "max": int(max(per_frame6))}
        res["shapes_4sigma"] = summarise(objs4, len(sample), "4")
        res["shapes_6sigma"] = summarise(objs6, len(sample), "6")
        log(f"  objects per frame: {res['objects_per_frame_4sigma']['median']:.0f} above 4 sigma "
            f"({res['objects_per_frame_4sigma']['min']}-{res['objects_per_frame_4sigma']['max']}), "
            f"{res['objects_per_frame_6sigma']['median']:.0f} above 6 sigma")
        for lab, s in (("4 sigma", res["shapes_4sigma"]), ("6 sigma", res["shapes_6sigma"])):
            log(f"  classes at {lab}: " + ", ".join(f"{c} {100 * s['frac_' + c]:.0f} %" for c in
                                                    ("H", "V", "ellipse", "round", "cross")))
            for c in ("H", "V"):
                if f"{c}_length_px_p10_50_90_99" in s:
                    L, T = s[f"{c}_length_px_p10_50_90_99"], s[f"{c}_thickness_px_p10_50_90"]
                    log(f"    {c}: length (10/50/90/99 %) {L[0]:.0f}/{L[1]:.0f}/{L[2]:.0f}/{L[3]:.0f} px, "
                        f"thickness {T[0]:.0f}/{T[1]:.0f}/{T[2]:.0f} px")
        # orientation of long H and V components
        angH = np.array([o["angle_deg"] for o in objs4 if o["cls"] == "H" and o["w"] >= 12])
        angV = np.array([o["angle_deg"] for o in objs4 if o["cls"] == "V" and o["h"] >= 10])
        angV = np.where(angV < 0, angV + 180, angV) - 90.0      # V: major axis near 90 deg -> deviation from 90
        res["roll_deg"] = {"H_median": float(np.median(angH)) if len(angH) else None, "H_n": int(len(angH)),
                           "H_iqr": (np.percentile(angH, [25, 75]).tolist() if len(angH) else None),
                           "V_median": float(np.median(angV)) if len(angV) else None, "V_n": int(len(angV)),
                           "V_iqr": (np.percentile(angV, [25, 75]).tolist() if len(angV) else None)}
        log(f"  orientation (intensity moments): long H lines {res['roll_deg']['H_median']:+.2f} deg "
            f"(IQR {res['roll_deg']['H_iqr'][0]:+.2f}..{res['roll_deg']['H_iqr'][1]:+.2f}, n {len(angH)}); "
            f"long V lines {res['roll_deg']['V_median']:+.2f} deg from vertical "
            f"(IQR {res['roll_deg']['V_iqr'][0]:+.2f}..{res['roll_deg']['V_iqr'][1]:+.2f}, n {len(angV)})")
        # contrast and flux against length (H and V at 4 sigma)
        cvl = {}
        for c in ("H", "V"):
            sel = [o for o in objs4 if o["cls"] == c]
            L = np.array([o["w"] if c == "H" else o["h"] for o in sel])
            pk = np.array([o["peak"] for o in sel])
            fl = np.array([o["flux"] for o in sel])
            bins = [(4, 7), (7, 10), (10, 14), (14, 20), (20, 30), (30, 60)]
            rows = []
            for lo, hi in bins:
                s = (L >= lo) & (L < hi)
                if s.sum() >= 5:
                    rows.append({"length": [lo, hi], "n": int(s.sum()), "peak_median": float(np.median(pk[s])),
                                 "flux_median": float(np.median(fl[s]))})
            cvl[c] = rows
            log(f"    {c}: length bin -> n, median peak, median summed signal (counts): " +
                "; ".join(f"{r['length'][0]}-{r['length'][1]}: {r['n']}, {r['peak_median']:.1f}, {r['flux_median']:.0f}"
                          for r in rows))
        res["contrast_vs_length"] = cvl
        pk_all = np.array([o["peak"] for o in objs4])
        res["peak_percentiles_4sigma"] = np.percentile(pk_all, [50, 90, 99, 99.9]).tolist()
        log(f"  peak hp of the objects (counts): median {res['peak_percentiles_4sigma'][0]:.1f}, 90 % "
            f"{res['peak_percentiles_4sigma'][1]:.1f}, 99 % {res['peak_percentiles_4sigma'][2]:.1f}, 99.9 % "
            f"{res['peak_percentiles_4sigma'][3]:.1f}")
        shapes_all[cam] = objs4
        out["cameras"][cam] = res
        log(f"  ({time.time() - t0:.0f} s)")

    # figure: shapes
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for j, cam in enumerate(t1.CAMS):
        o = shapes_all[cam]
        LH = [x["w"] for x in o if x["cls"] == "H"]
        LV = [x["h"] for x in o if x["cls"] == "V"]
        ax[j, 0].hist([LH, LV], bins=np.arange(3, 61, 2), label=["H (width)", "V (height)"], histtype="step")
        ax[j, 0].set_title(f"{cam}: lengths above 4 sigma (px)")
        ax[j, 0].legend()
        for c, col in (("H", "C0"), ("V", "C1"), ("ellipse", "C2"), ("round", "C3"), ("cross", "C4")):
            sel = [x for x in o if x["cls"] == c]
            ax[j, 1].scatter([max(x["w"], x["h"]) for x in sel], [x["peak"] for x in sel], s=3, c=col, label=c, alpha=0.5)
            ax[j, 2].scatter([max(x["w"], x["h"]) for x in sel], [x["flux"] for x in sel], s=3, c=col, label=c, alpha=0.5)
        ax[j, 1].set_xlabel("long side (px)")
        ax[j, 1].set_ylabel("peak hp (counts)")
        ax[j, 1].set_yscale("log")
        ax[j, 1].legend(markerscale=3)
        ax[j, 2].set_xlabel("long side (px)")
        ax[j, 2].set_ylabel("summed hp (counts)")
        ax[j, 2].set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "p0b_shapes.png"), dpi=110)
    with open(os.path.join(HERE, "p0b_t1_survey.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    np.save(os.path.join(BG_DIR, "p0b_objects_4sigma.npy"), np.array(
        [(c, o["frame"], o["x"], o["y"], o["w"], o["h"], o["peak"], o["flux"], o["angle_deg"], o["corner"],
          o["rms_x"], o["rms_y"]) for c in t1.CAMS for o in shapes_all[c]],
        dtype=[("cam", "U4"), ("frame", "i4"), ("x", "f8"), ("y", "f8"), ("w", "i4"), ("h", "i4"), ("peak", "f8"),
               ("flux", "f8"), ("angle", "f8"), ("corner", "f8"), ("rms_x", "f8"), ("rms_y", "f8")]))
    with open(os.path.join(HERE, "p0b_t1_survey_out.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
