"""D1 (diagnosis; plan section 3, "residuals"): why does T0 fit to 0.98 px when 8 Oct's run fit to 0.70 px?

    python tests/d1_residual_causes.py [RESULT_FOLDER]      (default: config out_dir, C1's result)

Per camera, the used detections' residuals (calibration_data.json: du, dv = model - detection) against what the
detector knew about each camera-frame (the kept rows of case.load_selection):
  - the ball's aspect ratio, and in particular the camera-frames that only the raised aspect limit admits (some ball
    above 8 Oct's 1.45: plan D2);
  - the gap between the two balls' ellipses (px; small when the wand is seen nearly end-on);
  - the edge RMS of the ball's fit;
  - the ball's distance from the image's border;
  - the frames whose free-triangulation wand length is off by more than 150 um (the bursts of c1_wand_length.png):
    what their detections look like in each camera.
Each as the 2-D residual RMS per bin with the count, for ball A (small) and B (large) together.
Writes d1_residual_causes_out.txt and d1_residual_causes.json beside this script.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

import numpy as np  # noqa: E402


def ellipse_gap(e1, e2):
    """ball_ellipses.ellipse_gap on kept rows [u, v, r, metric, a, b, theta, dropped]: the gap (px) between the two
    ellipses along the line joining their centres."""
    du, dv = e2[0] - e1[0], e2[1] - e1[1]
    phi = math.atan2(dv, du)

    def rad(e, ang):
        return 1.0 / math.sqrt((math.cos(ang - e[6]) / e[4]) ** 2 + (math.sin(ang - e[6]) / e[5]) ** 2)
    return math.hypot(du, dv) - rad(e1, phi) - rad(e2, phi + math.pi)


def binned(log, title, x, r2, edges, fmt="{:.2f}"):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi)
        n = int(s.sum())
        if n == 0:
            continue
        rows.append({"lo": lo, "hi": hi, "n": n, "rms_px": float(np.sqrt(np.mean(r2[s])))})
    log(f"    {title}: " + "; ".join(f"[{fmt.format(d['lo'])}, {fmt.format(d['hi'])}) n {d['n']} RMS "
                                    f"{d['rms_px']:.3f}" for d in rows))
    return rows


def main():
    cfg = case.load_config()
    folder = sys.argv[1] if len(sys.argv) > 1 else cfg["out_dir"]
    log = case.Log(os.path.join(HERE, "d1_residual_causes"))
    _dets, kept, _reasons, _medians, _frames = case.load_selection(cfg)
    with open(os.path.join(folder, "calibration_data.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    with open(os.path.join(folder, "calibration_result.json"), encoding="utf-8") as fh:
        result = json.load(fh)
    ob = data["observations"]
    fid, cam, end = np.array(ob["frame_id"]), np.array(ob["cam_id"]), np.array(ob["end"])
    u, v = np.array(ob["u"], float), np.array(ob["v"], float)
    du = np.array([np.nan if x is None else x for x in ob["du"]], float)
    dv = np.array([np.nan if x is None else x for x in ob["dv"]], float)
    act = np.array(ob["active"], bool) & np.isfinite(du)
    r2 = du ** 2 + dv ** 2
    lo45 = 1.45
    log(f"D1: {folder}: RMS {result['rms_px']:.4f} px, {int(act.sum())} of {len(act)} detections used")
    n = len(fid)
    asp_ball, asp_max, gap, erms, border = (np.full(n, np.nan) for _ in range(5))
    for i in range(n):
        k = (int(fid[i]), int(cam[i]))
        if k not in kept:
            continue
        a, b = kept[k]
        row = (a, b)[int(end[i])]
        asp_ball[i] = case.aspect(row)
        asp_max[i] = max(case.aspect(a), case.aspect(b))
        gap[i] = ellipse_gap(a, b)
        erms[i] = case.edge_rms(row)
        W, H = case.size(cfg, int(cam[i]))
        border[i] = min(row[0], row[1], W - 1 - row[0], H - 1 - row[1])
    out = {"cameras": {}}
    for c in sorted(set(cam)):
        s = act & (cam == c)
        rec = {"n": int(s.sum()), "rms_px": float(np.sqrt(np.mean(r2[s])))}
        log(f"  cam{c}: {rec['n']} used detections, RMS {rec['rms_px']:.3f} px")
        hi = s & (asp_max > lo45)
        rec["admitted_by_aspect_1_60"] = {"n": int(hi.sum()), "frames": int(len(set(fid[hi]))),
                                          "rms_px": float(np.sqrt(np.mean(r2[hi]))) if hi.any() else None,
                                          "rms_rest_px": float(np.sqrt(np.mean(r2[s & ~(asp_max > lo45)])))}
        d = rec["admitted_by_aspect_1_60"]
        log(f"    camera-frames with a ball above aspect {lo45} (admitted by the 1.60 limit): {d['n']} detections in "
            f"{d['frames']} frames, RMS {d['rms_px'] if d['rms_px'] is None else round(d['rms_px'], 3)} px; the "
            f"others {d['rms_rest_px']:.3f} px")
        rec["by_aspect"] = binned(log, "by the ball's aspect", asp_ball[s], r2[s],
                                  [1.0, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.6, 2.0])
        rec["by_gap"] = binned(log, "by the gap between the ellipses (px)", gap[s], r2[s],
                               [-1e9, 3, 10, 20, 40, 80, 160, 320, 1e9], "{:.0f}")
        rec["by_edge_rms"] = binned(log, "by the fit's edge RMS (px)", erms[s], r2[s],
                                    [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
        rec["by_border"] = binned(log, "by the distance from the image's border (px)", border[s], r2[s],
                                  [0, 25, 50, 100, 200, 400, 1e9], "{:.0f}")
        for name in ("A", "B"):
            e_ = 0 if name == "A" else 1
            t = s & (end == e_)
            rec[f"rms_{name}"] = float(np.sqrt(np.mean(r2[t])))
        out["cameras"][f"cam{c}"] = rec

    # the frames whose free-triangulation wand length is far off (the bursts)
    L = np.array([np.nan if x is None else x for x in data["wand_length_free_mm"]], float)
    fids = np.array(data["frame_ids"])
    okL = np.isfinite(L) & np.array(data["frame_active"], bool)
    dL = 1e3 * (L - float(result["wand_length_mm"]))
    bad = set(int(f) for f in fids[okL & (np.abs(dL) > 150.0)])
    log(f"  frames with |wand length error| > 150 um: {len(bad)} of {int(okL.sum())}")
    burst = {}
    for c in sorted(set(cam)):
        s = act & (cam == c)
        inb = s & np.isin(fid, list(bad))
        burst[f"cam{c}"] = {"n": int(inb.sum()), "rms_px": float(np.sqrt(np.mean(r2[inb]))) if inb.any() else None,
                            "gap_median_px": float(np.nanmedian(gap[inb])) if inb.any() else None,
                            "gap_median_all_px": float(np.nanmedian(gap[s])),
                            "aspect_max_median": float(np.nanmedian(asp_max[inb])) if inb.any() else None}
        b = burst[f"cam{c}"]
        log(f"    cam{c} in those frames: {b['n']} detections, RMS {b['rms_px'] if b['rms_px'] is None else round(b['rms_px'], 3)} px; "
            f"gap median {b['gap_median_px'] if b['gap_median_px'] is None else round(b['gap_median_px'], 1)} px "
            f"(all frames {b['gap_median_all_px']:.1f})")
    nc = {f: sum(1 for c in sorted(set(cam)) if (f, int(c)) in kept) for f in bad}
    log(f"    cameras that kept those frames: " + ", ".join(f"{k} cameras: {sum(1 for x in nc.values() if x == k)}"
                                                       for k in (2, 3)))
    out["wand_length_bursts"] = {"frames": len(bad), "per_camera": burst,
                                 "cameras_kept": {str(k): sum(1 for x in nc.values() if x == k) for k in (2, 3)}}
    log.save(out)


if __name__ == "__main__":
    main()
