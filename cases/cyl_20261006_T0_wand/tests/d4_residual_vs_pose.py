"""D4 (diagnosis after T2, D3): do S2's residuals follow the wand's pose in the image?

    python tests/d4_residual_vs_pose.py [RESULT_FOLDER]      (default: config out_dir, S2's result)

A centre bias from the rod depends on how the rod lies in the image: its angle (the tube stretches the image 1.3x
across its axis, so the rod looks wider when it is steep) and its foreshortening (the image distance A-B).  This
bins each camera's residuals, split along the wand's image (positive = away from the other ball) and across it, by
the rod's angle from the image rows and by the image distance A-B, for each ball; and by the ball's edge RMS (a
proxy for blur and for the string crossing the outline).  A model error (tube, lens) would not depend on these.
Writes d4_residual_vs_pose_out.txt and d4_residual_vs_pose.json beside this script.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

import numpy as np  # noqa: E402


def binned(x, y, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi) & np.isfinite(y)
        if s.sum() >= 20:
            out.append({"bin": [float(lo), float(hi)], "n": int(s.sum()), "mean": float(np.mean(y[s])),
                        "se": float(np.std(y[s]) / math.sqrt(s.sum()))})
    return out


def main():
    cfg = case.load_config()
    folder = sys.argv[1] if len(sys.argv) > 1 else cfg["out_dir"]
    log = case.Log(os.path.join(HERE, "d4_residual_vs_pose"))
    _dets, kept, _reasons, _medians, _frames = case.load_selection(cfg)
    with open(os.path.join(folder, "calibration_data.json"), encoding="utf-8") as fh:
        ob = json.load(fh)["observations"]
    fid, cam, end = np.array(ob["frame_id"]), np.array(ob["cam_id"]), np.array(ob["end"])
    du = np.array([np.nan if x is None else x for x in ob["du"]], float)
    dv = np.array([np.nan if x is None else x for x in ob["dv"]], float)
    act = np.array(ob["active"], bool) & np.isfinite(du)
    n = len(fid)
    ang, dist, along, across, erms = (np.full(n, np.nan) for _ in range(5))
    for i in range(n):
        k = (int(fid[i]), int(cam[i]))
        if k not in kept:
            continue
        A, B = kept[k]
        e = np.array(B[:2]) - np.array(A[:2])
        L = float(np.linalg.norm(e))
        e /= L
        s = 1.0 if end[i] == 1 else -1.0
        along[i] = s * (du[i] * e[0] + dv[i] * e[1])
        across[i] = -du[i] * e[1] + dv[i] * e[0]
        ang[i] = abs(((math.degrees(math.atan2(e[1], e[0])) + 90.0) % 180.0) - 90.0)
        dist[i] = L
        erms[i] = 1.0 / (A if end[i] == 0 else B)[3] - 1.0
    out = {}
    log(f"D4: residuals of {folder} by the wand's pose in the image (along = away from the other ball)")
    for c in sorted(set(cam)):
        for e, name in ((0, "A"), (1, "B")):
            s = act & (cam == c) & (end == e)
            rec = {"angle": binned(ang[s], along[s], [0, 10, 20, 30, 45, 60, 90.1]),
                   "angle_across": binned(ang[s], across[s], [0, 10, 20, 30, 45, 60, 90.1]),
                   "distance": binned(dist[s], along[s], [0, 200, 300, 400, 500, 600, 800]),
                   "edge_rms": binned(erms[s], np.hypot(du[s], dv[s]), [0, 0.2, 0.25, 0.3, 0.4, 0.6, 1.01])}
            out[f"cam{c}_{name}"] = rec
            log(f"  cam{c} ball {name}:")
            log("    along, by rod angle (deg): " + "; ".join(
                f"{b['bin'][0]:.0f}-{b['bin'][1]:.0f}: {b['mean']:+.3f}+-{b['se']:.3f} (n {b['n']})" for b in rec["angle"]))
            log("    across, by rod angle (deg): " + "; ".join(
                f"{b['bin'][0]:.0f}-{b['bin'][1]:.0f}: {b['mean']:+.3f}+-{b['se']:.3f}" for b in rec["angle_across"]))
            log("    along, by image distance A-B (px): " + "; ".join(
                f"{b['bin'][0]:.0f}-{b['bin'][1]:.0f}: {b['mean']:+.3f}+-{b['se']:.3f} (n {b['n']})" for b in rec["distance"]))
            log("    |residual|, by edge RMS (px): " + "; ".join(
                f"{b['bin'][0]:.2f}-{b['bin'][1]:.2f}: {b['mean']:.3f} (n {b['n']})" for b in rec["edge_rms"]))
    log.save(out)


if __name__ == "__main__":
    main()
