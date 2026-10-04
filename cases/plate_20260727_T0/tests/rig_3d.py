"""The rig in 3-D and the wand's volume, for the report page (tests/rig_3d.json) and the report.

The frame is anchored on the two walls.  y is along wall 1's normal, into the water. x is wall 2's normal made
perpendicular to y, also into the water. z = x cross y runs along the walls' corner line. The origin is on that line
(both walls' camera-side surfaces) at the wand's median height.  "Up" is the direction from C2 to C1 (the photo: C1
above C2, C3 above C4; checked against C3 and C4).  The script writes, in that frame (mm):
  * both fits' cameras (the delivered T2 and the GUI's calib_3): centre, image axes, f and image size;
  * the walls' normals and thicknesses;
  * a subsample of the delivered fit's wand ends;
  * the wand's volume.  Its extent along each wall's normal is measured from that wall's water-side surface, with
    the height along the corner line, at the percentiles 0, 1, 5, 50, 95, 99, 100.  Also the 1-99 % box, the convex
    hull's volume, and each camera's distance to the volume's centre.

    python tests/rig_3d.py
Writes rig_3d.json and rig_3d_out.txt beside this script.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.spatial import ConvexHull

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402
import evaluate as ev  # noqa: E402

PCT = [0, 1, 5, 50, 95, 99, 100]
N_POINTS = 2000


def frame(cams, rig, P):
    """(origin, axes rows ex, ey, ez, up sign) of the walls' frame for this rig and wand ends P."""
    w1, w2 = rig.windows[0], rig.windows[1]
    ey = w1.n
    ex = w2.n - (w2.n @ ey) * ey
    ex /= np.linalg.norm(ex)
    ez = np.cross(ex, ey)
    h0 = float(np.median(P @ ez))
    O = np.linalg.solve(np.vstack([w1.n, w2.n, ez]), np.array([w1.d, w2.d, h0]))
    B = np.vstack([ex, ey, ez])
    up = 1.0 if (B @ (cams[0].C - O))[2] > (B @ (cams[1].C - O))[2] else -1.0
    return O, B, up


def cameras(cams, rig, O, B):
    """Each camera in the frame, with its rays through the image's four corners and its centre, refracted through
    its wall by the plate model: where each leaves the wall into the water (o) and its direction there (d)."""
    out = []
    for c, cam in enumerate(cams):
        uv = np.array([[0.0, 0.0], [cam.width - 1.0, 0.0], [cam.width - 1.0, cam.height - 1.0],
                       [0.0, cam.height - 1.0], [cam.cx, cam.cy]])
        idx = np.full(len(uv), c)
        o, d, ok = case.pm.rays(case.pw.cm.rows(cams, idx),
                                case.pm.wrows(rig.windows, np.full(len(uv), rig.cam_win[c])), uv)
        out.append({"name": f"C{c + 1}", "cam": c, "wall": rig.cam_win[c] + 1, "C": (B @ (cam.C - O)).tolist(),
                    "x_axis": (B @ cam.R[0]).tolist(), "y_axis": (B @ cam.R[1]).tolist(),
                    "z_axis": (B @ cam.R[2]).tolist(), "f": cam.f, "w": cam.width, "h": cam.height,
                    "rays": [{"o": (B @ (oi - O)).tolist(), "d": (B @ di).tolist()} for oi, di in zip(o, d)],
                    "rays_ok": bool(np.all(ok))})
    return out


def walls(rig, B):
    return [{"n": (B @ w.n).tolist(), "t": w.T} for w in rig.windows]


def pct(v):
    return {str(p): float(np.percentile(v, p)) for p in PCT}


def main():
    cfg = case.load_config()
    st = case.load_state(os.path.join(HERE, "states", "t2_all.pkl"))
    cams, rig = st["cams"], st["rig"]
    fr = case.pw.Frames(np.asarray(st["frames_m"]), np.asarray(st["frames_w"]))
    XA, XB = fr.ends(st["L"])
    act = np.zeros(len(XA), bool)
    act[np.asarray(st["obs_frame"])[np.asarray(st["active"], bool)]] = True
    P = np.vstack([XA[act], XB[act]])
    O, B, up = frame(cams, rig, P)
    Q = (B @ (P - O).T).T
    w1, w2 = rig.windows
    depth1 = P @ w1.n - w1.far_d          # past wall 1's water-side surface
    depth2 = P @ w2.n - w2.far_d
    height = up * Q[:, 2]
    box = {k: [float(np.percentile(v, 1)), float(np.percentile(v, 99))]
           for k, v in (("depth1", depth1), ("depth2", depth2), ("height", height))}
    hull = ConvexHull(P)
    centre = np.median(P, axis=0)
    cam_to_centre = [float(np.linalg.norm(c.C - centre)) for c in cams]
    # calib_3 in its own walls' frame, its wand ends triangulated from every frame without a gross detection error
    g_cams, g_rig = case.load_gui(cfg, "calib_3")
    rc = case.rig_config(cfg)
    cd = case.read_detections(cfg, "all")
    obs, X, r = ev.triangulate_all(g_cams, g_rig, cd.uv, rc.wand_length_mm, rc.sizes)
    Lm = np.linalg.norm(X[1::2] - X[0::2], axis=1)
    good = np.abs(Lm - rc.wand_length_mm) < 0.5
    Pg = np.vstack([X[0::2][good], X[1::2][good]])
    Og, Bg, up_g = frame(g_cams, g_rig, Pg)
    take = np.linspace(0, len(Q) - 1, N_POINTS).round().astype(int)
    out = {
        "frame": "x: wall 2's normal made perpendicular to y; y: wall 1's normal (both into the water); z = x cross y "
                 "along the walls' corner line; origin on the corner line (the walls' camera-side surfaces) at the wand's "
                 "median height; mm",
        "up": up, "up_calib_3": up_g,
        "T2": {"cameras": cameras(cams, rig, O, B), "walls": walls(rig, B),
               "points": np.round(Q[take], 2).tolist()},
        "calib_3": {"cameras": cameras(g_cams, g_rig, Og, Bg), "walls": walls(g_rig, Bg)},
        "box_frame_mm": [[float(np.percentile(Q[:, k], 1)), float(np.percentile(Q[:, k], 99))] for k in range(3)],
        "volume": {"depth_from_wall1_mm": pct(depth1), "depth_from_wall2_mm": pct(depth2), "height_mm": pct(height),
                   "camera_height_mm": [float(up * (B @ (c.C - O))[2]) for c in cams],
                   "box_1_99_mm": box, "box_1_99_size_mm": {k: v[1] - v[0] for k, v in box.items()},
                   "hull_cm3": hull.volume / 1000.0, "box_1_99_cm3": float(np.prod([v[1] - v[0] for v in box.values()]) / 1000.0),
                   "ends": int(len(P)), "camera_to_centre_mm": cam_to_centre,
                   "walls_angle_deg": case.angle_deg(w1.n, w2.n)}}
    with open(os.path.join(HERE, "rig_3d.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    v = out["volume"]
    log = case.Tee(os.path.join(HERE, "rig_3d_out.txt"))
    log(f"The wand's volume in the delivered calibration (T2): {v['ends']} wand ends of {int(act.sum())} frames")
    for k, name in (("depth_from_wall1_mm", "from wall 1's water side (C1, C2's wall)"),
                    ("depth_from_wall2_mm", "from wall 2's water side (C3, C4's wall)"),
                    ("height_mm", "height along the corner line, from the median (up = C2 -> C1)")):
        q = v[k]
        log(f"  {name}: min {q['0']:.1f}, 1 % {q['1']:.1f}, 5 % {q['5']:.1f}, median {q['50']:.1f}, 95 % {q['95']:.1f}, "
            f"99 % {q['99']:.1f}, max {q['100']:.1f} mm")
    s = v["box_1_99_size_mm"]
    log(f"  1-99 % box: {s['depth1']:.1f} (along wall 1's normal) x {s['depth2']:.1f} (along wall 2's normal) x "
        f"{s['height']:.1f} (height) mm = {v['box_1_99_cm3']:.0f} cm3; convex hull of every end {v['hull_cm3']:.0f} cm3")
    log("  each camera to the volume's centre (the ends' median): " + ", ".join(
        f"C{c + 1} {d:.0f} mm" for c, d in enumerate(cam_to_centre)))
    zc = [cc["C"][2] * up for cc in out["T2"]["cameras"]]
    log(f"  heights of the cameras (up = C2 -> C1): " + ", ".join(f"C{c + 1} {z:+.0f}" for c, z in enumerate(zc))
        + f" mm; C3 above C4: {zc[2] > zc[3]}")
    log.close()


if __name__ == "__main__":
    main()
