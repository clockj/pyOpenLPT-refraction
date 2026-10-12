"""The rig in 3-D and the wand's volume, for the report page (tests/rig_3d.json) and the reports.

Three calibrations, each in its own walls' frame: the first one (T2, tests/states/t2_all.pkl) and the two with the
channel (W4 of 2026-10-03-walls-plan.md): the chosen one with the principal point held at the image centre
(tests/states/w4c_ref.pkl) and the one with it fitted (tests/states/w4_ref.pkl), each when it exists.  The frame is anchored on the two
walls.  y is along wall 1's normal, into the water. x is wall 2's normal made perpendicular to y, also into the water.
z = x cross y runs along the walls' corner line. The origin is on that line (both walls' camera-side surfaces) at the
wand's median height.  "Up" is the direction from C2 to C1 (the photo: C1 above C2, C3 above C4; checked against C3
and C4).  The script writes, in that frame (mm), per calibration:
  * the cameras: centre, image axes, f and image size, and the rays through the image's corners and centre refracted
    through their wall;
  * the walls' normals and thicknesses;
  * a subsample of the wand ends;
  * the wand's volume.  Its extent along each wall's normal is measured from that wall's water-side surface and from
    the channel's far wall across it (config constraints.channel: 100 mm inside across wall 1, 150 mm across wall 2),
    with the height along the corner line, at the percentiles 0, 1, 5, 50, 95, 99, 100.  Also the 1-99 % box, the
    convex hull's volume, and each camera's distance to the volume's centre.
And calib_3's cameras and walls in its own walls' frame, and the channel's size.

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
FITS = {"T2": ("t2_all", "calib_plate_wand: the first calibration, no channel"),
        "ref": ("w4c_ref", "calib_plate_wand_ref_centre: the channel held, principal point at the image centre (chosen)"),
        "ref_pp": ("w4_ref", "calib_plate_wand_ref: the channel held, principal point fitted")}


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
        win = rig.windows[rig.cam_win[c]]
        out.append({"name": f"C{c + 1}", "cam": c, "wall": rig.cam_win[c] + 1, "C": (B @ (cam.C - O)).tolist(),
                    "x_axis": (B @ cam.R[0]).tolist(), "y_axis": (B @ cam.R[1]).tolist(),
                    "z_axis": (B @ cam.R[2]).tolist(), "f": cam.f, "w": cam.width, "h": cam.height,
                    "pp_off_px": [cam.cx - (cam.width - 1) / 2.0, cam.cy - (cam.height - 1) / 2.0],
                    "k1": float(cam.dist[0]), "distance_to_wall_mm": win.distance(cam.C),
                    "incidence_deg": case.angle_deg(cam.R[2], win.n),
                    "rays": [{"o": (B @ (oi - O)).tolist(), "d": (B @ di).tolist()} for oi, di in zip(o, d)],
                    "rays_ok": bool(np.all(ok))})
    return out


def walls(rig, B):
    return [{"n": (B @ w.n).tolist(), "t": w.T} for w in rig.windows]


def pct(v):
    return {str(p): float(np.percentile(v, p)) for p in PCT}


def fit_data(st, label: str, width) -> dict:
    """One calibration in its walls' frame: cameras, walls, wand ends and the wand's volume (see the module's doc)."""
    cams, rig = st["cams"], st["rig"]
    fr = case.pw.Frames(np.asarray(st["frames_m"]), np.asarray(st["frames_w"]))
    XA, XB = fr.ends(st["L"])
    act = np.zeros(len(XA), bool)
    act[np.asarray(st["obs_frame"])[np.asarray(st["active"], bool)]] = True
    P = np.vstack([XA[act], XB[act]])
    O, B, up = frame(cams, rig, P)
    Q = (B @ (P - O).T).T
    w1, w2 = rig.windows
    depth1 = w1.depth(P)                  # past wall 1's water-side surface
    depth2 = w2.depth(P)
    height = up * Q[:, 2]
    box = {k: [float(np.percentile(v, 1)), float(np.percentile(v, 99))]
           for k, v in (("depth1", depth1), ("depth2", depth2), ("height", height))}
    centre = np.median(P, axis=0)
    take = np.linspace(0, len(Q) - 1, N_POINTS).round().astype(int)
    return {"label": label, "up": up, "cameras": cameras(cams, rig, O, B), "walls": walls(rig, B),
            "points": np.round(Q[take], 2).tolist(),
            "box_frame_mm": [[float(np.percentile(Q[:, k], 1)), float(np.percentile(Q[:, k], 99))] for k in range(3)],
            "rms_px": st["rms"],
            "volume": {"depth_from_wall1_mm": pct(depth1), "depth_from_wall2_mm": pct(depth2),
                       "depth_from_far1_mm": pct(width[0] - depth1), "depth_from_far2_mm": pct(width[1] - depth2),
                       "height_mm": pct(height),
                       "camera_height_mm": [float(up * (B @ (c.C - O))[2]) for c in cams],
                       "box_1_99_mm": box, "box_1_99_size_mm": {k: v[1] - v[0] for k, v in box.items()},
                       "hull_cm3": ConvexHull(P).volume / 1000.0,
                       "box_1_99_cm3": float(np.prod([v[1] - v[0] for v in box.values()]) / 1000.0),
                       "ends": int(len(P)), "frames": int(act.sum()),
                       "camera_to_centre_mm": [float(np.linalg.norm(c.C - centre)) for c in cams],
                       "walls_angle_deg": case.angle_deg(w1.n, w2.n),
                       "aim_C1C2_from_wall2_mm": case.pw.aim_depth(cams, rig, [0, 1], 1),
                       "aim_C3C4_from_wall1_mm": case.pw.aim_depth(cams, rig, [2, 3], 0)}}


def main():
    cfg = case.load_config()
    ch = {k: v for k, v in cfg["constraints"]["channel"].items() if not k.startswith("_")}
    width = ch["width_mm"]
    out = {"frame": "x: wall 2's normal made perpendicular to y; y: wall 1's normal (both into the water); z = x cross y "
                    "along the walls' corner line; origin on the corner line (the walls' camera-side surfaces) at the "
                    "wand's median height; mm",
           "channel": {**ch, "note": "inside widths across wall 1 and wall 2 (the user's 15 x 10 cm, C1 and C2 on the "
                                     "long side); the far walls are drawn with wall 1's thickness"},
           "fits": {}}
    for key, (state, label) in FITS.items():
        p = os.path.join(HERE, "states", f"{state}.pkl")
        if os.path.exists(p):
            out["fits"][key] = fit_data(case.load_state(p), label, width)
    # calib_3 in its own walls' frame, its wand ends triangulated from every frame without a gross detection error
    g_cams, g_rig = case.load_gui(cfg, "calib_3")
    rc = case.rig_config(cfg)
    cd = case.read_detections(cfg, "all")
    obs, X, r = ev.triangulate_all(g_cams, g_rig, cd.uv, rc.wand_length_mm, rc.sizes)
    Lm = np.linalg.norm(X[1::2] - X[0::2], axis=1)
    good = np.abs(Lm - rc.wand_length_mm) < 0.5
    Pg = np.vstack([X[0::2][good], X[1::2][good]])
    Og, Bg, up_g = frame(g_cams, g_rig, Pg)
    out["calib_3"] = {"up": up_g, "cameras": cameras(g_cams, g_rig, Og, Bg), "walls": walls(g_rig, Bg)}
    with open(os.path.join(HERE, "rig_3d.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log = case.Tee(os.path.join(HERE, "rig_3d_out.txt"))
    for key, f in out["fits"].items():
        v = f["volume"]
        log(f"The wand's volume in {f['label']} ({key}): {v['ends']} wand ends of {v['frames']} frames; channel "
            f"{width[1]:.0f} x {width[0]:.0f} mm inside")
        for k, name in (("depth_from_wall1_mm", "from wall 1's water side (C1, C2's wall)"),
                        ("depth_from_far1_mm", f"from the far wall across wall 1 ({width[0]:.0f} mm)"),
                        ("depth_from_wall2_mm", "from wall 2's water side (C3, C4's wall)"),
                        ("depth_from_far2_mm", f"from the far wall across wall 2 ({width[1]:.0f} mm)"),
                        ("height_mm", "height along the corner line, from the median (up = C2 -> C1)")):
            q = v[k]
            log(f"  {name}: min {q['0']:.1f}, 1 % {q['1']:.1f}, 5 % {q['5']:.1f}, median {q['50']:.1f}, 95 % "
                f"{q['95']:.1f}, 99 % {q['99']:.1f}, max {q['100']:.1f} mm")
        s = v["box_1_99_size_mm"]
        log(f"  1-99 % box: {s['depth1']:.1f} (along wall 1's normal) x {s['depth2']:.1f} (along wall 2's normal) x "
            f"{s['height']:.1f} (height) mm = {v['box_1_99_cm3']:.0f} cm3; convex hull of every end {v['hull_cm3']:.0f} "
            "cm3")
        log("  each camera to the volume's centre (the ends' median): " + ", ".join(
            f"C{c + 1} {d:.0f} mm" for c, d in enumerate(v["camera_to_centre_mm"])))
        zc = [cc["C"][2] * f["up"] for cc in f["cameras"]]
        log("  heights of the cameras (up = C2 -> C1): " + ", ".join(f"C{c + 1} {z:+.0f}" for c, z in enumerate(zc))
            + f" mm; C3 above C4: {zc[2] > zc[3]}")
        log(f"  where the cameras aim: C1 and C2 {v['aim_C1C2_from_wall2_mm']:.1f} mm from wall 2's water side, C3 "
            f"and C4 {v['aim_C3C4_from_wall1_mm']:.1f} mm from wall 1's; walls {v['walls_angle_deg']:.3f} deg apart")
    log.close()


if __name__ == "__main__":
    main()
