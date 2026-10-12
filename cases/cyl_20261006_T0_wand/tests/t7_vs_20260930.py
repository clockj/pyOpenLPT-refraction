"""T7: cam1 against the dot-board calibration of 30 Sep (plan: 2026-10-06-plan.md, section 2).

    python tests/t7_vs_20260930.py [RESULT_FOLDER]      (default: config out_dir, S2's result)

Meaningful only if cam1's lens and focus are unchanged since 30 Sep (not confirmed when this was written); the camera
may also have moved a little.  Compares the quantities that do not depend on either calibration's frame:
  - f (px), against 41,459 +- 69 px (dot board through the tube) and 41,610 px (the board in air, 5 Oct);
  - the camera's distance from the tube's axis;
  - the elevation of the line of sight (its angle to the plane perpendicular to the axis);
  - the aim: the angle, seen along the axis, between the line of sight and the line from the camera to the axis;
  - the roll: the angle of the image rows (the camera's x axis) to the plane perpendicular to the axis.
Writes t7_vs_20260930_out.txt and t7_vs_20260930.json beside this script.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def geometry(C, R, axis_point, axis_dir):
    """Distance from the axis, elevation, aim and roll (deg) of a camera (centre C, rotation R: rows = camera x, y,
    z in the world) about a tube axis."""
    a = np.asarray(axis_dir, float) / np.linalg.norm(axis_dir)
    C, R, p = np.asarray(C, float), np.asarray(R, float), np.asarray(axis_point, float)
    w = C - p
    radial = w - (w @ a) * a                      # from the axis to the camera, perpendicular to the axis
    dist = float(np.linalg.norm(radial))
    d = R[2] / np.linalg.norm(R[2])                # line of sight
    elev = math.degrees(math.asin(float(d @ a)))
    dp = d - (d @ a) * a                           # seen along the axis
    to_axis = -radial / dist
    aim = math.degrees(math.atan2(float(np.cross(to_axis, dp) @ a), float(to_axis @ dp)))
    roll = math.degrees(math.asin(float(R[0] @ a) / np.linalg.norm(R[0])))
    return {"distance_from_axis_mm": dist, "elevation_deg": elev, "aim_deg": aim, "roll_deg": roll}


def main():
    cfg = case.load_config()
    folder = sys.argv[1] if len(sys.argv) > 1 else cfg["out_dir"]
    log = case.Log(os.path.join(HERE, "t7_vs_20260930"))
    with open(os.path.join(folder, "calibration_result.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    ref_dir = cfg["reference_20260930"]["result_dir"]
    with open(os.path.join(ref_dir, "calibration_result.json"), encoding="utf-8") as fh:
        old = json.load(fh)
    cam = next(c for c in res["cameras"] if c["name"] == "cam1")
    s = cam.get("sigma") or {}
    new_g = geometry(cam["centre_mm"], cam["R"], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    rig = old["rig"]
    R_old, _ = cv2.Rodrigues(np.asarray(rig["rvec"], float))
    old_g = geometry(rig["camera_centre_mm"], R_old, rig["axis_point_at_Y0_mm"], rig["axis_direction"])
    f_new, f_old = float(cam["f_px"]), float(rig["f_px"])
    sf_new, sf_old = float(s.get("f_px") or float("nan")), float(old["sigma"]["f_px"])
    comb = math.hypot(sf_new, sf_old)
    log("T7: cam1 of today's wand calibration against the dot-board calibration of 30 Sep")
    log(f"  (only meaningful if cam1's lens and focus are unchanged; the camera may also have moved a little)")
    log(f"  f: wand {f_new:,.1f} +- {sf_new:,.1f} px; dot board {f_old:,.1f} +- {sf_old:,.1f} px "
        f"(in air: {cfg['reference_20260930']['f_air_px']:,} px)")
    log(f"     difference {f_new - f_old:+,.1f} px = {100 * (f_new / f_old - 1):+.3f} % "
        f"({(f_new - f_old) / comb:+.2f} combined sigma)")
    for k, unit in (("distance_from_axis_mm", "mm"), ("elevation_deg", "deg"), ("aim_deg", "deg"),
                    ("roll_deg", "deg")):
        log(f"  {k:<22s} wand {new_g[k]:+11.4f} {unit}   dot board {old_g[k]:+11.4f} {unit}   "
            f"difference {new_g[k] - old_g[k]:+.4f}")
    if s.get("distance_mm"):
        log(f"  sigma of the wand fit's distance from the axis: {s['distance_mm']:.3f} mm; the dot board's centre "
            f"sigma {old['sigma']['camera_centre_mm']} mm (board frame)")
    log.save({"f": {"wand": f_new, "wand_sigma": sf_new, "dot_board": f_old, "dot_board_sigma": sf_old,
                    "air": cfg["reference_20260930"]["f_air_px"]},
              "geometry": {"wand": new_g, "dot_board": old_g}})


if __name__ == "__main__":
    main()
