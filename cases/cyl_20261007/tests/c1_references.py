"""C1, references (plan: 2026-10-07-plan.md, Phase C; the 6 Oct T2 case's t7_references.py, adapted): this run's
cameras against the earlier calibrations of the same rig.

    python tests/c1_references.py [RESULT_FOLDER]      (default: config out_dir, C1's result)

  - cam1 against the dot-board calibration of 30 Sep (meaningful only if cam1's lens and focus are unchanged since):
    f against 41,459 +- 69 px (through the tube) and 41,610 px (in air, 5 Oct);
  - cam1 and cam2 against the delivered wand calibrations of 6 Oct (T0 and T2).  On 6 Oct every fit (full, halves,
    quarters of both runs) and the 30 Sep board lay on one line f = k D through the origin per camera (k about
    38.2 px/mm for cam1, 34.5 for cam2: the near-telecentric valley), so f / D is compared as well as f: the same
    k with a different f means the same magnification with the valley's f ambiguity; a different k means the
    magnification (focus, or the camera's distance) changed.
Per camera the quantities that do not depend on either calibration's frame: f (px); the distance from the tube's
axis; the elevation of the line of sight (its angle to the plane perpendicular to the axis); the aim (seen along the
axis, the angle between the line of sight and the line from the camera to the axis); the roll (the angle of the
image rows to the plane perpendicular to the axis).
Writes c1_references_out.txt and c1_references.json beside this script.
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

KEYS = (("distance_from_axis_mm", "mm"), ("elevation_deg", "deg"), ("aim_deg", "deg"), ("roll_deg", "deg"))
OCT6 = {"6 Oct T0": r"X:\Shijie Zhong\Refraction\Processed\20261006\T0\Results",
        "6 Oct T2": None}                                  # T2's folder from config.reference_T2


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


def wand_geometry(cam):
    return geometry(cam["centre_mm"], cam["R"], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])


def jload(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    cfg = case.load_config()
    folder = sys.argv[1] if len(sys.argv) > 1 else cfg["out_dir"]
    log = case.Log(os.path.join(HERE, "c1_references"))
    res = jload(os.path.join(folder, "calibration_result.json"))
    old = jload(os.path.join(cfg["reference_20260930"]["result_dir"], "calibration_result.json"))
    refs = {"6 Oct T0": jload(os.path.join(OCT6["6 Oct T0"], "calibration_result.json")),
            "6 Oct T2": jload(os.path.join(cfg["reference_T2"]["result_dir"], "calibration_result.json"))}
    out = {"vs_20260930": {}, "vs_6oct": {}, "f_over_D": {}}

    cam = next(c for c in res["cameras"] if c["name"] == "cam1")
    s = cam.get("sigma") or {}
    new_g = wand_geometry(cam)
    rig = old["rig"]
    R_old, _ = cv2.Rodrigues(np.asarray(rig["rvec"], float))
    old_g = geometry(rig["camera_centre_mm"], R_old, rig["axis_point_at_Y0_mm"], rig["axis_direction"])
    f_new, f_old = float(cam["f_px"]), float(rig["f_px"])
    sf_new, sf_old = float(s.get("f_px") or float("nan")), float(old["sigma"]["f_px"])
    log(f"C1, references: this run's cameras ({folder}) against the earlier calibrations")
    log("\ncam1 against the dot board of 30 Sep (only meaningful if cam1's lens and focus are unchanged)")
    log(f"  f: 7 Oct wand {f_new:,.1f} +- {sf_new:,.1f} px; dot board {f_old:,.1f} +- {sf_old:,.1f} px "
        f"(in air: {cfg['reference_20260930']['f_air_px']:,} px); difference {f_new - f_old:+,.1f} px = "
        f"{100 * (f_new / f_old - 1):+.3f} % ({(f_new - f_old) / math.hypot(sf_new, sf_old):+.2f} combined sigma)")
    log(f"  f / distance: 7 Oct {f_new / new_g['distance_from_axis_mm']:.4f}, dot board "
        f"{f_old / old_g['distance_from_axis_mm']:.4f} px/mm")
    for k, unit in KEYS:
        log(f"  {k:<22s} 7 Oct {new_g[k]:+11.4f} {unit}   dot board {old_g[k]:+11.4f} {unit}   "
            f"difference {new_g[k] - old_g[k]:+.4f}")
    out["vs_20260930"] = {"f": {"wand": f_new, "wand_sigma": sf_new, "dot_board": f_old, "dot_board_sigma": sf_old,
                                "air": cfg["reference_20260930"]["f_air_px"]},
                          "geometry": {"wand": new_g, "dot_board": old_g},
                          "f_over_D": {"wand": f_new / new_g["distance_from_axis_mm"],
                                       "dot_board": f_old / old_g["distance_from_axis_mm"]}}

    for ref_name, ref in refs.items():
        log(f"\nBoth cameras against {ref_name}'s delivered calibration")
        out["vs_6oct"][ref_name] = {}
        for c2 in res["cameras"]:
            c0 = next(c for c in ref["cameras"] if c["name"] == c2["name"])
            g2, g0 = wand_geometry(c2), wand_geometry(c0)
            s2, s0 = c2.get("sigma") or {}, c0.get("sigma") or {}
            f2, f0 = float(c2["f_px"]), float(c0["f_px"])
            sf2, sf0 = float(s2.get("f_px") or float("nan")), float(s0.get("f_px") or float("nan"))
            k2, k0 = f2 / g2["distance_from_axis_mm"], f0 / g0["distance_from_axis_mm"]
            log(f"  {c2['name']}: f 7 Oct {f2:,.1f} +- {sf2:,.1f} px, {ref_name} {f0:,.1f} +- {sf0:,.1f} px; difference "
                f"{f2 - f0:+,.1f} px = {100 * (f2 / f0 - 1):+.3f} % ({(f2 - f0) / math.hypot(sf2, sf0):+.2f} combined "
                f"sigma); f / distance 7 Oct {k2:.4f}, {ref_name} {k0:.4f} px/mm ({100 * (k2 / k0 - 1):+.3f} %)")
            for k, unit in KEYS:
                log(f"      {k:<22s} 7 Oct {g2[k]:+11.4f} {unit}   {ref_name} {g0[k]:+11.4f} {unit}   "
                    f"difference {g2[k] - g0[k]:+.4f}")
            out["vs_6oct"][ref_name][c2["name"]] = {"f": {"7oct": f2, "7oct_sigma": sf2, "ref": f0, "ref_sigma": sf0},
                                                    "f_over_D": {"7oct": k2, "ref": k0},
                                                    "geometry": {"7oct": g2, "ref": g0}}
    log.save(out)


if __name__ == "__main__":
    main()
