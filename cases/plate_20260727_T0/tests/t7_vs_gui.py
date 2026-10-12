"""T7: this calibration against the GUI's (calib, calib_2, calib_3), all judged the same way (evaluate.py: free
triangulation of every frame, reprojection error, wand length) on the same detections:
  * every frame of the circle finder's table (10,936 frames; a few gross detection errors in it, so medians and
    percentiles are the fair measures);
  * calib_3's own 5,563 frames (the GUI's filtered set: the frames calib_3 fits best).
The calibrations of this case: T2 (every frame, states/t2_all.pkl) and the same pipeline fitted on calib_3's 5,563
frames (states/t7_gui5563.pkl); a missing state is skipped.  Then the rigs side by side: f, distances and angles to the
walls, the walls' angle, the baselines.  The text is also written to the output folder as
comparison_with_calib_3.txt when that folder exists.

    python tests/t7_vs_gui.py
Writes t7_vs_gui_out.txt and t7_vs_gui.json beside this script.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402
import evaluate as ev  # noqa: E402

OURS = {"plate_wand (T2, every frame)": "t2_all", "plate_wand on calib_3's frames": "t7k_gui5563",
        "same, k1 held at 0": "t7_gui5563"}


def main():
    cfg = case.load_config()
    rc = case.rig_config(cfg)
    names = rc.names
    rigs = {}
    for g in ("calib", "calib_2", "calib_3"):
        rigs[f"GUI {g}"] = case.load_gui(cfg, g)
    for label, stn in OURS.items():
        p = os.path.join(HERE, "states", f"{stn}.pkl")
        if os.path.isfile(p):
            st = case.load_state(p)
            rigs[label] = (st["cams"], st["rig"])
    sets = {"every frame of the table": case.read_detections(cfg, "all"),
            "calib_3's 5,563 frames": case.read_detections(cfg, "gui_5563")}
    log = case.Tee(os.path.join(HERE, "t7_vs_gui_out.txt"))
    log("T7: the calibrations judged the same way (evaluate.py: free triangulation of every frame)")
    out = {"evaluations": {}, "rigs": {}}
    for sname, cd in sets.items():
        log(f"\n== {sname}: {cd.uv.shape[0]} frames")
        log("  calibration                       median px  p90 px  p99 px   per camera median px            "
            "|dL| median  p90     p99 mm   mean L mm")
        for label, (cams, rig) in rigs.items():
            e = ev.evaluate(cams, rig, cd.uv, rc.wand_length_mm, rc.sizes, frame_ids=cd.frame_ids)
            out["evaluations"].setdefault(sname, {})[label] = e
            w = e["wand_error_mm"]
            log(f"  {label:32s}  {e['euclidean_px']['median']:.4f}   {e['euclidean_px']['p90']:6.3f}  "
                f"{e['euclidean_px']['p99']:6.2f}   " + " ".join(f"{pc['euclidean_px']['median']:.3f}"
                                                         for pc in e["per_camera"])
                + f"      {w['abs']['median']:.4f}   {w['abs']['p90']:.4f}  {w['abs']['p99']:.4f}   "
                  f"{e['L'] + w['mean']:.4f}")
    log("\n== the rigs (frame-independent; f in mm with 13.5 um pixels)")
    for label, (cams, rig) in rigs.items():
        d = case.describe(cams, rig)
        out["rigs"][label] = d
        log(f"  {label}: walls {d['walls_angle_deg']:.3f} deg apart")
        for c, cd_ in enumerate(d["cameras"]):
            log(f"    {names[c]} ({rc.cameras[c]['photo']}): f {cd_['f_px']:7.0f} px ({cd_['f_px'] * 0.0135:6.1f} mm), "
                f"{cd_['distance_to_wall_mm']:7.1f} mm from its wall, {cd_['incidence_deg']:5.2f} deg to its normal")
        log("    baselines " + ", ".join(f"{p['cams'][0]}-{p['cams'][1]} {p['baseline_mm']:.1f} mm" for p in d["pairs"]))
    with open(os.path.join(HERE, "t7_vs_gui.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    log.close()
    if os.path.isdir(cfg["out_dir"]) and os.path.isfile(os.path.join(cfg["out_dir"], "calibration_result.json")):
        shutil.copyfile(os.path.join(HERE, "t7_vs_gui_out.txt"), os.path.join(cfg["out_dir"], "comparison_with_calib_3.txt"))
        print(f"copied to {os.path.join(cfg['out_dir'], 'comparison_with_calib_3.txt')}")


if __name__ == "__main__":
    main()
