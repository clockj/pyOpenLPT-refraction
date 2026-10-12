"""C1, a file check (plan: 2026-10-07-plan.md, Phase C; the 6 Oct T2 case's t0_gui_csv.py, copied): after
calibrate.py, <out_dir>/wand_points.csv loads in the GUI.

    python tests/c1_gui_csv.py

Loads the file with the GUI's own loader (WandCalibrator.load_wand_data_from_csv, "Load Wand Points (from CSV)") and
converts it as the GUI's cylinder calibration does (cylinder_wand_gui.uv_from_wand_points).  Checks: every frame
kept in both cameras is there, with cameras 0 and 1 (cam1, cam2) and the balls labelled Filtered_Small (A) and
Filtered_Large (B); the centres equal the selection's within the file's rounding: 4 decimals, so at most 5e-5 px (criterion 5.1e-5: the first run
used 5e-5 itself and failed at 5.00e-05 px, the rounding bound).
Writes c1_gui_csv_out.txt beside this script.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

import numpy as np  # noqa: E402


def main():
    cfg = case.load_config()
    log = case.Log(os.path.join(HERE, "c1_gui_csv"))
    _dets, kept, _reasons, _medians, frames = case.load_selection(cfg)
    lib = case.library()
    from modules.camera_calibration.wand_calibration.wand_calibrator import WandCalibrator
    wc = WandCalibrator()
    path = os.path.join(cfg["out_dir"], "wand_points.csv")
    ok, msg = wc.load_wand_data_from_csv(path)
    log(f"C1 (file check): {path}: the GUI's loader says: {ok}, {msg}")
    n = len(wc.wand_points)
    labels_ok = all(sorted(str(p[4]) for p in pts) == ["Filtered_Large", "Filtered_Small"]
                    for f in wc.wand_points for pts in wc.wand_points[f].values())
    uv, fids, cids = lib.cg.uv_from_wand_points(wc.wand_points)
    ref = case.uv_array(kept, frames, case.cam_ids(cfg))
    same_frames = list(fids) == list(frames)
    err = float(np.nanmax(np.abs(uv - ref))) if same_frames else float("nan")
    checks = [("every kept frame loaded", n == len(frames), f"{n} of {len(frames)}"),
              ("cameras 0 and 1", list(cids) == [0, 1], str(cids)),
              ("labels Filtered_Small / Filtered_Large in every camera-frame", labels_ok, ""),
              ("the same frames, in order", same_frames, ""),
              ("centres equal the selection's within 5.1e-5 px", same_frames and err <= 5.1e-5, f"largest {err:.2e} px")]
    for name, passed, detail in checks:
        log(f"  {'PASS' if passed else 'FAIL'}: {name} {detail}")
    log.save()


if __name__ == "__main__":
    main()
