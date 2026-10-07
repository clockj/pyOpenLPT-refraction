"""F1, a finding (not a fix): the Pinhole model with wand points whose cameras are not numbered 0..N-1 without gaps.

    python f1_pinhole_ids.py

The pinhole calibrator builds cam_id_map = {camera id: internal index} (wand_calibrator.py, calibrate_wand), but
_parse_results reads it the other way, cam_id_map[internal index].  That holds only while the ids are 0..N-1.  Here
W10's synthetic rig (three cameras of 1280 x 800 px, f 2,000 px, 300 frames, 0.1 px) gets the ids 0, 2 and 3 (the
camera table has four rows; camera 1 has no points), and Run Calibration with the model "Pinhole" is pressed.
Recorded: the GUI's message.  For comparison the same points with the ids 0, 1, 2 (W10) calibrate.
Writes f1_pinhole_ids_out.txt and .json.
"""
import common as C  # noqa: I001

import shutil  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

from w9_w10_pinhole import EQUAL, gui_points, rig  # noqa: E402

log = C.Log("f1_pinhole_ids")


def main():
    run_dir = C.DATA / "f1_run"
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    app = C.qt_app()  # noqa: F841
    dlg = C.Dialogs(folder=str(run_dir))
    from PySide6.QtWidgets import QMessageBox
    QMessageBox.exec = lambda box: 0                        # the result box: "Not Save"
    from modules.camera_calibration import CameraCalibrationView
    _true, uv, _A, _B, _on = rig(np.random.default_rng(1314), EQUAL)
    out = {}
    for label, ids in (("ids 0, 1, 2", (0, 1, 2)), ("ids 0, 2, 3", (0, 2, 3))):
        view = CameraCalibrationView()
        view.wand_num_cams.setValue(max(ids) + 1)
        for r in range(max(ids) + 1):
            view.wand_cam_table.cellWidget(r, 2).setValue(2000)
        wp = {f: {ids[c]: pts for c, pts in fr.items()} for f, fr in gui_points(uv).items()}
        view.wand_calibrator.wand_points = wp
        view.wand_calibrator.wand_points_filtered = None
        view.wand_model_combo.setCurrentText("Pinhole")
        view.wand_len_spin.setValue(10.0)
        view.dist_model_combo.setCurrentIndex(0)
        n = len(dlg.messages)
        view._run_wand_calibration()
        w = view._calib_worker
        C.wait_for(lambda: w.isFinished() and getattr(view, "_calib_dialog", None) is not None
                   and not view._calib_dialog.isVisible(), 900)
        C.wait_for(lambda: False, 1.0)
        msgs = dlg.messages[n:]
        params = sorted(view.wand_calibrator.final_params)
        out[label] = {"messages": [list(m) for m in msgs], "final_params_cameras": params,
                      "status": view.status_label.text()}
        log(f"{label}: status '{view.status_label.text()}'; messages {[m[:2] for m in msgs]}; final_params cameras "
            f"{params}")
        for m in msgs:
            log("    " + m[2].replace("\n", "\n    ")[:400])
        view.close()
    ok = "Calibration Successful" in out["ids 0, 1, 2"]["status"]
    log.check("ids 0, 1, 2 (W10's rig) calibrate", ok, out["ids 0, 1, 2"]["status"])
    log(f"ids 0, 2, 3: {'FAILS' if 'Successful' not in out['ids 0, 2, 3']['status'] else 'calibrates'} "
        f"('{out['ids 0, 2, 3']['status']}')")
    return log.save({"runs": out})


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
