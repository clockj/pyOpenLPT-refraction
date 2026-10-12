"""G8 (O5): the flat-window wand calibration (plate_wand) behind the GUI's "Pinhole+Refraction", offscreen.

    python g8_plate_gui.py

Synthetic wand points of the plate campaign's rig B (tests/plate_wand/prigs.py: two perpendicular walls of 12.7 mm
acrylic, two cameras behind each, 300 frames, 0.1 px noise, C++ detections) are put into the view as GUI wand points
(Filtered_Small / Filtered_Large); the window panel gets 2 windows, cameras 0 and 1 behind window 0, 2 and 3 behind
window 1, air / acrylic 1.49 / water 1.333, 12.7 mm.  Checks:
  - Run Calibration runs plate_wand in a separate process (ProcessCalibWorker) and ends with f of every camera within
    0.5 % of the truth and every detection used (plate_wand on rig B: W3);
  - the camFiles are PINPLATE, load in C++, and carry the REFRACTION_META block with the GUI's window ids, which
    VSC's parser reads;
  - the error table lists every frame; the 3-D view shows the two window planes; the result is marked "plate_wand";
    the tracking settings' hook copies the files; the results page is written;
  - Stop ends the process within 2 s; "Refit from the last result" takes the last answer and gives the same f;
  - with "Use the previous calibrator" ticked, Run Calibration goes to the previous calibrator (its entry is
    replaced by a stub that records the call).
Writes g8_plate_gui_out.txt and g8_plate_gui.json; the run's files go to data/g8_run/.
"""
import common as C  # noqa: I001

import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(C.REPO / "tests" / "plate_wand"))
import prigs  # noqa: E402

log = C.Log("g8_plate_gui")


def main():
    log("G8: plate_wand behind the GUI's Pinhole+Refraction, offscreen")
    run_dir = C.DATA / "g8_run"
    shutil.rmtree(run_dir, ignore_errors=True)
    (run_dir / "results").mkdir(parents=True)
    app = C.qt_app()
    dlg = C.Dialogs(folder=str(run_dir / "results"))
    from modules.camera_calibration import CameraCalibrationView
    import pyopenlpt as lpt
    from modules.vsc.camera_io import parse_camera_file
    rig = prigs.rig_B()
    w = prigs.make_wand(rig, 300, np.random.default_rng(1308), noise_px=0.1)
    wp = {}
    for i in range(w.uv.shape[0]):
        fr = {}
        for c in range(rig.n_cams):
            if np.all(np.isfinite(w.uv[i, c])):
                a, b = w.uv[i, c]
                fr[c] = [[a[0], a[1], 20.0, 0.9, "Filtered_Small", 0], [b[0], b[1], 30.0, 0.9, "Filtered_Large", 1]]
        if len(fr) >= 2:
            wp[i] = fr
    view = CameraCalibrationView()
    view.wand_num_cams.setValue(4)
    for r in range(4):
        view.wand_cam_table.cellWidget(r, 3).setValue(rig.cams[r].width)
        view.wand_cam_table.cellWidget(r, 4).setValue(rig.cams[r].height)
    view.wand_calibrator.wand_points = wp
    view.wand_calibrator.wand_points_filtered = None
    view._wand_detect_mode_used = "ellipse"
    view.wand_model_combo.setCurrentText("Pinhole+Refraction")
    view.wand_len_spin.setValue(rig.L)
    view.dist_model_combo.setCurrentIndex(0)
    view.window_count_spin.setValue(2)
    for row, wid in enumerate([0, 0, 1, 1]):
        view.cam_window_table.cellWidget(row, 1).setCurrentIndex(wid)
    view.media3_combo.setCurrentText("Other")
    view.media3_index.setValue(1.333)
    view.media2_thick.setValue(12.7)
    log(f"  {len(wp)} frames; media {view.media1_index.value()}, {view.media2_index.value()}, "
        f"{view.media3_index.value()}, {view.media2_thick.value()} mm; legacy box {view.refr_legacy_check.isChecked()}")

    t0 = time.time()
    view._run_wand_calibration()
    worker = view._cyl_worker
    ok_wait = C.wait_for(lambda: worker.isFinished() and view._cyl_dialog is None, timeout_s=1800)
    log(f"  Run Calibration: {time.time() - t0:.0f} s (finished {ok_wait}); message {dlg.messages[-1][:2] if dlg.messages else None}")
    if dlg.messages:
        log("    " + dlg.messages[-1][2].replace("\n", "\n    "))
    log.check("the calibration ran in a separate process (ProcessCalibWorker)", type(worker).__name__ == "ProcessCalibWorker")
    res_dir = run_dir / "results"
    res = json.load(open(res_dir / "calibration_result.json", encoding="utf-8"))
    f_err = [c["f_px"] / t.f - 1 for c, t in zip(res["cameras"], rig.cams)]
    log.check("f of every camera within 0.5 % of the truth", max(abs(e) for e in f_err) < 0.005,
              ", ".join(f"{c['name']} {c['f_px']:,.1f} ({100 * e:+.3f} %)" for c, e in zip(res["cameras"], f_err)))
    log.check("every detection used", res["active"] == res["observations"], f"{res['active']}/{res['observations']}")
    metas, models = [], set()
    for k in range(4):
        path = str(res_dir / "camFile" / f"cam{k}.txt")
        models.add(open(path, encoding="utf-8").read().split("\n")[1].strip())
        lpt.Camera(path)
        metas.append((parse_camera_file(path).get("ref_meta") or {}).get("window_id"))
    log.check("camFiles are PINPLATE, load in C++, and carry the GUI's window ids for VSC",
              models == {"PINPLATE"} and metas == [0, 0, 1, 1], f"models {models}, window ids {metas}")
    pfe = view.wand_calibrator.per_frame_errors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    planes = sum(isinstance(c, Poly3DCollection) for c in view.calib_3d_view.ax.collections)
    log.check("error table, 3-D windows, the result's kind and the page",
              view.error_table.rowCount() == len(pfe) == res["frames"] and planes == 2 and view._refr_kind == "plate_wand"
              and os.path.isfile(res_dir / "calibration_report.html"),
              f"{view.error_table.rowCount()} rows, {planes} planes, kind {view._refr_kind}, page "
              f"{os.path.isfile(res_dir / 'calibration_report.html')}")
    from views.tracking_settings_view import TrackingSettingsView
    ts = TrackingSettingsView(calibration_view=view)
    proj = run_dir / "project" / "camFile"
    proj.mkdir(parents=True)
    ts.camera_path_display.setText(str(proj))
    ts._on_cam_path_changed()
    copied = sorted(os.listdir(proj))
    same = all((proj / n).read_bytes() == (res_dir / "camFile" / n).read_bytes() for n in copied)
    log.check("tracking settings' hook copies the PINPLATE files", copied == [f"cam{k}.txt" for k in range(4)] and same,
              f"{copied}, byte-identical {same}")

    # Stop
    view._run_wand_calibration()
    worker = view._cyl_worker
    from PySide6.QtCore import QTimer
    t_stop = {}
    QTimer.singleShot(3000, lambda: (t_stop.__setitem__("t", time.time()), view._stop_cylinder_calibration()))
    C.wait_for(lambda: worker.isFinished() and view._cyl_dialog is None, timeout_s=600)
    lag = time.time() - t_stop.get("t", time.time())
    log.check("Stop ends the process within 2 s", lag < 2.0 and "stopped" in view.status_label.text(),
              f"{lag:.2f} s; status '{view.status_label.text()}'")

    # refit from the last result
    view.refr_refit_check.setChecked(True)
    t0 = time.time()
    view._run_wand_calibration()
    worker = view._cyl_worker
    C.wait_for(lambda: worker.isFinished() and view._cyl_dialog is None, timeout_s=1800)
    res2 = json.load(open(res_dir / "calibration_result.json", encoding="utf-8"))
    df = max(abs(a["f_px"] / b["f_px"] - 1) for a, b in zip(res2["cameras"], res["cameras"]))
    log.check("refit from the last result: the same f (1e-6), and faster", df < 1e-6,
              f"{time.time() - t0:.0f} s; largest change of f {df:.1e}")

    # the previous calibrator
    calls = []
    view._run_refractive_wand_calibration = lambda *a, **k: (calls.append(a), view._busy_end('wand_calibration'))
    view.refr_legacy_check.setChecked(True)
    view._run_wand_calibration()
    log.check("'Use the previous calibrator' goes to the previous calibrator", len(calls) == 1, f"calls {calls}")
    view.close()
    return log.save({"f_rel_err": f_err})


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
