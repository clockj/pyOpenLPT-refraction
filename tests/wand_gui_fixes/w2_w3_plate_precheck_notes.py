"""W3 (B3): "Precalibrate to Check" with the flat-window model runs plate_wand's start search and coarse fit; and W2
(B2): the notes on axis points.  Offscreen, on the plate campaign's synthetic rig B as G8 (tests/plate_wand/prigs.py:
two perpendicular walls of 12.7 mm acrylic, two cameras behind each, 300 frames, 0.1 px noise, C++ detections),
put into the view as GUI wand points; the window panel: 2 windows, cameras 0 and 1 behind window 0, 2 and 3 behind
window 1, air / acrylic 1.49 / water (its preset, 1.333), 12.7 mm.

    python w2_w3_plate_precheck_notes.py

W3:
  - Precalibrate to Check runs plate_wand_gui.child_precheck in a separate process (ProcessCalibWorker), with no
    folder prompt; the error table lists every frame; the 3D View shows the two windows; a "Check Complete" box
    gives the RMS; the status line says so; nothing is written and nothing becomes a result (no camFiles to export);
  - the RMS is that of plate_wand_gui.precheck run in this process on the GUI's job (the same stages 1-3);
  - Stop 2 s into the check ends it within 2 s, with no box;
  - with "Use the previous calibrator" ticked, Precalibrate to Check runs the pinhole check (CalibrationWorker with
    precalibrate), as before.
W2:
  - no text in the calibration module tells to tick the previous calibrator to align axis points;
  - with axis points loaded: the flat-window check's box and the flat-window calibration's box say that they are
    not applied to the flat-window model; the tube's check box says it for the tube (rig: tests/cylinder_wand's
    reference rig of 3 cameras, 300 frames, 0.1 px);
  - the previous calibrator's result box says that axis points were not applied when they are loaded (its
    alignment needs C++ cameras that it never gets), and says nothing about them otherwise.
Writes w2_w3_plate_precheck_notes_out.txt and .json; the run's files go to data/w2_w3_run/.
"""
import common as C  # noqa: I001

import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(C.REPO / "tests" / "plate_wand"))
sys.path.insert(0, str(C.REPO / "tests" / "cylinder_wand"))
log = C.Log("w2_w3_plate_precheck_notes")
AXIS = {0: {"center": [600.0, 400.0], "+X": [700.0, 400.0], "+Y": [600.0, 300.0], "+Z": [650.0, 450.0]},
        1: {"center": [640.0, 410.0], "+X": [740.0, 410.0], "+Y": [640.0, 310.0], "+Z": [690.0, 460.0]}}


def gui_points(uv):
    wp = {}
    for i in range(uv.shape[0]):
        fr = {}
        for c in range(uv.shape[1]):
            if np.all(np.isfinite(uv[i, c])):
                a, b = uv[i, c]
                fr[c] = [[a[0], a[1], 20.0, 0.9, "Filtered_Small", 0], [b[0], b[1], 30.0, 0.9, "Filtered_Large", 1]]
        if len(fr) >= 2:
            wp[i] = fr
    return wp


def main():
    run_dir = C.DATA / "w2_w3_run"
    shutil.rmtree(run_dir, ignore_errors=True)
    (run_dir / "results").mkdir(parents=True)
    app = C.qt_app()  # noqa: F841
    dlg = C.Dialogs(folder=str(run_dir / "results"))
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QFileDialog
    prompts = []
    QFileDialog.getExistingDirectory = lambda *a, **k: (prompts.append(a[1] if len(a) > 1 else ""), dlg.folder)[1]
    import prigs
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from modules.camera_calibration import CameraCalibrationView
    from modules.camera_calibration.curved_calibration import plate_wand_gui as pg

    rig = prigs.rig_B()
    w = prigs.make_wand(rig, 300, np.random.default_rng(1308), noise_px=0.1)
    wp = gui_points(w.uv)
    view = CameraCalibrationView()
    captured = {}

    def wrap(name):
        orig = getattr(view, name)

        def f(payload):
            captured.setdefault(name, []).append(payload)
            return orig(payload)
        setattr(view, name, f)                       # the workers' signals are connected to these attributes
    for name in ("_on_plate_precheck_finished", "_on_plate_finished", "_on_cylinder_finished"):
        wrap(name)

    def plate_setup():
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
        view.media3_combo.setCurrentText("Water")
        view.media2_thick.setValue(12.7)

    def files_written():
        return sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file())

    plate_setup()
    log(f"Rig B: {len(wp)} frames; media {view.media1_index.value()}, {view.media2_index.value()}, "
        f"{view.media3_index.value()}, {view.media2_thick.value()} mm; legacy box {view.refr_legacy_check.isChecked()}")

    # ---------------- W3: the flat-window check
    log("W3: Precalibrate to Check with the flat-window model")
    n_msg, n_prompt = len(dlg.messages), len(prompts)
    t0 = time.time()
    view._run_wand_calibration(precalibrate=True)
    worker = view._cyl_worker
    C.wait_for(lambda: worker.isFinished() and view._cyl_dialog is None, 900)
    secs = time.time() - t0
    payload = (captured.get("_on_plate_precheck_finished") or [None])[-1]
    new = dlg.messages[n_msg:]
    log(f"  {secs:.0f} s; messages {[m[:2] for m in new]}")
    for m in new:
        log("    " + m[2].replace("\n", "\n    "))
    log.check("it ran plate_wand_gui.child_precheck in a separate process, with no folder",
              type(worker).__name__ == "ProcessCalibWorker" and worker.target is pg.child_precheck
              and worker.folder is None and len(prompts) == n_prompt,
              f"{type(worker).__name__}, target {getattr(worker.target, '__name__', worker.target)}, folder "
              f"{worker.folder}, folder prompts {len(prompts) - n_prompt}")
    planes = sum(isinstance(c, Poly3DCollection) for c in view.calib_3d_view.ax.collections)
    n_frames = len(payload["per_frame"]) if payload else -1
    log.check("the error table lists every frame and the 3D View shows the two windows",
              payload is not None and view.error_table.rowCount() == n_frames == len(wp) and planes == 2,
              f"{view.error_table.rowCount()} rows, {n_frames} frames in the payload, {len(wp)} frames; {planes} planes")
    box = [m for m in new if m[1] == "Check Complete"]
    rms_txt = f"RMS {payload['rms_px']:.3f} px" if payload else "?"
    log.check("a 'Check Complete' box with the RMS of stages 1-3; the status line says so",
              len(box) == 1 and "Start and coarse fit (stages 1-3)" in box[0][2] and rms_txt in box[0][2]
              and "Flat-window check done" in view.status_label.text(),
              f"{rms_txt}; status '{view.status_label.text()}'")
    log.check("nothing written, nothing a result (no camFiles to export)",
              not files_written() and not view._refr_has_result and getattr(view, "_plate_last", None) is None
              and not view.export_refractive_camfiles_to_dir(str(run_dir / "export")),
              f"files {files_written()}; has result {view._refr_has_result}")
    ref = pg.precheck(view._plate_job)
    log.check("the RMS is that of plate_wand_gui.precheck on the GUI's job (in this process)",
              payload is not None and ref.rms == payload["rms_px"],
              f"GUI {payload['rms_px'] if payload else None!r}, here {ref.rms!r}; stage-3 RMS "
              f"{ref.info['stage3_rms_px']:.6f} px")

    # Stop
    n_msg = len(dlg.messages)
    view._run_wand_calibration(precalibrate=True)
    worker = view._cyl_worker
    t_stop = {}
    QTimer.singleShot(2000, lambda: (t_stop.__setitem__("t", time.time()), view._stop_cylinder_calibration()))
    C.wait_for(lambda: worker.isFinished() and view._cyl_dialog is None, 600)
    lag = time.time() - t_stop.get("t", time.time())
    new = dlg.messages[n_msg:]
    log.check("Stop 2 s into the check ends it within 2 s, with no box",
              lag < 2.0 and "stopped" in view.status_label.text() and not new,
              f"{lag:.2f} s; status '{view.status_label.text()}'; messages {[m[:2] for m in new]}")

    # the previous calibrator: the pinhole check, as before
    view.refr_legacy_check.setChecked(True)
    before_cyl = view._cyl_worker
    n_msg = len(dlg.messages)
    view._run_wand_calibration(precalibrate=True)
    cw = getattr(view, "_calib_worker", None)
    started = type(cw).__name__ == "CalibrationWorker" and bool(getattr(cw, "precalibrate", False))
    if cw is not None:
        C.wait_for(lambda: cw.isFinished(), 600)
        C.wait_for(lambda: False, 1.0)
    log.check("with 'Use the previous calibrator', Precalibrate to Check runs the pinhole check (as before)",
              started and view._cyl_worker is before_cyl,
              f"{type(cw).__name__}, precalibrate {getattr(cw, 'precalibrate', None)}; "
              f"messages {[m[:2] for m in dlg.messages[n_msg:]]}")
    view.refr_legacy_check.setChecked(False)
    dlg.messages.append(("-", "-", "-"))
    plate_setup()

    # ---------------- W2: the notes on axis points
    log("\nW2: the notes on axis points")
    src = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                    for p in (C.REPO / "modules" / "camera_calibration").rglob("*.py"))
    told = re.findall(r"[Tt]ick 'Use the previous calibrator'[^\"]*", src)
    log.check("no text tells to tick the previous calibrator to align axis points", not told, f"{told[:2]}")

    view.axis_direction_map = dict(AXIS)
    note_plate = "Axis points are not applied to the flat-window model"
    n_msg = len(dlg.messages)
    view._run_wand_calibration(precalibrate=True)
    worker = view._cyl_worker
    C.wait_for(lambda: worker.isFinished() and view._cyl_dialog is None, 900)
    box = [m for m in dlg.messages[n_msg:] if m[1] == "Check Complete"]
    log.check("axis points loaded: the flat-window check's box says they are not applied",
              len(box) == 1 and note_plate in box[0][2], box[0][2].splitlines()[-1] if box else "no box")

    n_msg = len(dlg.messages)
    t0 = time.time()
    view._run_wand_calibration()
    worker = view._cyl_worker
    C.wait_for(lambda: worker.isFinished() and view._cyl_dialog is None, 1800)
    new = dlg.messages[n_msg:]
    text = "\n".join(m[2] for m in new)
    log(f"  the flat-window calibration: {time.time() - t0:.0f} s; messages {[m[:2] for m in new]}")
    log.check("axis points loaded: the flat-window calibration's box says they are not applied, and tells "
              "nothing to tick", bool(new) and note_plate in text and "previous calibrator" not in text,
              next((ln for ln in text.splitlines() if "Axis points" in ln), "no note"))

    # the tube's check
    import t13_common  # noqa: F401  (puts build-py/Release and the curved folder on sys.path)
    import rigs
    cams = rigs.reference_rig(3)
    wt = rigs.wand_frames(cams, rigs.CASE_TUBE, 300, 10.0, np.random.default_rng(1313), noise_px=0.1)
    view.wand_num_cams.setValue(3)
    for r in range(3):
        view.wand_cam_table.cellWidget(r, 3).setValue(cams[r].width)
        view.wand_cam_table.cellWidget(r, 4).setValue(cams[r].height)
    view.wand_calibrator.wand_points = gui_points(wt.uv)
    view.wand_calibrator.wand_points_filtered = None
    view.wand_model_combo.setCurrentText("Pinhole+Cylinder")
    view.wand_len_spin.setValue(10.0)
    view.cyl_od_spin.setValue(201.6125)
    view.cyl_wall_spin.setValue(3.175)
    n_msg = len(dlg.messages)
    view._run_wand_calibration(precalibrate=True)
    worker = view._cyl_worker
    C.wait_for(lambda: worker.isFinished() and view._cyl_dialog is None, 900)
    box = [m for m in dlg.messages[n_msg:] if m[1] == "Check Complete"]
    log.check("axis points loaded: the tube's check box says they are not applied to the tube model",
              len(box) == 1 and "Axis points are not applied to the cylinder model" in box[0][2],
              box[0][2].splitlines()[-1] if box else f"messages {[m[:2] for m in dlg.messages[n_msg:]]}")

    # the previous calibrator's result box
    def previous_box(axis):
        view.axis_direction_map = dict(AXIS) if axis else {}
        view._refr_cams_cpp = None
        n = len(dlg.messages)
        view._on_refractive_finished(True, {0: np.zeros(11), 1: np.zeros(11)}, {"window_planes": {}},
                                     {"per_frame_errors": {}})
        boxes = [m for m in dlg.messages[n:] if m[1] == "Success"]
        return boxes[-1][2] if boxes else ""
    with_axis, without = previous_box(True), previous_box(False)
    note_prev = "Axis points were not applied"
    log.check("the previous calibrator's box: says axis points were not applied when they are loaded, nothing "
              "otherwise", note_prev in with_axis and note_prev not in without and "Axis" not in without,
              next((ln for ln in with_axis.splitlines() if note_prev in ln), "no note"))
    view.close()
    return log.save({"files": files_written()})


if __name__ == "__main__":
    ok = main()
    sys.stdout.flush()
    os._exit(0 if ok else 1)
