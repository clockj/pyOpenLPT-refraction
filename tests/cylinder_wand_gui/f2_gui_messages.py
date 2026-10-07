"""F2 (fixes plan 2026-10-06-fixes-plan.md, section 3): the GUI's messages after a detection and when loading wand
points (fixes F-d, F-e), offscreen, with the dialogs answered by the test (common.Dialogs: every message recorded,
questions answered No).

    python tests/cylinder_wand_gui/f2_gui_messages.py

The detection's state is set as Process All Frames leaves it, from the CSVs of P4 (T2 frames 4000-5199, read with
the GUI's loader): the GUI's default ellipse settings (no frame paired) and the case's settings (159 paired); the
images are T2's real frames.  Then CameraCalibrationView's handlers run:
  a. _on_process_finished after the default-settings detection: a warning that no frame was paired, per camera the
     frames with two or more blobs / one / none (P4: cam1 8 / 557 / 635, cam2 0 / 835 / 365), the rod and the rod
     opening named; no export question; the status line says 0 of 1200;
  b. after the case-settings detection: the export question says 159 of 1200 frames; no warning;
  c. after a Stop: no message; the status line says the file is not paired yet and names Resume;
  d. after a failure with partial data: only the rescue question, naming the raw detections and Resume; no
     "Detection complete" question (before the fix it came after every failure);
  e. after a failure with no data: only the "Detection Failed" warning;
  f. Load Wand Points on P4's raw-only CSV: a warning with its raw count (1,408 detections in 1,032 frames), the two
     causes and what to do; no "Success";
  g. Load Wand Points on P3's CSV (the GUI's own detection, autosaved): "Success ... Loaded 159 valid frames", as
     before the fix.
Writes f2_gui_messages_out.txt and f2_gui_messages.json beside this script.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

RAW = r"X:\Shijie Zhong\Refraction\Raw\20261006\T2"
FIRST, LAST = 4000, 5199
CSV_DEFAULTS = os.path.join(HERE, "data", "p4_ellipse_defaults.csv")
CSV_CASE = os.path.join(HERE, "data", "p4_ellipse_case.csv")
CSV_P3 = os.path.join(HERE, "data", "p3_wand_points_gui.csv")

log = C.Log("f2_gui_messages")


class _Dialog:
    def close(self):
        pass


def main():
    log("F2: the GUI's messages after a detection and when loading wand points (offscreen)")
    app = C.qt_app()
    dlg = C.Dialogs(save_file=os.path.join(HERE, "data", "f2_export.csv"), answer="No")
    from PySide6.QtWidgets import QFileDialog
    from modules.camera_calibration import CameraCalibrationView
    from modules.camera_calibration.wand_calibration.wand_calibrator import WandCalibrator
    view = CameraCalibrationView()
    view.resize(1500, 900)
    view.show()
    app.processEvents()
    images = {k: [os.path.join(RAW, cam, f"frame{f:06d}.tif") for f in range(FIRST, LAST + 1)]
              for k, cam in enumerate(("cam1", "cam2"))}

    def state(csv_path, mode="ellipse"):
        """The calibrator as Process All Frames leaves it (detections read back from its autosave)."""
        view.wand_calibrator = WandCalibrator()
        if csv_path:
            ok, msg = view.wand_calibrator.load_wand_data_from_csv(csv_path)
            assert ok, msg
        view.wand_images = {k: list(v) for k, v in images.items()}
        view._wand_detect_mode_used = mode
        view._proc_dialog = _Dialog()
        dlg.messages.clear()

    def shown():
        return [(k, t, x) for k, t, x in dlg.messages]

    # a. the default settings: nothing paired
    state(CSV_DEFAULTS)
    view._on_process_finished(True, "Detection completed.")
    m = shown()
    log(f"\na. after the default-settings detection: {len(m)} message(s); status: {view.status_label.text()!r}")
    for k, t, x in m:
        log(f"   [{k}] {t}: {x}")
    warn = [x for k, t, x in m if k == "warning"]
    text = warn[0] if warn else ""
    log.check("a: one warning, no question", len(warn) == 1 and not any(k == "question" for k, _t, _x in m))
    log.check("a: it says no frame was paired, of 1200", "paired none" in text and "1200 frames" in text)
    log.check("a: per camera counts as P4 (cam1 8 / 557 / 635, cam2 0 / 835 / 365)",
              "Camera 1: two or more blobs in 8 frames, one blob in 557, none in 635" in text
              and "Camera 2: two or more blobs in 0 frames, one blob in 835, none in 365" in text)
    log.check("a: the rod and the Rod opening named, and the file's Raw rows",
              "rod probably joins" in text and "Rod opening" in text and "'Raw' rows" in text)
    log.check("a: the status line says 0 of 1200 frames paired", "0 of 1200 frames paired" in view.status_label.text())

    # b. the case's settings: 159 paired
    state(CSV_CASE)
    view._on_process_finished(True, "Detection completed.")
    m = shown()
    log(f"\nb. after the case-settings detection: {len(m)} message(s); status: {view.status_label.text()!r}")
    for k, t, x in m:
        log(f"   [{k}] {t}: {x}")
    log.check("b: only the export question, with 159 of 1200 frames",
              len(m) == 1 and m[0][0] == "question" and "159 of 1200 frames have both balls" in m[0][2])

    # c. Stop
    state(CSV_DEFAULTS)
    view._on_process_finished(False, "Stopped by user (Data Saved).")
    m = shown()
    log(f"\nc. after a Stop: {len(m)} message(s); status: {view.status_label.text()!r}")
    log.check("c: no message; the status line says the file is not paired yet and names Resume",
              not m and "not paired" in view.status_label.text() and "Resume" in view.status_label.text())

    # d. a failure with partial data
    state(CSV_DEFAULTS)
    view._on_process_finished(False, "Error: A process in the process pool was terminated abruptly while the "
                                     "future was running or pending.")
    m = shown()
    log(f"\nd. after a failure with partial data: {len(m)} message(s)")
    for k, t, x in m:
        log(f"   [{k}] {t}: {x}")
    log.check("d: only the rescue question, naming the raw detections and Resume; no 'Detection complete'",
              len(m) == 1 and m[0][1] == "Detection Failed" and "not yet paired" in m[0][2] and "Resume" in m[0][2]
              and not any("Detection complete" in x for _k, _t, x in m))

    # e. a failure with no data
    state(None)
    view._on_process_finished(False, "Error: x")
    m = shown()
    log(f"\ne. after a failure with no data: {len(m)} message(s): {m}")
    log.check("e: only the 'Detection Failed' warning", len(m) == 1 and m[0][0] == "warning"
              and m[0][1] == "Detection Failed")

    # f / g. Load Wand Points
    for tag, path in (("f", CSV_DEFAULTS), ("g", CSV_P3)):
        view.wand_calibrator = WandCalibrator()
        view.wand_images = {k: list(v) for k, v in images.items()}
        dlg.messages.clear()
        QFileDialog.getOpenFileName = lambda *a, _p=path, **k: (_p, "CSV Files (*.csv)")
        view._load_wand_points_for_calibration()
        m = shown()
        log(f"\n{tag}. Load Wand Points on {os.path.basename(path)}: {len(m)} message(s); status: "
            f"{view.status_label.text()!r}")
        for k, t, x in m:
            log(f"   [{k}] {t}: {x}")
        if tag == "f":
            log.check("f: a warning with the raw count, the causes and what to do; no 'Success'",
                      len(m) == 1 and m[0][0] == "warning" and "1408 raw detections in 1032 frames" in m[0][2]
                      and "Resume" in m[0][2] and "Test Detect" in m[0][2]
                      and not any(t == "Success" for _k, t, _x in m))
        else:
            log.check("g: 'Success ... Loaded 159 valid frames', as before the fix",
                      len(m) == 1 and m[0][1] == "Success" and "Loaded 159 valid frames." in m[0][2]
                      and len(view.wand_calibrator.wand_points) == 159)
    view.close()
    log.save()


if __name__ == "__main__":
    main()
