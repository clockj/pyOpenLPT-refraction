"""G4b: pictures of the wand tab with run 20261008/T0's three cameras (plan: 2026-10-08-plan.md, test G4).

    python g4b_screens.py

G4 runs offscreen, where Qt has no fonts; these pictures come from the GUI's own main window placed off the screen
(as tests/wand_gui_fixes/w8_panel_widths.py): the Wand Calibration tab, three cameras loaded through their Load
buttons (the folder dialog answered), Pixel (um) 20 / 20 / 13.5, the ellipse mode with cams 1-2's settings
(Radius Range 20-70 px, Rod opening 51 px, Rod mask 28 px), then Test Detect on image 5000 (the frame list's row
2604 after the alignment).
  - g4b_detection_panel.png: the Point Detection panel (the camera table with the Pixel column, the status line);
  - g4b_cam3_test_detect.png: cam3's image tab after Test Detect (its two balls, found with the scaled settings);
  - g4b_cam1_test_detect.png: cam1's, for comparison.
Writes g4b_screens_out.txt beside itself.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
T0 = Path(os.environ.get("OPENLPT_T0_1008", r"X:\Shijie Zhong\Refraction\Raw\20261008\T0"))


def main():
    lines = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(str(msg))

    sys.path[:0] = [str(REPO / "gui"), str(REPO)]
    import openlpt  # noqa: F401
    import app as guiapp
    guiapp.prepare_qtwebengine()
    from PySide6.QtCore import QEventLoop, QPoint, QRect, QTimer
    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QTabWidget
    qapp = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    from modules.camera_calibration import CameraCalibrationView

    def settle(s=0.5):
        loop = QEventLoop()
        QTimer.singleShot(int(1000 * s), loop.quit)
        loop.exec()

    guiapp.OpenLPTMainWindow._check_for_updates = lambda self: None
    win = guiapp.OpenLPTMainWindow()
    win.move(-9000, -9000)
    win.resize(1700, 1080)
    win.show()
    view = win.findChildren(CameraCalibrationView)[0]
    view.tabs.setCurrentIndex(1)                                   # Wand Calibration
    p = view.det_scroll.parentWidget()
    while p is not None and not isinstance(p, QTabWidget):
        p = p.parentWidget()
    ctrl = p
    ctrl.setCurrentIndex(0)                                        # Point Detection
    msgs = []
    QMessageBox.information = lambda *a, **k: msgs.append(("info", str(a[1:3]))) or QMessageBox.StandardButton.Ok
    QMessageBox.warning = lambda *a, **k: msgs.append(("warning", str(a[1:3]))) or QMessageBox.StandardButton.Ok
    QMessageBox.critical = lambda *a, **k: msgs.append(("critical", str(a[1:3]))) or QMessageBox.StandardButton.Ok
    folder = {"f": None}
    QFileDialog.getExistingDirectory = lambda *a, **k: folder["f"]
    view.wand_num_cams.setValue(3)
    for k in range(3):
        folder["f"] = str(T0 / f"cam{k + 1}")
        view._load_wand_folder_for_cam(k)
        settle(0.3)
    t = view.wand_cam_table
    for k, px in enumerate((20.0, 20.0, 13.5)):
        t.cellWidget(k, 5).setValue(px)
    view.detect_mode_ellipse.setChecked(True)
    view.radius_range.setMaxValue(70)
    view.radius_range.setMinValue(20)
    view.ellipse_open_spin.setValue(51)
    view.ellipse_mask_spin.setValue(28.0)
    view.wand_type_combo.setCurrentIndex(0)
    row = 5000 - 2396
    view.frame_table.setCurrentCell(row, 0)
    view.frame_table.scrollToItem(view.frame_table.item(row, 0))
    settle(1.0)
    log(f"loaded: {[len(view.wand_images[k]) for k in range(3)]} images; status: {view.status_label.text()!r}")
    log(f"scales: {view._wand_cam_scales()}")

    def grab(widget, name):
        pm = win.grab()
        r = QRect(widget.mapTo(win, QPoint(0, 0)), widget.size())
        dpr = pm.devicePixelRatio()
        pm = pm.copy(QRect(int(r.x() * dpr), int(r.y() * dpr), int(r.width() * dpr), int(r.height() * dpr)))
        pm.save(str(HERE / name))
        log(f"  picture: {name} ({pm.width()} x {pm.height()})")

    view.det_scroll.verticalScrollBar().setValue(0)
    settle(0.8)
    grab(view.det_scroll.viewport(), "g4b_detection_panel.png")
    got = {}
    orig = view._on_single_detection_finished

    def keep(res):
        got["res"] = res
        orig(res)
    view._on_single_detection_finished = keep
    view._detect_single_frame()
    loop_t = 0.0
    while "res" not in got and loop_t < 300:
        settle(0.5)
        loop_t += 0.5
    res = got.get("res", {})
    log(f"Test Detect: balls per camera {[0 if res.get(k) is None else len(res.get(k)) for k in range(3)]}")
    settle(1.5)
    for k, name in ((2, "g4b_cam3_test_detect.png"), (0, "g4b_cam1_test_detect.png")):
        view.vis_tabs.setCurrentIndex(k)
        settle(1.0)
        grab(view.vis_tabs, name)
    log(f"dialogs: {msgs}")
    (HERE / "g4b_screens_out.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.flush()
    os._exit(0)                                         # skip Qt's teardown (the full window can fault at shutdown)


if __name__ == "__main__":
    main()
