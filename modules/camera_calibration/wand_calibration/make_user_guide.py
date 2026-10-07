"""Build the wand calibration user guide: WAND_CALIBRATION_USER_GUIDE.html and its GUI screenshots (guide_*.png).

    python make_user_guide.py html
        WAND_CALIBRATION_USER_GUIDE.html from WAND_CALIBRATION_USER_GUIDE.md (edit the .md), with pandoc and
        user_guide_template.html (the GUI's dark style).  Needs pandoc on the PATH, or --pandoc PATH.
    python make_user_guide.py shots [--raw DIR] [--frames N] [--csv FILE] [--frame N] [--out DIR]
        The screenshots, from the real GUI.

shots runs the GUI's main window (gui/app.py, its own dark style) on the native platform, placed off the screen
(QtWebEngine, which draws the Report tab, cannot run offscreen), and answers its dialogs itself:
  1. Point Detection: 2 cameras with the image folders of a recording (default: run 20261006/T2, X:, frames
     frame000000.tif ...; the file lists are built from that pattern, as the folder's Load button would list them,
     instead of listing 12,716 files on the network share), Dark on Bright, the "ellipse" mode, Radius Range
     40-120 px, Rod opening 95 px, Rod mask 50 px (that run's settings), Test Detect on frame --frame.
  2. Calibration: the run's wand_points.csv loaded with "Load Wand Points (from CSV)" (its points were detected by the
     case's scripts with the GUI's ellipse detector, so the detection mode is set to "ellipse"), the model
     "Pinhole+Cylinder" with the tube's defaults (that run's tube), wand 10 mm; Run Calibration into --out; then the 3D
     View, the error table with a frame's overlay, and the Report tab.
  3. The Tube Settings and Refraction Settings panels (for the picture: 4 cameras behind 2 windows).
Writes guide_*.png beside this script.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MD = HERE / "WAND_CALIBRATION_USER_GUIDE.md"
HTML = HERE / "WAND_CALIBRATION_USER_GUIDE.html"
TEMPLATE = HERE / "user_guide_template.html"
T2 = r"X:\Shijie Zhong\Refraction"


def build_html(pandoc):
    cmd = [pandoc, "-f", "gfm", "-t", "html5", "-s", "--template", str(TEMPLATE),
           "--metadata", "pagetitle=Wand Calibration User Guide", "-o", str(HTML), str(MD)]
    subprocess.run(cmd, check=True, cwd=str(HERE))
    print("written", HTML)


def wait_for(predicate, timeout_s=900.0, step_ms=100):
    """Run the Qt event loop until predicate() is True or the time is up."""
    from PySide6.QtCore import QEventLoop, QTimer
    loop, t0 = QEventLoop(), time.time()

    def poll():
        if predicate() or time.time() - t0 > timeout_s:
            loop.quit()
    tm = QTimer()
    tm.timeout.connect(poll)
    tm.start(step_ms)
    loop.exec()
    tm.stop()
    return bool(predicate())


def answer_dialogs(state):
    """The GUI's message boxes are recorded and answered (questions: No); its file dialogs return state's paths."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    def box(kind):
        def f(*a, **k):
            state["messages"].append((kind, str(a[1]) if len(a) > 1 else "", str(a[2]) if len(a) > 2 else ""))
            return QMessageBox.StandardButton.No if kind == "question" else QMessageBox.StandardButton.Ok
        return f
    for kind in ("information", "warning", "critical", "question"):
        setattr(QMessageBox, kind, box(kind))
    QFileDialog.getExistingDirectory = lambda *a, **k: state["folder"]
    QFileDialog.getOpenFileName = lambda *a, **k: (state["open"], "CSV Files (*.csv)")


def grab(win, path, web=None, part=None):
    """Save the window's image, or the part of it that the widget `part` covers (cut from the window's image, so it
    looks as on the screen: a widget grabbed alone loses the window's style).  A QtWebEngine view inside is painted
    over it (a parent's grab leaves it blank)."""
    from PySide6.QtCore import QPoint, QRect
    from PySide6.QtGui import QPainter
    pm = win.grab()
    if web is not None and web.isVisible():
        p = QPainter(pm)
        p.drawPixmap(web.mapTo(win, QPoint(0, 0)), web.grab())
        p.end()
    if part is not None:
        r = QRect(part.mapTo(win, QPoint(0, 0)), part.size())
        dpr = pm.devicePixelRatio()
        pm = pm.copy(QRect(int(r.x() * dpr), int(r.y() * dpr), int(r.width() * dpr), int(r.height() * dpr)))
    pm.save(str(path))
    print("written", path, f"{pm.width()} x {pm.height()}")


def tab_widget_of(w):
    """The QTabWidget whose page holds the widget w."""
    from PySide6.QtWidgets import QTabWidget
    p = w.parentWidget()
    while p is not None and not isinstance(p, QTabWidget):
        p = p.parentWidget()
    return p


def shots(args):
    sys.path[:0] = [str(REPO / "gui"), str(REPO)]
    import openlpt  # noqa: F401  (the GUI's entry: the local pyopenlpt build first on the path)
    import app as guiapp
    guiapp.prepare_qtwebengine()                  # as the GUI's main() does, before the application exists
    from PySide6.QtWidgets import QApplication, QScrollArea
    qapp = QApplication.instance() or QApplication(sys.argv)
    qapp.setStyle("Fusion")
    from modules.camera_calibration import CameraCalibrationView

    state = {"messages": [], "folder": str(args.out), "open": str(args.csv)}
    answer_dialogs(state)
    guiapp.OpenLPTMainWindow._check_for_updates = lambda self: None
    win = guiapp.OpenLPTMainWindow()
    win.resize(1600, 960)
    win.move(-9000, -9000)                       # native window, off the screen
    win.show()
    qapp.processEvents()
    view = win.findChildren(CameraCalibrationView)[0]
    view.tabs.setCurrentIndex(1)                 # Wand Calibration
    ctrl = tab_widget_of(view.det_scroll)        # its sub-tabs Point Detection / Calibration / Tutorial

    # 1. Point Detection
    view.wand_num_cams.setValue(2)
    raw = Path(args.raw)
    for c in range(2):
        files = [str(raw / f"cam{c + 1}" / f"frame{f:06d}.tif") for f in range(args.frames)]
        view.wand_images[c] = files
        view.wand_cam_table.cellWidget(c, 0).setText(f"{len(files)}")      # as the Load button shows it
        view._update_wand_cam_size_from_first_image(c, files)
    view._refresh_wand_radius_range_limit()
    view.populate_wand_table()
    view.wand_type_combo.setCurrentText("Dark on Bright")
    view.detect_mode_ellipse.setChecked(True)
    view.radius_range.setMinValue(40)
    view.radius_range.setMaxValue(120)
    view.ellipse_open_spin.setValue(95)
    view.ellipse_mask_spin.setValue(50.0)
    ctrl.setCurrentIndex(0)
    view.frame_table.setCurrentCell(args.frame, 0)
    view.frame_table.scrollToItem(view.frame_table.item(args.frame, 0))
    wait_for(lambda: False, 1.5)
    view._detect_single_frame()
    wait_for(lambda: "Found" in view.status_label.text(), 120)
    view.vis_tabs.setCurrentIndex(0)
    wait_for(lambda: False, 1.0)
    grab(win, HERE / "guide_detection.png")
    print("  status:", view.status_label.text())

    # 2. Calibration with the tube model
    ctrl.setCurrentIndex(1)
    view._load_wand_points_for_calibration()
    view._wand_detect_mode_used = "ellipse"      # the file's points came from the ellipse detector (the case's scripts)
    view.wand_model_combo.setCurrentText("Pinhole+Cylinder")
    view.wand_len_spin.setValue(10.0)
    view.dist_model_combo.setCurrentIndex(0)
    t0 = time.time()
    view._run_wand_calibration()
    wait_for(lambda: getattr(view, "_cyl_worker", None) is not None and view._cyl_worker.isFinished()
             and getattr(view, "_cyl_dialog", None) is None, 1800)
    print(f"  calibration: {time.time() - t0:.0f} s; messages: {[m[:2] for m in state['messages']][-3:]}")
    ctrl.setCurrentIndex(1)
    view.vis_tabs.setCurrentWidget(view.calib_3d_view)
    calib_scroll = next(s for s in view.findChildren(QScrollArea) if s.widget() and s.widget().isAncestorOf(
        view.cylinder_group))
    calib_scroll.verticalScrollBar().setValue(0)
    wait_for(lambda: False, 1.5)
    grab(win, HERE / "guide_calibration.png")
    calib_scroll.ensureWidgetVisible(view.cylinder_group)
    wait_for(lambda: False, 0.5)
    grab(win, HERE / "guide_tube_settings.png", part=view.cylinder_group)
    view._display_frame_with_overlay(args.frame, target_cam=0)
    ctrl.setCurrentIndex(1)
    calib_scroll.ensureWidgetVisible(view.error_table)
    calib_scroll.horizontalScrollBar().setValue(0)
    wait_for(lambda: False, 1.5)
    grab(win, HERE / "guide_error_table.png")

    tab = view.report_tab
    view.vis_tabs.setCurrentWidget(tab)
    ready = {"v": None}
    for _ in range(120):
        tab.web.page().runJavaScript("window.__reportReady === true", 0, lambda v: ready.__setitem__("v", v))
        wait_for(lambda: ready["v"] is not None, 2)
        if ready["v"]:
            break
        ready["v"] = None
    wait_for(lambda: False, 2.0)
    grab(win, HERE / "guide_report.png", web=tab.web)

    # 3. The flat-window panel (for the picture: 4 cameras behind 2 windows)
    view.wand_model_combo.setCurrentText("Pinhole+Refraction")
    view.wand_num_cams.setValue(4)
    view.window_count_spin.setValue(2)
    for c in (2, 3):
        view.cam_window_table.cellWidget(c, 1).setCurrentIndex(1)
    win.resize(1600, 1300)                       # room for the whole panel
    ctrl.setCurrentIndex(1)
    wait_for(lambda: False, 1.0)
    calib_scroll.ensureWidgetVisible(view.refraction_group)
    wait_for(lambda: False, 1.0)
    grab(win, HERE / "guide_refraction_settings.png", part=view.refraction_group)
    print("  messages:", [m[:2] for m in state["messages"]])
    win.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["html", "shots"])
    ap.add_argument("--pandoc", default="pandoc")
    ap.add_argument("--raw", default=T2 + r"\Raw\20261006\T2")
    ap.add_argument("--frames", type=int, default=12716)
    ap.add_argument("--csv", default=T2 + r"\Processed\20261006\T2\Results\wand_points.csv")
    ap.add_argument("--frame", type=int, default=4069)
    ap.add_argument("--out", default=str(Path(os.environ.get("TEMP", ".")) / "wand_calibration"))
    args = ap.parse_args()
    if args.what == "html":
        build_html(args.pandoc)
    else:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        shots(args)
        sys.stdout.flush()
        os._exit(0)                              # skip Qt's teardown (the full window can fault at shutdown)


if __name__ == "__main__":
    main()
