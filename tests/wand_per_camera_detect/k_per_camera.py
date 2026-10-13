"""K: detection settings per camera in the wand tab (2026-10-13-plan.md, section 4).

    python k_per_camera.py

The GUI's main window (gui/app.py) on the native platform, placed off the screen (as tests/wand_gui_fixes W8), on
the six frames of run 20261006/T2 that W1 wrote (tests/wand_gui_fixes/data/w1/dark/: cameras 0 and 1, dark balls on
a bright background), with T2's settings (radius 40-120 px, opening 95 px, mask 50 px).
  - K1, nothing changes: Test Detect on frame 4069 in the fast, reliable and ellipse modes, without and with pixel
    sizes (20 and 13.5 um): with the box unticked (the shared settings, as before) and with it ticked, its rows left as
    the tick filled them.  The detections must be the same to the bit.
  - K2, each camera gets its own values (ellipse): camera 1's row set to 5-20 px, below its balls.
    - Test Detect: camera 1 finds neither of its balls (no point within 3 px of them), and camera 0 gives its points
      of K1 to the bit.  A row with R min above R max stops the detection with a warning.
    - Process All Frames on the six frames, through the GUI's own path (its dialogs answered): the written file has
      camera 0's balls as a run with the shared settings has them, and none of camera 1's.
  - K3, the panel: the table follows Num Cameras 2 -> 3 -> 2 (rows kept, the new one from the shared values); its
    columns follow the mode (Sens. not in the ellipse mode unless the custom axis is on; Opening and Mask only in
    it); the shared settings are greyed while the box is ticked; the Point Detection panel has no horizontal scroll
    bar at 1500 x 900 and 1200 x 700 with four cameras and the box ticked; a picture of the panel.
Writes k_per_camera_out.txt, k_per_camera.json and k_per_camera_panel_<mode>.png; data/ holds Process All's files.
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE / "data"
W1 = REPO / "tests" / "wand_gui_fixes" / "data" / "w1" / "dark"
FRAMES = (1000, 2000, 3000, 4069, 5000, 8000)
ROW = FRAMES.index(4069)
sys.path.insert(0, str(REPO / "tests" / "wand_gui_fixes"))      # its common: Dialogs, wait_for (paths, one BLAS thread)
import common as C  # noqa: E402,I001

LINES, CHECKS = [], []


def say(msg=""):
    print(msg, flush=True)
    LINES.append(str(msg))


def check(label, ok, detail=""):
    CHECKS.append({"check": label, "ok": bool(ok), "detail": detail})
    say(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))


def same(a, b):
    """Two Test Detect results ({camera: points or None}) identical to the last bit."""
    a = {k: v for k, v in (a or {}).items() if v is not None}
    b = {k: v for k, v in (b or {}).items() if v is not None}
    return set(a) == set(b) and all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a)


def counts(res):
    return {k: (0 if v is None else len(v)) for k, v in sorted((res or {}).items())}


def near(points, centres, tol=3.0):
    """How many of `centres` have a point of `points` within tol px."""
    if points is None or len(points) == 0 or centres is None:
        return 0
    P = np.asarray(points, float)[:, :2]
    return sum(bool(np.min(np.hypot(*(P - c).T)) <= tol) for c in np.asarray(centres, float)[:, :2])


def raw_rows(path):
    """{camera: [(frame, x, y, r), ...]} of the file's Raw rows."""
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["Status"] == "Raw":
                out.setdefault(int(r["Camera"]), []).append((int(r["Frame"]), float(r["X"]), float(r["Y"]),
                                                             float(r["Radius"])))
    return out


def main():
    t_start = time.time()
    DATA.mkdir(exist_ok=True)
    import openlpt  # noqa: F401  (the GUI's entry: the local pyopenlpt build first on the path)
    import app as guiapp
    guiapp.prepare_qtwebengine()
    from PySide6.QtCore import QEventLoop, QPoint, QRect, QTimer
    from PySide6.QtWidgets import QApplication, QTabWidget
    qapp = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    from modules.camera_calibration import CameraCalibrationView

    def settle(s=0.4):
        loop = QEventLoop()
        QTimer.singleShot(int(1000 * s), loop.quit)
        loop.exec()

    dialogs = C.Dialogs(answer="No")
    guiapp.OpenLPTMainWindow._check_for_updates = lambda self: None
    win = guiapp.OpenLPTMainWindow()
    win.resize(1500, 900)
    win.move(-9000, -9000)
    win.show()
    view = win.findChildren(CameraCalibrationView)[0]
    view.tabs.setCurrentIndex(1)                         # Wand Calibration
    ctrl = view.det_scroll.parentWidget()
    while ctrl is not None and not isinstance(ctrl, QTabWidget):
        ctrl = ctrl.parentWidget()
    ctrl.setCurrentIndex(0)                              # Point Detection
    settle(0.8)

    view.wand_num_cams.setValue(2)
    paths = {c: [str(W1 / f"cam{c + 1}" / f"frame{f:06d}.png") for f in FRAMES] for c in (0, 1)}
    assert all(os.path.exists(p) for v in paths.values() for p in v), "run tests/wand_gui_fixes/w1_bright_balls.py"
    view.wand_images = {c: list(v) for c, v in paths.items()}
    view.populate_wand_table()
    view._refresh_wand_radius_range_limit()
    view.wand_type_combo.setCurrentText("Dark on Bright")
    view.radius_range.setMinValue(40)
    view.radius_range.setMaxValue(120)
    view.ellipse_open_spin.setValue(95)
    view.ellipse_mask_spin.setValue(50.0)
    t = view.wand_percam_table
    say("K: detection settings per camera; T2 frames " + ", ".join(map(str, FRAMES)) + " (cameras 0 and 1); shared "
        f"settings radius {view.radius_range.value()}, sensitivity {view.sensitivity_slider.value()}, opening 95 px, "
        "mask 50 px")

    got = []
    orig = view._on_single_detection_finished

    def capture(res):
        got.append(res)
        orig(res)
    view._on_single_detection_finished = capture       # the worker's signal is connected to this attribute

    def test_detect(row=ROW):
        view.frame_table.setCurrentCell(row, 0)
        n0 = len(got)
        view._detect_single_frame()
        ok = C.wait_for(lambda: len(got) > n0, 300)    # (the GUI deletes the worker once it has finished)
        C.wait_for(lambda: False, 0.2)
        return got[-1] if ok and len(got) > n0 else None

    def set_pixels(pix):
        for c in (0, 1):
            view.wand_cam_table.cellWidget(c, 5).setValue(pix[c] if pix else 0.0)

    radios = {"fast": view.detect_mode_fast, "reliable": view.detect_mode_reliable,
              "ellipse": view.detect_mode_ellipse}

    # ---- K1
    say("\nK1: the box unticked (as before) and ticked with its rows as filled: the same detections")
    ref = {}
    for mode, radio in radios.items():
        radio.setChecked(True)
        for pix in (None, (20.0, 13.5)):
            set_pixels(pix)
            view.wand_percam_check.setChecked(False)
            a = test_detect()
            view.wand_percam_check.setChecked(True)            # fills the rows from the shared settings
            rows = view._wand_cam_settings()
            b = test_detect()
            view.wand_percam_check.setChecked(False)
            what = f"{mode}, " + (f"pixel sizes {pix[0]:g} and {pix[1]:g} um" if pix else "no pixel sizes")
            check(f"K1 {what}: ticking the box changes no detection", a is not None and same(a, b),
                  f"points per camera {counts(a)} and {counts(b)}; rows " + "; ".join(
                      f"cam {r}: r {s['min_radius']:g}-{s['max_radius']:g}, sens {s['sensitivity']:g}, opening "
                      f"{s['open_px']}, mask {s['rod_mask_px']:g}" for r, s in sorted(rows.items())))
            if pix is None:
                ref[mode] = a
    set_pixels(None)

    # ---- K2
    say("\nK2: each camera its own values (ellipse): camera 1's row 5-20 px, below its balls")
    radios["ellipse"].setChecked(True)
    view._percam_edited = False
    view.wand_percam_check.setChecked(True)
    t.cellWidget(1, 1).setValue(5.0)
    t.cellWidget(1, 2).setValue(20.0)
    rows = view._wand_cam_settings()
    check("K2 the rows: camera 0 the shared settings, camera 1 its own; editing a row marks the table edited",
          rows[0]["min_radius"] == 40 and rows[0]["max_radius"] == 120 and rows[1]["min_radius"] == 5
          and rows[1]["max_radius"] == 20 and view._percam_edited, json.dumps(rows))
    r2 = test_detect()
    balls0 = ref["ellipse"].get(0)
    balls1 = ref["ellipse"].get(1)
    check("K2 Test Detect: camera 1 finds neither of its balls, camera 0 its points of K1 to the bit",
          r2 is not None and near(r2.get(1), balls1) == 0 and np.array_equal(np.asarray(r2.get(0)),
                                                                           np.asarray(balls0)),
          f"camera 1: {counts(r2).get(1, 0)} points, {near(r2.get(1), balls1)} of its 2 balls; camera 0: "
          f"{counts(r2).get(0, 0)} points")
    t.cellWidget(1, 1).setValue(40.0)                          # R min above R max: refused
    n_msgs, n_got = len(dialogs.messages), len(got)
    view.frame_table.setCurrentCell(ROW, 0)
    view._detect_single_frame()
    C.wait_for(lambda: False, 1.0)
    msg = dialogs.messages[n_msgs:]
    check("K2 a row with R min above R max stops Test Detect with a warning",
          len(got) == n_got and any(m[0] == "warning" and "R min must be below R max" in m[2] for m in msg),
          f"{msg[:1]}")
    t.cellWidget(1, 1).setValue(5.0)

    def process_all(name):
        out = DATA / name
        if out.exists():
            out.unlink()
        dialogs.save_file = str(out)
        n_msgs = len(dialogs.messages)
        view._process_wand_frames()                            # modal until the detection has finished
        C.wait_for(lambda: False, 0.5)
        return out, dialogs.messages[n_msgs:]

    view.wand_percam_check.setChecked(False)
    f_shared, m_shared = process_all("k2_shared.csv")
    view.wand_percam_check.setChecked(True)
    f_own, m_own = process_all("k2_per_camera.csv")
    a, b = raw_rows(f_shared), raw_rows(f_own)
    balls1_all = {fr: [(x, y) for (g, x, y, _r) in a.get(1, []) if g == fr] for fr in range(len(FRAMES))}
    hits1 = sum(near(np.array([(x, y) for (g, x, y, _r) in b.get(1, []) if g == fr]).reshape(-1, 2),
                     np.array(balls1_all[fr]).reshape(-1, 2)) for fr in range(len(FRAMES)))
    check("K2 Process All Frames: camera 0's rows as with the shared settings, none of camera 1's balls",
          f_shared.exists() and f_own.exists() and sorted(a.get(0, [])) == sorted(b.get(0, []))
          and len(a.get(0, [])) > 0
          and sum(len(v) for v in balls1_all.values()) > 0 and hits1 == 0,
          f"shared: camera 0 {len(a.get(0, []))} and camera 1 {len(a.get(1, []))} detections; per camera: camera 0 "
          f"{len(b.get(0, []))}, camera 1 {len(b.get(1, []))}, of them on camera 1's balls {hits1}; messages "
          f"{[m[1] for m in m_own][:3]}")

    # ---- K3
    say("\nK3: the panel")
    t.cellWidget(0, 1).setValue(33.0)
    view.wand_num_cams.setValue(3)
    rows3 = view._wand_cam_settings()
    shared2 = view._shared_percam_values(2)
    view.wand_num_cams.setValue(2)
    rows2 = view._wand_cam_settings()
    check("K3 the table follows Num Cameras 2 -> 3 -> 2: the rows kept, the new one from the shared settings",
          t.rowCount() == 2 and rows3[0]["min_radius"] == 33 and rows3[1]["max_radius"] == 20
          and [rows3[2][k] for k in ("min_radius", "max_radius", "sensitivity", "open_px", "rod_mask_px")] == shared2
          and rows2[0]["min_radius"] == 33 and rows2[1]["max_radius"] == 20, f"{rows3}")
    states = {}
    for mode in ("fast", "reliable", "ellipse"):
        radios[mode].setChecked(True)
        for axis in (False, True):
            (view.axis_mode_custom if axis else view.axis_mode_default).setChecked(True)
            states[(mode, axis)] = [not t.isColumnHidden(k) for k in (3, 4, 5)]
    view.axis_mode_default.setChecked(True)
    want = {(m, ax): [m != "ellipse" or ax, m == "ellipse", m == "ellipse"] for m in radios for ax in (False, True)}
    check("K3 the columns follow the mode: Sens. unless ellipse (shown with the custom axis), Opening and Mask in "
          "ellipse only", states == want, "; ".join(f"{m}{' + custom axis' if ax else ''}: Sens. {s[0]}, Opening "
                                                    f"{s[1]}, Mask {s[2]}" for (m, ax), s in states.items()))
    grey = {}
    for on in (True, False):
        view.wand_percam_check.setChecked(on)
        for mode in ("reliable", "ellipse"):
            radios[mode].setChecked(True)
            grey[(on, mode)] = [w.isEnabled() for w in (view.radius_range, view.sensitivity_slider,
                                                        view.ellipse_open_spin, view.ellipse_mask_spin)]
    want_g = {(True, "reliable"): [False] * 4, (True, "ellipse"): [False] * 4,
              (False, "reliable"): [True] * 4, (False, "ellipse"): [True, False, True, True]}
    check("K3 the shared settings greyed while the box is ticked (the Sensitivity also in the ellipse mode)",
          grey == want_g, "; ".join(f"box {'on' if on else 'off'}, {m}: radius, sensitivity, opening, mask enabled "
                                    f"{v}" for (on, m), v in grey.items()))

    view.wand_num_cams.setValue(4)
    view.wand_percam_check.setChecked(True)
    for size in ((1500, 900), (1200, 700)):
        win.resize(*size)
        for mode in ("reliable", "ellipse"):
            radios[mode].setChecked(True)
            settle(0.6)
            sc = view.det_scroll
            hb = sc.horizontalScrollBar()
            check(f"K3 {size[0]} x {size[1]}, 4 cameras, box ticked, {mode}: no horizontal scroll bar",
                  not hb.isVisible(), f"viewport {sc.viewport().width()} px, content minimum "
                                      f"{sc.widget().minimumSizeHint().width()} px, table "
                                      f"{t.width()} px wide")
    win.resize(1500, 1300)
    for mode in ("reliable", "ellipse"):
        radios[mode].setChecked(True)
        settle(0.8)
        view.det_scroll.ensureWidgetVisible(view.btn_detect_single)
        settle(0.6)
        pm = win.grab()
        part = view.det_scroll.viewport()
        r = QRect(part.mapTo(win, QPoint(0, 0)), part.size())
        dpr = pm.devicePixelRatio()
        pm = pm.copy(QRect(int(r.x() * dpr), int(r.y() * dpr), int(r.width() * dpr), int(r.height() * dpr)))
        out = HERE / f"k_per_camera_panel_{mode}.png"
        pm.save(str(out))
        say(f"  picture: {out.name} ({pm.width()} x {pm.height()})")

    n_fail = sum(not c["ok"] for c in CHECKS)
    say(f"\n{len(CHECKS) - n_fail} of {len(CHECKS)} checks passed; {time.time() - t_start:.0f} s")
    (HERE / "k_per_camera_out.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    (HERE / "k_per_camera.json").write_text(json.dumps({"checks": CHECKS}, indent=1, default=str), encoding="utf-8")
    sys.stdout.flush()
    os._exit(0 if n_fail == 0 else 1)                    # skip Qt's teardown (the full window can fault at shutdown)


if __name__ == "__main__":
    main()
