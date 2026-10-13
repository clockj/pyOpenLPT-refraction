"""R: the detection record, for the results page and for reproduction (2026-10-13-plan-record.md, section 4).

    python r_record.py

The GUI's main window (gui/app.py) on the native platform, placed off the screen, on 24 frames of run 20261006/T2
(frames 500 to 12,000, every 500; cameras 0 and 1; copied once from X: into data/frames/ as the 8-bit images the
detector reads: the pairing needs 10 detections per camera, more than six frames give), ellipse mode, shared
settings T2's (radius 40-120 px, opening 95 px, mask 50 px).  Its dialogs are answered by their titles.
  - R1: Process All Frames writes <points>_detection.json beside the points, equal to what the detection got:
    each camera's own settings (camera 1: 35-125 px, opening 90 px, mask 48 px); then the shared settings scaled by
    pixel sizes (20 and 13.5 um: camera 1's values times 20/13.5 to the last bit, with its factor).
  - R2: Process All Frames / Resume on R1's file, camera 1's R max changed to 130 px: the record keeps R1's run and
    adds this one (resumed); Export Data writes the record beside the exported copy too.
  - R3: Load Wand Points reads the record back (the mode and every run); a CSV without one stays unknown, as before.
  - R4: the tube calibration's Job carries the record (captured before its worker starts: no fit is run; its check
    of the frame count is skipped, 24 frames being too few for a fit); the flat-window Job's settings carry it too.
  - R5: the results page's settings rows (report_from_cylinder on G3's saved result, report_from_plate on G8's,
    report_from_pinhole on W9's rig, each with R2's record; G3's without one as before), and the cylinder page in
    QtWebEngine: no console error; a picture of its settings table.
Writes r_record_out.txt, r_record.json and r_record_page_settings.png; data/ holds the points and records.
"""
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE / "data"
FRAMES = tuple(range(500, 12001, 500))      # 24 frames of T2: the pairing needs 10 detections per camera or more
IMGS = DATA / "frames"                       # their 8-bit copies, as the detector reads them (as W1 writes them)
sys.path.insert(0, str(REPO / "tests" / "wand_gui_fixes"))      # its common (T2 on X:, wait_for) and W9's rig
import common as C  # noqa: E402,I001

LINES, CHECKS = [], []


def say(msg=""):
    print(msg, flush=True)
    LINES.append(str(msg))


def check(label, ok, detail=""):
    CHECKS.append({"check": label, "ok": bool(ok), "detail": detail})
    say(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))


class Answers:
    """The GUI's dialogs, answered: questions by their title (default No), save dialogs from a queue of paths."""

    def __init__(self):
        self.messages, self.save, self.question = [], [], {}
        self.open = self.folder = None
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        def box(kind):
            def f(*a, **k):
                title = str(a[1]) if len(a) > 1 else ""
                self.messages.append((kind, title, str(a[2]) if len(a) > 2 else ""))
                if kind == "question":
                    return getattr(QMessageBox.StandardButton, self.question.get(title, "No"))
                return QMessageBox.StandardButton.Ok
            return f
        QMessageBox.information = box("info")
        QMessageBox.warning = box("warning")
        QMessageBox.critical = box("critical")
        QMessageBox.question = box("question")
        QFileDialog.getSaveFileName = lambda *a, **k: (self.save.pop(0) if self.save else "", "CSV Files (*.csv)")
        QFileDialog.getOpenFileName = lambda *a, **k: (self.open or "", "CSV Files (*.csv)")
        QFileDialog.getExistingDirectory = lambda *a, **k: self.folder or ""


def n_pairs(path):
    """The frames paired into wand points in a points file: its frames with Filtered_Small rows (one per camera)."""
    with open(path, newline="") as fh:
        return len({r["Frame"] for r in csv.DictReader(fh) if r["Status"] == "Filtered_Small"})


def main():
    t_start = time.time()
    DATA.mkdir(exist_ok=True)
    for p in DATA.glob("r[0-9]*"):
        if p.is_file():
            p.unlink()
    import openlpt  # noqa: F401
    import app as guiapp
    guiapp.prepare_qtwebengine()
    from PySide6.QtCore import QEventLoop, QPoint, QRect, QTimer, QUrl
    from PySide6.QtWidgets import QApplication, QTabWidget
    qapp = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    from modules.camera_calibration import CameraCalibrationView
    import modules.camera_calibration.view as V
    from modules.camera_calibration.curved_calibration import cylinder_wand_gui as cg
    from modules.camera_calibration.curved_calibration import plate_wand_gui as pg
    from modules.camera_calibration.results_page import page as rp
    from modules.camera_calibration.wand_calibration.wand_calibrator import detection_record_path

    def settle(s=0.4):
        loop = QEventLoop()
        QTimer.singleShot(int(1000 * s), loop.quit)
        loop.exec()

    ans = Answers()
    guiapp.OpenLPTMainWindow._check_for_updates = lambda self: None
    win = guiapp.OpenLPTMainWindow()
    win.resize(1500, 900)
    win.move(-9000, -9000)
    win.show()
    view = win.findChildren(CameraCalibrationView)[0]
    view.tabs.setCurrentIndex(1)
    ctrl = view.det_scroll.parentWidget()
    while ctrl is not None and not isinstance(ctrl, QTabWidget):
        ctrl = ctrl.parentWidget()
    ctrl.setCurrentIndex(0)
    settle(0.8)

    import cv2
    paths = {}
    for c, srcs in C.t2_frames(FRAMES).items():                # X: -> data/frames/cam1, cam2 (once)
        (IMGS / f"cam{c + 1}").mkdir(parents=True, exist_ok=True)
        paths[c] = []
        for f, src in zip(FRAMES, srcs):
            out = IMGS / f"cam{c + 1}" / f"frame{f:06d}.png"
            if not out.exists():
                img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
                if img.dtype != np.uint8:
                    img = (img / img.max() * 255).astype(np.uint8)
                cv2.imwrite(str(out), img)
            paths[c].append(str(out))
    view.wand_num_cams.setValue(2)
    view.wand_images = {c: list(v) for c, v in paths.items()}
    view.populate_wand_table()
    view._refresh_wand_radius_range_limit()
    view.wand_type_combo.setCurrentText("Dark on Bright")
    view.radius_range.setMinValue(40)
    view.radius_range.setMaxValue(120)
    view.ellipse_open_spin.setValue(95)
    view.ellipse_mask_spin.setValue(50.0)
    view.detect_mode_ellipse.setChecked(True)
    t = view.wand_percam_table
    say(f"R: the detection record; {len(FRAMES)} T2 frames, {FRAMES[0]} to {FRAMES[-1]} every 500 (cameras 0 and 1), "
        "ellipse mode")

    def process_all(csv_name, export=None, resume=False):
        out = DATA / csv_name
        ans.save = [str(out)] + ([str(DATA / export)] if export else [])
        ans.question = {"Resume?": "Yes" if resume else "No", "Export Data": "Yes" if export else "No"}
        n0 = len(ans.messages)
        view._process_wand_frames()                            # modal until the detection has finished
        C.wait_for(lambda: False, 0.5)
        return out, ans.messages[n0:]

    def record_of(csv_path):
        p = detection_record_path(csv_path)
        return json.loads(Path(p).read_text(encoding="utf-8")) if os.path.exists(p) else None

    def values(run):
        return {c["cam_id"]: {k: c.get(k) for k in ("min_radius", "max_radius", "sensitivity", "open_px",
                                                    "rod_mask_px")} for c in run["cameras"]}

    # ---- R1
    say("\nR1: Process All Frames writes the record beside the points")
    view._percam_edited = False
    view.wand_percam_check.setChecked(True)
    t.cellWidget(1, 1).setValue(35.0)
    t.cellWidget(1, 2).setValue(125.0)
    t.cellWidget(1, 4).setValue(90)
    t.cellWidget(1, 5).setValue(48.0)
    want = view._wand_cam_settings()
    f1, m1 = process_all("r1_points.csv")
    rec1 = record_of(f1)
    run = (rec1 or {"runs": [{}]})["runs"][-1]
    imgs = {c["cam_id"]: c.get("images") for c in run.get("cameras", [])}
    ok_imgs = all(imgs.get(c) == {"folder": str(IMGS / f"cam{c + 1}"), "first": "frame000500.png",
                                  "last": "frame012000.png", "count": len(FRAMES)} for c in (0, 1))
    check("R1 each camera's own settings: the record beside the points holds them as the detection got them",
          rec1 is not None and len(rec1["runs"]) == 1 and run.get("mode") == "ellipse"
          and run.get("wand_type") == "dark" and run.get("per_camera") is True and run.get("resumed") is False
          and values(run) == {c: {k: want[c][k] for k in want[c]} for c in want} and ok_imgs
          and view._wand_detect_record_used == {**rec1, "file": detection_record_path(f1)},
          f"{detection_record_path(f1)}: {json.dumps(values(run))}; images {'as loaded' if ok_imgs else imgs}")
    say(f"    points: {n_pairs(f1)} frames paired; messages {[m[1] for m in m1]}")

    view.wand_percam_check.setChecked(False)
    for c, p in ((0, 20.0), (1, 13.5)):
        view.wand_cam_table.cellWidget(c, 5).setValue(p)
    fb, _mb = process_all("r1b_points.csv")
    recb = record_of(fb)
    runb = (recb or {"runs": [{}]})["runs"][-1]
    s = 20.0 / 13.5
    cb = {c["cam_id"]: c for c in runb.get("cameras", [])}
    check("R1 shared settings with pixel sizes: camera 1's values are the shared ones times 20/13.5, to the bit",
          recb is not None and runb.get("per_camera") is False and cb[0]["min_radius"] == 40.0
          and cb[1]["min_radius"] == 40 * s and cb[1]["max_radius"] == 120 * s and cb[1]["open_px"] == round(95 * s)
          and cb[1]["rod_mask_px"] == 50.0 * s and cb[1]["scale"] == s and cb[1]["pixel_um"] == 13.5,
          f"camera 1: {json.dumps({k: cb.get(1, {}).get(k) for k in ('min_radius', 'max_radius', 'open_px', 'rod_mask_px', 'scale', 'pixel_um')})}")
    for c in (0, 1):
        view.wand_cam_table.cellWidget(c, 5).setValue(0.0)

    # ---- R2
    say("\nR2: a resume adds its run; Export Data writes the record beside the copy")
    view.wand_percam_check.setChecked(True)
    t.cellWidget(1, 2).setValue(130.0)
    f2, m2 = process_all("r1_points.csv", export="r2_export.csv", resume=True)
    rec2 = record_of(f1)
    runs = (rec2 or {}).get("runs", [])
    check("R2 the resume keeps R1's run and adds its own (resumed, camera 1's R max 130 px)",
          len(runs) == 2 and runs[0] == rec1["runs"][0] and runs[1].get("resumed") is True
          and values(runs[1])[1]["max_radius"] == 130.0 and values(runs[1])[0] == values(runs[0])[0],
          f"{len(runs)} runs; questions {[m[1] for m in m2 if m[0] == 'question']}")
    rexp = record_of(DATA / "r2_export.csv")
    check("R2 Export Data: the exported copy has the record beside it, the same runs",
          (DATA / "r2_export.csv").exists() and rexp is not None and rexp["runs"] == runs
          and rexp.get("points") == "r2_export.csv", f"{detection_record_path(DATA / 'r2_export.csv')}")

    # ---- R3
    say("\nR3: Load Wand Points reads the record back")
    view._wand_detect_mode_used = view._wand_detect_record_used = None
    ans.open = str(f1)
    view._load_wand_points_for_calibration()
    got = view._wand_detect_record_used
    check("R3 a file with a record: its mode and every run come back",
          view._wand_detect_mode_used == "ellipse" and got is not None and got["runs"] == runs
          and got["file"] == detection_record_path(f1), f"mode {view._wand_detect_mode_used}; "
                                                         f"{len((got or {}).get('runs', []))} runs")
    plain = DATA / "r3_no_record.csv"
    shutil.copyfile(f1, plain)
    ans.open = str(plain)
    view._load_wand_points_for_calibration()
    check("R3 a file without one: the mode stays unknown, as before",
          view._wand_detect_mode_used is None and view._wand_detect_record_used is None
          and view._wand_points_source == str(plain), f"mode {view._wand_detect_mode_used}")

    # ---- R4
    say("\nR4: the calibration's Job carries the record")
    ans.open = str(f1)
    view._load_wand_points_for_calibration()
    keep_check, keep_start = cg.Job.check, V.CylinderCalibWorker.start
    cg.Job.check = lambda self: None                       # 24 frames: too few for the fit's start (not run)
    V.CylinderCalibWorker.start = lambda self: None        # the worker is not started: no fit
    ans.folder = str(DATA / "r4_cylinder")
    view._run_cylinder_wand_calibration(10.0, 0)
    cg.Job.check, V.CylinderCalibWorker.start = keep_check, keep_start
    job = getattr(view, "_cyl_job", None)
    check("R4 the tube calibration's Job: detection = the record, detect_mode ellipse; in its settings",
          job is not None and job.detection == view._wand_detect_record_used and job.detect_mode == "ellipse"
          and job.settings()["detection"]["runs"] == runs, f"source {getattr(job, 'source', None)}")
    pjob = pg.Job(uv=np.zeros((1, 2, 2, 2)), frame_ids=[0], cam_ids=[0, 1], sizes=[(1280, 800)] * 2,
                  wand_length=10.0, windows=[{"name": "", "thickness_mm": [10.0], "n": [1.0, 1.49, 1.333]}],
                  cam_to_window=[0, 0], detection=view._wand_detect_record_used)
    check("R4 the flat-window Job's settings carry it too", pjob.settings()["detection"]["runs"] == runs)

    # ---- R5
    say("\nR5: the results page's settings rows")
    rec = view._wand_detect_record_used
    g3 = REPO / "tests" / "cylinder_wand_gui" / "data" / "g3_run" / "results"
    res = json.load(open(g3 / "calibration_result.json", encoding="utf-8"))
    data = json.load(open(g3 / "calibration_data.json", encoding="utf-8"))
    plain_rows = rp.report_from_cylinder(res, data, str(g3))["settings"]
    res.setdefault("gui", {}).setdefault("settings", {})["detection"] = rec
    rep = rp.report_from_cylinder(res, data, str(g3))
    rows = {k: v for k, v in rep["settings"]}
    for k, v in rep["settings"]:
        if k.startswith(("Detection", "Images")):
            say(f"    {k}: {v}")
    check("R5 cylinder: the Detection rows from the record (two runs with different settings, the last shown)",
          rows.get("Detection", "").startswith("ellipse mode, Dark on Bright; each camera's own settings; detected in "
                                               "2 runs (resumed), with different settings")
          and rows.get("Detection, cam1") == "radius 35–130 px; Rod opening 90 px; Rod mask 48 px"
          and rows.get("Detection, cam0") == "radius 40–120 px; Rod opening 95 px; Rod mask 50 px"
          and rows.get("Images, cam0", "").endswith(f": {len(FRAMES)} images, frame000500.png to frame012000.png")
          and rows.get("Detection record") == detection_record_path(f1), "")
    check("R5 cylinder without a record: the settings rows as before (one Detection row)",
          [k for k, _ in plain_rows if k.startswith(("Detection", "Images"))] == ["Detection"],
          f"{[r for r in plain_rows if r[0] == 'Detection']}")
    g8 = REPO / "tests" / "cylinder_wand_gui" / "data" / "g8_run" / "results"
    res8 = json.load(open(g8 / "calibration_result.json", encoding="utf-8"))
    res8.setdefault("gui", {}).setdefault("settings", {})["detection"] = rec
    rows8 = {k: v for k, v in rp.report_from_plate(res8, json.load(open(g8 / "calibration_data.json",
                                                                        encoding="utf-8")), str(g8))["settings"]}
    from w9_w10_pinhole import gui_points, rig
    final, uv, _a, _b, _on = rig(np.random.default_rng(9), [(1280, 800)] * 3)
    rows_p = {k: v for k, v in rp.report_from_pinhole(final, gui_points(uv), 10.0, detect_mode="ellipse",
                                                      detection=rec)["settings"]}
    check("R5 flat windows and pinhole: the same rows", all(r.get("Detection, cam1") == rows["Detection, cam1"]
                                                            and r.get("Detection record") == rows["Detection record"]
                                                            for r in (rows8, rows_p)))
    page = DATA / "r5_g3_page.html"
    rp.write_page(rep, str(page))
    from PySide6.QtWebEngineCore import QWebEnginePage
    from PySide6.QtWebEngineWidgets import QWebEngineView
    msgs = []

    class Page(QWebEnginePage):
        def javaScriptConsoleMessage(self, level, message, line, source):
            msgs.append((int(level.value) if hasattr(level, "value") else int(level), message))
    v = QWebEngineView()
    pgw = Page(v)
    v.setPage(pgw)
    v.resize(1200, 1100)
    v.move(-6000, -6000)
    v.show()
    done = {}
    v.loadFinished.connect(lambda ok: done.__setitem__("ok", ok))
    url = QUrl.fromLocalFile(str(page))
    url.setQuery("theme=dark")
    v.load(url)
    C.wait_for(lambda: "ok" in done, timeout_s=30)
    got_js = {}
    for _ in range(100):
        pgw.runJavaScript("window.__reportReady === true", 0, lambda r: got_js.__setitem__("r", r))
        C.wait_for(lambda: "r" in got_js, timeout_s=5)
        if got_js.get("r"):
            break
        got_js.clear()
        time.sleep(0.1)
    pgw.runJavaScript("document.getElementById('files').scrollIntoView(); true", 0, lambda r: None)
    settle(1.0)
    v.grab().save(str(HERE / "r_record_page_settings.png"))
    check("R5 the page with the rows draws in QtWebEngine without a console error",
          bool(done.get("ok")) and got_js.get("r") is True and not [m for m in msgs if m[0] >= 2],
          f"{len(msgs)} console messages")
    say("  picture: r_record_page_settings.png")

    n_fail = sum(not c["ok"] for c in CHECKS)
    say(f"\n{len(CHECKS) - n_fail} of {len(CHECKS)} checks passed; {time.time() - t_start:.0f} s")
    (HERE / "r_record_out.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    (HERE / "r_record.json").write_text(json.dumps({"checks": CHECKS, "record": rec}, indent=1, default=str),
                                        encoding="utf-8")
    sys.stdout.flush()
    os._exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
