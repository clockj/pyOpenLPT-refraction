"""G5 (Stop and bad input of the cylinder model) and G9 (O4: the pinhole calibration's results page), offscreen.

    python g5_g9_stop_input_pinhole.py

G5:
  - Stop pressed 3 s into a cylinder calibration (the start search) ends it within 6 s; the GUI comes back idle
    (buttons enabled, no dialog) and no camFile is written;
  - bad input gives a message and no run: one camera; fewer than 12 frames seen whole by every camera; a camera
    without a row in the camera table (its image size missing; the table's own spin boxes cannot be emptied); a wall
    thicker than the tube's radius; k1+k2 requested (the model fits k1 at most).
G9 (O4), the pinhole page:
  - three pinhole cameras (OpenCV model, 1280 x 800, f 2,000 px, k1 0.05) around a 100 mm volume, 300 wand frames
    (10 mm), 0.1 px noise; the true cameras stand in for a calibration's final_params (the page reports what it is
    given; the calibrators are tested elsewhere);
  - _show_pinhole_report writes the page and shows it in the Report tab; the page's triangulated wand-length error
    and reprojection RMS are those of 0.1 px noise (RMS below 0.2 px; wand error below 0.05 mm RMS).
Writes g5_g9_stop_input_pinhole_out.txt and .json.
"""
import common as C  # noqa: I001

import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

log = C.Log("g5_g9_stop_input_pinhole")


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
    app = C.qt_app()
    import rigs
    from modules.camera_calibration import CameraCalibrationView
    from PySide6.QtCore import QTimer
    run_dir = C.DATA / "g5_run"
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    dlg = C.Dialogs(folder=str(run_dir))
    view = CameraCalibrationView()
    cams = rigs.reference_rig(3)
    w = rigs.wand_frames(cams, rigs.CASE_TUBE, 300, 10.0, np.random.default_rng(1313), noise_px=0.1)

    def setup(n_cams, wp, sizes=True):
        view.wand_num_cams.setValue(n_cams)
        for r in range(n_cams):
            view.wand_cam_table.cellWidget(r, 3).setValue(cams[r].width if sizes else 1)
            view.wand_cam_table.cellWidget(r, 4).setValue(cams[r].height if sizes else 1)
        view.wand_calibrator.wand_points = wp
        view.wand_calibrator.wand_points_filtered = None
        view.wand_model_combo.setCurrentText("Pinhole+Cylinder")
        view.wand_len_spin.setValue(10.0)
        view.dist_model_combo.setCurrentIndex(0)
        view.cyl_od_spin.setValue(201.6125)
        view.cyl_wall_spin.setValue(3.175)

    # ---------------- G5: Stop
    log("G5: Stop and bad input (cylinder model)")
    setup(3, gui_points(w.uv))
    t = {}
    view._run_wand_calibration()
    worker = view._cyl_worker
    QTimer.singleShot(3000, lambda: (t.__setitem__("stop", time.time()), view._stop_cylinder_calibration()))
    C.wait_for(lambda: worker.isFinished() and view._cyl_dialog is None, timeout_s=300)
    lag = time.time() - t.get("stop", time.time())
    idle = view.btn_calibrate_wand.isEnabled() and view.btn_precalibrate.isEnabled() and view._cyl_dialog is None
    files = os.path.isdir(run_dir / "camFile") and os.listdir(run_dir / "camFile")
    log.check("Stop 3 s into the start search ends the run within 6 s; idle again; no camFile", lag < 6 and idle and not files,
              f"{lag:.1f} s after Stop; status '{view.status_label.text()}'; camFiles {files or 'none'}")

    # ---------------- G5: bad input
    def bad(label, n_cams, wp, expect, sizes=True, tweak=None):
        setup(n_cams, wp, sizes)
        if tweak:
            tweak()
        before = len(dlg.messages)
        prev = getattr(view, "_cyl_worker", None)
        view._run_wand_calibration()
        new = dlg.messages[before:]
        started = getattr(view, "_cyl_worker", None) is not prev
        ok = (not started) and any(m[0] == "warning" and re.search(expect, m[2]) for m in new)
        log.check(f"bad input, {label}: a message, no run", ok, (new[-1][2][:110] if new else "no message"))

    wp1 = {f: {0: fr[0]} for f, fr in gui_points(w.uv).items() if 0 in fr}
    bad("one camera", 1, wp1, "at least two cameras")
    few = {f: fr for f, fr in list(gui_points(w.uv).items())[:8]}
    bad("8 frames", 3, few, "12")
    # the table's spin boxes cannot be emptied (1 px at least): a camera without a table row is the missing size
    bad("a camera without a row in the camera table", 2, gui_points(w.uv), "no row")
    bad("a wall thicker than the radius", 3, gui_points(w.uv), "not thinner",
        tweak=lambda: view.cyl_wall_spin.setValue(150.0))
    view.cyl_wall_spin.setValue(3.175)
    bad("k1+k2", 3, gui_points(w.uv), "k1 at most", tweak=lambda: (view.dist_model_combo.model().item(2).setEnabled(True),
                                                                   view.dist_model_combo.setCurrentIndex(2)))

    # ---------------- G9: the pinhole page
    log("\nG9: the pinhole calibration's results page")
    import cv2
    rng = np.random.default_rng(1314)
    W, H, f, k1 = 1280, 800, 2000.0, 0.05
    K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]])
    dist = np.array([k1, 0, 0, 0, 0])
    final, Cs = {}, []
    for k, az in enumerate((-30.0, 0.0, 30.0)):
        a = np.radians(az)
        Cc = np.array([600 * np.sin(a), 80.0, -600 * np.cos(a)])
        z = -Cc / np.linalg.norm(Cc)
        x = np.cross([0, 1, 0], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.vstack([x, y, z])
        final[k] = {"R": R, "T": (-R @ Cc).reshape(3, 1), "K": K, "dist": dist, "img_size": (H, W)}
        Cs.append(Cc)
    mid = rng.uniform(-40, 40, (300, 3))
    d = rng.normal(size=(300, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    A, B = mid - 5 * d, mid + 5 * d
    uv = np.full((300, 3, 2, 2), np.nan)
    for k, p in final.items():
        for e, X in enumerate((A, B)):
            q, _j = cv2.projectPoints(X, cv2.Rodrigues(p["R"])[0], p["T"], K, dist)
            uv[:, k, e] = q.reshape(-1, 2) + rng.normal(scale=0.1, size=(300, 2))
    view.wand_calibrator.final_params = final
    view.wand_calibrator.wand_points = gui_points(uv)
    view.wand_calibrator.wand_points_filtered = None
    view.wand_len_spin.setValue(10.0)
    view._wand_detect_mode_used = "fast"
    view._show_pinhole_report(output_dir=str(run_dir / "pinhole"))
    page = run_dir / "pinhole" / "calibration_report.html"
    txt = page.read_text(encoding="utf-8") if page.exists() else ""
    m = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', txt, re.S)
    rep = json.loads(m.group(1).replace("<\\/", "</")) if m else {}
    S = rep.get("summary", {})
    shown = view.report_tab.page_path == str(page)
    log.check("the page is written and shown in the Report tab", page.exists() and shown and rep.get("model") == "PINHOLE",
              f"{page} ({page.stat().st_size // 1024 if page.exists() else 0} kB); shown {shown}")
    log.check("its reprojection RMS and wand-length error are those of 0.1 px noise",
              S.get("rms_px", 9) < 0.2 and S.get("wand_rms_um", 1e9) < 50,
              f"RMS {S.get('rms_px', float('nan')):.3f} px; wand {S.get('wand_rms_um', float('nan')):.1f} um RMS over "
              f"{S.get('wand_frames')} frames; {len(rep.get('residuals', []))} residual panels")
    view.close()
    return log.save()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
