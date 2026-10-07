"""W9 (B9) and W10 (B10): the Pinhole model's camFiles and error table.  Offscreen.

    python w9_w10_pinhole.py

Synthetic rig (G9's, tests/cylinder_wand_gui/g5_g9_stop_input_pinhole.py, without lens distortion): three OpenCV
cameras at -30, 0 and 30 deg, 600 mm from a 100 mm volume, f 2,000 px; 300 wand frames of 10 mm, 0.1 px noise.

W10, through the GUI: the cameras all 1280 x 800 px; the camera table gets Focal 2,000; Run Calibration with the
model "Pinhole", distortion None; the result saved (Save, then the folder data/w9_run/equal/):
  - every frame's entry of the error table has its cameras' projections (proj_pts), and they give the table's error
    (the larger of the two balls' distances to their detections) to 1e-9 px;
  - showing a frame (a click on its row) draws the red crosses at those projections (cv2.drawMarker recorded), on
    blank images written for frames 0-4 (before the fix it drew none: the pinhole entries had no proj_pts, and the
    branch that would project them is skipped for frames that have an entry).
W9, cameras of two sizes (camera 1 1024 x 768 px, its principal point at its centre):
  - the GUI's pinhole calibration of this rig fails ("Initial guess is outside of provided bounds"): its stages 2
    and 4 bound every camera's principal point to the first camera's image centre (B11, found here, not fixed:
    recorded, not a check);
  - so the export is checked on the calibrator's own result of the true cameras: the camera table's sizes, the
    calibrator's image_size set as Run Calibration sets it (the first camera's), _parse_results with the true
    parameters, the true wand ends as points_3d; export_to_file then writes each camera's own Image Size (800,1280 /
    768,1024 / 800,1280); git HEAD's export_to_file (before_lib/) on the same result writes the first camera's in
    every file (for the record).
(The first version ran W9 through the GUI's calibration and stopped at its failure: w9_w10_pinhole_first_try_console.txt.)
Writes w9_w10_pinhole_out.txt and .json.
"""
import common as C  # noqa: I001

import importlib.util  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

log = C.Log("w9_w10_pinhole")
EQUAL = [(1280, 800)] * 3                                  # (width, height)
MIXED = [(1280, 800), (1024, 768), (1280, 800)]


def rig(rng, sizes, f=2000.0, n=300):
    import cv2
    final = {}
    for k, az in enumerate((-30.0, 0.0, 30.0)):
        W, H = sizes[k]
        K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
        a = np.radians(az)
        Cc = np.array([600 * np.sin(a), 80.0, -600 * np.cos(a)])
        z = -Cc / np.linalg.norm(Cc)
        x = np.cross([0, 1, 0], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.vstack([x, y, z])
        final[k] = {"R": R, "T": (-R @ Cc).reshape(3, 1), "K": K, "dist": np.zeros(5), "img_size": (H, W)}
    mid = rng.uniform(-40, 40, (n, 3))
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    A, B = mid - 5 * d, mid + 5 * d
    uv = np.full((n, 3, 2, 2), np.nan)
    for k, p in final.items():
        for e, X in enumerate((A, B)):
            q, _j = cv2.projectPoints(X, cv2.Rodrigues(p["R"])[0], p["T"], p["K"], p["dist"])
            uv[:, k, e] = q.reshape(-1, 2) + rng.normal(scale=0.1, size=(n, 2))
    on = all(np.all((uv[:, k, :, 0] >= 0) & (uv[:, k, :, 0] < sizes[k][0]) & (uv[:, k, :, 1] >= 0)
                    & (uv[:, k, :, 1] < sizes[k][1])) for k in range(3))
    return final, uv, A, B, on


def gui_points(uv):
    wp = {}
    for i in range(uv.shape[0]):
        fr = {}
        for c in range(uv.shape[1]):
            a, b = uv[i, c]
            fr[c] = [[a[0], a[1], 20.0, 0.9, "Filtered_Small", 0], [b[0], b[1], 30.0, 0.9, "Filtered_Large", 1]]
        wp[i] = fr
    return wp


def image_size_line(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("# Image Size"))
    return lines[i + 1].strip()


def setup(view, sizes, uv, folder):
    import cv2
    view.wand_num_cams.setValue(3)
    for k, (W, H) in enumerate(sizes):
        view.wand_cam_table.cellWidget(k, 2).setValue(2000)
        view.wand_cam_table.cellWidget(k, 3).setValue(W)
        view.wand_cam_table.cellWidget(k, 4).setValue(H)
    imgs = {}
    for k, (W, H) in enumerate(sizes):                      # blank images for frames 0-4 (the overlay draws on them)
        (folder / "images" / f"cam{k}").mkdir(parents=True)
        imgs[k] = [str(folder / "images" / f"cam{k}" / f"frame_{i:03d}.png") for i in range(uv.shape[0])]
        for i in range(5):
            cv2.imwrite(imgs[k][i], np.zeros((H, W), np.uint8))
    view.wand_images = imgs
    wc = view.wand_calibrator
    wc.wand_points = gui_points(uv)
    wc.wand_points_filtered = None
    wc.per_frame_errors = {}
    view.wand_model_combo.setCurrentText("Pinhole")
    view.wand_len_spin.setValue(10.0)
    view.dist_model_combo.setCurrentIndex(0)


def main():
    import cv2
    run_dir = C.DATA / "w9_run"
    shutil.rmtree(run_dir, ignore_errors=True)
    (run_dir / "equal").mkdir(parents=True)
    (run_dir / "mixed").mkdir(parents=True)
    app = C.qt_app()  # noqa: F841
    dlg = C.Dialogs(folder=str(run_dir / "equal"))
    from PySide6.QtWidgets import QMessageBox

    def click_save(box):                                    # the result box's "Save" (a custom QMessageBox)
        for b in box.buttons():
            if b.text() == "Save":
                b.click()
                break
        return 0
    QMessageBox.exec = click_save
    from modules.camera_calibration import CameraCalibrationView
    spec = importlib.util.spec_from_file_location("wand_calibrator_before", C.HERE / "before_lib" / "wand_calibrator.py")
    old = importlib.util.module_from_spec(spec)
    sys.modules["wand_calibrator_before"] = old
    spec.loader.exec_module(old)

    # ---------------- W10: equal sizes, through the GUI
    _true, uv, _A, _B, on = rig(np.random.default_rng(1314), EQUAL)
    log(f"W10: 3 cameras of 1280 x 800 px, {uv.shape[0]} frames (every detection on its sensor: {on})")
    view = CameraCalibrationView()
    setup(view, EQUAL, uv, run_dir / "equal")
    wc = view.wand_calibrator
    t0 = time.time()
    view._run_wand_calibration()
    worker = view._calib_worker
    C.wait_for(lambda: worker.isFinished() and any(m[1] in ("Saved", "Calibration Failed") for m in dlg.messages), 900)
    log(f"  calibration and save: {time.time() - t0:.0f} s; messages {[m[:2] for m in dlg.messages]}")
    f_fit = {k: float(p["K"][0, 0]) for k, p in wc.final_params.items()}
    log(f"  fitted f: {', '.join(f'cam{k} {v:,.1f} px' for k, v in f_fit.items())}")
    cam_dir = run_dir / "equal" / "camFile"
    equal_lines = {k: image_size_line(cam_dir / f"cam{k}.txt") for k in range(3)} if cam_dir.is_dir() else {}
    log(f"  camFiles' Image Size: {equal_lines}")

    pfe = wc.per_frame_errors
    wand = wc.wand_points_filtered or wc.wand_points
    worst, missing = 0.0, 0
    for fid, err in pfe.items():
        pp = err.get("proj_pts") or {}
        for c, e in err["cam_errors"].items():
            if c not in pp:
                missing += 1
                continue
            pa, pb = (np.asarray(v, float) for v in pp[c])
            oa, ob = (np.asarray(v[:2], float) for v in wand[fid][c][:2])
            worst = max(worst, abs(max(np.linalg.norm(pa - oa), np.linalg.norm(pb - ob)) - e))
    log.check("every frame's entry has its cameras' projections, which give the table's errors (1e-9 px)",
              bool(pfe) and len(pfe) == len(wand) and missing == 0 and worst < 1e-9
              and view.error_table.rowCount() == len(pfe),
              f"{len(pfe)} frames, {view.error_table.rowCount()} rows, {missing} missing, largest mismatch {worst:.1e} px")

    marks = []
    orig_marker = cv2.drawMarker

    def record(img, pos, *a, **k):
        marks.append(tuple(pos))
        return orig_marker(img, pos, *a, **k)
    cv2.drawMarker = record
    ok_frames = 0
    for fid in range(5):
        n0 = len(marks)
        view._display_frame_with_overlay(fid, target_cam=0)
        want_pts = sorted((int(p[0]), int(p[1])) for c in range(3) for p in pfe[fid]["proj_pts"][c]) if pfe else []
        ok_frames += bool(want_pts) and sorted(marks[n0:]) == want_pts
    cv2.drawMarker = orig_marker
    log.check("showing a frame draws the red crosses at the projections (frames 0-4, all cameras)", ok_frames == 5,
              f"{ok_frames} of 5 frames; {len(marks)} crosses")
    old_pfe = old.WandCalibrator.calculate_per_frame_errors
    wc_old = old.WandCalibrator()
    for name in ("final_params", "points_3d", "wand_points", "wand_points_filtered", "wand_length"):
        setattr(wc_old, name, getattr(wc, name))
    wc_old.per_frame_errors = {}
    n_old = sum("proj_pts" in e for e in old_pfe(wc_old).values())
    log(f"      before the fix (git HEAD's calculate_per_frame_errors on the same result): {n_old} of "
        f"{len(wc_old.per_frame_errors)} entries with projections")

    # ---------------- W9: two sizes
    log("\nW9: camera 1 of 1024 x 768 px")
    true_m, uv_m, A, B, on = rig(np.random.default_rng(1315), MIXED)
    view2 = CameraCalibrationView()
    setup(view2, MIXED, uv_m, run_dir / "mixed")
    dlg.folder = str(run_dir / "mixed")
    n_msg = len(dlg.messages)
    view2._run_wand_calibration()
    w2 = view2._calib_worker
    C.wait_for(lambda: w2.isFinished() and any(m[1] in ("Saved", "Calibration Failed")
                                               for m in dlg.messages[n_msg:]), 900)
    fail = [m for m in dlg.messages[n_msg:] if m[1] == "Calibration Failed"]
    log(f"  (B11, recorded) the GUI's pinhole calibration of this rig: "
        f"{'FAILS: ' + fail[0][2] if fail else 'ran: ' + str([m[:2] for m in dlg.messages[n_msg:]])}")

    wc2 = view2.wand_calibrator
    settings = view2._collect_camera_settings_from_table()
    first = list(settings.values())[0]
    wc2.image_size = (first["height"], first["width"])     # as Run Calibration sets it
    wc2.camera_settings = settings
    wc2.wand_length = 10.0
    wc2.wand_points = gui_points(uv_m)
    wc2.wand_points_filtered = None
    wc2.per_frame_errors = {}
    wc2.points_3d = np.vstack([np.vstack([A[i], B[i]]) for i in range(len(A))])
    wc2.dist_coeff_num = 0
    params = []
    for k in range(3):
        p = true_m[k]
        params += list(cv2.Rodrigues(p["R"])[0].ravel()) + list(p["T"].ravel()) + [p["K"][0, 0], p["K"][0, 2],
                                                                                p["K"][1, 2], 0.0, 0.0]
    wc2.final_params = {}
    wc2._parse_results(np.array(params), {0: 0, 1: 1, 2: 2})
    (run_dir / "mixed" / "camFile").mkdir(parents=True, exist_ok=True)
    (run_dir / "mixed" / "camFile_before_fix").mkdir()
    got, got_old = {}, {}
    for k in range(3):
        wc2.export_to_file(k, str(run_dir / "mixed" / "camFile" / f"cam{k}.txt"))
        got[k] = image_size_line(run_dir / "mixed" / "camFile" / f"cam{k}.txt")
        old.WandCalibrator.export_to_file(wc2, k, str(run_dir / "mixed" / "camFile_before_fix" / f"cam{k}.txt"))
        got_old[k] = image_size_line(run_dir / "mixed" / "camFile_before_fix" / f"cam{k}.txt")
    want = {k: f"{H},{W}" for k, (W, H) in enumerate(MIXED)}
    log.check("each camFile's Image Size is its own camera's (rows,cols)", got == want, f"{got}")
    log(f"      before the fix (git HEAD's export_to_file on the same result): {got_old}")
    view.close()
    view2.close()
    return log.save({"equal_image_size_lines": equal_lines, "mixed_image_size_lines": got,
                     "mixed_before_fix": got_old, "f_px": f_fit, "b11_failure": fail[0][2] if fail else None})


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
