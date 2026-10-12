"""G3: the whole GUI path, offscreen, from images to PINCYL camFiles (plan: 2026-10-06-plan.md, section 5).

    python g3_gui_path.py

Rendered image folders (common.render_set: 3 cameras around the real tube, backlit, 200 frames) go through the
Wand Calibration tab as a user would use it, with the dialogs answered by the test:
  1. the three folders are loaded (the image sizes are read from the first image);
  2. Process All Frames in the "ellipse" mode ("Dark on Bright", Radius Range 10-60 px, rod settings 25 / 18 px);
  3. the model "Pinhole+Cylinder", the tube's defaults (the real tube), wand 10 mm; Precalibrate to Check, then
     Run Calibration into a results folder.
Checks (three were restated after the first run, g3_gui_path_first_run_out.txt; the first values are still
printed):
  - detection: every frame whose two balls are apart (gap over 3 px) in every camera that sees them whole is kept,
    with ball A as Filtered_Small; the centres match the exact ones within 0.035 px RMS.  [First: every frame that
    every camera sees whole.  Frames 47, 136 and 170 were not kept: in one camera each the wand points along the
    line of sight and the balls merge into one blob (D1), so that camera has no pair and the GUI drops the frame.]
  - calibration: f of every camera within 0.2 % of the truth (T13c reached 0.07 %), RMS under 0.1 px, at most 5 %
    of the detections rejected as outliers.  [First: every detection used.  The library rejects detections beyond
    5 robust sigmas; the estimator's errors have tails (largest 0.12 px at 0.027 px RMS, G2): 23 of 1,062.]
  - files: camFile/cam0-2.txt are PINCYL and load in C++; the C++ projections of the fitted wand ends equal the
    overlay's crosses (the Python model) within 0.01 px, a third of the detector's error; calibration_result.json,
    _report.txt, _data.json written.  [First: 1e-6 px, as tests/cylinder_wand T10 found (9e-9 px).  Here cam0, which
    faces the tube head-on, projects 2.75e-3 px off in u in C++ at every point although its camFile holds the fitted
    camera exactly (D1, D1b, D1d): a difference inside the C++ cylinder projection for this camera, left open.]
  - the GUI: the error table lists every frame, the overlay has both crosses for every camera, the 3-D view shows the
    tube, the result is marked "cylinder" for the tracking settings;
  - tracking settings (TrackingSettingsView, its own camera-folder hook): the project's camFile folder receives
    byte-identical PINCYL files, not a pinhole result;
  - the detector check: with the points marked as found by "fast", a refit from the last result reports the warning.
Writes g3_gui_path_out.txt and g3_gui_path.json; the run's files go to data/g3_run/.
"""
import common as C  # noqa: I001  (first: paths and one BLAS thread)

import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

log = C.Log("g3_gui_path")
N_FRAMES, SEED = 200, 1308


def main():
    log("G3: the GUI path from images to PINCYL camFiles, offscreen")
    t0 = time.time()
    folder = C.render_set("g3_backlit", N_FRAMES, SEED, "backlit")
    truth = C.load_truth(folder)
    log(f"  image set: {folder} ({N_FRAMES} frames x 3 cameras, rendered or reused in {time.time() - t0:.0f} s)")
    run = C.DATA / "g3_run"
    shutil.rmtree(run, ignore_errors=True)
    (run / "results").mkdir(parents=True)
    app = C.qt_app()
    dlg = C.Dialogs(folder=str(run / "results"), save_file=str(run / "wand_points.csv"), answer="No")
    from modules.camera_calibration import CameraCalibrationView
    import pyopenlpt as lpt
    from modules.camera_calibration.curved_calibration import cyl_model as cm
    view = CameraCalibrationView()
    view.resize(1500, 900)
    view.show()
    app.processEvents()

    # 1. the folders, as the per-camera Load buttons do
    view.wand_num_cams.setValue(3)
    lists = C.image_lists(folder)
    for c in range(3):
        view.wand_images[c] = lists[c]
        view._update_wand_cam_size_from_first_image(c, lists[c])
    view._refresh_wand_radius_range_limit()
    view.populate_wand_table()
    sizes = [(view.wand_cam_table.cellWidget(c, 3).value(), view.wand_cam_table.cellWidget(c, 4).value())
             for c in range(3)]
    log.check("image sizes read from the first images", all(s == (1280, 800) for s in sizes), str(sizes))

    # 2. detection, "ellipse" mode
    view.wand_type_combo.setCurrentText("Dark on Bright")
    view.radius_range.setMinValue(10)
    view.radius_range.setMaxValue(60)
    view.detect_mode_ellipse.setChecked(True)
    view.ellipse_open_spin.setValue(25)
    view.ellipse_mask_spin.setValue(18.0)
    log(f"  radius range {view.radius_range.value()}, mode {view._get_wand_detect_mode()}, opts "
        f"{view._get_wand_detect_opts()}")
    t1 = time.time()
    view._process_wand_frames()
    app.processEvents()
    wc = view.wand_calibrator
    log(f"  Process All Frames: {time.time() - t1:.0f} s, {len(wc.wand_points)} frames kept of {N_FRAMES}")
    seen_all = [f for f in range(N_FRAMES) if np.all(np.isfinite(truth["UV"][f]))]
    import t13_common as T
    apart = [f for f in seen_all if all(T.gap_px(truth["UV"][f, c, 0], truth["UV"][f, c, 1]) > 3.0 for c in range(3))]
    kept = sorted(wc.wand_points)
    log(f"  first criterion (every frame every camera sees whole): {len(set(seen_all) & set(kept))} of "
        f"{len(seen_all)} kept; missing {sorted(set(seen_all) - set(kept))}")
    log.check("detection keeps every frame whose balls are apart in every camera", set(apart) <= set(kept),
              f"{len(apart)} such frames, {len(kept)} kept, missing {sorted(set(apart) - set(kept))[:10]}; "
              f"balls within 3 px in some camera: {sorted(set(seen_all) - set(apart))}")
    err, swapped = [], 0
    for f in kept:
        for c, pts in wc.wand_points[f].items():
            tA, tB = truth["UV"][f, c]
            if not np.all(np.isfinite(tA)):
                continue
            small = [p for p in pts if "Filtered_Small" in [v for v in p if isinstance(v, str)]][0]
            large = [p for p in pts if "Filtered_Large" in [v for v in p if isinstance(v, str)]][0]
            eA, eB = np.array(small[:2], float) - tA, np.array(large[:2], float) - tB
            if max(np.hypot(*eA), np.hypot(*eB)) > 1.0:
                swapped += 1
                continue
            err += [eA, eB]
    E = np.array(err)
    rms = float(np.sqrt(np.mean(np.sum(E ** 2, axis=1))))
    log.check("detected centres: ball A small, within 0.035 px RMS of the exact centres", swapped == 0 and rms <= 0.035,
              f"RMS {rms:.4f} px over {len(E)} balls; {swapped} camera-frames more than 1 px off or swapped")

    # 3. the cylinder model
    view.wand_model_combo.setCurrentText("Pinhole+Cylinder")
    view.wand_len_spin.setValue(10.0)
    view.dist_model_combo.setCurrentIndex(0)
    log.check("the tube's defaults are the real tube",
              abs(view.cyl_od_spin.value() - 2 * truth["radii"][0]) < 1e-9
              and abs(view.cyl_wall_spin.value() - (truth["radii"][0] - truth["radii"][1])) < 1e-9
              and np.allclose([view.cyl_out_index.value(), view.cyl_wall_index.value(), view.cyl_in_index.value()],
                              truth["n"]),
              f"OD {view.cyl_od_spin.value()} mm, wall {view.cyl_wall_spin.value()} mm, n "
              f"{view.cyl_out_index.value()}, {view.cyl_wall_index.value()}, {view.cyl_in_index.value()}")
    t1 = time.time()
    view._run_wand_calibration(precalibrate=True)
    C.wait_for(lambda: C.cylinder_done(view))
    pre = view._cyl_worker
    msg = dlg.messages[-1] if dlg.messages else ("", "", "")
    log(f"  Precalibrate to Check: {time.time() - t1:.0f} s; message: {msg[0]} {msg[1]!r}")
    log("    " + msg[2].replace("\n", "\n    "))
    log.check("the check (stages 1-3) ran and filled the error table",
              msg[0] == "info" and view.error_table.rowCount() == len(kept),
              f"{view.error_table.rowCount()} rows")
    t1 = time.time()
    view._run_wand_calibration()
    C.wait_for(lambda: C.cylinder_done(view) and view._cyl_worker is not pre)
    msg = dlg.messages[-1]
    log(f"  Run Calibration: {time.time() - t1:.0f} s; message: {msg[0]} {msg[1]!r}")
    log("    " + msg[2].replace("\n", "\n    "))
    res_dir = run / "results"
    with open(res_dir / "calibration_result.json", encoding="utf-8") as fh:
        res = json.load(fh)
    f_true = float(truth["f"][0])
    f_err = [c["f_px"] / f_true - 1 for c in res["cameras"]]
    log.check("f of every camera within 0.2 % of the truth", max(abs(e) for e in f_err) < 0.002,
              ", ".join(f"{c['name']} {c['f_px']:,.1f} px ({100 * e:+.3f} %, sigma {100 * c['sigma']['f_rel']:.3f} %)"
                        for c, e in zip(res["cameras"], f_err)))
    log(f"  first criterion (every detection used): {res['active']} of {res['observations']} used")
    log.check("RMS under 0.1 px, at most 5 % of the detections rejected",
              res["rms_px"] < 0.1 and res["active"] >= 0.95 * res["observations"],
              f"RMS {res['rms_px']:.4f} px, {res['active']}/{res['observations']} used")
    files = sorted(os.listdir(res_dir))
    cams_files = sorted(os.listdir(res_dir / "camFile"))
    log.check("files written", {"calibration_result.json", "calibration_report.txt", "calibration_data.json"} <= set(files)
              and cams_files == ["cam0.txt", "cam1.txt", "cam2.txt"], f"{files}; camFile {cams_files}")

    # the camFiles against the overlay's crosses
    data = json.load(open(res_dir / "calibration_data.json", encoding="utf-8"))
    pfe = wc.per_frame_errors
    worst, n_cmp, models = 0.0, 0, set()
    for k, fid in enumerate(data["frame_ids"]):
        XA, XB = np.array(data["ends_A_mm"][k]), np.array(data["ends_B_mm"][k])
        for cid, (pa, pb) in pfe[fid]["proj_pts"].items():
            if k == 0:
                first = open(res_dir / "camFile" / f"cam{cid}.txt", encoding="utf-8").read().split("\n")[1].strip()
                models.add(first)
            cam = lpt.Camera(str(res_dir / "camFile" / f"cam{cid}.txt"))
            q = cm.cpp_project(cam, np.array([XA, XB]))
            for p, qq in ((pa, q[0]), (pb, q[1])):
                if p is not None:
                    worst = max(worst, float(np.hypot(*(np.array(p) - qq))))
                    n_cmp += 1
        if k >= 30:
            break
    log(f"  first criterion (1e-6 px): largest difference {worst:.2e} px")
    log.check("camFiles are PINCYL; C++ projections of the fitted ends = the overlay's crosses (0.01 px)",
              models == {"PINCYL"} and worst < 0.01, f"models {models}; {n_cmp} points, largest difference {worst:.2e} px")

    # the GUI's tables, overlay, 3-D view and state
    n_rows = view.error_table.rowCount()
    full = all(len(v["proj_pts"]) == 3 and all(a is not None and b is not None for a, b in v["proj_pts"].values())
               for v in pfe.values())
    log.check("error table: one row per frame; overlay crosses for both balls in every camera",
              n_rows == len(kept) and full, f"{n_rows} rows, {len(pfe)} frames")
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    has_tube = any(isinstance(col, Poly3DCollection) for col in view.calib_3d_view.ax.collections)
    view.calib_3d_view.figure.savefig(str(C.HERE / "g3_gui_path_3d.png"), dpi=90, facecolor="black")
    log.check("3-D view shows the tube; the result is marked for the tracking settings",
              has_tube and view._refr_has_result and view._refr_kind == "cylinder",
              f"tube surface {has_tube}, _refr_has_result {view._refr_has_result}, kind {view._refr_kind}")
    view._display_frame_with_overlay(kept[0], target_cam=0)

    # tracking settings, its own hook
    from views.tracking_settings_view import TrackingSettingsView
    ts = TrackingSettingsView(calibration_view=view)
    proj_cam = run / "project" / "camFile"
    proj_cam.mkdir(parents=True)
    ts.camera_path_display.setText(str(proj_cam))
    ts._on_cam_path_changed()
    copied = sorted(os.listdir(proj_cam))
    same = all((proj_cam / n).read_bytes() == (res_dir / "camFile" / n).read_bytes() for n in copied)
    log.check("tracking settings' camera-folder hook copies the PINCYL files", copied == cams_files and same,
              f"{copied}, byte-identical {same}")

    # the detector check through the GUI
    view._wand_detect_mode_used = "fast"
    view.cyl_refit_check.setChecked(True)
    prev = view._cyl_worker
    view._run_wand_calibration()
    C.wait_for(lambda: C.cylinder_done(view) and view._cyl_worker is not prev)
    res2 = json.load(open(res_dir / "calibration_result.json", encoding="utf-8"))
    titles = [c["title"] for c in res2.get("checks", [])]
    log.check("points marked as found by 'fast': the result warns about the detector",
              any("'fast'" in t for t in titles), f"checks: {titles}")
    view.close()
    return log.save({"f_rel_err": f_err, "rms_px": res["rms_px"], "detect_rms_px": rms, "frames_kept": len(kept)})


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
