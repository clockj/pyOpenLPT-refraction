"""P1: the real case's delivered fit W4c (calib_plate_wand_ref_centre) made through the GUI's path (plan P1).

    python p1_real_case.py

The case's own command was run_calibration.py --fit-k1 --channel --aim centre --start tests/states/t2_all.pkl. Here the
GUI's view (offscreen) is set as a user would set it for the 20260727 case (X:, read only):
- Load Wand Points (from CSV): the case's detections, T0\\circle_results_table_new.csv, through the GUI's own button;
- camera table: 4 cameras, 2048 x 1952; wand length 10 mm; Pinhole+Refraction with "Radial k1" (now its default);
- 2 windows: cameras 0 and 1 behind window 0, 2 and 3 behind window 1; 12.7 mm; n 1.0 / 1.49 / 1.333;
- Start: Saved fit, T0\\calib_plate_wand (T2's delivered result; that it holds W4c's start, tests/states/t2_all.pkl,
  value for value, is checked here first);
- Constraints: the channel (inside widths 100 / 150 mm in the Windows table, tolerance 3 mm, balls 1.5 / 2.0 mm) and an
  aim (cameras 0, 1 at 75 mm behind window 1).
The Job the view builds runs as the GUI's process runs it (plate_wand_gui.run; here in this process, about 4 min) and
is written by plate_wand_gui.write_outputs to data/p1_run/. Pass: cameras, windows, RMS and the detections used equal
W4c's delivered calibration_result.json (bit for bit, else to the optimizer's last digits: f 1e-9 relative, positions
1e-6 mm, rotations 1e-9, RMS 1e-9 px); the record and the page list the start, the constraints and k1.
Writes p1_real_case_out.txt and .json here.
"""
import pgcommon as P  # noqa: I001

import json  # noqa: E402
import pickle  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

log = P.Log("p1_real_case")
CSV = P.X_T0 / "circle_results_table_new.csv"
T2 = P.X_T0 / "calib_plate_wand"
W4C = P.X_T0 / "calib_plate_wand_ref_centre"


def main():
    P.qt_app()
    run_dir = P.DATA / "p1_run"
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    dlg = P.Dialogs(folder=str(run_dir))
    from PySide6.QtWidgets import QFileDialog, QTableWidgetItem
    QFileDialog.getOpenFileName = lambda *a, **k: (str(CSV), "CSV Files (*.csv)")
    from modules.camera_calibration import CameraCalibrationView
    from modules.camera_calibration.curved_calibration import plate_wand_gui as pg

    # the start folder holds W4c's start
    st = pickle.load(open(P.CASE / "tests" / "states" / "t2_all.pkl", "rb"))
    t2 = json.load(open(T2 / "calibration_result.json", encoding="utf-8"))
    dev = max(max(float(np.max(np.abs(np.asarray(R) - np.asarray(c["R"])))), float(np.max(np.abs(np.asarray(C) - np.asarray(c["centre_mm"])))),
                  abs(float(f) - c["f_px"]), float(np.max(np.abs(np.asarray(dist) - np.asarray(c["dist"])))))
              for (R, C, f, _cx, _cy, _w, _h, dist), c in zip(st["cams"], t2["cameras"]))
    log.check("T2's saved folder holds W4c's start (tests/states/t2_all.pkl) value for value", dev == 0.0,
              f"largest difference {dev}")

    view = CameraCalibrationView()
    view.wand_num_cams.setValue(4)
    for r in range(4):
        view.wand_cam_table.cellWidget(r, 3).setValue(2048)
        view.wand_cam_table.cellWidget(r, 4).setValue(1952)
    t0 = time.time()
    view._load_wand_points_for_calibration()
    log(f"  Load CSV: {len(view.wand_calibrator.wand_points)} frames in {time.time() - t0:.0f} s; "
        f"message {dlg.messages[-1][:2] if dlg.messages else None}")
    view.wand_model_combo.setCurrentText("Pinhole+Refraction")
    view.wand_len_spin.setValue(10.0)
    view.window_count_spin.setValue(2)
    for row, wid in enumerate([0, 0, 1, 1]):
        view.cam_window_table.cellWidget(row, 1).setCurrentIndex(wid)
    view.media3_combo.setCurrentText("Other")
    view.media3_index.setValue(1.333)
    view.media2_index.setValue(1.49)
    view.media2_thick.setValue(12.7)
    view.refr_window_table.item(0, 3).setText("100")
    view.refr_window_table.item(1, 3).setText("150")
    view.refr_start_combo.setCurrentIndex(2)
    view.refr_start_folder.setText(str(T2))
    view.refr_cons_group.setChecked(True)
    view.refr_channel_check.setChecked(True)
    view.refr_channel_tol.setValue(3.0)
    view.refr_ball_a.setValue(1.5)
    view.refr_ball_b.setValue(2.0)
    view.refr_aim_table.insertRow(0)
    for j, t in enumerate(["0,1", "1", "75", ""]):
        view.refr_aim_table.setItem(0, j, QTableWidgetItem(t))
    log(f"  distortion: {view.dist_model_combo.currentText()} (the default)")
    got = P.catch_job(view)
    job = got[1] if got else None
    w4 = json.load(open(W4C / "calibration_result.json", encoding="utf-8"))
    log.check("the view's Job: k1, no lens terms, the saved start, the constraints as the case's",
              job is not None and job.fit_k1 and job.fit_lens == () and job.start is not None
              and job.start_source == str(T2) and job.constraints == w4["settings"]["constraints"],
              f"constraints {job.constraints if job else None}; case {w4['settings']['constraints']}")
    if job is None:
        log.save({"messages": dlg.messages})
        return

    t0 = time.time()
    sol = pg.run(job)
    secs = time.time() - t0
    res = pg.write_outputs(sol, job, str(run_dir))
    log(f"  run: {secs:.0f} s; RMS {sol.rms:.10f} px (W4c {w4['rms_px']:.10f}); active {int(sol.obs.active.sum())} of "
        f"{len(sol.obs)} (W4c {w4['active']} of {w4['observations']})")
    d = {"f_rel": max(abs(c.f / w["f_px"] - 1) for c, w in zip(sol.cams, w4["cameras"])),
         "C_mm": max(float(np.max(np.abs(c.C - np.asarray(w["centre_mm"])))) for c, w in zip(sol.cams, w4["cameras"])),
         "R": max(float(np.max(np.abs(c.R - np.asarray(w["R"])))) for c, w in zip(sol.cams, w4["cameras"])),
         "pp_px": max(max(abs(c.cx - w["cx"]), abs(c.cy - w["cy"])) for c, w in zip(sol.cams, w4["cameras"])),
         "dist": max(float(np.max(np.abs(c.dist - np.asarray(w["dist"])))) for c, w in zip(sol.cams, w4["cameras"])),
         "n": max(float(np.max(np.abs(win.n - np.asarray(w["normal"])))) for win, w in zip(sol.rig.windows, w4["windows"])),
         "d_mm": max(abs(win.d - w["d_mm"]) for win, w in zip(sol.rig.windows, w4["windows"])),
         "rms_px": abs(sol.rms - w4["rms_px"])}
    exact = all(v == 0.0 for v in d.values()) and int(sol.obs.active.sum()) == w4["active"]
    close = (d["f_rel"] <= 1e-9 and d["C_mm"] <= 1e-6 and d["R"] <= 1e-9 and d["pp_px"] <= 1e-6 and d["dist"] <= 1e-9
             and d["n"] <= 1e-9 and d["d_mm"] <= 1e-6 and d["rms_px"] <= 1e-9 and int(sol.obs.active.sum()) == w4["active"])
    log.check("W4c reproduced through the GUI's path: cameras, windows, RMS and detections used",
              exact or close, ("bit for bit" if exact else "to the last digits") + f"; largest differences {d}")
    gui = res.get("gui", {}).get("settings", {})
    page = (run_dir / "calibration_report.html").read_text(encoding="utf-8") if (run_dir / "calibration_report.html").exists() else ""
    log.check("the record and the page list the start, the constraints and k1",
              gui.get("start") == str(T2) and gui.get("constraints") == job.constraints and res.get("fit_k1")
              and "calib_plate_wand" in page and "channel" in page and isinstance(res.get("gui", {}).get("seconds"), float),
              f"start {gui.get('start')}; run time {res.get('gui', {}).get('seconds')}")
    lines = pg.summary_lines(res)
    log("  the GUI's message:\n    " + "\n    ".join(lines))
    log.save({"differences": d, "seconds": secs, "summary_lines": lines})


if __name__ == "__main__":
    main()
