"""P3: the wand tab's flat-window panel (a light check): it builds, its new controls behave, and the Job they make.

    python p3_panel.py

An offscreen CameraCalibrationView (G8's set-up: rig B of tests/plate_wand, 2 windows, cameras 0-1 behind window 0
and 2-3 behind window 1). Checks:
- D1: Pinhole+Refraction shows "Radial k1" until the user picks a distortion model; Pinhole and Pinhole+Cylinder show
  "None"; after a pick, switching models keeps it;
- the Windows table follows Number of Windows; new rows take the diagram's thickness and plate index; the diagram's
  values set every row; a row's own value reaches the Job;
- Start and Hints enable their folder fields only where they apply; the old refit checkbox is not shown;
- the Job: per-window windows, k1 (+ k2 for "Radial k1+k2"), the lens terms, the constraints from the panel (rig
  indices), hints only for a searched start; Precalibrate drops the start and the constraints with a note;
- bad inputs are refused with a message (an aim's unknown camera, a channel without widths, a missing start folder).
The panel's pictures: p3b_screens.py.
Writes p3_panel_out.txt and .json here.
"""
import pgcommon as P  # noqa: I001

import numpy as np  # noqa: E402

log = P.Log("p3_panel")


def main():
    app = P.qt_app()
    dlg = P.Dialogs(folder=str(P.DATA / "p3_unused"))
    import prigs
    from PySide6.QtWidgets import QTableWidgetItem
    rig = prigs.rig_B()
    wp = P.rig_b_points(rig, 120)
    view = P.make_view(rig, wp, [0, 0, 1, 1])

    # D1: the distortion box per model
    seen = {}
    for m in ("Pinhole", "Pinhole+Refraction", "Pinhole+Cylinder", "Pinhole+Refraction"):
        view.wand_model_combo.setCurrentText(m)
        app.processEvents()
        seen.setdefault(m, []).append(view.dist_model_combo.currentText())
    log.check("D1: Radial k1 for Pinhole+Refraction, None for Pinhole and Pinhole+Cylinder, until the user picks",
              seen == {"Pinhole": ["None (0)"], "Pinhole+Refraction": ["Radial k1 (1)", "Radial k1 (1)"],
                       "Pinhole+Cylinder": ["None (0)"]}, str(seen))
    view.dist_model_combo.setCurrentIndex(0)
    view.dist_model_combo.activated.emit(0)          # the user picks None
    view.wand_model_combo.setCurrentText("Pinhole")
    view.wand_model_combo.setCurrentText("Pinhole+Refraction")
    log.check("after the user's pick, switching models keeps it", view.dist_model_combo.currentIndex() == 0,
              view.dist_model_combo.currentText())
    view.dist_model_combo.setCurrentIndex(1)

    # the Windows table
    tbl = view.refr_window_table
    rows = tbl.rowCount()
    vals0 = view._refr_window_values()
    view.window_count_spin.setValue(3)
    view.media2_index.setValue(1.52)
    vals3 = view._refr_window_values()
    view.window_count_spin.setValue(2)
    tbl.item(1, 1).setText("10.0")              # window 1's own thickness
    tbl.item(0, 3).setText("100.0")             # channel widths
    tbl.item(1, 3).setText("150.0")
    vals2 = view._refr_window_values()
    log.check("the Windows table follows Number of Windows and the diagram's values",
              rows == 2 and vals0 == {0: (12.7, 1.49, 0.0), 1: (12.7, 1.49, 0.0)} and len(vals3) == 3
              and all(v[1] == 1.52 for v in vals3.values()) and vals2 == {0: (12.7, 1.52, 100.0), 1: (10.0, 1.52, 150.0)},
              f"rows {rows}; {vals0}; 3 windows {vals3}; edited {vals2}")

    # Start and Hints enabling
    st = []
    for k in (0, 1, 2):
        view.refr_start_combo.setCurrentIndex(k)
        view.refr_hints_combo.setCurrentIndex(2)
        st.append((view.refr_start_folder.isEnabled(), view.refr_hints_combo.isEnabled(),
                   view.refr_hints_folder.isEnabled()))
    view.refr_start_combo.setCurrentIndex(0)
    view.refr_hints_combo.setCurrentIndex(0)
    log.check("folder fields only where they apply; hints only for a searched start; the refit box not shown",
              st == [(False, True, True), (False, False, False), (True, False, False)]
              and not view.refr_refit_check.isVisibleTo(view.refraction_group), str(st))

    # the Job
    got = P.catch_job(view)
    job = got[1] if got else None
    log.check("Run builds the Job: per-window windows, k1, no lens terms, no start, hints or constraints",
              job is not None and job.windows == [
                  {"name": "window 0", "thickness_mm": [12.7], "n": [1.0, 1.52, 1.333]},
                  {"name": "window 1", "thickness_mm": [10.0], "n": [1.0, 1.52, 1.333]}]
              and job.fit_k1 and job.fit_lens == () and job.start is None and job.hints == {} and job.constraints is None,
              f"target {got[0] if got else None}; windows {job.windows if job else None}; fit_k1 {job.fit_k1 if job else None}")
    view.dist_model_combo.setCurrentIndex(2)
    view.refr_pp_check.setChecked(True)
    view.refr_tangential_check.setChecked(True)
    view.refr_cons_group.setChecked(True)
    view.refr_channel_check.setChecked(True)
    view.refr_channel_tol.setValue(3.0)
    view.refr_ball_a.setValue(1.5)
    view.refr_ball_b.setValue(2.0)
    view.refr_aim_table.insertRow(0)
    for j, t in enumerate(["0,1", "1", "75", ""]):
        view.refr_aim_table.setItem(0, j, QTableWidgetItem(t))
    view.refr_dist_table.insertRow(0)
    for j, t in enumerate(["3", "250", "5"]):
        view.refr_dist_table.setItem(0, j, QTableWidgetItem(t))
    view.refr_hints_combo.setCurrentIndex(1)
    job = P.catch_job(view)[1]
    want = {"channel": {"width_mm": [100.0, 150.0], "tolerance_mm": 3.0, "ball_radius_mm": [1.5, 2.0]},
            "aim": [{"cams": [0, 1], "wall": 1, "mm": 75.0}], "distance": [{"cam": 3, "mm": 250.0, "sigma_mm": 5.0}]}
    log.check("the options reach the Job: k1 + k2, pp, tangential, the constraints in rig indices, the hints",
              job.fit_k1 and job.fit_lens == ("pp", "k2", "tangential") and job.constraints == want
              and job.hints == {"normals": [[0.0, 0.0, 1.0]] * 4} and job.hints_source == "windows face the cameras",
              f"fit_lens {job.fit_lens}; constraints {job.constraints}; hints {job.hints_source}")
    n_msgs = len(dlg.messages)
    got = P.catch_job(view, precalibrate=True)
    log.check("Precalibrate: no constraints (with a note), the hints kept",
              got is not None and got[0] == "child_precheck" and got[1].constraints is None and got[1].hints
              and any("constraints" in n for n in view._cyl_notes), f"notes {view._cyl_notes}")

    # bad inputs
    refused = []
    view.refr_aim_table.setItem(0, 0, QTableWidgetItem("0,7"))
    refused.append(P.catch_job(view) is None and "Aim 1" in (dlg.messages[-1][2] if dlg.messages else ""))
    view.refr_aim_table.setItem(0, 0, QTableWidgetItem("0,1"))
    tbl.item(0, 3).setText("0.0")
    tbl.item(1, 3).setText("0.0")
    refused.append(P.catch_job(view) is None and "inside width" in dlg.messages[-1][2])
    tbl.item(0, 3).setText("100.0")
    view.refr_start_combo.setCurrentIndex(2)
    view.refr_start_folder.setText(str(P.DATA / "no_such_folder"))
    refused.append(P.catch_job(view) is None and "saved fit" in dlg.messages[-1][2])
    view.refr_start_combo.setCurrentIndex(0)
    log.check("bad inputs refused with a message (aim camera, channel without widths, missing start folder)",
              all(refused), f"{refused}; last messages {[m[2][:80] for m in dlg.messages[n_msgs:]]}")

    log.save({"messages": dlg.messages})


if __name__ == "__main__":
    main()
