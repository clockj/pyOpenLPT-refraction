"""P0: with no new option set, the flat-window calibration runs as before (2026-10-09-plan.md, P0).

    python p0_unchanged.py

Today's code is commit cffed7d's (git show): its view (modules/camera_calibration/view.py, loaded as a module of the
same package, so that it builds the same Job class) and its plate_wand_gui.run (loaded on its own). On G8's rig B (tests/
plate_wand prigs: two perpendicular 12.7 mm walls, two cameras behind each; 150 frames, 0.1 px), with the distortion box
set by the user to "None", then to "Radial k1" (D1 changes only the default):
- the Job that today's view builds and the Job that the new view builds: every field equal, the new ones at their
  defaults (no lens terms, constraints, start or hints);
- today's run and the new run on them: cameras, windows, RMS and the detections used bit for bit equal;
- Precalibrate (plate_wand_gui.precheck, stages 1-3) on the first Job: the same, bit for bit.
Writes p0_unchanged_out.txt and .json here.
"""
import pgcommon as P  # noqa: I001

import importlib.util  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

log = P.Log("p0_unchanged")
OLD_FIELDS = ["uv", "frame_ids", "cam_ids", "sizes", "wand_length", "windows", "cam_to_window", "fit_k1",
              "noise_weights", "n_final", "start", "pixel_mm", "detect_mode", "source", "hints", "window_ids"]


def head_view_class():
    path = P.committed_file("modules/camera_calibration/view.py", f"view_{P.REF_COMMIT}.py")
    name = "modules.camera_calibration._view_head"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def set_up(view, rig, wp, dist_index):
    """G8's settings, through the widgets every version has."""
    view.wand_num_cams.setValue(rig.n_cams)
    for r in range(rig.n_cams):
        view.wand_cam_table.cellWidget(r, 3).setValue(rig.cams[r].width)
        view.wand_cam_table.cellWidget(r, 4).setValue(rig.cams[r].height)
    view.wand_calibrator.wand_points = wp
    view.wand_calibrator.wand_points_filtered = None
    view._wand_detect_mode_used = "ellipse"
    view.wand_model_combo.setCurrentText("Pinhole+Refraction")
    view.wand_len_spin.setValue(rig.L)
    view.dist_model_combo.setCurrentIndex(dist_index)
    view.dist_model_combo.activated.emit(dist_index)
    view.window_count_spin.setValue(2)
    for row, wid in enumerate([0, 0, 1, 1]):
        view.cam_window_table.cellWidget(row, 1).setCurrentIndex(wid)
    view.media3_combo.setCurrentText("Other")
    view.media3_index.setValue(1.333)
    view.media2_thick.setValue(12.7)


def catch(module, view):
    real = module.ProcessCalibWorker
    module.ProcessCalibWorker = P.JobCatcher
    P.JobCatcher.caught = []
    try:
        view._run_wand_calibration()
    finally:
        module.ProcessCalibWorker = real
    view._busy_end('wand_calibration')
    return P.JobCatcher.caught[-1][1] if P.JobCatcher.caught else None


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    return a == b


def sol_summary(sol):
    return {"R": np.array([c.R for c in sol.cams]), "C": np.array([c.C for c in sol.cams]),
            "f": np.array([c.f for c in sol.cams]), "pp": np.array([[c.cx, c.cy] for c in sol.cams]),
            "dist": np.array([c.dist for c in sol.cams]), "n": np.array([w.n for w in sol.rig.windows]),
            "d": np.array([w.d for w in sol.rig.windows]), "rms": np.array([sol.rms]),
            "active": np.asarray(sol.obs.active)}


def main():
    P.qt_app()
    P.Dialogs(folder=str(P.DATA / "p0_unused"))
    import prigs
    import modules.camera_calibration.view as V
    from modules.camera_calibration.curved_calibration import plate_wand_gui as pg
    H = head_view_class()
    pg_head = P.head_plate_wand_gui()
    rig = prigs.rig_B()
    wp = P.rig_b_points(rig, 150)
    log(f"P0: today's view {H.__file__} and run {pg_head.__file__}; rig B, {len(wp)} frames")
    out = {}
    for dist_index, label in ((0, "None"), (1, "Radial k1")):
        v_old, v_new = H.CameraCalibrationView(), V.CameraCalibrationView()
        set_up(v_old, rig, wp, dist_index)
        set_up(v_new, rig, wp, dist_index)
        j_old, j_new = catch(H, v_old), catch(V, v_new)
        diff = [f for f in OLD_FIELDS if not same(getattr(j_old, f), getattr(j_new, f))]
        new_defaults = (j_new.fit_lens == () and j_new.constraints is None and j_new.start_source == ""
                        and j_new.hints_source == "")
        log.check(f"distortion {label}: the Job of today's view = the new view's (new fields at their defaults)",
                  j_old is not None and j_new is not None and not diff and new_defaults,
                  f"differing fields {diff}; windows {j_new.windows}; fit_k1 {j_new.fit_k1}")
        s_old = sol_summary(pg_head.run(j_old))
        s_new = sol_summary(pg.run(j_new))
        neq = [k for k in s_old if not np.array_equal(s_old[k], s_new[k])]
        log.check(f"distortion {label}: today's run = the new run, bit for bit (cameras, windows, RMS, detections used)",
                  not neq, f"differing {neq}; RMS {s_new['rms'][0]:.6f} px; f {np.round(s_new['f'], 1).tolist()}")
        out[label] = {"rms_px": float(s_new["rms"][0]), "f_px": s_new["f"].tolist(), "differing": neq}
        if dist_index == 0:   # Precalibrate (stages 1-3) on the same Job
            p_old, p_new = sol_summary(pg_head.precheck(j_old)), sol_summary(pg.precheck(j_new))
            neq_p = [k for k in p_old if not np.array_equal(p_old[k], p_new[k])]
            log.check("Precalibrate: today's stages 1-3 = the new ones, bit for bit", not neq_p,
                      f"differing {neq_p}; RMS {p_new['rms'][0]:.6f} px")
    log.save(out)


if __name__ == "__main__":
    main()
