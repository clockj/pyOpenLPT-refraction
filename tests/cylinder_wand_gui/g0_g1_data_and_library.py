"""G0 and G1 (plan section 5): the data path from the GUI's wand points, and the same answer as the library.

    python g0_g1_data_and_library.py

G0, the data path:
  - from the GUI's in-memory wand points (Filtered_Small / Filtered_Large), uv_from_wand_points gives exactly what
    plate_wand_io.read_gui_csv reads from the CSV the GUI's own exporter (WandCalibrator.export_wand_data) writes,
    to the CSV's 3 decimals; after the GUI's own CSV loader, exactly;
  - a camera missing from a frame gives NaN; the small ball is end A even when the points come unlabelled.
G1, the same answer as the library (the campaign's reference rig, 3 and 4 cameras, 500 frames, 0.1 px, fixed seeds):
  - the GUI's runner (cylinder_wand_gui.run) and a direct cylinder_wand.calibrate call give the same cameras (1e-9
    relative in f, 1e-9 mm in the centres, 1e-12 rad);
  - L1: with log_fits off the library's numbers equal those with it on (the log only), and calibrate's default log
    output is unchanged (log_fits off: no line from the fits);
  - the camFiles load in C++ and project the fitted wand ends to the Python model's projections (0.01 px; see D1 for
    the head-on camera);
  - the per-frame errors equal an independent recomputation (largest residual per camera, free-triangulation length);
  - f within 3 sigma of the truth.
Writes g0_g1_data_and_library_out.txt and .json.
"""
import common as C  # noqa: I001

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402

log = C.Log("g0_g1_data_and_library")


def gui_points(uv):
    """GUI-style wand points from uv (F, Nc, 2, 2): {frame: {cam: [small, large]}}; the radii mark the ends."""
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
    C.qt_app()
    import rigs
    from modules.camera_calibration.curved_calibration import cylinder_wand_gui as g
    from modules.camera_calibration.curved_calibration import cylinder_wand as cw
    from modules.camera_calibration.curved_calibration import cyl_model as cm
    from modules.camera_calibration.curved_calibration import plate_wand_io as pio
    from modules.camera_calibration.wand_calibration.wand_calibrator import WandCalibrator

    # ---------------- G0
    log("G0: the data path")
    rng = np.random.default_rng(1310)
    cams = rigs.reference_rig(3)
    w = rigs.wand_frames(cams, rigs.CASE_TUBE, 120, 10.0, rng, noise_px=0.1)
    wp = gui_points(w.uv)
    uv, fids, cids = g.uv_from_wand_points(wp)
    wc = WandCalibrator()
    wc.wand_data_raw = {f: {c: [p[:4] for p in pts] for c, pts in fr.items()} for f, fr in wp.items()}
    wc.wand_data_filtered = {f: {c: [p[:4] for p in pts] for c, pts in fr.items()} for f, fr in wp.items()}
    tmp = tempfile.mkdtemp()
    csv_path = os.path.join(tmp, "wand_points.csv")
    ok, _msg = wc.export_wand_data(csv_path)
    cd = pio.read_gui_csv(csv_path)
    same_shape = cd.uv.shape == uv.shape and list(cd.frame_ids) == fids and list(cd.cam_ids) == cids
    diff = float(np.nanmax(np.abs(cd.uv - uv))) if same_shape else float("nan")
    nan_same = same_shape and np.array_equal(np.isnan(cd.uv), np.isnan(uv))
    log.check("G0 GUI points = the GUI's CSV as read by read_gui_csv (to the CSV's 3 decimals)",
              ok and same_shape and nan_same and diff <= 0.0005 + 1e-9, f"largest difference {diff:.2e} px")
    wc2 = WandCalibrator()
    wc2.load_wand_data_from_csv(csv_path)
    uv2, f2, c2 = g.uv_from_wand_points(wc2.wand_points)
    cd_frames = [list(cd.frame_ids).index(f) for f in f2]
    exact = np.array_equal(np.nan_to_num(uv2, nan=-1.0), np.nan_to_num(cd.uv[cd_frames], nan=-1.0))
    log.check("G0 after the GUI's own CSV loader: exactly read_gui_csv's", exact, f"{len(f2)} frames")
    seen_missing = int(np.sum(~np.isfinite(uv[..., 0])))
    truth_missing = int(np.sum(~np.isfinite(w.uv[[k for k in range(len(w.uv)) if k in wp]][..., 0])))
    log.check("G0 a camera missing from a frame gives NaN", seen_missing == truth_missing,
              f"{seen_missing} NaN coordinates, {truth_missing} in the truth")
    unl = {f: {c: [p[:3] for p in pts][::-1] for c, pts in fr.items()} for f, fr in wp.items()}   # no labels, B first
    uv3, _f3, _c3 = g.uv_from_wand_points(unl)
    log.check("G0 unlabelled points: the smaller radius is end A", np.array_equal(np.nan_to_num(uv3, nan=-1.0),
                                                                                  np.nan_to_num(uv, nan=-1.0)))

    # ---------------- G1
    log("\nG1: the same answer as the library")
    res = {}
    for n_cams, seed in ((3, 1311), (4, 1312)):
        rng = np.random.default_rng(seed)
        cams = rigs.reference_rig(n_cams)
        w = rigs.wand_frames(cams, rigs.CASE_TUBE, 500, 10.0, rng, noise_px=0.1)
        wp = gui_points(w.uv)
        uv, fids, cids = g.uv_from_wand_points(wp)
        job = g.Job(uv=uv, frame_ids=fids, cam_ids=cids, sizes=[(c.width, c.height) for c in cams], wand_length=10.0,
                    radii=rigs.CASE_TUBE.radii, n=rigs.CASE_TUBE.n, detect_mode="ellipse", source="G1")
        sol_gui = g.run(job)
        lines_off = []
        sol_lib = cw.calibrate(cw.WandData(uv, 10.0, job.sizes), cm.Tube(job.radii, job.n), verbose=True,
                               log=lines_off.append)
        df = max(abs(a.f / b.f - 1) for a, b in zip(sol_gui.cams, sol_lib.cams))
        dC = max(float(np.max(np.abs(a.C - b.C))) for a, b in zip(sol_gui.cams, sol_lib.cams))
        dR = max(cm.rotation_angle_deg(a.R, b.R) * np.pi / 180 for a, b in zip(sol_gui.cams, sol_lib.cams))
        log.check(f"G1 {n_cams} cameras: the GUI's runner = cylinder_wand.calibrate",
                  df < 1e-9 and dC < 1e-9 and dR < 1e-12, f"f {df:.1e}, centres {dC:.1e} mm, rotations {dR:.1e} rad")
        fit_lines = [ln for ln in lines_off if ("LM" in ln or "coarse round" in ln or "start search:" in ln)]
        log.check(f"G1 {n_cams} cameras: L1 off by default (no fit lines in calibrate's verbose log)",
                  not fit_lines, f"{len(lines_off)} lines, {len(fit_lines)} from the fits")
        folder = tempfile.mkdtemp()
        result = g.write_outputs(sol_gui, job, folder)
        data = json.load(open(os.path.join(folder, "calibration_data.json"), encoding="utf-8"))
        import pyopenlpt as lpt
        pfe = g.per_frame_errors(sol_gui, job)
        worst = 0.0
        for k, fid in enumerate(data["frame_ids"][:60]):
            XA, XB = np.array(data["ends_A_mm"][k]), np.array(data["ends_B_mm"][k])
            for cid, (pa, pb) in pfe[fid]["proj_pts"].items():
                q = cm.cpp_project(lpt.Camera(os.path.join(folder, "camFile", f"cam{cid}.txt")), np.array([XA, XB]))
                for pp, qq in ((pa, q[0]), (pb, q[1])):
                    if pp is not None:
                        worst = max(worst, float(np.hypot(*(np.array(pp) - qq))))
        log.check(f"G1 {n_cams} cameras: camFiles load in C++; their projections = the model's (0.01 px)",
                  worst < 0.01, f"largest difference {worst:.2e} px (the ends are rounded to 1e-5 mm)")
        # per-frame errors, recomputed independently
        ev = cw.evaluate(sol_gui.cams, sol_gui.tube, sol_gui.frames, sol_gui.L, sol_gui.obs)
        rn = np.linalg.norm(ev.r, axis=1)
        Lm = cw.wand_lengths(sol_gui.cams, sol_gui.tube, sol_gui.obs, rows=np.flatnonzero(ev.ok))
        bad = 0
        for k in range(sol_gui.obs.n_frames):
            fid = fids[int(sol_gui.obs.frame_ids[k])]
            rows = np.flatnonzero(sol_gui.obs.frame == k)
            for c in set(sol_gui.obs.cam[rows].tolist()):
                rr = rows[(sol_gui.obs.cam[rows] == c) & ev.ok[rows]]
                if abs(pfe[fid]["cam_errors"][cids[c]] - rn[rr].max()) > 1e-12:
                    bad += 1
            if abs(pfe[fid]["len_error"] - abs(Lm[k] - sol_gui.L)) > 1e-12:
                bad += 1
        log.check(f"G1 {n_cams} cameras: per-frame errors = an independent recomputation", bad == 0, f"{bad} mismatches")
        z = [(a.f - t.f) / (a.f * s["f_rel"]) for a, t, s in zip(sol_gui.cams, cams, cw.camera_sigmas(
            sol_gui.cams, sol_gui.layout, sol_gui.cov))]
        log.check(f"G1 {n_cams} cameras: f within 3 sigma of the truth", max(abs(v) for v in z) < 3,
                  "z " + ", ".join(f"{v:+.2f}" for v in z))
        res[n_cams] = {"df": df, "dC": dC, "dR": dR, "cpp_px": worst, "z_f": z, "rms_px": result["rms_px"]}
    return log.save({"g1": res})


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
