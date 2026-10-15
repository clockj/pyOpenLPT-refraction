"""C1: the delivered calibration of T0 (plan: 2026-10-15-plan.md, Phase C; cases/cyl_20261008/calibrate.py with one
addition, the detection record).

    python calibrate.py

Reads C1's detections, keeps the camera-frames that pass config.accept (case.classify) and the frames that at least
config.accept.min_cameras cameras keep (the others are "not seen" in the fit), writes them in the GUI's format
(<out_dir>/wand_points.csv) with the selection's counts (<out_dir>/detection/selection.json), and runs
cylinder_wand_gui.run (cylinder_wand.calibrate as the GUI's "Pinhole+Cylinder" runs it: tube and wand of
config.json, k1 off, per-camera noise weights, two starts carried; each camera's pixel size for f in mm).
cylinder_wand_gui.write_outputs then writes the GUI's files into <out_dir>: camFile/cam1.txt, cam2.txt, cam3.txt
(PINCYL), calibration_result.json, calibration_report.txt, calibration_data.json, calibration_report.html (the
results page).

The addition: the detection settings are recorded as the GUI records them since 13 Oct (wand_calibrator's detection
record): <out_dir>/wand_points_detection.json beside wand_points.csv, with the GUI's camera numbers (0, 1, 2), so that
Load Wand Points in the GUI shows how the points were found; and the Job's `detection` (calibration_result.json
gui.settings.detection, and the results page's Detection rows), with the case's camera numbers (1, 2, 3), as the
camFiles and the page name them.
Writes calibrate_out.txt and calibrate.json (the summary the tests compare against) beside this script.
"""
import datetime
import json
import os
from collections import Counter

import case


def detection_record(cfg: dict, numbers: str) -> dict:
    """The detection record (wand_calibrator.save_detection_record's layout, one run): the mode, the Wand Type and
    each camera's settings and images.  numbers: "gui" (cam_id = the GUI's camera index) or "case" (the case's id)."""
    cams = []
    for c in case.cam_ids(cfg):
        d = case.det_settings(cfg, c)
        cc = case.cam_cfg(cfg, c)
        frames = case.cam_frames(cfg, c)
        name = lambda f: cfg["frames"]["pattern"].format(frame=f - case.first_image(cfg, c))  # noqa: E731
        cams.append({"cam_id": int(cc["gui_index"]) if numbers == "gui" else int(c),
                     "min_radius": float(d["min_r_px"]), "max_radius": float(d["max_r_px"]),
                     "open_px": int(d["open_px"]), "rod_mask_px": float(d["rod_mask_px"]),
                     "guard_px": float(d["guard_px"]), "pixel_um": round(1000.0 * float(cc["pixel_mm"]), 6),
                     "images": {"folder": os.path.join(cfg["raw_dir"], cc["folder"]), "first": name(frames[0]),
                                "last": name(frames[-1]), "count": len(frames)}})
    det_file = case.detections_path(cfg)
    written = datetime.datetime.fromtimestamp(os.path.getmtime(det_file)).strftime("%Y-%m-%d %H:%M:%S")
    acc = cfg["accept"]
    run = {"written": written, "resumed": False, "mode": "ellipse",
           "wand_type": "bright" if case.det_settings(cfg, case.cam_ids(cfg)[0])["polarity"] == "bright" else "dark",
           "per_camera": True, "cameras": cams,
           "note": (f"Detected by pyOpenLPT/cases/{cfg['case']}/detect.py with ball_ellipses.detect_wand_balls (the "
                    "GUI's ellipse mode) and these settings, not in the GUI's wand tab. Kept by the case's rules "
                    f"(config.accept: two balls apart, edge RMS at most {acc['edge_rms_max_px']} px, aspect "
                    f"{acc['aspect_range'][0]}-{acc['aspect_range'][1]}, sizes within {100 * acc['size_tol']:.0f} % "
                    f"of the camera's median), frames kept in at least {acc['min_cameras']} cameras. Camera numbers: "
                    + ("the GUI's (0, 1, 2 = cam1, cam2, cam3)." if numbers == "gui" else "the case's (cam1-cam3)."))}
    return {"about": "How the wand points of this file were detected: one entry per detection run, with the mode, the "
                     "Wand Type, and each camera's settings as its detector got them (radii in px, in that camera's "
                     "pixels) and its images.",
            "runs": [run]}


def main():
    cfg = case.load_config()
    log = case.Log(os.path.join(case.HERE, "calibrate"))
    cams = case.cam_ids(cfg)
    dets, kept, reasons, medians, frames = case.load_selection(cfg)
    log("C1: the delivered calibration")
    log(f"  detections: {case.detections_path(cfg)}")
    sel = {"cameras": {}, "frames_kept_in_min_cameras": len(frames), "frames_total": len(case.all_frames(cfg)),
           "accept": cfg["accept"], "detect": {f"cam{c}": case.det_settings(cfg, c) for c in cams}}
    for c in cams:
        keys = [k for k in dets if k[1] == c]
        why = Counter(reasons[k] for k in keys if k in reasons)
        nk = sum(1 for k in keys if k in kept)
        ma, mb = medians[c]
        sel["cameras"][f"cam{c}"] = {"camera_frames": len(keys), "kept": nk, "rejected": dict(why),
                                     "median_r_px": [ma, mb]}
        log(f"  cam{c}: kept {nk} of {len(keys)} camera-frames; median r (sqrt(a b)) A {ma:.1f} px, B {mb:.1f} px; "
            "rejected: " + ", ".join(f"{n} {r}" for r, n in why.most_common()))
    n3 = len(case.frames_in_all(kept, cams))
    log(f"  frames kept in at least {cfg['accept']['min_cameras']} cameras: {len(frames)} of "
        f"{len(case.all_frames(cfg))}; in all three: {n3}")
    sel["frames_kept_in_all_three"] = n3
    os.makedirs(os.path.join(cfg["out_dir"], "detection"), exist_ok=True)
    sel["frames"] = frames
    with open(os.path.join(cfg["out_dir"], "detection", "selection.json"), "w", encoding="utf-8") as fh:
        json.dump(sel, fh, indent=1)
    csv_path = os.path.join(cfg["out_dir"], "wand_points.csv")
    case.write_gui_csv(csv_path, kept, frames, cfg)
    log(f"  written {csv_path} (the GUI's format; cam1, cam2, cam3 = the GUI's cameras 0, 1, 2)")

    lib = case.library()
    from modules.camera_calibration.wand_calibration.wand_calibrator import write_detection_record
    rec = write_detection_record(csv_path, detection_record(cfg, "gui"))
    log(f"  written {rec['file'] if rec else 'NO detection record'} (the GUI's camera numbers)")

    uv = case.uv_array(kept, frames, cams)
    job = case.make_job(lib, cfg, frames, uv)
    job.pixel_mm = [float(case.cam_cfg(cfg, c)["pixel_mm"]) for c in cams]     # one per camera (GUI change, 8 Oct)
    job.detection = {**detection_record(cfg, "case"), "file": rec["file"] if rec else None}
    log(f"  calibration: {len(frames)} frames, wand {job.wand_length} mm, tube radii {job.radii} mm, n {job.n}, "
        f"fit_k1 {job.fit_k1}, fit_L {job.fit_L}, noise weights {job.noise_weights}, n_final {job.n_final}")
    sol, result, secs = case.run_calibration(lib, job, cfg["out_dir"], log)
    log(f"  done in {secs:.0f} s; written to {cfg['out_dir']}")
    for line in lib.cg.summary_lines(result):
        log("  " + line)
    log("")
    with open(os.path.join(cfg["out_dir"], "calibration_report.txt"), encoding="utf-8") as fh:
        for line in fh.read().splitlines():
            log("  | " + line)
    log.save({"summary": case.summary(result, cfg), "selection": {k: v for k, v in sel.items() if k != "frames"},
              "seconds": secs, "checks": result.get("checks"), "page": result.get("page")})


if __name__ == "__main__":
    main()
