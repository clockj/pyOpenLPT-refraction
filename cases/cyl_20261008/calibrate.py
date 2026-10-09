"""C1: the delivered calibration of T0 (plan: 2026-10-08-plan.md, Phase C; the 7 Oct case's calibrate.py, adapted
to three cameras of two kinds).

    python calibrate.py

Reads C1's detections, keeps the camera-frames that pass config.accept (case.classify) and the frames that at least
config.accept.min_cameras cameras keep (the others are "not seen" in the fit), writes them in the GUI's format
(<out_dir>/wand_points.csv) with the selection's counts (<out_dir>/detection/selection.json), and runs
cylinder_wand_gui.run (cylinder_wand.calibrate as the GUI's "Pinhole+Cylinder" runs it: tube and wand of
config.json, k1 off, per-camera noise weights, two starts carried; each camera's pixel size for f in mm).
cylinder_wand_gui.write_outputs then writes the GUI's files into <out_dir>: camFile/cam1.txt, cam2.txt, cam3.txt
(PINCYL), calibration_result.json, calibration_report.txt, calibration_data.json, calibration_report.html (the
results page).
Writes calibrate_out.txt and calibrate.json (the summary the tests compare against) beside this script.
"""
import json
import os
from collections import Counter

import case


def main():
    cfg = case.load_config()
    log = case.Log(os.path.join(case.HERE, "calibrate"))
    cams = case.cam_ids(cfg)
    dets, kept, reasons, medians, frames = case.load_selection(cfg)
    log("S2: the delivered calibration")
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
    uv = case.uv_array(kept, frames, cams)
    job = case.make_job(lib, cfg, frames, uv)
    job.pixel_mm = [float(case.cam_cfg(cfg, c)["pixel_mm"]) for c in cams]     # one per camera (GUI change, 8 Oct)
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
