"""C1-H (plan: 2026-10-08-plan.md, Phase C): each camera's ball images predicted from the other two, with C1's cameras.

    python tests/c1h_third_camera.py          (after calibrate.py and c4_track_camfiles.py)

For every frame all three cameras kept: each ball's lines of sight from two cameras (the kept detections; the
tracking camFiles of C4 in pyOpenLPT, build p4c as for T1), triangulated (myMATH.triangulation: the point nearest
both lines, and the RMS distance of the lines from it), then projected into the third camera and compared with its
detection.  Per held-out camera: the distance's RMS, median and 95th percentile against that camera's residual RMS
in the fit; per pair: the triangulation error.  The third camera's distance is the error a three-camera match must
allow (StereoMatch's tol_2d_px): it adds the pair's triangulation error, seen from the third camera, to the
detection's own.
Writes c1h_third_camera_out.txt and c1h_third_camera.json beside this script.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASE_DIR = os.path.dirname(HERE)
sys.path.insert(0, CASE_DIR)
import case  # noqa: E402
import t1_setup as ts  # noqa: E402

import numpy as np  # noqa: E402


def main():
    cfg = case.load_config()
    log = case.Log(os.path.join(HERE, "c1h_third_camera"))
    lpt = ts.lib()
    cams = {c: lpt.loadCameraModel(f) for c, f in zip(ts.CAM_IDS, ts.copy_cameras())}
    _dets, kept, _reasons, _medians, _frames = case.load_selection(cfg)
    ids = case.cam_ids(cfg)
    frames = case.frames_in_all(kept, ids)
    with open(os.path.join(os.path.dirname(HERE), "calibrate.json"), encoding="utf-8") as fh:
        fit = {c["name"]: c for c in json.load(fh)["summary"]["cameras"]}
    log(f"C1-H: {len(frames)} frames kept in all three cameras; cameras {ts.CAMS}")
    # lines of sight of every kept ball, per camera
    rays = {}
    for c in ids:
        pts = [lpt.Pt2D(float(kept[(f, c)][e][0]), float(kept[(f, c)][e][1])) for f in frames for e in (0, 1)]
        res = cams[c].lineOfSightBatchStatus(pts)
        rays[c] = res
    out = {"frames": len(frames), "held_out": {}, "pairs": {}}
    for k in ids:
        a, b = [c for c in ids if c != k]
        X, err, ok = [], [], []
        for i in range(2 * len(frames)):
            ra, rb = rays[a][i], rays[b][i]
            if not (ra[0] and rb[0]):
                ok.append(False)
                X.append(lpt.Pt3D(0.0, 0.0, 0.0))
                err.append(np.nan)
                continue
            p, e = lpt.myMATH.triangulation([ra[1], rb[1]])
            X.append(p)
            err.append(e)
            ok.append(True)
        proj = cams[k].projectBatchStatus(X)
        d, uvk = [], []
        for i, (f, e) in enumerate((f, e) for f in frames for e in (0, 1)):
            if not ok[i] or not proj[i][0]:
                continue
            q = proj[i][1]
            u, v = kept[(f, k)][e][0], kept[(f, k)][e][1]
            d.append(np.hypot(q[0] - u, q[1] - v))
        d = np.array(d)
        err = np.array(err, float)
        rec = {"pair": [a, b], "n": int(len(d)), "rms_px": float(np.sqrt(np.mean(d ** 2))),
               "median_px": float(np.median(d)), "p95_px": float(np.percentile(d, 95)),
               "p99_px": float(np.percentile(d, 99)), "share_within_1px": float(np.mean(d <= 1.0)),
               "share_within_1_5px": float(np.mean(d <= 1.5)),
               "fit_rms_px": fit[f"cam{k}"]["rms_px"], "triangulation_error_mm_median": float(np.nanmedian(err)),
               "failed": int(2 * len(frames) - len(d))}
        out["held_out"][f"cam{k}"] = rec
        log(f"  cam{k} predicted from cams {a} and {b}: {rec['n']} balls; distance RMS {rec['rms_px']:.3f} px, median "
            f"{rec['median_px']:.3f}, 95 % {rec['p95_px']:.3f}, 99 % {rec['p99_px']:.3f}; within 1 px "
            f"{100 * rec['share_within_1px']:.1f} %, within 1.5 px {100 * rec['share_within_1_5px']:.1f} % (the fit's "
            f"RMS for cam{k}: {rec['fit_rms_px']:.3f} px); pair's triangulation error median "
            f"{1e3 * rec['triangulation_error_mm_median']:.2f} um; failed {rec['failed']}")
    log.save(out)


if __name__ == "__main__":
    main()
