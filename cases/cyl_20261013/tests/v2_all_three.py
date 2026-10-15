"""V2 (your question, 15 Oct: "what if you only use the frames that all cameras can see for calibration? will that
reduce the error?"): C1 refitted on the frames all three cameras keep, compared with C1.

    python tests/v2_all_three.py        (after calibrate.py and v1_variants.py)

all3: config.json's settings with accept.min_cameras = 3 (the GUI's own rule: it keeps a frame only when every camera
has a pair), written to <out_dir>/checks/all3/.  Compared with C1 (frames kept in at least two cameras) on:
  - the fit: RMS per camera, the wand length of a free triangulation over each fit's own frames, f;
  - the cameras' agreement: each camera predicted from the other two (V1's held_out, library p4c) on the 5,050 frames
    all three cameras keep under 8 Oct's rule, as V1;
  - the frames only two cameras keep, which all3 never saw: each ball triangulated from those two cameras
    (myMATH.triangulation), and the wand length's error |X_A - X_B| - L.  C1 fitted these frames, so the comparison
    favours C1 there.
Writes v2_all_three_out.txt and v2_all_three.json beside this script.
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import case  # noqa: E402
import v1_variants as v1  # noqa: E402

import numpy as np  # noqa: E402


def all3_cfg():
    cfg = copy.deepcopy(case.load_config())
    cfg["accept"]["min_cameras"] = 3
    return cfg


def two_camera_lengths(lpt, files, kept, frames2, ids, L):
    """Per pair of cameras: the wand length error (um) of the frames that pair alone keeps."""
    cams = {c: lpt.loadCameraModel(files[c]) for c in ids}
    out = {}
    allerr = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            fr = [f for f in frames2 if (f, a) in kept and (f, b) in kept]
            if not fr:
                continue
            rays = {c: cams[c].lineOfSightBatchStatus([lpt.Pt2D(float(kept[(f, c)][e][0]), float(kept[(f, c)][e][1]))
                                                        for f in fr for e in (0, 1)]) for c in (a, b)}
            err = []
            for j in range(len(fr)):
                P = []
                for e in (0, 1):
                    ra, rb = rays[a][2 * j + e], rays[b][2 * j + e]
                    if not (ra[0] and rb[0]):
                        break
                    p, _d = lpt.myMATH.triangulation([ra[1], rb[1]])
                    P.append(np.array([p[0], p[1], p[2]], float))
                if len(P) == 2:
                    err.append(1e3 * (float(np.linalg.norm(P[1] - P[0])) - L))
            err = np.array(err)
            allerr.append(err)
            out[f"cams {a}{b}"] = {"frames": len(fr), "n": int(len(err)), "rms_um": float(np.sqrt(np.mean(err ** 2))),
                                   "mean_um": float(err.mean()), "median_abs_um": float(np.median(np.abs(err))),
                                   "p95_abs_um": float(np.percentile(np.abs(err), 95))}
    e = np.concatenate(allerr)
    out["all"] = {"n": int(len(e)), "rms_um": float(np.sqrt(np.mean(e ** 2))), "mean_um": float(e.mean()),
                  "median_abs_um": float(np.median(np.abs(e))), "p95_abs_um": float(np.percentile(np.abs(e), 95))}
    return out


def run_all3():
    """The all-three fit, in its own process (the library build-py/Release; the comparisons use p4c, as C1-H)."""
    c3 = all3_cfg()
    cams = case.cam_ids(c3)
    _d, kept3, _r, _m, frames3 = case.load_selection(c3)
    lib = case.library()
    job = case.make_job(lib, c3, frames3, case.uv_array(kept3, frames3, cams))
    job.pixel_mm = [float(case.cam_cfg(c3, c)["pixel_mm"]) for c in cams]
    folder = os.path.join(c3["out_dir"], "checks", "all3")
    _sol, result, secs = case.run_calibration(lib, job, folder, log=lambda m: None)
    return case.summary(result, c3), secs, folder, len(frames3)


def main():
    cfg = case.load_config()
    log = case.Log(os.path.join(HERE, "v2_all_three"))
    cams = case.cam_ids(cfg)
    with open(os.path.join(os.path.dirname(HERE), "calibrate.json"), encoding="utf-8") as fh:
        ref = json.load(fh)["summary"]

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=1) as ex:
        s3, secs, folder, n3 = ex.submit(run_all3).result()
    log(f"V2: C1 refitted on the {n3} frames all three cameras keep (C1: {ref['frames']} frames in at least two)")
    out = {"C1": {"summary": ref}, "all3": {"summary": s3, "seconds": secs, "folder": folder}}

    # the comparisons (library p4c, as C1-H and V1)
    sys.path.insert(0, v1.BUILD)
    import pyopenlpt as lpt  # noqa: E402
    _d, kept45, _r, _m, _f = case.load_selection(v1.variant_cfg("a145"))
    frames_ho = case.frames_in_all(kept45, cams)
    _d, kept, _r, _m, frames_c1 = case.load_selection(cfg)
    frames2 = [f for f in frames_c1 if sum((f, c) in kept for c in cams) == 2]
    L = float(cfg["wand"]["length_mm"])
    sets = {"C1": os.path.join(cfg["out_dir"], "camFile"), "all3": os.path.join(folder, "camFile")}
    log(f"  held-out comparison on {len(frames_ho)} frames (V1's); two-camera frames: {len(frames2)} (C1 fitted them, "
        f"all3 did not)")
    for name, d in sets.items():
        files = {c: os.path.join(d, f"cam{c}.txt") for c in cams}
        s = ref if name == "C1" else s3
        ho = v1.held_out(lpt, files, kept45, frames_ho, cams)
        tw = two_camera_lengths(lpt, files, kept, frames2, cams, L)
        out[name].update({"held_out": ho, "two_camera_frames": tw})
        cs = {c["name"]: c for c in s["cameras"]}
        log(f"  {name}: {s['frames']} frames, {s['active']}/{s['observations']} used, RMS {s['rms_px']:.4f} px (cam1 "
            f"{cs['cam1']['rms_px']:.3f}, cam2 {cs['cam2']['rms_px']:.3f}, cam3 {cs['cam3']['rms_px']:.3f}); wand length "
            f"RMS {s['wand_free_rms_um']:.1f} um over its own frames"
            + (f"; {secs:.0f} s" if name == "all3" else ""))
        log("      f: " + ", ".join(f"{c['name']} {c['f_px']:.0f} px ({c['distance_mm']:.0f} mm)" for c in s["cameras"]))
        log("      held out: " + "; ".join(f"{k} RMS {v['rms_px']:.3f} px, median {v['median_px']:.3f}, 95 % "
                                         f"{v['p95_px']:.3f}" for k, v in ho.items()))
        log("      two-camera frames, wand length error: " + "; ".join(
            f"{k}: n {v['n']}, RMS {v['rms_um']:.1f} um, mean {v['mean_um']:+.1f}, median |e| {v['median_abs_um']:.1f}, "
            f"95 % {v['p95_abs_um']:.1f}" for k, v in tw.items()))
    log.save(out)


if __name__ == "__main__":
    main()
