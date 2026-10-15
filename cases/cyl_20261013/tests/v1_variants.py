"""V1 (diagnosis of C1's 0.98 px; plan section 3): would another setting give a better calibration?

    python tests/v1_variants.py [NAME ...]      (default: all; after calibrate.py and c4_track_camfiles.py)

Three full calibrations (cylinder_wand_gui.run with config.json's settings and one change each), each in its own
process, written to <out_dir>/checks/<name>/:
  a145:  the selection with 8 Oct's aspect limit 1.15-1.45 (plan D2 raised it to 1.60);
  wball: noise weights per camera and ball (the GUI's "Per camera and ball", cylinder_wand's "camera_end"): the small
         ball's centres are noisier (C1: its residual RMS 1.20 / 0.67 / 1.25 px against 0.80 / 0.60 / 1.06);
  k1:    the lens's radial distortion k1 fitted per camera (the GUI fits it when asked).
Judged on held-out cameras, which a fit cannot tune itself to: every camera's balls predicted from the other two
cameras (C1-H's method: lines of sight, myMATH.triangulation, projection; library build p4c, as C1-H), for C1 and for
each variant on the same detections: the frames that all three cameras keep under 8 Oct's rule (a subset of every
variant's frames).
Writes v1_variants_out.txt and v1_variants.json beside this script.
"""
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

import numpy as np  # noqa: E402

BUILD = os.path.join(case.REPO, "build-stb", "p4c", "out")


def variant_cfg(name):
    cfg = copy.deepcopy(case.load_config())
    if name == "a145":
        cfg["accept"]["aspect_range"] = [1.15, 1.45]
    elif name == "wball":
        cfg["calibrate"]["noise_weights"] = "camera_end"
    elif name == "k1":
        cfg["calibrate"]["fit_k1"] = True
    else:
        raise ValueError(name)
    return cfg


VARIANTS = ("a145", "wball", "k1")


def run_variant(name):
    cfg = variant_cfg(name)
    _dets, kept, _reasons, _medians, frames = case.load_selection(cfg)
    cams = case.cam_ids(cfg)
    lib = case.library()
    job = case.make_job(lib, cfg, frames, case.uv_array(kept, frames, cams))
    job.pixel_mm = [float(case.cam_cfg(cfg, c)["pixel_mm"]) for c in cams]
    folder = os.path.join(cfg["out_dir"], "checks", name)
    try:
        _sol, result, secs = case.run_calibration(lib, job, folder, log=lambda m: None)
    except Exception as e:                               # report it, keep the other variants
        return name, {"error": f"{type(e).__name__}: {e}", "frames": len(frames)}
    s = case.summary(result, cfg)
    s["seconds"] = secs
    s["folder"] = folder
    return name, s


def held_out(lpt, files, kept, frames, ids):
    """Each camera's kept balls predicted from the other two cameras: RMS, median, 95 % of the distance (px)."""
    cams = {c: lpt.loadCameraModel(files[c]) for c in ids}
    rays = {c: cams[c].lineOfSightBatchStatus([lpt.Pt2D(float(kept[(f, c)][e][0]), float(kept[(f, c)][e][1]))
                                                for f in frames for e in (0, 1)]) for c in ids}
    out = {}
    for k in ids:
        a, b = [c for c in ids if c != k]
        X, ok = [], []
        for i in range(2 * len(frames)):
            ra, rb = rays[a][i], rays[b][i]
            if not (ra[0] and rb[0]):
                X.append(lpt.Pt3D(0.0, 0.0, 0.0))
                ok.append(False)
                continue
            p, _e = lpt.myMATH.triangulation([ra[1], rb[1]])
            X.append(p)
            ok.append(True)
        proj = cams[k].projectBatchStatus(X)
        d = []
        for i, (f, e) in enumerate((f, e) for f in frames for e in (0, 1)):
            if ok[i] and proj[i][0]:
                q = proj[i][1]
                d.append(np.hypot(q[0] - kept[(f, k)][e][0], q[1] - kept[(f, k)][e][1]))
        d = np.array(d)
        out[f"cam{k}"] = {"n": int(len(d)), "rms_px": float(np.sqrt(np.mean(d ** 2))), "median_px": float(np.median(d)),
                          "p95_px": float(np.percentile(d, 95)), "failed": int(2 * len(frames) - len(d))}
    return out


def main():
    names = sys.argv[1:] or list(VARIANTS)
    cfg = case.load_config()
    log = case.Log(os.path.join(HERE, "v1_variants"))
    with open(os.path.join(os.path.dirname(HERE), "calibrate.json"), encoding="utf-8") as fh:
        ref = json.load(fh)["summary"]
    log(f"V1: {len(names)} variants of C1 (RMS {ref['rms_px']:.4f} px, {ref['frames']} frames)")
    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(max_workers=len(names)) as ex:
        for name, s in ex.map(run_variant, names):
            res[name] = s
    log(f"  the fits ran in {time.time() - t0:.0f} s")

    # the held-out comparison, on the frames all three cameras keep under 8 Oct's rule
    sys.path.insert(0, BUILD)
    import pyopenlpt as lpt  # noqa: E402
    ids = case.cam_ids(cfg)
    _d, kept45, _r, _m, _f = case.load_selection(variant_cfg("a145"))
    frames = case.frames_in_all(kept45, ids)
    log(f"  held-out comparison: {len(frames)} frames kept by all three cameras under the 1.45 rule; library "
        f"{lpt.__file__}")
    out = {"C1": {"summary": ref}, "variants": {}, "held_out_frames": len(frames)}
    sets = {"C1": os.path.join(cfg["out_dir"], "camFile")}
    for name in names:
        if "error" not in res[name]:
            sets[name] = os.path.join(res[name]["folder"], "camFile")
    for name, folder in sets.items():
        ho = held_out(lpt, {c: os.path.join(folder, f"cam{c}.txt") for c in ids}, kept45, frames, ids)
        s = ref if name == "C1" else res[name]
        rec = out["C1"] if name == "C1" else out["variants"].setdefault(name, {"summary": s})
        rec["held_out"] = ho
        cams = {c["name"]: c for c in s["cameras"]}
        log(f"  {name}: {s['frames']} frames, {s['active']}/{s['observations']} used, RMS {s['rms_px']:.4f} px "
            f"(cam1 {cams['cam1']['rms_px']:.3f}, cam2 {cams['cam2']['rms_px']:.3f}, cam3 {cams['cam3']['rms_px']:.3f}); "
            f"wand length RMS {s['wand_free_rms_um']:.1f} um"
            + (f"; {s['seconds']:.0f} s" if "seconds" in s else ""))
        log("      f: " + ", ".join(f"{c['name']} {c['f_px']:.0f} px ({c['distance_mm']:.0f} mm)" for c in s["cameras"])
            + ("; k1: " + ", ".join(f"{c['name']} {c['k1']:+.3f}" for c in s["cameras"]) if name == "k1" else ""))
        log("      held out: " + "; ".join(f"{k} RMS {v['rms_px']:.3f} px, median {v['median_px']:.3f}, 95 % "
                                         f"{v['p95_px']:.3f}" for k, v in ho.items()))
    for name in names:
        if "error" in res[name]:
            log(f"  {name}: FAILED: {res[name]['error']}")
            out["variants"][name] = res[name]
    log.save(out)


if __name__ == "__main__":
    main()
