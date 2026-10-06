"""P0 (before the plan): T0's calibration on T2's surveyed frames.  Read-only.

    python tests/p0_t0_cameras.py      (after tests/p0_survey.py)

T0's delivered cameras (S2) and its variants h1, h2, q1-q4 (cam1 f 32,800-40,200 px), on T0's 1,519 kept frames and
on the frames the T2 survey kept, by free triangulation (as T0's D8):
  - the wand-length error per frame, against 10 mm;
  - the residual of a maximum-likelihood triangulation of each ball from both cameras (the cameras' disagreement,
    1 degree of freedom per ball).  Per camera: its mean (a camera that moved between the runs leaves a common
    offset) and the RMS about the mean, by ball and by how end-on the wand is to that camera.
And where the balls are, in T0's frame (y = the tube's axis, z towards cam1): the depth along each camera's line of
sight, the extent, and the share of end-on poses, T0 against T2.
Uses the T0 case's code (cases/cyl_20261006_T0_wand/case.py) unchanged.
Writes p0_t0_cameras_out.txt and p0_t0_cameras.json beside itself.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASE_DIR = os.path.dirname(HERE)
T0_DIR = os.path.join(os.path.dirname(CASE_DIR), "cyl_20261006_T0_wand")
sys.path.insert(0, T0_DIR)
import case  # noqa: E402  (the T0 case's code)

import numpy as np  # noqa: E402

VARIANTS = ["S2", "h1", "h2", "q1", "q2", "q3", "q4"]


def load_cams(lib, out_dir, name):
    folder = out_dir if name == "S2" else os.path.join(out_dir, "checks", name)
    with open(os.path.join(folder, "calibration_result.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    cams = [lib.cm.Camera(np.array(c["R"]), np.array(c["centre_mm"]), c["f_px"], c["cx"], c["cy"], c["width"],
                          c["height"], np.array(c["dist"])) for c in res["cameras"]]
    tube = lib.cm.Tube(tuple(res["tube"]["radii_mm"]), tuple(res["tube"]["n"]))
    return res, cams, tube


def triangulate(lib, cams, tube, uv, L):
    """Free triangulation (least squares, then maximum likelihood) of every kept frame.  Returns the length error
    (um) per frame, the residuals (px) per observation, the observations, and the balls' least-squares positions."""
    cw = lib.cw
    obs = cw.observations(cw.WandData(uv, L, [(1280, 800)] * len(cams)))
    XA, XB, cnt = cw.triangulate(cams, tube, obs, np.arange(len(obs)))
    Lm = np.linalg.norm(XB - XA, axis=1)
    Lm[np.any(cnt < 2, axis=1)] = np.nan
    point = obs.frame * 2 + obs.end
    X0 = np.empty((2 * obs.n_frames, 3))
    X0[0::2], X0[1::2] = XA, XB
    ok = np.all(np.isfinite(X0), axis=1)
    X0[~ok] = 0.0
    _X, r, _cov = cw.triangulate_ml(cams, tube, obs.cam, obs.uv, point, X0)
    return 1e3 * (Lm - L), r, obs, XA, XB


def rms(a):
    a = np.asarray(a)
    return float(np.sqrt(np.mean(a ** 2))) if a.size else float("nan")


def main():
    cfg2 = case.load_config(os.path.join(CASE_DIR, "config.json"))
    cfg0 = case.load_config(os.path.join(T0_DIR, "config.json"))
    log = case.Log(os.path.join(HERE, "p0_t0_cameras"))
    lib = case.library()
    L = float(cfg0["wand"]["length_mm"])
    _d, kept0, _r, _m, frames0 = case.load_selection(cfg0)
    with open(os.path.join(HERE, "p0_survey.json"), encoding="utf-8") as fh:
        sv = json.load(fh)
    dets2 = {tuple(int(x) for x in k.split("_")): v for k, v in sv["dets"].items()}
    kept2, _r2, _m2 = case.classify(dets2, cfg2)
    frames2 = case.frames_in_all(kept2, case.cam_ids(cfg2))
    runs = {"T0": case.uv_array(kept0, frames0, [1, 2]), "T2": case.uv_array(kept2, frames2, [1, 2])}
    log(f"P0: T0's cameras on T0's {len(frames0)} kept frames and on the T2 survey's {len(frames2)} "
        f"(every {sv['step']}th frame)")
    out = {"frames": {"T0": len(frames0), "T2": len(frames2)}, "variants": {}, "offsets": {}, "coverage": {}}

    log("\nWand length and reprojection, free triangulation (no wand length, no fitted poses)")
    pos = {}
    for name in VARIANTS:
        res, cams, tube = load_cams(lib, cfg0["out_dir"], name)
        f = [c["f_px"] for c in res["cameras"]]
        rec = {"f_px": f}
        for run, uv in runs.items():
            dL, r, obs, XA, XB = triangulate(lib, cams, tube, uv, L)
            d = dL[np.isfinite(dL)]
            rn = np.linalg.norm(r, axis=1)
            rec[run] = {"length_mean_um": float(np.mean(d)), "length_rms_um": rms(d),
                        "reproj_rms_px": rms(rn[np.isfinite(rn)]), "frames": int(len(d))}
            log(f"  {name:>3s} (f {f[0]:6.0f}, {f[1]:6.0f}) on {run}: length error mean {np.mean(d):+6.1f} um, "
                f"RMS {rms(d):5.1f} um; reprojection RMS {rec[run]['reproj_rms_px']:.3f} px")
            if name == "S2":
                pos[run] = (r, obs, XA, XB)
        out["variants"][name] = rec

    res, _cams, _tube = load_cams(lib, cfg0["out_dir"], "S2")
    log("\nT0's S2 cameras: the residual per camera (px): mean (u, v), RMS about the mean; RMS by ball and by the "
        "wand's angle to the camera's line of sight")
    for run, (r, obs, XA, XB) in pos.items():
        w = XB - XA
        w = w / np.linalg.norm(w, axis=1)[:, None]
        for j, c in enumerate(res["cameras"]):
            v = np.array(c["viewing_direction"], float)
            ang = np.degrees(np.arccos(np.clip(np.abs(w @ v), 0.0, 1.0)))[obs.frame]
            m = obs.cam == j
            mean = r[m].mean(axis=0)
            about = float(np.sqrt(np.mean(np.sum((r[m] - mean) ** 2, axis=1))))
            rec = {"mean_px": mean.tolist(), "rms_about_mean_px": about}
            parts = []
            for e, nm in ((0, "A"), (1, "B")):
                me = m & (obs.end == e)
                rec[f"rms_ball_{nm}"] = rms(np.linalg.norm(r[me], axis=1))
                parts.append(f"ball {nm} {rec[f'rms_ball_{nm}']:.3f}")
            for lo, hi in ((0, 45), (45, 70), (70, 91)):
                ma = m & (ang >= lo) & (ang < hi)
                rec[f"rms_angle_{lo}_{hi}"] = rms(np.linalg.norm(r[ma], axis=1))
                parts.append(f"{lo}-{min(hi, 90)} deg {rec[f'rms_angle_{lo}_{hi}']:.3f} (n {int(ma.sum())})")
            out["offsets"][f"{run}_{c['name']}"] = rec
            log(f"  {run} {c['name']}: mean ({mean[0]:+.3f}, {mean[1]:+.3f}), RMS about it {about:.3f}; "
                + "; ".join(parts))
            if run == "T2":                      # before and after the light's step (frames 2,000-3,000)
                fid = np.array(frames2)[obs.frame]
                for tag, sel in (("before frame 2500", fid < 2500), ("after frame 3000", fid > 3000)):
                    ms = m & sel
                    mv = r[ms].mean(axis=0)
                    rec[f"mean_px_{tag.replace(' ', '_')}"] = mv.tolist()
                    log(f"      {tag}: mean ({mv[0]:+.3f}, {mv[1]:+.3f}) over {int(ms.sum()) // 2} frames")

    log("\nWhere the balls are (T0's frame, mm; 5th-95th percentiles), from T0's S2 cameras")
    for c in res["cameras"]:
        v = np.array(c["viewing_direction"], float)
        C = np.array(c["centre_mm"], float)
        for run, (_r, _obs, XA, XB) in pos.items():
            X = np.vstack([XA, XB])
            X = X[np.all(np.isfinite(X), axis=1)]
            depth = (X - C) @ v
            w = XB - XA
            ok = np.all(np.isfinite(w), axis=1)
            w = w[ok] / np.linalg.norm(w[ok], axis=1)[:, None]
            ang = np.degrees(np.arccos(np.clip(np.abs(w @ v), 0.0, 1.0)))
            p5, p95 = np.percentile(depth, [5, 95])
            rec = {"depth_p5_mm": float(p5), "depth_p95_mm": float(p95), "angle_median_deg": float(np.median(ang)),
                   "share_within_45_deg": float(np.mean(ang < 45))}
            out["coverage"][f"{run}_{c['name']}"] = rec
            log(f"  {c['name']} {run}: depth along its line of sight {p5:7.1f} .. {p95:7.1f} mm (span {p95 - p5:4.1f} mm);"
                f" the wand's angle to it: median {np.median(ang):4.1f} deg, within 45 deg in "
                f"{100 * np.mean(ang < 45):4.1f} % of frames")
    for run, (_r, _obs, XA, XB) in pos.items():
        X = np.vstack([XA, XB])
        X = X[np.all(np.isfinite(X), axis=1)]
        lo, hi = np.percentile(X, 5, axis=0), np.percentile(X, 95, axis=0)
        out["coverage"][f"{run}_extent"] = {"p5": lo.tolist(), "p95": hi.tolist()}
        log(f"  {run} extent: x {lo[0]:6.1f} .. {hi[0]:5.1f}, y (axis) {lo[1]:6.1f} .. {hi[1]:5.1f}, "
            f"z {lo[2]:6.1f} .. {hi[2]:5.1f} mm")
    log.save(out)


if __name__ == "__main__":
    main()
