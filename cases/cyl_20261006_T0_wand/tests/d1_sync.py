"""D1 (diagnosis after T2): are the two cameras' frames taken at the same time?

    python tests/d1_sync.py [RESULT_FOLDER]      (default: config out_dir, S2's result)

S2's residuals are large (0.45 px RMS) and the wand moves about 20 px between frames.  If a camera's exposures are
offset in time by dt (in frames) from the other's, each detection sits where the ball was dt frames away from the
common instant, so its residual (model - detection) is about -dt times the ball's image velocity, with opposite
signs in the two cameras.  This regresses each camera's residuals (du, dv) on the ball's image velocity (centred
difference of the kept detections at frames f-1 and f+1, px per frame): residual = -k * velocity + c.  The slope k is
the camera's offset from the fit's common instant, in frames; k1 - k2 is the offset between the cameras.  Also: the
residual RMS of the frames where the wand is slow against those where it is fast.
Writes d1_sync_out.txt and d1_sync.json beside this script.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

import numpy as np  # noqa: E402


def main():
    cfg = case.load_config()
    folder = sys.argv[1] if len(sys.argv) > 1 else cfg["out_dir"]
    log = case.Log(os.path.join(HERE, "d1_sync"))
    _dets, kept, _reasons, _medians, _frames = case.load_selection(cfg)
    with open(os.path.join(folder, "calibration_data.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    ob = data["observations"]
    fid, cam, end = np.array(ob["frame_id"]), np.array(ob["cam_id"]), np.array(ob["end"])
    du = np.array([np.nan if x is None else x for x in ob["du"]], float)
    dv = np.array([np.nan if x is None else x for x in ob["dv"]], float)
    act = np.array(ob["active"], bool) & np.isfinite(du)
    vel = np.full((len(fid), 2), np.nan)
    for i in range(len(fid)):
        f, c, e = int(fid[i]), int(cam[i]), int(end[i])
        if (f - 1, c) in kept and (f + 1, c) in kept:
            p0, p1 = np.array(kept[(f - 1, c)][e][:2]), np.array(kept[(f + 1, c)][e][:2])
            vel[i] = 0.5 * (p1 - p0)
    ok = act & np.all(np.isfinite(vel), axis=1)
    log(f"D1: residuals against the ball's image velocity; {int(ok.sum())} of {int(act.sum())} used detections have "
        "both neighbouring frames kept")
    out = {}
    for c in sorted(set(cam)):
        s = ok & (cam == c)
        # one slope for both coordinates: [du; dv] = -k [vu; vv] + [cu; cv]
        y = np.concatenate([du[s], dv[s]])
        X = np.zeros((2 * s.sum(), 3))
        X[:s.sum(), 0] = -vel[s, 0]
        X[s.sum():, 0] = -vel[s, 1]
        X[:s.sum(), 1] = 1.0
        X[s.sum():, 2] = 1.0
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ coef
        s2 = float(r @ r) / (len(y) - 3)
        se = math.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
        rms_before = float(np.sqrt(np.mean(y ** 2)))
        rms_after = float(np.sqrt(np.mean(r ** 2)))
        speed = np.linalg.norm(vel[s], axis=1)
        q = np.percentile(speed, [33.3, 66.7])
        res2 = du[s] ** 2 + dv[s] ** 2
        slow = float(np.sqrt(np.mean(res2[speed <= q[0]])))
        fast = float(np.sqrt(np.mean(res2[speed >= q[1]])))
        out[f"cam{c}"] = {"n": int(s.sum()), "k_frames": float(coef[0]), "k_se": se, "rms_before_px": rms_before,
                          "rms_after_px": rms_after, "speed_median_px_per_frame": float(np.median(speed)),
                          "rms_slow_third_px": slow, "rms_fast_third_px": fast}
        log(f"  cam{c}: n {int(s.sum())}; offset k = {coef[0]:+.4f} +- {se:.4f} frames ({coef[0] / se:+.1f} SE); "
            f"residual RMS per coordinate {rms_before:.3f} -> {rms_after:.3f} px with the velocity term; ball speed "
            f"median {np.median(speed):.1f} px/frame; residual RMS (2-D) slowest third {slow:.3f} px, fastest third "
            f"{fast:.3f} px")
    ks = [out[k]["k_frames"] for k in sorted(out)]
    if len(ks) == 2:
        dk = ks[0] - ks[1]
        se = math.hypot(*[out[k]["k_se"] for k in sorted(out)])
        out["offset_cam1_minus_cam2_frames"] = dk
        out["offset_se"] = se
        log(f"  offset between the cameras (cam1 - cam2): {dk:+.4f} +- {se:.4f} frames")
    log.save(out)


if __name__ == "__main__":
    main()
