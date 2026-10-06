"""D5 (diagnosis after D3): is the frame-to-frame jitter the wand's motion or the cameras' own?

    python tests/d5_jitter_cameras.py [RESULT_FOLDER]      (default: config out_dir, S2's result)

D3 found both balls of a camera jumping together by about 0.5 px (robust; 1.0-1.3 px RMS in v) from one frame to
the next, beyond a smooth (quadratic) track, while their distance stays within 0.16-0.18 px.  If the wand really
moves like that, both cameras see the jump in the same frame (the cameras are synchronised, D1); if a camera vibrates
(or its image jumps), its jumps are its own.  This correlates the two cameras' jitter (the mean of the two balls'
residual from the 7-frame quadratic, per coordinate) frame by frame, and compares S2's residuals in u and v, and the
fraction of S2's residual that the same-frame jitter accounts for.
Writes d5_jitter_cameras_out.txt and d5_jitter_cameras.json beside this script.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

import numpy as np  # noqa: E402

W = 3


def jitter(kept, c):
    """{frame: (ju, jv)}: mean of the two balls' residuals from a quadratic over frames f-3..f+3 at f."""
    t = np.arange(-W, W + 1, dtype=float)
    A = np.column_stack([np.ones_like(t), t, t * t])
    P = np.linalg.pinv(A)
    fs = {f for (f, cc) in kept if cc == c}
    out = {}
    for f in sorted(fs):
        win = [f + d for d in range(-W, W + 1)]
        if not all(g in fs for g in win):
            continue
        m = np.array([0.5 * (np.array(kept[(g, c)][0][:2]) + np.array(kept[(g, c)][1][:2])) for g in win])
        res = m[W] - (A @ (P @ m))[W]
        out[f] = res
    return out


def main():
    cfg = case.load_config()
    folder = sys.argv[1] if len(sys.argv) > 1 else cfg["out_dir"]
    log = case.Log(os.path.join(HERE, "d5_jitter_cameras"))
    _dets, kept, _reasons, _medians, _frames = case.load_selection(cfg)
    j1, j2 = jitter(kept, 1), jitter(kept, 2)
    common = sorted(set(j1) & set(j2))
    a = np.array([j1[f] for f in common])
    b = np.array([j2[f] for f in common])
    out = {"frames": len(common)}
    log(f"D5: jitter of the wand's midpoint (beyond a 7-frame quadratic) in both cameras, {len(common)} frames")
    for k, name in ((0, "u"), (1, "v")):
        r = float(np.corrcoef(a[:, k], b[:, k])[0, 1])
        # robust: the same on the frames within 3 robust sigmas in both
        def rs(x):
            return 1.4826 * float(np.median(np.abs(x - np.median(x))))
        ok = (np.abs(a[:, k]) < 3 * rs(a[:, k])) & (np.abs(b[:, k]) < 3 * rs(b[:, k]))
        r_core = float(np.corrcoef(a[ok, k], b[ok, k])[0, 1])
        slope = float(np.polyfit(a[ok, k], b[ok, k], 1)[0])
        out[name] = {"corr": r, "corr_core": r_core, "slope_cam2_on_cam1": slope,
                     "rms_cam1": float(np.sqrt(np.mean(a[:, k] ** 2))), "rms_cam2": float(np.sqrt(np.mean(b[:, k] ** 2))),
                     "robust_cam1": rs(a[:, k]), "robust_cam2": rs(b[:, k])}
        log(f"  {name}: jitter RMS cam1 {out[name]['rms_cam1']:.3f} px (robust {out[name]['robust_cam1']:.3f}), cam2 "
            f"{out[name]['rms_cam2']:.3f} (robust {out[name]['robust_cam2']:.3f}); correlation between the cameras "
            f"{r:+.3f} (core {r_core:+.3f}, slope cam2 on cam1 {slope:+.3f})")
    with open(os.path.join(folder, "calibration_data.json"), encoding="utf-8") as fh:
        ob = json.load(fh)["observations"]
    fid, cam = np.array(ob["frame_id"]), np.array(ob["cam_id"])
    du = np.array([np.nan if x is None else x for x in ob["du"]], float)
    dv = np.array([np.nan if x is None else x for x in ob["dv"]], float)
    act = np.array(ob["active"], bool) & np.isfinite(du)
    for c, jj in ((1, j1), (2, j2)):
        s = act & (cam == c)
        log(f"  S2's residuals, cam{c}: RMS u {np.sqrt(np.mean(du[s] ** 2)):.3f} px, v {np.sqrt(np.mean(dv[s] ** 2)):.3f} px")
        # residual (model - detection) against the jitter (detection - smooth): expect slope -1 if the jitter is the
        # camera's own (the model cannot follow it) and about 0 if the wand really moved (the model follows it)
        sel = [i for i in np.flatnonzero(s) if int(fid[i]) in jj]
        J = np.array([jj[int(fid[i])] for i in sel])
        for k, name, d in ((0, "u", du), (1, "v", dv)):
            y = d[sel]
            sl = float(np.polyfit(J[:, k], y, 1)[0])
            log(f"    residual {name} against the same frame's jitter {name}: slope {sl:+.3f} "
                "(-1: the model does not follow the jump, the camera's own; 0: the wand moved)")
            out[f"cam{c}_slope_{name}"] = sl
    log.save(out)


if __name__ == "__main__":
    main()
