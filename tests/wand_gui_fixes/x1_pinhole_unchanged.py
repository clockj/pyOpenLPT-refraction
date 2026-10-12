"""X1 (B11, B12): the Pinhole calibration of cameras of one size is unchanged by the fixes.

    python x1_pinhole_unchanged.py

The calibrator from git HEAD (before_lib/wand_calibrator.py) and the fixed one, called as the GUI's CalibrationWorker
calls them (calibrate_wand / run_precalibration_check with the wand length and the start focal length), on W10's
synthetic rig (w9_w10_pinhole.py: three OpenCV cameras of 1280 x 800 px, f 2,000 px, 300 frames, 0.1 px noise):
  - the full calibration with the three cameras (Phases 1-3);
  - with its first two cameras (the two-camera path: Phase 1 only);
  - Precalibrate's check with the three cameras.
Each is run twice with the fixed code (a control: the calibration must give the same answer twice for the comparison
to mean anything).  Passes when every camera's parameters (R, T, K, distortion, image size), the triangulated wand
ends, the per-frame errors (camera errors, length error, triangulation errors) and the message are identical to the
bit.
Writes x1_pinhole_unchanged_out.txt and .json.
"""
import common as C  # noqa: I001

import importlib.util  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from w9_w10_pinhole import EQUAL, gui_points, rig  # noqa: E402

log = C.Log("x1_pinhole_unchanged")


def load_old():
    spec = importlib.util.spec_from_file_location("wand_calibrator_before", C.HERE / "before_lib" / "wand_calibrator.py")
    old = importlib.util.module_from_spec(spec)
    sys.modules["wand_calibrator_before"] = old
    spec.loader.exec_module(old)
    return old


def run(module, uv, cams, precheck=False):
    wc = module.WandCalibrator()
    wp = gui_points(uv)
    wc.wand_points = {f: {c: v for c, v in fr.items() if c in cams} for f, fr in wp.items()}
    wc.wand_points_filtered = None
    wc.camera_settings = {c: {"focal": 2000.0, "width": EQUAL[c][0], "height": EQUAL[c][1]} for c in cams}
    wc.image_size = (EQUAL[cams[0]][1], EQUAL[cams[0]][0])          # as Run Calibration sets it
    wc.dist_coeff_num = 0
    t0 = time.time()
    if precheck:
        ok, msg, _res = wc.run_precalibration_check(wand_length=10.0, init_focal_length=2000.0, callback=None)
    else:
        ok, msg, _res = wc.calibrate_wand(wand_length=10.0, init_focal_length=2000.0, callback=None)
    return wc, ok, msg, time.time() - t0


def same(a, b, path, out):
    """Bitwise comparison of nested dicts / arrays / numbers; differences appended to out."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'new' if k in b else 'old'}")
            else:
                same(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and not isinstance(a, str):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} against {len(b)}")
        for i, (u, v) in enumerate(zip(a, b)):
            same(u, v, f"{path}[{i}]", out)
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        x, y = np.asarray(a), np.asarray(b)
        if x.shape != y.shape or x.tobytes() != y.tobytes():
            out.append(f"{path}: arrays differ" + (f" (largest {np.max(np.abs(x - y)):.3e})" if x.shape == y.shape
                                                   else f" ({x.shape} against {y.shape})"))
    elif a != b and not (isinstance(a, float) and isinstance(b, float) and a != a and b != b):
        out.append(f"{path}: {str(a)[:60]!r} against {str(b)[:60]!r}")
    return out


def state(wc, msg):
    pfe = {f: {k: v for k, v in e.items() if k != "proj_pts"} for f, e in (wc.per_frame_errors or {}).items()}
    return {"final_params": wc.final_params, "points_3d": wc.points_3d, "per_frame_errors": pfe, "msg": msg}


def main():
    old = load_old()
    import modules.camera_calibration.wand_calibration.wand_calibrator as new
    _true, uv, _A, _B, on = rig(np.random.default_rng(1314), EQUAL)
    log(f"X1: W10's rig, 3 cameras of 1280 x 800 px, {uv.shape[0]} frames (on the sensors: {on})")
    out = {}
    for label, cams, pre in (("three cameras, calibration", (0, 1, 2), False),
                             ("two cameras, calibration", (0, 1), False),
                             ("three cameras, Precalibrate's check", (0, 1, 2), True)):
        r_old = run(old, uv, cams, pre)
        r_new = run(new, uv, cams, pre)
        r_ctl = run(new, uv, cams, pre)
        ok_all = r_old[1] and r_new[1] and r_ctl[1]
        d_ctl = same(state(r_new[0], r_new[2]), state(r_ctl[0], r_ctl[2]), "", [])
        d = same(state(r_old[0], r_old[2]), state(r_new[0], r_new[2]), "", [])
        f = [float(p["K"][0, 0]) for _c, p in sorted(r_new[0].final_params.items())]
        log.check(f"{label}: the fixed code gives git HEAD's answer to the bit (control: the same twice)",
                  ok_all and not d_ctl and not d,
                  f"ok {r_old[1]}/{r_new[1]}/{r_ctl[1]}; control {len(d_ctl)} differences, against HEAD {len(d)}; "
                  f"f {', '.join(f'{v:,.2f}' for v in f)}; {r_old[3]:.0f} s old, {r_new[3]:.0f} s new")
        for line in (d_ctl + d)[:10]:
            log(f"      {line}")
        out[label] = {"differences": d, "control_differences": d_ctl, "f_px": f, "msg": r_new[2]}
    return log.save({"cases": out})


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
