"""D3 (diagnosis of X2c's rotation difference): how far does the two-camera Pinhole calibration move when its data
change by rounding-sized amounts?

    python d3_two_camera_flatness.py

X2c (two cameras, camera 1 of 1024 x 768 px) differed from ref2S (the same noise, both cameras 1280 x 800 px) by
3.2e-4 rad in the rotation between the cameras and 0.43 px in camera 1's principal point, with RMS within 2e-5 px.
With two cameras the principal point and the rotation trade off along a nearly flat direction (0.43 px at
f = 2,000 px is 2.2e-4 rad), so where the optimiser stops along it may depend on rounding.  Here the calibrator is
called directly (as X1), on ref2S's data and on the same data with every detection moved by +-1e-9 px and +-1e-6 px
(random signs, fixed seed), and on X2c's: each answer's differences from ref2S's, in the frame-free quantities of
X2 (f, the principal point's offset from the image centre, the distance and the rotation between the cameras, RMS).
Also the three-camera rig (refS and X2a) for comparison.
Writes d3_two_camera_flatness_out.txt and .json.
"""
import common as C  # noqa: I001

import math  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

from w9_w10_pinhole import gui_points, rig  # noqa: E402

log = C.Log("d3_two_camera_flatness")
BIG, SMALL = (1280, 800), (1024, 768)


def angle(Ra, Rb):
    c = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


def calibrate(sizes, cams, seed, eps=0.0):
    import modules.camera_calibration.wand_calibration.wand_calibrator as wcm
    _true, uv, _A, _B, _on = rig(np.random.default_rng(seed), list(sizes) + [BIG] * (3 - len(sizes)))
    uv = uv[:, list(cams)].copy()
    if eps:
        uv = uv + eps * np.random.default_rng(7).choice([-1.0, 1.0], size=uv.shape)
    wc = wcm.WandCalibrator()
    wc.wand_points = gui_points(uv)
    wc.wand_points_filtered = None
    wc.camera_settings = {k: {"focal": 2000.0, "width": sizes[k][0], "height": sizes[k][1]} for k in range(len(cams))}
    wc.image_size = (sizes[0][1], sizes[0][0])
    wc.dist_coeff_num = 0
    ok, msg, _res = wc.calibrate_wand(wand_length=10.0, init_focal_length=2000.0, callback=None)
    fp = wc.final_params
    n = len(cams)
    rms = float(re_rms(msg))
    return {"ok": ok, "f": [float(fp[k]["K"][0, 0]) for k in range(n)],
            "pp": [[float(fp[k]["K"][0, 2] - sizes[k][0] / 2.0), float(fp[k]["K"][1, 2] - sizes[k][1] / 2.0)]
                   for k in range(n)],
            "R": [np.asarray(fp[k]["R"], float) for k in range(n)],
            "C": [(-fp[k]["R"].T @ fp[k]["T"]).ravel() for k in range(n)], "rms": rms}


def re_rms(msg):
    import re
    m = re.search(r"Reprojection RMS: ([0-9.]+) px", msg or "")
    return float(m.group(1)) if m else float("nan")


def diff(r, q):
    n = len(r["f"])
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    return {"f_rel": max(abs(r["f"][k] / q["f"][k] - 1) for k in range(n)),
            "pp_px": max(float(np.hypot(r["pp"][k][0] - q["pp"][k][0], r["pp"][k][1] - q["pp"][k][1])) for k in range(n)),
            "baseline_rel": max(abs(np.linalg.norm(r["C"][a] - r["C"][b]) / np.linalg.norm(q["C"][a] - q["C"][b]) - 1)
                                for a, b in pairs),
            "rotation_rad": max(angle(r["R"][a] @ r["R"][b].T, q["R"][a] @ q["R"][b].T) for a, b in pairs),
            "rms_px": abs(r["rms"] - q["rms"])}


def main():
    out = {}
    for label, cams, ref_sizes, test_sizes in (("two cameras", (0, 1), [BIG, BIG], [BIG, SMALL]),
                                               ("three cameras", (0, 1, 2), [BIG, BIG, BIG], [BIG, SMALL, BIG])):
        ref = calibrate(ref_sizes, cams, 1315)
        log(f"{label}: reference (all 1280 x 800, seed 1315): ok {ref['ok']}, RMS {ref['rms']:.4f} px")
        rows = {}
        for name, sizes, eps in (("data +-1e-9 px", ref_sizes, 1e-9), ("data +-1e-6 px", ref_sizes, 1e-6),
                                 ("camera 1 of 1024 x 768 (X2)", test_sizes, 0.0)):
            r = calibrate(sizes, cams, 1315, eps)
            d = diff(r, ref)
            rows[name] = d
            log(f"  {name:<28}: f {d['f_rel']:.1e}, principal point {d['pp_px']:.3f} px, baseline "
                f"{d['baseline_rel']:.1e}, rotation {d['rotation_rad']:.1e} rad, RMS {d['rms_px']:.1e} px")
        out[label] = rows
    two = out["two cameras"]
    log.check("two cameras: the 1024 x 768 camera moves the answer no more than the data's rounding-sized changes do "
              "(within 3x the larger of the +-1e-9 and +-1e-6 px runs, in every quantity)",
              all(two["camera 1 of 1024 x 768 (X2)"][k] <= 3 * max(two["data +-1e-9 px"][k], two["data +-1e-6 px"][k])
                  for k in ("f_rel", "pp_px", "baseline_rel", "rotation_rad")),
              "; ".join(f"{k} {two['camera 1 of 1024 x 768 (X2)'][k]:.1e} against "
                        f"{max(two['data +-1e-9 px'][k], two['data +-1e-6 px'][k]):.1e}"
                        for k in ("f_rel", "pp_px", "baseline_rel", "rotation_rad")))
    return log.save({"differences": out})


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
