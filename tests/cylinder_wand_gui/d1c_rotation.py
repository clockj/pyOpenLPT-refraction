"""D1c: is G3's cam0 offset (D1b: the C++ camera's projections 2.75e-3 px off in u, constant) its rotation?

SUPERSEDED by d1d_cam0_offset.py: the angles below come from acos((trace - 1) / 2), which cannot resolve rotations
below about 1e-7 rad (acos near 1 turns round-off of 1e-15 into ~5e-8 rad), so its "6.66e-08 rad" for cam1 is an
artefact; its output is kept as d1c_rotation_acos_out.txt.  D1d compares the matrices element by element.
cam0 faces the tube head-on, so its rotation is close to a half-turn.  Compares, per camera:
  - the fitted R (calibration_result.json) with exp(log(R)) in Python (cyl_model.rvec_of / rot_of);
  - with the "Rotation Matrix" the C++ camera writes into its camFile (17 digits), and with cv2.Rodrigues of the
    camFile's rotation vector;
  - the angle of R from a half-turn.

    python d1c_rotation.py
"""
import common as C  # noqa: I001

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

log = C.Log("d1c_rotation")


def section(txt, head, rows):
    lines = txt.split("\n")
    i = next(k for k, ln in enumerate(lines) if ln.startswith("# " + head))
    return np.array([[float(v) for v in lines[i + 1 + r].split(",")] for r in range(rows)])


def angle(Ra, Rb):
    return math.acos(max(-1.0, min(1.0, 0.5 * (float(np.trace(Ra.T @ Rb)) - 1.0))))


def main():
    from modules.camera_calibration.curved_calibration import cyl_model as cm
    res_dir = C.DATA / "g3_run" / "results"
    res = json.load(open(res_dir / "calibration_result.json", encoding="utf-8"))
    out = {}
    for k, c in enumerate(res["cameras"]):
        R = np.array(c["R"])
        th = math.acos(max(-1.0, min(1.0, 0.5 * (np.trace(R) - 1.0))))
        w = cm.rvec_of(R)
        R_py = cm.rot_of(w)
        txt = open(res_dir / "camFile" / f"cam{k}.txt", encoding="utf-8").read()
        R_file = section(txt, "Rotation Matrix", 3)
        w_file = section(txt, "Rotation Vector", 1)[0]
        R_cv, _ = cv2.Rodrigues(w_file.reshape(3, 1))
        a_py, a_file, a_cv = angle(R, R_py), angle(R, R_file), angle(R, R_cv)
        log(f"cam{k}: rotation angle {math.degrees(th):.6f} deg ({math.pi - th:.3e} rad from a half-turn); "
            f"|rvec_of - file rvec| {np.linalg.norm(w - w_file):.2e}")
        log(f"   R vs exp(log R) in Python {a_py:.2e} rad; vs the camFile's matrix {a_file:.2e} rad; vs "
            f"cv2.Rodrigues(file rvec) {a_cv:.2e} rad; offset that makes at f = {c['f_px']:.0f} px: "
            f"{a_file * c['f_px']:.2e} px")
        out[c["name"]] = {"angle_from_half_turn_rad": math.pi - th, "python_roundtrip_rad": a_py,
                          "camfile_matrix_rad": a_file, "cv2_of_file_rvec_rad": a_cv}
    return log.save({"cameras": out})


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
