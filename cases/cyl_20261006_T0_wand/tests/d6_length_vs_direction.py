"""D6 (diagnosis after T2, D5): does the wand's length error follow its 3-D direction (a stretched calibration)?

    python tests/d6_length_vs_direction.py [RESULT_FOLDER]      (default: config out_dir, S2's result)

T2: the wand length of a free triangulation scatters by 22 um RMS, in streaks over time, far more than the image
distance's frame-to-frame jitter (D3: 0.16-0.18 px, about 4-5 um).  If the calibration is stretched along some
direction (e.g. a wrong f against distance, or a wrong index, scales one camera's depth), the length error is
L e^T S e for the wand's direction e and a symmetric matrix S.  This fits dL = L e^T S e + c (6 + 1 parameters) and
reports how much of dL's variance it explains, S's eigenvalues (relative stretch) and directions; then the same with
the wand's position (X, Y, Z) added.  Writes d6_length_vs_direction_out.txt and .json beside this script.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

import numpy as np  # noqa: E402


def main():
    cfg = case.load_config()
    folder = sys.argv[1] if len(sys.argv) > 1 else cfg["out_dir"]
    log = case.Log(os.path.join(HERE, "d6_length_vs_direction"))
    with open(os.path.join(folder, "calibration_data.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    with open(os.path.join(folder, "calibration_result.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    L0 = float(res["wand_length_mm"])
    Lf = np.array([np.nan if x is None else x for x in data["wand_length_free_mm"]], float)
    XA, XB = np.array(data["ends_A_mm"], float), np.array(data["ends_B_mm"], float)
    ok = np.isfinite(Lf) & np.array(data["frame_active"], bool)
    e = (XB - XA)[ok]
    e /= np.linalg.norm(e, axis=1)[:, None]
    mid = 0.5 * (XA + XB)[ok]
    dL = 1e3 * (Lf[ok] - L0)                       # um
    q = np.column_stack([e[:, 0] ** 2, e[:, 1] ** 2, e[:, 2] ** 2, 2 * e[:, 0] * e[:, 1], 2 * e[:, 0] * e[:, 2],
                         2 * e[:, 1] * e[:, 2]]) * L0 * 1e3       # um per unit stretch
    out = {"frames": int(ok.sum()), "dL_rms_um": float(np.sqrt(np.mean(dL ** 2)))}
    log(f"D6: wand length error ({ok.sum()} frames, RMS {out['dL_rms_um']:.2f} um) against the wand's direction")
    log("  direction components (mean |e|): x (across) %.2f, y (axis) %.2f, z (towards cam1) %.2f"
        % tuple(np.mean(np.abs(e), axis=0)))
    for name, X in (("direction (stretch matrix)", np.column_stack([q, np.ones(len(dL))])),
                    ("direction + position", np.column_stack([q, mid, np.ones(len(dL))]))):
        coef, *_ = np.linalg.lstsq(X, dL, rcond=None)
        r = dL - X @ coef
        r2 = 1.0 - float(np.var(r)) / float(np.var(dL))
        log(f"  {name}: explains {100 * r2:.1f} % of the variance; residual RMS {np.sqrt(np.mean(r ** 2)):.2f} um")
        S = np.array([[coef[0], coef[3], coef[4]], [coef[3], coef[1], coef[5]], [coef[4], coef[5], coef[2]]])
        w, V = np.linalg.eigh(S)
        for k in range(3):
            log(f"    stretch {1e2 * w[k]:+.3f} % along ({V[0, k]:+.3f}, {V[1, k]:+.3f}, {V[2, k]:+.3f})")
        out[name] = {"r2": r2, "residual_rms_um": float(np.sqrt(np.mean(r ** 2))), "S": S.tolist(),
                     "eigenvalues": w.tolist(), "eigenvectors": V.T.tolist(), "coef": coef.tolist()}
    log.save(out)


if __name__ == "__main__":
    main()
