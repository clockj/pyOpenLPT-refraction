"""D3 (diagnosis after T2): the random part of the detection error, from the smoothness of the wand's motion.

    python tests/d3_jitter.py

The wand swings smoothly (about 20 px per frame), so over 7 consecutive frames each ball's image track is close to a
quadratic in time.  The scatter of S1's centres about a quadratic fitted to frames f-3 ... f+3, taken at frame f,
is the frame-to-frame (random) detection noise: sigma = RMS / sqrt(2/3) (the centre point's leverage is 1/3).
Done per camera and ball for u and v, and for the image distance A-B.  A detection error that changes slowly with
the wand's pose (e.g. a bias that depends on the rod's angle) does not show here; S2's residuals (T2) contain both.
Writes d3_jitter_out.txt and d3_jitter.json beside this script.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

import numpy as np  # noqa: E402

W = 3                                     # half-window (frames)


def centre_residuals(t, y):
    """Residual at the window's centre of a quadratic fitted to (t, y), t = -W..W."""
    A = np.column_stack([np.ones_like(t), t, t * t])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y[W] - coef[0]


def main():
    cfg = case.load_config()
    log = case.Log(os.path.join(HERE, "d3_jitter"))
    _dets, kept, _reasons, _medians, _frames = case.load_selection(cfg)
    t = np.arange(-W, W + 1, dtype=float)
    k = 1.0 / math.sqrt(2.0 / 3.0)
    out = {}
    log("D3: frame-to-frame detection noise (quadratic over 7 frames), S1's centres")
    for c in case.cam_ids(cfg):
        fr = sorted(f for (f, cc) in kept if cc == c)
        fs = set(fr)
        res = {"Au": [], "Av": [], "Bu": [], "Bv": [], "AB": []}
        for f in fr:
            win = [f + d for d in range(-W, W + 1)]
            if not all(g in fs for g in win):
                continue
            A = np.array([kept[(g, c)][0][:2] for g in win])
            B = np.array([kept[(g, c)][1][:2] for g in win])
            res["Au"].append(centre_residuals(t, A[:, 0]))
            res["Av"].append(centre_residuals(t, A[:, 1]))
            res["Bu"].append(centre_residuals(t, B[:, 0]))
            res["Bv"].append(centre_residuals(t, B[:, 1]))
            res["AB"].append(centre_residuals(t, np.linalg.norm(B - A, axis=1)))
        rec = {}
        for name, r in res.items():
            r = np.array(r)
            mad = 1.4826 * float(np.median(np.abs(r - np.median(r))))
            rec[name] = {"n": int(len(r)), "sigma_rms_px": k * float(np.sqrt(np.mean(r ** 2))),
                         "sigma_robust_px": k * mad}
        out[f"cam{c}"] = rec
        log(f"  cam{c} ({rec['Au']['n']} windows): noise sigma per coordinate, RMS (robust): "
            + ", ".join(f"{n} {v['sigma_rms_px']:.3f} ({v['sigma_robust_px']:.3f})" for n, v in rec.items()) + " px")
    log.save(out)


if __name__ == "__main__":
    main()
