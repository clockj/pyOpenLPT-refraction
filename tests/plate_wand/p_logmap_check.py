"""How accurate are the rotation matrix -> vector conversions near a half-turn?
cv2.Rodrigues, cyl_model.log_so3 (as is), and a corrected log map (the axis column normalised)."""
import math
import sys

import numpy as np

sys.path.insert(0, r"D:\Refraction\pyOpenLPT\modules\camera_calibration\curved_calibration")
import cv2  # noqa: E402
import cyl_model as cm  # noqa: E402


def log_fixed(R):
    R = np.asarray(R, float)
    v = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = float(np.linalg.norm(v))
    c = 0.5 * (float(np.trace(R)) - 1.0)
    th = math.atan2(s, c)
    if s < 1e-12 and c > 0:
        return v
    if th < 3.0:
        return v * (th / s)
    B = 0.5 * (R + R.T) - c * np.eye(3)
    k = int(np.argmax(np.diag(B)))
    axis = B[:, k] / np.linalg.norm(B[:, k])
    if axis @ v < 0:
        axis = -axis
    return axis * th


rng = np.random.default_rng(0)
worst = {"cv2": 0.0, "cm.log_so3": 0.0, "fixed": 0.0}
rows = []
for delta in (0.0, 1e-9, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.05):
    w = {"cv2": [], "cm.log_so3": [], "fixed": []}
    for _ in range(200):
        a = rng.normal(size=3)
        a /= np.linalg.norm(a)
        b = rng.normal(size=3)
        b /= np.linalg.norm(b)
        R = cm.exp_so3(a * math.pi) @ cm.exp_so3(b * delta)      # within delta of a half-turn
        for name, f in (("cv2", lambda M: cv2.Rodrigues(M)[0].ravel()), ("cm.log_so3", cm.log_so3),
                        ("fixed", log_fixed)):
            Rb = cm.exp_so3(f(R))
            err = cm.rotation_angle_deg(Rb, R) * math.pi / 180.0
            w[name].append(err)
    rows.append((delta, {k: max(v) for k, v in w.items()}))
for delta, d in rows:
    print(f"within {delta:7.0e} rad of a half-turn: round-trip error (rad, worst of 200)  "
          + "  ".join(f"{k} {v:.1e}" for k, v in d.items()))
