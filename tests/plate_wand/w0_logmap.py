"""W0 (log map) and Phase 1 check: the fixed cyl_model.log_so3 / rvec_of at every rotation angle.

The round trip exp(log(R)) must return R to within 1e-14 rad, at random angles in [0, pi], near zero, and within
1e-9 .. 0.14 rad of a half-turn (where the old half-turn branch was 1.3 rad off and cv2.Rodrigues loses 2e-5 rad).
Also checks |log(R)| <= pi and that rvec_of is log_so3.  Writes w0_logmap_out.txt beside itself.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "modules" / "camera_calibration" / "curved_calibration"))
import cyl_model as cm  # noqa: E402

try:
    import cv2  # noqa: E402
except ImportError:  # pragma: no cover
    cv2 = None

out_lines = []


def say(s=""):
    print(s)
    out_lines.append(s)


def angle_between(Ra, Rb):
    """Angle of Ra Rb^T (rad), accurate at every angle: 2 asin(|Ra - Rb|_F / (2 sqrt 2))."""
    return 2.0 * math.asin(min(1.0, np.linalg.norm(Ra - Rb) / (2.0 * math.sqrt(2.0))))


rng = np.random.default_rng(20261002)


def rand_axis():
    a = rng.normal(size=3)
    return a / np.linalg.norm(a)


results = {}
groups = [("uniform angle in [0, pi]", lambda: rng.uniform(0.0, math.pi)),
          ("tiny angle 1e-12 .. 1e-6", lambda: 10.0 ** rng.uniform(-12, -6)),
          ("exactly 0", lambda: 0.0)]
for delta in (0.0, 1e-12, 1e-9, 1e-7, 1e-5, 1e-3, 0.05, 0.14, 0.2):
    groups.append((f"pi - {delta:.0e}", (lambda d=delta: math.pi - d)))
worst_all = 0.0
cv2_worst = 0.0
for name, ang in groups:
    worst = 0.0
    worst_norm = 0.0
    for _ in range(2000):
        th = ang()
        R = cm.exp_so3(rand_axis() * th)
        if rng.random() < 0.5:          # also perturb off the exact family by a random small rotation
            R = R @ cm.exp_so3(rand_axis() * 1e-13)
        w = cm.log_so3(R)
        assert np.array_equal(cm.rvec_of(R), w)
        worst = max(worst, angle_between(cm.exp_so3(w), R))
        worst_norm = max(worst_norm, float(np.linalg.norm(w)) - math.pi)
        if cv2 is not None:
            cv2_worst = max(cv2_worst, angle_between(cm.exp_so3(cv2.Rodrigues(R)[0].ravel()), R))
    results[name] = worst
    worst_all = max(worst_all, worst)
    say(f"{name:28s}: worst round-trip error {worst:.2e} rad over 2000 rotations; max |w| - pi = {worst_norm:+.1e}")
# look-at cameras with image up = world +Y (OpenCV y row = -Y): exact half-turns
for k in range(8):
    phi = 2 * math.pi * k / 8
    z = np.array([math.sin(phi), 0.0, math.cos(phi)])
    y = np.array([0.0, -1.0, 0.0])
    x = np.cross(y, z)
    R = np.vstack([x, y, z])
    if np.linalg.det(R) < 0:
        R[0] = -R[0]
    w = cm.log_so3(R)
    e = angle_between(cm.exp_so3(w), R)
    worst_all = max(worst_all, e)
    say(f"look-at camera, azimuth {math.degrees(phi):5.1f} deg: |w| = {np.linalg.norm(w):.15f}, round trip {e:.1e}")
say()
say(f"worst round-trip error of the fixed log map: {worst_all:.2e} rad (criterion 1e-14)")
if cv2 is not None:
    say(f"for comparison, cv2.Rodrigues on the same rotations: worst {cv2_worst:.2e} rad")
ok = worst_all <= 1e-14
say("W0 log map: " + ("PASS" if ok else "FAIL"))
(HERE / "w0_logmap_out.txt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
(HERE / "w0_logmap.json").write_text(json.dumps({"worst": worst_all, "groups": results, "cv2_worst": cv2_worst,
                                                 "pass": ok}, indent=1), encoding="utf-8")
sys.exit(0 if ok else 1)
