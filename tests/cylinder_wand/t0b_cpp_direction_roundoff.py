"""T0b: why the C++ lineOfSight carries ~1e-13 rad of round-off for a distant camera (direction = difference of
two world points |C| from the origin and 1 mm apart).  Emulates that step and compares with the direct R^T m."""
import sys
sys.path.insert(0, r"D:\Refraction\pyOpenLPT\tests\cylinder_wand")
import numpy as np
import rigs
from rigs import cm
# a narrow-field camera like rig 11: f 13000, D 900 mm, tube frame
tube = rigs.CASE_TUBE
cam = rigs.ring_camera(30.0, 1000.0, 13000.0, 2048, 2048)
cpp = cm.to_cpp(cam, tube, tol=1e-13, nmax=300)
rng = np.random.default_rng(1)
uv = np.column_stack([rng.uniform(0, 2048, 4000), rng.uniform(0, 2048, 4000)])
o_c, d_c, ok_c = cm.cpp_rays(cpp, uv)
rw = cm.rows([cam], np.zeros(len(uv), int))
xy = cm.undistort(uv, rw)
o_m, d_m, ok_m, _, _ = cm.trace(rw, xy, tube)
# emulate the C++ step 1: point at camera-frame (x, y, 1) mm in the world, then direction = normalise(point - C)
Rcpp = cm.rot_of(cam.rvec)
Ccpp = -np.linalg.inv(Rcpp) @ (-cam.R @ cam.C)
m = np.column_stack([xy, np.ones(len(xy))])
pw = m @ np.linalg.inv(Rcpp).T + Ccpp
d0 = pw - Ccpp
d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
# continue the trace from this direction: reuse cm.trace by building a camera whose R maps d0 exactly... simpler:
# compare the initial directions only, and the final ones
v = m @ cam.R            # R^T m (rows)
v /= np.linalg.norm(v, axis=1, keepdims=True)
both = ok_c & ok_m
print("initial-direction difference (emulated C++ cancellation vs direct): max %.2e rad, median %.2e" % (
    np.max(np.linalg.norm(d0 - v, axis=1)), np.median(np.linalg.norm(d0 - v, axis=1))))
print("expected scale |C| eps / 1mm = %.2e rad" % (np.linalg.norm(cam.C) * 2.2e-16))
print("final-direction difference C++ vs mine: max %.2e median %.2e rad" % (
    np.max(np.linalg.norm(d_c[both] - d_m[both], axis=1)), np.median(np.linalg.norm(d_c[both] - d_m[both], axis=1))))
