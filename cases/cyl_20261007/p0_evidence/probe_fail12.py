import os, sys, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pyd"))
import pyopenlpt as lpt
RNG = np.random.default_rng(7)
# rebuild the A2 point set exactly as probe_rig.py did (same RNG stream up to A2)
xs = np.arange(-30.0, 30.0 + 1e-9, 1.5); ys = np.arange(-20.0, 20.0 + 1e-9, 2.0); zs = np.arange(-30.0, 30.0 + 1e-9, 1.5)
grid = np.array([(x, y, z) for x in xs for y in ys for z in zs])
cams9 = [lpt.loadCameraModel(os.path.join(HERE, "variants", "cam%d_tol9.txt" % i)) for i in (1, 2)]
cams12 = [lpt.loadCameraModel(os.path.join(HERE, "cams", "cam%d.txt" % i)) for i in (1, 2)]
NROW, NCOL = 800, 1280
def onsens(c, X):
    try:
        rays, _ = cams9[c].projectAll(lpt.Pt3D(*map(float, X)))
    except RuntimeError:
        return 0
    return sum(1 for r in rays if not r["is_ring"] and 0 <= r["pt"][0] < NCOL and 0 <= r["pt"][1] < NROW)
both = np.array([onsens(0, X) >= 1 and onsens(1, X) >= 1 for X in grid])
common = grid[both]
pts = common[RNG.integers(0, len(common), 20000)] + RNG.uniform(-0.75, 0.75, (20000, 3))
P = [lpt.Pt3D(*map(float, X)) for X in pts]
for c in range(2):
    r12 = cams12[c].projectBatchStatus(P); r9 = cams9[c].projectBatchStatus(P)
    for i in range(len(P)):
        if not r12[i][0]:
            X = pts[i]
            rays9, _ = cams9[c].projectAll(P[i])
            rays12, cov12 = cams12[c].projectAll(P[i])
            print("cam%d" % (c + 1), "X", np.round(X, 4), "r=%.3f" % math.hypot(X[0], X[2]),
                  "| tol1e-9 images:", [(round(r["pt"][0], 3), round(r["pt"][1], 3), "far" if r["is_far"] else "near", round(r["lambda"], 6)) for r in rays9],
                  "| tol1e-12:", len(rays12), cov12)
