"""Where does a PINCYL projection spend its time? (scratch)

For points of the T2 cameras' common volume, in the port frame of cam1: time the near-branch classification,
the far-branch box search (classify_far), a near-branch Newton from the true answer moved a little (the warm
start a tracker would have), and the whole project_cylinder. Also: the projections that failed at the camFile's
proj_tol 1e-12.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pyd"))
import pyopenlpt as lpt  # noqa: E402

rc = lpt.refract_curved
n = [1.0, 1.49, 1.333]
R = [100.80625, 97.63125]
RNG = np.random.default_rng(11)

lines = open(os.path.join(HERE, "cams", "cam1.txt")).read().splitlines()
C = np.array([float(x) for x in lines[lines.index("# Inverse of Translation Vector: ") + 1].split(",")])


def local(X):  # buildFrame for axis (0,1,0): u=(1,0,0), v=w x u=(0,0,-1), w=(0,1,0)
    return np.array([X[0], -X[2], X[1]])


pin = local(C)
r_p = math.hypot(pin[0], pin[1])
u = 2.0 ** math.floor(math.log2(r_p))
nR = [x / u for x in R]
pin_u = pin / u

out = json.load(open(os.path.join(HERE, "probe_rig_out.json")))
lo, hi = np.array(out["A1"]["common_bbox_mm"][0]), np.array(out["A1"]["common_bbox_mm"][1])
pts = RNG.uniform(lo, hi, (2000, 3))


def reduce(X):
    x = local(X) / u
    r_x = math.hypot(x[0], x[1])
    d = math.remainder(math.atan2(x[1], x[0]) - math.atan2(pin_u[1], pin_u[0]), 2 * math.pi)
    return r_x, abs(d), x[2] - pin_u[2]


def tmin(f, reps=3):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        r = f()
        best = min(best, time.perf_counter() - t0)
    return best, r


red = [reduce(X) for X in pts]
N = len(red)
res = {}
t, _ = tmin(lambda: [rc.project_cylinder(n, R, list(pin), list(local(X)), 1e-9, -1) for X in pts])
res["project_cylinder_us"] = t / N * 1e6
t, seeds = tmin(lambda: [rc.classify_near(n, nR, r_p / u, rx, dphi, dz) for rx, dphi, dz in red])
res["classify_near_us"] = t / N * 1e6
res["classify_near_found"] = sum(s is not None for s in seeds)
t, far = tmin(lambda: [rc.classify_far(n, nR, r_p / u, rx, dphi, dz, 1e-13, 200000) for rx, dphi, dz in red], 1)
res["classify_far_us"] = t / N * 1e6
res["classify_far_boxes_median"] = float(np.median([f[1]["boxes"] for f in far]))
res["classify_far_seeds_hist"] = np.bincount([len(f[0]) for f in far]).tolist()
# the second target 2 pi - dphi (the other sense) also gets both searches in project_cylinder
t, far2 = tmin(lambda: [rc.classify_far(n, nR, r_p / u, rx, 2 * math.pi - dphi, dz, 1e-13, 200000)
                        for rx, dphi, dz in red], 1)
res["classify_far_other_sense_us"] = t / N * 1e6

# warm start: the converged near solution, then Newton from a seed moved as a 0.05 mm step of the point would
sol = [rc.solve_cylinder(n, nR, r_p / u, rx, dphi, dz, s[0], s[1], False) if s else None
       for (rx, dphi, dz), s in zip(red, seeds)]
ok = [i for i, s in enumerate(sol) if s is not None and s[0]]
res["near_solved"] = len(ok)
dX = RNG.normal(0, 0.05, (N, 3))
red2 = [reduce(pts[i] + dX[i]) for i in range(N)]
t, warm = tmin(lambda: [rc.solve_cylinder(n, nR, r_p / u, *red2[i], sol[i][1], sol[i][2], False)
                        for i in ok])
res["warm_newton_us"] = t / len(ok) * 1e6
res["warm_newton_iters_hist"] = np.bincount([w[4] for w in warm]).tolist()
res["warm_newton_all_ok"] = all(w[0] for w in warm)
t, _ = tmin(lambda: [rc.phi_z(sol[i][1], sol[i][2], n, nR, r_p / u, red[i][0], False) for i in ok])
res["one_phi_z_eval_us"] = t / len(ok) * 1e6

# python-call overhead reference
t, _ = tmin(lambda: [rc.nus(n, 0.1) for _ in range(N)])
res["binding_overhead_us"] = t / N * 1e6
print(json.dumps(res, indent=1))

# --- the failures at proj_tol 1e-12 (the camFile's value)
cam12 = lpt.loadCameraModel(os.path.join(HERE, "cams", "cam1.txt"))
cam9 = lpt.loadCameraModel(os.path.join(HERE, "variants", "cam1_tol9.txt"))
P = [lpt.Pt3D(*map(float, X)) for X in RNG.uniform(lo, hi, (40000, 3))]
r12 = cam12.projectBatchStatus(P)
r9 = cam9.projectBatchStatus(P)
fails = [(i, r12[i][2]) for i in range(len(P)) if not r12[i][0]]
print("fails at 1e-12:", len(fails), "of", len(P), "; at 1e-9:", sum(1 for r in r9 if not r[0]))
for i, msg in fails[:5]:
    X = np.array([P[i][0], P[i][1], P[i][2]])
    rays, cov = cam12.projectAll(P[i])
    rays9, cov9 = cam9.projectAll(P[i])
    print(" X", np.round(X, 3), "r", round(math.hypot(X[0], X[2]), 2), "| 1e-12:", len(rays), cov,
          "| 1e-9:", [(round(r["pt"][0], 2), round(r["pt"][1], 2), r["is_far"]) for r in rays9])
    print("   ", msg[:160])
