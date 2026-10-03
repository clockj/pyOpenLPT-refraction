"""W1: every derivative column of plate_model.project (J_cam, J_win, J_X) against central differences, and (block 2)
plate_wand.evaluate's global and frame columns against differences of the whole residual.

Block 1: 300 random camera / window / point sets (one and two plates, incidence to 60 deg, distortion on and off, an
anchor anywhere near the window, some points on the camera's normal line), 20 points each.  The finite differences
move the parameter exactly as the solver does: R <- R exp([h e_i]x), C + h e_i, f e^h, the normal along the sphere
with the surface pivoting about the anchor, d + h, X + h e_i.  Criterion: per column, max |J_fd - J| / max |J| below
1e-6 (columns that are identically zero, e.g. the distortion of a camera at the principal point, are skipped).
Output: w1_jacobians_out.txt / .json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prigs  # noqa: E402

pm = prigs.pm
log = prigs.Tee(HERE / "w1_jacobians_out.txt")
rng = np.random.default_rng(11)


def rand_unit():
    a = rng.normal(size=3)
    return a / np.linalg.norm(a)


def case(distorted, two_plates):
    n = rand_unit()
    d = rng.uniform(-100, 100)
    if two_plates:
        win = pm.Window(n, d, (rng.uniform(3, 20), rng.uniform(2, 10)), (1.0, 1.49, 1.6, 1.333))
    else:
        win = pm.Window(n, d, (rng.uniform(3, 30),), (1.0, rng.uniform(1.4, 1.6), rng.uniform(1.3, 1.5)))
    h1 = math.exp(rng.uniform(math.log(80), math.log(1500)))
    th = math.radians(rng.uniform(0, 60))
    t1, t2 = pm.tangent_basis(n)
    ph = rng.uniform(0, 2 * math.pi)
    e = math.cos(ph) * t1[0] + math.sin(ph) * t2[0]
    u = math.cos(th) * n + math.sin(th) * e
    C = d * n + rng.uniform(-200, 200) * t1[0] - (h1 / math.cos(th)) * u
    R = prigs.look_dir(u, up=rand_unit(), roll_deg=rng.uniform(-180, 180))
    f = math.exp(rng.uniform(math.log(1500), math.log(40000)))
    dist = np.zeros(5)
    if distorted:
        dist = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.1, 0.1), rng.uniform(-1e-3, 1e-3),
                         rng.uniform(-1e-3, 1e-3), rng.uniform(-0.05, 0.05)])
    cam = pm.Camera(R, C, f, 640 + rng.uniform(-30, 30), 400 + rng.uniform(-30, 30), 1280, 800, dist)
    # points: along random camera-side rays (within 25 deg of the chief ray), 5-300 mm into the object medium
    m = 20
    X = []
    for k in range(m):
        q = u + 0.45 * rng.uniform(-1, 1) * t1[0] + 0.45 * rng.uniform(-1, 1) * t2[0]
        if k == 0:
            q = n.copy()                                  # the camera's normal line: rho = 0
        if k == 1:
            q = n + 1e-11 * t1[0]                         # next to it
        rw = pm.rows([cam], [0])
        xyq = (R @ q)[:2] / (R @ q)[2]
        o, dd, ok = pm.trace(rw, pm.wrows([win], [0]), xyq[None])
        depth = rng.uniform(5, 300)
        X.append(o[0] + (depth / float(dd[0] @ n)) * dd[0])
    A = d * n + rng.uniform(-300, 300) * t1[0] + rng.uniform(-300, 300) * t2[0]
    return cam, win, np.array(X), A


def proj(cam, win, X, A, jac=False):
    m = len(X)
    return pm.project(pm.rows([cam], np.zeros(m, int)), pm.wrows([win], np.zeros(m, int), anchors=[A]), X, jac=jac)


def moved(cam, win, X, A, kind, i, h):
    c, w, Y = cam.copy(), win.copy(), X.copy()
    if kind == "cam":
        if i < 3:
            c.R = cam.R @ pm.exp_so3(h * np.eye(3)[i])
        elif i < 6:
            c.C = cam.C + h * np.eye(3)[i - 3]
        elif i == 6:
            c.f = cam.f * math.exp(h)
        elif i == 7:
            c.cx += h
        elif i == 8:
            c.cy += h
        else:
            c.dist = cam.dist.copy()
            c.dist[i - 9] += h
    elif kind == "win":
        if i < 2:
            ab = [0.0, 0.0]
            ab[i] = h
            n_new = pm.retract_normal(win.n, ab)
            w = pm.Window(n_new, float(n_new @ A) + (win.d - float(win.n @ A)), win.t, win.idx)
        else:
            w.d = win.d + h
    else:
        Y[:, i] += h
    return c, w, Y


STEPS = {"cam": [1e-6] * 3 + [1e-4] * 3 + [1e-7, 1e-4, 1e-4] + [1e-6] * 5, "win": [1e-7, 1e-7, 1e-5],
         "X": [1e-5] * 3}
NAMES = {"cam": list(pm.CAM_PARAMS), "win": list(pm.WIN_PARAMS), "X": ["X1", "X2", "X3"]}
worst = {k: [0.0] * len(v) for k, v in NAMES.items()}
n_sets = 0
n_fail_rows = 0
for s in range(300):
    cam, win, X, A = case(distorted=s % 2 == 1, two_plates=s % 3 == 2)
    base = proj(cam, win, X, A, jac=True)
    good = base.ok
    n_fail_rows += int((~good).sum())
    if good.sum() < 3:
        continue
    n_sets += 1
    J = {"cam": base.J_cam, "win": base.J_win, "X": base.J_X}
    for kind in ("cam", "win", "X"):
        for i, h in enumerate(STEPS[kind]):
            cp, wp, Xp = moved(cam, win, X, A, kind, i, h)
            cn, wn, Xn = moved(cam, win, X, A, kind, i, -h)
            up = proj(cp, wp, Xp, A).uv
            un = proj(cn, wn, Xn, A).uv
            fd = (up - un) / (2 * h)
            an = J[kind][:, :, i]
            sel = good & np.all(np.isfinite(fd), axis=1)
            scale = float(np.max(np.abs(an[sel])))
            if scale == 0.0:
                continue
            err = float(np.max(np.abs(fd[sel] - an[sel]))) / scale
            worst[kind][i] = max(worst[kind][i], err)
log(f"block 1: {n_sets} camera/window/point sets, 20 points each ({n_fail_rows} rows not projected)")
ok1 = True
for kind in ("cam", "win", "X"):
    for i, name in enumerate(NAMES[kind]):
        flag = worst[kind][i] <= 1e-6
        ok1 &= flag
        log(f"  {kind:3s} {name:6s}: worst relative error {worst[kind][i]:.2e}" + ("" if flag else "   <-- FAIL"))
res = {"block1": {"sets": n_sets, "worst": worst, "pass": bool(ok1)}}

# ---- block 2: plate_wand.evaluate against differences of the whole residual (needs plate_wand)
ok2 = True
try:
    import plate_wand as pw
except ImportError:
    pw = None
if pw is not None:
    import w1_block2
    ok2, res["block2"] = w1_block2.run(log, rng)
else:
    log("block 2: plate_wand not written yet -- skipped")
    res["block2"] = "skipped"
ok = ok1 and ok2
res["pass"] = bool(ok)
log("W1: " + ("PASS" if ok else "FAIL"))
(HERE / "w1_jacobians.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
sys.exit(0 if ok else 1)
