"""C5b (diagnosis): the image our C++ search missed in C5's shadow-edge set of the reference window.

C5 found, in the 500 shadow-edge points (seed 20261102), one point where our list lacks an image that the endpoint
bracket of the ground truth recovers, at 1 - Lambda/Lmax = 1.2e-14, and one AFRT answer that lands within 1e-9 m but
matches no image. This redraws the same points, finds the point(s), and prints what each projection returns there:
the C++ search, the Python reference (project_cylinder.project), the recovered ray, and AFRT's answer.
"""
import math

import numpy as np

from common import Log, SCENES, args, cpp, e1

log = Log("c5b_edge_miss")
sc = SCENES["ref"]
rng = np.random.default_rng(20261102)            # run_comparison.PLAN[("ref", "edge")]
pts = e1.point_set(sc, "edge", 500, rng)
X = np.array(pts, float)
raw = cpp.ours(*args(sc), X)
s = cpp.single_start(*args(sc), X, 1)
same = lambda a, b: a[0] == b[0] and abs(a[1] - b[1]) < 1e-6 * sc.R[0] / sc.rP and abs(a[2] - b[2]) < 1e-6
found = 0
for i, x in enumerate(pts):
    rays = [(b, L / (math.sqrt(sc.n[0] ** 2 - C * C) * sc.rP), C, np.asarray(d, float)) for b, L, C, d in raw[i]]
    good = [r for r in rays if e1.landing(sc, x, r[3])[0] <= e1.TOL]
    extra = [r for r in e1.endpoint_rays(sc, x, good)
             if e1.landing(sc, x, r[3])[0] <= e1.TOL and not any(same(r, g) for g in good)]
    afrt_d = np.array(s["d"][0, i]) if s["status"][0, i] in (1, 2) else None
    afrt_unmatched = False
    if afrt_d is not None and e1.landing(sc, x, afrt_d)[0] <= e1.TOL:
        truth = good + extra
        afrt_unmatched = not any(e1._angle(afrt_d, g[3]) <= e1.SAME_RAY for g in truth)
    if not extra and not afrt_unmatched:
        continue
    found += 1
    rX, dphi, dz = math.hypot(x[0], x[1]), math.atan2(x[1], x[0]), x[2]
    log(f"point {i}: X = ({x[0]:.12g}, {x[1]:.12g}, {x[2]:.12g}) m; r_X {1e3 * rX:.6f} mm, dphi "
        f"{math.degrees(dphi):.9f} deg, dz {1e3 * dz:.6f} mm")
    log(f"  C++ search: {len(raw[i])} ray(s): " + "; ".join(f"{b} Lambda {L:.15g} C_z {C:.15g}" for b, L, C, _ in raw[i]))
    py = e1.pc.project(sc.n, sc.R, np.array(sc.P0), np.array(x))
    log(f"  Python reference: {len(py)} ray(s): " + "; ".join(f"{b} Lambda {L:.15g} C_z {C:.15g}" for b, L, C, _ in py))
    for r in extra:
        line = e1.landing(sc, x, r[3])[0]
        log(f"  recovered by the endpoint bracket: {r[0]}, sin alpha {r[1]:.15g}, C_z {r[2]:.15g}, 1 - Lambda/Lmax "
            f"{e1._from_grazing(sc, x, r):.2e}, landing miss {line:.2e} m")
    if afrt_d is not None:
        line = e1.landing(sc, x, afrt_d)[0]
        L, C = sc.n[0] * sc.rP * afrt_d[1], sc.n[0] * afrt_d[2]       # the momenta of AFRT's ray at the pinhole
        delta = 1.0 - abs(L) / e1.pc.Lmax_kstar(sc.n, sc.R, rX, abs(C))[0]
        log(f"  AFRT's ray: Lambda {L:.15g}, C_z {C:.15g}, 1 - Lambda/Lmax {delta:.2e}")
        ang = [e1._angle(afrt_d, g[3]) for g in good + extra]
        log(f"  AFRT: exit {s['status'][0, i]}, landing miss {line:.2e} m, angle to the images "
            + ", ".join(f"{a:.2e}" for a in ang) + " rad" + ("; UNMATCHED" if afrt_unmatched else ""))
log(f"C5b: {found} point(s) of 500 where our list lacks an image or AFRT's correct answer matches none")
log.close(points=found)
