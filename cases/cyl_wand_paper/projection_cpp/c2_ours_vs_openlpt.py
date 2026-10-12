"""C2: our search as compiled in this module returns what OpenLPT's compiled extension returns.

The reference is the frozen copy of pyopenlpt (built 2026-10-05 22:12, the copy cases E0-E2 used), from the folder in
CYL_WAND_PAPER_EXT (default: the session's frozen copy). Both compile the same RefractCurved.cpp (unchanged since
ce6ae77), so the rays should agree to the last bit. 2000 points: 500 each, uniform and far-branch-rich (60 <= r_X <=
93.5 mm, angle from the pinhole 1.8..pi rad; |dz| <= 150 mm for the cylinder), in the reference window and its sphere.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np

from common import Log, SCENES, args, cpp, e1

FROZEN = os.environ.get("CYL_WAND_PAPER_EXT", r"C:\Users\zcloc\AppData\Local\Temp\claude\D--Refraction"
                        r"\a41554c2-317f-4657-a430-107eb353b7ed\scratchpad\frozen_pyopenlpt")
sys.path.insert(0, FROZEN)
import pyopenlpt  # noqa: E402

assert Path(pyopenlpt.__file__).resolve().parent == Path(FROZEN).resolve(), pyopenlpt.__file__
lib = pyopenlpt.refract_curved

log = Log("c2_ours_vs_openlpt")
log(f"reference: {pyopenlpt.__file__}")


def far_rich(sc, m, rng):
    out = []
    for _ in range(m):
        rX, ang = rng.uniform(0.060, 0.0935), rng.uniform(1.8, math.pi)
        if sc.cyl:
            out.append((rX * math.cos(ang), rX * math.sin(ang) * rng.choice([-1, 1]), rng.uniform(-0.150, 0.150)))
        else:
            roll = rng.uniform(0, 2 * math.pi)
            out.append((rX * math.cos(ang), rX * math.sin(ang) * math.cos(roll), rX * math.sin(ang) * math.sin(roll)))
    return out


rng = np.random.default_rng(20261006)
worst = {"lambda": 0.0, "c_z": 0.0, "dir": 0.0}
total, same_list, bitwise = 0, 0, 0
for key in ("ref", "sph"):
    sc = SCENES[key]
    for kind, pts in (("uniform", e1.point_set(sc, "uniform", 500, rng)), ("far-branch-rich", far_rich(sc, 500, rng))):
        X = np.array(pts)
        mine = cpp.ours(*args(sc), X)
        n_same, n_bit, counts = 0, 0, [0, 0, 0, 0]
        for x, a in zip(X, mine):
            if sc.cyl:
                b, _ = lib.project_cylinder(sc.n, sc.R, list(sc.P0), list(x), 1e-9, -1)
                b = [(br, L, C, tuple(d)) for br, L, C, d in b]
            else:
                b = [(br, L, 0.0, tuple(d)) for br, L, d in
                     lib.project_sphere(sc.n, sc.R, [0.0, 0.0, 0.0], list(sc.P0), list(x), 1e-9, -1)]
            counts[min(len(b), 3)] += 1
            if len(a) == len(b) and all(p[0] == q[0] for p, q in zip(a, b)):
                n_same += 1
                for p, q in zip(a, b):
                    worst["lambda"] = max(worst["lambda"], abs(p[1] - q[1]))
                    worst["c_z"] = max(worst["c_z"], abs(p[2] - q[2]))
                    worst["dir"] = max(worst["dir"], max(abs(u - v) for u, v in zip(p[3], q[3])))
                n_bit += all(p[1] == q[1] and p[2] == q[2] and tuple(p[3]) == tuple(q[3]) for p, q in zip(a, b))
        total += len(X)
        same_list += n_same
        bitwise += n_bit
        log(f"{key:3s} {kind:16s}: {len(X)} points, images 0/1/2/3+ {counts}; same list {n_same}; identical to the "
            f"last bit {n_bit}")
log(f"largest difference: |dLambda| {worst['lambda']:.3g}, |dC_z| {worst['c_z']:.3g}, direction {worst['dir']:.3g}")
ok = same_list == total
log(f"C2 {'PASS' if ok else 'FAIL'}: {same_list} of {total} points have the same list of rays "
    f"({bitwise} identical to the last bit)")
log.close(passed=ok, points=total, same_list=same_list, bitwise=bitwise, worst=worst)
