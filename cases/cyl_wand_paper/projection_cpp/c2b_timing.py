"""C2b (diagnosis): is our search as fast in this module as in OpenLPT's extension?

The smoke run timed our search on the sphere at about 350 us per point, against 0.092 ms in the paper's B1 run of
OpenLPT's extension. Here the same points go through both: the frozen pyopenlpt binding, timed per call from Python
(as B1 timed it, so a call's overhead of about 1 us is included), and this module's single_start (timed in C++),
on 500 uniform points of the reference window and of its sphere. Also B1's own point sets are redrawn as B1 drew
them (paper_openlpt_synthetic.py), to see whether the point distribution explains the difference.
"""
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from common import Log, SCENES, args, cpp, e1

FROZEN = os.environ.get("CYL_WAND_PAPER_EXT", r"C:\Users\zcloc\AppData\Local\Temp\claude\D--Refraction"
                        r"\a41554c2-317f-4657-a430-107eb353b7ed\scratchpad\frozen_pyopenlpt")
sys.path.insert(0, FROZEN)
import pyopenlpt  # noqa: E402

assert Path(pyopenlpt.__file__).resolve().parent == Path(FROZEN).resolve()
lib = pyopenlpt.refract_curved
log = Log("c2b_timing")


def b1_sphere_uniform(m, rng):
    """B1's sphere uniform set: r_X = 94 u^(1/3) mm, direction uniform (paper_openlpt_synthetic.py)."""
    out = []
    for _ in range(m):
        r = 0.094 * rng.uniform() ** (1 / 3)
        v = rng.normal(size=3)
        out.append(tuple(r * v / np.linalg.norm(v)))
    return out


rng = np.random.default_rng(20261012)
for key in ("ref", "sph"):
    sc = SCENES[key]
    for what, pts in (("e1 uniform", e1.point_set(sc, "uniform", 500, rng)),
                      ("B1-style uniform", b1_sphere_uniform(500, rng) if key == "sph" else None)):
        if pts is None:
            continue
        X = np.array(pts, float)
        s = cpp.single_start(*args(sc), X, 3)
        t_mod = np.median(s["ours_t_ns"], axis=0) / 1e3
        t_lib = []
        for _ in range(3):
            tt = []
            for x in X:
                t0 = time.perf_counter()
                if sc.cyl:
                    lib.project_cylinder(sc.n, sc.R, list(sc.P0), list(x), 1e-9, -1)
                else:
                    lib.project_sphere(sc.n, sc.R, [0.0, 0.0, 0.0], list(sc.P0), list(x), 1e-9, -1)
                tt.append(time.perf_counter() - t0)
            t_lib.append(tt)
        t_lib = np.median(np.array(t_lib), axis=0) * 1e6
        rX = np.linalg.norm(X if not sc.cyl else X[:, :2], axis=1)
        log(f"{key} {what}: 500 points, r_X median {1e3 * np.median(rX):.1f} mm; ours median per point: this module "
            f"{np.median(t_mod):.1f} us (C++ timer), OpenLPT's extension {np.median(t_lib):.1f} us (Python timer); "
            f"p95 {np.percentile(t_mod, 95):.1f} / {np.percentile(t_lib, 95):.1f} us")
log.close()
