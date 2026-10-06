"""E0b-trace: where do the C++ cameras of E0 lose the 27 images that E0b found?

E0b (e0b_zero_images.py) found 27 point-camera pairs, all inside r_c, for which the
C++ `projectAll` returns no image while the Python reference returns one.  Each lies where the ray grazes the
point's own radius, at the junction of the near and far branches (1 - Lambda/Lmax from 3e-12 to 8e-9).  This traces
each pair through the steps of the algorithm in both implementations, at the scaled lengths project() uses
(u = 2^floor(log2 r_P)): step (3) classify (near: classify; far: the box search) and step (4) solve, and reports
the first step at which the C++ drops a ray that the Python keeps.  Read-only: it changes no code.
Result (2026-10-06): no step drops it.  The C++ core, at its default tolerance of 1e-9, finds every one of the
27 rays, step for step with the Python; the cameras of E0 rejected them because cyl_model.to_cpp sets the
tolerance to 1e-12, below the residual that rounding allows next to grazing (e0c_tolerance.py).

Run in the refract env with CYL_WAND_PAPER_EXT pointing at a copy of pyopenlpt*.pyd (or the build folder).
Outputs: e0b_trace_out.txt, e0b_trace.json.
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, r"D:\Refraction\code")                     # project_cylinder, the Python reference
EXT = os.environ.get("CYL_WAND_PAPER_EXT")
if EXT:
    sys.path.insert(0, EXT)
else:
    sys.path.insert(0, r"D:\Refraction\pyOpenLPT\build-py\Release")
import pyopenlpt as lpt                                          # noqa: E402
import project_cylinder as pc                                    # noqa: E402

rc = lpt.refract_curved
N, R = [1.0, 1.49, 1.333], [100.80625, 97.63125]
E0B = HERE / "e0b_zero_images.json"
LINES = []


def log(msg=""):
    print(msg, flush=True)
    LINES.append(msg)


def scaled(rP, rX, dz):
    u = math.ldexp(1.0, math.frexp(rP)[1] - 1)
    return u, [r / u for r in R], rP / u, rX / u, dz / u


def trace(p):
    rP, rX, dphi, dz = p["r_P_mm"], p["r_X_mm"], abs(math.radians(p["dphi_deg"])), p["dz_mm"]
    u, Rs, rPs, rXs, dzs = scaled(rP, rX, dz)
    out = {"rX_mm": rX, "dphi_rad": dphi, "dz_mm": dz}
    # end to end
    P0 = [rP, 0.0, 0.0]
    X = [rX * math.cos(dphi), rX * math.sin(dphi), dz]
    cpp_rays, cov = rc.project_cylinder(N, R, P0, X)
    py_rays = pc.project(N, R, np.array(P0), np.array(X))
    out["cpp_rays"] = [(b, L / u, C) for b, L, C, _d in cpp_rays]
    out["py_rays"] = [(b, L / u, C) for b, L, C, _d in py_rays]
    out["cpp_coverage"] = dict(cov)
    # the reference's ray, in scaled units, and how close to grazing it is
    b0, L0, C0 = out["py_rays"][0]
    Lm = pc.Lmax_kstar(N, Rs, rXs, C0)
    out["py_branch"], out["py_one_minus_lam_ratio"], out["kstar"] = b0, 1 - abs(L0) / Lm[0], int(Lm[1])
    # step (3) near: classify, both implementations
    out["py_classify_near"] = pc.classify(N, Rs, rPs, rXs, dphi, dzs)
    out["cpp_classify_near"] = rc.classify_near(N, Rs, rPs, rXs, dphi, dzs)
    # step (3) far: the box search
    out["py_classify_far"] = [tuple(s) for s in pc.classify_far(N, Rs, rPs, rXs, dphi, dzs)]
    seeds, stats = rc.classify_far(N, Rs, rPs, rXs, dphi, dzs)
    out["cpp_classify_far"] = [tuple(s) for s in seeds]
    out["cpp_far_stats"] = dict(stats)
    # step (4): solve from every seed, in both implementations
    solves = []
    for who, seed, far in ([("py", s, False) for s in [out["py_classify_near"]] if s is not None]
                           + [("cpp", s, False) for s in [out["cpp_classify_near"]] if s is not None]
                           + [("py", s, True) for s in out["py_classify_far"]]
                           + [("cpp", s, True) for s in out["cpp_classify_far"]]):
        if far:
            py = pc.solve_far(N, Rs, rPs, rXs, dphi, dzs, seed[0], seed[1])
            cpp = rc.solve_far(N, Rs, rPs, rXs, dphi, dzs, seed[0], seed[1])
        else:
            py = pc.solve(N, Rs, rPs, rXs, dphi, dzs, seed[0], seed[1])
            cpp = rc.solve_cylinder(N, Rs, rPs, rXs, dphi, dzs, seed[0], seed[1], False)
        solves.append({"seed_from": who, "far": far, "seed": tuple(seed),
                       "py_solve": None if py[0] is None else (tuple(py[0]), py[1], py[2]),
                       "cpp_solve": tuple(cpp)})
    out["solves"] = solves
    return out


def main():
    pairs = json.load(open(E0B, encoding="utf-8"))["pairs"]
    log(f"E0b-trace: tracing the {len(pairs)} pairs of E0b through steps (3) and (4); extension {lpt.__file__}")
    res = []
    for k, p in enumerate(pairs):
        t = trace(p)
        res.append(t)
        log(f"[{k:2d}] r_X {t['rX_mm']:6.2f} mm, dphi {math.degrees(t['dphi_rad']):8.4f} deg | reference: "
            f"{t['py_branch']} ray, 1 - Lam/Lmax {t['py_one_minus_lam_ratio']:.1e}, k* {t['kstar']} | end to end: "
            f"C++ {len(t['cpp_rays'])} ray(s), Python {len(t['py_rays'])}")
        log(f"      step 3 near: Python seed {t['py_classify_near']}, C++ seed {t['cpp_classify_near']}")
        log(f"      step 3 far:  Python seeds {t['py_classify_far']}, C++ seeds {t['cpp_classify_far']} "
            f"(boxes {t['cpp_far_stats'].get('boxes')}, undecided {t['cpp_far_stats'].get('undecided')})")
        for s in t["solves"]:
            log(f"      step 4 {'far ' if s['far'] else 'near'} from the {s['seed_from']} seed: Python "
                f"{s['py_solve']}, C++ (ok, Lam, C_z, residual, iters) {s['cpp_solve']}")
    with open(HERE / "e0b_trace.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    (HERE / "e0b_trace_out.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
