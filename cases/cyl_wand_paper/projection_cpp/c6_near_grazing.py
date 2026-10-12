"""C6: the near-grazing test of the paper (T2 of code/tests/classify_grazing/, as rerun by
code/tests/openlpt_audit/u6_t2_metres.py) on the C++ core.

The same 1950 draws (seed 20260924; the reference window and two random stacks; 400 near-branch and 250 far-branch
rays each, at 1 - Lambda/Lmax log-uniform in 1e-15..1e-3), each defining its own point. A projection passes on a draw
if it returns the drawn ray (same branch, C_z and |Lambda|/Lmax to 1e-6, traced landing within 1e-9 m), by u6's
found(). Run on the Python reference (the paper quoted 1937 of 1950 from u6's run of 24 Sep, on the code of that day)
and on this module's C++ search, on the same draws.

Pass: every ray lost lies in the rounding band. Next to grazing the rounding error of Phi is about eps/sqrt(2 delta),
delta = 1 - Lambda/Lmax (Supplement S8); it comes within a factor of two of the projection's tolerance tol = 1e-9 when
eps/sqrt(2 delta) >= tol/2, i.e. delta <= 2 eps^2 / tol^2 = 9.9e-14. (u6 itself used 1e-14, stricter than that
argument, and so reported FAIL for its 3 losses between 1e-14 and 5.6e-14.)
"""
import time

import numpy as np

from common import CODE, Log, cpp

import importlib.util
import sys

sys.path.insert(0, str(CODE / "tests" / "openlpt_audit"))
spec = importlib.util.spec_from_file_location("u6", CODE / "tests" / "openlpt_audit" / "u6_t2_metres.py")
u6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u6)          # its main() is not run; it imports project_cylinder and the frozen "before"
pc = u6.pc


class CppProjection:
    """project(n, R, P0, X) of the C++ core, in project_cylinder's form (branch, Lambda, C_z, direction)."""

    @staticmethod
    def project(n, R, P0, X):
        rays = cpp.ours(list(n), list(R), float(P0[0]), False, np.array([X], float))[0]
        return [(b, L, C, np.array(d)) for b, L, C, d in rays]


log = Log("c6_near_grazing")
rng = np.random.default_rng(u6.SEED)
t0 = time.perf_counter()
grand = {"python": [0, 0], "cpp": [0, 0]}
lost = {"python": [], "cpp": []}
for name, n, R, rP in u6.scenes(rng):
    log(f"{name}: n = {[round(v, 4) for v in n]}, R = {[round(1e3 * v, 3) for v in R]} mm, r_P = {1e3 * rP:.1f} mm")
    for branch, count in u6.PER_SCENE.items():
        far = branch == "far"
        kept = 0
        hits = {"python": 0, "cpp": 0}
        while kept < count:
            d = u6.draw(rng, n, R, rP, far)
            if d is None:
                continue
            X, L, C, delta = d
            kept += 1
            for tag, mod in (("python", pc), ("cpp", CppProjection)):
                ok, _ = u6.found(mod, n, R, rP, X, L, C, far)
                hits[tag] += ok
                if not ok:
                    lost[tag].append((name, branch, delta))
        for tag in hits:
            grand[tag][0] += hits[tag]
            grand[tag][1] += kept
        log(f"  {branch:4s}: Python {hits['python']} of {kept}, C++ {hits['cpp']} of {kept}")
for tag in grand:
    deltas = sorted(r[2] for r in lost[tag])
    log(f"{tag}: {grand[tag][0]} of {grand[tag][1]} near-grazing rays returned; missed "
        f"{len(deltas)}" + (f", all with 1 - Lambda/Lmax <= {max(deltas):.1e}" if deltas else ""))
EPS, TOL = np.finfo(float).eps, 1e-9
ROUND = 2 * EPS ** 2 / TOL ** 2                  # eps/sqrt(2 delta) >= tol/2
beyond = [r for r in lost["cpp"] if r[2] > ROUND]
ok = not beyond
log(f"C6 {'PASS' if ok else 'FAIL'}: C++ {grand['cpp'][0]} of {grand['cpp'][1]}, Python {grand['python'][0]} of "
    f"{grand['python'][1]}; C++ rays lost outside the rounding band (1 - Lambda/Lmax > 2 eps^2/tol^2 = {ROUND:.2g}): "
    f"{len(beyond)}; {time.perf_counter() - t0:.0f} s")
log.close(passed=ok, cpp=grand["cpp"], python=grand["python"],
          cpp_lost=[list(r) for r in lost["cpp"]], python_lost=[list(r) for r in lost["python"]])
