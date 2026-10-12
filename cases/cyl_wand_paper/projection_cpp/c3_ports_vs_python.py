"""C3: the C++ ports of the iterative methods return what the Python methods of code/paper_baselines.py return.

For 200 points of every set of every window (the band set unfiltered), each method runs from its paper's start and
from 4 further starts of ms_starts(), in Python and in C++. Compared: the exit status, the iterations, the ray-path
evaluations (and those of the Jacobians), and the angle between the two returned directions. Pass: the same status and
iterations in at least 99 % of the runs, and directions within 1e-12 rad wherever both runs converge to the same exit.
A last-bit difference between Python's and the C runtime's hypot can move a run that sits on a stopping threshold;
every difference is listed.
"""
import numpy as np

from common import Log, SCENES, NAMES, STATUS_NAMES, args, cpp, e1, angle

SETS = {"ref": ("uniform", "edge", "band", "steep", "axis"), "pa": ("uniform", "steep", "axis"),
        "sph": ("uniform", "edge", "band", "axis")}
M, EXTRA = 200, 4

log = Log("c3_ports_vs_python")
rng = np.random.default_rng(20261007)
runs = same_status = same_it = same_ev = 0
worst_angle, worst_stat, diffs, rows = 0.0, 0.0, [], []
for key, kinds in SETS.items():
    sc = SCENES[key]
    for kind in kinds:
        pts = e1.point_set(sc, kind, M, rng)
        tally = {n: [0, 0, 0, 0, 0.0, 0.0] for n in NAMES}  # runs, same status/it/ev, worst angle: zero, stationary
        for X in pts:
            starts = [None] + e1.ms_starts(sc, X, EXTRA)
            for k, name in enumerate(NAMES):
                for d0 in starts:
                    e1._JAC[0] = 0
                    dp, sp, evp, itp = e1._call(sc, name, X, d0)
                    evjp = e1._JAC[0]
                    rc = cpp.baseline(*args(sc), k, list(X), None if d0 is None else list(d0))
                    t = tally[name]
                    t[0] += 1
                    s_ok = e1.STATUS[sp] == rc["status"]
                    i_ok = itp == rc["it"]
                    v_ok = evp == rc["ev"] and evjp == rc["evj"]
                    t[1] += s_ok
                    t[2] += i_ok
                    t[3] += v_ok
                    if dp is not None and rc["has_d"] and s_ok and sp in ("zero", "stationary"):
                        a = angle(dp, rc["d"])
                        j = 4 if sp == "zero" else 5
                        t[j] = max(t[j], a)
                    if not (s_ok and i_ok and v_ok) or (dp is None) != (not rc["has_d"]):
                        diffs.append(f"{key}/{kind} {name} X={tuple(round(float(v), 9) for v in X)} "
                                     f"start={'paper' if d0 is None else 'drawn'}: Python {sp} it {itp} ev {evp}"
                                     f"+{evjp}; C++ {STATUS_NAMES[rc['status']]} it {rc['it']} ev {rc['ev']}"
                                     f"+{rc['evj']}")
        for name in NAMES:
            r, s, i, v, a, b = tally[name]
            runs, same_status, same_it, same_ev = runs + r, same_status + s, same_it + i, same_ev + v
            worst_angle = max(worst_angle, a)
            worst_stat = max(worst_stat, b)
            rows.append(dict(scene=key, set=kind, method=name, runs=r, same_status=s, same_it=i, same_ev=v,
                             worst_angle_zero=a, worst_angle_stationary=b))
            log(f"{key:3s} {kind:8s} {name:5s}: {r} runs; same status {s}, iterations {i}, evaluations {v}; "
                f"directions within {a:.2g} rad (both converged) and {b:.2g} rad (both stationary)")
log("")
log(f"differences ({len(diffs)}):")
for line in diffs[:60]:
    log("  " + line)
if len(diffs) > 60:
    log(f"  ... and {len(diffs) - 60} more (all in c3_ports_vs_python.json)")
pairs = {}
for line in diffs:
    py_st = line.split("Python ")[1].split()[0]
    c_st = line.split("C++ ")[1].split()[0]
    pairs[f"{py_st}/{c_st}"] = pairs.get(f"{py_st}/{c_st}", 0) + 1
log(f"differences by exit (Python/C++): {pairs}")
ok = same_status >= 0.99 * runs and same_it >= 0.99 * runs and worst_angle <= 1e-12
log(f"C3 {'PASS' if ok else 'FAIL'}: {runs} runs; the same status in {same_status} ({100 * same_status / runs:.2f} %), "
    f"iterations in {same_it} ({100 * same_it / runs:.2f} %), evaluations in {same_ev}; directions within "
    f"{worst_angle:.2g} rad where both converge and {worst_stat:.2g} rad where both stop at a stationary point")
log.close(passed=bool(ok), runs=runs, same_status=same_status, same_it=same_it, same_ev=same_ev,
          worst_angle_zero=worst_angle, worst_angle_stationary=worst_stat, by_exit=pairs, rows=rows,
          differences=diffs)
