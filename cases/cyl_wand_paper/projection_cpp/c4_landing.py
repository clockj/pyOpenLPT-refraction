"""C4: the C++ re-trace (the landing miss with which the multi-start keeps an answer) equals the Python re-traces.

10 000 rays per window: for random points of the uniform set, rays drawn by ms_starts() (most of them miss the point)
and the rays of our search (which land on it). Compared with paper_baselines._miss (the same algorithm in pure
Python: bitwise expected, since C3 found the ports bitwise) and with paper_baselines.landing (NumPy, the scoring's
re-trace: equal to rounding expected). Pass: _miss bitwise everywhere; landing() within 1e-14 m + 1e-12 relative
(rounding over the quadratics of a trace at 0.35 m); and the same side of the tolerance 1e-9 m for every ray.
"""
import numpy as np

from common import Log, SCENES, args, cpp, e1

log = Log("c4_landing")
rng = np.random.default_rng(20261009)
ok = True
summary = {}
for key, sc in SCENES.items():
    X, D = [], []
    for P in e1.point_set(sc, "uniform", 500, rng):
        for d in e1.ms_starts(sc, P, 20):
            X.append(P)
            D.append(d)
        for _, _, _, d in cpp.ours(*args(sc), np.array([P]))[0]:
            X.append(P)
            D.append(tuple(d))
    X, D = np.array(X[:10000]), np.array(D[:10000])
    c = cpp.landing(*args(sc), X, D)
    py = np.array([e1._miss(sc, tuple(x), tuple(d)) for x, d in zip(X, D)])
    lnd = np.array([e1.landing(sc, x, d)[0] for x, d in zip(X, D)])
    bit = int(np.sum((c == py) | (np.isinf(c) & np.isinf(py))))
    fin = np.isfinite(c) & np.isfinite(lnd)
    within = int(np.sum(np.abs(c[fin] - lnd[fin]) <= 1e-14 + 1e-12 * np.abs(lnd[fin])))
    both_inf = int(np.sum(np.isinf(c) & np.isinf(lnd)))
    side = int(np.sum((c <= e1.TOL) == (lnd <= e1.TOL)))
    near = int(np.sum(fin & (c <= e1.TOL)))
    log(f"{key}: {len(X)} rays ({near} landing within 1e-9 m); C++ = _miss bitwise at {bit}; finite in both "
        f"{int(fin.sum())} (within 1e-14 m + 1e-12 relative of landing() at {within}), infinite in both {both_inf}; the "
        f"same side of 1e-9 m at {side}; largest difference from landing() {np.max(np.abs(c[fin] - lnd[fin])):.2g} m")
    ok &= bit == len(X) and side == len(X) and within == int(fin.sum()) and int(fin.sum()) + both_inf == len(X)
    summary[key] = dict(rays=len(X), bitwise=bit, finite=int(fin.sum()), within=within, both_inf=both_inf, side=side)
log(f"C4 {'PASS' if ok else 'FAIL'}")
log.close(passed=bool(ok), summary=summary)
