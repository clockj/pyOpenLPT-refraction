"""C1: the module imports, and every method runs on one point of each window, from its paper's start and from one
other start; the single-start and multi-start entry points return arrays of the expected shapes."""
import numpy as np

from common import Log, SCENES, NAMES, STATUS_NAMES, args, cpp, e1

log = Log("c1_smoke")
pts = {"ref": (0.05, 0.03, 0.02), "pa": (0.02, 0.01, 0.01), "sph": (0.05, 0.03, 0.02)}
ok = True
for key, sc in SCENES.items():
    X = np.array([pts[key]])
    rays = cpp.ours(*args(sc), X)[0]
    log(f"{key}: X = {pts[key]}; ours: {len(rays)} ray(s): " +
        "; ".join(f"{b} Lambda {L:.6g} C_z {C:.6g}" for b, L, C, _ in rays))
    for k, name in enumerate(NAMES):
        r = cpp.baseline(*args(sc), k, list(X[0]))
        miss = cpp.landing(*args(sc), X, np.array([r["d"]]))[0]
        log(f"  {name:5s} {STATUS_NAMES[r['status']]:10s} ev {r['ev']:4d} it {r['it']:4d} miss {miss:.3g} m")
        ok &= r["status"] in (1, 2)
    s = cpp.single_start(*args(sc), X, 2)
    assert s["d"].shape == (4, 1, 3) and s["t_ns"].shape == (2, 4, 1) and s["ours_t_ns"].shape == (2, 1)
    starts = np.array([e1.ms_starts(sc, tuple(X[0]), 3)])
    m = cpp.multistart(*args(sc), X, starts)
    assert m["d"].shape == (4, 1, 4, 3) and m["kept"].shape == (4, 1, 4)
    log(f"  single_start: ours {s['ours_t_ns'][0, 0] / 1e3:.1f} us, baselines "
        + ", ".join(f"{n} {s['t_ns'][0, k, 0] / 1e3:.1f} us" for k, n in enumerate(NAMES))
        + f"; passes differ {s['passes_differ']}; multistart kept per method {m['kept'].sum(axis=(1, 2)).tolist()}")
log(f"C1 {'PASS' if ok else 'FAIL'}: every baseline converged on these interior points")
log.close(passed=bool(ok))
