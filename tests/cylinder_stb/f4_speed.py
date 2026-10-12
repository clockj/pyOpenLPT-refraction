"""F4 (plan, Phase 1): the cost of a projection, with and without a hint.

    python f4_speed.py [--build p1] [--n 4000] [--cams DIR]

Cameras: 6 Oct T2's camFiles at the tracking tolerance 1e-9, and a PINHOLE camera with the same K, R, t (the
camFile cut after its pose) for scale.  N random points both cameras see.
  1. Per call, from Python: Camera.project() and Camera.projectWithHint() with the hint of the point 0.05 mm away
     (the median over calls of batches of 200 calls, per camera).  The Python call itself costs about as much as a
     pinhole projection, which is reported beside them.
  2. A shake-like sequence inside C++: Tracer3D.projectObject2D (both cameras per call) on a particle moved through
     the shake's trial positions (x0 - D, x0, x0 + D, then the best, on each axis; D = 0.1, 0.05, 0.025, 0.0125 mm
     over four loops: 48 positions), with OPENLPT_PROJECTION_HINTS on (this process) and the full search (project()
     via a hint that is never valid, i.e. Camera.project on each position) for comparison.
Judged by (plan): median cost per hinted call <= 3 us (today about 180 us).  Writes f4_speed_out.txt and f4_speed.json.
"""
import argparse
import os
import time

import numpy as np

import cstb

BOX = ((-20.0, -15.0, -30.0), (25.0, 15.0, 35.0))


def pinhole_copy(src, dst):
    with open(src, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    i = next(k for k, ln in enumerate(lines) if ln.startswith("# Port Center"))
    lines = lines[:i]
    j = next(k for k, ln in enumerate(lines) if ln.strip() == "PINCYL")
    lines[j] = "PINHOLE"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def per_call(fn, items, batch=200):
    times = []
    for k in range(0, len(items) - batch + 1, batch):
        chunk = items[k:k + batch]
        t0 = time.perf_counter()
        for it in chunk:
            fn(it)
        times.append((time.perf_counter() - t0) / batch)
    return float(np.median(times)) * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="p1")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--cams", default=cstb.CAMS_T2)
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()
    lpt, build = cstb.use_build(args.build)
    log = cstb.Log(os.path.join(cstb.HERE, "f4_speed"))
    rng = np.random.default_rng(args.seed)
    files = cstb.cams_with_tol(args.cams, 1e-9)
    cams = [lpt.loadCameraModel(f) for f in files]
    pin_files = [os.path.join(cstb.DATA, "cams_T2_pinhole", f"cam{c}.txt") for c in (1, 2)]
    for f, p in zip(files, pin_files):
        pinhole_copy(f, p)
    pins = [lpt.loadCameraModel(f) for f in pin_files]
    log(f"F4: projection cost; build {build['name']} ({build['git_head']} + diff "
        f"{(build['git_diff_sha256'] or '')[:12]}); cameras {args.cams} at tol 1e-9")
    pts, _pix = cstb.sample_shared(lpt, cams, args.n, BOX, rng)
    out = {"build": build, "n": len(pts), "per_call_us": {}, "shake_like": {}}
    for c, (cam, pin) in enumerate(zip(cams, pins)):
        P = [cstb.pt3(lpt, X) for X in pts]
        hints = [cam.projectWithHint(p, lpt.ProjectionHint())[2] for p in P]
        moved = []
        for X, h in zip(pts, hints):
            d = rng.normal(size=3)
            moved.append((cstb.pt3(lpt, X + 0.05 * d / np.linalg.norm(d)), h))
        t_pin = per_call(lambda p: pin.project(p), P)
        t_slow = per_call(lambda p: cam.project(p), P)
        t_fast = per_call(lambda it: cam.projectWithHint(it[0], it[1]), moved)
        t_pin_hint = per_call(lambda it: pin.projectWithHint(it[0], it[1]), moved)
        out["per_call_us"][f"cam{c + 1}"] = {"pinhole_project": t_pin, "pinhole_projectWithHint": t_pin_hint,
                                             "pincyl_project": t_slow, "pincyl_projectWithHint": t_fast}
        log(f"  cam{c + 1}, per call from Python (median of batches): pinhole project {t_pin:.2f} us, "
            f"pinhole projectWithHint {t_pin_hint:.2f} us; PINCYL project {t_slow:.1f} us, "
            f"PINCYL projectWithHint (0.05 mm) {t_fast:.2f} us")

    # 2. shake-like sequence through projectObject2D (both cameras per call, inside C++)
    def shake_positions(x0):
        seq = []
        for D in (0.1, 0.05, 0.025, 0.0125):
            for ax in range(3):
                for s in (-1.0, 0.0, 1.0):
                    x = x0.copy()
                    x[ax] += s * D
                    seq.append(x)
                x0 = x0.copy()
                x0[ax] += 0.3 * D       # "the best"
                seq.append(x0.copy())
        return seq

    seqs = [shake_positions(X) for X in pts[:1000]]
    n_calls = sum(len(s) for s in seqs)
    tr = lpt.Tracer3D()
    t0 = time.perf_counter()
    for seq in seqs:
        tr._proj_hint = []           # a new particle: the first projection runs the full search
        for x in seq:
            tr._pt_center = cstb.pt3(lpt, x)
            tr.projectObject2D(cams)
    t_hint = (time.perf_counter() - t0) / n_calls * 1e6
    t0 = time.perf_counter()
    for seq in seqs[:100]:
        for x in seq:
            for cam in cams:
                cam.project(cstb.pt3(lpt, x))
    t_full = (time.perf_counter() - t0) / sum(len(s) for s in seqs[:100]) * 1e6
    out["shake_like"] = {"particles": len(seqs), "positions_per_particle": len(seqs[0]),
                         "us_per_position_both_cams_hinted": t_hint, "us_per_position_both_cams_full": t_full}
    log(f"  shake-like sequences ({len(seqs)} particles x {len(seqs[0])} positions, the first one without a hint): "
        f"projectObject2D (2 cameras, hinted) {t_hint:.2f} us per position; project() on both cameras "
        f"{t_full:.1f} us per position ({t_full / t_hint:.0f}x)")
    med = max(v["pincyl_projectWithHint"] for v in out["per_call_us"].values())
    out["pass"] = bool(med <= 3.0)
    log(f"  judged: hinted call <= 3 us (from Python, call overhead included): {'PASS' if out['pass'] else 'FAIL'}")
    log.save(out)


if __name__ == "__main__":
    main()
