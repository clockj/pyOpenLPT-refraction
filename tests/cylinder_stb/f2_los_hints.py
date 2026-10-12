"""F2 (plan, Phase 1): hints from lines of sight, as StereoMatch gives them to a newly triangulated particle.

    python f2_los_hints.py [--build p1] [--n 20000] [--cams DIR]

Cameras: 6 Oct T2's camFiles at the tracking tolerance 1e-9.  For N random points both cameras see: each camera's
pixel, plus Gaussian detection noise of SIGMA px (0, 0.1 and 0.5 px); each pixel's line of sight
(Camera.lineOfSight); the point triangulated from the two lines (the midpoint of their common perpendicular, as
myMATH::triangulation for two lines); then, per camera, the hint Camera.hintFromLineOfSight(line, point) and
Camera.projectWithHint(point, hint) against Camera.project(point).
Judged as F1: pixels within 1e-6 px wherever both succeed; no new failure; the fast-path rate reported (from
refract_curved.project_cylinder_from on the same input).  Writes f2_los_hints_out.txt and f2_los_hints.json.
"""
import argparse
import os

import numpy as np

import cstb

BOX = ((-20.0, -15.0, -30.0), (25.0, 15.0, 35.0))


def midpoint(l1, l2):
    p1, d1 = np.array([l1.pt[i] for i in range(3)]), np.array([l1.unit_vector[i] for i in range(3)])
    p2, d2 = np.array([l2.pt[i] for i in range(3)]), np.array([l2.unit_vector[i] for i in range(3)])
    A = np.array([[d1 @ d1, -d1 @ d2], [d1 @ d2, -d2 @ d2]])
    b = np.array([(p2 - p1) @ d1, (p2 - p1) @ d2])
    s, t = np.linalg.solve(A, b)
    return 0.5 * ((p1 + s * d1) + (p2 + t * d2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="p1")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--cams", default=cstb.CAMS_T2)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()
    lpt, build = cstb.use_build(args.build)
    log = cstb.Log(os.path.join(cstb.HERE, "f2_los_hints"))
    rng = np.random.default_rng(args.seed)
    files = cstb.cams_with_tol(args.cams, 1e-9)
    cams = [lpt.loadCameraModel(f) for f in files]
    geos = [cstb.pincyl_geometry(f) for f in files]
    frames = [cstb.port_frame(g["center"], g["axis"]) for g in geos]
    rc = lpt.refract_curved
    log(f"F2: hints from lines of sight; build {build['name']} ({build['git_head']} + diff "
        f"{(build['git_diff_sha256'] or '')[:12]}); cameras {args.cams} at tol 1e-9")
    pts, pix = cstb.sample_shared(lpt, cams, args.n, BOX, rng)
    out = {"build": build, "n": len(pts), "levels": []}
    for sigma in (0.0, 0.1, 0.5):
        rec = {"sigma_px": sigma, "pairs": 0, "both_ok": 0, "slow_fail_fast_ok": 0, "slow_ok_fast_fail": 0,
               "both_fail": 0, "fast_path": 0, "hint_failed": 0, "branch_wrong": 0, "max_dpx": 0.0,
               "tri_dist_mm_median": None}
        dpx, tri = [], []
        for X, uv in zip(pts, pix):
            uvn = uv + sigma * rng.normal(size=uv.shape)
            los = [cam.lineOfSight(lpt.Pt2D(float(u), float(v))) for cam, (u, v) in zip(cams, uvn)]
            Xh = midpoint(los[0], los[1])
            tri.append(float(np.linalg.norm(Xh - X)))
            P = cstb.pt3(lpt, Xh)
            for c, cam in enumerate(cams):
                rec["pairs"] += 1
                okh, h = cam.hintFromLineOfSight(los[c], P)
                if not okh:
                    rec["hint_failed"] += 1
                    continue
                try:
                    slow = cam.project(P)
                    slow_ok = True
                except RuntimeError:
                    slow_ok = False
                fast_ok, fast, h2, _e = cam.projectWithHint(P, h)
                if fast_ok and h2.is_far != h.is_far:
                    rec["branch_wrong"] += 1
                g, fr = geos[c], frames[c]
                core = rc.project_cylinder_from(g["n"], g["R"], cstb.to_local(fr, g["pinhole"]),
                                                cstb.to_local(fr, Xh), h.lam, h.c_z, h.is_far, g["tol"], g["nmax"])
                rec["fast_path"] += core is not None
                if slow_ok and fast_ok:
                    rec["both_ok"] += 1
                    d = float(np.hypot(slow[0] - fast[0], slow[1] - fast[1]))
                    dpx.append(d)
                    rec["max_dpx"] = max(rec["max_dpx"], d)
                elif fast_ok:
                    rec["slow_fail_fast_ok"] += 1
                elif slow_ok:
                    rec["slow_ok_fast_fail"] += 1
                else:
                    rec["both_fail"] += 1
        rec["tri_dist_mm_median"] = float(np.median(tri))
        rec["dpx_p99"] = float(np.percentile(dpx, 99))
        rec["pass"] = bool(rec["max_dpx"] <= 1e-6 and rec["slow_ok_fast_fail"] == 0)
        out["levels"].append(rec)
        n = rec["pairs"]
        log(f"\n  pixel noise {sigma} px: {n} point-camera pairs; triangulated point {rec['tri_dist_mm_median']:.4f} mm "
            f"from the true one (median)")
        log(f"    hints made: {n - rec['hint_failed']}; fast path taken: {rec['fast_path']} "
            f"({100.0 * rec['fast_path'] / n:.3f} %); the hint's branch was not the image's: {rec['branch_wrong']}")
        log(f"    both succeed {rec['both_ok']}, slow fails/fast succeeds {rec['slow_fail_fast_ok']}, "
            f"slow succeeds/fast fails {rec['slow_ok_fast_fail']}, both fail {rec['both_fail']}")
        log(f"    |pixel difference| 99th percentile {rec['dpx_p99']:.3g}, max {rec['max_dpx']:.3g} px: "
            f"{'PASS' if rec['pass'] else 'FAIL'}")
    log.save(out)


if __name__ == "__main__":
    main()
