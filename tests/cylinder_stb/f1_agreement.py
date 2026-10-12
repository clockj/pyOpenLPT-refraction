"""F1 (plan, Phase 1): the hinted projection against project().

    python f1_agreement.py [--build p1] [--base 20000] [--moves 10] [--cams DIR]

Cameras: 6 Oct T2's delivered PINCYL camFiles (data/cams_T2, copied from X:), written with the tracking tolerance
1e-9 (C4); a second pass uses them as delivered (1e-12).  Points: BASE random points that both cameras see on their
sensors (cstb.sample_shared), each moved MOVES times by a random step of 1 um - 0.5 mm (log-uniform length, random
direction).  For every moved point and camera:
  - slow: Camera.project() (the full search);
  - fast: Camera.projectWithHint() from the hint the base point's own projection left (as a tracked particle would);
  - whether the fast path was taken: refract_curved.project_cylinder_from() on the same input returned a ray, and if
    not, whether single_image() held (else the solve failed).
Judged by (plan): pixels within 1e-6 px of project() wherever both succeed; the same failures; the fallback rate
reported.  Writes f1_agreement_out.txt and f1_agreement.json beside itself.
"""
import argparse
import os
import time

import numpy as np

import cstb

BOX = ((-20.0, -15.0, -30.0), (25.0, 15.0, 35.0))   # world mm; the tube's axis is world y


def run_pass(lpt, cam_files, base_pts, rng, moves, log, tag):
    cams = [lpt.loadCameraModel(f) for f in cam_files]
    geos = [cstb.pincyl_geometry(f) for f in cam_files]
    frames = [cstb.port_frame(g["center"], g["axis"]) for g in geos]
    rc = lpt.refract_curved
    rec = {"tag": tag, "tol": geos[0]["tol"], "pairs": 0, "both_ok": 0, "slow_fail_fast_ok": 0,
           "slow_ok_fast_fail": 0, "both_fail": 0, "fast_path": 0, "fallback_not_single": 0,
           "fallback_solve": 0, "max_dpx": 0.0, "dpx_hist": {}, "worst": None}
    dpx_all = []
    t0 = time.time()
    for i, X in enumerate(base_pts):
        hints = []
        for cam in cams:
            ok, _p, h, _e = cam.projectWithHint(cstb.pt3(lpt, X), lpt.ProjectionHint())
            hints.append(h if ok else None)
        for _ in range(moves):
            step = 10.0 ** rng.uniform(-3.0, np.log10(0.5))
            d = rng.normal(size=3)
            Xm = X + step * d / np.linalg.norm(d)
            P = cstb.pt3(lpt, Xm)
            for c, cam in enumerate(cams):
                if hints[c] is None:
                    continue
                rec["pairs"] += 1
                try:
                    slow = cam.project(P)
                    slow_ok = True
                except RuntimeError:
                    slow_ok = False
                fast_ok, fast, _h, _e = cam.projectWithHint(P, hints[c])
                g, fr = geos[c], frames[c]
                core = rc.project_cylinder_from(g["n"], g["R"], cstb.to_local(fr, g["pinhole"]),
                                                cstb.to_local(fr, Xm), hints[c].lam, hints[c].c_z,
                                                hints[c].is_far, g["tol"], g["nmax"])
                if core is not None:
                    rec["fast_path"] += 1
                else:
                    pl, xl = cstb.to_local(fr, g["pinhole"]), cstb.to_local(fr, Xm)
                    single = rc.single_image(g["n"], g["R"], float(np.hypot(pl[0], pl[1])),
                                             float(np.hypot(xl[0], xl[1])), xl[2] - pl[2])
                    rec["fallback_solve" if single else "fallback_not_single"] += 1
                if slow_ok and fast_ok:
                    rec["both_ok"] += 1
                    dpx = float(np.hypot(slow[0] - fast[0], slow[1] - fast[1]))
                    dpx_all.append(dpx)
                    if dpx > rec["max_dpx"]:
                        rec["max_dpx"] = dpx
                        rec["worst"] = {"X": Xm.tolist(), "cam": c + 1, "step_mm": step,
                                        "slow": [slow[0], slow[1]], "fast": [fast[0], fast[1]],
                                        "fast_path": core is not None}
                elif fast_ok:
                    rec["slow_fail_fast_ok"] += 1
                elif slow_ok:
                    rec["slow_ok_fast_fail"] += 1
                else:
                    rec["both_fail"] += 1
        if (i + 1) % 2000 == 0:
            log(f"    {tag}: {i + 1} / {len(base_pts)} base points, {time.time() - t0:.0f} s")
    dpx_all = np.array(dpx_all)
    edges = [0.0, 1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-4, np.inf]
    h = np.histogram(dpx_all, bins=edges)[0]
    rec["dpx_hist"] = {f"[{a:g}, {b:g})": int(n) for a, b, n in zip(edges[:-1], edges[1:], h)}
    rec["dpx_median"] = float(np.median(dpx_all)) if len(dpx_all) else None
    rec["dpx_p99"] = float(np.percentile(dpx_all, 99)) if len(dpx_all) else None
    rec["seconds"] = time.time() - t0
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="p1")
    ap.add_argument("--base", type=int, default=20000)
    ap.add_argument("--moves", type=int, default=10)
    ap.add_argument("--cams", default=cstb.CAMS_T2)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    lpt, build = cstb.use_build(args.build)
    log = cstb.Log(os.path.join(cstb.HERE, "f1_agreement"))
    rng = np.random.default_rng(args.seed)
    log(f"F1: hinted projection against project(); build {build['name']} ({build['git_head']} + diff "
        f"{(build['git_diff_sha256'] or '')[:12]}); cameras {args.cams}")
    cams9 = cstb.cams_with_tol(args.cams, 1e-9)
    cams_lpt = [lpt.loadCameraModel(f) for f in cams9]
    base, _pix = cstb.sample_shared(lpt, cams_lpt, args.base, BOX, rng)
    r = np.hypot(base[:, 0], base[:, 2])
    log(f"  {len(base)} base points seen by both cameras; radius from the axis {r.min():.1f}-{r.max():.1f} mm, "
        f"along it (y) {base[:, 1].min():.1f} to {base[:, 1].max():.1f} mm; {args.moves} moves each")
    out = {"build": build, "cams": args.cams, "base": len(base), "moves": args.moves, "box": BOX, "passes": []}
    for tag, files, nb in (("tol 1e-9 (tracking)", cams9, len(base)),
                           ("tol 1e-12 (as delivered)", [os.path.join(args.cams, f"cam{c}.txt") for c in (1, 2)],
                            len(base) // 4)):
        rec = run_pass(lpt, files, base[:nb], rng, args.moves, log, tag)
        out["passes"].append(rec)
        n = rec["pairs"]
        log(f"\n  {tag}: {n} point-camera pairs, {rec['seconds']:.0f} s")
        log(f"    fast path taken: {rec['fast_path']} ({100.0 * rec['fast_path'] / n:.3f} %); fallbacks: "
            f"{rec['fallback_not_single']} not certified single-image, {rec['fallback_solve']} solve not converged")
        log(f"    both succeed: {rec['both_ok']}; slow fails, fast succeeds: {rec['slow_fail_fast_ok']}; "
            f"slow succeeds, fast fails: {rec['slow_ok_fast_fail']}; both fail: {rec['both_fail']}")
        log(f"    |pixel difference|: median {rec['dpx_median']:.3g}, 99th percentile {rec['dpx_p99']:.3g}, "
            f"max {rec['max_dpx']:.3g} px; histogram {rec['dpx_hist']}")
        if rec["worst"]:
            log(f"    worst: {rec['worst']}")
        ok = rec["max_dpx"] <= 1e-6 and rec["slow_ok_fast_fail"] == 0
        rec["pass"] = bool(ok)
        log(f"    judged: pixels within 1e-6 px and no new failure: {'PASS' if ok else 'FAIL'}")
    log.save(out)


if __name__ == "__main__":
    main()
