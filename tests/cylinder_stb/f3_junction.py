"""F3 (plan, Phase 1): particles moving through a camera's near/far junction, with chained hints.

    python f3_junction.py [--build p1] [--lines 200] [--cams DIR]

A point lies on its camera's near branch when it comes before its ray's closest approach to the axis, and on the
far branch after it; the junction is where the ray grazes the point's own radius.  For each camera and LINES lines of
sight (through the pixels of random points both cameras see), two paths of 401 points 0.01 mm apart, centred on the
line's closest approach to the axis:
  - A: along the line of sight itself: every point has the same pixel;
  - B: along a random direction.
Each point is projected with the hint the previous point's projection left (as a tracked particle), and with
project().  Judged by (plan): identical pixels along the path (within 1e-6 px of project()); fallbacks to the full
search only next to the junction (reported with their distance from it).  Path A also checks the pixel stays
constant (against the line's own pixel).  Writes f3_junction_out.txt and f3_junction.json.
"""
import argparse
import os

import numpy as np

import cstb

BOX = ((-20.0, -15.0, -30.0), (25.0, 15.0, 35.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="p1")
    ap.add_argument("--lines", type=int, default=200)
    ap.add_argument("--cams", default=cstb.CAMS_T2)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    lpt, build = cstb.use_build(args.build)
    log = cstb.Log(os.path.join(cstb.HERE, "f3_junction"))
    rng = np.random.default_rng(args.seed)
    files = cstb.cams_with_tol(args.cams, 1e-9)
    cams = [lpt.loadCameraModel(f) for f in files]
    geos = [cstb.pincyl_geometry(f) for f in files]
    frames = [cstb.port_frame(g["center"], g["axis"]) for g in geos]
    rc = lpt.refract_curved
    log(f"F3: the near/far junction with chained hints; build {build['name']} ({build['git_head']} + diff "
        f"{(build['git_diff_sha256'] or '')[:12]}); cameras {args.cams} at tol 1e-9")
    pts, pix = cstb.sample_shared(lpt, cams, args.lines, BOX, rng)
    out = {"build": build, "lines": args.lines, "cams": []}
    for c, cam in enumerate(cams):
        g, fr = geos[c], frames[c]
        c_, u_, v_, w_ = fr
        rec = {"cam": c + 1, "paths": {"A": {}, "B": {}}}
        for name in ("A", "B"):
            n_pts = n_fast = n_both = 0
            max_d, max_const = 0.0, 0.0
            fb_dist, branches_seen = [], set()
            new_fail = 0
            worst, fb_events = None, []
            for k in range(args.lines):
                uv = pix[k][c]
                los = cam.lineOfSight(lpt.Pt2D(float(uv[0]), float(uv[1])))
                p = np.array([los.pt[i] for i in range(3)])
                d = np.array([los.unit_vector[i] for i in range(3)])
                # closest approach to the axis along p + t d, in the port frame
                pl, dl = np.array(cstb.to_local(fr, p)), np.array([d @ u_, d @ v_, d @ w_])
                tc = -(pl[0] * dl[0] + pl[1] * dl[1]) / (dl[0] ** 2 + dl[1] ** 2)
                Xc = p + tc * d
                e = d if name == "A" else rng.normal(size=3)
                e = e / np.linalg.norm(e)
                hint = lpt.ProjectionHint()
                for j in range(-200, 201):
                    X = Xc + 0.01 * j * e
                    P = cstb.pt3(lpt, X)
                    h_in = hint
                    fast_ok, fast, hint, _e = cam.projectWithHint(P, h_in)
                    try:
                        slow = cam.project(P)
                        slow_ok = True
                    except RuntimeError:
                        slow_ok = False
                    n_pts += 1
                    if slow_ok and not fast_ok:
                        new_fail += 1
                    if fast_ok:
                        branches_seen.add(hint.is_far)
                    if h_in.valid:
                        core = rc.project_cylinder_from(g["n"], g["R"], cstb.to_local(fr, g["pinhole"]),
                                                        cstb.to_local(fr, X), h_in.lam, h_in.c_z, h_in.is_far,
                                                        g["tol"], g["nmax"])
                        if core is not None:
                            n_fast += 1
                        else:
                            fb_dist.append(abs(0.01 * j))
                            fb_events.append({"X": X.tolist(), "j": j, "line": k, "slow_ok": slow_ok,
                                              "fast_ok": fast_ok,
                                              "hint_in": [h_in.lam, h_in.c_z, h_in.is_far]})
                            # keep the 40 farthest from the closest approach
                            fb_events.sort(key=lambda e: -abs(e["j"]))
                            del fb_events[40:]
                    if slow_ok and fast_ok:
                        n_both += 1
                        dd = float(np.hypot(slow[0] - fast[0], slow[1] - fast[1]))
                        if dd > max_d:
                            worst = {"X": X.tolist(), "j": j, "line": k, "slow": [slow[0], slow[1]],
                                     "fast": [fast[0], fast[1]], "dpx": dd,
                                     "hint_in": [h_in.lam, h_in.c_z, h_in.is_far] if h_in.valid else None,
                                     "hint_out": [hint.lam, hint.c_z, hint.is_far]}
                        max_d = max(max_d, dd)
                        if name == "A":
                            max_const = max(max_const, float(np.hypot(fast[0] - uv[0], fast[1] - uv[1])))
            pr = {"points": n_pts, "with_hint": n_pts - args.lines, "fast_path": n_fast, "both_ok": n_both,
                  "new_failures": new_fail, "max_dpx_vs_project": max_d, "fallbacks": len(fb_dist),
                  "fallback_dist_mm_max": max(fb_dist) if fb_dist else None,
                  "branches_seen": sorted(int(b) for b in branches_seen)}
            if name == "A":
                pr["max_dpx_from_line_pixel"] = max_const
            pr["worst"] = worst
            pr["fallback_events"] = fb_events
            pr["pass"] = bool(max_d <= 1e-6 and new_fail == 0)
            rec["paths"][name] = pr
            log(f"  cam{c + 1} path {name}: {n_pts} points ({pr['with_hint']} with a hint), fast path {n_fast}, "
                f"fallbacks {len(fb_dist)}" + (f" (at most {pr['fallback_dist_mm_max']:.2f} mm from the junction)"
                                                 if fb_dist else "")
                + f"; branches seen {pr['branches_seen']}; max |fast - project()| {max_d:.3g} px"
                + (f"; max |pixel - the line's pixel| {max_const:.3g} px" if name == "A" else "")
                + f"; new failures {new_fail}: {'PASS' if pr['pass'] else 'FAIL'}")
        out["cams"].append(rec)
    log.save(out)


if __name__ == "__main__":
    main()
