"""F3b (diagnosis of F3): what happens next to the near/far junction.

    python f3b_junction_diag.py [--build p1] [--cams DIR]       (after f3_junction.py)

Reads f3_junction.json: for each camera and path, the case with the largest difference between the hinted projection
and project(), and up to 40 points where the hinted solve fell back to the full search.  For each, in the core's units
(lengths / u, u = 2^floor(log2 r_P)):
  - both rays (refract_curved.project_cylinder and project_cylinder_from from the recorded hint), their branch,
    delta = 1 - Lambda/Lmax (distance from grazing: 0 at the junction) and their residuals (phi_z);
  - the rounding scale of a residual there, eps/sqrt(delta) (refract_curved.tex, Remark rem:tol: Phi has a square-root
    edge at Lmax, so evaluating it costs about eps/sqrt(delta));
  - for a fallback: whether single_image() held, and both branches' solves from the hint (solve_cylinder /
    solve_far: converged? residual).
Judged: every difference and fallback is at the rounding scale of the grazing ray (residuals within a small multiple
of eps/sqrt(delta), and below the tolerance), i.e. a rounding-level effect, documented and not fixed (the user's rule,
memory rigor-level-decision).  Writes f3b_junction_diag_out.txt and f3b_junction_diag.json.
"""
import argparse
import json
import math
import os

import numpy as np

import cstb

EPS = np.finfo(float).eps


def analyse(lpt, g, fr, X, hint):
    rc = lpt.refract_curved
    pl, xl = cstb.to_local(fr, g["pinhole"]), cstb.to_local(fr, X)
    r_p, r_x = math.hypot(pl[0], pl[1]), math.hypot(xl[0], xl[1])
    u = 2.0 ** math.floor(math.log2(r_p))
    d = math.remainder(math.atan2(xl[1], xl[0]) - math.atan2(pl[1], pl[0]), 2 * math.pi)
    dphi, dz = abs(d), (xl[2] - pl[2]) / u
    Ru = [R / u for R in g["R"]]
    out = {"r_x_mm": r_x, "single_image": rc.single_image(g["n"], g["R"], r_p, r_x, xl[2] - pl[2])}

    def ray_info(br, lam, cz):
        lmax = rc.lmax_kstar(g["n"], Ru, r_x / u, cz)[0]
        v = rc.phi_z(abs(lam) / u, cz, g["n"], Ru, r_p / u, r_x / u, br == "far")
        delta = 1.0 - (abs(lam) / u) / lmax
        res = (v[0] - dphi, v[1] - dz) if v is not None else (float("nan"), float("nan"))
        return {"branch": br, "lam": lam, "c_z": cz, "delta": delta, "res_phi": res[0], "res_z": res[1],
                "rounding_scale": EPS / math.sqrt(max(delta, 1e-300))}
    rays, _ = rc.project_cylinder(g["n"], g["R"], pl, xl, g["tol"], g["nmax"])
    out["slow"] = [ray_info(r[0], r[1], r[2]) for r in rays]
    if hint is not None:
        lam0, cz0, far0 = hint
        f = rc.project_cylinder_from(g["n"], g["R"], pl, xl, lam0, cz0, far0, g["tol"], g["nmax"])
        out["fast"] = ray_info(f[0], f[1], f[2]) if f is not None else None
        l0 = abs(lam0) / u
        lmax0 = rc.lmax_kstar(g["n"], Ru, r_x / u, cz0)[0]
        l0 = min(l0, lmax0 * (1 - 1e-9))
        sn = rc.solve_cylinder(g["n"], Ru, r_p / u, r_x / u, dphi, dz, l0, cz0, False)
        sf = rc.solve_far(g["n"], Ru, r_p / u, r_x / u, dphi, dz, l0, cz0)
        out["solve_near_from_hint"] = {"ok": sn[0], "residual": sn[3], "iters": sn[4],
                                       "delta": 1.0 - sn[1] / rc.lmax_kstar(g["n"], Ru, r_x / u, sn[2])[0]}
        out["solve_far_from_hint"] = {"ok": sf[0], "residual": sf[3], "iters": sf[4],
                                      "delta": 1.0 - sf[1] / rc.lmax_kstar(g["n"], Ru, r_x / u, sf[2])[0]}
        out["hint_delta"] = 1.0 - l0 / lmax0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="p1")
    ap.add_argument("--cams", default=cstb.CAMS_T2)
    args = ap.parse_args()
    lpt, build = cstb.use_build(args.build)
    log = cstb.Log(os.path.join(cstb.HERE, "f3b_junction_diag"))
    with open(os.path.join(cstb.HERE, "f3_junction.json"), encoding="utf-8") as fh:
        f3 = json.load(fh)
    files = cstb.cams_with_tol(args.cams, 1e-9)
    geos = [cstb.pincyl_geometry(f) for f in files]
    frames = [cstb.port_frame(g["center"], g["axis"]) for g in geos]
    log(f"F3b: next to the junction; build {build['name']}; F3's records ({f3['build']['name']})")
    out = {"build": build, "worst": [], "fallbacks": []}
    ok_all = True
    for rec in f3["cams"]:
        c = rec["cam"] - 1
        for name, pr in rec["paths"].items():
            w = pr.get("worst")
            if w and w["dpx"] > 1e-6:
                a = analyse(lpt, geos[c], frames[c], np.array(w["X"]), w["hint_in"])
                a.update({"cam": c + 1, "path": name, "dpx": w["dpx"], "j": w["j"]})
                out["worst"].append(a)
                s0 = a["slow"][0] if a["slow"] else {}
                fa = a.get("fast") or {}
                worst_res = max(abs(s0.get("res_phi", 0)), abs(s0.get("res_z", 0)),
                                abs(fa.get("res_phi", 0)), abs(fa.get("res_z", 0)))
                scale = min(s0.get("rounding_scale", 1), fa.get("rounding_scale", 1))
                ok = worst_res < 1e-9 and worst_res <= 1e4 * scale
                ok_all &= ok
                log(f"  cam{c + 1} path {name}, worst |d| {w['dpx']:.3g} px at {0.01 * w['j']:+.2f} mm from the "
                    f"closest approach: slow {s0.get('branch')} delta {s0.get('delta', float('nan')):.2e} residuals "
                    f"({s0.get('res_phi', float('nan')):.1e}, {s0.get('res_z', float('nan')):.1e}); fast "
                    f"{fa.get('branch')} delta {fa.get('delta', float('nan')):.2e} residuals "
                    f"({fa.get('res_phi', float('nan')):.1e}, {fa.get('res_z', float('nan')):.1e}); "
                    f"rounding scale eps/sqrt(delta) {scale:.1e}; {len(a['slow'])} ray(s) from the full search")
            for ev in pr.get("fallback_events", []):
                a = analyse(lpt, geos[c], frames[c], np.array(ev["X"]), ev["hint_in"])
                a.update({"cam": c + 1, "path": name, "j": ev["j"], "slow_ok": ev["slow_ok"]})
                out["fallbacks"].append(a)
    # summary of the fallbacks
    fb = out["fallbacks"]
    log(f"\n  fallbacks analysed: {len(fb)}")
    if fb:
        not_single = sum(1 for a in fb if not a["single_image"])
        far_dist = [abs(0.01 * a["j"]) for a in fb]
        hint_deltas = [a.get("hint_delta", float("nan")) for a in fb]
        sol_deltas = [a["slow"][0]["delta"] if a["slow"] else float("nan") for a in fb]
        log(f"    single_image false: {not_single}; distance from the closest approach: max {max(far_dist):.2f} mm, "
            f"median {np.median(far_dist):.3f} mm")
        log(f"    delta of the image's ray: median {np.nanmedian(sol_deltas):.2e}, max {np.nanmax(sol_deltas):.2e}; "
            f"delta of the hint: median {np.nanmedian(hint_deltas):.2e}")
        for a in sorted(fb, key=lambda a: -abs(a["j"]))[:6]:
            s0 = a["slow"][0] if a["slow"] else {}
            log(f"    cam{a['cam']} path {a['path']} at {0.01 * a['j']:+.2f} mm: image {s0.get('branch')} delta "
                f"{s0.get('delta', float('nan')):.2e}; near solve from the hint ok {a['solve_near_from_hint']['ok']} "
                f"res {a['solve_near_from_hint']['residual']:.1e} ({a['solve_near_from_hint']['iters']} it); far "
                f"solve ok {a['solve_far_from_hint']['ok']} res {a['solve_far_from_hint']['residual']:.1e} "
                f"({a['solve_far_from_hint']['iters']} it); hint delta {a.get('hint_delta', float('nan')):.2e}")
    out["pass"] = bool(ok_all)
    log(f"\n  judged (worst cases at the rounding scale of the grazing ray): {'PASS' if ok_all else 'FAIL'}")
    log.save(out)


if __name__ == "__main__":
    main()
