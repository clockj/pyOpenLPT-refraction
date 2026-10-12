"""F1b (diagnosis of F1-F3's largest pixel differences): are they the projection's own tolerance?

    python f1b_residuals.py [--build p1] [--base 20000] [--moves 10] [--cams DIR]

F1, F2 and F3 found the hinted projection and project() up to 3.4e-6 px apart (F1, tolerance 1e-9; 2.3e-8 px at
1e-12), against the plan's 1e-6 px.  Both accept a ray when its residual hypot(Phi - dphi, (Z - dz)/u) is below the
camFile's tolerance (u = 2^floor(log2 r_P) = 512 mm here; Newton aims at min(1e-10, tol)), so an axial residual of
1e-10 alone is 5e-8 mm, about 2e-6 px at 1 m with f = 38,000 px.
This repeats F1's moves (same sampling, tolerance 1e-9) and, for every pair more than 3e-7 px apart, recomputes both
rays in the core (refract_curved.project_cylinder / project_cylinder_from) and their residuals in the core's units
(phi_z), and predicts the pixel difference from the residual difference with the pixel's derivatives (finite
differences of project() in the point's azimuth and axial position).  Judged: the difference is explained by the
residuals (predicted within a factor 2 of the measured), and every residual is below the tolerance.
Writes f1b_residuals_out.txt and f1b_residuals.json.
"""
import argparse
import math
import os

import numpy as np

import cstb

BOX = ((-20.0, -15.0, -30.0), (25.0, 15.0, 35.0))


def core_case(lpt, g, fr, X, hint):
    rc = lpt.refract_curved
    pl, xl = cstb.to_local(fr, g["pinhole"]), cstb.to_local(fr, X)
    r_p, r_x = math.hypot(pl[0], pl[1]), math.hypot(xl[0], xl[1])
    u = 2.0 ** math.floor(math.log2(r_p))
    d = math.remainder(math.atan2(xl[1], xl[0]) - math.atan2(pl[1], pl[0]), 2 * math.pi)
    dphi, dz = abs(d), (xl[2] - pl[2]) / u
    Ru = [R / u for R in g["R"]]
    rays, _cov = rc.project_cylinder(g["n"], g["R"], pl, xl, g["tol"], g["nmax"])
    fast = rc.project_cylinder_from(g["n"], g["R"], pl, xl, hint.lam, hint.c_z, hint.is_far, g["tol"], g["nmax"])

    def resid(br, lam, cz):
        v = rc.phi_z(abs(lam) / u, cz, g["n"], Ru, r_p / u, r_x / u, br == "far")
        return (v[0] - dphi, v[1] - dz) if v is not None else (float("nan"), float("nan"))
    out = {"r_x_mm": r_x, "dz_mm": dz * u, "u_mm": u, "n_slow_rays": len(rays)}
    if rays:
        br, lam, cz, _d = rays[0]
        out["slow"] = {"branch": br, "lam": lam, "c_z": cz, "res_phi": resid(br, lam, cz)[0],
                       "res_z": resid(br, lam, cz)[1]}
    if fast is not None:
        br, lam, cz, _d = fast
        out["fast"] = {"branch": br, "lam": lam, "c_z": cz, "res_phi": resid(br, lam, cz)[0],
                       "res_z": resid(br, lam, cz)[1]}
    return out


def pixel_derivs(lpt, cam, g, fr, X, h_phi=1e-6, h_z=1e-4):
    """d(pixel)/d(azimuth) [px/rad] and d(pixel)/d(axial position) [px/mm] by central differences of project()."""
    c, uu, vv, ww = fr
    xl = np.array(cstb.to_local(fr, X))
    r = math.hypot(xl[0], xl[1])
    phi = math.atan2(xl[1], xl[0])

    def world(phi_, z_):
        loc = np.array([r * math.cos(phi_), r * math.sin(phi_), z_])
        return c + loc[0] * uu + loc[1] * vv + loc[2] * ww

    def px(Y):
        p = cam.project(cstb.pt3(lpt, Y))
        return np.array([p[0], p[1]])
    dp_phi = (px(world(phi + h_phi, xl[2])) - px(world(phi - h_phi, xl[2]))) / (2 * h_phi)
    dp_z = (px(world(phi, xl[2] + h_z)) - px(world(phi, xl[2] - h_z))) / (2 * h_z)
    return dp_phi, dp_z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="p1")
    ap.add_argument("--base", type=int, default=20000)
    ap.add_argument("--moves", type=int, default=10)
    ap.add_argument("--cams", default=cstb.CAMS_T2)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    lpt, build = cstb.use_build(args.build)
    log = cstb.Log(os.path.join(cstb.HERE, "f1b_residuals"))
    rng = np.random.default_rng(args.seed)
    files = cstb.cams_with_tol(args.cams, 1e-9)
    cams = [lpt.loadCameraModel(f) for f in files]
    geos = [cstb.pincyl_geometry(f) for f in files]
    frames = [cstb.port_frame(g["center"], g["axis"]) for g in geos]
    log(f"F1b: residuals behind F1's largest pixel differences; build {build['name']}; cameras {args.cams} at tol 1e-9")
    base, _pix = cstb.sample_shared(lpt, cams, args.base, BOX, rng)   # the same sampling as F1 (same seed)
    cases = []
    n_pairs = 0
    for X in base:
        hints = [cam.projectWithHint(cstb.pt3(lpt, X), lpt.ProjectionHint())[2] for cam in cams]
        for _ in range(args.moves):
            step = 10.0 ** rng.uniform(-3.0, np.log10(0.5))
            d = rng.normal(size=3)
            Xm = X + step * d / np.linalg.norm(d)
            P = cstb.pt3(lpt, Xm)
            for c, cam in enumerate(cams):
                n_pairs += 1
                slow = cam.project(P)
                ok, fast, _h, _e = cam.projectWithHint(P, hints[c])
                dpx = math.hypot(slow[0] - fast[0], slow[1] - fast[1])
                if dpx > 3e-7:
                    cases.append({"cam": c, "X": Xm, "hint": (hints[c].lam, hints[c].c_z, hints[c].is_far),
                                  "dpx": dpx, "d_uv": (fast[0] - slow[0], fast[1] - slow[1])})
    log(f"  {n_pairs} point-camera pairs; {len(cases)} more than 3e-7 px apart")
    rows = []
    for k in cases:
        c = k["cam"]
        h = lpt.ProjectionHint()
        h.lam, h.c_z, h.is_far, h.valid = k["hint"][0], k["hint"][1], k["hint"][2], True
        info = core_case(lpt, geos[c], frames[c], k["X"], h)
        dp_phi, dp_z = pixel_derivs(lpt, cams[c], geos[c], frames[c], k["X"])
        s, f = info.get("slow"), info.get("fast")
        pred = None
        if s and f:
            dres_phi = f["res_phi"] - s["res_phi"]
            dres_z_mm = (f["res_z"] - s["res_z"]) * info["u_mm"]
            # a ray whose sweep is short by e lands where a point at azimuth phi - e would: pixel moves by -e dp/dphi
            pred = -(dres_phi * dp_phi + dres_z_mm * dp_z)
        row = {"cam": c + 1, "X": k["X"].tolist(), "dpx": k["dpx"], "d_uv": k["d_uv"], **info,
               "dpix_dphi_px_per_rad": dp_phi.tolist(), "dpix_dz_px_per_mm": dp_z.tolist(),
               "predicted_d_uv": pred.tolist() if pred is not None else None}
        rows.append(row)
    rows.sort(key=lambda r: -r["dpx"])
    explained = 0
    worst_res = 0.0
    for r in rows:
        ratio = None
        if r["predicted_d_uv"] is not None:
            pm = math.hypot(*r["predicted_d_uv"])
            ratio = pm / r["dpx"] if r["dpx"] > 0 else None
            explained += (ratio is not None and 0.5 <= ratio <= 2.0)
        for side in ("slow", "fast"):
            if side in r:
                worst_res = max(worst_res, math.hypot(r[side]["res_phi"], r[side]["res_z"]))
        r["pred_over_measured"] = ratio
    for r in rows[:12]:
        s, f = r.get("slow", {}), r.get("fast", {})
        log(f"  cam{r['cam']} r_X {r['r_x_mm']:.2f} mm, dz {r['dz_mm']:.2f} mm: |d| {r['dpx']:.3g} px "
            f"(du {r['d_uv'][0]:.2e}, dv {r['d_uv'][1]:.2e}); residuals slow (phi {s.get('res_phi', float('nan')):.2e}, "
            f"z {s.get('res_z', float('nan')):.2e}), fast (phi {f.get('res_phi', float('nan')):.2e}, "
            f"z {f.get('res_z', float('nan')):.2e}); predicted (du, dv) "
            f"({r['predicted_d_uv'][0]:.2e}, {r['predicted_d_uv'][1]:.2e}); ratio {r['pred_over_measured']:.2f}")
    log(f"\n  explained (predicted within a factor 2): {explained} of {len(rows)}; largest residual of either ray "
        f"{worst_res:.3g} (the tolerance 1e-9)")
    out = {"build": build, "pairs": n_pairs, "cases": rows, "explained": explained, "largest_residual": worst_res,
           "pass": bool(explained == len(rows) and worst_res < 1e-9)}
    log(f"  judged: {'PASS' if out['pass'] else 'FAIL'}")
    log.save(out)


if __name__ == "__main__":
    main()
