"""T1: the forward projection's derivatives (implicit differentiation) against central differences, on the cylinder.

For random tubes, cameras (with and without distortion) and points (inside the fold radius, and beyond it next to
each of a point's images), `cm.project(..., jac=True)` gives d(u, v)/d(camera) in the order of `cm.CAM_PARAMS`
(rotation increment w with R <- R exp([w]x), centre C, log f, cx, cy, k1, k2, p1, p2, k3) and d(u, v)/dX.  Each
column is compared with a central difference of the forward projection itself (Newton restarted from the
unperturbed solution, converged to round-off), applying the perturbation exactly as the calibration's manifold
update does.

Numeric derivative: Richardson-extrapolated central differences, (4 D(h/2) - D(h)) / 3, error O(h^4).  The pixel
is exactly linear in each distortion coefficient (the ray, hence (x, y), does not depend on it), so those use a
large step (0.1), where a small one would sit on round-off for columns as small as f x r^6.

Error measure: per column, |analytic - numeric| over the column's RMS size across the rig's points (so a column
that is near zero at one point is not divided by zero).  Criterion (plan section 7): at most 1e-6, on points inside
the fold radius r_c.  Beyond r_c, next to the fold, the projection curves sharply and finite differences lose
accuracy, so there the numeric derivative is taken at two steps (h and h/4) and the better of the two must agree
to 1e-6 (truncation dominates at h next to the fold, round-off at h/4 elsewhere).
Outputs: t1_derivatives_out.txt, t1_derivatives.json.
"""
import json
import time

import numpy as np

import rigs
from rigs import cm

OUT = rigs.HERE / "t1_derivatives"
log = rigs.Tee(str(OUT) + "_out.txt")
STEPS = {"w": 1e-5, "C": 1e-2, "logf": 1e-6, "c": 1e-3, "k": 0.1, "X": 1e-3}


def perturbed(cam: cm.Camera, name: str, h: float) -> cm.Camera:
    c = cam.copy()
    i = cm.CAM_PARAMS.index(name)
    if name in ("w1", "w2", "w3"):
        w = np.zeros(3)
        w[i] = h
        c.R = cam.R @ cm.exp_so3(w)
    elif name in ("C1", "C2", "C3"):
        c.C[i - 3] += h
    elif name == "logf":
        c.f = cam.f * np.exp(h)
    elif name == "cx":
        c.cx += h
    elif name == "cy":
        c.cy += h
    else:
        c.dist[i - 9] += h
    return c


def step_of(name):
    if name.startswith("w"):
        return STEPS["w"]
    if name.startswith("C"):
        return STEPS["C"]
    if name == "logf":
        return STEPS["logf"]
    if name in ("cx", "cy"):
        return STEPS["c"]
    return STEPS["k"]


def proj(cam, tube, X, xy0):
    rw = cm.rows([cam], np.zeros(len(X), int))
    return cm.project(rw, X, xy0, tube, tol=1e-16, max_iter=60)


def main():
    rng = np.random.default_rng(7)
    out = {"rigs": []}
    worst = 0.0
    worst_col = None
    worst_beyond_h = worst_beyond_h4 = worst_beyond_best = 0.0
    t0 = time.time()
    for k in range(12):
        tube = rigs.CASE_TUBE if k % 2 == 0 else cm.Tube(radii=(60.0, 56.0, 55.0), n=(1.0, 1.52, 1.47, 1.333))
        D = rng.uniform(2.0, 11.0) * tube.r_outer
        f = float(rng.uniform(0.6, 1.4) * 2048 / (2 * tube.r_inner / D))
        dist = np.zeros(5)
        if k % 3 == 1:
            dist = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.05, 0.05), rng.uniform(-1e-3, 1e-3),
                             rng.uniform(-1e-3, 1e-3), rng.uniform(-0.01, 0.01)])
        aim = np.array([rng.uniform(-10, 10), rng.uniform(-10, 10), rng.uniform(-10, 10)])
        cam = rigs.ring_camera(rng.uniform(-180, 180), D, f, 2048, 2048, height_mm=rng.uniform(-30, 30), aim=aim,
                               roll_deg=rng.uniform(-180, 180), dist=dist)
        cpp = cm.to_cpp(cam, tube, tol=1e-13, nmax=300)
        rc = tube.fold_radius()
        # points: inside the fold radius, and beyond it (each of their images)
        m = 300
        rr = np.sqrt(rng.uniform(0, (0.98 * rc) ** 2, m))
        th = rng.uniform(0, 2 * np.pi, m)
        P_in = np.column_stack([rr * np.cos(th), rng.uniform(-30, 30, m), rr * np.sin(th)])
        uv_in = cm.cpp_project(cpp, P_in)
        keep = np.isfinite(uv_in[:, 0]) & (uv_in[:, 0] >= 0) & (uv_in[:, 0] < 2048) & (uv_in[:, 1] >= 0) & (
            uv_in[:, 1] < 2048)
        P, UV, region = list(P_in[keep]), list(uv_in[keep]), ["inside"] * int(keep.sum())
        if rc < 0.99 * tube.r_inner:
            rr = np.sqrt(rng.uniform(rc ** 2, (0.99 * tube.r_inner) ** 2, m))
            th = rng.uniform(0, 2 * np.pi, m)
            P_b = np.column_stack([rr * np.cos(th), rng.uniform(-30, 30, m), rr * np.sin(th)])
            for p, ims in zip(P_b, cm.cpp_images(cpp, P_b)):
                for q in ims:
                    if 0 <= q[0] < 2048 and 0 <= q[1] < 2048:
                        P.append(p)
                        UV.append(q)
                        region.append("beyond")
        P, UV, region = np.array(P), np.array(UV), np.array(region)
        rw = cm.rows([cam], np.zeros(len(P), int))
        base = cm.project(rw, P, cm.undistort(UV, rw), tube, jac=True, tol=1e-16, max_iter=60)
        good = base.ok & (np.linalg.norm(base.uv - UV, axis=1) < 1e-6)
        P, UV, region = P[good], UV[good], region[good]
        xy0 = base.xy[good]
        Ja = base.J_cam[good]
        JX = base.J_X[good]
        errs = {}

        def fd(fun, h):
            """Richardson-extrapolated central difference of fun(+-h) -> pixels; returns (deriv, ok)."""
            def cd(hh):
                pp, pm = fun(+hh), fun(-hh)
                return (pp.uv - pm.uv) / (2 * hh), pp.ok & pm.ok
            d1, ok1 = cd(h)
            d2, ok2 = cd(h / 2)
            return (4 * d2 - d1) / 3, ok1 & ok2

        def score(name, col, fun, h):
            num, ok = fd(fun, h)
            num4, ok4 = fd(fun, h / 4)
            ok &= ok4
            scale = np.sqrt(np.mean(np.sum(col[ok] ** 2, axis=1))) + 1e-300
            e = np.linalg.norm(col[ok] - num[ok], axis=1) / scale
            e4 = np.linalg.norm(col[ok] - num4[ok], axis=1) / scale
            ins = region[ok] == "inside"
            bey = ~ins
            errs[name] = {"max_rel_inside": float(e[ins].max()) if ins.any() else 0.0,
                          "max_rel_beyond_h": float(e[bey].max()) if bey.any() else None,
                          "max_rel_beyond_h4": float(e4[bey].max()) if bey.any() else None,
                          "rms_size": float(scale), "n_fail": int((~ok).sum())}

        for name in cm.CAM_PARAMS:
            col = Ja[:, :, cm.CAM_PARAMS.index(name)]
            score(name, col, lambda hh, name=name: proj(perturbed(cam, name, hh), tube, P, xy0), step_of(name))
        for i in range(3):
            dP = np.zeros(3)
            dP[i] = 1.0
            score(f"X{i + 1}", JX[:, :, i], lambda hh, dP=dP: proj(cam, tube, P + hh * dP, xy0), STEPS["X"])
        rig_worst = max(v["max_rel_inside"] for v in errs.values())
        wc = max(errs, key=lambda q: errs[q]["max_rel_inside"])
        if rig_worst > worst:
            worst, worst_col = rig_worst, (k, wc)
        bey = [(v["max_rel_beyond_h"], v["max_rel_beyond_h4"]) for v in errs.values()
               if v["max_rel_beyond_h"] is not None]
        if bey:
            worst_beyond_h = max(worst_beyond_h, max(b[0] for b in bey))
            worst_beyond_h4 = max(worst_beyond_h4, max(b[1] for b in bey))
            worst_beyond_best = max(worst_beyond_best, max(min(b) for b in bey))
        out["rigs"].append({"rig": k, "tube": {"radii": tube.radii, "n": tube.n}, "D": D, "f": f,
                            "distorted": bool(np.any(dist)), "points_inside": int(np.sum(region == "inside")),
                            "points_beyond": int(np.sum(region == "beyond")), "columns": errs})
        log(f"rig {k:2d}: D {D:7.1f} mm f {f:8.1f} dist {'y' if np.any(dist) else 'n'} | points inside "
            f"{np.sum(region == 'inside'):3d} beyond {np.sum(region == 'beyond'):3d} | inside: worst {wc} "
            f"{errs[wc]['max_rel_inside']:.1e}; " + " ".join(f"{q}:{v['max_rel_inside']:.0e}" for q, v in errs.items())
            + (f" | beyond: worst at h {max(b[0] for b in bey):.1e}, at h/4 {max(b[1] for b in bey):.1e}"
               if bey else ""))
    out["worst_rel_inside"] = worst
    out["worst_column"] = worst_col
    out["beyond_worst_rel_at_h"] = worst_beyond_h
    out["beyond_worst_rel_at_h_over_4"] = worst_beyond_h4
    out["pass"] = bool(worst < 1e-6)
    out["beyond_worst_rel_best_step"] = worst_beyond_best
    out["beyond_pass"] = bool(worst_beyond_best < 1e-6)
    out["seconds"] = time.time() - t0
    log("")
    log(f"inside r_c: worst relative error over all columns and rigs {worst:.3e} (rig, column {worst_col})")
    log(f"beyond r_c: worst relative error {worst_beyond_h:.3e} at the base steps, {worst_beyond_h4:.3e} at a "
        f"quarter of them, {worst_beyond_best:.3e} at the better of the two (per column)")
    log(f"  {'PASS' if out['pass'] else 'FAIL'}  inside r_c: derivatives agree with central differences to 1e-6")
    log(f"  {'PASS' if out['beyond_pass'] else 'FAIL'}  beyond r_c: derivatives agree to 1e-6 at the better of "
        f"the two steps")
    log(f"time {out['seconds']:.1f} s")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
