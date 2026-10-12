"""T0: the fast model (cyl_model) against pyOpenLPT's C++ PINCYL camera, on random rigs.

Each rig has a random tube (2-4 media), a random world frame (axis tilted up to 30 deg and offset up to 50 mm, so
the conversion to the tube frame is exercised), a random camera (distance, azimuth, height, aim, roll, f) and,
on half the rigs, random distortion.

Checks (plan section 7):
  back-projection  my trace vs the C++ lineOfSight, for pixels over the sensor.  The C++ undistorts only to 1e-5 px
                   (UNDISTORT_EPS), so with distortion the comparison uses an exact copy of its fixed-point
                   iteration (`undistort_like_cpp`); the library's own exact undistortion is compared to it
                   separately.  The rays are compared as lines, in pixels: the C++ ray's mid-chord point is
                   reprojected by the fast model from the C++'s own normalised coordinates.  Criterion 1e-8 px.
                   The plan's 1e-9 mm is below the C++'s own round-off for a distant camera: its lineOfSight forms
                   the direction as normalise((R^-1 (x, y, 1) + C) - C), the difference of two points |C| ~ 1 m
                   from the origin and 1 mm apart, which costs ~|C| eps / 1 mm ~ 1e-13 rad, i.e. ~1e-9 px at
                   f = 13,000 (checked by emulating that step).  Raw millimetre differences are logged too.
  forward          (a) the single-image region r < r_c: my Newton from the C++ pixel + 5 px, and from a crude
                   pinhole start, vs the C++ primary image: within 1e-6 px, every start converged.
                   (b) beyond r_c (diagnostic): a start next to each C++ image on the sensor must return an image
                   (that one, or another), never a false solution.
Outputs: t0_fast_vs_cpp_out.txt, t0_fast_vs_cpp.json beside this script.
"""
import json
import math
import time

import numpy as np

import rigs
from rigs import cm

OUT = rigs.HERE / "t0_fast_vs_cpp"
log = rigs.Tee(str(OUT) + "_out.txt")
UNDISTORT_EPS, UNDISTORT_MAX_ITER = 1e-5, 50


def undistort_like_cpp(uv, f, cx, cy, dist):
    """The C++ Camera::undistort for 5 coefficients, transcribed (fixed point, stops at 1e-5 px)."""
    k1, k2, p1, p2, k3 = dist
    out = np.empty((len(uv), 2))
    for i, (u, v) in enumerate(uv):
        x = x0 = (u - cx) / f
        y = y0 = (v - cy) / f
        if not np.any(np.abs(dist) > cm.SMALLNUMBER):
            out[i] = (x, y)
            continue
        err, it = np.inf, 0
        while err > UNDISTORT_EPS and it < UNDISTORT_MAX_ITER:
            r2 = x * x + y * y
            a1, a2, a3 = 2 * x * y, r2 + 2 * x * x, r2 + 2 * y * y
            icdist = 1 / (1 + ((k3 * r2 + k2) * r2 + k1) * r2)
            if icdist < 0:
                x, y = (u - cx) / f, (v - cy) / f
                break
            dx = p1 * a1 + p2 * a2
            dy = p1 * a3 + p2 * a1
            x = (x0 - dx) * icdist
            y = (y0 - dy) * icdist
            xd = cm.distort(np.array([[x, y]]), dist)[0]
            err = math.hypot(xd[0] * f + cx - u, xd[1] * f + cy - v)
            it += 1
        out[i] = (x, y)
    return out


def random_rig(rng, k):
    """A random tube, frame and camera.  Returns (tube, camera in the tube frame, world pose, axis)."""
    if k % 3 == 0:
        tube = rigs.CASE_TUBE
    else:
        R1 = rng.uniform(30, 150)
        walls = rng.uniform(1, 10, size=rng.integers(1, 4))
        radii = tuple(R1 - np.concatenate([[0.0], np.cumsum(walls)[:-1]]))
        radii = tuple(float(r) for r in radii) + ()
        n_mid = [float(rng.uniform(1.4, 1.6)) for _ in range(len(radii) - 1)]
        tube = cm.Tube(radii=radii, n=tuple([1.0] + n_mid + [float(rng.uniform(1.33, 1.34))]))
    Ro, Ri = tube.r_outer, tube.r_inner
    # camera in the tube frame
    D = rng.uniform(1.5, 12.0) * Ro
    az = rng.uniform(-180, 180)
    hgt = rng.uniform(-0.5, 0.5) * Ro
    aim = np.array([rng.uniform(-0.4, 0.4) * Ri, rng.uniform(-0.3, 0.3) * Ri, rng.uniform(-0.4, 0.4) * Ri])
    width, height = (1280, 800) if rng.random() < 0.5 else (2048, 2048)
    # f so that the inner tube (2 Ri across) spans about the sensor width
    f = float(width / (2 * Ri / D) * rng.uniform(0.5, 1.5))
    dist = np.zeros(5)
    if k % 2 == 1:
        # moderate, so that r * (1 + k1 r^2 + ...) stays monotonic over the sensor and undistortion is unique
        dist = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.05, 0.05), rng.uniform(-1e-3, 1e-3),
                         rng.uniform(-1e-3, 1e-3), rng.uniform(-0.01, 0.01)])
    cam = rigs.ring_camera(az, D, f, width, height, height_mm=hgt, aim=aim, roll_deg=rng.uniform(-180, 180),
                           dist=dist)
    # a random world frame: X_world = Q^T X_t + p, axis direction Q^T e_y
    tilt = math.radians(rng.uniform(0, 30))
    a = rng.normal(size=3)
    a -= (a @ [0, 1, 0]) * np.array([0, 1, 0])
    a /= np.linalg.norm(a)
    Qw = cm.exp_so3(a * tilt) @ cm.exp_so3(np.array([0, rng.uniform(-np.pi, np.pi), 0]))   # tube -> world rotation
    p = rng.uniform(-50, 50, size=3)
    axis_dir = Qw @ np.array([0.0, 1.0, 0.0])
    # the world rotation is rebuilt from the rotation vector the C++ receives, so both sides start from the same
    # rvec (R -> rvec loses digits near pi, which is not what T0 tests)
    R_world = cm.rot_of(cm.rvec_of(cam.R @ Qw.T))
    C_world = Qw @ cam.C + p
    return tube, cam, (R_world, C_world), (p, axis_dir), Qw


def main():
    rng = np.random.default_rng(20261001)
    n_rigs = 24
    res = {"rigs": [], "fail_cases": []}
    cams = {}
    worst = {"bp_o": 0.0, "bp_far": 0.0, "bp_ang": 0.0, "bp_equiv_px": 0.0, "bp_lib_vs_cppundist_px": 0.0, "fw_px": 0.0,
             "fw_ok_frac_min": 1.0, "pinhole_start_ok_frac_min": 1.0, "pinhole_start_max_px": 0.0,
             "ok_mismatch": 0, "beyond_starts": 0, "beyond_same": 0, "beyond_other": 0, "beyond_fail": 0,
             "beyond_false": 0}
    t_start = time.time()
    for k in range(n_rigs):
        tube, cam_t, (Rw, Cw), (p, axis_dir), Qw = random_rig(rng, k)
        cpp = cm.to_cpp(cam_t, tube, tol=1e-13, nmax=300, axis_pt=p, axis_dir=axis_dir, R_world=Rw, C_world=Cw)
        # my camera in the tube frame, rebuilt from the world pose by the conversion under test
        R_t, C_t, Q, p0 = cm.to_tube_frame(Rw, Cw, p, axis_dir)
        mine = cam_t.copy()
        mine.R, mine.C = R_t, C_t
        rw1 = cm.rows([mine], np.zeros(1, int))
        cams[k] = {"R": mine.R.tolist(), "C": mine.C.tolist(), "f": mine.f, "cx": mine.cx, "cy": mine.cy,
                   "width": mine.width, "height": mine.height, "dist": mine.dist.tolist(),
                   "radii": list(tube.radii), "n": list(tube.n)}
        # ---- back-projection over the sensor
        n = 3000
        uv = np.column_stack([rng.uniform(0, mine.width, n), rng.uniform(0, mine.height, n)])
        o_c, d_c, ok_c = cm.cpp_rays(cpp, uv)
        rw = cm.rows([mine], np.zeros(n, int))
        xy_cpp_like = undistort_like_cpp(uv, mine.f, mine.cx, mine.cy, mine.dist)
        o_t, d_t, ok_m, _, _ = cm.trace(rw, xy_cpp_like, tube)
        # back to the world frame: X_w = Q^T X_t + p
        o_w = o_t @ Q + p0
        d_w = d_t @ Q
        both = ok_c & ok_m
        mism = int(np.sum(ok_c != ok_m))
        dob = np.linalg.norm(o_w[both] - o_c[both], axis=1)
        far = 2.0 * tube.r_inner
        dfar = np.linalg.norm((o_w[both] + far * d_w[both]) - (o_c[both] + far * d_c[both]), axis=1)
        ang = np.linalg.norm(np.cross(d_w[both], d_c[both]), axis=1)
        # the same difference in pixels: the C++ ray's mid-chord point (nearest the axis in the cross-section),
        # projected by my forward model from the C++'s own normalised coordinates, lands this far from them (times
        # f).  It compares lines as lines, and the C++'s 1e-5 px undistortion stays out of it.
        ib = np.flatnonzero(both)
        oc_t = (o_c[ib] - p0) @ Q.T
        dc_t = d_c[ib] @ Q.T
        dperp = dc_t[:, [0, 2]]
        tmid = -np.einsum("ij,ij->i", oc_t[:, [0, 2]], dperp) / np.einsum("ij,ij->i", dperp, dperp)
        Pmid = oc_t + tmid[:, None] * dc_t
        prb = cm.project(rw.take(ib), Pmid, xy_cpp_like[ib], tube, tol=1e-16, max_iter=12)
        eq_px = np.linalg.norm(prb.xy - xy_cpp_like[ib], axis=1) * mine.f
        eq_px[~prb.ok] = np.inf
        # the library's exact undistortion vs the C++-like one, in pixels of the undistorted image
        xy_lib = cm.undistort(uv, rw)
        dxy_px = np.max(np.abs(xy_lib - xy_cpp_like)) * mine.f if np.any(mine.dist) else 0.0
        # ---- forward projection (a): the single-image region r < r_c, the plan's criterion
        rc = tube.fold_radius()
        m = 1500

        def points(r_lo, r_hi, count):
            rr = np.sqrt(rng.uniform(r_lo ** 2, r_hi ** 2, count))
            th = rng.uniform(0, 2 * np.pi, count)
            Pt = np.column_stack([rr * np.cos(th), rng.uniform(-0.5, 0.5, count) * tube.r_inner, rr * np.sin(th)])
            return Pt @ Qw.T + p                  # world frame (Qw maps the generating tube frame -> world)

        def on_sensor(q):
            return np.isfinite(q[:, 0]) & (q[:, 0] >= 0) & (q[:, 0] < mine.width) & (q[:, 1] >= 0) & (
                q[:, 1] < mine.height)

        Pw = points(0.0, 0.98 * min(rc, tube.r_inner), m)
        prim = cm.cpp_project(cpp, Pw)
        sel = on_sensor(prim)
        # the same points in the tube frame that `to_tube_frame` built (it differs from the generating one by a
        # turn about the axis)
        P1 = ((Pw - p0) @ Q.T)[sel]
        uv1 = prim[sel]
        rwm = cm.rows([mine], np.zeros(len(P1), int))
        start_uv = uv1 + rng.normal(scale=5.0, size=uv1.shape)
        pr = cm.project(rwm, P1, cm.undistort(start_uv, rwm), tube)
        fw = np.linalg.norm(pr.uv - uv1, axis=1)
        for i in np.flatnonzero(~pr.ok | ~(fw < 1e-6)):
            res["fail_cases"].append({"kind": "forward", "rig": k, "X_tube": P1[i].tolist(),
                                      "start_uv": start_uv[i].tolist(), "cpp_uv": uv1[i].tolist(),
                                      "my_uv": pr.uv[i].tolist(), "ok": bool(pr.ok[i])})
        fw_ok = float(np.mean(pr.ok)) if len(P1) else 1.0
        pr0 = cm.project(rwm, P1, cm.pinhole_start(rwm, P1), tube)
        ph_ok = float(np.mean(pr0.ok)) if len(P1) else 1.0
        ph_err = np.linalg.norm(pr0.uv[pr0.ok] - uv1[pr0.ok], axis=1)
        # ---- forward projection (b): beyond the fold radius (diagnostic).  Every image on the sensor is a start
        # target; the start is offset from it by min(1 px, a quarter of the distance to the nearest other image),
        # so it is nearest that image.  Outcomes: that image, another image, no convergence, or a false solution.
        beyond = {"starts": 0, "same": 0, "other": 0, "fail": 0, "false": 0, "points_2img": 0}
        if rc < 0.999 * tube.r_inner:
            Pw2 = points(rc, 0.999 * tube.r_inner, m)
            imgs = cm.cpp_images(cpp, Pw2)
            Pm2 = (Pw2 - p0) @ Q.T
            for i, im in enumerate(imgs):
                q_on = im[on_sensor(im)] if len(im) else im
                if len(q_on) == 0:
                    continue
                beyond["points_2img"] += int(len(im) >= 2)
                for q in q_on:
                    others = [o for o in im if np.linalg.norm(o - q) > 1e-9]
                    sep = min([np.linalg.norm(o - q) for o in others], default=np.inf)
                    rwq = cm.rows([mine], np.zeros(1, int))
                    off = rng.normal(size=2)
                    off *= min(1.0, 0.25 * sep) / np.linalg.norm(off)
                    sq = (q + off)[None, :]
                    pq = cm.project(rwq, Pm2[i:i + 1], cm.undistort(sq, rwq), tube)
                    beyond["starts"] += 1
                    if not pq.ok[0]:
                        beyond["fail"] += 1
                        kind = "fail"
                    elif np.linalg.norm(pq.uv[0] - q) < 1e-6:
                        beyond["same"] += 1
                        kind = None
                    elif any(np.linalg.norm(pq.uv[0] - o) < 1e-6 for o in others):
                        beyond["other"] += 1
                        kind = "other"
                    else:
                        beyond["false"] += 1
                        kind = "false"
                    if kind:
                        res["fail_cases"].append({"kind": "beyond-" + kind, "rig": k, "X_tube": Pm2[i].tolist(),
                                                  "start_uv": sq[0].tolist(), "cpp_uv": q.tolist(),
                                                  "images": im.tolist(), "my_uv": pq.uv[0].tolist(),
                                                  "ok": bool(pq.ok[0])})
        rec = {
            "rig": k, "media": len(tube.n), "radii": list(tube.radii), "n": list(tube.n),
            "D_over_Router": float(np.linalg.norm(mine.C[[0, 2]]) / tube.r_outer), "f": mine.f,
            "distorted": bool(np.any(mine.dist)), "sensor": [mine.width, mine.height],
            "bp_rays": int(both.sum()), "bp_ok_mismatch": mism,
            "bp_origin_max_mm": float(dob.max()) if dob.size else 0.0,
            "bp_far_side_max_mm": float(dfar.max()) if dfar.size else 0.0,
            "bp_angle_max_rad": float(ang.max()) if ang.size else 0.0,
            "bp_equiv_px_max": float(eq_px.max()) if eq_px.size else 0.0,
            "bp_equiv_px_median": float(np.median(eq_px)) if eq_px.size else 0.0,
            "lib_undistort_vs_cpp_like_px": float(dxy_px),
            "fw_points": int(len(P1)), "fw_ok_frac": fw_ok,
            "fw_max_px": float(np.nanmax(fw[pr.ok])) if np.any(pr.ok) else 0.0,
            "fw_iters": int(pr.iters),
            "pinhole_start_ok_frac": ph_ok, "pinhole_start_max_px": float(ph_err.max()) if ph_err.size else 0.0,
            "beyond_fold": beyond,
        }
        res["rigs"].append(rec)
        worst["bp_o"] = max(worst["bp_o"], rec["bp_origin_max_mm"])
        worst["bp_far"] = max(worst["bp_far"], rec["bp_far_side_max_mm"])
        worst["bp_ang"] = max(worst["bp_ang"], rec["bp_angle_max_rad"])
        worst["bp_equiv_px"] = max(worst["bp_equiv_px"], rec["bp_equiv_px_max"])
        worst["bp_lib_vs_cppundist_px"] = max(worst["bp_lib_vs_cppundist_px"], rec["lib_undistort_vs_cpp_like_px"])
        worst["fw_px"] = max(worst["fw_px"], rec["fw_max_px"])
        worst["fw_ok_frac_min"] = min(worst["fw_ok_frac_min"], fw_ok)
        worst["pinhole_start_ok_frac_min"] = min(worst["pinhole_start_ok_frac_min"], ph_ok)
        worst["pinhole_start_max_px"] = max(worst["pinhole_start_max_px"], rec["pinhole_start_max_px"])
        for kk in ("starts", "same", "other", "fail", "false"):
            worst["beyond_" + kk] += beyond[kk]
        worst["ok_mismatch"] += mism
        log(f"rig {k:2d}: media {len(tube.n)}, D/Ro {rec['D_over_Router']:5.2f}, f {mine.f:9.1f}, "
            f"dist {'y' if rec['distorted'] else 'n'} | bp rays {rec['bp_rays']:4d} mism {mism} "
            f"o {rec['bp_origin_max_mm']:.1e} far {rec['bp_far_side_max_mm']:.1e} mm ang "
            f"{rec['bp_angle_max_rad']:.1e} eq {rec['bp_equiv_px_max']:.1e} px | lib-undist {dxy_px:.1e} px | fw {len(P1):4d} pts ok "
            f"{fw_ok:.3f} max {rec['fw_max_px']:.1e} px it {pr.iters} | pinhole start ok {ph_ok:.3f} | "
            f"beyond r_c: {beyond['starts']} starts, same {beyond['same']} other {beyond['other']} "
            f"fail {beyond['fail']} false {beyond['false']}")
    res["worst"] = worst
    res["fail_cameras"] = {str(c["rig"]): cams[c["rig"]] for c in res["fail_cases"]}
    crit = {
        "rays agree to 1e-8 px (the C++ ray reprojected by the fast model)": worst["bp_equiv_px"] < 1e-8,
        "same ok flags as the C++": worst["ok_mismatch"] == 0,
        "single-image region: forward within 1e-6 px, every start 5 px off converged":
            worst["fw_px"] < 1e-6 and worst["fw_ok_frac_min"] == 1.0,
        "single-image region: every crude pinhole start converged to the C++ image":
            worst["pinhole_start_ok_frac_min"] == 1.0 and worst["pinhole_start_max_px"] < 1e-6,
        "beyond r_c: no false solution (a converged pixel is always one of the C++ images)": worst["beyond_false"] == 0,
    }
    res["criteria"] = crit
    res["seconds"] = time.time() - t_start
    log("")
    log("worst over rigs:", json.dumps(worst, indent=1))
    for kk, v in crit.items():
        log(f"  {'PASS' if v else 'FAIL'}  {kk}")
    log(f"time {res['seconds']:.1f} s")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
