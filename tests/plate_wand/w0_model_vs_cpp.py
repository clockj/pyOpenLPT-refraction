"""W0 (model against the C++): plate_model.project / rays against pyOpenLPT's C++ PINPLATE camera.

2,000 random camera / window / point triples:
  - windows with one plate (3-30 mm, n 1.40-1.60) or two (+ 2-20 mm, n 1.45-1.75), the object medium n 1.30-1.50,
    the camera in air; random normals and offsets;
  - cameras 50-2,000 mm from the window, chief ray at 0-60 deg incidence, 1280 x 800 to 4096 x 3000 sensors with a
    half-diagonal field of view of 3-40 deg (f about 900-48,000 px), no distortion (and a separate group with
    distortion);
  - points: a random pixel on the sensor whose ray enters the window at most 80 deg from its normal, back-projected
    by the C++ and taken 0.01-500 mm beyond the object-side surface; plus points on or next to the camera's normal
    line (rho = 0, 1e-12, 1e-9, 1e-6 mm).
Criteria: projections within 1e-6 px of the C++'s (and of the pixel the point came from); rays (origin on the
object-side surface, unit direction) within 1e-10 (mm, and per unit length).  With distortion the C++ stops
undistorting at a tolerance (Camera.cpp, UNDISTORT_EPS), so that group is recorded with its own numbers.
Also recorded: what the C++ returns for points on the camera side of the object-side surface (the model: not ok).
Output: w0_model_vs_cpp_out.txt / .json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prigs  # noqa: E402

pm = prigs.pm
cm = prigs.cm
log = prigs.Tee(HERE / "w0_model_vs_cpp_out.txt")
rng = np.random.default_rng(20261002)


def rand_unit():
    a = rng.normal(size=3)
    return a / np.linalg.norm(a)


def random_case(distorted=False):
    n = rand_unit()
    d = rng.uniform(-200.0, 200.0)
    if rng.random() < 0.5:
        t, idx = (rng.uniform(3, 30),), (1.0, rng.uniform(1.40, 1.60), rng.uniform(1.30, 1.50))
    else:
        t = (rng.uniform(3, 30), rng.uniform(2, 20))
        idx = (1.0, rng.uniform(1.40, 1.60), rng.uniform(1.45, 1.75), rng.uniform(1.30, 1.50))
    win = pm.Window(n, d, t, idx)
    h1 = math.exp(rng.uniform(math.log(50.0), math.log(2000.0)))
    theta = rng.uniform(0.0, 60.0)
    t1, t2 = pm.tangent_basis(n)
    ph = rng.uniform(0, 2 * math.pi)
    e = math.cos(ph) * t1[0] + math.sin(ph) * t2[0]
    u = math.cos(math.radians(theta)) * n + math.sin(math.radians(theta)) * e
    foot = d * n + rng.uniform(-300, 300) * t1[0] + rng.uniform(-300, 300) * t2[0]   # a point on the surface
    C = foot - (h1 / math.cos(math.radians(theta))) * u
    R = prigs.look_dir(u, up=rand_unit(), roll_deg=rng.uniform(-180, 180))
    W, H = [(1280, 800), (2048, 2048), (4096, 3000)][int(rng.integers(0, 3))]
    f = 0.5 * math.hypot(W, H) / math.tan(math.radians(rng.uniform(3.0, 40.0)))
    dist = np.zeros(5)
    if distorted:
        dist = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.1, 0.1), rng.uniform(-1e-3, 1e-3),
                         rng.uniform(-1e-3, 1e-3), 0.0])
    cam = pm.Camera(R, C, f, (W - 1) / 2.0 + rng.uniform(-20, 20), (H - 1) / 2.0 + rng.uniform(-20, 20), W, H, dist)
    return cam, win


def one(rows, k):
    return pm.rows([rows[0]], np.zeros(k, int)), pm.wrows([rows[1]], np.zeros(k, int))


res = {"groups": {}}
groups = {"no distortion": False, "with distortion": True}
for gname, distorted in groups.items():
    n_cases = 2000 if not distorted else 400
    worst = {"proj_vs_cpp_px": 0.0, "proj_vs_pixel_px": 0.0, "ray_dir": 0.0, "ray_origin_mm": 0.0,
             "ray_origin_rel": 0.0, "cpp_closure_px": 0.0}
    n_pts = 0
    n_fail_model = n_fail_cpp = 0
    tau_max = 0.0
    for case in range(n_cases):
        cam, win = random_case(distorted)
        cpp = pm.to_cpp(cam, win)
        rw, ww = one((cam, win), 1)
        for _try in range(100):
            uv0 = np.column_stack([rng.uniform(0, cam.width - 1, 1), rng.uniform(0, cam.height - 1, 1)])
            xy0 = pm.undistort(uv0, rw)
            q0 = cam.R.T @ np.array([xy0[0, 0], xy0[0, 1], 1.0])
            if float(q0 @ win.n) > math.cos(math.radians(80.0)) * np.linalg.norm(q0):
                break
        else:
            continue
        o, dvec, okr = cm.cpp_rays(cpp, uv0)
        if not okr[0]:
            n_fail_cpp += 1
            continue
        depth = math.exp(rng.uniform(math.log(0.01), math.log(500.0)))
        X = o + (depth / float(dvec[0] @ win.n)) * dvec
        pr = pm.project(rw, ww, X)
        uc = cm.cpp_project(cpp, X)
        if not pr.ok[0]:
            n_fail_model += 1
            log(f"  model failed: case {case}, tau {pr.tau[0]}, h1 {pr.h1[0]}, hW {pr.hW[0]}")
            continue
        n_pts += 1
        tau_max = max(tau_max, float(pr.tau[0]))
        worst["proj_vs_cpp_px"] = max(worst["proj_vs_cpp_px"], float(np.linalg.norm(pr.uv[0] - uc[0])))
        worst["proj_vs_pixel_px"] = max(worst["proj_vs_pixel_px"], float(np.linalg.norm(pr.uv[0] - uv0[0])))
        worst["cpp_closure_px"] = max(worst["cpp_closure_px"], float(np.linalg.norm(uc[0] - uv0[0])))
        om, dm, okm = pm.rays(rw, ww, uv0)
        scale = float(np.linalg.norm(o[0] - cam.C))
        worst["ray_dir"] = max(worst["ray_dir"], float(np.linalg.norm(dm[0] - dvec[0])))
        worst["ray_origin_mm"] = max(worst["ray_origin_mm"], float(np.linalg.norm(om[0] - o[0])))
        worst["ray_origin_rel"] = max(worst["ray_origin_rel"], float(np.linalg.norm(om[0] - o[0])) / scale)
    res["groups"][gname] = {"cases": n_cases, "points": n_pts, "model_failed": n_fail_model,
                            "cpp_ray_failed": n_fail_cpp, "tau_max": tau_max, **worst}
    log(f"{gname}: {n_pts} points of {n_cases} cases (model failed {n_fail_model}, C++ ray failed {n_fail_cpp}); "
        f"largest air angle {math.degrees(math.atan(tau_max)):.1f} deg")
    log(f"  projection vs C++ {worst['proj_vs_cpp_px']:.2e} px, vs the pixel {worst['proj_vs_pixel_px']:.2e} px "
        f"(C++ closure {worst['cpp_closure_px']:.2e} px)")
    log(f"  ray direction {worst['ray_dir']:.2e}, ray origin {worst['ray_origin_mm']:.2e} mm "
        f"({worst['ray_origin_rel']:.2e} of the camera's distance)")

# points on or next to the camera's normal line
worst_axis = 0.0
rows_axis = []
for case in range(200):
    cam, win = random_case(False)
    cpp = pm.to_cpp(cam, win)
    h1 = win.distance(cam.C)
    t1, _t2 = pm.tangent_basis(win.n)
    for rho in (0.0, 1e-12, 1e-9, 1e-6):
        s = h1 + win.T + rng.uniform(1.0, 300.0)
        X = (cam.C + s * win.n + rho * t1[0])[None]
        rw, ww = one((cam, win), 1)
        pr = pm.project(rw, ww, X)
        uc = cm.cpp_project(cpp, X)
        if pr.ok[0] and np.all(np.isfinite(uc)):
            dlt = float(np.linalg.norm(pr.uv[0] - uc[0]))
            worst_axis = max(worst_axis, dlt)
            rows_axis.append(dlt)
res["normal_line"] = {"points": len(rows_axis), "worst_px": worst_axis}
log(f"on / next to the camera's normal line (rho 0 .. 1e-6 mm): {len(rows_axis)} points, worst {worst_axis:.2e} px")

# points on the camera side of the object-side surface
quirk = {"inside_plate": [0, 0, 0], "camera_side": [0, 0, 0]}    # [count, C++ returned a pixel, model ok]
mirror_dev = []
for case in range(300):
    cam, win = random_case(False)
    cpp = pm.to_cpp(cam, win)
    uv0 = np.array([[cam.cx, cam.cy]])
    o, dvec, okr = cm.cpp_rays(cpp, uv0)
    if not okr[0]:
        continue
    for name, depth in (("inside_plate", -0.5 * win.T), ("camera_side", -win.T - 0.5 * win.distance(cam.C))):
        # a point on the line through o along the object-medium ray, `depth` along the normal from the far surface
        X = o + (depth / float(dvec[0] @ win.n)) * dvec
        rw, ww = one((cam, win), 1)
        pr = pm.project(rw, ww, X)
        uc = cm.cpp_project(cpp, X)
        quirk[name][0] += 1
        quirk[name][1] += int(np.all(np.isfinite(uc)))
        quirk[name][2] += int(pr.ok[0])
        if np.all(np.isfinite(uc)):
            # the mirror image of X in the object-side surface, projected by the model
            Xm = X - 2.0 * (float(X[0] @ win.n) - win.far_d) * win.n
            pmr = pm.project(rw, ww, Xm)
            if pmr.ok[0]:
                mirror_dev.append(float(np.linalg.norm(pmr.uv[0] - uc[0])))
res["camera_side_points"] = {k: {"count": v[0], "cpp_returned_pixel": v[1], "model_ok": v[2]}
                             for k, v in quirk.items()}
res["camera_side_points"]["cpp_equals_mirror_image_max_px"] = max(mirror_dev) if mirror_dev else None
for k, v in quirk.items():
    log(f"points {k.replace('_', ' ')} (not in the object medium): {v[0]}; the C++ returned a pixel for {v[1]}, "
        f"the model for {v[2]}")
if mirror_dev:
    log(f"  the C++ pixel is the image of the point mirrored in the object-side surface (worst {max(mirror_dev):.1e} px "
        f"over {len(mirror_dev)}): the C++ uses the unsigned distance to that surface (Camera.cpp, Snell1D step 1)")

g = res["groups"]["no distortion"]
ok = (g["proj_vs_cpp_px"] <= 1e-6 and g["proj_vs_pixel_px"] <= 1e-6 and g["ray_dir"] <= 1e-10
      and g["ray_origin_rel"] <= 1e-10 and worst_axis <= 1e-6)
res["pass"] = bool(ok)
log("W0 model vs C++: " + ("PASS" if ok else "FAIL"))
(HERE / "w0_model_vs_cpp.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
sys.exit(0 if ok else 1)
