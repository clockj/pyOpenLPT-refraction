"""The 2-, 3- and 4-camera rigs of Part S (plan: 2026-10-08-plan.md, sections 3 and 7).

    python make_rigs.py

Writes data/rig/cam1..cam4.txt (git-ignored, regenerable):
  - cam1, cam2: T0's tracking camFiles (cases/cyl_20261007/data/cams_T0), byte copies;
  - cam3: cam2's lens (K, image size) at cam2's distance from the aim point, moved to azimuth 30 deg (between cam1 at 0
    and cam2 at 60) and 25 deg above the horizontal plane; cam4: cam1's lens and distance, azimuth 30 deg, 25 deg below;
  - the aim point T: the centroid of the points of the central region (r < 30 mm) that both real cameras see on their
    sensors.  A new camera's optical axis is set along T's ray in air at its pinhole (the refracted path from T to a
    fixed pinhole does not depend on the camera's orientation), so T images at the principal point; its x axis is
    horizontal, as cam1's.
Checks (make_rigs_out.txt, make_rigs.json):
  - T's pixel in each new camera (the principal point, to rounding);
  - per camera, the share of the central region's points it sees, and of T1's view volume's points seen by 2, 3 and
    4 cameras of the 4-camera rig;
  - the certificate: the share of each camera's points in the view volume that provablySingleImage() certifies (the
    hinted projection), and imageStatus() of the rest;
  - at T, the image direction of each camera's line of sight in every other camera (the epipolar lines' angles) and
    of the tube's axis (the stretch is across it).
"""
from __future__ import annotations

import math
import os
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

import study
import cstb
import shape_model as sm

NEW = [  # name, source camera (lens and distance), azimuth [deg], elevation [deg]
    ("cam3", 2, 30.0, 25.0),
    ("cam4", 1, 30.0, -25.0),
]


def lookat_rotation(z_dir):
    """World-to-camera rotation (rows x, y, z) with the optical axis z_dir and x horizontal (y the world's up)."""
    z = np.asarray(z_dir, float) / np.linalg.norm(z_dir)
    x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def write_camfile(src, dst, R, t):
    """src's camFile with the pose (R, t) and its derived lines replaced; calibration errors set to 0."""
    with open(src, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    fmt = lambda v: ",".join("%.16e" % x for x in v)

    def put(header, rows):
        i = next(k for k, ln in enumerate(lines) if ln.startswith(header)) + 1
        for k, r in enumerate(rows):
            lines[i + k] = r

    put("# Camera Calibration Error", ["0,0"])
    put("# Pose Calibration Error", ["0,0"])
    put("# Rotation Vector", [fmt(Rotation.from_matrix(R).as_rotvec())])
    put("# Rotation Matrix", [fmt(R[k]) for k in range(3)])
    put("# Inverse of Rotation Matrix", [fmt(R.T[k]) for k in range(3)])
    put("# Translation Vector", [fmt(t)])
    put("# Inverse of Translation Vector", [fmt(-R.T @ t)])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def seen(lpt, cam, X):
    """On the sensor (bool array) and the pixels (n, 2) of points X."""
    res = cam.projectBatchStatus([lpt.Pt3D(*map(float, x)) for x in X])
    W, H = cam.getNCol(), cam.getNRow()
    uv = np.array([[r[1][0], r[1][1]] if r[0] else [np.nan, np.nan] for r in res])
    on = np.isfinite(uv[:, 0]) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return on, uv


def main():
    lpt, build = study.lib()
    log = study.Log("make_rigs")
    out = {"build": build}
    log(f"The study's rigs; build {build['name']}")
    os.makedirs(study.RIG, exist_ok=True)
    for c, f in enumerate(study.CAMS_T0):
        shutil.copy2(f, os.path.join(study.RIG, f"cam{c + 1}.txt"))
    real = [lpt.loadCameraModel(f) for f in study.rig_files(2)]

    rng = np.random.default_rng(11)
    n = 200000
    X = np.column_stack([rng.uniform(-study.R_IN, study.R_IN, n), rng.uniform(-45, 45, n),
                         rng.uniform(-study.R_IN, study.R_IN, n)])
    X = X[np.hypot(X[:, 0], X[:, 2]) < study.R_IN]
    on_real = [seen(lpt, cam, X)[0] for cam in real]
    both = on_real[0] & on_real[1]
    T = X[both].mean(axis=0)
    out["aim_point_mm"] = T
    out["central_points_sampled"] = int(len(X))
    log(f"  aim point T (centroid of the {int(both.sum())} of {len(X)} central points both real cameras see): "
        f"({T[0]:.2f}, {T[1]:.2f}, {T[2]:.2f}) mm")

    for name, src_c, az, el in NEW:
        src = study.CAMS_T0[src_c - 1]
        R_s, t_s, K = sm.camera_geometry(src)
        P_s = -R_s.T @ t_s
        D = float(np.linalg.norm(P_s - T))
        a, e = math.radians(az), math.radians(el)
        P = T + D * np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])
        dst = os.path.join(study.RIG, f"{name}.txt")
        R0 = lookat_rotation(T - P)
        write_camfile(src, dst, R0, -R0 @ P)
        cam = lpt.loadCameraModel(dst)
        p = cam.project(lpt.Pt3D(*map(float, T)))
        first = float(math.hypot(p[0] - K[0, 2], p[1] - K[1, 2]))   # aimed at T in air: off by the refraction
        d = R0.T @ np.linalg.solve(K, [p[0], p[1], 1.0])          # T's ray in air at the pinhole, world frame
        R = lookat_rotation(d)
        write_camfile(src, dst, R, -R @ P)
        cam = lpt.loadCameraModel(dst)
        p = cam.project(lpt.Pt3D(*map(float, T)))
        rec = {"source": f"cam{src_c}", "azimuth_deg": az, "elevation_deg": el, "distance_mm": D,
               "pinhole_mm": P, "f_px": float(K[0, 0]), "T_pixel": [p[0], p[1]],
               "T_pixel_error_px": float(math.hypot(p[0] - K[0, 2], p[1] - K[1, 2])),
               "first_aim_error_px": first}
        out[name] = rec
        log(f"  {name}: cam{src_c}'s lens (f {K[0, 0]:.0f} px) at {D:.1f} mm from T, azimuth {az:g} deg, elevation "
            f"{el:+g} deg; pinhole ({P[0]:.1f}, {P[1]:.1f}, {P[2]:.1f}) mm; T images at ({p[0]:.6f}, {p[1]:.6f}) "
            f"(principal point ({K[0, 2]}, {K[1, 2]}): {rec['T_pixel_error_px']:.1e} px off; aimed in air "
            f"{first:.0f} px off)")

    cams = [lpt.loadCameraModel(f) for f in study.rig_files(4)]
    on, uv = zip(*[seen(lpt, cam, X) for cam in cams])
    on = np.array(on)
    vol = study.view_volume()
    inv = study.in_volume(X, vol)
    out["central"] = {f"cam{c + 1}": float(on[c].mean()) for c in range(4)}
    log("  share of the central region's points (r < 30 mm, |y| < 45 mm) each camera sees: "
        + ", ".join(f"cam{c + 1} {100 * on[c].mean():.1f} %" for c in range(4)))
    k = on[:, inv].sum(axis=0)
    out["view_volume"] = {"points": int(inv.sum()),
                          "rig2_both": float(np.mean(on[0, inv] & on[1, inv])),
                          "rig3_ge2": float(np.mean(on[:3, inv].sum(axis=0) >= 2)),
                          "rig3_all": float(np.mean(on[:3, inv].sum(axis=0) == 3)),
                          "rig4_ge2": float(np.mean(k >= 2)), "rig4_ge3": float(np.mean(k >= 3)),
                          "rig4_all": float(np.mean(k == 4))}
    v = out["view_volume"]
    log(f"  T1's view volume (x {np.round(vol['x'], 1)}, y {np.round(vol['y'], 1)}, z {np.round(vol['z'], 1)} mm), "
        f"{v['points']} of the sampled points inside the water's central region: seen by both real cameras "
        f"{100 * v['rig2_both']:.1f} %; 3-camera rig: >= 2 {100 * v['rig3_ge2']:.1f} %, all 3 "
        f"{100 * v['rig3_all']:.1f} %; 4-camera rig: >= 2 {100 * v['rig4_ge2']:.1f} %, >= 3 {100 * v['rig4_ge3']:.1f} %, "
        f"all 4 {100 * v['rig4_all']:.1f} %")

    # the certificate and the statuses
    cert = {}
    sub = np.flatnonzero(inv)[:6000]
    for c, cam in enumerate(cams):
        idx = sub[on[c, sub]]
        ok = np.array([cam.provablySingleImage(lpt.Pt3D(*map(float, X[i]))) for i in idx])
        rest = idx[~ok]
        st = [cam.imageStatus(lpt.Pt3D(*map(float, X[i])), 0.0, 4.0)[0] for i in rest[:500]]
        cert[f"cam{c + 1}"] = {"points": int(len(idx)), "certified": float(ok.mean()) if len(ok) else None,
                               "rest_status": {s: st.count(s) for s in sorted(set(st))}}
        log(f"  cam{c + 1}: {len(idx)} points of the view volume on its sensor; certified single image "
            f"{100 * ok.mean():.2f} %; the rest's imageStatus (up to 500): {cert[f'cam{c + 1}']['rest_status']}")
    out["certificate"] = cert

    # the epipolar lines' and the axis' image directions at T
    PT = lpt.Pt3D(*map(float, T))
    pix = [cam.project(PT) for cam in cams]
    ang = {}
    for i, ci in enumerate(cams):
        los = ci.lineOfSight(pix[i])
        u = np.array([los.unit_vector[k] for k in range(3)])
        for j, cj in enumerate(cams):
            if i == j:
                continue
            a = cj.project(lpt.Pt3D(*map(float, T - 2.0 * u)))
            b = cj.project(lpt.Pt3D(*map(float, T + 2.0 * u)))
            th = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0
            ang[f"cam{i + 1}_in_cam{j + 1}"] = th
    axis = {f"cam{c + 1}": (math.degrees(sm.axis_angle(lpt, cam, T, (0.0, 1.0, 0.0))) + 90.0) % 180.0
            for c, cam in enumerate(cams)}
    out["epipolar_angle_deg"] = ang
    out["axis_image_angle_deg"] = axis
    for j in range(4):
        log(f"  in cam{j + 1} (image angles from the u axis, mod 180; the tube's axis at {axis[f'cam{j + 1}']:.1f} deg): "
            + ", ".join(f"cam{i + 1}'s lines {ang[f'cam{i + 1}_in_cam{j + 1}']:.1f} deg" for i in range(4) if i != j))
    log.save(out)


if __name__ == "__main__":
    main()
