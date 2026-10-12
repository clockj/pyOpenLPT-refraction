"""Pre-plan sizing for the flat-window (PINPLATE) wand calibration plan: how far a camera behind a flat window is
from a pinhole camera, and how visible the window's distance and tilt are, for typical lenses and incidence angles.

Setting (all from pyOpenLPT's C++ PINPLATE camera, the source of truth):
  - window: camera-side surface Z = 0, normal +Z (into the water), acrylic 12.7 mm (n 1.49), air 1.0, water 1.333;
  - camera: its chief ray meets the window at the origin at incidence alpha (in the X-Z plane), 400 mm from the
    camera along that ray; f = 3000 or 12000 px on a 1280 x 800 sensor (half-angles about 12 and 3 deg);
  - points: a 13 x 9 grid of pixels (5 % margin) back-projected into the water and placed 20, 45, 70, 95 and 120 mm
    below the water-side surface (585 points), so the grid pixels are their exact images.

Fits, with the points known (as with a dot target; a wand gives less):
  (a) the best pinhole camera (R, C, f; cx, cy at the centre), then + k1, then + k1 + cx, cy;
  (b) the refractive camera refitted (R, C, f) with the window moved along its normal towards the camera by 10 % and
      25 % of the camera's distance to it, and 15 mm away from it (the nearest points are 20 mm below the water-side
      surface, so all stay in the water): how well a wrong window distance can be absorbed;
  (c) the same with the window's normal tilted by 2 and 5 deg, in and out of the plane of incidence (pivot: the
      chief ray's crossing).
An RMS far below the detection noise (about 0.1 px) means the data cannot see that change.
Output: p_pre_sizing_out.txt, p_pre_sizing.json.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO / "build-py" / os.environ.get("CONFIG", "Release")), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import cv2  # noqa: E402
import pyopenlpt as lpt  # noqa: E402

W, H = 1280, 800
CX, CY = (W - 1) / 2.0, (H - 1) / 2.0
T_WIN = 12.7
N_AIR, N_WIN, N_WAT = 1.0, 1.49, 1.333
D_CHIEF = 400.0
DEPTHS = (20.0, 45.0, 70.0, 95.0, 120.0)
NORMAL = np.array([0.0, 0.0, 1.0])


def rvec_from_R(R):
    """Rotation vector of R, accurate to round-off at every angle (half-turns included)."""
    v = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s_ = float(np.linalg.norm(v))
    c = 0.5 * (float(np.trace(R)) - 1.0)
    th = math.atan2(s_, c)
    if s_ < 1e-12 and c > 0:
        return v
    if th < 3.0:
        return v * (th / s_)
    B = 0.5 * (R + R.T) - c * np.eye(3)
    k = int(np.argmax(np.diag(B)))
    axis = B[:, k] / np.linalg.norm(B[:, k])
    if axis @ v < 0:
        axis = -axis
    return axis * th


def look_at(C, target, up=(0.0, 1.0, 0.0)):
    z = np.asarray(target, float) - np.asarray(C, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    x /= np.linalg.norm(x)
    return np.vstack([x, np.cross(z, x), z])


def build(R, C, f, plane_z=0.0, normal=NORMAL, k1=0.0, pivot=None):
    """A C++ PINPLATE camera; the window's camera-side surface passes through `pivot` + plane_z * normal."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    p_near = (np.zeros(3) if pivot is None else np.asarray(pivot, float)) + plane_z * n
    cam = lpt.Camera()
    cam.setPinplateImageSize(H, W)
    cam.setPinplateIntrinsics(float(f), float(f), CX, CY, [float(k1), 0.0, 0.0, 0.0, 0.0])
    # not cv2.Rodrigues: within ~1e-4 rad of a half-turn (every look-at camera here is one) OpenCV's matrix -> vector
    # conversion loses up to 2e-5 rad, which made finite differences of a 1e-7 rad turn vanish or come out 4x too
    # large; and not cyl_model.log_so3, whose half-turn branch does not normalise the axis (1.3 rad off;
    # p_logmap_check.py)
    rvec = rvec_from_R(np.asarray(R, float))
    t = -np.asarray(R, float) @ np.asarray(C, float)
    cam.setPinplateExtrinsics(lpt.Pt3D(*map(float, rvec)), lpt.Pt3D(*map(float, t)))
    p_far = p_near + T_WIN * n
    cam.setPinplateRefraction(lpt.Pt3D(*map(float, p_far)), lpt.Pt3D(*map(float, n)), [N_WAT, N_WIN, N_AIR],
                              [T_WIN], 1e-12, 1000, 0.1)
    cam.commitPinplateUpdate(True, 255.0)
    return cam


def project(cam, X):
    res = cam.projectBatchStatus([lpt.Pt3D(*map(float, p)) for p in X], False)
    out = np.full((len(X), 2), np.nan)
    for k, (good, pt, _m) in enumerate(res):
        if good:
            out[k] = (pt[0], pt[1])
    return out


def points_for(cam):
    us = np.linspace(0.05 * W, 0.95 * W, 13)
    vs = np.linspace(0.05 * H, 0.95 * H, 9)
    uv = np.array([(u, v) for v in vs for u in us])
    res = cam.lineOfSightBatchStatus([lpt.Pt2D(float(a), float(b)) for a, b in uv])
    X, U = [], []
    for (good, line, _m), q in zip(res, uv):
        if not good:
            continue
        o = np.array([line.pt[0], line.pt[1], line.pt[2]])
        d = np.array([line.unit_vector[0], line.unit_vector[1], line.unit_vector[2]])
        for z in DEPTHS:
            s = (T_WIN + z - o[2]) / d[2]
            X.append(o + s * d)
            U.append(q)
    return np.array(X), np.array(U)


# ------------------------------------------------------------------ pinhole fits
def pin_project(R, C, f, X, k1=0.0, cx=CX, cy=CY):
    Xc = (X - C) @ R.T
    x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
    r2 = x * x + y * y
    s = 1.0 + k1 * r2
    return np.column_stack([f * x * s + cx, f * y * s + cy])


def dlt_start(X, U):
    A = []
    for (x, y, z), (u, v) in zip(X, U):
        A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])
    P = np.linalg.svd(np.asarray(A))[2][-1].reshape(3, 4)
    M = P[:, :3]
    if np.linalg.det(M) < 0:
        P, M = -P, -M
    C = -np.linalg.solve(M, P[:, 3])
    K, R = _rq(M)
    D = np.diag(np.sign(np.diag(K)))
    K, R = K @ D, D @ R
    if np.linalg.det(R) < 0:
        R = -R
    K = K / K[2, 2]
    return R, C, 0.5 * (abs(K[0, 0]) + abs(K[1, 1]))


def _rq(M):
    Q, Rr = np.linalg.qr(np.flipud(M).T)
    Rr = np.flipud(Rr.T)
    Q = Q.T
    return Rr[:, ::-1], Q[::-1, :]


def fit_pinhole(X, U, use_k1=False, free_pp=False):
    R0, C0, f0 = dlt_start(X, U)

    def unpack(p):
        R = cv2.Rodrigues(p[0:3])[0] @ R0
        k = 7
        k1 = p[k] if use_k1 else 0.0
        k += int(use_k1)
        cx, cy = (p[k], p[k + 1]) if free_pp else (CX, CY)
        return R, C0 + p[3:6], f0 * math.exp(p[6]), k1, cx, cy

    def fun(p):
        R, C, f, k1, cx, cy = unpack(p)
        return (pin_project(R, C, f, X, k1, cx, cy) - U).ravel()

    p0 = np.zeros(7 + int(use_k1) + 2 * int(free_pp))
    if free_pp:
        p0[-2:] = (CX, CY)
    res = least_squares(fun, p0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
    R, C, f, k1, cx, cy = unpack(res.x)
    r = res.fun.reshape(-1, 2)
    e = np.linalg.norm(r, axis=1)
    return {"rms_px": float(np.sqrt(np.mean(e ** 2))), "max_px": float(e.max()), "f": f, "C": C, "R": R,
            "k1": float(k1), "cx": float(cx), "cy": float(cy)}


# ------------------------------------------------------------------ refractive refits
def refit_refractive(X, U, R0, C0, f0, plane_z=0.0, normal=NORMAL, pivot=None, steps=4):
    """Refit R, C, f with the window changed; the change is applied in `steps` stages (continuation), each refit
    from the last, so a large change never starts the fit far from its answer."""
    n0 = np.asarray(NORMAL, float)
    n1 = np.asarray(normal, float) / np.linalg.norm(normal)
    x = np.zeros(7)
    res = None
    for k in range(1, steps + 1):
        w = k / steps
        nk = (1 - w) * n0 + w * n1
        nk /= np.linalg.norm(nk)
        zk = w * plane_z

        def fun(p, zk=zk, nk=nk):
            R = cv2.Rodrigues(p[0:3])[0] @ R0
            cam = build(R, C0 + p[3:6], f0 * math.exp(p[6]), zk, nk, pivot=pivot)
            uv = project(cam, X)
            r = uv - U
            r[~np.isfinite(r)] = 1e3
            return r.ravel()

        res = least_squares(fun, x, method="trf", x_scale="jac", xtol=1e-15, ftol=1e-15, gtol=1e-15,
                            max_nfev=2000, diff_step=1e-6)
        x = res.x
    r = res.fun.reshape(-1, 2)
    e = np.linalg.norm(r, axis=1)
    return {"rms_px": float(np.sqrt(np.mean(e ** 2))), "max_px": float(e.max()),
            "f_rel": math.exp(x[6]) - 1.0, "dC_mm": float(np.linalg.norm(x[3:6])),
            "drot_deg": float(np.degrees(np.linalg.norm(x[0:3])))}


def rot_about(axis, deg):
    return cv2.Rodrigues(np.asarray(axis, float) * math.radians(deg))[0]


def main():
    out = {}
    lines = [__doc__.strip().splitlines()[0], ""]
    for f in (3000.0, 12000.0):
        for alpha in (0.0, 15.0, 30.0, 45.0):
            a = math.radians(alpha)
            dchief = np.array([math.sin(a), 0.0, math.cos(a)])
            C = -D_CHIEF * dchief
            R = look_at(C, np.zeros(3))
            cam = build(R, C, f)
            pp = cam._pinplate_param
            R_cpp = np.array([[pp.r_mtx[i, j] for j in range(3)] for i in range(3)])
            assert np.max(np.abs(R_cpp - R)) < 1e-14, "the C++ camera is not the one intended"
            X, U = points_for(cam)
            back = project(cam, X)
            closure = float(np.nanmax(np.linalg.norm(back - U, axis=1)))
            h1 = -C[2]
            key = f"f{int(f)}_a{int(alpha)}"
            row = {"n_points": len(X), "closure_px": closure, "h1_mm": h1}
            p0 = fit_pinhole(X, U)
            p1 = fit_pinhole(X, U, use_k1=True)
            p2 = fit_pinhole(X, U, use_k1=True, free_pp=True)
            row["pinhole"] = {k: p0[k] for k in ("rms_px", "max_px")}
            row["pinhole_k1"] = {k: p1[k] for k in ("rms_px", "max_px", "k1")}
            row["pinhole_k1_pp"] = {k: p2[k] for k in ("rms_px", "max_px", "k1", "cx", "cy")}
            row["apparent_f_ratio"] = p0["f"] / f
            row["apparent_dist_ratio"] = float(np.linalg.norm(p0["C"]) / D_CHIEF)
            # (b) the window moved along its normal: towards the camera by 10 % and 25 % of its distance, away by 15 mm
            shifts = {"-25%": -0.25 * h1, "-10%": -0.10 * h1, "+15mm": 15.0}
            for name, dz in shifts.items():
                row[f"window_shift_{name}"] = refit_refractive(X, U, R, C, f, plane_z=dz)
            # (c) the normal tilted, pivot at the chief ray's crossing (the origin)
            for deg in (2.0, 5.0):
                for name, ax in (("in_plane", (0.0, 1.0, 0.0)), ("out_of_plane", (1.0, 0.0, 0.0))):
                    nrm = rot_about(ax, deg) @ NORMAL
                    row[f"tilt_{name}_{deg:g}deg"] = refit_refractive(X, U, R, C, f, 0.0, nrm)
            out[key] = row
            lines.append(f"f = {f:.0f} px, incidence {alpha:.0f} deg (camera {h1:.0f} mm from the window along the "
                         f"normal; {len(X)} points; C++ closure {closure:.1e} px)")
            lines.append(f"  pinhole:            rms {p0['rms_px']:.4f} px, max {p0['max_px']:.4f} px; apparent f "
                         f"x{row['apparent_f_ratio']:.4f}, apparent distance x{row['apparent_dist_ratio']:.4f}")
            lines.append(f"  pinhole + k1:       rms {p1['rms_px']:.4f} px, max {p1['max_px']:.4f} px (k1 "
                         f"{p1['k1']:+.3e})")
            lines.append(f"  pinhole + k1 + pp:  rms {p2['rms_px']:.4f} px, max {p2['max_px']:.4f} px (cx "
                         f"{p2['cx']:.1f}, cy {p2['cy']:.1f})")
            for name, dz in shifts.items():
                r = row[f"window_shift_{name}"]
                lines.append(f"  window moved {name} ({dz:+.0f} mm), camera refitted: rms "
                             f"{r['rms_px']:.4f} px, max {r['max_px']:.4f} px (f {r['f_rel'] * 100:+.2f} %, centre "
                             f"{r['dC_mm']:.1f} mm, rotation {r['drot_deg']:.3f} deg)")
            for deg in (2.0, 5.0):
                for name in ("in_plane", "out_of_plane"):
                    r = row[f"tilt_{name}_{deg:g}deg"]
                    lines.append(f"  normal tilted {deg:g} deg {name.replace('_', ' ')}, camera refitted: rms "
                                 f"{r['rms_px']:.4f} px, max {r['max_px']:.4f} px (f {r['f_rel'] * 100:+.2f} %, "
                                 f"rotation {r['drot_deg']:.3f} deg)")
            lines.append("")
            print("\n".join(lines[-13:]), flush=True)
    (HERE / "p_pre_sizing_out.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / "p_pre_sizing.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
