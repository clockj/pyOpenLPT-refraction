"""Shared helpers for the cylinder wand-calibration tests: paths, the case's tube, synthetic rigs and wand data.

Synthetic detections come from pyOpenLPT's C++ PINCYL camera (the source of truth), never from the fast model
being tested.  Rigs are built in the tube frame (axis = Y through the origin, camera 1 at azimuth 0 and height 0),
so the ground truth is already in the calibration's gauge.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CURVED = REPO / "modules" / "camera_calibration" / "curved_calibration"
for _p in (str(REPO / "build-py" / os.environ.get("CONFIG", "Release")), str(CURVED), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cyl_model as cm  # noqa: E402

# the real rig (cases/cyl_20260930_T0_cam1/config.json): OD 7.9375 in, wall 0.125 in; air | acrylic | water
CASE_TUBE = cm.Tube(radii=(100.80625, 97.63125), n=(1.0, 1.49, 1.333))
CAM1_F = 41459.1          # px, the case's fit (2026-10-01-report.md, M1bx)
CAM1_D = 1094.13          # mm, camera centre to axis
PIXEL_MM = 0.020


def look_at(C, target, up=(0.0, 1.0, 0.0), roll_deg: float = 0.0) -> np.ndarray:
    """World -> camera rotation of a camera at C looking at `target`, image up along `up`, then rolled."""
    z = np.asarray(target, float) - np.asarray(C, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    if roll_deg:
        c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
        R = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) @ R
    return R


def ring_camera(azimuth_deg: float, D: float, f: float, width: int, height: int, height_mm: float = 0.0,
                aim=(0.0, 0.0, 0.0), roll_deg: float = 0.0, dist=None) -> cm.Camera:
    """A camera at distance D from the axis, at an azimuth measured from +Z towards +X, looking at `aim`."""
    a = math.radians(azimuth_deg)
    C = np.array([D * math.sin(a), height_mm, D * math.cos(a)])
    R = look_at(C, aim, roll_deg=roll_deg)
    return cm.Camera(R, C, f, (width - 1) / 2.0, (height - 1) / 2.0, width, height,
                     np.zeros(5) if dist is None else dist)


@dataclass
class Wand:
    uv: np.ndarray            # (F, Nc, 2, 2): frame, camera, end (A, B), (u, v); NaN where not seen
    XA: np.ndarray            # (F, 3) truth
    XB: np.ndarray            # (F, 3) truth
    L: float


def wand_frames(cams, tube: cm.Tube, n_frames: int, L: float, rng, half=(20.0, 20.0, 20.0), centre=(0, 0, 0),
                noise_px: float = 0.1, min_cams: int = 2, r_max: float = None, cpp_tol: float = 1e-12,
                all_cams: bool = False, max_tries: int = 200) -> Wand:
    """Wand frames with midpoints uniform in a box (half-sizes `half` about `centre`), directions uniform on the
    sphere, both ends inside r_max (default: the fold radius, so every point has one image).  Detections are the
    C++ primary images, on the sensor, plus Gaussian noise.  A frame is kept when both ends are seen by at least
    `min_cams` cameras (or by every camera with all_cams)."""
    cpp = [cm.to_cpp(c, tube, tol=cpp_tol) for c in cams]
    r_max = tube.fold_radius() * 0.98 if r_max is None else r_max
    half = np.asarray(half, float)
    centre = np.asarray(centre, float)
    uvs, XA, XB = [], [], []
    tries = 0
    while len(uvs) < n_frames and tries < max_tries:
        tries += 1
        m = n_frames - len(uvs)
        k = max(2 * m, 64)
        mid = centre + rng.uniform(-1, 1, size=(k, 3)) * half
        w = rng.normal(size=(k, 3))
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        A, B = mid - 0.5 * L * w, mid + 0.5 * L * w
        rA = np.hypot(A[:, 0], A[:, 2])
        rB = np.hypot(B[:, 0], B[:, 2])
        keep = (rA < r_max) & (rB < r_max)
        A, B = A[keep], B[keep]
        uv = np.full((len(A), len(cams), 2, 2), np.nan)
        for c, (cam, cc) in enumerate(zip(cams, cpp)):
            for e, P in enumerate((A, B)):
                q = cm.cpp_project(cc, P)
                on = (q[:, 0] >= 0) & (q[:, 0] < cam.width) & (q[:, 1] >= 0) & (q[:, 1] < cam.height)
                q[~on] = np.nan
                uv[:, c, e] = q
        seen = np.all(np.isfinite(uv[..., 0]), axis=2)            # (k, Nc) both ends seen
        n_seen = seen.sum(axis=1)
        good = n_seen == len(cams) if all_cams else n_seen >= min_cams
        # an end seen in a camera where the other end is not: drop that camera's half-observation as well
        for i in np.flatnonzero(good):
            for c in range(len(cams)):
                if not seen[i, c]:
                    uv[i, c] = np.nan
        uvs += list(uv[good][:m])
        XA += list(A[good][:m])
        XB += list(B[good][:m])
    uv = np.array(uvs)
    if noise_px > 0:
        uv = uv + rng.normal(scale=noise_px, size=uv.shape)
    return Wand(uv=uv, XA=np.array(XA), XB=np.array(XB), L=float(L))


def reference_rig(n_cams: int = 3, spacing_deg: float = 45.0, D: float = CAM1_D, f: float = None,
                  sensor=(2600, 2000), dist=None):
    """Replicas of cam1 (f/D as the case's camera) at azimuths 0, spacing, 2 spacing, ... with a sensor large enough
    for the tested volume (the user adjusts the field of view).  With D other than cam1's, f keeps f/D."""
    f = CAM1_F * D / CAM1_D if f is None else f
    return [ring_camera(k * spacing_deg, D, f, sensor[0], sensor[1], dist=dist) for k in range(n_cams)]


def gauge_fix(cams, points=None):
    """Rotate about Y and slide along Y so camera 1 sits at X = 0 (Z > 0) and Y = 0 (the calibration's gauge).
    Returns new cameras (and points)."""
    C0 = cams[0].C
    phi = math.atan2(C0[0], C0[2])
    c, s = math.cos(phi), math.sin(phi)
    Ry = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])     # rotates C0 onto the +Z half-plane
    shift = np.array([0.0, C0[1], 0.0])
    out = []
    for cam in cams:
        k = cam.copy()
        k.R = cam.R @ Ry.T
        k.C = Ry @ cam.C - shift
        out.append(k)
    if points is None:
        return out
    return out, (np.asarray(points, float) @ Ry.T) - shift


class Tee:
    """Print to the console and to a log file beside the test.  The file is opened at the first line written, so a
    test module imported by another (for its helpers) does not truncate its own log."""

    def __init__(self, path):
        self.path = path
        self.f = None

    def __call__(self, *args):
        s = " ".join(str(a) for a in args)
        print(s, flush=True)
        if self.f is None:
            self.f = open(self.path, "w", encoding="utf-8")
        self.f.write(s + "\n")
        self.f.flush()

    def close(self):
        if self.f is not None:
            self.f.close()


def irregular_rig(sensor=(5000, 5000)):
    """Four cameras of unequal geometry: azimuths 0, 70, 160, 250 deg; 0.6-1.2 m from the axis; heights -60..+50 mm;
    aimed a few mm off the axis; rolled; f/D from 0.8x to 1.3x cam1's.  Camera 1 at azimuth 0, height 0 (gauge)."""
    spec = [(0.0, 1094.13, 0.0, (0.0, 0.0, 0.0), 0.0, 1.0),
            (70.0, 800.0, 50.0, (3.0, -2.0, 1.0), 8.0, 1.3),
            (160.0, 1200.0, -60.0, (-2.0, 4.0, -3.0), -5.0, 0.8),
            (250.0, 600.0, 20.0, (1.0, 1.0, 4.0), 12.0, 1.1)]
    out = []
    for az, D, h, aim, roll, k in spec:
        f = CAM1_F * D / CAM1_D * k
        out.append(ring_camera(az, D, f, sensor[0], sensor[1], height_mm=h, aim=aim, roll_deg=roll))
    return out


def perturb_rig(cams, rng, rot_deg=2.0, trans_mm=20.0, f_rel=0.05):
    """Each camera turned by rot_deg about a random axis (about its centre), moved by trans_mm in a random direction,
    f scaled by 1 +- f_rel; then put back into the gauge (camera 1 at X = Y = 0)."""
    out = []
    for cam in cams:
        k = cam.copy()
        a = rng.normal(size=3)
        a /= np.linalg.norm(a)
        k.R = cam.R @ cm.exp_so3(a * np.radians(rot_deg))
        t = rng.normal(size=3)
        k.C = cam.C + trans_mm * t / np.linalg.norm(t)
        k.f = cam.f * (1.0 + f_rel * rng.choice([-1.0, 1.0]))
        out.append(k)
    return gauge_fix(out)


def compare(est, truth):
    """Per camera: rotation error (deg), centre error (mm), relative f error."""
    out = []
    for e, t in zip(est, truth):
        out.append({"rot_deg": cm.rotation_angle_deg(e.R, t.R), "centre_mm": float(np.linalg.norm(e.C - t.C)),
                    "f_rel": float(e.f / t.f - 1.0)})
    return out


def local_delta(est, truth):
    """The estimate's offset from the truth in the fit's local coordinates, per camera:
    (w with R_est = R_true exp([w]x), C_est - C_true, log(f_est / f_true))."""
    out = []
    for e, t in zip(est, truth):
        out.append(np.r_[cm.log_so3(t.R.T @ e.R), e.C - t.C, np.log(e.f / t.f)])
    return out


def random_rig(rng, sensor=(5000, 5000)):
    """A random rig for T5: 2-4 cameras (mostly 3) ringing the tube with gaps of 30-120 deg, 0.4-1.5 m from the axis,
    heights within +-80 mm, f/D 0.7-1.4x cam1's, aimed within 5 mm of the cloud's centre, rolled +-20 deg (one in
    five turned to portrait); the cloud centred within +-12 mm of the axis (+-10 mm along it), half-size 12-25 mm.
    Camera 1 at azimuth 0 and height 0 (the gauge).  Returns (cameras, cloud centre, half-size)."""
    n = int(rng.choice([2, 3, 3, 3, 4, 4]))
    centre = np.array([rng.uniform(-12, 12), rng.uniform(-10, 10), rng.uniform(-12, 12)])
    half = float(rng.uniform(12, 25))
    az = 0.0
    cams = []
    for k in range(n):
        if k:
            az += float(rng.uniform(30, 120))
        D = float(rng.uniform(400, 1500))
        f = CAM1_F / CAM1_D * D * float(rng.uniform(0.7, 1.4))
        h = 0.0 if k == 0 else float(rng.uniform(-80, 80))
        aim = centre + rng.uniform(-5, 5, 3)
        roll = float(rng.uniform(-20, 20)) + (90.0 if rng.random() < 0.2 else 0.0)
        cams.append(ring_camera(az, D, f, sensor[0], sensor[1], height_mm=h, aim=aim, roll_deg=roll))
    return cams, centre, half


def realised_unseen(fit, tube_fit, truth, tube_true, P):
    """Realised calibration error on unseen points P (tube frame), RMS per 2-D image: (fixed, modulo the
    stabiliser, re-triangulated).  The fitted cameras project through `tube_fit`, the true ones through
    `tube_true` (they differ when T9 fits with wrong constants)."""
    import cylinder_wand as cw
    ey = np.array([0.0, 1.0, 0.0])
    nc = len(truth)
    uv_t, uv_f, JX = [], [], []
    ok = np.ones(len(P), bool)
    for cf, ct in zip(fit, truth):
        rwt = cm.rows([ct], np.zeros(len(P), int))
        pt = cm.project(rwt, P, cm.pinhole_start(rwt, P), tube_true)
        rwf = cm.rows([cf], np.zeros(len(P), int))
        pf = cm.project(rwf, P, cm.undistort(pt.uv, rwf), tube_fit, jac=True)
        ok &= pt.ok & pf.ok
        uv_t.append(pt.uv)
        uv_f.append(pf.uv)
        JX.append(pf.J_X)
    d = np.stack([uv_f[c] - uv_t[c] for c in range(nc)], axis=1)[ok]
    fixed = math.sqrt(float(np.mean(np.sum(d ** 2, axis=2))))
    G = []
    for i in np.flatnonzero(ok):
        g = np.column_stack([np.cross(ey, P[i]), ey])
        G.append(np.vstack([JX[c][i] @ g for c in range(nc)]))
    G = np.vstack(G)
    dv = d.reshape(-1)
    coef = np.linalg.lstsq(G, dv, rcond=None)[0]
    stab = math.sqrt(float(np.mean(np.sum((dv - G @ coef).reshape(-1, 2) ** 2, axis=1))))
    m = int(ok.sum())
    cam_idx = np.tile(np.arange(nc), m)
    pt_idx = np.repeat(np.arange(m), nc)
    uvs = np.stack([uv_t[c][ok] for c in range(nc)], axis=1).reshape(-1, 2)
    _X, r, _cov = cw.triangulate_ml(fit, tube_fit, cam_idx, uvs, pt_idx, P[ok])
    retri = math.sqrt(float(np.nanmean(np.sum(r ** 2, axis=1))))
    return fixed, stab, retri
