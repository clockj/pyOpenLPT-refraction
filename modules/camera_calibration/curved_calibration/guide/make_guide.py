"""Data and page of the wand-calibration guide, WAND_CALIBRATION_GUIDE.html beside this script.

Every number the page quotes and every ray it draws is computed here with the calibration's own camera models
(plate_model, cyl_model) and functions (plate_wand, cylinder_wand):
  * the flat-wall rig: the real run 20260727/T0 and its delivered calibration (cases/plate_20260727_T0,
    tests/states/w4c_ref.pkl, written to Results/T0/calib_plate_wand_ref_centre), in the walls' frame of the case's
    tests/rig_3d.py;
  * the tube rig: tests/cylinder_wand's reference rig (three copies of the 20260930 camera, 45 deg apart, around the
    20260930 tube) with 500 synthetic wand frames from cyl_model (0.1 px noise), calibrated from scratch by
    cylinder_wand.calibrate;
  * the windows: how much an image changes when a flat wall moves 10 mm away and the wand (n_water - 1) x 10 mm towards
    the camera, by the camera's angle to the wall; a 2 mm ball's image through the tube across the volume;
  * stage 1 on T0: plate_wand.affine_start, its singular values, the metric upgrade, and both mirror images against the
    delivered calibration;
  * stage 2: each T0 camera's perspective line (plate_wand._distance_from_magnification), the pinhole start, the window
    search's candidate normals (plate_wand.fibonacci_cap), and cylinder_wand.bootstrap's placement search replayed
    candidate by candidate, with each camera's stretch-ratio curve (cylinder_wand._distance_from_ratio).
Rays drawn through a window are traced here by Snell's law in vector form from the camera model's own line of sight
for that point, and each must pass within 1e-6 mm of its point.

    python make_guide.py                                   # compute guide_data.json, then build the page
    python make_guide.py --page-only                       # rebuild the page from guide_data.json alone
    python make_guide.py --page-only --fragment OUT.html   # also the page without its document skeleton (publishing)
Writes guide_data.json, WAND_CALIBRATION_GUIDE.html and make_guide_out.txt beside this script.  Needs the refract
environment, the T0 case's detections (X:) and its saved fit (git-ignored), unless --page-only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CURVED = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(CURVED)))
CASE = os.path.join(REPO, "cases", "plate_20260727_T0")
CYL_TESTS = os.path.join(REPO, "tests", "cylinder_wand")
DATA_PATH = os.path.join(HERE, "guide_data.json")
TEMPLATE = os.path.join(HERE, "guide_template.html")
PAGE = os.path.join(HERE, "WAND_CALIBRATION_GUIDE.html")
LOG_PATH = os.path.join(HERE, "make_guide_out.txt")

SEED = 20261004
N_SEG = 160                  # wand segments drawn in each 3-D scene
N_SCATTER = 700              # points per camera and mirror image in the perspective plot
TUBE_FRAMES = 500            # synthetic wand frames through the tube
TUBE_NOISE_PX = 0.1
TUBE_HALF_MM = 20.0          # the wand's midpoints: a box of +-20 mm about the axis
WAND_L = 10.0
BALL_R_MM = 1.0              # a 2 mm ball
BALL_X = [-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]     # across camera 1's view
BALL_Z = [-20.0, 0.0, 20.0]  # along its view: far side, axis, near side
N_SURF = 6000
WALL_MOVE_MM = 10.0
WALL_DEPTH_MM = 60.0         # the points' depth below the wall's water-side surface
TILTS = [round(0.5 * k, 1) for k in range(61)]                      # 0 to 30 deg
N_ROW = 21                   # pixels across the image row


class Log:
    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def rl(a, k=3):
    """A nested list of floats rounded to k decimals (JSON-friendly)."""
    return np.round(np.asarray(a, float), k).tolist()


# ------------------------------------------------------------------ geometry for drawing
def refract(d, nrm, n1, n2):
    """Snell's law in vector form: unit direction d crossing a surface whose unit normal nrm faces the incoming ray
    (nrm . d < 0), from index n1 into n2."""
    cosi = -float(nrm @ d)
    r = n1 / n2
    k = 1.0 - r * r * (1.0 - cosi * cosi)
    if k < 0:
        raise ValueError("total internal reflection")
    t = r * d + (r * cosi - math.sqrt(k)) * nrm
    return t / np.linalg.norm(t)


def trace_plate(C, d, win):
    """The ray from C along unit d through the flat window win (plate_model.Window): its points [C, camera-side hit,
    ..., object-side hit] and its direction in the object medium."""
    n = win.n
    pts = [np.asarray(C, float)]
    P, dd, level = pts[0], np.asarray(d, float), win.d
    for k in range(len(win.t) + 1):
        P = P + (level - float(n @ P)) / float(n @ dd) * dd
        pts.append(P)
        dd = refract(dd, -n, win.idx[k], win.idx[k + 1])
        if k < len(win.t):
            level += win.t[k]
    return pts, dd


def trace_tube(C, d, tube):
    """The ray from C along unit d into the tube (axis Y through the origin): its points [C, outer hit, ..., inner
    hit] and its direction in the water."""
    pts = [np.asarray(C, float)]
    P, dd = pts[0], np.asarray(d, float)
    for k, R in enumerate(tube.radii):
        A = dd[0] ** 2 + dd[2] ** 2
        B = 2.0 * (P[0] * dd[0] + P[2] * dd[2])
        Cq = P[0] ** 2 + P[2] ** 2 - R * R
        disc = B * B - 4.0 * A * Cq
        if disc < 0:
            raise ValueError("the ray misses the tube")
        P = P + (-B - math.sqrt(disc)) / (2.0 * A) * dd
        pts.append(P)
        dd = refract(dd, np.array([P[0], 0.0, P[2]]) / R, tube.n[k], tube.n[k + 1])
    return pts, dd


def miss(X, P, d):
    """Distance of X from the line through P along unit d."""
    w = np.asarray(X, float) - P
    return float(np.linalg.norm(w - (w @ d) * d))


def air_direction(cam, xy):
    """The camera's line of sight in air for normalised undistorted image coordinates xy."""
    v = cam.R.T @ np.array([xy[0], xy[1], 1.0])
    return v / np.linalg.norm(v)


def look_at(z, up=(0.0, 1.0, 0.0)):
    """World -> camera rotation of a camera looking along z, image x = z x up."""
    z = np.asarray(z, float) / np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    x /= np.linalg.norm(x)
    return np.vstack([x, np.cross(z, x), z])


def sphere_points(n):
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    th = np.pi * (1 + 5 ** 0.5) * k
    return np.column_stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)])


def polygon_moments(P):
    """Area, centroid and the second central moments (xx, yy, xy) of the polygon P (m, 2), counter-clockwise."""
    x, y = P[:, 0], P[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    c = x * y1 - x1 * y
    A = c.sum() / 2
    cx = ((x + x1) * c).sum() / (6 * A)
    cy = ((y + y1) * c).sum() / (6 * A)
    ixx = ((x * x + x * x1 + x1 * x1) * c).sum() / 12 - A * cx * cx
    iyy = ((y * y + y * y1 + y1 * y1) * c).sum() / 12 - A * cy * cy
    ixy = ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * c).sum() / 24 - A * cx * cy
    return A, np.array([cx, cy]), np.array([[ixx, ixy], [ixy, iyy]])


def resample(P, n):
    """n points evenly spaced along the closed polygon P."""
    Q = np.vstack([P, P[:1]])
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, Q[:, 0]), np.interp(t, s, Q[:, 1])])


def rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def similarity(P, Q):
    """s, R (proper rotation), t minimising |s R P + t - Q| (Umeyama); and the RMS left."""
    mp, mq = P.mean(axis=0), Q.mean(axis=0)
    P0, Q0 = P - mp, Q - mq
    U, S, Vt = np.linalg.svd(Q0.T @ P0)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / np.sum(P0 ** 2))
    t = mq - s * R @ mp
    return s, R, t, float(np.sqrt(np.mean(np.sum((s * P @ R.T + t - Q) ** 2, axis=1))))


def polar(v, g, t1, t2):
    """Angle (deg) of unit v from g, and its azimuth (deg) in the basis t1, t2 perpendicular to g."""
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(v @ g)))))
    return tilt, math.degrees(math.atan2(float(v @ t2), float(v @ t1)))


# ------------------------------------------------------------------ the flat-wall rig: run 20260727/T0
def plate_data(log):
    sys.path.insert(0, CASE)
    sys.path.insert(0, os.path.join(CASE, "tests"))
    import case          # noqa: E402  (puts the curved-calibration modules on the path)
    import rig_3d        # noqa: E402
    pm, pw = case.pm, case.pw
    cm = pw.cm
    cfg = case.load_config()
    st = case.load_state(os.path.join(CASE, "tests", "states", "w4c_ref.pkl"))
    rc, cd, rows, data, obs, frames = case.rebuild(st, cfg)
    cams, rig = st["cams"], st["rig"]
    L = float(st["L"])
    Nc = len(cams)
    with open(os.path.join(CASE, "tests", "w4c_delivery.json"), encoding="utf-8") as fh:
        delivery = json.load(fh)
    ev = delivery["every_frame"]["w4c_ref"]
    out = {"source": "cases/plate_20260727_T0, tests/states/w4c_ref.pkl (Results/T0/calib_plate_wand_ref_centre)",
           "n_frames": int(data.n_frames), "wand_mm": L,
           "result": {"rms_px": delivery["w4c_ref"]["rms_px"], "sigma_px": delivery["w4c_ref"]["sigma_px"],
                      "median_px": ev["median_px"], "wand_abs_median_mm": ev["wand_abs_median_mm"],
                      "active": delivery["w4c_ref"]["active"], "detections": int(len(obs)),
                      "seconds_from_start": delivery["w4c_ref"]["seconds"]},
           "params": len(st["layout"]["names"]), "param_names": list(st["layout"]["names"])}
    log(f"T0: {data.n_frames} frames, {len(obs)} detections; delivered fit RMS {out['result']['rms_px']:.4f} px, "
        f"median {ev['median_px']:.4f} px, wand {ev['wand_abs_median_mm']:.4f} mm; {out['params']} rig parameters")

    # ---- the 3-D scene, in rig_3d's walls' frame
    XA, XB = frames.ends(L)
    act = np.zeros(len(XA), bool)
    act[np.asarray(st["obs_frame"])[np.asarray(st["active"], bool)]] = True
    P = np.vstack([XA[act], XB[act]])
    O, B, up = rig_3d.frame(cams, rig, P)
    to = lambda X: (B @ (np.asarray(X, float) - O))     # noqa: E731
    ch = {k: v for k, v in cfg["constraints"]["channel"].items() if not k.startswith("_")}
    fd = rig_3d.fit_data(st, "calib_plate_wand_ref_centre", ch["width_mm"])
    idx = np.flatnonzero(act)
    take = idx[np.linspace(0, len(idx) - 1, N_SEG).round().astype(int)]
    segs = [rl([to(XA[i]), to(XB[i])], 2) for i in take]
    # rays from every camera to two wand ends at opposite ends of the cloud's longest axis
    Pc = P - P.mean(axis=0)
    ax = np.linalg.svd(Pc, full_matrices=False)[2][0]
    proj = (XA[idx] + XB[idx]) / 2 @ ax
    pick = [idx[int(np.argsort(proj)[int(0.15 * len(idx))])], idx[int(np.argsort(proj)[int(0.85 * len(idx))])]]
    ends = [XA[pick[0]], XB[pick[1]]]
    end_rays, worst = [], 0.0
    for k, X in enumerate(ends):
        for c, cam in enumerate(cams):
            w = rig.cam_win[c]
            win = rig.windows[w]
            pr = pm.project(cm.rows([cam], np.zeros(1, int)), pm.wrows(rig.windows, np.array([w])), X[None])
            if not pr.ok[0]:
                continue
            pts, dw = trace_plate(cam.C, air_direction(cam, pr.xy[0]), win)
            worst = max(worst, miss(X, pts[-1], dw))
            end_rays.append({"cam": c, "end": k, "path": rl([to(p) for p in pts] + [to(X)], 2)})
    if worst > 1e-6:
        raise AssertionError(f"a traced plate ray misses its point by {worst:.2e} mm")
    log(f"  3-D: {len(segs)} wand segments, {len(end_rays)} rays to 2 wand ends (traced rays meet their points within "
        f"{worst:.1e} mm)")
    out["scene"] = {"up": fd["up"], "cameras": fd["cameras"], "walls": fd["walls"], "box": fd["box_frame_mm"],
                    "segments": segs, "end_rays": end_rays, "channel": ch}

    # ---- the optics: a flat wall's move, by the camera's angle to it
    out["wall_move"] = wall_move(cams, rig, pm, cm, log)
    out["incidence_deg"] = [case.angle_deg(c.R[2], rig.windows[rig.cam_win[c_]].n) for c_, c in enumerate(cams)]
    log("  incidence on their walls: " + ", ".join(f"C{c + 1} {a:.2f} deg" for c, a in enumerate(out["incidence_deg"])))

    # ---- stage 1: the affine start
    obs.active = np.ones(len(obs), bool)              # as at the start of calibrate
    seen = np.zeros((obs.n_frames, Nc, 2), bool)
    seen[obs.frame, obs.cam, obs.end] = True
    whole = int(seen.all(axis=(1, 2)).sum())
    t0 = time.time()
    aff, subset = pw.affine_start(obs, L)
    t_aff = time.time() - t0
    col = np.full(obs.n_frames, -1)
    col[aff.frames] = np.arange(len(aff.frames))
    W = np.zeros((2 * Nc, 2 * len(aff.frames)))
    for i in np.flatnonzero(col[obs.frame] >= 0):
        c, e, k = obs.cam[i], obs.end[i], col[obs.frame[i]]
        W[2 * c:2 * c + 2, 2 * k + e] = obs.uv[i]
    sv = np.linalg.svd(W - W.mean(axis=1, keepdims=True), compute_uv=False)
    Pf = np.vstack([XA[aff.frames], XB[aff.frames]])
    centre = Pf.mean(axis=0)
    mags = []
    for c, cam in enumerate(cams):
        w = rig.cam_win[c]
        h = 0.5
        X = np.vstack([centre + h * np.eye(3)[k] for k in range(3)] + [centre - h * np.eye(3)[k] for k in range(3)])
        pr = pm.project(cm.rows([cam], np.zeros(6, int)), pm.wrows(rig.windows, np.full(6, w)), X)
        J = (pr.uv[:3] - pr.uv[3:]).T / (2 * h)
        mags.append({"affine": rl(np.linalg.svd(aff.M[c], compute_uv=False), 4),
                     "final": rl(np.linalg.svd(J, compute_uv=False), 4)})
    Pa = np.vstack([aff.XA, aff.XB])
    mirrors = []
    for mirror in (False, True):
        Pm = np.diag([1.0, 1.0, -1.0]) if mirror else np.eye(3)
        s, Rp, _t, rms = similarity(Pa @ Pm, Pf)
        errs = []
        for c in range(Nc):
            U, S, Vt = np.linalg.svd(aff.M[c] @ Pm, full_matrices=False)
            Oc = U @ Vt
            errs.append(rot_angle_deg(cams[c].R, np.vstack([Oc[0], Oc[1], np.cross(Oc[0], Oc[1])]) @ Rp.T))
        mirrors.append({"mirror": mirror, "rms_mm": rms, "scale": s, "rot_deg": errs})
    right = min(mirrors, key=lambda m: m["rms_mm"])
    spread = float(np.sqrt(np.mean(np.sum((Pf - centre) ** 2, axis=1))))
    out["affine"] = {"frames_total": int(data.n_frames), "frames_whole": whole, "frames_kept": int(len(aff.frames)),
                     "subset": [int(c) for c in subset], "rows": int(W.shape[0]), "cols_all": 2 * whole,
                     "cols_kept": int(W.shape[1]), "singular_values": rl(sv, 2), "rms_px": aff.rms_px,
                     "length_rms_rel": aff.length_rms_rel, "Q_eig": rl(aff.Q_eig, 3), "seconds": t_aff,
                     "magnifications": mags, "mirrors": mirrors, "right_mirror": right["mirror"],
                     "cloud_rms_radius_mm": spread}
    log(f"  stage 1: {whole} of {data.n_frames} frames seen whole by all cameras, {len(aff.frames)} kept; matrix "
        f"{W.shape[0]} x {W.shape[1]}; singular values " + ", ".join(f"{v:.4g}" for v in sv)
        + f"; rank-3 residual {aff.rms_px:.3f} px; lengths |d|^2/L^2-1 RMS {aff.length_rms_rel:.4f}; {t_aff:.2f} s")
    for m in mirrors:
        log(f"    mirror {m['mirror']!s:5}: {m['rms_mm']:.3f} mm RMS from the delivered fit's wand ends (cloud RMS radius "
            f"{spread:.1f} mm), scale {m['scale']:.4f}, rotations " + ", ".join(f"{e:.2f}" for e in m["rot_deg"])
            + " deg")
    for c, m in enumerate(mags):
        log(f"    C{c + 1} magnifications: affine {m['affine'][0]:.4f}, {m['affine'][1]:.4f} px/mm; final model at the "
            f"cloud's centre {m['final'][0]:.4f}, {m['final'][1]:.4f}")

    # ---- stage 2: perspective (near wands look longer), per camera and mirror image
    persp = []
    for mirror in (False, True):
        Pm = np.diag([1.0, 1.0, -1.0]) if mirror else np.eye(3)
        XAm, XBm = aff.XA @ Pm, aff.XB @ Pm
        cen = np.vstack([XAm, XBm]).mean(axis=0)
        per = []
        for c in range(Nc):
            M = aff.M[c] @ Pm
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            Oc = U @ Vt
            R = np.vstack([Oc[0], Oc[1], np.cross(Oc[0], Oc[1])])
            rr = np.flatnonzero(obs.active & (obs.cam == c) & (col[obs.frame] >= 0))
            uvw = np.full((len(aff.frames), 2, 2), np.nan)
            uvw[col[obs.frame[rr]], obs.end[rr]] = obs.uv[rr]
            D, f, slope = pw._distance_from_magnification(R, XAm, XBm, uvw[:, 0], uvw[:, 1], L, cen)
            dX = XBm - XAm
            wv = dX / np.linalg.norm(dX, axis=1, keepdims=True)
            z = (0.5 * (XAm + XBm) - cen) @ R[2]
            lp = L * np.linalg.norm(wv - (wv @ R[2])[:, None] * R[2], axis=1)
            lu = np.linalg.norm(uvw[:, 1] - uvw[:, 0], axis=1)
            sel = (lp > 0.5 * L) & np.isfinite(lu) & (lu > 0)
            y = lp[sel] / lu[sel]
            zz = z[sel]
            A = np.column_stack([np.ones(len(y)), zz])
            wts = np.ones(len(y))
            coef = np.zeros(2)
            for _ in range(10):                      # _distance_from_magnification's Huber line, for its intercept
                coef = np.linalg.lstsq(A * np.sqrt(wts)[:, None], y * np.sqrt(wts), rcond=None)[0]
                r = y - A @ coef
                sr = 1.4826 * float(np.median(np.abs(r)))
                wts = np.minimum(1.0, 1.345 * sr / np.maximum(np.abs(r), 1e-300))
            if abs(coef[1] - slope) > 1e-12 * max(1.0, abs(slope)):
                raise AssertionError("the perspective line differs from plate_wand's")
            pick = np.linspace(0, len(y) - 1, min(N_SCATTER, len(y))).round().astype(int)
            win = rig.windows[rig.cam_win[c]]
            per.append({"z": rl(zz[pick], 2), "y": rl(y[pick] * 1000.0, 4), "a": float(coef[1]),
                        "b": float(coef[0]), "D": None if not np.isfinite(D) else float(D),
                        "f": None if not np.isfinite(f) else float(f), "n_used": int(sel.sum()),
                        "f_final": float(cams[c].f), "f_final_water": float(cams[c].f * win.idx[-1] / win.idx[0])})
            log(f"    perspective, mirror {mirror!s:5}, C{c + 1}: {int(sel.sum())} wands, slope {coef[1]:+.3e} /px, "
                f"intercept {coef[0]:.4e} mm/px -> D {D:.1f} mm, f {f:.0f} px (delivered f x n_water/n_air "
                f"{cams[c].f * win.idx[-1] / win.idx[0]:.0f} px)")
        persp.append({"mirror": mirror, "cams": per})
    out["perspective"] = persp
    pins, pinfo = pw.pinhole_start(obs, L, rc.sizes, [((w - 1) / 2.0, (h - 1) / 2.0) for (w, h) in rc.sizes], rig)
    out["pinhole_start"] = {"rms_px": [float(s.rms) for s in pins], "mirrors": [bool(s.info["mirror"]) for s in pins],
                            "votes": pinfo["mirror_votes"],
                            "f_px": [[float(c.f) for c in s.cams] for s in pins]}
    log(f"  pinhole start: kept {len(pins)} (mirror votes {pinfo['mirror_votes']}); RMS "
        + ", ".join(f"{s.rms:.3f} px (mirror {s.info['mirror']})" for s in pins)
        + "; f " + ", ".join(f"{c.f:.0f}" for c in pins[0].cams))

    # ---- stage 2 (walls): the window search's candidate normals about each wall's cameras' viewing direction
    cands = []
    for w in range(len(rig.windows)):
        cw_ = rig.cams_of(w)
        S = sum(np.outer(cams[c].R[2], cams[c].R[2]) for c in cw_)
        g = np.linalg.eigh(S)[1][:, -1]
        if float(g @ (P.mean(axis=0) - np.mean([cams[c].C for c in cw_], axis=0))) < 0:
            g = -g
        normals = pw.fibonacci_cap(g, 50.0, 9.0)
        t1, t2 = pm.tangent_basis(g)
        t1, t2 = t1[0], t2[0]
        cands.append({"wall": w + 1, "cams": [int(c) for c in cw_],
                      "normals": [rl(polar(n, g, t1, t2), 2) for n in normals],
                      "final": rl(polar(rig.windows[w].n, g, t1, t2), 2),
                      "views": [rl(polar(cams[c].R[2], g, t1, t2), 2) for c in cw_]})
        log(f"  window search, wall {w + 1}: {len(normals)} candidate normals x 3 offsets; the delivered normal is "
            f"{cands[-1]['final'][0]:.2f} deg from the cameras' mean view")
    out["window_candidates"] = cands
    return out


def wall_move(cams, rig, pm, cm, log):
    """How much a flat wall's position shows in an image: a camera like T0's C1 (its f, image, distance to its wall,
    the wall's plate and indices) tilted by each angle to the wall's normal; the points 60 mm below the wall's
    water-side surface on each pixel's line of sight along the middle image row; the wall moved 10 mm away from the
    camera and the points (n_water - 1) x 10 mm towards it, the paraxial compensation.  Returns each tilt's image
    change along the row (px)."""
    ref = cams[0]
    wr = rig.windows[rig.cam_win[0]]
    f, Wd, Hd = ref.f, ref.width, ref.height
    h = wr.distance(ref.C)
    n_w = wr.idx[-1]
    nrm = np.array([0.0, 0.0, 1.0])
    us = np.linspace(0.0, Wd - 1.0, N_ROW)
    uv = np.column_stack([us, np.full(N_ROW, (Hd - 1) / 2.0)])
    zero = np.zeros(N_ROW, int)
    rows, worst = [], 0.0
    for th in TILTS:
        t = math.radians(th)
        cam = pm.Camera(look_at([math.sin(t), 0.0, math.cos(t)]), np.zeros(3), f, (Wd - 1) / 2.0, (Hd - 1) / 2.0,
                        Wd, Hd, np.zeros(5))
        rw = cm.rows([cam], zero)
        w0 = pm.Window(nrm, h, wr.t, wr.idx)
        w1 = pm.Window(nrm, h + WALL_MOVE_MM, wr.t, wr.idx)
        o, d, ok = pm.rays(rw, pm.wrows([w0], zero), uv)
        X = o + (WALL_DEPTH_MM / (d @ nrm))[:, None] * d
        p0 = pm.project(rw, pm.wrows([w0], zero), X)
        p1 = pm.project(rw, pm.wrows([w1], zero), X - (n_w - 1.0) * WALL_MOVE_MM * nrm)
        worst = max(worst, float(np.max(np.abs(p0.uv - uv))))
        du = p1.uv - uv
        ang = [math.degrees(math.acos(float(air_direction(cam, ((u - cam.cx) / f, 0.0)) @ nrm))) for u in us]
        # what neither a uniform shift (a slight turn of the camera) nor a change of scale (f, distance) can absorb
        A = np.column_stack([np.ones(N_ROW), us - cam.cx])
        left = du[:, 0] - A @ np.linalg.lstsq(A, du[:, 0], rcond=None)[0]
        rows.append({"tilt": th, "du": rl(du[:, 0], 6), "dv": rl(du[:, 1], 6), "angle": rl(ang, 3),
                     "left": rl(left, 7)})
    if worst > 1e-6:
        raise AssertionError(f"the plate model does not reproduce its own rays ({worst:.1e} px)")
    half = math.degrees(math.atan((Wd - 1) / 2.0 / f))
    log(f"  a wall's move: camera f {f:.0f} px, {h:.0f} mm from its wall (plate {wr.T} mm, n {wr.idx}); field half-angle "
        f"{half:.2f} deg; image change at the centre / across the row (max - min, px):")
    for r in rows:
        if r["tilt"] in (0.0, 1.0, 3.0, 5.0, 10.0, 13.0, 16.5, 20.0, 30.0):
            c = r["du"][N_ROW // 2]
            log(f"    tilt {r['tilt']:4.1f} deg: centre {c:+.2e} px, across the row {max(r['du']) - min(r['du']):.2e} px, "
                f"left after the best shift and scale {max(r['left']) - min(r['left']):.2e} px")
    return {"f": f, "h_mm": h, "plate_mm": wr.T, "n": list(wr.idx), "move_mm": WALL_MOVE_MM,
            "depth_mm": WALL_DEPTH_MM, "half_field_deg": half, "u": rl(us, 1), "rows": rows}


# ------------------------------------------------------------------ the tube rig: synthetic, the reference rig
def synth_wand(cams, tube, cm, n, L, rng, half, noise):
    """Wand frames with midpoints uniform in a +-half box about the origin and random directions, both ends inside
    0.98 of the fold radius; detections from cyl_model on the sensor plus Gaussian noise; kept when both ends are seen
    by at least two cameras."""
    r_max = 0.98 * tube.fold_radius()
    Nc = len(cams)
    uvs, As, Bs = [], [], []
    while len(uvs) < n:
        k = 2 * (n - len(uvs)) + 64
        mid = rng.uniform(-half, half, size=(k, 3))
        w = rng.normal(size=(k, 3))
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        A, B = mid - 0.5 * L * w, mid + 0.5 * L * w
        keep = (np.hypot(A[:, 0], A[:, 2]) < r_max) & (np.hypot(B[:, 0], B[:, 2]) < r_max)
        A, B = A[keep], B[keep]
        uv = np.full((len(A), Nc, 2, 2), np.nan)
        for c, cam in enumerate(cams):
            for e, X in enumerate((A, B)):
                rw = cm.rows([cam], np.zeros(len(X), int))
                pr = cm.project(rw, X, cm.pinhole_start(rw, X), tube)
                q = pr.uv.copy()
                on = pr.ok & (q[:, 0] >= 0) & (q[:, 0] < cam.width) & (q[:, 1] >= 0) & (q[:, 1] < cam.height)
                q[~on] = np.nan
                uv[:, c, e] = q
        seen = np.all(np.isfinite(uv[..., 0]), axis=2)
        good = seen.sum(axis=1) >= 2
        for i in np.flatnonzero(good):
            uv[i, ~seen[i]] = np.nan
        uvs += list(uv[good])
        As += list(A[good])
        Bs += list(B[good])
    uv = np.array(uvs[:n])
    return uv + rng.normal(scale=noise, size=uv.shape), np.array(As[:n]), np.array(Bs[:n])


def ratio_curve(cm, R, q, cloud, tube, ratio_obs, s_low_obs, D_lo=150.0, D_hi=6000.0, n=36, m_pts=120):
    """cylinder_wand._distance_from_ratio, also returning its curve: the model camera's stretch ratio and low
    magnification at each distance along the principal ray."""
    z = R[2]
    pts = np.asarray(cloud, float)
    if len(pts) > m_pts:
        pts = pts[np.linspace(0, len(pts) - 1, m_pts).round().astype(int)]
    m = len(pts)
    s_grid = np.geomspace(D_lo, D_hi, n)
    C = q[None, :] - s_grid[:, None] * z[None, :]
    outside = np.hypot(C[:, 0], C[:, 2]) > tube.r_outer * 1.02
    s_grid, C = s_grid[outside], C[outside]
    k = len(s_grid)
    rw = cm.Rows(np.repeat(R[None], k * m, axis=0), np.repeat(C, m, axis=0), np.ones(k * m), np.zeros(k * m),
                 np.zeros(k * m), np.zeros((k * m, 5)))
    X = np.tile(pts, (k, 1))
    pr = cm.project(rw, X, cm.pinhole_start(rw, X), tube)
    xy = pr.xy.reshape(k, m, 2)
    ok = pr.ok.reshape(k, m)
    rat = np.full(k, np.nan)
    slow = np.full(k, np.nan)
    A = np.column_stack([pts, np.ones(m)])
    for i in range(k):
        g = ok[i]
        if g.sum() < 12:
            continue
        coef = np.linalg.lstsq(A[g], xy[i][g], rcond=None)[0]
        sv = np.linalg.svd(coef[:3].T, compute_uv=False)
        rat[i], slow[i] = sv[0] / sv[1], sv[1]
    good = np.isfinite(rat)
    s_all, rat_all = s_grid.copy(), rat.copy()
    s_grid, rat, slow = s_grid[good], rat[good], slow[good]
    j0 = int(np.argmin(np.abs(rat - ratio_obs)))
    s_best, sl = float(s_grid[j0]), float(slow[j0])
    for j in (j0 - 1, j0):
        if 0 <= j < len(rat) - 1 and (rat[j] - ratio_obs) * (rat[j + 1] - ratio_obs) <= 0 and rat[j] != rat[j + 1]:
            w = (ratio_obs - rat[j]) / (rat[j + 1] - rat[j])
            s_best = float(np.exp((1 - w) * np.log(s_grid[j]) + w * np.log(s_grid[j + 1])))
            sl = float((1 - w) * slow[j] + w * slow[j + 1])
            break
    return s_best, s_low_obs / sl, s_all, rat_all


def tube_data(log):
    sys.path.insert(0, CYL_TESTS)
    import rigs                                      # noqa: E402
    import cyl_model as cm                           # noqa: E402
    import cylinder_wand as cw                       # noqa: E402
    tube = rigs.CASE_TUBE
    truth = rigs.reference_rig()
    sizes = [(c.width, c.height) for c in truth]
    cxcy = [((w - 1) / 2.0, (h - 1) / 2.0) for (w, h) in sizes]
    Nc = len(truth)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    uv, XA, XB = synth_wand(truth, tube, cm, TUBE_FRAMES, WAND_L, rng, TUBE_HALF_MM, TUBE_NOISE_PX)
    data = cw.WandData(uv, WAND_L, sizes)
    obs = cw.observations(data)
    log(f"tube: radii {tube.radii} mm, n {tube.n}; {Nc} cameras {rigs.CAM1_D} mm from the axis, f {rigs.CAM1_F} px, "
        f"{sizes[0]} px; {TUBE_FRAMES} frames ({len(obs)} detections, {TUBE_NOISE_PX} px noise) in "
        f"{time.time() - t0:.1f} s; fold radius {tube.fold_radius():.2f} mm")
    out = {"radii": list(tube.radii), "n": list(tube.n), "D_mm": rigs.CAM1_D, "f_px": rigs.CAM1_F,
           "sensor": list(sizes[0]), "frames": TUBE_FRAMES, "noise_px": TUBE_NOISE_PX, "half_mm": TUBE_HALF_MM,
           "seed": SEED, "fold_radius_mm": tube.fold_radius()}

    # ---- the 3-D scene, in the tube frame (axis = Y)
    cams3 = []
    for k, cam in enumerate(truth):
        rays = []
        for (u, v) in ((0.0, 0.0), (cam.width - 1.0, 0.0), (cam.width - 1.0, cam.height - 1.0),
                       (0.0, cam.height - 1.0), (cam.cx, cam.cy)):
            pts, dw = trace_tube(cam.C, air_direction(cam, ((u - cam.cx) / cam.f, (v - cam.cy) / cam.f)), tube)
            rays.append({"pts": rl(pts, 2), "d": rl(dw, 6)})
        cams3.append({"name": f"C{k + 1}", "C": rl(cam.C, 2), "z_axis": rl(cam.R[2], 6), "rays": rays})
    take = np.linspace(0, len(XA) - 1, N_SEG).round().astype(int)
    segs = [rl([XA[i], XB[i]], 2) for i in take]
    mids = (XA + XB) / 2
    pick = [int(np.argmin(mids[:, 0] + mids[:, 1])), int(np.argmax(mids[:, 0] - mids[:, 1]))]
    end_rays, worst = [], 0.0
    for k, X in enumerate((XA[pick[0]], XB[pick[1]])):
        for c, cam in enumerate(truth):
            rw = cm.rows([cam], np.zeros(1, int))
            pr = cm.project(rw, X[None], cm.pinhole_start(rw, X[None]), tube)
            if not pr.ok[0]:
                continue
            pts, dw = trace_tube(cam.C, air_direction(cam, pr.xy[0]), tube)
            worst = max(worst, miss(X, pts[-1], dw))
            end_rays.append({"cam": c, "end": k, "path": rl(list(pts) + [X], 2)})
    if worst > 1e-6:
        raise AssertionError(f"a traced tube ray misses its point by {worst:.2e} mm")
    out["scene"] = {"cameras": cams3, "segments": segs, "end_rays": end_rays}
    log(f"  3-D: {len(segs)} wand segments, {len(end_rays)} rays to 2 wand ends (within {worst:.1e} mm)")

    # ---- a ball's image through the tube (camera 1)
    cam = truth[0]
    S = sphere_points(N_SURF) * BALL_R_MM
    balls = []
    for z in BALL_Z:
        for x in BALL_X:
            X0 = np.array([x, 0.0, z])
            X = np.vstack([X0, X0 + S])
            rw = cm.rows([cam], np.zeros(len(X), int))
            pr = cm.project(rw, X, cm.pinhole_start(rw, X), tube)
            if not pr.ok.all():
                log(f"    ball at x {x}, z {z}: {int((~pr.ok).sum())} surface points without an image; skipped")
                continue
            from scipy.spatial import ConvexHull
            c0, q = pr.uv[0], pr.uv[1:]
            poly = q[ConvexHull(q).vertices]
            A, cen, I = polygon_moments(poly)
            evs, V = np.linalg.eigh(I)
            balls.append({"x": x, "z": z, "ratio": float(math.sqrt(evs[1] / evs[0])),
                          "major_px": float(4.0 * math.sqrt(evs[1] / A)), "minor_px": float(4.0 * math.sqrt(evs[0] / A)),
                          "major_dir": rl(V[:, 1], 4), "centroid_offset_px": float(np.linalg.norm(cen - c0)),
                          "outline": rl(resample(poly, 96) - c0, 2)})
    rs = [b["ratio"] for b in balls]
    near = [b["ratio"] for b in balls if abs(b["x"]) <= 8 and abs(b["z"]) <= 8]
    log(f"  a {2 * BALL_R_MM:g} mm ball through the tube (camera 1): axis ratio {min(rs):.3f}-{max(rs):.3f} over "
        f"+-20 mm ({min(near):.3f}-{max(near):.3f} within 8 mm of the axis); image {balls[len(balls) // 2]['major_px']:.1f}"
        f" x {balls[len(balls) // 2]['minor_px']:.1f} px on the axis; centroid offsets up to "
        f"{max(b['centroid_offset_px'] for b in balls):.4f} px")
    out["balls"] = balls

    # ---- stages 1-2: cylinder_wand.bootstrap's placement search, candidate by candidate
    t0 = time.time()
    aff = cw.affine_factorization(obs, WAND_L)
    rows = cw._subset_rows(obs, 150)
    quick_rows = cw._subset_rows(obs, 40)
    layout = cw.Layout(Nc)
    cands, oris, grids = [], {}, {}
    for mirror in (False, True):
        ori = cw.orient(aff, cxcy, mirror)
        oris[mirror] = ori
        r_cloud = float(np.max(np.hypot(np.r_[ori.XA[:, 0], ori.XB[:, 0]], np.r_[ori.XA[:, 2], ori.XB[:, 2]])))
        rho = 0.98 * tube.r_inner - r_cloud
        if rho <= 0:
            continue
        step = max(4.0, rho / 4.0)
        g = np.arange(-rho, rho + 1e-9, step)
        g = g - g[np.argmin(np.abs(g))]
        grids[mirror] = {"rho": rho, "step": step, "r_cloud": r_cloud}
        for px in g:
            for pz in g:
                if px * px + pz * pz > rho * rho:
                    continue
                pl = cw.place(ori, (px, pz), tube, sizes, cxcy, D_range=(150.0, 6000.0), f_range=None)
                if pl is None:
                    cands.append({"mirror": mirror, "p": [float(px), float(pz)], "placed": False})
                    continue
                cs0, _ends = pl
                cost0 = cw._robust_cost(cw.backprojection_residual(cs0, tube, obs, WAND_L, rows))
                cs, _ = cw.coarse_fit(cs0, tube, obs, WAND_L, layout, max_outer=1, max_nfev=8, rows=quick_rows)
                cost_q = cw._robust_cost(cw.backprojection_residual(cs, tube, obs, WAND_L, quick_rows))
                cands.append({"mirror": mirror, "p": [float(px), float(pz)], "placed": True, "cost0": cost0,
                              "cost_q": cost_q, "_cams": cs})
    placed = [c for c in cands if c["placed"]]
    placed.sort(key=lambda c: c["cost_q"])
    for c in placed[:6]:
        cs, _ = cw.coarse_fit(c["_cams"], tube, obs, WAND_L, layout, max_outer=2, max_nfev=40, rows=rows)
        c["cost_s"] = cw._robust_cost(cw.backprojection_residual(cs, tube, obs, WAND_L, rows))
    best = min(placed[:6], key=lambda c: c["cost_s"])
    t_replay = time.time() - t0
    ranked, binfo = cw.bootstrap(obs, tube, WAND_L, sizes)
    same = (bool(binfo["best"]["mirror"]) == best["mirror"]
            and np.allclose(binfo["best"]["p"], best["p"]) and binfo["candidates"] == len(placed))
    if not same:
        raise AssertionError(f"the replayed placement search differs from cylinder_wand.bootstrap: {binfo['best']}")
    log(f"  placement search (replayed in {t_replay:.1f} s, identical to cylinder_wand.bootstrap): {len(placed)} "
        f"placements of {len(cands)} grid points; best: mirror {best['mirror']}, axis at ({best['p'][0]:+.1f}, "
        f"{best['p'][1]:+.1f}) mm from the cloud's centroid; quick cost {best['cost_q']:.4g}, short cost "
        f"{best['cost_s']:.4g}")
    # the true axis position relative to the affine cloud's centroid (the oriented frame matches the tube frame:
    # Y along the axis, Z towards camera 1)
    fid = obs.frame_ids[aff.frames]
    tc = np.vstack([XA[fid], XB[fid]]).mean(axis=0)
    p_true = [-float(tc[0]), -float(tc[2])]
    worst_wrong = min((c for c in placed if c["mirror"] != best["mirror"]), key=lambda c: c["cost_q"], default=None)
    out["placement"] = {
        "candidates": [{k: v for k, v in c.items() if not k.startswith("_")} for c in cands],
        "grids": {str(k): v for k, v in grids.items()}, "best": {"mirror": best["mirror"], "p": best["p"],
                                                                  "cost_q": best["cost_q"], "cost_s": best["cost_s"]},
        "short": [{"mirror": c["mirror"], "p": c["p"], "cost_q": c["cost_q"], "cost_s": c["cost_s"]}
                  for c in sorted(placed[:6], key=lambda c: c["cost_s"])],
        "p_true": p_true, "best_other_mirror_cost_q": None if worst_wrong is None else worst_wrong["cost_q"],
        "seconds": t_replay, "affine_rms_px": aff.rms_px, "affine_frames": int(len(aff.frames))}
    log(f"  the true axis relative to the cloud's centroid: ({p_true[0]:+.2f}, {p_true[1]:+.2f}) mm; the other mirror "
        f"image's best quick cost {worst_wrong['cost_q'] if worst_wrong else float('nan'):.4g}; affine residual "
        f"{aff.rms_px:.3f} px on {len(aff.frames)} frames")
    for c in sorted(placed[:6], key=lambda c: c["cost_s"]):
        log(f"    short list: mirror {c['mirror']!s:5} ({c['p'][0]:+6.1f}, {c['p'][1]:+6.1f}) mm: as placed "
            f"{c['cost0']:.4g}, quick {c['cost_q']:.4g}, short {c['cost_s']:.4g}")
    ori = oris[best["mirror"]]
    curves = []
    for c in range(Nc):
        cur = {"ratio_obs": float(ori.ratio[c])}
        for key, p in (("chosen", best["p"]), ("true", p_true)):
            p3 = np.array([p[0], 0.0, p[1]])
            cloud = np.vstack([ori.XA - p3, ori.XB - p3])
            q = ori.q[c] - p3
            s, f, s_all, rat_all = ratio_curve(cm, ori.R[c], q, cloud, tube, ori.ratio[c], ori.s_low[c])
            s_lib, f_lib = cw._distance_from_ratio(ori.R[c], q, cloud, tube, ori.ratio[c], ori.s_low[c], 150.0, 6000.0)
            if abs(s - s_lib) > 1e-9 * s or abs(f - f_lib) > 1e-9 * f:
                raise AssertionError("the replayed ratio curve differs from cylinder_wand._distance_from_ratio")
            Cc = q - s * ori.R[c][2]
            # the true camera's distance along this principal ray: where its centre's distance from the axis is D
            cur[key] = {"s": rl(s_all, 2), "ratio": [None if not np.isfinite(v) else round(float(v), 5) for v in rat_all],
                        "s_best": s, "f": f, "D_axis": float(math.hypot(Cc[0], Cc[2]))}
            log(f"    C{c + 1} at the {key} axis position: stretch ratio {ori.ratio[c]:.4f} -> {s:.1f} mm along its "
                f"principal ray, {cur[key]['D_axis']:.1f} mm from the axis (true {rigs.CAM1_D}), f {f:.0f} px (true "
                f"{rigs.CAM1_F})")
        curves.append(cur)
    out["ratio_curves"] = curves

    # ---- the whole calibration from scratch
    t0 = time.time()
    sol = cw.calibrate(data, tube)
    secs = time.time() - t0
    f_err = [(sol.cams[c].f - truth[c].f) / truth[c].f * 100.0 for c in range(Nc)]
    d_err = [math.hypot(sol.cams[c].C[0], sol.cams[c].C[2]) - math.hypot(truth[c].C[0], truth[c].C[2])
             for c in range(Nc)]
    out["calibration"] = {"seconds": secs, "rms_px": float(sol.rms), "f_err_pct": f_err, "D_err_mm": d_err,
                          "stage3_rms_px": sol.info["starts"]["stage3_rms_px"], "starts_tried": sol.info["starts_tried"],
                          "params": int(sol.layout.P), "param_names": list(sol.layout.names),
                          "active": int(sol.obs.active.sum()), "detections": int(len(sol.obs))}
    log(f"  calibrate from scratch: {secs:.1f} s, RMS {sol.rms:.4f} px, f errors "
        + ", ".join(f"{e:+.3f} %" for e in f_err) + ", distances to the axis " + ", ".join(f"{e:+.3f} mm" for e in d_err)
        + f"; {sol.layout.P} rig parameters, {int(sol.obs.active.sum())}/{len(sol.obs)} detections kept")
    return out


# ------------------------------------------------------------------ the page
def build_page(data, fragment=None):
    with open(TEMPLATE, encoding="utf-8") as fh:
        tpl = fh.read()
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    tpl = tpl.replace("__GUIDE_DATA__", blob)
    head, body = tpl.split("<!--BODY-->", 1)
    page = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            + head.strip() + "\n</head>\n<body>\n" + body.strip() + "\n</body>\n</html>\n")
    with open(PAGE, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    if fragment:
        with open(fragment, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(head.strip() + "\n" + body.strip() + "\n")
    return PAGE


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--page-only", action="store_true", help="rebuild the page from guide_data.json")
    ap.add_argument("--fragment", help="also write the page without its document skeleton here")
    a = ap.parse_args()
    if not a.page_only:
        log = Log(LOG_PATH)
        t0 = time.time()
        log(f"make_guide.py, {time.strftime('%Y-%m-%d %H:%M')}")
        data = {"built": time.strftime("%Y-%m-%d"), "plate": plate_data(log), "tube": tube_data(log)}
        with open(DATA_PATH, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, separators=(",", ":"), allow_nan=False)
        log(f"wrote {DATA_PATH} ({os.path.getsize(DATA_PATH) / 1024:.0f} kB) in {time.time() - t0:.0f} s")
        log.close()
    with open(DATA_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    if os.path.exists(TEMPLATE):
        print("wrote", build_page(data, a.fragment))
    else:
        print("no template yet:", TEMPLATE)


if __name__ == "__main__":
    main()
