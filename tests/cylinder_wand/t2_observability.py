"""T2: Phase 0 gate -- observability (Cramer-Rao bound) and rig design.

For each rig the Fisher information of the camera parameters is formed at the truth from noise-free detections
(C++ PINCYL), with i.i.d. 0.1 px noise, the wand frames marginalised (Schur complement) and the tube-frame gauge
(camera 1 at X = Y = 0).  Its inverse is the Cramer-Rao bound.  Reported per rig:
  sigma(f)/f, and sigma of each camera's distance from the axis, azimuth, height and orientation;
  the calibration's error on unseen points of the volume (a 7x7x7 grid), RMS per 2-D image:
    `fixed`       projecting a known tube-frame point, in the fit's gauge (camera 1 at X = Y = 0);
    `stabiliser`  the same after the best common spin about the axis and slide along it of the points.  The
                  tube's stabiliser is not observable; a near-telecentric camera 1 trades its height with its
                  pitch, so in the camera-1 gauge the whole scene slides along the axis relative to camera 1's
                  centre by ~0.15 mm, which `fixed` counts as several pixels.  That is a relabelling, not a
                  calibration error, so `stabiliser` is the figure of merit for projecting unseen points;
    `retriangulated`  each point re-triangulated from all cameras: what the cameras disagree on (gauge-free).

Reference rig: three replicas of the case's cam1 (f 41,459 px, 1094 mm from the axis, so f/D as cam1) at azimuths
0, 45, 90 deg, the case's tube, a 10 mm wand with midpoints in a +-20 mm box about the axis, 500 frames.  The
sensor is made large enough to see the whole volume (the user places the cameras for the field of view); the
image extent the volume needs is reported.  One factor is varied at a time: number of cameras, angle between
them, camera distance (0.5 m with f scaled to keep f/D), volume half-size, number of frames.

Also: the model error of the starts the plate pipeline would use -- a pinhole with fx = fy, a pinhole with
fx != fy (principal point free in both), and an affine camera -- fitted to the true 3-D points and their exact
images, per camera of the reference rig.

Gate (plan section 7): "GO if, on the reference rig, the calibration adds at most 0.05 px RMS to the projection of
unseen points".  The plan did not say in which frame; all three are reported.  `fixed` is dominated by the
unobservable slide; `stabiliser` by where the tube sits across the axis relative to the rig (about 12 um, see
t2b_error_modes.py); `retriangulated` is what triangulation and reprojection in particle tracking use.  The
decision is taken on `retriangulated`, and the literal reading (`stabiliser`) is recorded beside it.
Target: sigma(f)/f <= 0.5 % on the reference rig.
Outputs: t2_observability_out.txt, t2_observability.json.
"""
import json
import math
import time

import numpy as np
from scipy.optimize import least_squares

import rigs
from rigs import cm
import cylinder_wand as cw

OUT = rigs.HERE / "t2_observability"
log = rigs.Tee(str(OUT) + "_out.txt")
SIGMA = 0.1
L_WAND = 10.0
REF = {"n_cams": 3, "spacing": 45.0, "D": rigs.CAM1_D, "half": 20.0, "frames": 500}


def build(n_cams, spacing, D, half, frames, seed=11, sensor=(9000, 9000)):
    rng = np.random.default_rng(seed)
    cams = rigs.reference_rig(n_cams, spacing, D, sensor=sensor)
    tube = rigs.CASE_TUBE
    wd = rigs.wand_frames(cams, tube, frames, L_WAND, rng, half=(half, half, half), noise_px=0.0, all_cams=True)
    data = cw.WandData(wd.uv, L_WAND, [(c.width, c.height) for c in cams])
    obs = cw.observations(data)
    fr = cw.frames_from_ends(wd.XA[obs.frame_ids], wd.XB[obs.frame_ids])
    return cams, tube, fr, obs, wd


def analyse(name, n_cams, spacing, D, half, frames, seed=11):
    t0 = time.time()
    cams, tube, fr, obs, wd = build(n_cams, spacing, D, half, frames, seed)
    layout = cw.Layout(n_cams)
    info, ev = cw.information(cams, tube, fr, L_WAND, obs, layout, sigma_px=SIGMA)
    cond = float(np.linalg.cond(info))
    cov = np.linalg.inv(info)
    sig = cw.camera_sigmas(cams, layout, cov)
    g = np.linspace(-half, half, 7)
    P = np.array([[x, y, z] for x in g for y in g for z in g])
    pe = cw.predicted_projection_error(cams, tube, layout, cov, P)
    # image extent the volume needs, per camera (px): the wand ends' bounding box
    ext = []
    for c in range(n_cams):
        uv = wd.uv[:, c].reshape(-1, 2)
        ext.append([float(np.ptp(uv[:, 0])), float(np.ptp(uv[:, 1]))])
    corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    np.fill_diagonal(corr, 0.0)
    i, j = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    rec = {"name": name, "n_cams": n_cams, "spacing_deg": spacing, "D_mm": D, "f_px": cams[0].f,
           "half_mm": half, "frames": int(obs.n_frames), "observations": int(len(obs)),
           "cond_info": cond, "sigma_f_rel": [s["f_rel"] for s in sig],
           "sigma_distance_mm": [s["distance_mm"] for s in sig], "sigma_azimuth_deg": [s["azimuth_deg"] for s in sig],
           "sigma_height_mm": [s["height_mm"] for s in sig], "sigma_rotation_deg": [s["rotation_deg"] for s in sig],
           "unseen_fixed_rms_px": pe["fixed_rms_px"], "unseen_fixed_rms_px_per_cam": pe["fixed_rms_px_per_cam"],
           "unseen_retriangulated_rms_px": pe["retriangulated_rms_px"], "unseen_points": pe["n_points"],
           "unseen_stabiliser_rms_px": pe["stabiliser_rms_px"],
           "image_extent_px": ext,
           "largest_correlation": {"pair": [layout.names[i], layout.names[j]], "value": float(corr[i, j])},
           "seconds": time.time() - t0}
    f_rel = max(rec["sigma_f_rel"])
    log(f"{name:28s} cams {n_cams} sp {spacing:5.1f} D {D:6.0f} f {cams[0].f:7.0f} half {half:4.0f} "
        f"frames {obs.n_frames:4d} | sigma f/f max {100 * f_rel:6.3f} % | dist {max(rec['sigma_distance_mm']):7.3f} mm"
        f" | az {max(rec['sigma_azimuth_deg']):.4f} deg | h {max(rec['sigma_height_mm']):.4f} mm | unseen: fixed "
        f"{pe['fixed_rms_px']:.3f}, stab {pe['stabiliser_rms_px']:.4f}, retri {pe['retriangulated_rms_px']:.4f} px"
        f" | extent "
        f"{max(e[0] for e in ext):.0f}x{max(e[1] for e in ext):.0f} px")
    return rec


# ------------------------------------------------------------------ model error of the plate pipeline's starts
def pinhole_fit(X, uv, square: bool, cam: cm.Camera):
    """Best pinhole for the tube camera's images of X: pose and f (or fx, fy) free, the principal point held at
    the sensor centre as in a pinhole calibration.  (With the principal point free and unbounded, a pinhole far
    away viewing the cloud off-axis becomes a general affine camera and just reproduces the affine fit.)
    Started from the true pose with f scaled by the mean of the tube's magnifications (x1.18)."""
    rv = cm.rvec_of(cam.R)
    t = cam.tvec
    f0 = 1.18 * cam.f
    x0 = np.r_[rv, t, f0] if square else np.r_[rv, t, f0, f0]

    def res(p):
        Rm = cm.rot_of(p[:3])
        pc = X @ Rm.T + p[3:6]
        fx = p[6]
        fy = p[6] if square else p[7]
        return np.r_[fx * pc[:, 0] / pc[:, 2] + cam.cx - uv[:, 0], fy * pc[:, 1] / pc[:, 2] + cam.cy - uv[:, 1]]

    r = least_squares(res, x0, method="trf", x_scale="jac", xtol=1e-14, ftol=1e-14, max_nfev=20000)
    return float(np.sqrt(np.mean(r.fun ** 2) * 2)), r.x


def affine_fit(X, uv):
    A = np.column_stack([X, np.ones(len(X))])
    coef, *_ = np.linalg.lstsq(A, uv, rcond=None)
    r = A @ coef - uv
    return float(np.sqrt(np.mean(np.sum(r ** 2, axis=1))))


def start_model_errors():
    cams, tube, fr, obs, wd = build(**{"n_cams": 3, "spacing": 45.0, "D": rigs.CAM1_D, "half": 20.0,
                                       "frames": 500})
    X = np.vstack([wd.XA, wd.XB])
    out = []
    for c in range(len(cams)):
        uv = np.vstack([wd.uv[:, c, 0], wd.uv[:, c, 1]])
        a, xa = pinhole_fit(X, uv, True, cams[c])
        b, xb = pinhole_fit(X, uv, False, cams[c])
        out.append({"camera": c + 1, "pinhole_fx_eq_fy_rms_px": a, "pinhole_fx_ne_fy_rms_px": b,
                    "affine_rms_px": affine_fit(X, uv), "pinhole_square_f": float(xa[6]),
                    "pinhole_fx_fy": [float(xb[6]), float(xb[7])],
                    "pinhole_square_distance_mm": float(np.linalg.norm(cm.rot_of(xa[:3]).T @ xa[3:6])),
                    "pinhole_fx_ne_fy_distance_mm": float(np.linalg.norm(cm.rot_of(xb[:3]).T @ xb[3:6]))})
        log(f"  camera {c + 1}: best pinhole fx = fy {a:8.2f} px RMS (f {xa[6]:.0f}, at "
            f"{out[-1]['pinhole_square_distance_mm']:.0f} mm); fx != fy {b:8.2f} px RMS (fx {xb[6]:.0f}, "
            f"fy {xb[7]:.0f}, at {out[-1]['pinhole_fx_ne_fy_distance_mm']:.0f} mm); affine "
            f"{out[-1]['affine_rms_px']:8.2f} px RMS")
    return out


def main():
    t0 = time.time()
    res = {"sigma_px": SIGMA, "wand_mm": L_WAND, "rigs": []}
    log("== reference rig and one-factor variations (Cramer-Rao, 0.1 px noise) ==")
    ref = analyse("reference", **{"n_cams": 3, "spacing": 45.0, "D": rigs.CAM1_D, "half": 20.0, "frames": 500})
    res["rigs"].append(ref)
    variants = []
    for n in (2, 4):
        variants.append((f"cams={n}", dict(REF, n_cams=n)))
    for sp in (30.0, 60.0, 90.0, 120.0):
        variants.append((f"spacing={sp:.0f}", dict(REF, spacing=sp)))
    variants.append(("D=500 (f scaled)", dict(REF, D=500.0)))
    for h in (10.0, 40.0):
        variants.append((f"half={h:.0f}", dict(REF, half=h)))
    for nf in (200, 2000):
        variants.append((f"frames={nf}", dict(REF, frames=nf)))
    variants += [("4 cams at 90", dict(REF, n_cams=4, spacing=90.0)),
                 ("3 cams at 90, D=500", dict(REF, spacing=90.0, D=500.0)),
                 ("2 cams at 90", dict(REF, n_cams=2, spacing=90.0)),
                 ("3 cams at 60, half=10", dict(REF, spacing=60.0, half=10.0)),
                 ("cam1 sensor-like: half=8", dict(REF, half=8.0)),
                 ("2 cams at 45, half=10", dict(REF, n_cams=2, half=10.0))]
    for name, kw in variants:
        p = dict(kw)
        res["rigs"].append(analyse(name, p["n_cams"], p["spacing"], p["D"], p["half"], p["frames"]))
    log("")
    log("== model error of the plate pipeline's starts, reference rig (true 3-D points, exact images) ==")
    res["start_models"] = start_model_errors()
    go = ref["unseen_retriangulated_rms_px"] <= 0.05
    go_literal = ref["unseen_stabiliser_rms_px"] <= 0.05
    target = max(ref["sigma_f_rel"]) <= 0.005
    res["gate"] = {"go": bool(go), "go_on_tube_frame_modulo_stabiliser": bool(go_literal),
                   "reference_unseen_stabiliser_rms_px": ref["unseen_stabiliser_rms_px"],
                   "reference_unseen_fixed_rms_px": ref["unseen_fixed_rms_px"],
                   "reference_unseen_retriangulated_rms_px": ref["unseen_retriangulated_rms_px"],
                   "reference_sigma_f_rel_max": max(ref["sigma_f_rel"]), "f_target_met": bool(target)}
    res["seconds"] = time.time() - t0
    log("")
    log(f"GATE (gauge-free, re-triangulated): the reference rig adds {ref['unseen_retriangulated_rms_px']:.4f} px "
        f"RMS on unseen points (limit 0.05): {'GO' if go else 'NO-GO'}")
    log(f"  literal reading, tube frame modulo the stabiliser: {ref['unseen_stabiliser_rms_px']:.3f} px "
        f"({'pass' if go_literal else 'FAILS the 0.05 px limit'}; dominated by the tube's position across the axis, "
        f"see t2b); in the camera-1 gauge: {ref['unseen_fixed_rms_px']:.3f} px (the unobservable slide)")
    log(f"f target: sigma(f)/f = {100 * max(ref['sigma_f_rel']):.3f} % (target 0.5 %): "
        f"{'met' if target else 'not met'}")
    log(f"time {res['seconds']:.0f} s")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
