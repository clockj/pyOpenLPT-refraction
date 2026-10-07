"""Files for the cylinder wand calibration (`cylinder_wand.py`): the rig configuration (JSON), PINCYL camFiles, the
result JSON and a short text report.

The camFiles are written by pyOpenLPT's own C++ writer (`saveParameters`, 17 significant digits), so they load with
`pyopenlpt.Camera(path)` exactly as calibrated; only the two error lines of the header are filled in afterwards,
as the existing wand calibrator does: "Camera Calibration Error" = mean,std of the camera's reprojection error (px,
Euclidean, active observations) and "Pose Calibration Error" = mean,std of the wand-length error of a free
triangulation (mm, all frames).

The report also says what the noise weighting did (`noise_summary`): each camera's (or camera and wand end's)
estimated noise level and how well it is determined, the detections used and not used, the residuals, whether the
weights were applied and why not if they were not, and how many bootstrap starts were tried.

Frame of the outputs: the tube frame (axis = Y through the origin, Z towards camera 1), with the origin slid along
the axis to the wand's mean height (`cylinder_wand.relabel`); the camera files carry that frame's axis
(center (0, 0, 0), axis (0, 1, 0)).  The sigmas are those of the fit's gauge (camera 1 at X = Y = 0): the slide
along the axis is a relabelling and changes no residual.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

try:
    from . import cyl_model as cm
    from . import cylinder_wand as cw
except ImportError:  # imported as a top-level module (tests put this folder on sys.path)
    import cyl_model as cm
    import cylinder_wand as cw


# ------------------------------------------------------------------ rig configuration
@dataclass
class RigConfig:
    radii: Sequence[float]                 # interface radii, outermost first (mm)
    n: Sequence[float]                     # refractive indices, the pinhole's medium first
    wand_length_mm: float
    cameras: List[Dict]                    # [{"name": "cam1", "width": 1280, "height": 800, "pixel_mm": 0.02}, ...]
    hints: Dict = field(default_factory=dict)

    @property
    def tube(self) -> cm.Tube:
        return cm.Tube(tuple(self.radii), tuple(self.n))

    @property
    def sizes(self):
        return [(int(c["width"]), int(c["height"])) for c in self.cameras]


def load_config(path: str) -> RigConfig:
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    cfg = RigConfig(radii=d["tube"]["radii"], n=d["tube"]["n"], wand_length_mm=float(d["wand_length_mm"]),
                    cameras=list(d["cameras"]), hints=dict(d.get("hints", {})))
    cfg.tube                                       # validates the tube
    return cfg


def save_config(cfg: RigConfig, path: str) -> None:
    d = {"tube": {"radii": list(cfg.radii), "n": list(cfg.n)}, "wand_length_mm": cfg.wand_length_mm,
         "cameras": cfg.cameras, "hints": cfg.hints}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1)


# ------------------------------------------------------------------ camFiles
def write_camfile(cam: cm.Camera, tube: cm.Tube, path: str, proj_err=(None, None), pose_err=(None, None)) -> None:
    """A PINCYL camFile through the C++ writer, with the two error lines filled in (mean,std)."""
    cm.to_cpp(cam, tube).saveParameters(path)
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.read().split("\n")

    def fill(header, vals):
        if vals[0] is None:
            return
        for k, ln in enumerate(lines[:-1]):
            if ln.strip().startswith(header):
                lines[k + 1] = f"{float(vals[0]):.8g},{float(vals[1]):.8g}"
                return
    fill("# Camera Calibration Error", proj_err)
    fill("# Pose Calibration Error", pose_err)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ------------------------------------------------------------------ result
def statistics(sol: cw.Solution) -> Dict:
    """Per camera: reprojection error of the active observations (RMS, mean and std of the 2-D norm, max, count);
    the wand length of a free triangulation of every frame; the predicted error on unseen points."""
    obs = sol.obs
    ev = cw.evaluate(sol.cams, sol.tube, sol.frames, sol.L, obs)
    rn = np.linalg.norm(ev.r, axis=1)
    per_cam = []
    for c in range(len(sol.cams)):
        sel = obs.active & ev.ok & (obs.cam == c)
        v = rn[sel]
        per_cam.append({"n_active": int(sel.sum()), "n_total": int(np.sum(obs.cam == c)),
                        "rms_px": float(np.sqrt(np.mean(v ** 2))) if v.size else float("nan"),
                        "mean_px": float(v.mean()) if v.size else float("nan"),
                        "std_px": float(v.std()) if v.size else float("nan"),
                        "max_px": float(v.max()) if v.size else float("nan")})
    act_frames = np.zeros(obs.n_frames, bool)
    act_frames[obs.frame[obs.active]] = True
    Lm = cw.wand_lengths(sol.cams, sol.tube, obs)
    err = (Lm - sol.L)[act_frames & np.isfinite(Lm)]
    wand = {"frames": int(err.size), "mean_mm": float(err.mean()) if err.size else float("nan"),
            "std_mm": float(err.std()) if err.size else float("nan"),
            "rms_mm": float(np.sqrt(np.mean(err ** 2))) if err.size else float("nan"),
            "abs_mean_mm": float(np.abs(err).mean()) if err.size else float("nan")}
    return {"per_camera": per_cam, "wand_length_free_triangulation": wand}


def noise_summary(sol: cw.Solution, names: Sequence[str] = None) -> Dict:
    """What the noise weighting did (cylinder_wand.refine, tests/cylinder_wand T12), for the report: per noise group
    (camera, or camera and wand end) the estimated noise sigma (px per coordinate) with its relative standard error,
    the detections used and not used (rejected, or dropped with their frame), the residual RMS (px, 2-D) and the
    scale outliers were judged against; whether the weights were applied, and if not, why; the pooled sigma, the
    variance factor and the number of rounds; and the bootstrap starts tried."""
    nw = sol.info.get("noise_weights") or {}
    obs = sol.obs
    n = len(sol.cams)
    names = list(names) if names is not None else [f"cam{c + 1}" for c in range(n)]
    mode = nw.get("mode", "none")
    given = bool(nw.get("given"))
    gmode = "camera" if given else mode
    groups = cw.noise_groups(obs, gmode if gmode in cw.NOISE_MODES else "none")
    if gmode == "camera":
        labels = list(names)
    elif gmode == "camera_end":
        labels = [f"{names[g // 2]} end {'AB'[g % 2]}" for g in range(2 * n)]
    else:
        labels = ["all cameras"]
    used = bool(nw.get("used"))
    if given:
        why = "given by the user (sigma_cam_px); not estimated"
    elif mode == "none":
        why = "equal weights requested (noise_weights='none')"
    elif used:
        why = "estimated from the residuals; each detection weighted by 1 / sigma^2 of its group"
    elif nw.get("flipped"):
        fl = nw["flipped"]
        states = " px and ".join(", ".join(f"{v:.3f}" for v in s) for s in fl["sigma_px"])
        if fl.get("kind", "alternates") == "alternates":
            why = ("the estimated noise flipped between two states every round instead of converging (from round "
                   f"{fl['round']}: {states} px per group): equal weights, as without the weighting")
        else:
            why = (f"the estimated noise did not converge: it moved by more than {100 * cw.FLIP_JUMP:.0f} % in each "
                   f"of {cw.FLIP_RUN} rounds in a row (to round {fl['round']}; the last two: {states} px per group): "
                   "equal weights, as without the weighting")
    else:
        rs = [x for x in (nw.get("rel_se_group") or []) if x is not None]
        why = ("the data could not separate the groups' noise (largest relative standard error "
               + (f"{100 * max(rs):.1f} %" if rs else "unknown")
               + f", limit {100 * nw.get('max_rel_se', 0.10):.0f} %): equal weights, as without the weighting")
    ev = cw.evaluate(sol.cams, sol.tube, sol.frames, sol.L, obs)
    rn = np.linalg.norm(np.where(ev.ok[:, None], ev.r, np.nan), axis=1)
    sig_used = nw.get("sigma_group_px")
    sig_sep = nw.get("sigma_px_separate")
    rel_se = nw.get("rel_se_group")
    robust = sol.info.get("robust_sigma_group_px") or {}
    rows = []
    for g, lab in enumerate(labels):
        sel = groups == g
        act = sel & obs.active & ev.ok
        v = rn[act]
        rows.append({"group": lab,
                     "sigma_px": (sig_used[g] if used and sig_used and g < len(sig_used) else None),
                     "sigma_estimate_px": (sig_sep[g] if sig_sep and g < len(sig_sep) else None),
                     "rel_se": (rel_se[g] if rel_se and g < len(rel_se) else None),
                     "n_total": int(sel.sum()), "n_used": int(act.sum()), "n_not_used": int(sel.sum() - act.sum()),
                     "residual_rms_px": float(np.sqrt(np.mean(v ** 2))) if v.size else None,
                     "residual_rms_px_per_coordinate": float(np.sqrt(np.mean(v ** 2) / 2.0)) if v.size else None,
                     "rejection_scale_px": robust.get(g) if used else sol.info.get("robust_sigma_px")})
    st = sol.info.get("starts") or {}
    sig_list = [r["sigma_px"] for r in rows if r["sigma_px"]]
    return {"mode": mode, "given": given, "weights_used": used, "why": why, "groups": rows,
            "pooled_sigma_px": sol.sigma_px, "variance_factor": sol.info.get("variance_factor"),
            "rounds": sol.info.get("rounds"), "cycle": sol.info.get("cycle"),
            "noisiest_over_quietest": (max(sig_list) / min(sig_list)) if len(sig_list) >= 2 else None,
            "max_rel_se": nw.get("max_rel_se"),
            "starts": {"short_listed": st.get("short_listed"), "tried": st.get("tried"),
                       "extra_starts": st.get("extra_starts"), "more_starts_px": st.get("more_starts_px"),
                       "stage3_rms_px": st.get("stage3_rms_px"), "start_given": st.get("start_given"),
                       "stage4_shared": st.get("stage4_shared")}}


def write_result(sol: cw.Solution, folder: str, names: Sequence[str] = None, pixel_mm: Sequence[float] = None,
                 extra: Dict = None) -> Dict:
    """camFile/<name>.txt for every camera, calibration_result.json and calibration_report.txt in `folder`.
    Returns the result dictionary."""
    os.makedirs(os.path.join(folder, "camFile"), exist_ok=True)
    n = len(sol.cams)
    names = list(names) if names is not None else [f"cam{c + 1}" for c in range(n)]
    pixel_mm = list(pixel_mm) if pixel_mm is not None else [None] * n
    stats = statistics(sol)
    cams_out, frames_out, y0 = cw.relabel(sol.cams, sol.frames)
    sig = cw.camera_sigmas(sol.cams, sol.layout, sol.cov) if sol.cov is not None else [None] * n
    wand = stats["wand_length_free_triangulation"]
    cams_json = []
    for c, cam in enumerate(cams_out):
        pc = stats["per_camera"][c]
        path = os.path.join(folder, "camFile", f"{names[c]}.txt")
        write_camfile(cam, sol.tube, path, (pc["mean_px"], pc["std_px"]), (wand["abs_mean_mm"], wand["std_mm"]))
        D = math.hypot(cam.C[0], cam.C[2])
        cams_json.append({
            "name": names[c], "camfile": os.path.relpath(path, folder), "width": cam.width, "height": cam.height,
            "f_px": cam.f, "f_mm": cam.f * pixel_mm[c] if pixel_mm[c] else None, "cx": cam.cx, "cy": cam.cy,
            "dist": cam.dist.tolist(), "R": cam.R.tolist(), "rvec": cam.rvec.tolist(), "tvec": cam.tvec.tolist(),
            "centre_mm": cam.C.tolist(), "distance_from_axis_mm": D,
            "azimuth_deg": math.degrees(math.atan2(cam.C[0], cam.C[2])), "height_mm": float(cam.C[1]),
            "viewing_direction": cam.R[2].tolist(), "sigma": sig[c], "residuals": pc})
    result = {
        "model": "PINCYL", "frame": "tube frame: axis = Y through the origin, Z towards camera 1, origin at the "
                                    "wand's mean height", "origin_shift_along_axis_mm": y0,
        "tube": {"radii_mm": list(sol.tube.radii), "n": list(sol.tube.n), "fold_radius_mm": sol.tube.fold_radius()},
        "wand_length_mm": sol.L, "fit_L": bool(sol.layout.fit_L), "fit_k1": bool(sol.layout.fit_k1),
        "gauge": "camera 1's centre at X = 0, Y = 0 during the fit (sigmas are in that gauge)",
        "rms_px": sol.rms, "sigma_px": sol.sigma_px, "frames": int(sol.obs.n_frames),
        "observations": int(len(sol.obs)), "active": int(sol.obs.active.sum()), "cameras": cams_json,
        "wand_length_free_triangulation": wand, "noise": noise_summary(sol, names),
        "info": {k: v for k, v in sol.info.items() if _jsonable(v)}}
    if sol.layout.fit_L and sol.cov is not None:
        result["L_sigma_mm"] = math.sqrt(sol.cov[sol.layout.L_index, sol.layout.L_index])
    if extra:
        result.update(extra)
    with open(os.path.join(folder, "calibration_result.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    with open(os.path.join(folder, "calibration_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(report_text(result))
    return result


def _jsonable(v) -> bool:
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


def report_text(res: Dict) -> str:
    out = [f"Cylinder wand calibration (PINCYL), {len(res['cameras'])} cameras",
           f"tube: radii {res['tube']['radii_mm']} mm, n {res['tube']['n']}; wand {res['wand_length_mm']:.4f} mm"
           + (f" (fitted, sigma {res['L_sigma_mm']:.4f} mm)" if 'L_sigma_mm' in res else ""),
           f"frames {res['frames']}, observations {res['active']} active of {res['observations']}; "
           f"reprojection RMS {res['rms_px']:.4f} px (2-D), sigma {res['sigma_px']:.4f} px per coordinate",
           f"wand length, free triangulation: mean error {res['wand_length_free_triangulation']['mean_mm'] * 1e3:+.2f}"
           f" um, RMS {res['wand_length_free_triangulation']['rms_mm'] * 1e3:.2f} um over "
           f"{res['wand_length_free_triangulation']['frames']} frames",
           f"frame: {res['frame']} (slid {res['origin_shift_along_axis_mm']:+.3f} mm)", ""]
    for c in res["cameras"]:
        s = c["sigma"] or {}
        out.append(f"{c['name']}: f {c['f_px']:.1f} px"
                   + (f" ({c['f_mm']:.3f} mm)" if c["f_mm"] else "")
                   + (f" +- {100 * s['f_rel']:.3f} %" if s else "")
                   + f"; distance from the axis {c['distance_from_axis_mm']:.2f} mm"
                   + (f" +- {s['distance_mm']:.3f}" if s else "")
                   + f"; azimuth {c['azimuth_deg']:.3f} deg; height {c['height_mm']:.3f} mm; reprojection RMS "
                   f"{c['residuals']['rms_px']:.4f} px ({c['residuals']['n_active']}/{c['residuals']['n_total']} "
                   f"active); camFile {c['camfile']}")
    if res.get("noise"):
        out += [""] + noise_text(res["noise"])
    return "\n".join(out) + "\n"


def noise_text(nz: Dict) -> List[str]:
    """The report's section on the noise levels and weights (see `noise_summary`)."""
    w = max([14] + [len(r["group"]) + 1 for r in nz["groups"]])
    out = ["Noise levels and weights (noise_weights=" + repr(nz["mode"]) + ("; given by the user" if nz["given"] else "")
           + ")",
           f"  weights {'APPLIED' if nz['weights_used'] else 'NOT applied'}: {nz['why']}",
           f"  {'group':<{w}s} {'noise sigma':>11s} {'+- rel. SE':>11s} {'used / total':>15s} {'not used':>9s} "
           f"{'residual RMS':>13s} {'outlier scale':>14s}",
           f"  {'':<{w}s} {'(px)':>11s} {'':>11s} {'':>15s} {'':>9s} {'(px/coord.)':>13s} {'(px)':>14s}"]
    for r in nz["groups"]:
        sig = r["sigma_px"] if r["sigma_px"] is not None else r["sigma_estimate_px"]
        res = r.get("residual_rms_px_per_coordinate")
        out.append(f"  {r['group']:<{w}s} "
                   + (f"{sig:>11.4f} " if sig is not None else f"{'-':>11s} ")
                   + (f"{100 * r['rel_se']:>9.1f} % " if r["rel_se"] is not None else f"{'-':>11s} ")
                   + f"{r['n_used']:>7d} / {r['n_total']:<5d} {r['n_not_used']:>9d} "
                   + (f"{res:>13.4f} " if res is not None else f"{'-':>13s} ")
                   + (f"{r['rejection_scale_px']:>14.4f}" if r["rejection_scale_px"] is not None else f"{'-':>14s}")
                   + ("   (estimate, not used)" if r["sigma_px"] is None and sig is not None else ""))
    vf = nz.get("variance_factor")
    out.append(f"  pooled sigma {nz['pooled_sigma_px']:.4f} px per coordinate (unweighted)"
               + (f"; variance factor {vf:.3f} (about 1 when the weights match the data)" if vf is not None else "")
               + f"; {nz['rounds']} rounds"
               + (f"; noisiest / quietest group {nz['noisiest_over_quietest']:.2f}x"
                  if nz.get("noisiest_over_quietest") else ""))
    cy = nz.get("cycle")
    if cy:
        out.append(f"  the rounds stopped at round {cy['round']}: the active set repeated round "
                   f"{cy['repeats_round']}'s, and the {cy['left_out']} detections that alternated (rejected in the "
                   "fit, re-admitted after it) are left out")
    out += ["  noise sigma: each group's own detection noise, estimated with what the fit absorbs accounted for.",
            "  residual RMS: what is left after the fit, usually a little below sigma; with equal weights a quiet camera",
            "    next to a noisy one also shows part of the noisy camera's error (the cameras share the wand positions).",
            "  not used: rejected as outliers, or dropped with their frame when an end was left in fewer than two cameras.",
            "  outlier scale: the robust scale each detection was judged against (rejected beyond 5 of it)."]
    st = nz.get("starts") or {}
    if st.get("start_given"):
        out.append("  starts: the calibration was started from given cameras (no bootstrap)")
    elif st.get("short_listed"):
        s3 = st.get("stage3_rms_px") or []
        out.append(f"  starts: {st['tried']} of {st['short_listed']} short-listed bootstrap starts carried through stages 3-4"
                   + (f" (extra starts: the best stage-3 RMS of the first ones exceeded {st['more_starts_px']:g} px)"
                      if st.get("extra_starts") else "")
                   + (f"; stage-3 RMS {', '.join(f'{x:.3f}' for x in s3)} px" if s3 else "")
                   + (f"; {st['stage4_shared']} of them ended stage 3 as an earlier one (the same detections, RMS "
                      "within 1e-5) and share its stage 4" if st.get("stage4_shared") else ""))
    return out
