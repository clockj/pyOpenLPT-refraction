"""Files for the cylinder wand calibration (`cylinder_wand.py`): the rig configuration (JSON), PINCYL camFiles, the
result JSON and a short text report.

The camFiles are written by pyOpenLPT's own C++ writer (`saveParameters`, 17 significant digits), so they load with
`pyopenlpt.Camera(path)` exactly as calibrated; only the two error lines of the header are filled in afterwards,
as the existing wand calibrator does: "Camera Calibration Error" = mean,std of the camera's reprojection error (px,
Euclidean, active observations) and "Pose Calibration Error" = mean,std of the wand-length error of a free
triangulation (mm, all frames).

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
        "wand_length_free_triangulation": wand,
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
    return "\n".join(out) + "\n"
