"""Files for the flat-window wand calibration (`plate_wand.py`): the GUI's wand CSV, the rig configuration (JSON),
PINPLATE camFiles, the result JSON and a short text report.

**The wand CSV** is the GUI's (wand_calibrator.export_wand_data): header Frame, Camera, Status, PointIdx, X, Y, Radius,
Metric; the rows whose Status starts with "Filtered" are the detections, Filtered_Small = end A (0), Filtered_Large =
end B (1), at most two per camera and frame.  `read_gui_csv` returns them as uv[frame, camera, end, (u, v)] with NaN
where not seen.

**The camFiles** are written by pyOpenLPT's own C++ writer (`saveParameters`, 17 significant digits), so they load with
`pyopenlpt.Camera(path)` exactly as calibrated (tests/plate_wand/w9_files.py); only the two error lines of the header
are filled in afterwards, as the existing wand calibrator does: "Camera Calibration Error" = mean,std of the camera's
reprojection error (px, Euclidean, active observations) and "Pose Calibration Error" = mean,std of the wand-length
error of a free triangulation (mm).

Frame of the outputs: plate_wand.to_output_frame's (Z along window 1's normal, camera -> water; Y along n1 x n2 with two
windows, else camera 1's image up in the window's plane; the origin at the wand's mean position).  The sigmas are those
of the fit's gauge (camera 1's pose held).
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from . import cyl_model as cm
    from . import plate_model as pm
    from . import plate_wand as pw
except ImportError:  # imported as a top-level module (tests put this folder on sys.path)
    import cyl_model as cm
    import plate_model as pm
    import plate_wand as pw

CSV_HEADER = ["Frame", "Camera", "Status", "PointIdx", "X", "Y", "Radius", "Metric"]


# ------------------------------------------------------------------ the GUI's wand CSV
@dataclass
class CsvData:
    uv: np.ndarray                 # (F, Nc, 2, 2), NaN where not seen
    frame_ids: np.ndarray          # (F,) the CSV's frame numbers
    cam_ids: List[int]             # (Nc,) the CSV's camera numbers
    radii: np.ndarray              # (F, Nc, 2) blob radii (NaN where not seen)


def read_gui_csv(path: str, cam_ids: Sequence[int] = None) -> CsvData:
    """The Filtered detections of a GUI wand CSV.  cam_ids fixes the cameras and their order (default: every camera
    in the file, sorted)."""
    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header:
            raise ValueError(f"{path}: empty file")
        for row in reader:
            if len(row) < 8 or not row[2].startswith("Filtered"):
                continue
            try:
                f_idx, c_idx = int(row[0]), int(row[1])
                x, y, r = float(row[4]), float(row[5]), float(row[6])
            except ValueError:
                continue
            end = 0 if row[2] == "Filtered_Small" else 1 if row[2] == "Filtered_Large" else int(row[3])
            rows.append((f_idx, c_idx, end, x, y, r))
    if not rows:
        raise ValueError(f"{path}: no Filtered detections")
    frames = sorted({r[0] for r in rows})
    cams = sorted({r[1] for r in rows}) if cam_ids is None else list(cam_ids)
    fpos = {f: k for k, f in enumerate(frames)}
    cpos = {c: k for k, c in enumerate(cams)}
    uv = np.full((len(frames), len(cams), 2, 2), np.nan)
    rad = np.full((len(frames), len(cams), 2), np.nan)
    count = np.zeros((len(frames), len(cams)), int)
    for f_idx, c_idx, end, x, y, r in rows:
        if c_idx not in cpos:
            continue
        i, c = fpos[f_idx], cpos[c_idx]
        if count[i, c] >= 2 or not np.isnan(uv[i, c, end, 0]):        # the GUI keeps the first two per camera and frame
            continue
        uv[i, c, end] = (x, y)
        rad[i, c, end] = r
        count[i, c] += 1
    return CsvData(uv, np.array(frames), cams, rad)


def write_gui_csv(path: str, uv: np.ndarray, radii=(4.0, 6.0), frame_ids=None, cam_ids=None, digits: int = 6) -> int:
    """Detections uv[frame, camera, end, (u, v)] as a GUI wand CSV: each seen end as a "Raw" row and as
    Filtered_Small (end A) / Filtered_Large (end B); a camera that does not see both ends of a frame gets no rows for
    it.  Returns the number of Filtered rows."""
    uv = np.asarray(uv, float)
    F, Nc = uv.shape[:2]
    frame_ids = list(range(F)) if frame_ids is None else list(frame_ids)
    cam_ids = list(range(Nc)) if cam_ids is None else list(cam_ids)
    n = 0
    fmt = f"{{:.{digits}f}}"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        for i in range(F):
            for c in range(Nc):
                if not np.all(np.isfinite(uv[i, c])):
                    continue
                for e in (0, 1):
                    w.writerow([frame_ids[i], cam_ids[c], "Raw", e, fmt.format(uv[i, c, e, 0]), fmt.format(uv[i, c, e, 1]),
                                f"{radii[e]:.3f}", "1.0000"])
        for i in range(F):
            for c in range(Nc):
                if not np.all(np.isfinite(uv[i, c])):
                    continue
                for e, st in ((0, "Filtered_Small"), (1, "Filtered_Large")):
                    w.writerow([frame_ids[i], cam_ids[c], st, e, fmt.format(uv[i, c, e, 0]), fmt.format(uv[i, c, e, 1]),
                                f"{radii[e]:.3f}", "1.0000"])
                    n += 1
    return n


# ------------------------------------------------------------------ rig configuration
@dataclass
class RigConfig:
    """windows: [{"name": "front", "thickness_mm": [12.7], "n": [1.0, 1.49, 1.333]}] (indices camera side first);
    cam_to_window: per camera the window index (-1: none); cameras: [{"name", "width", "height", "pixel_mm"}];
    hints (optional): {"f": [px per camera or null], "f_prior_rel": sigma}."""

    windows: List[Dict]
    cam_to_window: List[int]
    cameras: List[Dict]
    wand_length_mm: float
    hints: Dict = field(default_factory=dict)

    def rig(self) -> pw.Rig:
        """The rig with placeholder planes (normal +Z, d = 0): calibrate() finds the planes."""
        wins = [pm.Window([0.0, 0.0, 1.0], 0.0, tuple(w["thickness_mm"]), tuple(w["n"]), w.get("name", ""))
                for w in self.windows]
        return pw.Rig(wins, list(self.cam_to_window))

    @property
    def sizes(self):
        return [(int(c["width"]), int(c["height"])) for c in self.cameras]

    @property
    def names(self):
        return [c.get("name", f"cam{k + 1}") for k, c in enumerate(self.cameras)]


def load_config(path: str) -> RigConfig:
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    cfg = RigConfig(windows=list(d["windows"]), cam_to_window=list(d["cam_to_window"]), cameras=list(d["cameras"]),
                    wand_length_mm=float(d["wand_length_mm"]), hints=dict(d.get("hints", {})))
    cfg.rig()                                   # validates the windows
    if len(cfg.cam_to_window) != len(cfg.cameras):
        raise ValueError("cam_to_window needs one entry per camera")
    return cfg


def save_config(cfg: RigConfig, path: str) -> None:
    d = {"windows": cfg.windows, "cam_to_window": cfg.cam_to_window, "cameras": cfg.cameras,
         "wand_length_mm": cfg.wand_length_mm, "hints": cfg.hints}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1)


# ------------------------------------------------------------------ camFiles
def write_camfile(cam: pm.Camera, win: Optional[pm.Window], path: str, proj_err=(None, None),
                  pose_err=(None, None)) -> None:
    """A PINPLATE camFile through the C++ writer, with the two error lines filled in (mean,std).  A camera without a
    window cannot be written as PINPLATE."""
    if win is None:
        raise ValueError("a camera without a window has no PINPLATE file")
    pm.to_cpp(cam, win).saveParameters(path)
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


def load_camfile(path: str) -> Tuple[pm.Camera, pm.Window]:
    """(Camera, Window) of a PINPLATE camFile, through the C++ loader."""
    lpt = pm.import_lpt()
    return pm.from_cpp(lpt.Camera(path))


# ------------------------------------------------------------------ result
def statistics(sol: pw.Solution) -> Dict:
    """Per camera: reprojection error of the active observations (RMS, mean and std of the 2-D norm, max, count);
    the wand length of a free triangulation of every active frame; rejected frames."""
    obs = sol.obs
    ev = pw.evaluate(sol.cams, sol.rig, sol.frames, sol.L, obs)
    rn = np.linalg.norm(np.where(ev.ok[:, None], ev.r, np.nan), axis=1)
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
    Lm = pw.wand_lengths(sol.cams, sol.rig, obs)
    err = (Lm - sol.L)[act_frames & np.isfinite(Lm)]
    wand = {"frames": int(err.size), "mean_mm": float(err.mean()) if err.size else float("nan"),
            "std_mm": float(err.std()) if err.size else float("nan"),
            "rms_mm": float(np.sqrt(np.mean(err ** 2))) if err.size else float("nan"),
            "abs_mean_mm": float(np.abs(err).mean()) if err.size else float("nan")}
    rejected = obs.frame_ids[~act_frames].tolist()
    partly = sorted(set(obs.frame_ids[obs.frame[~obs.active]].tolist()) - set(rejected))
    return {"per_camera": per_cam, "wand_length_free_triangulation": wand,
            "frames_dropped": rejected, "frames_with_rejected_detections": partly}


def noise_summary(sol: pw.Solution, names: Sequence[str] = None) -> Dict:
    """What the noise weighting did, as cylinder_wand_io.noise_summary (per camera: sigma, its relative standard error,
    detections used and not used, residual RMS, outlier scale; whether the weights were applied and why)."""
    nw = sol.info.get("noise_weights") or {}
    obs = sol.obs
    n = len(sol.cams)
    names = list(names) if names is not None else [f"cam{c + 1}" for c in range(n)]
    mode = nw.get("mode", "none")
    given = bool(nw.get("given"))
    gmode = "camera" if given else mode
    groups = pw.noise_groups(obs, gmode if gmode in pw.NOISE_MODES else "none")
    if gmode == "camera":
        labels = list(names)
    elif gmode == "camera_end":
        labels = [f"{names[g // 2]} end {'AB'[g % 2]}" for g in range(2 * n)]
    else:
        labels = ["all cameras"]
    used = bool(nw.get("used"))
    if given:
        why = "given by the user (sigma_cam_px); not estimated"
    elif nw.get("two_cameras"):
        why = "two cameras: their noise cannot be told apart reliably, so equal weights"
    elif nw.get("drifting"):
        why = "the estimated weights did not settle: equal weights"
    elif mode == "none":
        why = "equal weights requested (noise_weights='none')"
    elif used:
        why = "estimated from the residuals; each detection weighted by 1 / sigma^2 of its group"
    else:
        rs = [x for x in (nw.get("rel_se_group") or []) if x is not None]
        why = ("the data could not separate the groups' noise (largest relative standard error "
               + (f"{100 * max(rs):.1f} %" if rs else "unknown")
               + f", limit {100 * nw.get('max_rel_se', 0.10):.0f} %): equal weights")
    ev = pw.evaluate(sol.cams, sol.rig, sol.frames, sol.L, obs)
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
    sig_list = [r["sigma_px"] for r in rows if r["sigma_px"]]
    return {"mode": mode, "given": given, "weights_used": used, "why": why, "groups": rows,
            "pooled_sigma_px": sol.sigma_px, "variance_factor": sol.info.get("variance_factor"),
            "rounds": sol.info.get("rounds"),
            "noisiest_over_quietest": (max(sig_list) / min(sig_list)) if len(sig_list) >= 2 else None,
            "max_rel_se": nw.get("max_rel_se")}


def predicted_precision(sol: pw.Solution, n_points: int = 300) -> Dict:
    """Cramer-Rao precision on points spread like the wand positions (plate_wand.predicted_projection_error), and a
    held-out wand's length (the calibration's share, mm)."""
    if sol.cov is None:
        return {}
    act = np.zeros(sol.obs.n_frames, bool)
    act[sol.obs.frame[sol.obs.active]] = True
    XA, XB = sol.frames.ends(sol.L)
    P = np.vstack([XA[act], XB[act]])
    if len(P) > n_points:
        P = P[np.linspace(0, len(P) - 1, n_points).round().astype(int)]
    sizes = [(c.width, c.height) for c in sol.cams]
    pe = pw.predicted_projection_error(sol.cams, sol.rig, sol.layout, sol.cov, P, sizes)
    return {"points": pe}


def write_result(sol: pw.Solution, folder: str, names: Sequence[str] = None, pixel_mm: Sequence[float] = None,
                 extra: Dict = None) -> Dict:
    """camFile/<name>.txt for every camera, calibration_result.json and calibration_report.txt in `folder`.
    Returns the result dictionary."""
    os.makedirs(os.path.join(folder, "camFile"), exist_ok=True)
    n = len(sol.cams)
    names = list(names) if names is not None else [f"cam{c + 1}" for c in range(n)]
    pixel_mm = list(pixel_mm) if pixel_mm is not None else [None] * n
    stats = statistics(sol)
    sig = pw.camera_sigmas(sol.cams, sol.rig, sol.layout, sol.cov) if sol.cov is not None else [None] * n
    wsig = pw.window_sigmas(sol.cams, sol.rig, sol.layout, sol.cov) if sol.cov is not None else [None] * len(
        sol.rig.windows)
    wand = stats["wand_length_free_triangulation"]
    cams_json = []
    for c, cam in enumerate(sol.cams):
        pc = stats["per_camera"][c]
        w = sol.rig.cam_win[c]
        win = sol.rig.windows[w] if w >= 0 else None
        path = os.path.join(folder, "camFile", f"{names[c]}.txt")
        if win is not None:
            write_camfile(cam, win, path, (pc["mean_px"], pc["std_px"]), (wand["abs_mean_mm"], wand["std_mm"]))
        cams_json.append({
            "name": names[c], "camfile": os.path.relpath(path, folder) if win is not None else None,
            "width": cam.width, "height": cam.height, "f_px": cam.f,
            "f_mm": cam.f * pixel_mm[c] if pixel_mm[c] else None, "cx": cam.cx, "cy": cam.cy,
            "dist": cam.dist.tolist(), "R": cam.R.tolist(), "rvec": cam.rvec.tolist(), "tvec": cam.tvec.tolist(),
            "centre_mm": cam.C.tolist(), "viewing_direction": cam.R[2].tolist(), "window": w,
            "distance_to_window_mm": win.distance(cam.C) if win is not None else None,
            "incidence_deg": (math.degrees(math.acos(max(-1.0, min(1.0, float(cam.R[2] @ win.n)))))
                              if win is not None else None),
            "sigma": sig[c], "residuals": pc})
    wins_json = []
    for k, win in enumerate(sol.rig.windows):
        wins_json.append({"name": win.name, "normal": win.n.tolist(), "d_mm": win.d, "far_d_mm": win.far_d,
                          "thickness_mm": list(win.t), "n": list(win.idx), "cameras": sol.rig.cams_of(k),
                          "sigma": wsig[k]})
    result = {
        "model": "PINPLATE",
        "frame": "Z = window 1's normal (camera -> water); Y = n1 x n2 (two windows) or camera 1's image up in "
                 "window 1's plane; origin = the wand's mean position",
        "wand_length_mm": sol.L, "fit_L": bool(sol.layout.fit_L), "fit_k1": bool(sol.layout.fit_k1),
        "gauge": "camera 1's pose held during the fit (sigmas are in that gauge)",
        "rms_px": sol.rms, "sigma_px": sol.sigma_px, "frames": int(sol.obs.n_frames),
        "observations": int(len(sol.obs)), "active": int(sol.obs.active.sum()), "cameras": cams_json,
        "windows": wins_json, "wand_length_free_triangulation": wand,
        "frames_dropped": stats["frames_dropped"],
        "frames_with_rejected_detections": stats["frames_with_rejected_detections"],
        "noise": noise_summary(sol, names), "predicted": predicted_precision(sol),
        "info": {k: v for k, v in sol.info.items() if _jsonable(v)}}
    if sol.layout.fit_L and sol.cov is not None:
        result["L_sigma_mm"] = math.sqrt(sol.cov[sol.layout.L_index, sol.layout.L_index])
    if extra:
        result.update(extra)
    with open(os.path.join(folder, "calibration_result.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=float)
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
    wl = res["wand_length_free_triangulation"]
    out = [f"Flat-window wand calibration (PINPLATE), {len(res['cameras'])} cameras, {len(res['windows'])} window(s)",
           f"wand {res['wand_length_mm']:.4f} mm" + (f" (fitted, sigma {res['L_sigma_mm']:.4f} mm)"
                                                     if 'L_sigma_mm' in res else ""),
           f"frames {res['frames']}, observations {res['active']} active of {res['observations']}; reprojection RMS "
           f"{res['rms_px']:.4f} px (2-D), sigma {res['sigma_px']:.4f} px per coordinate",
           f"wand length, free triangulation: mean error {wl['mean_mm'] * 1e3:+.2f} um, RMS {wl['rms_mm'] * 1e3:.2f} um "
           f"over {wl['frames']} frames",
           f"frames dropped: {len(res['frames_dropped'])}; frames with some detections rejected: "
           f"{len(res['frames_with_rejected_detections'])}",
           f"frame: {res['frame']}", f"gauge: {res['gauge']}", ""]
    for k, w in enumerate(res["windows"]):
        s = w["sigma"] or {}
        out.append(f"window {k + 1} ({w['name'] or 'unnamed'}): normal ({w['normal'][0]:+.6f}, {w['normal'][1]:+.6f}, "
                   f"{w['normal'][2]:+.6f})" + (f" +- {s['normal_deg']:.4f} deg" if s.get('normal_deg') is not None
                                               else "")
                   + f"; camera-side surface n.X = {w['d_mm']:.3f} mm"
                   + (f" (+- {s['offset_mm']:.3f} at the anchor)" if s.get('offset_mm') is not None else "")
                   + f"; plates {w['thickness_mm']} mm, n {w['n']}; cameras {[c + 1 for c in w['cameras']]}")
    for c in res["cameras"]:
        s = c["sigma"] or {}
        out.append(f"{c['name']}: f {c['f_px']:.2f} px" + (f" ({c['f_mm']:.3f} mm)" if c["f_mm"] else "")
                   + (f" +- {100 * s['f_rel']:.3f} %" if s else "")
                   + (f"; {c['distance_to_window_mm']:.2f} mm from its window" if c['distance_to_window_mm'] else "")
                   + (f" +- {s['distance_mm']:.3f}" if s.get('distance_mm') is not None else "")
                   + (f", incidence {c['incidence_deg']:.2f} deg" if c['incidence_deg'] is not None else "")
                   + f"; reprojection RMS {c['residuals']['rms_px']:.4f} px ({c['residuals']['n_active']}/"
                   f"{c['residuals']['n_total']} active); camFile {c['camfile']}")
    pr = (res.get("predicted") or {}).get("points")
    if pr:
        out += ["", f"predicted calibration error on points like the wand's ({pr['n_points']}): re-triangulated "
                    f"{pr['retriangulated_rms_px']:.4f} px; a known point projected in the window's frame "
                    f"{pr['stabiliser_rms_px']:.4f} px, in camera 1's {pr['fixed_rms_px']:.4f} px"]
    nz = res.get("noise")
    if nz:
        out += ["", "Noise levels and weights (noise_weights=" + repr(nz["mode"]) + ")",
                f"  weights {'APPLIED' if nz['weights_used'] else 'NOT applied'}: {nz['why']}"]
        for r in nz["groups"]:
            sigv = r["sigma_px"] if r["sigma_px"] is not None else r["sigma_estimate_px"]
            out.append(f"  {r['group']}: sigma " + (f"{sigv:.4f} px" if sigv is not None else "-")
                       + (f" (+- {100 * r['rel_se']:.1f} %)" if r["rel_se"] is not None else "")
                       + f", used {r['n_used']}/{r['n_total']}, residual RMS "
                       + (f"{r['residual_rms_px_per_coordinate']:.4f} px/coord." if r['residual_rms_px_per_coordinate']
                          is not None else "-"))
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ one call
def calibrate_csv(csv_path: str, config_path: str, out_folder: str, **kw) -> Dict:
    """Read the GUI's wand CSV and a rig configuration, calibrate, and write the results.  Returns the result
    dictionary."""
    cfg = load_config(config_path)
    cd = read_gui_csv(csv_path, cam_ids=[c.get("id", k) for k, c in enumerate(cfg.cameras)])
    data = pw.WandData(cd.uv, cfg.wand_length_mm, cfg.sizes)
    sol = pw.calibrate(data, cfg.rig(), hints=cfg.hints or None, **kw)
    return write_result(sol, out_folder, names=cfg.names, pixel_mm=[c.get("pixel_mm") for c in cfg.cameras])
