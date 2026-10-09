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

**Warnings** (`assess`): the result's "warnings" and the report's WARNINGS block name what the tests associate with a
wrong or a weak answer -- a window seen only by one or two narrow-view cameras, an answer pinned against a window, a
hint far from the answer, cameras that all look the same way, high noise on a multi-window rig, few frames, a fitted
wand length -- each with its evidence and what to do (tests/plate_wand/2026-10-03-angles-and-hints-report.md).
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
    hints (optional): {"f": [px per camera or null], "f_prior_rel": sigma, "normals": [per camera null or the normal
    of its window in the camera's own frame (x along the image rows, y down, z along the optical axis), e.g.
    [0.5, 0, 0.866] for a camera turned 30 deg to the left of square-on; a rough guess is enough]}."""

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


def find_camfile(folder: str, cam_id: int) -> Optional[str]:
    """Camera cam_id's camFile in a calibration folder: camFile/cam<id>.txt, else cam<id>.txt; None if neither."""
    for p in (os.path.join(folder, "camFile", f"cam{int(cam_id)}.txt"), os.path.join(folder, f"cam{int(cam_id)}.txt")):
        if os.path.isfile(p):
            return p
    return None


def load_start(folder: str, cam_ids: Sequence[int], cam_to_window: Sequence[int]):
    """A start for plate_wand.calibrate from a saved fit: (cameras in cam_ids' order, their rig, a description).

    - A plate_wand result folder (calibration_result.json, written by the GUI or by a case's scripts): its cameras and
      windows exactly as fitted; cameras matched by name (cam<id>).
    - Otherwise PINPLATE camFiles (camFile/cam<id>.txt or cam<id>.txt, e.g. an earlier calibration of the GUI): each
      camera with its window; a window's plane from its first camera's file.
    The saved fit must group the cameras into windows as cam_to_window does (rig indices); the windows come back in
    that order.  Thicknesses and indices are the saved ones: the caller applies its own (plate_wand_gui.run)."""
    cam_ids = [int(c) for c in cam_ids]
    cam_to_window = [int(k) for k in cam_to_window]
    n_win = max(cam_to_window) + 1
    groups = [sorted(c for c in range(len(cam_ids)) if cam_to_window[c] == k) for k in range(n_win)]
    path = os.path.join(folder, "calibration_result.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            res = json.load(fh)
        by_name = {c.get("name"): c for c in res.get("cameras", [])}
        missing = [c for c in cam_ids if f"cam{c}" not in by_name]
        if missing:
            raise ValueError(f"{path} has no camera named cam{missing[0]} (it has {sorted(by_name)}).")
        rows = [by_name[f"cam{c}"] for c in cam_ids]
        cams = [pm.Camera(np.asarray(r["R"], float), np.asarray(r["centre_mm"], float), r["f_px"], r["cx"], r["cy"],
                          int(r["width"]), int(r["height"]), np.asarray(r["dist"], float)) for r in rows]
        saved = [int(r["window"]) for r in rows]
        wins = []
        for k, grp in enumerate(groups):
            ws = {saved[c] for c in grp}
            if len(ws) != 1 or any(saved[c] in ws for c in range(len(cam_ids)) if c not in grp):
                raise ValueError(f"{path} groups these cameras into windows differently from the Camera-Window "
                                 "Mapping.")
            w = res["windows"][ws.pop()]
            wins.append(pm.Window(np.asarray(w["normal"], float), float(w["d_mm"]), tuple(w["thickness_mm"]),
                                  tuple(w["n"]), w.get("name", "")))
        return cams, pw.Rig(wins, cam_to_window), f"{path} (RMS {res.get('rms_px', float('nan')):.4f} px)"
    files = [find_camfile(folder, c) for c in cam_ids]
    missing = [c for c, f in zip(cam_ids, files) if f is None]
    if missing:
        raise ValueError(f"{folder} has neither calibration_result.json nor camFile/cam{missing[0]}.txt.")
    loaded = [load_camfile(f) for f in files]
    cams = [cam for cam, _w in loaded]
    wins = []
    for grp in groups:
        w = loaded[grp[0]][1]
        wins.append(pm.Window(w.n.copy(), float(w.d), tuple(w.t), tuple(w.idx), w.name))
    return cams, pw.Rig(wins, cam_to_window), f"{len(files)} camFiles in {folder}"


def camfile_normals(folder: str, cam_ids: Sequence[int]) -> List[List[float]]:
    """Window-normal hints from an earlier calibration's camFiles: each camera's window normal in its own frame
    (R n), as the real case's --hints gui:<calibration> (cases/plate_20260727_T0/run_calibration.py)."""
    out = []
    for c in cam_ids:
        f = find_camfile(folder, c)
        if f is None:
            raise ValueError(f"{folder} has no camFile for camera {c} (camFile/cam{c}.txt or cam{c}.txt).")
        cam, win = load_camfile(f)
        out.append([float(v) for v in cam.R @ win.n])
    return out


# ------------------------------------------------------------------ warnings: when to doubt the answer
WEAK_VIEW_DEG = pw.WEAK_VIEW_DEG   # a window whose one or two cameras all see the wand under less than this
PINNED_MM = 1e-6             # a wand end this close to a window's water-side surface: the answer may be pinned
HINT_FAR_DEG = 15.0          # a hinted normal this far from the answer's
SAME_WAY_DEG = 10.0          # every two cameras' lines of sight closer than this (rig C: 5 deg)
NOISY_PX = 0.5               # noise per coordinate above which multi-window rigs went wrong more often
FEW_FRAMES = 300             # active frames below this (the tests used 500-1,500)


def view_angles_deg(sol: pw.Solution) -> List[Optional[float]]:
    """Per camera, the angle under which it sees the wand (plate_wand.view_angles_deg)."""
    return pw.view_angles_deg(sol.cams, sol.obs)


def assess(sol: pw.Solution, names: Sequence[str] = None) -> List[Dict]:
    """What the tests associate with a wrong or weak answer, found in this one: [{"code", "text", ...}], empty when
    nothing is.  Each text gives the evidence (tests/plate_wand/) and what to do.  Codes: weak_window, pinned,
    hint_far, same_way, noisy, few_frames, wand_length_fitted."""
    n = len(sol.cams)
    names = list(names) if names is not None else [f"cam{c + 1}" for c in range(n)]
    out = []
    ang = view_angles_deg(sol)
    used = [w for w in range(len(sol.rig.windows)) if sol.rig.cams_of(w)]
    scanned = {s["window"]: s for s in ((sol.info.get("scan") or {}).get("windows") or [])}
    for w in pw.weak_windows(sol.cams, sol.rig, sol.obs):
        cs = sol.rig.cams_of(w)
        a = [ang[c] for c in cs]
        sc = scanned.get(w)
        done = ("" if sc is None else " The calibration scanned this window's normal" + (
            " and found a better answer (used)." if sc.get("accepted") else " and found nothing better."))
        out.append({"code": "weak_window", "window": w, "cameras": cs, "view_deg": a, "text": (
            f"window {w + 1} is seen only by {' and '.join(names[c] for c in cs)}, which see the wand under "
            f"{' and '.join(f'{x:.1f}' for x in a)} deg (a long lens or a far camera): its tilt shows only weakly, and "
            f"the cost can have several minima. Every wrong answer in the tests had such a window (W5 and R4: 5 of the "
            f"22 cases with one, none of the 98 without).{done} Check this window's sigma below. The remedy is a camera "
            f"with a wider view behind it; window-normal hints help when accurate.")})
    depths = pw.window_depths(sol.cams, sol.rig, sol.frames, sol.L, sol.obs)
    for w, d in enumerate(depths):
        if d is not None and d < PINNED_MM:
            out.append({"code": "pinned", "window": w, "min_depth_mm": d, "text": (
                f"a wand end lies on window {w + 1}'s water-side surface ({d:.1e} mm deep): the fit may be pinned "
                f"against the window rather than at a minimum. 8 of the 14 wrong multi-window answers in the tests "
                f"were, none of the 69 right ones. Less detection noise or more cameras behind the window help.")})
    off = (sol.info.get("hints") or {}).get("normals_off_deg") or []
    far = [(c, a) for c, a in enumerate(off) if a is not None and a > HINT_FAR_DEG]
    if far:
        out.append({"code": "hint_far", "cameras": [c for c, _a in far], "off_deg": [a for _c, a in far], "text": (
            "the answer's window normal is " + ", ".join(f"{a:.1f} deg from {names[c]}'s hint" for c, a in far)
            + ": either the hint is off by that much, or the answer is wrong. If you trust the hint to better than "
              "that, doubt the answer.")})
    if n >= 2:
        axes = [cam.R[2] for cam in sol.cams]
        spread = max(math.degrees(math.acos(min(1.0, abs(float(a @ b))))) for i, a in enumerate(axes)
                     for b in axes[i + 1:])
        if spread < SAME_WAY_DEG:
            out.append({"code": "same_way", "spread_deg": spread, "text": (
                f"every camera's line of sight is within {spread:.1f} deg of the others (as lines): the cameras see "
                f"the wand from nearly one direction, so f, the distances and the 3-D shape are poorly determined "
                f"(rig C, 5 deg apart: 3-9x worse than lines of sight 20-40 deg apart). Spread the cameras.")})
    if len(used) >= 2 and sol.sigma_px > NOISY_PX:
        out.append({"code": "noisy", "sigma_px": sol.sigma_px, "text": (
            f"the residuals are {sol.sigma_px:.2f} px per coordinate on a rig with several windows. If your detections "
            f"are that noisy: in the tests, answers pinned against a window came mostly at 1 px (6 of 8), and so did "
            f"R4's one wrong answer left; aim for about 0.1 px (sharp focus, short exposure, bright balls). If they are "
            f"much less noisy (a still wand imaged over many frames shows it), the answer itself is probably wrong: a "
            f"wrong minimum leaves large residuals (W5 seed 1031 without hints: 3.6 px RMS from 0.1 px detections).")})
    act = np.zeros(sol.obs.n_frames, bool)
    act[sol.obs.frame[sol.obs.active]] = True
    if int(act.sum()) < FEW_FRAMES:
        out.append({"code": "few_frames", "frames": int(act.sum()), "text": (
            f"only {int(act.sum())} frames used: the tests used 500-1,500. Record about 1,000 frames over the whole "
            f"volume.")})
    if sol.layout.fit_L:
        out.append({"code": "wand_length_fitted", "text": (
            "the wand length was fitted: through flat windows it is not identifiable (W8: 9-91 mm for a 25 mm wand), "
            "so the scale is not checked. Give the measured length.")})
    return out


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
        "fit_lens": list(getattr(sol.layout, "fit_lens", ())),
        "gauge": "camera 1's pose held during the fit (sigmas are in that gauge)",
        "rms_px": sol.rms, "sigma_px": sol.sigma_px, "frames": int(sol.obs.n_frames),
        "observations": int(len(sol.obs)), "active": int(sol.obs.active.sum()), "cameras": cams_json,
        "windows": wins_json, "wand_length_free_triangulation": wand,
        "frames_dropped": stats["frames_dropped"],
        "frames_with_rejected_detections": stats["frames_with_rejected_detections"],
        "noise": noise_summary(sol, names), "predicted": predicted_precision(sol),
        "warnings": assess(sol, names), "info": {k: v for k, v in sol.info.items() if _jsonable(v)}}
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


def _perturbed_line(pt: Dict) -> str:
    """The report's line on the restarts perturbed about the answer (plate_wand.perturbed_restart)."""
    rounds = pt.get("rounds") or []
    n = sum(r.get("screened", 0) for r in rounds)
    head = f"restarts perturbed about the answer: {n} draws at {pt['scale']:g} sigma in {len(rounds)} round(s); "
    if any(r.get("skipped") for r in rounds):
        return head + "skipped (" + "; ".join(r["skipped"] for r in rounds if r.get("skipped")) + ")"
    if pt.get("improved"):
        return head + (f"the answer IMPROVED, rms {pt['rms_before_px']:.5f} -> {pt['rms_px']:.5f} px (its start had "
                       f"ended in a wrong minimum)")
    lost = [r for r in rounds if r.get("rms_after_px") is not None]
    if lost:
        return head + (f"a restart refitted to rms {lost[0]['rms_after_px']:.5f} px was not better on the same terms "
                       f"(capped cost {lost[0].get('capped_rel', float('nan')):+.1e}); the answer stands")
    return head + "none lowered the cost; the answer stands"


def _lens_lines(cameras) -> list:
    """The report's lines on the lens terms of each camera, with their sigmas where fitted (`plate_wand.LENS_TERMS`)."""
    out = []
    for c in cameras:
        s = c["sigma"] or {}
        d = c["dist"]
        sd = lambda k, fmt: f" +- {s[k]:{fmt}}" if s.get(k) is not None else ""
        parts = [f"principal point ({c['cx']:.2f}{sd('cx_px', '.2f')}, {c['cy']:.2f}{sd('cy_px', '.2f')}) px",
                 f"k1 {d[0]:.4f}{sd('k1', '.4f')}"]
        for name, i in (("k2", 1), ("p1", 2), ("p2", 3)):
            if s.get(name) is not None or d[i] != 0.0:
                parts.append(f"{name} {d[i]:.4g}{sd(name, '.2g')}")
        out.append(f"{c['name']} lens: " + ", ".join(parts))
    return out


def _constraint_lines(cons) -> list:
    """The report's lines on the constraints at the answer (`plate_wand.constraint_report`)."""
    if not cons:
        return []
    out = []
    ch = cons.get("channel")
    if ch:
        parts = []
        for k, w in enumerate(ch["windows"]):
            lo, hi = w["band_mm"]
            parts.append(f"window {k + 1}: the wand ends {w['depth_mm'][0]:.1f}-{w['depth_mm'][1]:.1f} mm below it; it "
                         f"could move {'-inf' if lo is None else f'{lo:+.1f}'} to {hi:+.1f} mm along its normal (+: "
                         f"away from its cameras) and still hold them")
        out.append(f"channel (held): inside widths {ch['width']} mm + {ch['tol']:g} mm, ball radii {list(ch['ball'])} "
                   f"mm; " + "; ".join(parts) + f"; {ch['active_rows']} wand ends on a limit (largest breach "
                   f"{ch['max_breach_mm']:.3f} mm)")
    for p in cons.get("priors") or []:
        if p["kind"] == "aim":
            out.append(f"aim: cameras {[c + 1 for c in p['cam']]} aim {p['value']:.2f} mm below window {p['wall'] + 1}'s "
                       f"water side (given {p['mean']:.2f}, sigma {p['sigma']:g} mm)")
        elif p["kind"] == "distance":
            out.append(f"distance: camera {p['cam'] + 1} is {p['value']:.2f} mm from its window (given {p['mean']:.2f}, "
                       f"sigma {p['sigma']:g} mm)")
    return out


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
           f"{len(res['frames_with_rejected_detections'])}"]
    warns = res.get("warnings")
    if warns is not None:
        out.append(f"WARNINGS ({len(warns)}):" if warns else "warnings: none (nothing the tests associate with a wrong "
                                                             "or weak answer)")
        out += [f"  - {x['text']}" for x in warns]
    pt = ((res.get("info") or {}).get("starts") or {}).get("perturbed")
    if pt:
        out.append(_perturbed_line(pt))
    hh = (res.get("info") or {}).get("hints") or {}
    off = hh.get("normals_off_deg")
    if off and any(a is not None for a in off):
        line = ("window-normal hints: the answer's normals are " + ", ".join(f"{a:.1f}" for a in off if a is not None)
                + f" deg from them (cameras {', '.join(str(c + 1) for c, a in enumerate(off) if a is not None)}; a "
                  "large angle means a rough hint or a wrong answer)")
        names = {"default": "the start without hints", "held": "the hint-held start",
                 "searched": "the hint-searched start"}
        for nm, p in (hh.get("paths") or {}).items():
            lab = names.get(nm, nm)
            if p.get("error"):
                line += f"; {lab} failed ({p['error']})"
            elif p.get("same_as"):
                line += f"; {lab} reached the same minimum as {names.get(p['same_as'], p['same_as'])}"
            elif p.get("rms_px") is not None:
                line += (f"; {lab} ended at rms {p['rms_px']:.5f} px"
                         + (f" (capped cost {p['capped_rel']:+.1e} relative to the answer without hints)"
                            if p.get("capped_rel") is not None else ""))
        if hh.get("chosen"):
            line += f"; the answer is from {names.get(hh['chosen'], hh['chosen'])}"
        out.append(line)
    sc = (res.get("info") or {}).get("scan")
    if sc:
        parts = []
        for s in sc.get("windows") or []:
            p = f"window {s['window'] + 1}: "
            if s.get("skipped"):
                p += f"skipped ({s['skipped']})"
            else:
                p += (f"{s.get('fitted', 0)} of {s.get('points', 0)} normals fitted, {len(s.get('released') or [])} "
                      f"released, " + (f"the answer IMPROVED to rms {s['rms_px']:.5f} px" if s.get("accepted")
                                       else "nothing better"))
            parts.append(p)
        out.append(f"scan of the weakly seen windows' normals ({sc['seconds']:.0f} s): " + "; ".join(parts))
    out += _constraint_lines((res.get("info") or {}).get("constraints"))
    out += [f"frame: {res['frame']}", f"gauge: {res['gauge']}", ""]
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
    if res.get("fit_lens"):
        out += _lens_lines(res["cameras"])
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
