"""Build the results page of a wand calibration: a model-independent report dictionary, then one self-contained HTML
file (template.html; inline SVG and canvas, no network).  Adapters turn each calibration's own results into the
report: `report_from_cylinder` (PINCYL, curved_calibration/cylinder_wand_gui.py), `report_from_plate` (PINPLATE,
curved_calibration/plate_wand_gui.py) and `report_from_pinhole` (the GUI's pinhole wand calibration).

The report (schema 1): title, model, model_label, created, folder; summary {cameras, frames, frames_total, detections,
detections_used, rms_px, wand_length_mm, wand_rms_um, wand_mean_um, wand_frames}; checks [{level, title, detail,
what_to_do, evidence}]; cameras [{name, f_px, f_mm, f_sigma_rel, k1, cx, cy, width, height, centre, R, rms_px,
n_used, n_total, extra: [[label, text], ...]}]; port None | {type: "tube", radii, n, fold_radius} | {type: "windows",
windows: [{name, point, normal, thickness, n}]}; up (the views' vertical); points [[x, y, z], ...] (wand ends);
sight_lines [[camera index, [[x, y, z], ...]], ...]; residuals [{cam, u, v, du, dv, used, frame, end, subsampled}];
error_pdf None | {cover, uv, mag, joint: {lo, hi, width, bins (joint: max)}, cams [{n, n_total, du, dv, mag (counts per
bin), cells [[i, j, count]], mu, mv, mean_du, rms_du, mean_dv, rms_dv, median, rms, p95, max, beyond_du, beyond_dv,
beyond_mag, beyond_joint}]} (_error_pdf); wand {x, x_label, x_short, err_um, frame, lead}; noise {columns, rows,
num_cols, note}; settings [[label, text]]; files [{label, path}].  Lengths in mm, pixels in px; NaN and inf become
null.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

TEMPLATE = Path(__file__).with_name("template.html")
MAX_POINTS = 3000              # wand ends drawn (all frames are in the statistics)
MAX_RESIDUALS = 2500           # detections drawn per camera
ERR_COVER = 99.5               # the error charts' range holds this percentile of every camera's errors
ERR_BINS = (20, 150)           # the du, dv and |e| charts: bins across the range, at least and at most
ERR_CELLS = (16, 40)           # the du-against-dv chart: cells per side, at least and at most


# ------------------------------------------------------------------ page
def _clean(x):
    """JSON-safe: numpy scalars and arrays to lists, non-finite floats to None."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def render_html(report: Dict) -> str:
    html = TEMPLATE.read_text(encoding="utf-8")
    assert html.count("__REPORT_DATA__") == 1
    data = json.dumps(_clean(report), separators=(",", ":"), allow_nan=False)
    data = data.replace("</", "<\\/")                 # nothing in the data can close the script element
    return html.replace("__REPORT_DATA__", data)


def write_page(report: Dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(render_html(report))
    os.replace(tmp, path)
    return path


def _every(n: int, max_n: int) -> np.ndarray:
    """Indices of at most max_n of n items, evenly spread (all when n <= max_n)."""
    if n <= max_n:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_n).round().astype(int))


def _fmt(v, d, unit=""):
    if v is None or not np.isfinite(v):
        return "–"
    v = float(v)
    if abs(v) < 0.5 * 10.0 ** (-d):
        v = 0.0                                       # no "-0.00"
    return f"{v:,.{d}f}{unit}"


# ------------------------------------------------------------------ how the wand points were detected
WAND_TYPES = {"dark": "Dark on Bright", "bright": "Bright on Dark"}


def _num(v) -> str:
    """A detection setting as the panel shows it: at most two decimals, no trailing zeros."""
    return f"{float(v):.2f}".rstrip("0").rstrip(".")


def _detection_rows(det: Optional[Dict], mode: Optional[str] = None) -> List[List[str]]:
    """The settings table's rows on how the wand points were detected: from their detection record (the wand tab's
    <points>_detection.json, wand_calibrator.save_detection_record; its last run), else the mode alone."""
    runs = (det or {}).get("runs") or []
    if not runs:
        return [["Detection", str(mode or "unknown (points read from a file)")]]
    run = runs[-1]
    mode = run.get("mode") or mode
    cams = run.get("cameras") or []
    if run.get("per_camera"):
        how = "each camera's own settings"
    elif any(abs(float(c.get("scale", 1.0)) - 1.0) > 1e-12 for c in cams):
        how = "shared settings, scaled by each camera's pixel size"
    else:
        how = "shared settings"
    head = f"{mode} mode, {WAND_TYPES.get(run.get('wand_type'), run.get('wand_type') or '?')}; {how}"
    if len(runs) > 1:
        def key(r):
            return (r.get("mode"), r.get("wand_type"), r.get("per_camera"),
                    [{k: v for k, v in c.items() if k != "images"} for c in r.get("cameras") or []])
        head += (f"; detected in {len(runs)} runs (resumed), "
                 + ("all with these settings" if all(key(r) == key(run) for r in runs[:-1])
                    else "with different settings: the last run's are shown, the record has every run's"))
    rows = [["Detection", head]]
    for c in cams:
        parts = [f"radius {_num(c['min_radius'])}–{_num(c['max_radius'])} px"]
        if mode != "ellipse":
            parts.append(f"sensitivity {_num(c['sensitivity'])}")
        if "open_px" in c:
            parts += [f"Rod opening {int(c['open_px'])} px", f"Rod mask {_num(c['rod_mask_px'])} px"]
        if abs(float(c.get("scale", 1.0)) - 1.0) > 1e-12:
            parts.append(f"× {float(c['scale']):.4g} for its pixel size")
        rows.append([f"Detection, cam{c['cam_id']}", "; ".join(parts)])
    for c in cams:
        im = c.get("images") or {}
        if im:
            rows.append([f"Images, cam{c['cam_id']}", f"{im.get('folder', '')}: {int(im.get('count', 0)):,} images, "
                                                      f"{im.get('first', '')} to {im.get('last', '')}"])
    if det.get("file"):
        rows.append(["Detection record", str(det["file"])])
    return rows


# ------------------------------------------------------------------ the reprojection errors' distributions
def _nice_up(x: float) -> float:
    """The smallest of 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8 x 10^k not below x (x > 0)."""
    p = 10.0 ** math.floor(math.log10(x))
    return next(m * p for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10) if m * p >= x * (1 - 1e-9))


def _round_width(span: float, target: float, lo: int, hi: int):
    """(width, count): a bin width of 1, 2, 2.5, 4 or 5 x 10^k that splits `span` into lo..hi bins exactly, the
    nearest to `target` on a log scale."""
    best = None
    k0 = math.floor(math.log10(span))
    for k in range(k0 - 4, k0 + 1):
        for m in (1.0, 2.0, 2.5, 4.0, 5.0):
            w = m * 10.0 ** k
            nb = span / w
            if abs(nb - round(nb)) < 1e-6 and lo <= round(nb) <= hi:
                d = abs(math.log(w / target))
                if best is None or d < best[0]:
                    best = (d, w, int(round(nb)))
    if best is None:                                  # (not met by the spans _nice_up gives)
        return span / lo, lo
    return best[1], best[2]


def _error_pdf(errs: Sequence[np.ndarray], totals: Sequence[int]) -> Optional[Dict]:
    """The page's error charts: per camera the counts of du, dv (the model's position minus the detection, px) and
    |e| = sqrt(du^2 + dv^2) in bins shared by all cameras, the counts of du against dv in square cells, and the
    statistics.  errs[k]: camera k's (n, 2) array of (du, dv) over the detections the fit used; totals[k]: all its
    detections.  The range holds the ERR_COVER percentile of every camera's |du|, |dv| (and |e|); the bin width is
    a round number near the Freedman-Diaconis width of the narrowest camera.  None when no camera has an error."""
    errs = [np.asarray(e, float).reshape(-1, 2) for e in errs]
    errs = [e[np.all(np.isfinite(e), axis=1)] for e in errs]
    used = [e for e in errs if len(e)]
    if not used:
        return None

    def cover(vals):
        q = max(float(np.percentile(v, ERR_COVER)) for v in vals)
        if not q > 0:
            q = max(float(np.max(v)) for v in vals)
        return _nice_up(q) if q > 0 else 1.0

    def fd_width(vals):
        w = [2.0 * float(np.subtract(*np.percentile(v, [75, 25]))) * len(v) ** (-1.0 / 3.0)
             for v in vals if len(v) >= 20]
        w = [x for x in w if x > 0]
        return min(w) if w else None

    mags = [np.hypot(e[:, 0], e[:, 1]) for e in used]
    R = cover([np.abs(e[:, 0]) for e in used] + [np.abs(e[:, 1]) for e in used])
    Rm = cover(mags)
    w, nb = _round_width(2 * R, fd_width([e[:, 0] for e in used] + [e[:, 1] for e in used]) or 2 * R / 40, *ERR_BINS)
    wm, nbm = _round_width(Rm, fd_width(mags) or Rm / 40, *ERR_BINS)
    n_min = min(len(e) for e in used)
    cells = min(max(2 * round(math.sqrt(n_min) / 3), ERR_CELLS[0]), ERR_CELLS[1])
    wj, nj = _round_width(2 * R, 2 * R / cells, *ERR_CELLS)
    cams, jmax = [], 0
    for k, e in enumerate(errs):
        n = len(e)
        if n == 0:
            cams.append({"n": 0, "n_total": int(totals[k])})
            continue
        du, dv = e[:, 0], e[:, 1]
        m = np.hypot(du, dv)
        cu = np.histogram(du, bins=nb, range=(-R, R))[0]
        cv = np.histogram(dv, bins=nb, range=(-R, R))[0]
        cm = np.histogram(m, bins=nbm, range=(0.0, Rm))[0]
        H = np.histogram2d(du, dv, bins=nj, range=[[-R, R], [-R, R]])[0].astype(int)
        jmax = max(jmax, int(H.max()))
        cams.append({"n": n, "n_total": int(totals[k]), "du": cu.tolist(), "dv": cv.tolist(), "mag": cm.tolist(),
                     "cells": [[int(i), int(j), int(H[i, j])] for i, j in zip(*np.nonzero(H))],
                     "mu": np.histogram(du, bins=nj, range=(-R, R))[0].tolist(),
                     "mv": np.histogram(dv, bins=nj, range=(-R, R))[0].tolist(),
                     "mean_du": float(du.mean()), "rms_du": float(np.sqrt(np.mean(du ** 2))),
                     "mean_dv": float(dv.mean()), "rms_dv": float(np.sqrt(np.mean(dv ** 2))),
                     "median": float(np.median(m)), "rms": float(np.sqrt(np.mean(m ** 2))),
                     "p95": float(np.percentile(m, 95)), "max": float(m.max()),
                     "beyond_du": int(n - cu.sum()), "beyond_dv": int(n - cv.sum()), "beyond_mag": int(n - cm.sum()),
                     "beyond_joint": int(n - H.sum())})
    return {"cover": ERR_COVER, "uv": {"lo": -R, "hi": R, "width": w, "bins": nb},
            "mag": {"lo": 0.0, "hi": Rm, "width": wm, "bins": nbm},
            "joint": {"lo": -R, "hi": R, "width": wj, "bins": nj, "max": jmax}, "cams": cams}


def _errors_from_data(data: Dict, names: Sequence[str]):
    """([errors], [totals]) per camera (in the order of `names`, cam<id>): the (du, dv) of the detections the fit
    used, as an (n, 2) array, and the number of the camera's detections."""
    obs = data["observations"]
    index_of = {n: k for k, n in enumerate(names)}
    errs, totals = [[] for _ in names], [0] * len(names)
    for i, cid in enumerate(obs["cam_id"]):
        k = index_of.get(f"cam{cid}")
        if k is None:
            continue
        totals[k] += 1
        du, dv = obs["du"][i], obs["dv"][i]
        if obs["active"][i] and du is not None and dv is not None:
            errs[k].append((du, dv))
    return [np.asarray(e, float).reshape(-1, 2) for e in errs], totals


# ------------------------------------------------------------------ the cylinder (PINCYL)
def report_from_cylinder(result: Dict, data: Dict, folder: str = None) -> Dict:
    """The report of a cylinder wand calibration: cylinder_wand_gui.write_outputs's result dictionary and
    calibration_data.json (cylinder_wand_gui.plot_data)."""
    cams_res = result["cameras"]
    names = [c["name"] for c in cams_res]
    index_of = {n: k for k, n in enumerate(names)}
    obs = data["observations"]
    cam_ids = obs["cam_id"]
    # residuals per camera
    per_cam = {k: [] for k in range(len(cams_res))}
    for i, cid in enumerate(cam_ids):
        k = index_of.get(f"cam{cid}")
        if k is not None and obs["du"][i] is not None:
            per_cam[k].append(i)
    residuals = []
    for k, rows in per_cam.items():
        sel = np.asarray(rows, int)[_every(len(rows), MAX_RESIDUALS)] if rows else np.zeros(0, int)
        residuals.append({"cam": k, "u": [obs["u"][i] for i in sel], "v": [obs["v"][i] for i in sel],
                          "du": [obs["du"][i] for i in sel], "dv": [obs["dv"][i] for i in sel],
                          "used": [bool(obs["active"][i]) for i in sel], "frame": [obs["frame_id"][i] for i in sel],
                          "end": [obs["end"][i] for i in sel], "subsampled": len(sel) < len(rows)})
    # cameras
    cameras = []
    for c in cams_res:
        s = c.get("sigma") or {}
        rr = c.get("residuals") or {}
        cameras.append({"name": c["name"], "f_px": c["f_px"], "f_mm": c.get("f_mm"), "f_sigma_rel": s.get("f_rel"),
                        "k1": (c.get("dist") or [None])[0] if result.get("fit_k1") else None,
                        "cx": c["cx"], "cy": c["cy"], "width": c["width"],
                        "height": c["height"], "centre": c["centre_mm"], "R": c["R"], "rms_px": rr.get("rms_px"),
                        "n_used": rr.get("n_active"), "n_total": rr.get("n_total"),
                        "extra": [["Distance from the axis",
                                   _fmt(c["distance_from_axis_mm"], 1, " mm")
                                   + (f" ± {s['distance_mm']:.2f}" if s.get("distance_mm") else "")],
                                  ["Azimuth", _fmt(c["azimuth_deg"], 2, "°")],
                                  ["Height", _fmt(c["height_mm"], 2, " mm")]]})
    # wand ends and lengths
    A = np.asarray(data["ends_A_mm"], float).reshape(-1, 3)
    B = np.asarray(data["ends_B_mm"], float).reshape(-1, 3)
    act = np.asarray(data.get("frame_active", [True] * len(A)), bool)
    keep = np.flatnonzero(act)
    pts_idx = keep[_every(len(keep), MAX_POINTS // 2)]
    points = np.vstack([A[pts_idx], B[pts_idx]]).round(4).tolist()
    Lfree = np.array([np.nan if v is None else v for v in data["wand_length_free_mm"]], float)
    L = float(result["wand_length_mm"])
    mid = 0.5 * (A + B)
    rho = np.hypot(mid[:, 0], mid[:, 2])
    ok = act & np.isfinite(Lfree)
    wand = {"x": rho[ok].round(4).tolist(), "err_um": ((Lfree[ok] - L) * 1e3).round(4).tolist(),
            "frame": [data["frame_ids"][i] for i in np.flatnonzero(ok)],
            "x_label": "Distance of the wand's centre from the tube axis (mm)", "x_short": "from the axis (mm)",
            "lead": "Plotted against the wand's distance from the tube axis: a trend with that distance means the "
                    "tube's radii or indices are off."}
    w = result["wand_length_free_triangulation"]
    tube = result["tube"]
    gui = result.get("gui", {}).get("settings", {})
    settings = [["Wand length", f"{gui.get('wand_length_mm', L):.4f} mm"
                 + (f" (fitted: {L:.4f} ± {result['L_sigma_mm']:.4f} mm)" if result.get("fit_L") else "")],
                ["Tube radii", ", ".join(f"{r:.4f}" for r in tube["radii_mm"]) + " mm (outer first)"],
                ["Refractive indices", ", ".join(f"{v:.4f}" for v in tube["n"]) + " (camera side first)"],
                ["Fold radius", f"{tube['fold_radius_mm']:.2f} mm"],
                ["Lens distortion", "k1 fitted" if result.get("fit_k1") else "none"],
                ["Noise weights", str(gui.get("noise_weights", "camera"))],
                ["Starts carried", str(gui.get("n_final", 2)) + (" (refit from the last result)"
                                                                  if gui.get("refit_from_last_result") else "")],
                *_detection_rows(gui.get("detection"), gui.get("detect_mode")),
                ["Points from", str(gui.get("source") or "-")],
                ["Frame", result.get("frame", "")],
                ["Run time", _fmt(result.get("gui", {}).get("seconds"), 0, " s")]]
    nz = result.get("noise") or {}
    noise = None
    if nz.get("groups"):
        noise = {"columns": ["Group", "Noise sigma (px)", "± rel. SE", "Used / total", "Residual RMS (px/coord.)"],
                 "num_cols": [1, 2, 3, 4],
                 "rows": [[g["group"], _fmt(g["sigma_px"] if g["sigma_px"] is not None else g["sigma_estimate_px"], 4),
                           _fmt(100 * g["rel_se"], 1, " %") if g.get("rel_se") is not None else "–",
                           f"{g['n_used']} / {g['n_total']}", _fmt(g.get("residual_rms_px_per_coordinate"), 4)]
                          for g in nz["groups"]],
                 "note": ("Weights " + ("applied: " if nz.get("weights_used") else "not applied: ") + nz.get("why", "")
                          + ". Noise sigma: each group's own detection noise; residual RMS: what is left after the fit.")}
    files = [{"label": c["name"], "path": os.path.join(folder or "", c["camfile"])} for c in cams_res]
    files += [{"label": n, "path": os.path.join(folder or "", n)} for n in
              ("calibration_result.json", "calibration_report.txt", "calibration_data.json", "calibration_report.html")]
    return {
        "schema": 1, "title": "Wand calibration through a tube", "model": "PINCYL",
        "model_label": "Pinhole + cylinder (PINCYL)",
        "created": result.get("gui", {}).get("written") or time.strftime("%Y-%m-%d %H:%M"),
        "folder": folder,
        "summary": {"cameras": len(cams_res), "frames": int(act.sum()), "frames_total": len(act),
                    "detections": result["observations"], "detections_used": result["active"],
                    "rms_px": result["rms_px"], "wand_length_mm": L, "wand_rms_um": w["rms_mm"] * 1e3,
                    "wand_mean_um": w["mean_mm"] * 1e3, "wand_frames": w["frames"]},
        "checks": result.get("checks", []),
        "cameras": cameras,
        "port": {"type": "tube", "radii": tube["radii_mm"], "n": tube["n"], "fold_radius": tube["fold_radius_mm"]},
        "up": [0.0, 1.0, 0.0],
        "points": points,
        "sight_lines": data.get("sight_lines_mm", []),
        "residuals": residuals,
        "error_pdf": _error_pdf(*_errors_from_data(data, names)),
        "wand": wand,
        "noise": noise,
        "settings": settings,
        "files": files,
        "top_lead": "Looking down the tube's axis: each camera, its lines of sight bent at the walls, and the wand "
                    "ends. Left: the whole rig; right: the tube, with its fold radius dashed (beyond it a point can "
                    "have two images). Hover a camera for its numbers.",
        "cameras_lead": "f is the pinhole model's focal length in pixels: with a macro lens focused close it is much "
                        "larger than the lens's focal length divided by the pixel size. The frame is the tube's: Y "
                        "along the axis, Z towards the first camera.",
        "footer": f"Model PINCYL; {len(cams_res)} cameras.",
    }


def _residuals_from_data(data: Dict, names: Sequence[str]) -> List[Dict]:
    """Per camera (in the order of `names`, cam<id>), the detections and their residuals, at most MAX_RESIDUALS."""
    obs = data["observations"]
    index_of = {n: k for k, n in enumerate(names)}
    per_cam = {k: [] for k in range(len(names))}
    for i, cid in enumerate(obs["cam_id"]):
        k = index_of.get(f"cam{cid}")
        if k is not None and obs["du"][i] is not None:
            per_cam[k].append(i)
    out = []
    for k, rows in per_cam.items():
        sel = np.asarray(rows, int)[_every(len(rows), MAX_RESIDUALS)] if rows else np.zeros(0, int)
        out.append({"cam": k, "u": [obs["u"][i] for i in sel], "v": [obs["v"][i] for i in sel],
                    "du": [obs["du"][i] for i in sel], "dv": [obs["dv"][i] for i in sel],
                    "used": [bool(obs["active"][i]) for i in sel], "frame": [obs["frame_id"][i] for i in sel],
                    "end": [obs["end"][i] for i in sel], "subsampled": len(sel) < len(rows)})
    return out


def _up_of(cameras: Sequence[Dict]) -> List[float]:
    """The views' vertical: the cameras' mean image-up direction (minus their image y axes)."""
    v = -np.sum([np.asarray(c["R"], float)[1] for c in cameras], axis=0)
    n = float(np.linalg.norm(v))
    return (v / n).tolist() if n > 1e-9 else [0.0, 1.0, 0.0]


def _noise_table(nz: Dict) -> Optional[Dict]:
    if not nz or not nz.get("groups"):
        return None
    return {"columns": ["Group", "Noise sigma (px)", "± rel. SE", "Used / total", "Residual RMS (px/coord.)"],
            "num_cols": [1, 2, 3, 4],
            "rows": [[g["group"], _fmt(g["sigma_px"] if g["sigma_px"] is not None else g["sigma_estimate_px"], 4),
                      _fmt(100 * g["rel_se"], 1, " %") if g.get("rel_se") is not None else "–",
                      f"{g['n_used']} / {g['n_total']}", _fmt(g.get("residual_rms_px_per_coordinate"), 4)]
                     for g in nz["groups"]],
            "note": ("Weights " + ("applied: " if nz.get("weights_used") else "not applied: ") + nz.get("why", "")
                     + ". Noise sigma: each group's own detection noise; residual RMS: what is left after the fit.")}


# ------------------------------------------------------------------ flat windows (PINPLATE)
def _hints_text(hints) -> str:
    """The window-normal hints of a flat-window run, in a few words."""
    if not hints:
        return "none"
    n = sum(1 for v in hints.get("normals") or [] if v is not None)
    return f"{hints.get('source', 'given')}: window normals for {n} camera(s)"


def _constraints_text(cons, report) -> str:
    """A flat-window run's constraints (rig indices, 1-based here) and, from plate_wand.constraint_report, how they hold
    at the answer."""
    if not cons:
        return "none"
    parts = []
    ch = cons.get("channel")
    if ch:
        widths = ", ".join(f"{k + 1}: {w:g} mm" for k, w in enumerate(ch.get("width_mm") or []) if w is not None)
        balls = ch.get("ball_radius_mm", 0.0)
        balls = "/".join(f"{b:g}" for b in (balls if isinstance(balls, (list, tuple)) else [balls]))
        brk = ((report or {}).get("channel") or {}).get("max_breach_mm")
        parts.append(f"channel (inside width per window {widths}; tolerance {ch.get('tolerance_mm', 0):g} mm; balls "
                     f"{balls} mm" + (f"; largest breach {brk:.3f} mm" if brk is not None else "") + ")")
    held = {(p.get("kind"), tuple(np.atleast_1d(p.get("cam", [])).tolist()), p.get("wall")): p
            for p in (report or {}).get("priors") or []}
    for a in cons.get("aim") or []:
        p = held.get(("aim", tuple(a["cams"]), a["wall"]))
        parts.append(f"aim of camera(s) {', '.join(str(c + 1) for c in a['cams'])} at {a['mm']:g} mm behind window "
                     f"{a['wall'] + 1}" + (f" ({p['value']:.2f} mm at the answer)" if p else ""))
    for d in cons.get("distance") or []:
        parts.append(f"camera {d['cam'] + 1} at {d['mm']:g} mm from its window")
    return "; ".join(parts)


def report_from_plate(result: Dict, data: Dict, folder: str = None) -> Dict:
    """The report of a flat-window wand calibration: plate_wand_gui.write_outputs's result dictionary
    (plate_wand_io.write_result) and calibration_data.json (plate_wand_gui.plot_data)."""
    cams_res = result["cameras"]
    names = [c["name"] for c in cams_res]
    wins = result["windows"]
    cameras = []
    for c in cams_res:
        s = c.get("sigma") or {}
        rr = c.get("residuals") or {}
        w = c.get("window")
        extra = []
        if w is not None and w >= 0:
            extra = [["Window", f"{w + 1}" + (f" ({wins[w]['name']})" if wins[w].get("name") else "")],
                     ["Distance to the window", _fmt(c.get("distance_to_window_mm"), 1, " mm")],
                     ["Angle to the window's normal", _fmt(c.get("incidence_deg"), 1, "°")]]
        cameras.append({"name": c["name"], "f_px": c["f_px"], "f_mm": c.get("f_mm"), "f_sigma_rel": s.get("f_rel"),
                        "k1": (c.get("dist") or [None])[0] if result.get("fit_k1") else None, "cx": c["cx"],
                        "cy": c["cy"], "width": c["width"], "height": c["height"], "centre": c["centre_mm"],
                        "R": c["R"], "rms_px": rr.get("rms_px"), "n_used": rr.get("n_active"),
                        "n_total": rr.get("n_total"), "extra": extra})
    A = np.asarray(data["ends_A_mm"], float).reshape(-1, 3)
    B = np.asarray(data["ends_B_mm"], float).reshape(-1, 3)
    act = np.asarray(data.get("frame_active", [True] * len(A)), bool)
    keep = np.flatnonzero(act)
    pts_idx = keep[_every(len(keep), MAX_POINTS // 2)]
    points = np.vstack([A[pts_idx], B[pts_idx]]).round(4).tolist()
    mid = 0.5 * (A + B)
    centroid = mid[act].mean(axis=0) if act.any() else np.zeros(3)
    port_windows = []
    for w in wins:
        n = np.asarray(w["normal"], float)
        point = centroid - (float(n @ centroid) - float(w["d_mm"])) * n        # the centroid's foot on the camera side
        port_windows.append({"name": w.get("name") or "", "point": point.tolist(), "normal": n.tolist(),
                             "thickness": float(sum(w["thickness_mm"])), "n": w["n"]})
    n1 = np.asarray(wins[0]["normal"], float)
    depth = mid @ n1 - float(wins[0]["far_d_mm"])
    Lfree = np.array([np.nan if v is None else v for v in data["wand_length_free_mm"]], float)
    L = float(result["wand_length_mm"])
    ok = act & np.isfinite(Lfree)
    wand = {"x": depth[ok].round(4).tolist(), "err_um": ((Lfree[ok] - L) * 1e3).round(4).tolist(),
            "frame": [data["frame_ids"][i] for i in np.flatnonzero(ok)],
            "x_label": "Depth of the wand's centre behind window 1 (mm)", "x_short": "depth (mm)",
            "lead": "Plotted against the wand's depth behind window 1: a trend with depth means the window's "
                    "thickness or the indices are off."}
    w_stats = result["wand_length_free_triangulation"]
    gui = result.get("gui", {}).get("settings", {})
    settings = [["Wand length", f"{L:.4f} mm"],
                ["Windows", "; ".join(f"{k + 1}: " + ", ".join(f"{t:.3f}" for t in w["thickness_mm"]) + " mm, n "
                                      + ", ".join(f"{v:.4f}" for v in w["n"]) for k, w in enumerate(wins))],
                ["Cameras per window", "; ".join(f"{k + 1}: " + ", ".join(names[c] for c in w.get("cameras", []))
                                                 for k, w in enumerate(wins))],
                ["Lens terms", ", ".join((["k1"] if result.get("fit_k1") else []) + list(result.get("fit_lens") or []))
                 or "none (k1 held at 0)"],
                ["Noise weights", str(gui.get("noise_weights", "camera"))],
                ["Start", str(gui.get("start") or ("last result" if gui.get("refit_from_last_result") else "search"))],
                ["Starts carried", str(gui.get("n_final", 2)) + (" (refit from the last result)"
                                                                  if gui.get("refit_from_last_result") else "")],
                ["Window-normal hints", _hints_text(gui.get("hints"))],
                ["Constraints", _constraints_text(gui.get("constraints"), result.get("info", {}).get("constraints"))],
                *_detection_rows(gui.get("detection"), gui.get("detect_mode")),
                ["Points from", str(gui.get("source") or "-")],
                ["Frame", result.get("frame", "")],
                ["Run time", _fmt(result.get("gui", {}).get("seconds"), 0, " s")]]
    files = [{"label": c["name"], "path": os.path.join(folder or "", c["camfile"])} for c in cams_res
             if c.get("camfile")]
    files += [{"label": n, "path": os.path.join(folder or "", n)} for n in
              ("calibration_result.json", "calibration_report.txt", "calibration_data.json", "calibration_report.html")]
    return {
        "schema": 1, "title": "Wand calibration through flat windows", "model": "PINPLATE",
        "model_label": "Pinhole + flat windows (PINPLATE)",
        "created": result.get("gui", {}).get("written") or time.strftime("%Y-%m-%d %H:%M"), "folder": folder,
        "summary": {"cameras": len(cams_res), "frames": int(act.sum()), "frames_total": len(act),
                    "detections": result["observations"], "detections_used": result["active"],
                    "rms_px": result["rms_px"], "wand_length_mm": L, "wand_rms_um": w_stats["rms_mm"] * 1e3,
                    "wand_mean_um": w_stats["mean_mm"] * 1e3, "wand_frames": w_stats["frames"]},
        "checks": result.get("checks", []), "cameras": cameras,
        "port": {"type": "windows", "windows": port_windows}, "up": _up_of(cams_res), "points": points,
        "sight_lines": data.get("sight_lines_mm", []), "residuals": _residuals_from_data(data, names),
        "error_pdf": _error_pdf(*_errors_from_data(data, names)),
        "wand": wand, "noise": _noise_table(result.get("noise") or {}), "settings": settings, "files": files,
        "top_lead": "Looking down from above: each camera, its lines of sight bent at its window, and the wand ends. "
                    "Left: the whole rig; right: the measurement volume. Hover a camera for its numbers.",
        "cameras_lead": "The frame is plate_wand's: Z along window 1's normal (camera to water), the origin at the "
                        "wand's mean position.",
        "footer": f"Model PINPLATE; {len(cams_res)} cameras, {len(wins)} windows.",
    }


# ------------------------------------------------------------------ the pinhole calibration
def report_from_pinhole(final_params: Dict, wand_points: Dict, wand_length: float, per_frame_errors: Dict = None,
                        folder: str = None, detect_mode: str = None, source: str = "",
                        detection: Dict = None) -> Dict:
    """The report of the GUI's pinhole wand calibration (WandCalibrator.final_params: {cam: {R, T, K, dist}}), with
    the wand points it used ({frame: {cam: [ball A, ball B]}}).  The wand ends are triangulated here (linear, from
    every camera, distortion removed), so the wand length errors are free ones; the residuals are those of these
    points.  detection: the points' detection record (wand_calibrator.save_detection_record), listed in the settings
    table."""
    import cv2
    cids = sorted(int(c) for c in final_params)
    P = {}
    for c in cids:
        p = final_params[c]
        K = np.asarray(p["K"], float).reshape(3, 3)
        dist = np.asarray(p.get("dist", np.zeros(5)), float).reshape(-1)
        R = np.asarray(p["R"], float).reshape(3, 3)
        T = np.asarray(p.get("T", p.get("tvec")), float).reshape(3)
        P[c] = (K, dist, R, T)
    frames = sorted(int(f) for f in wand_points)

    def ends_of(pts):
        by = {}
        for q in pts:
            tags = [v for v in list(q) if isinstance(v, str)]
            if tags:
                by.setdefault(tags[0], q)
        if "Filtered_Small" in by and "Filtered_Large" in by:
            return by["Filtered_Small"], by["Filtered_Large"]
        two = sorted(list(pts)[:2], key=lambda q: float(q[2]) if len(q) > 2 else 0.0)
        return (two[0], two[1]) if len(two) == 2 else None

    rows = []                                      # (frame, cam, end, u, v)
    for f in frames:
        for c, pts in wand_points[f].items():
            if int(c) not in P:
                continue
            e = ends_of(pts)
            if e is None:
                continue
            for k in (0, 1):
                rows.append((f, int(c), k, float(e[k][0]), float(e[k][1])))
    # triangulate each end (DLT on undistorted normalised points)
    norm_xy = {}
    for (f, c, k, u, v) in rows:
        K, dist, R, T = P[c]
        xy = cv2.undistortPoints(np.array([[[u, v]]], float), K, dist).reshape(2)
        norm_xy[(f, c, k)] = xy
    X = {}
    for f in frames:
        for k in (0, 1):
            A = []
            for c in cids:
                xy = norm_xy.get((f, c, k))
                if xy is None:
                    continue
                _K, _d, R, T = P[c]
                M = np.hstack([R, T.reshape(3, 1)])
                A.append(xy[0] * M[2] - M[0])
                A.append(xy[1] * M[2] - M[1])
            if len(A) >= 4:
                _u, _s, vt = np.linalg.svd(np.asarray(A))
                h = vt[-1]
                if abs(h[3]) > 1e-12:
                    X[(f, k)] = h[:3] / h[3]
    # residuals
    res = {c: [] for c in cids}
    for (f, c, k, u, v) in rows:
        if (f, k) not in X:
            continue
        K, dist, R, T = P[c]
        q, _j = cv2.projectPoints(X[(f, k)].reshape(1, 3), cv2.Rodrigues(R)[0], T, K, dist)
        q = q.reshape(2)
        res[c].append((f, k, u, v, float(q[0] - u), float(q[1] - v)))
    cameras, residuals = [], []
    for i, c in enumerate(cids):
        K, dist, R, T = P[c]
        C = -R.T @ T
        rr = res[c]
        rn = np.array([math.hypot(r[4], r[5]) for r in rr]) if rr else np.zeros(0)
        p = final_params[c]
        size = p.get("img_size") or p.get("image_size")
        w_px, h_px = (int(size[1]), int(size[0])) if size is not None else (int(2 * K[0, 2] + 1), int(2 * K[1, 2] + 1))
        cameras.append({"name": f"cam{c}", "f_px": float(K[0, 0]), "f_mm": None, "f_sigma_rel": None,
                        "k1": float(dist[0]) if dist.size and dist[0] != 0 else None, "cx": float(K[0, 2]),
                        "cy": float(K[1, 2]), "width": w_px, "height": h_px, "centre": C.tolist(), "R": R.tolist(),
                        "rms_px": float(np.sqrt(np.mean(rn ** 2))) if rn.size else None, "n_used": len(rr),
                        "n_total": len(rr), "extra": [["Distance to the volume's centre", "–"]]})
        sel = _every(len(rr), MAX_RESIDUALS)
        residuals.append({"cam": i, "u": [rr[j][2] for j in sel], "v": [rr[j][3] for j in sel],
                          "du": [rr[j][4] for j in sel], "dv": [rr[j][5] for j in sel], "used": [True] * len(sel),
                          "frame": [rr[j][0] for j in sel], "end": [rr[j][1] for j in sel],
                          "subsampled": len(sel) < len(rr)})
    good = [f for f in frames if (f, 0) in X and (f, 1) in X]
    XA = np.array([X[(f, 0)] for f in good]).reshape(-1, 3)
    XB = np.array([X[(f, 1)] for f in good]).reshape(-1, 3)
    mid = 0.5 * (XA + XB)
    centre = mid.mean(axis=0) if len(mid) else np.zeros(3)
    for cam in cameras:
        cam["extra"] = [["Distance to the volume's centre",
                         _fmt(float(np.linalg.norm(np.asarray(cam["centre"]) - centre)), 1, " mm")]]
    Lm = np.linalg.norm(XB - XA, axis=1) if len(good) else np.zeros(0)
    err_um = (Lm - float(wand_length)) * 1e3
    rad = np.linalg.norm(mid - centre, axis=1) if len(good) else np.zeros(0)
    pts_idx = _every(len(good), MAX_POINTS // 2)
    allr = np.array([math.hypot(r[4], r[5]) for c in cids for r in res[c]]) if rows else np.zeros(0)
    rms = float(np.sqrt(np.mean(allr ** 2))) if allr.size else float("nan")
    checks = []
    if detect_mode is None:
        checks.append({"level": "note", "title": "Detection mode unknown", "detail": "These wand points were "
                       "loaded from a file, so it is not known how the balls were found.", "what_to_do": "",
                       "evidence": ""})
    if np.isfinite(rms) and rms > 1.0:
        checks.append({"level": "warning", "title": f"Large residuals: RMS {rms:.2f} px", "detail": "The cameras do "
                       "not fit the detections well: wrong detections, a moving camera, or refraction the pinhole "
                       "model does not have.", "what_to_do": "Look at the frames with large errors in the table.",
                       "evidence": ""})
    if len(good) < 200:
        checks.append({"level": "note", "title": f"{len(good)} frames used", "detail": "The precision grows with "
                       "the number of frames.", "what_to_do": "Record more frames over the whole volume.",
                       "evidence": ""})
    sight = []
    for i, cam in enumerate(cameras):
        Cc = np.asarray(cam["centre"], float)
        R = np.asarray(cam["R"], float)
        dist_c = float(np.linalg.norm(centre - Cc))
        for (u, v) in ((cam["cx"], cam["cy"]), (0, 0), (cam["width"] - 1, 0), (cam["width"] - 1, cam["height"] - 1),
                       (0, cam["height"] - 1)):
            d = R.T @ np.array([(u - cam["cx"]) / cam["f_px"], (v - cam["cy"]) / cam["f_px"], 1.0])
            d /= np.linalg.norm(d)
            sight.append([i, [Cc.tolist(), (Cc + 1.15 * dist_c * d).tolist()]])
    return {
        "schema": 1, "title": "Wand calibration", "model": "PINHOLE", "model_label": "Pinhole",
        "created": time.strftime("%Y-%m-%d %H:%M"), "folder": folder,
        "summary": {"cameras": len(cids), "frames": len(good), "frames_total": len(frames),
                    "detections": len(rows), "detections_used": len(rows), "rms_px": rms,
                    "wand_length_mm": float(wand_length),
                    "wand_rms_um": float(np.sqrt(np.mean(err_um ** 2))) if err_um.size else None,
                    "wand_mean_um": float(np.mean(err_um)) if err_um.size else None, "wand_frames": len(good)},
        "checks": checks, "cameras": cameras, "port": None, "up": _up_of(cameras),
        "points": np.vstack([XA[pts_idx], XB[pts_idx]]).round(4).tolist() if len(good) else [],
        "sight_lines": sight, "residuals": residuals,
        "error_pdf": _error_pdf([np.array([(r[4], r[5]) for r in res[c]], float).reshape(-1, 2) for c in cids],
                                [len(res[c]) for c in cids]),
        "wand": {"x": rad.round(4).tolist(), "err_um": err_um.round(4).tolist(), "frame": good,
                 "x_label": "Distance of the wand's centre from the volume's centre (mm)", "x_short": "from the centre (mm)",
                 "lead": "Plotted against the distance from the middle of the volume: errors growing outwards point "
                         "to lens distortion the model lacks."},
        "noise": None,
        "settings": [["Wand length", f"{float(wand_length):.4f} mm"],
                     *_detection_rows(detection, detect_mode),
                     ["Points from", str(source or "-")],
                     ["Wand ends", "triangulated for this page (linear, all cameras, distortion removed)"]],
        "files": [{"label": "camFile folder", "path": folder}] if folder else [],
        "top_lead": "Looking down from above: each camera, the edges of its field of view, and the wand ends. Left: "
                    "the whole rig; right: the measurement volume. Hover a camera for its numbers.",
        "footer": f"Model PINHOLE; {len(cids)} cameras.",
    }


def summary_html(report: Dict) -> str:
    """A short rich-text summary (for the GUI's Report tab when it cannot show the page itself)."""
    S = report.get("summary", {})
    rows = "".join(f"<tr><td>{c['name']}</td><td align='right'>{_fmt(c['f_px'], 0)}</td>"
                   f"<td align='right'>{_fmt(c.get('rms_px'), 3)}</td></tr>" for c in report.get("cameras", []))
    checks = "".join(f"<li>{'Warning' if ck['level'] == 'warning' else 'Note'}: {ck['title']}</li>"
                     for ck in report.get("checks", []))
    return (f"<h3>{report.get('title', '')}</h3><p>{report.get('model_label', '')} · {report.get('created', '')}</p>"
            f"<p>{S.get('cameras', '')} cameras, {S.get('frames', '')} frames; reprojection RMS "
            f"{_fmt(S.get('rms_px'), 3)} px; wand length error {_fmt(S.get('wand_rms_um'), 2)} µm RMS.</p>"
            f"<table cellpadding='4'><tr><th align='left'>Camera</th><th>f (px)</th><th>RMS (px)</th></tr>{rows}</table>"
            + (f"<ul>{checks}</ul>" if checks else ""))
