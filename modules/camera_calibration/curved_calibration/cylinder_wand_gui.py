"""The GUI's cylinder wand calibration (camera model "Pinhole+Cylinder"): from the GUI's wand points to PINCYL camFiles.

No Qt here, so that it can be tested alone (tests/cylinder_wand_gui).  The GUI (modules/camera_calibration/view.py)
collects the settings into a `Job`, runs `run` (or `precheck`) in a worker thread, writes the outputs with
`write_outputs` and fills its error table and frame overlays from `per_frame_errors`:

    job = Job(uv, frame_ids, cam_ids, sizes, wand_length, radii, n)   # uv from uv_from_wand_points(...)
    sol = run(job, progress=..., stop=...)       # cylinder_wand.calibrate; Stop raises Stopped
    result = write_outputs(sol, job, folder)     # camFile/cam<id>.txt, calibration_result.json, ..._report.txt,
                                                 # calibration_data.json (residuals and wand ends, for the page)
    table = per_frame_errors(sol, job)           # {frame id: {'cam_errors', 'len_error', 'proj_pts'}}

The calibration itself is cylinder_wand.calibrate, unchanged (tests/cylinder_wand): no camera input; f, pose and
optionally k1 per camera, from the wand alone.  The frame of the outputs is the tube's: the axis is Y through the
origin, Z points towards the first camera, and the origin sits at the wand's mean height (cylinder_wand_io).
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

try:
    from . import cyl_model as cm
    from . import cylinder_wand as cw
    from . import cylinder_wand_io as cio
except ImportError:  # imported as a top-level module (tests put this folder on sys.path)
    import cyl_model as cm
    import cylinder_wand as cw
    import cylinder_wand_io as cio

# the GUI's choice of noise weights -> cylinder_wand's noise_weights
NOISE_WEIGHTS = {"Per camera": "camera", "Per camera and ball": "camera_end", "Equal": "none"}
MIN_WHOLE_FRAMES = 12          # cylinder_wand.affine_factorization: frames seen whole by every camera


class Stopped(Exception):
    """Raised inside the calibration when Stop was pressed."""


# ------------------------------------------------------------------ inputs
def tube_from_settings(outer_diameter_mm: float, wall_mm: float, n_outside: float, n_wall: float,
                       n_inside: float) -> cm.Tube:
    """The tube of the GUI's settings: one wall between the camera's medium and the liquid."""
    R1 = 0.5 * float(outer_diameter_mm)
    if not float(wall_mm) > 0:
        raise ValueError("The wall thickness must be positive.")
    R2 = R1 - float(wall_mm)
    if not R2 > 0:
        raise ValueError(f"The wall ({float(wall_mm):g} mm) is not thinner than the tube's outer radius "
                         f"({R1:g} mm).")
    for name, v in (("camera side", n_outside), ("wall", n_wall), ("inside", n_inside)):
        if not float(v) >= 1.0:
            raise ValueError(f"The refractive index of the {name} must be at least 1 (got {float(v):g}).")
    return cm.Tube((R1, R2), (float(n_outside), float(n_wall), float(n_inside)))


def _ends(pts):
    """The two detections of one camera in one frame as (end A, end B): by the GUI's labels (Filtered_Small = A,
    Filtered_Large = B) when present, otherwise the smaller radius first.  None if there are not two."""
    if pts is None or len(pts) < 2:
        return None
    by_label = {}
    for p in pts:
        tags = [v for v in list(p) if isinstance(v, str)]
        if tags:
            by_label.setdefault(tags[0], p)
    if "Filtered_Small" in by_label and "Filtered_Large" in by_label:
        return by_label["Filtered_Small"], by_label["Filtered_Large"]
    two = sorted(list(pts)[:2], key=lambda p: float(p[2]) if len(p) > 2 else 0.0)
    return two[0], two[1]


def uv_from_wand_points(wand_points: Dict, cam_ids: Sequence[int] = None):
    """The GUI's wand points {frame: {camera: [ball A, ball B]}} (WandCalibrator.wand_points or its filtered copy)
    as uv[frame, camera, end, (u, v)], NaN where a camera does not see a frame.  End A is the small ball.
    Returns (uv, frame_ids, cam_ids), frames and cameras sorted (cam_ids fixes the cameras and their order)."""
    frames = sorted(int(f) for f in wand_points)
    cams = sorted({int(c) for f in wand_points for c in wand_points[f]}) if cam_ids is None else [int(c) for c in cam_ids]
    cpos = {c: k for k, c in enumerate(cams)}
    uv = np.full((len(frames), len(cams), 2, 2), np.nan)
    for i, f in enumerate(frames):
        for c, pts in wand_points[f].items():
            if int(c) not in cpos:
                continue
            ends = _ends(pts)
            if ends is None:
                continue
            for e, p in enumerate(ends):
                uv[i, cpos[int(c)], e] = (float(p[0]), float(p[1]))
    return uv, frames, cams


@dataclass
class Job:
    """Everything one calibration needs, as the GUI collected it."""

    uv: np.ndarray                         # (F, Nc, 2, 2), NaN = not seen; end 0 = A (small ball)
    frame_ids: List[int]                   # (F,) the GUI's frame numbers
    cam_ids: List[int]                     # (Nc,) the GUI's camera numbers
    sizes: List[tuple]                     # (width, height) per camera
    wand_length: float                     # mm
    radii: tuple                           # outer first, mm
    n: tuple                               # camera side first
    fit_k1: bool = False
    fit_L: bool = False
    noise_weights: str = "camera"          # cylinder_wand's: "camera", "camera_end" or "none"
    n_final: int = 2                       # starts carried through stages 3-4 (6: the thorough search)
    start: Optional[list] = None           # cameras of an earlier result (in the fit's gauge): skip the start search
    pixel_mm: Optional[float] = None       # pixel size, only to print f in mm
    detect_mode: Optional[str] = None      # how the points were found ("ellipse", "fast", "reliable"; None: unknown)
    source: str = ""                       # where the points came from (for the report)

    @property
    def tube(self) -> cm.Tube:
        return cm.Tube(tuple(self.radii), tuple(self.n))

    @property
    def names(self) -> List[str]:
        return [f"cam{c}" for c in self.cam_ids]

    def data(self) -> cw.WandData:
        return cw.WandData(np.asarray(self.uv, float), float(self.wand_length),
                           [(int(w), int(h)) for (w, h) in self.sizes])

    def settings(self) -> Dict:
        """The settings, for the result file (everything but the detections and the start)."""
        return {"wand_length_mm": float(self.wand_length), "radii_mm": [float(r) for r in self.radii],
                "n": [float(v) for v in self.n], "fit_k1": bool(self.fit_k1), "fit_L": bool(self.fit_L),
                "noise_weights": self.noise_weights, "n_final": int(self.n_final),
                "refit_from_last_result": self.start is not None, "pixel_mm": self.pixel_mm,
                "detect_mode": self.detect_mode, "source": self.source,
                "frame_ids": [int(f) for f in self.frame_ids], "cam_ids": [int(c) for c in self.cam_ids],
                "sizes": [[int(w), int(h)] for (w, h) in self.sizes]}

    def check(self) -> None:
        """Raise ValueError with a message for the user when the inputs cannot work."""
        uv = np.asarray(self.uv, float)
        if uv.ndim != 4 or uv.shape[2:] != (2, 2):
            raise ValueError("The wand points have the wrong shape.")
        F, Nc = uv.shape[:2]
        if Nc < 2:
            raise ValueError(f"The cylinder calibration needs at least two cameras; the wand points have {Nc}.")
        if len(self.sizes) != Nc or any(int(w) <= 0 or int(h) <= 0 for (w, h) in self.sizes):
            raise ValueError("Every camera needs its image width and height (camera table).")
        if not float(self.wand_length) > 0:
            raise ValueError("The wand length must be positive.")
        self.tube                                                        # raises on a bad tube
        seen = np.all(np.isfinite(uv), axis=3)                           # (F, Nc, 2)
        whole = int(np.sum(seen.all(axis=(1, 2))))
        if self.start is None and whole < MIN_WHOLE_FRAMES:
            raise ValueError(f"Only {whole} frames show both balls in every camera; the calibration's start needs "
                             f"at least {MIN_WHOLE_FRAMES} (several hundred are better).")
        usable = int(np.sum(np.all(seen.sum(axis=1) >= 2, axis=1)))
        if usable < MIN_WHOLE_FRAMES:
            raise ValueError(f"Only {usable} frames show both balls in two cameras or more.")


# ------------------------------------------------------------------ running
_RMS = re.compile(r"rms ([0-9]+(?:\.[0-9]*)?(?:[eE][-+]?[0-9]+)?) px")


def _stage_of(text: str, current: str) -> str:
    t = text.strip()
    if t.startswith("candidate") or t.startswith("start search:"):
        return "Start search (stages 1-2)"
    if t.startswith("coarse"):
        return "Coarse fit (stage 3)"
    if t.startswith("LM") or t.startswith("round"):
        return "Final fit (stage 4)"
    return current


def _logger(progress: Optional[Callable], stop: Optional[Callable]):
    """cylinder_wand's log function: each line goes to progress(stage, rms_px or nan, text); Stop raises Stopped."""
    state = {"stage": "Start search (stages 1-2)"}

    def log(msg):
        if stop is not None and stop():
            raise Stopped()
        text = str(msg)
        state["stage"] = _stage_of(text, state["stage"])
        m = _RMS.search(text)
        # the RMS of the start search's many candidates is not a progress curve: only stages 3-4 report one
        rms = float(m.group(1)) if m and not state["stage"].startswith("Start search") else float("nan")
        if progress is not None:
            progress(state["stage"], rms, text.strip())
    return log


def run(job: Job, progress: Callable = None, stop: Callable = None) -> cw.Solution:
    """The calibration (cylinder_wand.calibrate, stages 0-5).  progress(stage, rms_px, text) gets one call per log
    line; stop() is polled at every line and raises Stopped when it returns True."""
    job.check()
    data, tube = job.data(), job.tube
    log = _logger(progress, stop)
    start = job.start if job.start is not None and len(job.start) == data.n_cams else None
    if progress is not None:
        progress("Refit from the last result (stages 3-4)" if start is not None else "Start search (stages 1-2)",
                 float("nan"), f"{data.n_frames} frames, {data.n_cams} cameras")
    if stop is not None and stop():
        raise Stopped()
    sol = cw.calibrate(data, tube, start=start, fit_k1=bool(job.fit_k1), fit_L=bool(job.fit_L),
                       n_final=int(job.n_final), noise_weights=job.noise_weights, verbose=True, log=log, log_fits=True)
    if stop is not None and stop():
        raise Stopped()
    return sol


def precheck(job: Job, progress: Callable = None, stop: Callable = None) -> cw.Solution:
    """Stages 1-3 only (the start from the data and the coarse fit), for the GUI's "Precalibrate to Check": a
    Solution without covariance whose residuals show the data's quality before the full calibration."""
    job.check()
    data, tube = job.data(), job.tube
    log = _logger(progress, stop)
    obs = cw.observations(data)
    if progress is not None:
        progress("Start search (stages 1-2)", float("nan"), f"{obs.n_frames} frames, {data.n_cams} cameras")
    ranked, binfo = cw.bootstrap(obs, tube, data.L, data.sizes, verbose=True, log=log, log_fits=True)
    layout = cw.Layout(data.n_cams, fit_k1=bool(job.fit_k1))
    cams3, info3 = cw.coarse_fit(ranked[0][1], tube, obs, data.L, layout, verbose=True, log=log)
    cams3, _fr, _flip = cw.canonical(cams3)
    frames = cw.initial_frames(cams3, tube, obs, data.L)
    sol = cw.Solution(cams=cams3, tube=tube, frames=frames, L=data.L, obs=obs, layout=layout)
    ev = cw.evaluate(sol.cams, tube, frames, data.L, obs)
    ok = ev.ok & np.all(np.isfinite(ev.r), axis=1)
    sol.rms = float(np.sqrt(np.mean(np.sum(ev.r[ok] ** 2, axis=1)))) if ok.any() else float("nan")
    sol.info = {"precheck": True, "bootstrap": binfo, "stage3": info3}
    return sol


# ------------------------------------------------------------------ results for the GUI
def _rows_by_frame(obs: cw.Obs):
    """Row indices of each kept frame (obs rows are sorted by frame)."""
    starts = np.searchsorted(obs.frame, np.arange(obs.n_frames + 1))
    return [np.arange(starts[k], starts[k + 1]) for k in range(obs.n_frames)]


def residuals(sol: cw.Solution):
    """(r, ok, projections): each detection's reprojection residual (model - detection, px), whether its point
    projects, and where the model puts it."""
    obs = sol.obs
    ev = cw.evaluate(sol.cams, sol.tube, sol.frames, sol.L, obs)
    ok = ev.ok & np.all(np.isfinite(ev.r), axis=1)
    return ev.r, ok, obs.uv + np.where(ok[:, None], ev.r, np.nan)


def free_wand_lengths(sol: cw.Solution, ok: np.ndarray = None) -> np.ndarray:
    """Each kept frame's wand length from a free triangulation of all its projecting detections (NaN when an end
    has fewer than two)."""
    if ok is None:
        _r, ok, _p = residuals(sol)
    return cw.wand_lengths(sol.cams, sol.tube, sol.obs, rows=np.flatnonzero(ok))


def per_frame_errors(sol: cw.Solution, job: Job) -> Dict[int, Dict]:
    """The GUI's per-frame table (WandCalibrator.per_frame_errors): for each frame id, per camera id the larger
    reprojection error of the two balls (px), the wand-length error of a free triangulation (mm, absolute), and
    where the model puts the two balls (proj_pts, for the frame overlay).  Rejected detections are included, so
    that they can be found and deleted."""
    obs = sol.obs
    r, ok, proj = residuals(sol)
    norm = np.linalg.norm(np.where(ok[:, None], r, np.nan), axis=1)
    Lm = free_wand_lengths(sol, ok)
    out = {}
    for k, rows in enumerate(_rows_by_frame(obs)):
        fid = int(job.frame_ids[int(obs.frame_ids[k])])
        cam_err, proj_pts = {}, {}
        for c in np.unique(obs.cam[rows]):
            rr = rows[obs.cam[rows] == c]
            cid = int(job.cam_ids[int(c)])
            good = rr[ok[rr]]
            cam_err[cid] = float(np.max(norm[good])) if len(good) else float("nan")
            pa = pb = None
            for i in rr:
                if ok[i]:
                    if obs.end[i] == 0:
                        pa = [float(proj[i, 0]), float(proj[i, 1])]
                    else:
                        pb = [float(proj[i, 0]), float(proj[i, 1])]
            proj_pts[cid] = (pa, pb)
        le = abs(float(Lm[k]) - float(sol.L)) if np.isfinite(Lm[k]) else float("nan")
        out[fid] = {"cam_errors": cam_err, "len_error": le, "proj_pts": proj_pts,
                    "active": bool(np.any(obs.active[rows]))}
    return out


def _outer_hit(C: np.ndarray, d: np.ndarray, R: float) -> Optional[np.ndarray]:
    """Where the ray C + t d (t > 0) first meets the cylinder x^2 + z^2 = R^2 about the Y axis."""
    a = d[0] ** 2 + d[2] ** 2
    b = 2.0 * (C[0] * d[0] + C[2] * d[2])
    c = C[0] ** 2 + C[2] ** 2 - R * R
    disc = b * b - 4 * a * c
    if a <= 0 or disc < 0:
        return None
    t = (-b - math.sqrt(disc)) / (2 * a)
    return C + t * d if t > 0 else None


def sight_lines(cams: Sequence[cm.Camera], tube: cm.Tube, depth_mm: float = None, pixels: str = "centre") -> List:
    """Each camera's lines of sight as polylines through the tube, in the cameras' frame: from the centre to the
    outer wall (straight in air), across the wall to the inner wall, then into the liquid up to `depth_mm` past the
    axis (default: the inner radius).  pixels: "centre" (the principal point) or "corners" (the centre and the four
    image corners).  Returns [[camera index, [[x, y, z], ...]], ...]; the last two points of each line are the
    ray of cyl_model.trace (exact)."""
    depth = tube.r_inner if depth_mm is None else float(depth_mm)
    out = []
    for k, cam in enumerate(cams):
        uv = [(cam.cx, cam.cy)]
        if pixels == "corners":
            uv += [(0.0, 0.0), (cam.width - 1.0, 0.0), (cam.width - 1.0, cam.height - 1.0), (0.0, cam.height - 1.0)]
        uv = np.asarray(uv, float)
        rw = cm.rows([cam], np.zeros(len(uv), int))
        xy = cm.undistort(uv, rw)
        o, d, ok, _do, _dd = cm.trace(rw, xy, tube)
        for j in range(len(uv)):
            if not ok[j]:
                continue
            v = cam.R.T @ np.array([xy[j, 0], xy[j, 1], 1.0])
            p_out = _outer_hit(cam.C, v / np.linalg.norm(v), tube.r_outer)
            if p_out is None:
                continue
            # the last leg ends `depth` past its closest approach to the axis (seen along the axis)
            dxz = math.hypot(d[j, 0], d[j, 2])
            if dxz < 1e-9:
                continue
            t_end = float(-(o[j, 0] * d[j, 0] + o[j, 2] * d[j, 2]) / dxz ** 2 + depth / dxz)
            pts = [cam.C, p_out, o[j], o[j] + max(t_end, 0.0) * d[j]]
            out.append([k, [[float(x) for x in p] for p in pts]])
    return out


def plot_data(sol: cw.Solution, job: Job) -> Dict:
    """What the GUI's 3-D view and the results page draw, in the outputs' frame (cylinder_wand.relabel)."""
    obs = sol.obs
    cams_out, frames_out, y0 = cw.relabel(sol.cams, sol.frames)
    XA, XB = frames_out.ends(sol.L)
    r, ok, _proj = residuals(sol)
    Lm = free_wand_lengths(sol, ok)
    fid = [int(job.frame_ids[int(i)]) for i in obs.frame_ids]
    frame_active = np.zeros(obs.n_frames, bool)
    frame_active[obs.frame[obs.active]] = True
    return {
        "frame": "the outputs' frame: tube axis = Y through the origin, Z towards the first camera",
        "origin_shift_along_axis_mm": float(y0),
        "frame_ids": fid,
        "frame_active": frame_active.tolist(),
        "ends_A_mm": np.round(XA, 5).tolist(), "ends_B_mm": np.round(XB, 5).tolist(),
        "wand_length_free_mm": [None if not np.isfinite(v) else round(float(v), 6) for v in Lm],
        "observations": {
            "frame_id": [fid[int(k)] for k in obs.frame], "cam_id": [int(job.cam_ids[int(c)]) for c in obs.cam],
            "end": obs.end.astype(int).tolist(),
            "u": np.round(obs.uv[:, 0], 4).tolist(), "v": np.round(obs.uv[:, 1], 4).tolist(),
            "du": [None if not g else round(float(x), 5) for x, g in zip(r[:, 0], ok)],
            "dv": [None if not g else round(float(x), 5) for x, g in zip(r[:, 1], ok)],
            "active": obs.active.astype(bool).tolist()},
        "sight_lines_mm": sight_lines(cams_out, sol.tube, pixels="corners"),
    }


# ------------------------------------------------------------------ checks
def checks(result: Dict, data: Dict, job: Job) -> List[Dict]:
    """What the tests associate with a wrong or a weak answer, each with its evidence and what to do."""
    out = []

    def add(level, title, detail, todo, evidence):
        out.append({"level": level, "title": title, "detail": detail, "what_to_do": todo, "evidence": evidence})

    mode = job.detect_mode
    if mode is None:
        add("note", "Detection mode unknown",
            "These wand points were loaded from a file, so it is not known how the balls were found.",
            "Through a tube, detect the balls with the 'ellipse' mode.",
            "T13: the circle detectors miss the centres of the 1.3 : 1 images through the tube.")
    elif mode != "ellipse":
        add("warning", f"Balls detected with the '{mode}' mode",
            "Through a tube a ball's image is an ellipse; the circle detectors miss its centre by several pixels.",
            "Detect the balls again with the 'ellipse' mode and recalibrate.",
            "T13b/T13c: with 'fast' or 'reliable', f came out a few per cent to tens of per cent off.")
    if result.get("rms_px", 0.0) > 1.0:
        add("warning", f"Large residuals: RMS {result['rms_px']:.2f} px",
            "Detections this noisy can lead the start search to a wrong answer.",
            "Check the detections in a few frames ('Test Detect'); tick 'Thorough start search'.",
            "T11b: at 5 px one rig in ten took a wrong minimum from the top two starts.")
    for c in result.get("cameras", []):
        res = c.get("residuals") or {}
        tot, act = res.get("n_total", 0), res.get("n_active", 0)
        if tot and act < 0.8 * tot:
            add("warning", f"{c['name']}: {tot - act} of {tot} detections not used",
                "The calibration rejected them as outliers, or dropped their frames.",
                "Look at the frames with large errors in the table (sort by this camera) and check the detections.",
                "cylinder_wand.refine: a detection beyond 5 robust sigmas is rejected.")
    nf = int(sum(1 for a in data.get("frame_active", []) if a))
    if nf < 200:
        add("note", f"{nf} frames used",
            "The precision of f and of the poses grows with the number of frames.",
            "Record more frames, spread over the whole measurement volume.",
            "The tests used 500 frames (tests/cylinder_wand T5).")
    fold = float(result.get("tube", {}).get("fold_radius_mm") or 0.0)
    if fold > 0:
        ends = np.vstack([np.asarray(data.get("ends_A_mm", []), float).reshape(-1, 3),
                          np.asarray(data.get("ends_B_mm", []), float).reshape(-1, 3)])
        rho = np.hypot(ends[:, 0], ends[:, 2]) if len(ends) else np.zeros(0)
        far = int(np.sum(rho > fold))
        if far:
            add("warning", f"{far} wand ends beyond the fold radius ({fold:.1f} mm from the axis)",
                "There a point can have two images (or none), and a detection may belong to the other image.",
                "Keep the wand in the central volume, closer to the axis than the fold radius.",
                "refract_curved.tex, far branch; paper section 9.1.")
    if result.get("fit_L") and result.get("L_sigma_mm"):
        dL = float(result["wand_length_mm"]) - float(job.wand_length)
        if abs(dL) > 3.0 * float(result["L_sigma_mm"]):
            add("warning", f"The fitted wand length differs from the nominal by {dL * 1e3:+.1f} um",
                f"That is {abs(dL) / float(result['L_sigma_mm']):.1f} sigma: the tube's radii and indices fix the "
                "scale, so one of them (or the nominal length) is off.",
                "Check the tube's diameter, wall and the liquid's index, and the wand's length.",
                "tests/cylinder_wand T9: with L free the radii fix the scale to about 0.02 %.")
    return out


# ------------------------------------------------------------------ outputs
def write_outputs(sol: cw.Solution, job: Job, folder: str) -> Dict:
    """camFile/cam<id>.txt (PINCYL, C++ writer), calibration_result.json and calibration_report.txt
    (cylinder_wand_io.write_result), calibration_data.json (plot_data), the checks and the results page
    calibration_report.html (modules/camera_calibration/results_page), in `folder`.  Returns the result dictionary
    (with "checks", and "page": the page's path or None when it could not be written)."""
    os.makedirs(folder, exist_ok=True)
    n = len(sol.cams)
    pixel_mm = [float(job.pixel_mm)] * n if job.pixel_mm else None
    extra = {"gui": {"settings": job.settings(), "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "seconds": sol.info.get("total_seconds")}}
    result = cio.write_result(sol, folder, names=job.names, pixel_mm=pixel_mm, extra=extra)
    data = plot_data(sol, job)
    result["checks"] = checks(result, data, job)
    with open(os.path.join(folder, "calibration_data.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    with open(os.path.join(folder, "calibration_result.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    if result["checks"]:
        with open(os.path.join(folder, "calibration_report.txt"), "a", encoding="utf-8") as fh:
            fh.write("\nChecks\n")
            for ck in result["checks"]:
                fh.write(f"  {ck['level'].upper()}: {ck['title']}. {ck['detail']} What to do: {ck['what_to_do']} "
                         f"({ck['evidence']})\n")
    result["page"] = write_page(result, data, folder)
    return result


def write_page(result: Dict, data: Dict, folder: str) -> Optional[str]:
    """calibration_report.html: the results page (figures with values on hover, a 3-D view; offline)."""
    try:
        try:
            from ..results_page import page as rp
        except (ImportError, ValueError):
            from modules.camera_calibration.results_page import page as rp
        return rp.write_page(rp.report_from_cylinder(result, data, folder),
                             os.path.join(folder, "calibration_report.html"))
    except Exception as e:                     # the calibration's files stand without the page
        print(f"[Cylinder] The results page could not be written: {e}")
        return None


def camfile_paths(folder: str, job: Job) -> List[str]:
    return [os.path.join(folder, "camFile", f"{name}.txt") for name in job.names]


def copy_camfiles(folder: str, names: Sequence[str], out_dir: str) -> int:
    """Copy the result's camFiles (camFile/<name>.txt) into out_dir (the project's camera folder)."""
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for name in names:
        src = os.path.join(folder, "camFile", f"{name}.txt")
        dst = os.path.join(out_dir, f"{name}.txt")
        if os.path.isfile(src):
            if not (os.path.exists(dst) and os.path.samefile(src, dst)):      # the results may be in place already
                shutil.copyfile(src, dst)
            n += 1
    return n


def summary_lines(result: Dict) -> List[str]:
    """A few lines for the GUI's message box."""
    w = result["wand_length_free_triangulation"]
    out = [f"{len(result['cameras'])} cameras, {result['frames']} frames, {result['active']} of "
           f"{result['observations']} detections used.",
           f"Reprojection RMS {result['rms_px']:.3f} px; wand length (free triangulation) RMS error "
           f"{w['rms_mm'] * 1e3:.1f} um over {w['frames']} frames."]
    for c in result["cameras"]:
        s = c.get("sigma") or {}
        out.append(f"  {c['name']}: f {c['f_px']:,.0f} px" + (f" ({c['f_mm']:.1f} mm)" if c.get("f_mm") else "")
                   + (f" +- {100 * s['f_rel']:.2f} %" if s.get("f_rel") else "")
                   + f", {c['distance_from_axis_mm']:.0f} mm from the axis, RMS {c['residuals']['rms_px']:.3f} px")
    for ck in result.get("checks", []):
        out.append(("WARNING: " if ck["level"] == "warning" else "Note: ") + ck["title"])
    return out


def precheck_lines(sol: cw.Solution, job: Job) -> List[str]:
    """A few lines for the GUI's message box after stages 1-3."""
    r, ok, _p = residuals(sol)
    out = [f"Start and coarse fit (stages 1-3) on {sol.obs.n_frames} frames: RMS {sol.rms:.3f} px.",
           "This is only the start; run the calibration for the final answer."]
    norm = np.linalg.norm(np.where(ok[:, None], r, np.nan), axis=1)
    for c, cam in enumerate(sol.cams):
        sel = (sol.obs.cam == c) & ok
        rms = float(np.sqrt(np.mean(norm[sel] ** 2))) if sel.any() else float("nan")
        out.append(f"  cam{job.cam_ids[c]}: f {cam.f:,.0f} px, {math.hypot(cam.C[0], cam.C[2]):.0f} mm from the "
                   f"axis, RMS {rms:.3f} px")
    return out
