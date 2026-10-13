"""The GUI's flat-window wand calibration (camera model "Pinhole+Refraction"): from the GUI's wand points to PINPLATE
camFiles with plate_wand (tests/plate_wand), as cylinder_wand_gui does for the tube.

No Qt here, so that it can be tested alone (tests/cylinder_wand_gui, tests/plate_wand_gui).  The GUI collects the
windows (each window's thickness and plate index; the camera-side and object-side indices), which window each camera
looks through, and the options of the real case (tests/plate_wand_gui: a start from a saved fit, constraints, lens
terms, window-normal hints) into a `Job`, runs `run` in a worker process, writes the outputs with `write_outputs`
and fills its error table and frame overlays from `per_frame_errors`.

The calibration itself is plate_wand.calibrate, unchanged: no camera or window input; each camera's pose, f and
optionally k1, and each window's plane, from the wand alone.  The outputs' frame is plate_wand's (to_output_frame):
Z along window 1's normal (camera -> water), the origin at the wand's mean position.

The camFiles get the same comment block the GUI's previous calibrator wrote ("BEGIN_REFRACTION_META": CAM_ID,
WINDOW_ID, PLANE_PT_EXPORT, PLANE_N): VSC groups the cameras by window with it (modules/vsc/vsc_service.py
_build_refraction_window_map); the C++ loader ignores it.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

try:
    from . import cyl_model as cm
    from . import plate_model as pm
    from . import plate_wand as pw
    from . import plate_wand_io as pio
    from .cylinder_wand_gui import (NOISE_WEIGHTS, Stopped, _logger, _rows_by_frame, copy_camfiles,  # noqa: F401
                                    uv_from_wand_points)
except ImportError:  # imported as a top-level module (tests put this folder on sys.path)
    import cyl_model as cm
    import plate_model as pm
    import plate_wand as pw
    import plate_wand_io as pio
    from cylinder_wand_gui import (NOISE_WEIGHTS, Stopped, _logger, _rows_by_frame, copy_camfiles,  # noqa: F401
                                   uv_from_wand_points)

META_BEGIN = "# --- BEGIN_REFRACTION_META ---"
META_END = "# --- END_REFRACTION_META ---"


@dataclass
class Job:
    """Everything one flat-window calibration needs, as the GUI collected it."""

    uv: np.ndarray                         # (F, Nc, 2, 2), NaN = not seen; end 0 = A (small ball)
    frame_ids: List[int]
    cam_ids: List[int]
    sizes: List[tuple]                     # (width, height) per camera
    wand_length: float                     # mm
    windows: List[Dict]                    # [{"name", "thickness_mm": [t], "n": [n_camera, n_plate, n_object]}]
    cam_to_window: List[int]               # per camera (in cam_ids order), its window's index
    fit_k1: bool = False
    noise_weights: str = "camera"
    n_final: int = 2
    start: Optional[tuple] = None          # (cameras, rig) of an earlier result: skip the start search
    pixel_mm: Optional[object] = None      # pixel size (mm), one number or one per camera: only to print f in mm
    detect_mode: Optional[str] = None
    source: str = ""
    hints: Dict = field(default_factory=dict)
    window_ids: Optional[List[int]] = None  # the GUI's number of each window (for the camFiles' WINDOW_ID)
    fit_lens: tuple = ()                   # plate_wand.LENS_TERMS freed in stage 4: "pp", "k2", "tangential"
    constraints: Optional[Dict] = None     # plate_wand.calibrate's constraints (rig indices, mm)
    start_source: str = ""                 # where `start` came from: "last result" or a folder
    hints_source: str = ""                 # where `hints` came from
    detection: Optional[Dict] = None       # the points' detection record (wand_calibrator.save_detection_record)

    @property
    def names(self) -> List[str]:
        return [f"cam{c}" for c in self.cam_ids]

    def rig(self) -> pw.Rig:
        """The rig with placeholder planes (normal +Z, d = 0): calibrate() finds the planes."""
        wins = [pm.Window([0.0, 0.0, 1.0], 0.0, tuple(w["thickness_mm"]), tuple(w["n"]), w.get("name", ""))
                for w in self.windows]
        return pw.Rig(wins, list(self.cam_to_window))

    def data(self) -> pw.WandData:
        return pw.WandData(np.asarray(self.uv, float), float(self.wand_length),
                           [(int(w), int(h)) for (w, h) in self.sizes])

    def settings(self) -> Dict:
        return {"wand_length_mm": float(self.wand_length), "windows": self.windows,
                "cam_to_window": [int(k) for k in self.cam_to_window], "fit_k1": bool(self.fit_k1),
                "noise_weights": self.noise_weights, "n_final": int(self.n_final),
                "refit_from_last_result": self.start is not None and self.start_source == "last result",
                "start": ("search" if self.start is None else (self.start_source or "given")),
                "fit_lens": list(self.fit_lens), "constraints": self.constraints,
                "hints": ({"source": self.hints_source or "given", **self.hints} if self.hints else None),
                "pixel_mm": self.pixel_mm,
                "detect_mode": self.detect_mode, "source": self.source, "detection": self.detection,
                "frame_ids": [int(f) for f in self.frame_ids], "cam_ids": [int(c) for c in self.cam_ids],
                "sizes": [[int(w), int(h)] for (w, h) in self.sizes]}

    def check(self) -> None:
        uv = np.asarray(self.uv, float)
        if uv.ndim != 4 or uv.shape[2:] != (2, 2):
            raise ValueError("The wand points have the wrong shape.")
        F, Nc = uv.shape[:2]
        if Nc < 2:
            raise ValueError(f"The calibration needs at least two cameras; the wand points have {Nc}.")
        if len(self.sizes) != Nc or any(int(w) <= 0 or int(h) <= 0 for (w, h) in self.sizes):
            raise ValueError("Every camera needs its image width and height (camera table).")
        if len(self.cam_to_window) != Nc:
            raise ValueError("Every camera needs its window (Camera-Window Mapping).")
        if any(not 0 <= int(k) < len(self.windows) for k in self.cam_to_window):
            raise ValueError("A camera is mapped to a window that does not exist.")
        if not float(self.wand_length) > 0:
            raise ValueError("The wand length must be positive.")
        for w in self.windows:
            if not all(float(t) > 0 for t in w["thickness_mm"]):
                raise ValueError("Window thicknesses must be positive.")
            if not all(float(v) >= 1.0 for v in w["n"]):
                raise ValueError("Refractive indices must be at least 1.")
        bad = [t for t in self.fit_lens if t not in pw.LENS_TERMS]
        if bad:
            raise ValueError(f"Unknown lens terms {bad}: choose from {sorted(pw.LENS_TERMS)}.")
        check_constraints(self.constraints, Nc, len(self.windows))
        self.rig()
        seen = np.all(np.isfinite(uv), axis=3)
        usable = int(np.sum(np.all(seen.sum(axis=1) >= 2, axis=1)))
        if usable < 12:
            raise ValueError(f"Only {usable} frames show both balls in two cameras or more.")


# ------------------------------------------------------------------ running
def run(job: Job, progress: Callable = None, stop: Callable = None) -> pw.Solution:
    """plate_wand.calibrate (stages 0-5).  progress(stage, rms_px, text) gets one call per log line; stop() is polled
    at every line and raises Stopped when it returns True."""
    t0 = time.time()
    job.check()
    data = job.data()
    log = _logger(progress, stop)
    rig = job.rig()
    start = None
    if job.start is not None:
        start, rig = start_with_windows(job.start, rig)
    if progress is not None:
        label = "Start search" if start is None else ("Refit from the last result" if job.start_source == "last result"
                                                    else "Refit from a saved fit")
        progress(label, float("nan"), f"{data.n_frames} frames, {data.n_cams} cameras, {len(job.windows)} windows")
    if stop is not None and stop():
        raise Stopped()
    sol = pw.calibrate(data, rig, start=start, hints=job.hints or None, fit_k1=bool(job.fit_k1),
                       n_final=int(job.n_final), noise_weights=job.noise_weights, fit_lens=tuple(job.fit_lens),
                       constraints=job.constraints or None, verbose=True, log=log)
    if stop is not None and stop():
        raise Stopped()
    sol.info.setdefault("total_seconds", time.time() - t0)
    return sol


def start_with_windows(start: tuple, rig: pw.Rig) -> tuple:
    """A start (cameras, rig) with the panel's windows: the start's planes, the job's thicknesses and indices (as
    the real case's run_calibration.py did).  The start must group the cameras into windows as the job does."""
    cams0, rig0 = start
    if len(cams0) != len(rig.cam_win) or list(rig0.cam_win) != list(rig.cam_win):
        raise ValueError("The start's cameras or windows do not match these cameras and the Camera-Window "
                         "Mapping.")
    wins = [pm.Window(w0.n.copy(), float(w0.d), tuple(w.t), tuple(w.idx), w.name or w0.name)
            for w0, w in zip(rig0.windows, rig.windows)]
    return [c.copy() for c in cams0], pw.Rig(wins, list(rig.cam_win))


def check_constraints(cons: Optional[Dict], n_cams: int, n_windows: int) -> None:
    """The constraints' form (plate_wand.set_constraints reads them): rig indices in range, lengths positive."""
    if not cons:
        return
    unknown = set(cons) - {"channel", "aim", "distance"}
    if unknown:
        raise ValueError(f"Unknown constraints {sorted(unknown)}.")
    ch = cons.get("channel")
    if ch:
        widths = list(ch.get("width_mm") or [])
        if len(widths) != n_windows or not any(w is not None for w in widths):
            raise ValueError("The channel needs an inside width for at least one window.")
        if any(w is not None and not float(w) > 0 for w in widths):
            raise ValueError("The channel's widths must be positive.")
        balls = ch.get("ball_radius_mm", 0.0)
        balls = list(balls) if isinstance(balls, (list, tuple)) else [balls, balls]
        if len(balls) != 2 or any(float(b) < 0 for b in balls) or float(ch.get("tolerance_mm", 0.0)) < 0:
            raise ValueError("The channel's tolerance and ball radii must not be negative.")
    for a in cons.get("aim") or []:
        if not a.get("cams") or any(not 0 <= int(c) < n_cams for c in a["cams"]):
            raise ValueError("An aim names a camera that is not in this calibration.")
        if not 0 <= int(a.get("wall", -1)) < n_windows:
            raise ValueError("An aim names a window that no camera looks through.")
        if not float(a.get("mm", 0.0)) > 0 or float(a.get("sigma_mm", 1.0)) <= 0:
            raise ValueError("An aim's depth and sigma must be positive.")
    for d in cons.get("distance") or []:
        if not 0 <= int(d.get("cam", -1)) < n_cams:
            raise ValueError("A distance names a camera that is not in this calibration.")
        if not float(d.get("mm", 0.0)) > 0 or float(d.get("sigma_mm", 1.0)) <= 0:
            raise ValueError("A distance and its sigma must be positive.")


def precheck(job: Job, progress: Callable = None, stop: Callable = None) -> pw.Solution:
    """Stages 1-3 only (the start search with its coarse fit), for the GUI's "Precalibrate to Check", as
    cylinder_wand_gui.precheck: the first start calibrate would refine (`plate_wand.start_candidates`), moved into the
    output frame, with its residuals.  No outlier rejection and no stage 4: a check of the data before the full
    calibration."""
    job.check()
    data = job.data()
    log = _logger(progress, stop)
    rig = job.rig()
    obs = pw.observations(data)
    if progress is not None:
        progress("Start search", float("nan"),
                 f"{data.n_frames} frames, {data.n_cams} cameras, {len(job.windows)} windows")
    starts, binfo = pw.start_candidates(obs, rig, data, hints=job.hints or None, verbose=True, log=log)
    if not starts:
        raise ValueError("The start search found no start: check the detections and the Camera-Window Mapping.")
    _cost, rms3, cams3, rig3 = starts[0]
    cams3, rig3 = [c.copy() for c in cams3], rig3.copy()
    frames = pw.initial_frames(cams3, rig3, obs, data.L)
    sol = pw.Solution(cams=cams3, rig=rig3, frames=frames, L=data.L, obs=obs,
                      layout=pw.Layout(data.n_cams, len(rig3.windows), fit_k1=bool(job.fit_k1)))
    try:
        sol = pw.to_output_frame(sol)        # the frame of the full calibration's results
    except (np.linalg.LinAlgError, ValueError):
        pass
    ev = pw.evaluate(sol.cams, sol.rig, sol.frames, sol.L, obs)
    ok = ev.ok & np.all(np.isfinite(ev.r), axis=1)
    sol.rms = float(np.sqrt(np.mean(np.sum(ev.r[ok] ** 2, axis=1)))) if ok.any() else float("nan")
    sol.info.update({"precheck": True, "bootstrap": binfo, "stage3_rms_px": float(rms3)})
    return sol


def precheck_lines(sol: pw.Solution, job: Job) -> List[str]:
    """A few lines for the GUI's message box after stages 1-3."""
    r, ok, _p = residuals(sol)
    out = [f"Start and coarse fit (stages 1-3) on {sol.obs.n_frames} frames: RMS {sol.rms:.3f} px.",
           "This is only the start; run the calibration for the final answer."]
    norm = np.linalg.norm(np.where(ok[:, None], r, np.nan), axis=1)
    for c, cam in enumerate(sol.cams):
        sel = (sol.obs.cam == c) & ok
        rms = float(np.sqrt(np.mean(norm[sel] ** 2))) if sel.any() else float("nan")
        w = sol.rig.cam_win[c]
        out.append(f"  cam{job.cam_ids[c]}: f {cam.f:,.0f} px, window {job.window_ids[w] if job.window_ids else w}, "
                   f"RMS {rms:.3f} px")
    return out


def child_precheck(job: Job, folder, queue) -> None:
    """`precheck` in a separate process (the GUI's ProcessCalibWorker, as `child_run`): ("progress", ...) messages,
    then ("done", payload) with the error table, the message box's lines and the 3D View's cameras, windows and wand
    ends, or ("failed", message).  Writes nothing; `folder` is not used."""
    import traceback
    try:
        def prog(stage, rms, text):
            queue.put(("progress", str(stage), float(rms), str(text)))
        sol = precheck(job, progress=prog)
        XA, XB = sol.frames.ends(sol.L)
        plot = {"cameras": [{"R": c.R.tolist(), "tvec": c.tvec.tolist()} for c in sol.cams],
                "windows": [{"normal": w.n.tolist(), "d_mm": float(w.d)} for w in sol.rig.windows],
                "ends_A_mm": XA.tolist(), "ends_B_mm": XB.tolist()}
        queue.put(("done", {"precheck": True, "rms_px": float(sol.rms), "per_frame": per_frame_errors(sol, job),
                            "lines": precheck_lines(sol, job), "plot": plot}))
    except ValueError as e:
        queue.put(("failed", str(e)))
    except Exception:
        queue.put(("failed", traceback.format_exc()))


def residuals(sol: pw.Solution):
    obs = sol.obs
    ev = pw.evaluate(sol.cams, sol.rig, sol.frames, sol.L, obs)
    ok = ev.ok & np.all(np.isfinite(ev.r), axis=1)
    return ev.r, ok, obs.uv + np.where(ok[:, None], ev.r, np.nan)


def free_wand_lengths(sol: pw.Solution, ok: np.ndarray = None) -> np.ndarray:
    if ok is None:
        _r, ok, _p = residuals(sol)
    return pw.wand_lengths(sol.cams, sol.rig, sol.obs, rows=np.flatnonzero(ok))


def per_frame_errors(sol: pw.Solution, job: Job) -> Dict[int, Dict]:
    """The GUI's per-frame table, as cylinder_wand_gui.per_frame_errors."""
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


def sight_lines(cams, rig: pw.Rig, depth_mm: float, pixels: str = "corners") -> List:
    """Each camera's lines of sight through its window: centre -> camera-side surface -> water-side surface (plate
    model's trace) -> `depth_mm` past it.  Returns [[camera index, [[x, y, z], ...]], ...]."""
    out = []
    for k, cam in enumerate(cams):
        w = rig.cam_win[k]
        if w < 0:
            continue
        win = rig.windows[w]
        uv = [(cam.cx, cam.cy)]
        if pixels == "corners":
            uv += [(0.0, 0.0), (cam.width - 1.0, 0.0), (cam.width - 1.0, cam.height - 1.0), (0.0, cam.height - 1.0)]
        uv = np.asarray(uv, float)
        rw = cm.rows([cam], np.zeros(len(uv), int))
        ww = pm.wrows([win], np.zeros(len(uv), int))
        xy = pm.undistort(uv, rw)
        o, d, ok = pm.trace(rw, ww, xy)
        for j in range(len(uv)):
            if not ok[j]:
                continue
            q = cam.R.T @ np.array([xy[j, 0], xy[j, 1], 1.0])
            q /= np.linalg.norm(q)
            s = (win.d - float(win.n @ cam.C)) / float(win.n @ q)
            p1 = cam.C + s * q
            cz = float(win.n @ d[j])
            if cz <= 1e-9:
                continue
            pts = [cam.C, p1, o[j], o[j] + (depth_mm / cz) * d[j]]
            out.append([k, [[float(x) for x in p] for p in pts]])
    return out


def plot_data(sol: pw.Solution, job: Job) -> Dict:
    obs = sol.obs
    XA, XB = sol.frames.ends(sol.L)
    r, ok, _proj = residuals(sol)
    Lm = free_wand_lengths(sol, ok)
    fid = [int(job.frame_ids[int(i)]) for i in obs.frame_ids]
    frame_active = np.zeros(obs.n_frames, bool)
    frame_active[obs.frame[obs.active]] = True
    w0 = sol.rig.windows[0]
    mid = 0.5 * (XA + XB)
    depth = float(np.median(w0.depth(mid[frame_active]))) * 2.0 if frame_active.any() else 100.0
    return {
        "frame": "plate_wand's output frame: Z along window 1's normal (camera -> water), origin at the wand's mean",
        "frame_ids": fid, "frame_active": frame_active.tolist(),
        "ends_A_mm": np.round(XA, 5).tolist(), "ends_B_mm": np.round(XB, 5).tolist(),
        "wand_length_free_mm": [None if not np.isfinite(v) else round(float(v), 6) for v in Lm],
        "observations": {
            "frame_id": [fid[int(k)] for k in obs.frame], "cam_id": [int(job.cam_ids[int(c)]) for c in obs.cam],
            "end": obs.end.astype(int).tolist(),
            "u": np.round(obs.uv[:, 0], 4).tolist(), "v": np.round(obs.uv[:, 1], 4).tolist(),
            "du": [None if not g else round(float(x), 5) for x, g in zip(r[:, 0], ok)],
            "dv": [None if not g else round(float(x), 5) for x, g in zip(r[:, 1], ok)],
            "active": obs.active.astype(bool).tolist()},
        "sight_lines_mm": sight_lines(sol.cams, sol.rig, max(depth, 10.0)),
    }


WARNING_TITLES = {"weak_window": "A window seen only weakly", "pinned": "An answer pinned against a window",
                  "hint_far": "The answer is far from a hint", "same_way": "The cameras look the same way",
                  "noisy": "Large residuals", "few_frames": "Few frames",
                  "wand_length_fitted": "The wand length was fitted"}


def checks(result: Dict, job: Job) -> List[Dict]:
    """plate_wand_io.assess's warnings in the page's form, and the detector check."""
    out = []
    for w in result.get("warnings", []):
        out.append({"level": "note" if w.get("code") == "few_frames" else "warning",
                    "title": WARNING_TITLES.get(w.get("code"), str(w.get("code"))), "detail": w.get("text", ""),
                    "what_to_do": "", "evidence": "tests/plate_wand (plate_wand_io.assess)"})
    if not job.fit_k1:
        out.append({"level": "note", "title": "k1 not fitted",
                    "detail": "Radial distortion was held at zero. Real lenses usually need k1: the 20260727 "
                              "flat-window case did (cases/plate_20260727_T0).",
                    "what_to_do": "Choose 'Radial k1' unless the lenses are known to be distortion-free.",
                    "evidence": "cases/plate_20260727_T0"})
    if job.detect_mode is None:
        out.append({"level": "note", "title": "Detection mode unknown",
                    "detail": "These wand points were loaded from a file, so it is not known how the balls were found.",
                    "what_to_do": "", "evidence": ""})
    return out


def _append_meta(path: str, cam_id: int, window_id: int) -> None:
    """The comment block VSC reads (WINDOW_ID groups the cameras by window), with the C++ camera's own plane."""
    lpt = pm.import_lpt()
    plane = lpt.Camera(path)._pinplate_param.plane
    pt = [float(plane.pt[i]) for i in range(3)]
    nv = [float(plane.norm_vector[i]) for i in range(3)]
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if META_BEGIN in text:
        return
    block = (f"{META_BEGIN}\n# VERSION=2\n# CAM_ID={int(cam_id)}\n# WINDOW_ID={int(window_id)}\n"
             f"# PLANE_PT_EXPORT=[{pt[0]:.4f},{pt[1]:.4f},{pt[2]:.4f}]\n"
             f"# PLANE_N=[{nv[0]:.6f},{nv[1]:.6f},{nv[2]:.6f}]\n{META_END}\n")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
        fh.write(block)


def pixel_mm_list(pixel_mm, n: int):
    """Job.pixel_mm as one size per camera (None if none is known): a number is every camera's, a list gives each
    camera's (None or 0 for a camera without one)."""
    if pixel_mm is None:
        return None
    if isinstance(pixel_mm, (list, tuple)):
        out = [float(p) if p else None for p in pixel_mm]
        return out if any(out) else None
    return [float(pixel_mm)] * n if pixel_mm else None


def write_outputs(sol: pw.Solution, job: Job, folder: str) -> Dict:
    """camFile/cam<id>.txt (PINPLATE, C++ writer, plus the REFRACTION_META block), calibration_result.json and
    calibration_report.txt (plate_wand_io.write_result), calibration_data.json, the checks and the results page."""
    os.makedirs(folder, exist_ok=True)
    n = len(sol.cams)
    pixel_mm = pixel_mm_list(job.pixel_mm, n)
    extra = {"gui": {"settings": job.settings(), "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "seconds": sol.info.get("total_seconds")}}
    result = pio.write_result(sol, folder, names=job.names, pixel_mm=pixel_mm, extra=extra)
    for c, cam in enumerate(result["cameras"]):
        if cam.get("camfile"):
            w = sol.rig.cam_win[c]
            _append_meta(os.path.join(folder, cam["camfile"]), job.cam_ids[c],
                         job.window_ids[w] if job.window_ids else w)
    data = plot_data(sol, job)
    result["checks"] = checks(result, job)
    with open(os.path.join(folder, "calibration_data.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    with open(os.path.join(folder, "calibration_result.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=float)
    result["page"] = write_page(result, data, folder)
    return result


def write_page(result: Dict, data: Dict, folder: str) -> Optional[str]:
    try:
        try:
            from ..results_page import page as rp
        except (ImportError, ValueError):
            from modules.camera_calibration.results_page import page as rp
        return rp.write_page(rp.report_from_plate(result, data, folder), os.path.join(folder, "calibration_report.html"))
    except Exception as e:
        print(f"[Plate] The results page could not be written: {e}")
        return None


def child_run(job: Job, folder: str, queue) -> None:
    """The calibration in a separate process (the GUI's ProcessCalibWorker), so that Stop can end it at once: the
    start search of a flat-window rig can run for many minutes between two log lines.  Puts ("progress", stage, rms,
    text) messages, then ("done", payload) or ("failed", message) on `queue`."""
    import traceback
    try:
        def prog(stage, rms, text):
            queue.put(("progress", str(stage), float(rms), str(text)))
        sol = run(job, progress=prog)
        prog("Writing the results", float("nan"), str(folder))
        result = write_outputs(sol, job, folder)
        payload = {"result": result, "folder": folder, "per_frame": per_frame_errors(sol, job),
                   "lines": summary_lines(result), "start": ([c.copy() for c in sol.cams], sol.rig.copy())}
        queue.put(("done", payload))
    except ValueError as e:
        queue.put(("failed", str(e)))
    except Exception:
        queue.put(("failed", traceback.format_exc()))


def summary_lines(result: Dict) -> List[str]:
    w = result["wand_length_free_triangulation"]
    out = [f"{len(result['cameras'])} cameras, {len(result['windows'])} windows, {result['frames']} frames, "
           f"{result['active']} of {result['observations']} detections used.",
           f"Reprojection RMS {result['rms_px']:.3f} px; wand length (free triangulation) RMS error "
           f"{w['rms_mm'] * 1e3:.1f} um over {w['frames']} frames."]
    for c in result["cameras"]:
        s = c.get("sigma") or {}
        out.append(f"  {c['name']}: f {c['f_px']:,.0f} px" + (f" ({c['f_mm']:.1f} mm)" if c.get("f_mm") else "")
                   + (f" +- {100 * s['f_rel']:.2f} %" if s.get("f_rel") else "")
                   + (f", window {c['window'] + 1}, {c['distance_to_window_mm']:.0f} mm from it"
                      if c.get("window") is not None and c["window"] >= 0 else "")
                   + f", RMS {c['residuals']['rms_px']:.3f} px")
    out += constraint_lines(result)
    for ck in result.get("checks", []):
        out.append(("WARNING: " if ck["level"] == "warning" else "Note: ") + ck["title"])
    return out


def constraint_lines(result: Dict) -> List[str]:
    """The constraints at the answer (plate_wand.constraint_report in the result's info), in a line or two."""
    rep = (result.get("info") or {}).get("constraints") or {}
    out = []
    ch = rep.get("channel")
    if ch:
        out.append(f"Channel held (tolerance {ch.get('tol', 0):.1f} mm): the largest breach {ch.get('max_breach_mm', 0):.3f} mm.")
    for pr in rep.get("priors") or []:
        kind = pr.get("kind")
        if kind not in ("aim", "distance"):
            continue
        out.append(f"{kind.capitalize()} held at {pr.get('mean', float('nan')):.1f} mm: "
                   f"{pr.get('value', float('nan')):.2f} mm at the answer ({pr.get('residual_sigmas', float('nan')):+.2f} sigma).")
    return out
