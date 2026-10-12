"""Configuration, file access and the shared steps of the wand calibration (T0) of the cyl_20261007 case (plan:
2026-10-07-plan.md, Phase C).

A copy of the 6 Oct cases' case.py (cases/cyl_20261006_T2_wand), unchanged except for this paragraph: run 20261007/T0
differs from them only in config.json (paths, frame count).  The Planes and T1 have their own modules.

Everything case-specific (paths, cameras, tube, wand, detector and acceptance settings) lives in config.json.  The
library is used unchanged:
  - ball_ellipses.detect_wand_balls (the GUI's "ellipse" mode) for the detections;
  - cylinder_wand_gui (Job, run, write_outputs): cylinder_wand.calibrate exactly as the GUI's "Pinhole+Cylinder"
    runs it, and the GUI's output files (camFiles, result, report, results page).
This module only reads the images, keeps or rejects each camera-frame by the rules of config.accept, and writes the
case's tables.

Environment: CASE_PYOPENLPT_DIR, if set, is a folder holding a frozen copy of the pyopenlpt extension, imported
instead of build-py/Release (Windows locks a loaded .pyd, so another session could not rebuild while this runs).
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import sys
import time
from types import SimpleNamespace

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))          # pyOpenLPT root
CONFIG_PATH = os.path.join(HERE, "config.json")
CURVED = os.path.join(REPO, "modules", "camera_calibration", "curved_calibration")
DET_COLUMNS = ["frame", "camera", "n_balls", "k", "u", "v", "r", "metric", "a", "b", "theta_rad", "dropped"]

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)   # the Phantom TIFF tag warning


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Log:
    """Prints and keeps every line; save() writes <name>_out.txt (and <name>.json if data is given) beside the
    calling script."""

    def __init__(self, path_stem: str):
        self.stem = path_stem
        self.lines = []
        self.t0 = time.time()

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.lines.append(str(msg))

    def save(self, data: dict = None) -> None:
        with open(self.stem + "_out.txt", "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.lines) + "\n")
        if data is not None:
            with open(self.stem + ".json", "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=1, default=_json_default)


def _json_default(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    raise TypeError(type(v))


# ------------------------------------------------------------------ the library
def setup_paths() -> None:
    order = []
    frozen = os.environ.get("CASE_PYOPENLPT_DIR")
    if frozen:
        order.append(frozen)
    order += [os.path.join(REPO, "build-py", "Release"), REPO]
    for p in reversed(order):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


def library() -> SimpleNamespace:
    """pyopenlpt (first, so that a frozen copy is the one loaded) and the curved-calibration modules."""
    setup_paths()
    import pyopenlpt  # noqa: F401
    from modules.camera_calibration.curved_calibration import cyl_model as cm
    from modules.camera_calibration.curved_calibration import cylinder_wand as cw
    from modules.camera_calibration.curved_calibration import cylinder_wand_gui as cg
    from modules.camera_calibration.curved_calibration import cylinder_wand_io as cio
    return SimpleNamespace(lpt=pyopenlpt, cm=cm, cw=cw, cg=cg, cio=cio)


_BE = None


def ball_ellipses():
    """ball_ellipses.py loaded from its file (no package import, so the detection workers stay light)."""
    global _BE
    if _BE is None:
        spec = importlib.util.spec_from_file_location("ball_ellipses_case", os.path.join(CURVED, "ball_ellipses.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _BE = m
    return _BE


# ------------------------------------------------------------------ images and detections
def cam_ids(cfg: dict) -> list:
    return [int(c["id"]) for c in cfg["cameras"]]


def image_path(cfg: dict, cam_id: int, frame: int) -> str:
    folder = {int(c["id"]): c["folder"] for c in cfg["cameras"]}[int(cam_id)]
    return os.path.join(cfg["raw_dir"], folder, cfg["frames"]["pattern"].format(frame=int(frame)))


def all_frames(cfg: dict) -> list:
    return list(range(int(cfg["frames"]["first"]), int(cfg["frames"]["last"]) + 1))


def read_image(path: str) -> np.ndarray:
    """The image, read in one go and decoded in memory: over the network drive this is about 25 % faster than
    cv2.imread, which reads a TIFF in many small pieces (the same pixels)."""
    try:
        with open(path, "rb") as fh:
            buf = np.frombuffer(fh.read(), np.uint8)
    except OSError:
        raise IOError(f"cannot read {path}")
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def detect_image(img: np.ndarray, det: dict) -> list:
    """The GUI's "ellipse" estimator with the case's settings; rows [u, v, r, metric, a, b, theta, dropped]."""
    be = ball_ellipses()
    return be.detect_wand_balls(img, det["polarity"], min_r_px=float(det["min_r_px"]), max_r_px=float(det["max_r_px"]),
                                open_px=int(det["open_px"]), rod_mask_px=float(det["rod_mask_px"]),
                                guard_px=float(det["guard_px"]), full=True)


def detect_task(task):
    """Worker: (frame, camera id, path, detector settings) -> (frame, camera id, rows or None if unreadable)."""
    frame, cam, path, det = task
    cv2.setNumThreads(1)
    try:
        img = read_image(path)
    except IOError:
        return frame, cam, None
    return frame, cam, detect_image(img, det)


def detections_path(cfg: dict) -> str:
    return os.path.join(cfg["out_dir"], "detection", "detections_all.csv")


def write_detections(path: str, dets: dict) -> None:
    """One line per ball found (k = 0, 1, ... smallest first), or one line with n_balls = 0 for a camera-frame with
    none (n_balls = -1: the image could not be read)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(DET_COLUMNS)
        for (frame, cam) in sorted(dets):
            rows = dets[(frame, cam)]
            if rows is None:
                w.writerow([frame, cam, -1] + [""] * 9)
            elif not rows:
                w.writerow([frame, cam, 0] + [""] * 9)
            else:
                for k, r in enumerate(rows):
                    w.writerow([frame, cam, len(rows), k] + [f"{float(x):.5f}" for x in r[:4]]
                               + [f"{float(x):.5f}" for x in r[4:7]] + [int(r[7])])
    os.replace(tmp, path)


def read_detections(path: str) -> dict:
    dets = {}
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            key = (int(row["frame"]), int(row["camera"]))
            n = int(row["n_balls"])
            if n < 0:
                dets[key] = None
                continue
            dets.setdefault(key, [])
            if n > 0:
                dets[key].append([float(row[c]) for c in ("u", "v", "r", "metric", "a", "b", "theta_rad")]
                                 + [float(row["dropped"])])
    return dets


# ------------------------------------------------------------------ acceptance
def edge_rms(row) -> float:
    return 1.0 / float(row[3]) - 1.0


def aspect(row) -> float:
    return max(row[4], row[5]) / min(row[4], row[5])


def classify(dets: dict, cfg: dict):
    """Keep or reject every camera-frame by config.accept.  Returns (kept, reasons, medians):
    kept[(frame, cam)] = (row of ball A, row of ball B) (A = the smaller sqrt(a b)); reasons[(frame, cam)] = the
    first rule that rejected it; medians[cam] = (median r of A, median r of B) over the camera-frames that passed the
    shape rules (the size rule's reference)."""
    acc = cfg["accept"]
    lo, hi = acc["aspect_range"]
    shape_ok, reasons = {}, {}
    for key, rows in dets.items():
        if rows is None:
            reasons[key] = "image unreadable"
        elif len(rows) == 0:
            reasons[key] = "no ball"
        elif len(rows) == 1:
            reasons[key] = "one ball"
        elif len(rows) > 2:
            reasons[key] = "more than two balls"
        elif any(r[7] for r in rows):
            reasons[key] = "balls overlap (guard)"
        elif any(edge_rms(r) > float(acc["edge_rms_max_px"]) for r in rows):
            reasons[key] = "edge RMS"
        elif any(not lo <= aspect(r) <= hi for r in rows):
            reasons[key] = "aspect"
        else:
            a, b = sorted(rows, key=lambda r: r[2])
            shape_ok[key] = (a, b)
    medians = {}
    for cam in cam_ids(cfg):
        ra = [v[0][2] for k, v in shape_ok.items() if k[1] == cam]
        rb = [v[1][2] for k, v in shape_ok.items() if k[1] == cam]
        medians[cam] = (float(np.median(ra)) if ra else float("nan"), float(np.median(rb)) if rb else float("nan"))
    tol = float(acc["size_tol"])
    kept = {}
    for key, (a, b) in shape_ok.items():
        ma, mb = medians[key[1]]
        if abs(a[2] / ma - 1.0) > tol or abs(b[2] / mb - 1.0) > tol:
            reasons[key] = "size"
        else:
            kept[key] = (a, b)
    return kept, reasons, medians


def frames_in_all(kept: dict, cams: list) -> list:
    frames = sorted({k[0] for k in kept})
    return [f for f in frames if all((f, c) in kept for c in cams)]


def uv_array(kept: dict, frames: list, cams: list) -> np.ndarray:
    uv = np.full((len(frames), len(cams), 2, 2), np.nan)
    for i, f in enumerate(frames):
        for j, c in enumerate(cams):
            a, b = kept[(f, c)]
            uv[i, j, 0] = a[:2]
            uv[i, j, 1] = b[:2]
    return uv


def write_gui_csv(path: str, kept: dict, frames: list, cfg: dict) -> None:
    """The kept detections in the GUI's wand-points format (WandCalibrator.export_wand_data / load_wand_data_from_csv):
    Frame, Camera (the GUI's camera index), Status (Filtered_Small = ball A, Filtered_Large = ball B), PointIdx, X, Y,
    Radius (sqrt(a b)), Metric (1 / (1 + edge RMS))."""
    gui = {int(c["id"]): int(c["gui_index"]) for c in cfg["cameras"]}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Frame", "Camera", "Status", "PointIdx", "X", "Y", "Radius", "Metric"])
        for f in frames:
            for c in cam_ids(cfg):
                a, b = kept[(f, c)]
                for status, idx, p in (("Filtered_Small", 0, a), ("Filtered_Large", 1, b)):
                    w.writerow([f, gui[c], status, idx, f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.3f}", f"{p[3]:.4f}"])


def load_selection(cfg: dict):
    """Detections -> (kept, reasons, medians, frames kept in every camera)."""
    dets = read_detections(detections_path(cfg))
    kept, reasons, medians = classify(dets, cfg)
    return dets, kept, reasons, medians, frames_in_all(kept, cam_ids(cfg))


# ------------------------------------------------------------------ calibration
def make_job(lib, cfg: dict, frames: list, uv: np.ndarray, **over):
    cal = dict(cfg["calibrate"])
    cal.update(over)
    W, H = int(cfg["image"]["width"]), int(cfg["image"]["height"])
    return lib.cg.Job(uv=uv, frame_ids=list(frames), cam_ids=cam_ids(cfg), sizes=[(W, H)] * len(cfg["cameras"]),
                      wand_length=float(cfg["wand"]["length_mm"]), radii=tuple(cfg["tube"]["radii_mm"]),
                      n=tuple(cfg["tube"]["n"]), fit_k1=bool(cal["fit_k1"]), fit_L=bool(cal["fit_L"]),
                      noise_weights=str(cal["noise_weights"]), n_final=int(cal["n_final"]),
                      pixel_mm=float(cfg["image"]["pixel_mm"]), detect_mode="ellipse",
                      source=f"{cfg['raw_dir']} ({', '.join(c['folder'] for c in cfg['cameras'])}); case {cfg['case']}")


def run_calibration(lib, job, folder: str, log=print):
    """cylinder_wand_gui.run (the GUI's runner), every log line to <folder>/calibration_log.txt and the stage changes
    to `log`; then cylinder_wand_gui.write_outputs into folder.  Returns (sol, result, seconds)."""
    os.makedirs(folder, exist_ok=True)
    t0 = time.time()
    state = {"stage": None, "lines": 0}
    with open(os.path.join(folder, "calibration_log.txt"), "w", encoding="utf-8") as fh:
        def progress(stage, rms, text):
            fh.write(f"[{time.time() - t0:8.1f} s] {stage}: {text}\n")
            state["lines"] += 1
            if stage != state["stage"]:
                state["stage"] = stage
                log(f"    {time.time() - t0:7.1f} s  {stage}")
        sol = lib.cg.run(job, progress=progress)
    secs = time.time() - t0
    result = lib.cg.write_outputs(sol, job, folder)
    return sol, result, secs


def summary(result: dict) -> dict:
    """The quantities the tests compare: per camera f, distance from the axis, azimuth, the height relative to
    camera 1, the viewing direction, k1; their sigmas; and the run's statistics."""
    cams = result["cameras"]
    h0 = cams[0]["height_mm"]
    out = {"rms_px": result["rms_px"], "sigma_px": result["sigma_px"], "frames": result["frames"],
           "active": result["active"], "observations": result["observations"],
           "wand_length_mm": result["wand_length_mm"], "L_sigma_mm": result.get("L_sigma_mm"),
           "wand_free_rms_um": 1e3 * result["wand_length_free_triangulation"]["rms_mm"],
           "wand_free_mean_um": 1e3 * result["wand_length_free_triangulation"]["mean_mm"], "cameras": []}
    for c in cams:
        s = c.get("sigma") or {}
        out["cameras"].append({
            "name": c["name"], "f_px": c["f_px"], "f_sigma_px": s.get("f_px"),
            "distance_mm": c["distance_from_axis_mm"], "distance_sigma_mm": s.get("distance_mm"),
            "azimuth_deg": c["azimuth_deg"], "azimuth_sigma_deg": s.get("azimuth_deg"),
            "height_rel_mm": c["height_mm"] - h0, "height_sigma_mm": s.get("height_mm"),
            "viewing_direction": c["viewing_direction"], "k1": (c.get("dist") or [0.0])[0],
            "k1_sigma": s.get("k1"), "rms_px": c["residuals"]["rms_px"],
            "n_active": c["residuals"]["n_active"], "n_total": c["residuals"]["n_total"]})
    return out
