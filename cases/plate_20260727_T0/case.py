"""Configuration and data access for the plate_20260727_T0 calibration case: four cameras through two flat tank walls.

Everything case-specific (paths, walls, wand, cameras) lives in config.json; this module reads it, the wand detections
(the GUI's CSV format), the images and the GUI's earlier camFiles, and puts pyOpenLPT's flat-window calibration
modules (plate_model, plate_wand, plate_wand_io) and the local C++ build on the path.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))          # pyOpenLPT root
CURVED = os.path.join(REPO, "modules", "camera_calibration", "curved_calibration")
CONFIG_PATH = os.path.join(HERE, "config.json")
for _p in (os.path.join(REPO, "build-py", os.environ.get("CONFIG", "Release")), CURVED, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plate_model as pm          # noqa: E402
import plate_wand as pw           # noqa: E402
import plate_wand_io as pio       # noqa: E402


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rig_config(cfg: dict, n_water: float = None, thickness_mm: float = None, hints: dict = None) -> pio.RigConfig:
    """The rig of config.json as a plate_wand_io.RigConfig, optionally with another water index or wall thickness
    (both walls) or hints."""
    r = json.loads(json.dumps(cfg["rig"]))
    for w in r["windows"]:
        if n_water is not None:
            w["n"][-1] = float(n_water)
        if thickness_mm is not None:
            w["thickness_mm"] = [float(thickness_mm)]
    rc = pio.RigConfig(windows=r["windows"], cam_to_window=r["cam_to_window"], cameras=r["cameras"],
                       wand_length_mm=float(r["wand_length_mm"]), hints=dict(hints if hints is not None else r["hints"]))
    rc.rig()                                    # validates the windows
    return rc


def detections_path(cfg: dict, which: str = "all") -> str:
    return os.path.join(cfg["results_dir"], cfg["detections"][which])


def read_detections(cfg: dict, which: str = "all") -> pio.CsvData:
    """The Filtered detections of one of the configured CSVs: uv[frame, camera, end, (u, v)], cameras in the config's
    order, frame_ids = the CSV's frame numbers (= the image file numbers)."""
    ids = [c["id"] for c in cfg["rig"]["cameras"]]
    return pio.read_gui_csv(detections_path(cfg, which), cam_ids=ids)


def drop_truncated(cd: pio.CsvData, sizes, margin: float) -> int:
    """Set to NaN (in place) every detection whose ball (centre +- radius + margin) crosses the image border: a ball
    cut by the border has a biased centre.  Returns how many were dropped."""
    n = 0
    for c, (w, h) in enumerate(sizes):
        u, v, r = cd.uv[:, c, :, 0], cd.uv[:, c, :, 1], cd.radii[:, c, :] + margin
        cut = (u - r < 0) | (u + r > w - 1) | (v - r < 0) | (v + r > h - 1)
        cd.uv[:, c][cut] = np.nan
        n += int(cut.sum())
    return n


def wand_data(cd: pio.CsvData, rc: pio.RigConfig, rows: Optional[np.ndarray] = None) -> pw.WandData:
    """WandData of the detections' frames `rows` (indices into cd.uv; default: all)."""
    uv = cd.uv if rows is None else cd.uv[np.asarray(rows)]
    return pw.WandData(uv, rc.wand_length_mm, rc.sizes)


def interleaved(n: int, k: int, m: int) -> np.ndarray:
    """Every m-th of n frames, starting at k."""
    return np.arange(k, n, m)


def load_gui(cfg: dict, name: str):
    """The cameras and walls of one of the GUI's calibrations (camFiles cam0.txt ... cam3.txt): (cams, rig).  Cameras
    of one wall share their window; the first camera's copy is used."""
    folder = os.path.join(cfg["results_dir"], cfg["gui_calibrations"][name])
    cams, wins = [], {}
    cam_win = cfg["rig"]["cam_to_window"]
    for c, cc in enumerate(cfg["rig"]["cameras"]):
        cam, win = pio.load_camfile(os.path.join(folder, f"{cc['name']}.txt"))
        cams.append(cam)
        wins.setdefault(cam_win[c], win)
    rig = pw.Rig([wins[w] for w in sorted(wins)], list(cam_win))
    return cams, rig


def image_path(cfg: dict, cam: int, frame_id: int) -> str:
    folder = cfg["rig"]["cameras"][cam]["folder"]
    return os.path.join(cfg["image_dir"], cfg["image_pattern"].format(folder=folder, frame=int(frame_id)))


def read_image(path: str) -> np.ndarray:
    import cv2
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


# ------------------------------------------------------------------ a rig's frame-independent description
def angle_deg(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def describe(cams, rig) -> Dict:
    """What does not depend on the world frame: each camera's f and its distance and angle to its wall, the
    distances between camera centres, the angles between viewing directions, and the angle between the walls; and
    every camera centre and viewing direction in the walls' frame (origin camera 0's centre, z along wall 1's normal,
    y along n1 x n2)."""
    n = [w.n for w in rig.windows]
    ez = n[0]
    ey = np.cross(n[0], n[1]) if len(n) > 1 else np.cross(ez, [1.0, 0.0, 0.0])
    ey = ey / np.linalg.norm(ey)
    ex = np.cross(ey, ez)
    B = np.vstack([ex, ey, ez])
    C0 = cams[0].C
    out = {"cameras": [], "pairs": [], "walls_angle_deg": angle_deg(n[0], n[1]) if len(n) > 1 else None}
    for c, cam in enumerate(cams):
        win = rig.windows[rig.cam_win[c]]
        out["cameras"].append({
            "f_px": cam.f, "k1": float(cam.dist[0]), "distance_to_wall_mm": win.distance(cam.C),
            "incidence_deg": angle_deg(cam.R[2], win.n), "centre_in_walls_frame_mm": (B @ (cam.C - C0)).tolist(),
            "view_in_walls_frame": (B @ cam.R[2]).tolist()})
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            out["pairs"].append({"cams": [i, j], "baseline_mm": float(np.linalg.norm(cams[i].C - cams[j].C)),
                                 "views_angle_deg": angle_deg(cams[i].R[2], cams[j].R[2])})
    return out


# ------------------------------------------------------------------ saved fits
def save_state(path: str, sol, extra: Dict = None) -> None:
    """A fit as plain data: cameras, walls, wand frames, the observations' frames and active set, covariance."""
    st = {"cams": [(c.R, c.C, c.f, c.cx, c.cy, c.width, c.height, c.dist) for c in sol.cams],
          "windows": [(w.n, w.d, w.t, w.idx, w.name) for w in sol.rig.windows], "cam_win": list(sol.rig.cam_win),
          "L": sol.L, "rms": sol.rms, "sigma_px": sol.sigma_px, "cov": sol.cov, "weights": sol.weights,
          "layout": {"fit_k1": sol.layout.fit_k1, "fit_L": sol.layout.fit_L, "names": list(sol.layout.names)},
          "frame_ids": sol.obs.frame_ids, "active": sol.obs.active, "obs_frame": sol.obs.frame,
          "obs_cam": sol.obs.cam, "obs_end": sol.obs.end, "frames_m": sol.frames.m, "frames_w": sol.frames.w,
          "info": {k: v for k, v in sol.info.items() if _plain(v)}, "extra": extra or {}}
    with open(path, "wb") as fh:
        pickle.dump(st, fh)


def load_state(path: str) -> Dict:
    """A saved fit: the dict of save_state with 'cams' and 'rig' rebuilt."""
    with open(path, "rb") as fh:
        st = pickle.load(fh)
    st["cams"] = [pm.Camera(R, C, f, cx, cy, w, h, dist) for (R, C, f, cx, cy, w, h, dist) in st["cams"]]
    st["rig"] = pw.Rig([pm.Window(n, d, t, idx, name) for (n, d, t, idx, name) in st["windows"]], st["cam_win"])
    return st


def subset_rows(n: int, settings: Dict) -> np.ndarray:
    """The frame rows a run_calibration.py run used (its --every/--offset/--first/--last settings)."""
    rows = interleaved(n, int(settings.get("offset") or 0), int(settings.get("every") or 1))
    if settings.get("first") is not None:
        rows = rows[rows < int(round(settings["first"] * n))]
    if settings.get("last") is not None:
        rows = rows[rows >= n - int(round(settings["last"] * n))]
    return rows


def rebuild(st: Dict, cfg: dict):
    """The data of a saved fit again: (rig config, CSV data, rows, WandData, Obs with the fit's active set, Frames)."""
    s = st["extra"]["settings"]
    rc = rig_config(cfg, n_water=s.get("n_water"), thickness_mm=s.get("thickness"))
    cd = read_detections(cfg, s.get("detections") or "all")
    if s.get("edge_margin") is not None:
        drop_truncated(cd, rc.sizes, s["edge_margin"])
    rows = subset_rows(cd.uv.shape[0], s)
    data = wand_data(cd, rc, rows)
    obs = pw.observations(data)
    if not (np.array_equal(obs.frame, st["obs_frame"]) and np.array_equal(obs.cam, st["obs_cam"])
            and np.array_equal(obs.end, st["obs_end"])):
        raise ValueError("the rebuilt observations differ from the saved fit's")
    obs.active = np.asarray(st["active"], bool).copy()
    frames = pw.Frames(np.array(st["frames_m"], float), np.array(st["frames_w"], float))
    return rc, cd, rows, data, obs, frames


def _plain(v) -> bool:
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


class Tee:
    """print() to the console and to a file."""

    def __init__(self, path: str):
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, *args):
        s = " ".join(str(a) for a in args)
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()
