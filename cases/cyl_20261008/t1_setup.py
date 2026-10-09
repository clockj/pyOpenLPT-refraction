"""T1 tracking setup shared by Phase T's scripts (plan: 2026-10-08-plan.md, Phase T; the 7 Oct case's t1_setup.py,
for three cameras): the cameras, the view volume, the library build, and STB configs.

The cameras are T0's calibration for tracking (camFile_track: tolerance 1e-9), copied to data/cams_T0/.  The view
volume: the box around the points of the water (r < 97.6 mm) that project onto the sensors of at least
MIN_VIEW cameras, found by sampling (cached in data/t1_view_volume.json); the share every camera sees is reported
with it.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
CAM_IDS = (1, 2, 3)
CAMS_DIR = os.environ.get("CYL_T1_CAMS", "cams_T0")                 # of data/: cams_T0_c3crop with cam3 cropped (S1b)
CAMS = [os.path.join(DATA, CAMS_DIR, f"cam{c}.txt") for c in CAM_IDS]
PRE = os.path.join(DATA, os.environ.get("CYL_T1_PRE", "T1pre"))     # S1's frames; or another folder of data/
BUILD = os.environ.get("CYL_STB_BUILD", "p4c")
MIN_VIEW = 2


def load_config() -> dict:
    with open(os.path.join(HERE, "config.json"), encoding="utf-8") as fh:
        return json.load(fh)


def copy_cameras() -> list:
    """T0's tracking camFiles (X: T0\\camFile_track) to data/cams_T0/ (once); returns CAMS (CYL_T1_CAMS's)."""
    cfg = load_config()
    src = os.path.join(cfg["out_dir"], cfg["tracking_camfiles"]["folder"])
    os.makedirs(os.path.join(DATA, "cams_T0"), exist_ok=True)
    for c in CAM_IDS:
        dst = os.path.join(DATA, "cams_T0", f"cam{c}.txt")
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(src, f"cam{c}.txt"), dst)
    return CAMS


def lib():
    out = os.path.join(REPO, "build-stb", BUILD, "out")
    if out not in sys.path:
        sys.path.insert(0, out)
    sys.path.insert(0, os.path.join(REPO, "tests", "cylinder_stb"))
    import pyopenlpt as lpt  # noqa: E402
    return lpt


def view_volume(lpt, n=300000, seed=7):
    path = os.path.join(DATA, "t1_view_volume.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    cams = [lpt.loadCameraModel(f) for f in copy_cameras()]
    rng = np.random.default_rng(seed)
    X = rng.uniform(-70, 70, (n, 3))
    X = X[np.hypot(X[:, 0], X[:, 2]) < 97.0]
    P = [lpt.Pt3D(*map(float, x)) for x in X]
    seen = np.zeros((len(X), len(cams)), bool)
    for j, cam in enumerate(cams):
        res = cam.projectBatchStatus(P)
        W, H = cam.getNCol(), cam.getNRow()
        seen[:, j] = [r[0] and 0 <= r[1][0] < W and 0 <= r[1][1] < H for r in res]
    k = seen.sum(axis=1)
    S = X[k >= MIN_VIEW]
    S3 = X[k == len(cams)]
    vol = {"x": [float(S[:, 0].min()) - 1, float(S[:, 0].max()) + 1],
           "y": [float(S[:, 1].min()) - 1, float(S[:, 1].max()) + 1],
           "z": [float(S[:, 2].min()) - 1, float(S[:, 2].max()) + 1],
           "min_cameras": MIN_VIEW, "points_seen_by_min_cameras": int(len(S)),
           "points_seen_by_all": int(len(S3)), "sampled": int(len(X)),
           "seen_by_pairs": {f"{a}{b}": int(np.sum(seen[:, i] & seen[:, j])) for i, a in enumerate(CAM_IDS)
                             for j, b in enumerate(CAM_IDS) if j > i},
           "all_box": {"x": [float(S3[:, 0].min()), float(S3[:, 0].max())],
                       "y": [float(S3[:, 1].min()), float(S3[:, 1].max())],
                       "z": [float(S3[:, 2].min()), float(S3[:, 2].max())]} if len(S3) else None,
           "r_max_mm": float(np.hypot(S[:, 0], S[:, 2]).max())}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(vol, fh, indent=1)
    return vol


def image_lists(pre=None) -> list:
    pre = pre or PRE
    return [os.path.join(pre, f"cam{c}ImageNames.txt") for c in CAM_IDS]


def stb_config(path, frames, vol, tracer, voxel_mm=0.03, ipr=None, shake=None, output_dir="Results", extra=None,
               lists=None, fps=1000):
    """An STB config.json for T1's preprocessed frames (S1's image lists), three cameras."""
    cfg = {
        "_comment": "T1 of run 20261008 (cases/cyl_20261008, Phase T): S1's preprocessed frames, three cameras.",
        "frames": [int(frames[0]), int(frames[1])], "fps": int(fps), "threads": 0,
        "cameras": [{"file": f, "max_intensity": 255} for f in CAMS],
        "image_lists": lists or image_lists(),
        "view_volume_mm": {"x": vol["x"], "y": vol["y"], "z": vol["z"]},
        "voxel_mm": voxel_mm, "output_dir": output_dir,
        "objects": [{
            "type": "Tracer",
            "initial_phase": {"search_radius_vox": 10.0, "frames": 4},
            "convergence_phase": {"avg_spacing_vox": 30.0},
            "shake": shake or {"width_vox": 0.25, "loops": 4, "ghost_threshold": 0.01},
            "predictive_field": {"grid": [51, 51, 51], "search_radius_vox": 25.0},
            "ipr": ipr or {"loops": 3, "reduced_cameras": 1, "reduced_loops": 2, "tol_2d_px": 1.5,
                           "tol_3d_vox": 2.0},
            "tracer": tracer,
        }],
    }
    if extra:
        cfg.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=1)
    return cfg


def noise_per_camera(pre=None):
    with open(os.path.join(pre or PRE, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    return [man["cameras"][f"cam{c}"]["noise_8bit_median"] for c in CAM_IDS]
