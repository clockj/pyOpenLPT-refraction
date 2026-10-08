"""T1 tracking setup shared by Phase 6's scripts (plan: 2026-10-07-plan.md, Phase 6): the cameras, the view volume
both cameras see in the water, the library build, and STB configs.

The cameras are T0's delivered calibration for tracking (camFile_track: tolerance 1e-9), copied to data/cams_T0/.
The view volume: the box around the points of the water (r < 97.6 mm) that project onto both sensors, found by
sampling (cached in data/t1_view_volume.json).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
CAMS = [os.path.join(DATA, "cams_T0", f"cam{c}.txt") for c in (1, 2)]
PRE = os.path.join(DATA, os.environ.get("CYL_T1_PRE", "T1pre_s4"))   # S1's frames (every 4th); or another folder
BUILD = os.environ.get("CYL_STB_BUILD", "p4c")                        # of data/ (e.g. T1pre_s4_m15: a 15 px median)


def lib():
    out = os.path.join(REPO, "build-stb", BUILD, "out")
    if out not in sys.path:
        sys.path.insert(0, out)
    sys.path.insert(0, os.path.join(REPO, "tests", "cylinder_stb"))
    import pyopenlpt as lpt  # noqa: E402
    return lpt


def view_volume(lpt, n=200000, seed=7):
    path = os.path.join(DATA, "t1_view_volume.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    cams = [lpt.loadCameraModel(f) for f in CAMS]
    rng = np.random.default_rng(seed)
    X = rng.uniform(-60, 60, (n, 3))
    X = X[np.hypot(X[:, 0], X[:, 2]) < 97.0]
    P = [lpt.Pt3D(*map(float, x)) for x in X]
    ok = np.ones(len(X), bool)
    for cam in cams:
        res = cam.projectBatchStatus(P)
        W, H = cam.getNCol(), cam.getNRow()
        ok &= np.array([r[0] and 0 <= r[1][0] < W and 0 <= r[1][1] < H for r in res])
    S = X[ok]
    vol = {"x": [float(S[:, 0].min()) - 1, float(S[:, 0].max()) + 1],
           "y": [float(S[:, 1].min()) - 1, float(S[:, 1].max()) + 1],
           "z": [float(S[:, 2].min()) - 1, float(S[:, 2].max()) + 1],
           "points_seen_by_both": int(ok.sum()), "sampled": int(len(X)),
           "r_max_mm": float(np.hypot(S[:, 0], S[:, 2]).max())}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(vol, fh, indent=1)
    return vol


def stb_config(path, frames, vol, tracer, voxel_mm=0.03, ipr=None, shake=None, output_dir="Results", extra=None):
    """An STB config.json for T1's preprocessed frames (S1's image lists: frame index i = raw frame 4 i)."""
    cfg = {
        "_comment": "T1 of run 20261007 (cases/cyl_20261007, Phase 6): S1's preprocessed frames, every 4th raw frame.",
        "frames": [int(frames[0]), int(frames[1])], "fps": 1000, "threads": 0,
        "cameras": [{"file": f, "max_intensity": 255} for f in CAMS],
        "image_lists": [os.path.join(PRE, f"cam{c}ImageNames.txt") for c in (1, 2)],
        "view_volume_mm": {"x": vol["x"], "y": vol["y"], "z": vol["z"]},
        "voxel_mm": voxel_mm, "output_dir": output_dir,
        "objects": [{
            "type": "Tracer",
            "initial_phase": {"search_radius_vox": 10.0, "frames": 4},
            "convergence_phase": {"avg_spacing_vox": 30.0},
            "shake": shake or {"width_vox": 0.25, "loops": 4, "ghost_threshold": 0.01},
            "predictive_field": {"grid": [51, 51, 51], "search_radius_vox": 25.0},
            "ipr": ipr or {"loops": 3, "reduced_cameras": 0, "reduced_loops": 0, "tol_2d_px": 1.0,
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


def noise_per_camera():
    with open(os.path.join(PRE, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    return [man["cameras"][f"cam{c}"]["noise_8bit_median"] for c in (1, 2)]
