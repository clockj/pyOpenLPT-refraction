"""Configuration and file access for the cyl_20260930_T0_cam1 calibration case.

Everything case-specific (paths, port, pitch, planes) lives in config.json; this module only reads it.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))          # pyOpenLPT root
CONFIG_PATH = os.path.join(HERE, "config.json")

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)   # the Phantom TIFF tag warning


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plane_files(cfg: dict) -> list:
    """[(z_nominal_mm, path)] in increasing z, i.e. from the farthest plane towards the camera.

    The file names carry the stage labels (z_0mm ... z_38mm); the positions are config planes.z_mm (2.54 mm per
    step of 2 in the label)."""
    pl = cfg["planes"]
    if len(pl["labels"]) != len(pl["z_mm"]):
        raise ValueError("planes.labels and planes.z_mm differ in length")
    out = []
    for lab, z in zip(pl["labels"], pl["z_mm"]):
        p = os.path.join(cfg["raw_dir"], pl["file_pattern"].format(label=lab))
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        out.append((float(z), p))
    return out


def plane_labels(cfg: dict) -> list:
    """The stage label of each plane (as in its file name), in the order of plane_files."""
    return [int(v) for v in cfg["planes"]["labels"]]


def plane_tag(label: int) -> str:
    """Name stem of a plane's output files, after its raw image: z_00mm ... z_38mm."""
    return f"z_{int(label):02d}mm"


def read_image(path: str) -> np.ndarray:
    """Grey image as float32, checked against the configured sensor size."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def import_lpt():
    """Import pyopenlpt from the local build (build-py/Release), as conftest.py does."""
    for p in (REPO, os.path.join(REPO, "build-py", "Release"), os.path.join(REPO, "build-py")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    import pyopenlpt as lpt                      # noqa: E402
    return lpt


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
