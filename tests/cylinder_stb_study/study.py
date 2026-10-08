"""Shared helpers of the cylinder STB study (plan: 2026-10-08-plan.md).

- the library: build p4c (tests/cylinder_stb/cstb.use_build), the campaign's helpers on the path (cstb, shape_model,
  stbruns, synth, eval_truth);
- paths: data/ (rendered cases, git-ignored), runs/ (STB runs, git-ignored), the rig's camFiles (data/rig/), T1's case;
- the view volume every rig tracks: T1's (cases/cyl_20261007/data/t1_view_volume.json, the box both real cameras see);
- Log: prints, keeps and saves <stem>_out.txt and <stem>.json beside the scripts (cstb.Log).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CSTB = os.path.join(REPO, "tests", "cylinder_stb")
CASE_T1 = os.path.join(REPO, "cases", "cyl_20261007")
DATA = os.path.join(HERE, "data")
RUNS = os.path.join(HERE, "runs")
RIG = os.path.join(DATA, "rig")
CAMS_T0 = [os.path.join(CASE_T1, "data", "cams_T0", f"cam{c}.txt") for c in (1, 2)]
BUILD = os.environ.get("CYL_STUDY_BUILD", "p4c")
TODAY = "head1007"                  # today's code (the 7 Oct HEAD before this work)
R_WATER = 97.63125                  # the tube's inner radius [mm]
R_IN = 30.0                         # the central region: r < R_IN about the axis

for p in (CSTB, CASE_T1):
    if p not in sys.path:
        sys.path.insert(0, p)

import cstb  # noqa: E402


def lib(build: str | None = None):
    """(pyopenlpt module, its build record) of build BUILD."""
    return cstb.use_build(build or BUILD)


def Log(name: str):
    return cstb.Log(os.path.join(HERE, name))


def rig_files(n: int) -> list[str]:
    """The camFiles of the n-camera rig: cam1, cam2 (T0's) and, for 3 and 4, cam3 and cam4 (make_rigs.py)."""
    return [os.path.join(RIG, f"cam{c}.txt") for c in range(1, n + 1)]


def view_volume() -> dict:
    with open(os.path.join(CASE_T1, "data", "t1_view_volume.json"), encoding="utf-8") as fh:
        return json.load(fh)


def vol_box(vol: dict):
    return ((vol["x"][0], vol["y"][0], vol["z"][0]), (vol["x"][1], vol["y"][1], vol["z"][1]))


def in_volume(X, vol) -> np.ndarray:
    X = np.asarray(X, float)
    return ((X[..., 0] >= vol["x"][0]) & (X[..., 0] <= vol["x"][1]) & (X[..., 1] >= vol["y"][0])
            & (X[..., 1] <= vol["y"][1]) & (X[..., 2] >= vol["z"][0]) & (X[..., 2] <= vol["z"][1]))


def save_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, default=cstb._json_default)


def load_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
