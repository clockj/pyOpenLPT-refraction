"""Configuration and helpers for the stb_20260727_10Hz_r_b_1 case: tracking (STB) of run 20260727 / 10Hz_r_b_1 with
the old (GUI + VSC) and the new (flat-window wand) camFiles.

Everything case-specific (folders, camFile sets, frames, view volumes) lives in config.json.  This module reads it and
wraps the C++ cameras of the local pyOpenLPT build (build-py/Release) with vectorised project / line-of-sight /
triangulation helpers.
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))          # pyOpenLPT root
CONFIG_PATH = os.path.join(HERE, "config.json")
TESTS = os.path.join(HERE, "tests")
BUILD = os.path.join(REPO, "build-py", os.environ.get("CONFIG", "Release"))
for _p in (BUILD, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def case_path(cfg: dict, rel: str) -> str:
    """A config path: absolute as given, else relative to case_dir."""
    return rel if os.path.isabs(rel) else os.path.join(cfg["case_dir"], rel)


def camfile_paths(cfg: dict, which: str) -> List[str]:
    """The four camFiles of a set ("old" or "new"), absolute."""
    return [case_path(cfg, p) for p in cfg["camfiles"][which]]


def import_lpt():
    import pyopenlpt as lpt  # noqa: WPS433 (the local build, put on sys.path above)
    return lpt


def load_cameras(paths: Sequence[str]):
    lpt = import_lpt()
    return [lpt.Camera(p) for p in paths]


def project(cam, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Image points of world points X (n, 3) through one C++ camera: (uv (n, 2), ok (n,)); NaN where it fails."""
    lpt = import_lpt()
    X = np.asarray(X, float).reshape(-1, 3)
    res = cam.projectBatchStatus([lpt.Pt3D(float(a), float(b), float(c)) for a, b, c in X], False)
    uv = np.full((len(X), 2), np.nan)
    ok = np.zeros(len(X), bool)
    for i, (good, pt, _msg) in enumerate(res):
        if good:
            u, v = pt[0], pt[1]
            if np.isfinite(u) and np.isfinite(v) and abs(u) < 1e12 and abs(v) < 1e12:
                uv[i] = (u, v)
                ok[i] = True
    return uv, ok


def lines_of_sight(cam, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lines of sight of image points uv (n, 2): (origin (n, 3), unit direction (n, 3), ok (n,)).  The origin is on
    the window's water-side surface, as in the C++ code."""
    lpt = import_lpt()
    uv = np.asarray(uv, float).reshape(-1, 2)
    res = cam.lineOfSightBatchStatus([lpt.Pt2D(float(u), float(v)) for u, v in uv])
    o = np.full((len(uv), 3), np.nan)
    d = np.full((len(uv), 3), np.nan)
    ok = np.zeros(len(uv), bool)
    for i, (good, line, _msg) in enumerate(res):
        if good:
            o[i] = (line.pt[0], line.pt[1], line.pt[2])
            d[i] = (line.unit_vector[0], line.unit_vector[1], line.unit_vector[2])
            ok[i] = True
    return o, d, ok


def triangulate(origins: Sequence[np.ndarray], dirs: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Least-squares intersection of k lines per point (lists of k arrays (n, 3)), as myMATH::triangulation:
    (X (n, 3), mean distance from X to the lines (n,))."""
    n = len(origins[0])
    A = np.zeros((n, 3, 3))
    b = np.zeros((n, 3))
    eye = np.eye(3)
    for o, d in zip(origins, dirs):
        P = eye[None] - d[:, :, None] * d[:, None, :]
        A += P
        b += np.einsum("nij,nj->ni", P, o)
    X = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    dist = np.zeros(n)
    for o, d in zip(origins, dirs):
        w = X - o
        h = np.einsum("ni,ni->n", w, w) - np.einsum("ni,ni->n", w, d) ** 2
        dist += np.sqrt(np.maximum(h, 0.0))
    return X, dist / len(origins)


def in_image(uv: np.ndarray, cam) -> np.ndarray:
    """Whether each image point lies on the sensor (0 <= u <= n_col - 1, 0 <= v <= n_row - 1)."""
    u, v = uv[:, 0], uv[:, 1]
    with np.errstate(invalid="ignore"):
        return (u >= 0) & (u <= cam.getNCol() - 1) & (v >= 0) & (v <= cam.getNRow() - 1)


def plate_of(cam) -> dict:
    """The window of a PINPLATE camera: camera centre, water-side surface point and normal (camera -> water),
    thickness and indices (camera side first), and the solver settings."""
    pp = cam._pinplate_param
    vec = lambda p: np.array([p[0], p[1], p[2]], float)  # noqa: E731
    return {"centre": vec(pp.t_vec_inv), "water_pt": vec(pp.plane.pt), "normal": vec(pp.plane.norm_vector),
            "thickness": float(sum(pp.w_array)), "n": list(reversed(list(pp.refract_array))),
            "proj_tol": float(pp.proj_tol), "proj_nmax": int(pp.proj_nmax)}


def box_corners(vv: Sequence[float]) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = vv
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)


def box_grid(vv: Sequence[float], n: int = 11) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = vv
    g = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n), np.linspace(z0, z1, n), indexing="ij")
    return np.stack([a.ravel() for a in g], axis=1)


def box_random(vv: Sequence[float], n: int, seed: int = 0) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = vv
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n), rng.uniform(z0, z1, n)], axis=1)


def save_json(path: str, obj) -> None:
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=conv)
