"""Shared helpers of the cylinder STB tests (plan: cases/cyl_20261007/2026-10-07-plan.md, Phases 1-5).

- use_build(NAME): import pyopenlpt from build-stb/NAME/out, a variant built by tests/stb_improve/build_variant.py,
  and return (module, its build record) so that every result names the code it ran on.
- camfiles: read a camFile's PINCYL fields, write a copy with another projection tolerance.
- the port frame of Camera.cpp's buildFrame(), to call the refract_curved core with the camera's own coordinates.
- sample_shared(): random points that both cameras see on their sensors.
- Log: prints, keeps and saves the lines (<stem>_out.txt) and a JSON record (<stem>.json) beside the script.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
CAMS_T2 = os.path.join(DATA, "cams_T2")          # 6 Oct T2's delivered camFiles (copied from X:)


def use_build(name: str):
    out = os.path.join(REPO, "build-stb", name, "out")
    if not os.path.isdir(out):
        raise SystemExit(f"no build {out}")
    if out not in sys.path:
        sys.path.insert(0, out)
    import pyopenlpt as lpt  # noqa: E402
    with open(os.path.join(out, "build_info.json"), encoding="utf-8") as fh:
        info = json.load(fh)
    rec = {k: info.get(k) for k in ("name", "git_head", "git_diff_sha256", "start", "defs")}
    rec["module"] = lpt.__file__
    return lpt, rec


class Log:
    def __init__(self, stem: str):
        self.stem = stem
        self.lines = []
        self.t0 = time.time()

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.lines.append(str(msg))

    def save(self, data: dict | None = None) -> None:
        with open(self.stem + "_out.txt", "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.lines) + "\n")
        if data is not None:
            with open(self.stem + ".json", "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=1, default=_json_default)


def _json_default(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.bool_):
        return bool(v)
    raise TypeError(type(v))


# ------------------------------------------------------------------ camFiles
def read_camfile(path: str) -> dict:
    """The fields of a camFile, keyed by their '# ...' header line (without '#', up to ':'), each a list of rows of
    floats (or the raw text for the model line)."""
    with open(path, encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    out, key = {}, None
    for ln in lines:
        if ln.startswith("#"):
            key = ln[1:].split(":")[0].strip()
            out[key] = []
        elif key is not None and ln.strip():
            try:
                out[key].append([float(x) for x in ln.split(",") if x.strip() != ""])
            except ValueError:
                out[key].append(ln.strip())
    return out


def pincyl_geometry(path: str) -> dict:
    f = read_camfile(path)
    return {"center": np.array(f["Port Center (axis point for a cylinder, centre for a sphere)"][0]),
            "axis": np.array(f["Port Axis (unit; unused by a sphere)"][0]),
            "n": list(f["Refractive Index Array"][0]), "R": list(f["Interface Radii"][0]),
            "pinhole": np.array(f["Inverse of Translation Vector"][0]),
            "tol": f["Projection Tolerance"][0][0], "nmax": int(f["Maximum Number of Iterations for Projection"][0][0]),
            "size": tuple(int(x) for x in f["Image Size"][0])}


def write_tol_copy(src: str, dst: str, tol: float) -> None:
    """src with its 'Projection Tolerance' value replaced by tol (every other byte unchanged)."""
    with open(src, encoding="utf-8", newline="") as fh:
        lines = fh.read().split("\n")
    i = next(k for k, ln in enumerate(lines) if ln.startswith("# Projection Tolerance"))
    eol = "\r" if lines[i + 1].endswith("\r") else ""
    lines[i + 1] = f"{tol:.16e}" + eol
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(lines))


def cams_with_tol(src_dir: str, tol: float, n_cams: int = 2) -> list[str]:
    """Copies of src_dir/cam1..camN.txt with the given tolerance, in data/<folder>_tol<exp>/."""
    tag = f"{os.path.basename(os.path.normpath(src_dir))}_tol{int(round(-np.log10(tol)))}"
    out = []
    for c in range(1, n_cams + 1):
        dst = os.path.join(DATA, tag, f"cam{c}.txt")
        write_tol_copy(os.path.join(src_dir, f"cam{c}.txt"), dst, tol)
        out.append(dst)
    return out


# ------------------------------------------------------------------ the port frame (Camera.cpp buildFrame)
def port_frame(center, axis):
    w = np.asarray(axis, float) / np.linalg.norm(axis)
    trial = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = trial - trial.dot(w) * w
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    return np.asarray(center, float), u, v, w


def to_local(frame, X):
    c, u, v, w = frame
    d = np.asarray(X, float) - c
    return [float(d.dot(u)), float(d.dot(v)), float(d.dot(w))]


# ------------------------------------------------------------------ points
def pt3(lpt, X):
    return lpt.Pt3D(float(X[0]), float(X[1]), float(X[2]))


def pt2_tuple(p):
    return (float(p[0]), float(p[1]))


def sample_shared(lpt, cams, n, box, rng, batch=4000):
    """n points uniform in box [(x0,y0,z0),(x1,y1,z1)] whose project() lands on every camera's sensor; returns
    (points (n,3), pixels (n, n_cam, 2))."""
    lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
    pts, pix = [], []
    while len(pts) < n:
        X = lo + (hi - lo) * rng.random((batch, 3))
        P = [lpt.Pt3D(*map(float, x)) for x in X]
        res = [cam.projectBatchStatus(P) for cam in cams]
        for i in range(batch):
            ok = all(r[i][0] for r in res)
            if not ok:
                continue
            uv = [pt2_tuple(r[i][1]) for r in res]
            on = all(0 <= u < cam.getNCol() and 0 <= v < cam.getNRow() for (u, v), cam in zip(uv, cams))
            if on:
                pts.append(X[i])
                pix.append(uv)
                if len(pts) == n:
                    break
    return np.array(pts), np.array(pix)
