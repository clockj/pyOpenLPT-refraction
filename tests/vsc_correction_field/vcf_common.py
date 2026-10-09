"""Shared helpers of the correction-field tests (plan: 2026-10-08-plan.md, approved 8 Oct with D1 (a) and D2 (a)).

- paths: this folder's data/ and runs/ (git-ignored), T1's case (cases/cyl_20261007, read only), the self-calibration
  campaign (tests/cylinder_selfcal, read only), the builds (vcf: this change, from build_vcf.py; p4c: the baseline);
- lib(build): pyopenlpt of a build (one build per process);
- Log: prints, keeps and saves <name>_out.txt and <name>.json here;
- selfcal_correction(): a field of the self-calibration campaign (its dense map) as a library correction.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True   # the other campaigns' modules are imported, never rewritten

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
RUNS = os.path.join(HERE, "runs")
CASE = os.path.join(REPO, "cases", "cyl_20261007")
CASE_DATA = os.path.join(CASE, "data")
CASE_RUNS = os.path.join(CASE_DATA, "runs")
SELFCAL = os.path.join(REPO, "tests", "cylinder_selfcal")
CSTB = os.path.join(REPO, "tests", "cylinder_stb")
CAMS_T1 = [os.path.join(CASE_DATA, "cams_T0", f"cam{c}.txt") for c in (1, 2)]
PRE_M15 = os.path.join(CASE_DATA, "T1pre_s4_m15")
CAM_PINHOLE = os.path.join(CSTB, "data", "cams_T2_pinhole", "cam1.txt")
CAM_PINPLATE = os.path.join(REPO, "cases", "stb_20260727_10Hz_r_b_1", "tests", "t1_camfiles",
                            "new_cam0_tol1e-12_n1000.txt")
THREADS_MAX = 12      # the user's limit on OMP_NUM_THREADS

for _p in (REPO, CSTB):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from modules.vsc.correction_io import Correction, compose  # noqa: E402,F401

_LPT = {}


GUI_MODULE = os.path.join(REPO, "build-py", "Release")   # the GUI's module, as build_local.py installs it


def lib(build: str = "vcf"):
    """pyopenlpt of build-stb/<build>/out, or with build "gui" the GUI's installed module (one build per process)."""
    if "lpt" not in _LPT:
        if build == "gui":
            if GUI_MODULE not in sys.path:
                sys.path.insert(0, GUI_MODULE)
            import pyopenlpt as lpt_gui
            _LPT["lpt"], _LPT["rec"] = lpt_gui, {"name": "gui", "module": lpt_gui.__file__}
        else:
            import cstb
            _LPT["lpt"], _LPT["rec"] = cstb.use_build(build)
        _LPT["build"] = build
    elif _LPT["build"] != build:
        raise RuntimeError(f"this process already uses build {_LPT['build']}")
    return _LPT["lpt"]


def build_record():
    return _LPT.get("rec")


class Log:
    """Prints, keeps and saves <name>_out.txt and <name>.json beside the scripts."""

    def __init__(self, name: str):
        self.stem = os.path.join(HERE, name)
        self.lines = []
        self.t0 = time.time()

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.lines.append(str(msg))

    def save(self, data=None) -> None:
        with open(self.stem + "_out.txt", "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.lines) + "\n")
        if data is not None:
            save_json(self.stem + ".json", data)


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


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, default=_json_default)


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def selfcal_dense(name: str):
    """The dense map (n_cam, 2, H, W) of the self-calibration campaign's field data/fields/<name>.npz."""
    z = np.load(os.path.join(SELFCAL, "data", "fields", name + ".npz"), allow_pickle=False)
    return np.asarray(z["dense"], dtype=np.float64)


def dense_at(dense_cam: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """The campaign's eval_dense: the map read bilinearly at pixels uv (n, 2), clipped to the image."""
    from scipy import ndimage
    H, W = dense_cam.shape[1:]
    coords = np.vstack([np.clip(uv[:, 1], 0, H - 1), np.clip(uv[:, 0], 0, W - 1)])
    return np.column_stack([ndimage.map_coordinates(dense_cam[k], coords, order=1, mode="nearest")
                            for k in (0, 1)])


def selfcal_correction(name: str, cam: int, step: float = 8.0) -> tuple[Correction, dict]:
    """Camera cam's field of the campaign as a library correction: its dense map at the nodes (every `step` px),
    with how far the correction's bilinear read departs from the dense map at every pixel."""
    dense = selfcal_dense(name)[cam]
    H, W = dense.shape[1:]
    corr = Correction.from_function(W, H, lambda uv: dense_at(dense, uv), step=step,
                                    source=f"tests/cylinder_selfcal/data/fields/{name}.npz cam{cam + 1}")
    yy, xx = np.mgrid[0:H, 0:W]
    uv = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    diff = np.hypot(*(corr.at(uv) - dense.reshape(2, -1).T).T)
    mag = np.hypot(dense[0], dense[1])
    return corr, {"max_dev_px": float(diff.max()), "rms_dev_px": float(np.sqrt(np.mean(diff ** 2))),
                  "field_rms_px": float(np.sqrt(np.mean(mag ** 2))), "field_max_px": float(mag.max()),
                  "slope_bound": corr.slope_bound()}


REF_COMMIT = "cffed7d"   # "today's code" of the follow-up's before/after tests (2026-10-09-plan.md)


def committed_module(relpath: str, name: str):
    """A module as commit REF_COMMIT has it (git show), imported under `name`; the copy goes to data/ref/
    (git-ignored), so nothing is archived here."""
    import importlib.util
    import subprocess
    dst = os.path.join(DATA, "ref", f"{name}_{REF_COMMIT}.py")
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        blob = subprocess.run(["git", "show", f"{REF_COMMIT}:{relpath}"], cwd=REPO, capture_output=True,
                              check=True).stdout
        with open(dst, "wb") as fh:
            fh.write(blob)
    spec = importlib.util.spec_from_file_location(name, dst)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
