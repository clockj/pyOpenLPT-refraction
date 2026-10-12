"""The paper's wand calibration through the tube (D:\\Refraction\\paper, Section 9.1): pyOpenLPT's cylinder_wand with
the residual's image taken from the paper's projection, and no double-image guard.

    import projection_fit as pf
    sol = pf.calibrate_by_projection(data, tube, **kw)   # cylinder_wand.calibrate's options; guard_px is 0
    with pf.projection_residuals():                       # or any cylinder_wand call with the paper's residual
        sol = cw.refine(...)

What differs from the library, in memory only (no library file is edited):
  - the residual: cylinder_wand.evaluate starts Newton (cyl_model.project) at the detection.  evaluate_by_projection
    starts it at the image that the C++ PINCYL projectAll (the paper's projection) returns nearest the detection, and
    at the detection where the C++ returns no image (it searches only points inside the water);
  - no guard: guard_px = 0.  The paper keeps the wand in a central single-image volume instead.

The compiled extension: with CYL_WAND_PAPER_EXT set to a folder holding a copy of pyopenlpt*.pyd, that copy is
imported, so that another session may rebuild pyOpenLPT meanwhile (Windows locks a loaded .pyd).  Otherwise the
build folder is used, as tests/cylinder_wand/rigs.py arranges.  `runtime_record()` gives the pyOpenLPT commit, the
extension's path and SHA-256, and the SHA-256 of the library files; the tests record it at their start and end.
"""
from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                                   # pyOpenLPT
CURVED = REPO / "modules" / "camera_calibration" / "curved_calibration"
TESTS = REPO / "tests" / "cylinder_wand"

EXT = os.environ.get("CYL_WAND_PAPER_EXT")
if EXT:
    sys.path.insert(0, EXT)
    import pyopenlpt as _lpt                             # noqa: E402  first, so that rigs.py's build path cannot win
    assert Path(_lpt.__file__).resolve().parent == Path(EXT).resolve(), _lpt.__file__
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import rigs                                              # noqa: E402  (adds the build folder and the library)
from rigs import cm                                      # noqa: E402
import cylinder_wand as cw                               # noqa: E402

assert cw.cm is cm, "cylinder_wand and rigs must share one cyl_model"

LIB_FILES = (CURVED / "cyl_model.py", CURVED / "cylinder_wand.py", CURVED / "cylinder_wand_io.py",
             TESTS / "rigs.py")

# rows seen by evaluate_by_projection since the last reset: how many C++ images each had
STATS = {"calls": 0, "rows": 0, "no_image": 0, "one_image": 0, "more_images": 0}


def reset_stats():
    for k in STATS:
        STATS[k] = 0


# ------------------------------------------------------------------ the residual
def projection_start(cams, tube: cm.Tube, X: np.ndarray, cam_idx: np.ndarray, uv: np.ndarray):
    """Pixels at which Newton starts, (n, 2): per row, the C++ image of X nearest the detection uv, or uv itself where
    the C++ returns none; and the number of C++ images per row."""
    start = np.array(uv, float, copy=True)
    count = np.zeros(len(X), int)
    for c in np.unique(cam_idx):
        sel = np.flatnonzero(cam_idx == c)
        cc = cm.to_cpp(cams[c], tube)
        for k, im in zip(sel, cm.cpp_images(cc, X[sel])):
            count[k] = len(im)
            if len(im):
                start[k] = im[np.argmin(np.sum((im - uv[k]) ** 2, axis=1))]
    return start, count


def evaluate_by_projection(cams, tube: cm.Tube, frames, L: float, obs, layout=None, rows: np.ndarray = None):
    """cylinder_wand.evaluate with Newton started at the paper's projection (module docstring).  Apart from the
    start of cm.project, this is the library's evaluate line for line."""
    rows = np.arange(len(obs)) if rows is None else rows
    fr, ca, en = obs.frame[rows], obs.cam[rows], obs.end[rows]
    XA, XB = frames.ends(L)
    X = np.where((en == 0)[:, None], XA[fr], XB[fr])
    rw = cm.rows(cams, ca)
    start, count = projection_start(cams, tube, X, ca, obs.uv[rows])
    STATS["calls"] += 1
    STATS["rows"] += len(rows)
    STATS["no_image"] += int(np.sum(count == 0))
    STATS["one_image"] += int(np.sum(count == 1))
    STATS["more_images"] += int(np.sum(count > 1))
    pr = cm.project(rw, X, cm.undistort(start, rw), tube, jac=layout is not None)
    r = pr.uv - obs.uv[rows]
    ev = cw.Eval(r=r, ok=pr.ok.copy())
    if layout is None:
        return ev
    n = len(rows)
    Jg = np.zeros((n, 2, layout.P))
    for j, mi in enumerate(layout.model_idx):
        g = layout.pos[ca, j]
        sel = (g >= 0) & pr.ok
        Jg[np.flatnonzero(sel), :, g[sel]] = pr.J_cam[sel, :, mi]
    t1, t2 = frames.tangent_basis()
    sgn = np.where(en == 0, -0.5 * L, 0.5 * L)
    JX = np.where(pr.ok[:, None, None], pr.J_X, 0.0)
    Jf = np.zeros((n, 2, 5))
    Jf[:, :, 0:3] = JX
    Jf[:, :, 3] = np.einsum("nai,ni->na", JX, t1[fr]) * sgn[:, None]
    Jf[:, :, 4] = np.einsum("nai,ni->na", JX, t2[fr]) * sgn[:, None]
    if layout.fit_L:
        Jg[:, :, layout.L_index] = np.einsum("nai,ni->na", JX, frames.w[fr]) * np.where(en == 0, -0.5, 0.5)[:, None]
    ev.Jg, ev.Jf, ev.det = Jg, Jf, pr.det
    return ev


@contextlib.contextmanager
def projection_residuals():
    """Within the block, every cylinder_wand call evaluates its residuals with evaluate_by_projection."""
    saved = cw.evaluate
    cw.evaluate = evaluate_by_projection
    try:
        yield
    finally:
        cw.evaluate = saved


def calibrate_by_projection(data, tube: cm.Tube, **kw):
    """cylinder_wand.calibrate with the paper's residual and no guard; the answer's info carries the residual's
    name and the image counts its evaluations met."""
    if kw.get("guard_px", 0):
        raise ValueError("the paper's calibration has no double-image guard: guard_px must be 0")
    kw["guard_px"] = 0
    reset_stats()
    with projection_residuals():
        sol = cw.calibrate(data, tube, **kw)
    sol.info["residuals"] = "projection"
    sol.info["projection_stats"] = dict(STATS)
    return sol


# ------------------------------------------------------------------ provenance and running
def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args) -> str:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")      # read only: never refresh the other session's index
    res = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, env=env)
    return res.stdout.strip()


def runtime_record() -> dict:
    lpt = cm.import_lpt()
    rel = [str(p.relative_to(REPO)).replace("\\", "/") for p in LIB_FILES]
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "pyopenlpt_commit": _git("rev-parse", "HEAD"),
            "library_files_changed_since_commit": _git("status", "--porcelain", "--", *rel) or None,
            "extension": str(Path(lpt.__file__).resolve()), "extension_sha256": sha256(lpt.__file__),
            "files_sha256": {p.name: sha256(p) for p in LIB_FILES}}


def below_normal_priority():
    """This process at below-normal priority (Windows), so that other work keeps the machine."""
    if os.name == "nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)


class Run:
    """A test's output: `<name>_out.txt` (every line also printed) and `<name>.json`, with the provenance record at
    the start and the end, and whether the library files changed in between."""

    def __init__(self, name: str):
        below_normal_priority()
        self.name = name
        self.out = open(HERE / f"{name}_out.txt", "w", encoding="utf-8")
        self.t0 = time.time()
        self.start = runtime_record()
        self.log(f"{name}: started {self.start['time']}; pyOpenLPT {self.start['pyopenlpt_commit'][:7]}; extension "
                 f"{self.start['extension']} (SHA-256 {self.start['extension_sha256'][:12]}); library files "
                 f"{'unchanged since the commit' if not self.start['library_files_changed_since_commit'] else 'CHANGED: ' + self.start['library_files_changed_since_commit']}")

    def log(self, msg: str = ""):
        print(msg, flush=True)
        self.out.write(msg + "\n")
        self.out.flush()

    def finish(self, result: dict):
        end = runtime_record()
        same = end["files_sha256"] == self.start["files_sha256"] and \
            end["extension_sha256"] == self.start["extension_sha256"]
        self.log(f"library files and extension {'unchanged during the run' if same else 'CHANGED DURING THE RUN'}; "
                 f"{time.time() - self.t0:.0f} s")
        with open(HERE / f"{self.name}.json", "w", encoding="utf-8") as fh:
            json.dump({"start": self.start, "end": end, "unchanged_during_run": same,
                       "seconds": time.time() - self.t0, **result}, fh, indent=1)
        self.out.close()
        return same
