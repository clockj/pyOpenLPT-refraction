"""Shared helpers of the flat-window GUI options' tests (plan: 2026-10-09-plan.md, approved 9 Oct 2026 with D1 (b),
D2 (a), D3 (a), D4 (a)).

- paths: this folder's data/ (git-ignored), the real case (cases/plate_20260727_T0, read only) and its results on X:
  (read only);
- G8's offscreen Qt helpers (tests/cylinder_wand_gui/common.py: qt_app, Dialogs, wait_for), which also put the GUI's
  module (build-py/Release) first on sys.path;
- Log: prints and keeps the lines and the checks; save() writes <name>_out.txt and <name>.json here;
- rig B's wand points as the GUI holds them (G8's), a view set up as G8 sets it up, and JobCatcher, which takes the
  Job the view builds instead of starting the calibration process.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE / "data"
CASE = REPO / "cases" / "plate_20260727_T0"
X_T0 = Path(r"X:\Shijie Zhong\Bubble Shear Project\Processed\20260727\Results\T0")
REF_COMMIT = "cffed7d"

_spec = importlib.util.spec_from_file_location("cwg_common", REPO / "tests" / "cylinder_wand_gui" / "common.py")
C = importlib.util.module_from_spec(_spec)
sys.modules["cwg_common"] = C
_spec.loader.exec_module(C)
for _p in (str(REPO / "tests" / "plate_wand"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

qt_app, Dialogs, wait_for = C.qt_app, C.Dialogs, C.wait_for


class Log:
    """Prints and keeps every line and every check; save() writes <name>_out.txt and <name>.json here."""

    def __init__(self, name: str):
        self.name = name
        self.lines, self.checks = [], []
        self.t0 = time.time()

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.lines.append(str(msg))

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"check": label, "ok": bool(ok), "detail": detail})
        self(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))
        return bool(ok)

    def save(self, extra: dict = None) -> bool:
        n_fail = sum(not c["ok"] for c in self.checks)
        self(f"\n{len(self.checks) - n_fail} of {len(self.checks)} checks passed; {time.time() - self.t0:.0f} s")
        with open(HERE / f"{self.name}_out.txt", "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.lines) + "\n")
        with open(HERE / f"{self.name}.json", "w", encoding="utf-8") as fh:
            json.dump({"checks": self.checks, "seconds": time.time() - self.t0, **(extra or {})}, fh, indent=1,
                      default=_default)
        return n_fail == 0


def _default(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def committed_file(relpath: str, name: str) -> Path:
    """A file as commit REF_COMMIT has it (git show), written to data/ref/<name> (git-ignored)."""
    import subprocess
    dst = DATA / "ref" / name
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        blob = subprocess.run(["git", "show", f"{REF_COMMIT}:{relpath}"], cwd=REPO, capture_output=True,
                              check=True).stdout
        dst.write_bytes(blob)
    return dst


def head_plate_wand_gui():
    """plate_wand_gui.py as commit REF_COMMIT has it, imported on its own (its absolute-import fallback, with the
    curved_calibration folder on sys.path)."""
    curved = REPO / "modules" / "camera_calibration" / "curved_calibration"
    if str(curved) not in sys.path:
        sys.path.insert(0, str(curved))
    path = committed_file("modules/camera_calibration/curved_calibration/plate_wand_gui.py",
                          f"plate_wand_gui_{REF_COMMIT}.py")
    spec = importlib.util.spec_from_file_location("plate_wand_gui_head", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["plate_wand_gui_head"] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ rig B through the view (as G8)
def rig_b_points(rig, n_frames=300, seed=1308, noise_px=0.1):
    """Synthetic wand points of a plate rig as the GUI holds them (Filtered_Small = end A, Filtered_Large = end B)."""
    import prigs
    w = prigs.make_wand(rig, n_frames, np.random.default_rng(seed), noise_px=noise_px)
    wp = {}
    for i in range(w.uv.shape[0]):
        fr = {}
        for c in range(rig.n_cams):
            if np.all(np.isfinite(w.uv[i, c])):
                a, b = w.uv[i, c]
                fr[c] = [[a[0], a[1], 20.0, 0.9, "Filtered_Small", 0], [b[0], b[1], 30.0, 0.9, "Filtered_Large", 1]]
        if len(fr) >= 2:
            wp[i] = fr
    return wp


def make_view(rig, wp, mapping, thickness=12.7, dist_index=None):
    """CameraCalibrationView with the wand points, the camera table, Pinhole+Refraction, the windows and the mapping.
    dist_index: set the distortion box as a user would (None: leave the model's default)."""
    from modules.camera_calibration import CameraCalibrationView
    view = CameraCalibrationView()
    view.wand_num_cams.setValue(rig.n_cams)
    for r in range(rig.n_cams):
        view.wand_cam_table.cellWidget(r, 3).setValue(rig.cams[r].width)
        view.wand_cam_table.cellWidget(r, 4).setValue(rig.cams[r].height)
    view.wand_calibrator.wand_points = wp
    view.wand_calibrator.wand_points_filtered = None
    view._wand_detect_mode_used = "ellipse"
    view.wand_model_combo.setCurrentText("Pinhole+Refraction")
    view.wand_len_spin.setValue(rig.L)
    if dist_index is not None:
        view.dist_model_combo.setCurrentIndex(dist_index)
        view.dist_model_combo.activated.emit(dist_index)
    view.window_count_spin.setValue(max(mapping) + 1)
    for row, wid in enumerate(mapping):
        view.cam_window_table.cellWidget(row, 1).setCurrentIndex(wid)
    view.media3_combo.setCurrentText("Other")
    view.media3_index.setValue(1.333)
    view.media2_thick.setValue(thickness)
    return view


class _Sig:
    def __init__(self):
        self.slots = []

    def connect(self, f):
        self.slots.append(f)


class JobCatcher:
    """Stands in for view.ProcessCalibWorker: records (target, job, folder) and never runs."""
    caught = []

    def __init__(self, target, job, folder):
        JobCatcher.caught.append((getattr(target, "__name__", str(target)), job, folder))
        self.progress, self.finished_ok, self.failed, self.stopped = _Sig(), _Sig(), _Sig(), _Sig()

    def start(self):
        pass

    def isFinished(self):
        return True


def catch_job(view, precalibrate=False):
    """The Job the view builds for Run (or Precalibrate), taken instead of starting the process; the view is then
    reset as after a failure.  Returns (target name, job, folder) or None if the view refused (its message is in the
    Dialogs' record)."""
    import modules.camera_calibration.view as V
    real = V.ProcessCalibWorker
    V.ProcessCalibWorker = JobCatcher
    JobCatcher.caught = []
    try:
        view._run_wand_calibration(precalibrate=precalibrate)
    finally:
        V.ProcessCalibWorker = real
    got = JobCatcher.caught[-1] if JobCatcher.caught else None
    if got is not None:
        try:
            view._cylinder_cleanup()
        except Exception:
            pass
        view._busy_end('wand_calibration')
    return got
