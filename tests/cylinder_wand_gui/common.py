"""Shared helpers of the cylinder-wand GUI tests (plan: 2026-10-06-plan.md): paths, the synthetic image sets, an
offscreen Qt application with its dialogs answered by the test, and the check log.

Synthetic images come from T13's renderer (tests/cylinder_wand/t13_common.py): three replicas of cam1 (f 41,459 px,
1,094 mm from the axis, 1280 x 800 px) at azimuths 0, 45 and 90 deg around the real tube, a wand of a 1.0 mm and a
1.5 mm ball 10 mm apart on a 0.3 mm rod, the blur measured on the real camera, 2 grey levels of noise.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CAMPAIGN = REPO / "tests" / "cylinder_wand"
DATA = HERE / "data"
for _p in (str(CAMPAIGN), str(REPO / "gui"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import t13_common as T  # noqa: E402  (first: puts build-py/Release and the curved folder on sys.path)
import rigs  # noqa: E402

PY = sys.executable


class Log:
    """Prints and keeps every line and every check; save() writes <name>_out.txt and <name>.json beside the test."""

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
                      default=float)
        return n_fail == 0


# ------------------------------------------------------------------ synthetic image sets
def _render_one(args):
    """One camera's image of one frame (a worker task): the wand when the camera sees both balls whole, else the
    background with noise only."""
    import cv2
    cam, XA, XB, uv, lighting, seed, path = args
    rng = np.random.default_rng(seed)
    if np.all(np.isfinite(uv)):
        geo = T.render_geometry(cam, XA, XB, uv[0], uv[1])
        img = T.compose(cam, geo, lighting, T.ball_blur(cam, XA), T.ball_blur(cam, XB), rng=rng)
    else:
        out = np.full((cam.height, cam.width), T.BG[lighting], np.float64) + rng.normal(0.0, T.NOISE,
                                                                                      (cam.height, cam.width))
        img = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)
    return path


def render_set(name: str, n_frames: int, seed: int, lighting: str = "backlit", workers: int = None) -> Path:
    """Render (once) an image set into data/<name>/cam1..cam3/frame_NNNN.png with truth.npz: the exact images of
    the balls' centres (uv, NaN where a camera does not see both balls whole), the wand ends and the cameras."""
    from concurrent.futures import ProcessPoolExecutor
    folder = DATA / name
    truth = folder / "truth.npz"
    if truth.exists():
        return folder
    cams = T.cam1_rig()
    rng = np.random.default_rng(seed)
    XA, XB, UV = T.wand_frames(cams, n_frames, rng)
    tasks = []
    for c in range(len(cams)):
        (folder / f"cam{c + 1}").mkdir(parents=True, exist_ok=True)
        for f in range(n_frames):
            tasks.append((cams[c], XA[f], XB[f], UV[f, c], lighting, [seed, f, c],
                          str(folder / f"cam{c + 1}" / f"frame_{f:04d}.png")))
    with ProcessPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 4) - 2)) as ex:
        list(ex.map(_render_one, tasks, chunksize=4))
    np.savez(truth, XA=XA, XB=XB, UV=UV, f=[c.f for c in cams], C=[c.C for c in cams], R=[c.R for c in cams],
             width=cams[0].width, height=cams[0].height, lighting=lighting, seed=seed, L=T.L_WAND,
             radii=T.TUBE.radii, n=T.TUBE.n)
    return folder


def load_truth(folder: Path) -> dict:
    z = np.load(folder / "truth.npz")
    return {k: z[k] for k in z.files}


def image_lists(folder: Path) -> dict:
    """{camera index: sorted image paths}, as the GUI's per-camera Load button collects them."""
    out = {}
    for c in range(3):
        d = folder / f"cam{c + 1}"
        out[c] = sorted(str(p) for p in d.iterdir() if p.suffix.lower() == ".png")
    return out


# ------------------------------------------------------------------ offscreen Qt
def qt_app(offscreen: bool = True):
    if offscreen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    import openlpt  # noqa: F401  (the GUI's entry: build-py/Release first on sys.path)
    import app as gui_app                       # gui/app.py: QtWebEngine prepared as the GUI's main() does
    gui_app.prepare_qtwebengine()
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


class Dialogs:
    """Answers the GUI's message boxes and file dialogs: every message is recorded; questions get `answer`."""

    def __init__(self, folder: str = None, save_file: str = None, answer: str = "No"):
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        self.messages = []
        self.folder, self.save_file = folder, save_file
        btn = getattr(QMessageBox.StandardButton, answer)

        def box(kind):
            def f(*a, **k):
                self.messages.append((kind, str(a[1]) if len(a) > 1 else "", str(a[2]) if len(a) > 2 else ""))
                return btn if kind == "question" else QMessageBox.StandardButton.Ok
            return f
        QMessageBox.information = box("info")
        QMessageBox.warning = box("warning")
        QMessageBox.critical = box("critical")
        QMessageBox.question = box("question")
        QFileDialog.getExistingDirectory = lambda *a, **k: self.folder
        QFileDialog.getSaveFileName = lambda *a, **k: (self.save_file, "CSV Files (*.csv)")


def wait_for(predicate, timeout_s: float = 900.0, step_ms: int = 100) -> bool:
    """Run the Qt event loop until predicate() is True (or the time is up); returns predicate()."""
    from PySide6.QtCore import QEventLoop, QTimer
    loop = QEventLoop()
    t0 = time.time()

    def poll():
        if predicate() or time.time() - t0 > timeout_s:
            loop.quit()
    tm = QTimer()
    tm.timeout.connect(poll)
    tm.start(step_ms)
    loop.exec()
    tm.stop()
    return bool(predicate())


def cylinder_done(view) -> bool:
    w = getattr(view, "_cyl_worker", None)
    return w is not None and w.isFinished() and getattr(view, "_cyl_dialog", None) is None
