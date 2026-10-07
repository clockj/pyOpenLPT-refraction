"""Shared helpers of the wand-tab fix tests (plan: 2026-10-07-plan.md): paths, the check log, an offscreen Qt
application whose dialogs the test answers, and an event-loop wait.  (As tests/cylinder_wand_gui/common.py, whose
log writes beside that folder's tests.)"""
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
DATA = HERE / "data"
# run 20261006/T2: cam1, cam2, frameNNNNNN.tif (W1 copies six frames into data/w1/); OPENLPT_T2_RAW overrides the place
T2_RAW = Path(os.environ.get("OPENLPT_T2_RAW", r"X:\Shijie Zhong\Refraction\Raw\20261006\T2"))
for _p in (str(REPO / "gui"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


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
                      default=str)
        return n_fail == 0


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

    def __init__(self, folder: str = None, save_file: str = None, open_file: str = None, answer: str = "No"):
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        self.messages = []
        self.folder, self.save_file, self.open_file = folder, save_file, open_file
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
        QFileDialog.getOpenFileName = lambda *a, **k: (self.open_file, "CSV Files (*.csv)")


def wait_for(predicate, timeout_s: float = 900.0, step_ms: int = 50) -> bool:
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


def t2_frames(frames, cams=(1, 2)):
    """{camera index: [paths]} of run T2's raw frames (X:)."""
    return {k: [str(T2_RAW / f"cam{c}" / f"frame{f:06d}.tif") for f in frames] for k, c in enumerate(cams)}
