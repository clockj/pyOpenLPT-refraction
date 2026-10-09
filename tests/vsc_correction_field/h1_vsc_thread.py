"""H1: the tracking view lets go of the VSC's worker thread only after it has ended (decision 3, 2026-10-09-plan.md).

    python h1_vsc_thread.py                       # every case in its own child process; the verdict
    python h1_vsc_thread.py child CODE MODE N     # one case: CODE today | change, MODE fake | real, N rounds

The tracking view on Qt's offscreen platform (build vcf's pyopenlpt) presses Run VSC with message boxes that return
at once. A harness has no modal loop; the GUI's boxes run one, and that hid the race. When the worker's finished
signal reaches _on_vsc_finished, its QThread may still run, and letting go of a running QThread aborts the process.
- CODE today: gui/views/tracking_view.py as commit cffed7d has it; change: the working tree's.
- MODE fake: a stand-in VSC service that fails at once (the view's failure branch), on projects/t1_g3 (unchanged).
- MODE real: the field VSC on a fresh T1 project (projects/t1_h1, made by g1_project.py), its success branch.
A case passes when its child exits 0 after N rounds with the view idle again. Expected: today's code crashes (the
race is deterministic: the thread's quit is queued behind the handler); the change passes every case.
Writes h1_vsc_thread_out.txt and .json here.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import vcf_common as vc  # noqa: E402


class Immediate:
    """QMessageBox's static calls, recorded and returning at once (no event loop)."""
    shown = []
    Ok = Yes = No = Cancel = Close = 0   # the button constants the views pass

    @staticmethod
    def _rec(kind, title, text):
        Immediate.shown.append({"kind": kind, "title": title, "text": text})
        return 0

    @staticmethod
    def information(parent, title, text, *a, **k):
        return Immediate._rec("information", title, text)

    @staticmethod
    def warning(parent, title, text, *a, **k):
        return Immediate._rec("warning", title, text)

    @staticmethod
    def critical(parent, title, text, *a, **k):
        return Immediate._rec("critical", title, text)

    @staticmethod
    def question(parent, title, text, *a, **k):
        return Immediate._rec("question", title, text)


class FakeService:
    """Stands in for modules.vsc.VSCService: fails at once."""

    def __init__(self, proj_dir, log_callback=None):
        self.log = log_callback

    def set_params(self, **kwargs):
        pass

    def run(self):
        if self.log:
            self.log("H1: a stand-in VSC service")
        return False, "a stand-in VSC service (H1)", {}


def child(code, mode, rounds):
    lpt = vc.lib("vcf")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    tv = vc.committed_module("gui/views/tracking_view.py", "tracking_view_today") if code == "today" else \
        __import__("gui.views.tracking_view", fromlist=["TrackingView"])
    tv.QMessageBox = Immediate
    if mode == "fake":
        import modules.vsc as mv
        mv.VSCService = FakeService
        proj = os.path.join(vc.HERE, "projects", "t1_g3")
    else:
        proj = os.path.join(vc.HERE, "projects", "t1_h1")
    view = tv.TrackingView(settings_view=None)
    view.proj_path_edit.setText(proj)
    app.processEvents()
    print(f"[h1] {code} {mode}: {tv.__file__}, pyopenlpt {lpt.__file__}, project {proj}", flush=True)
    for r in range(rounds):
        t0 = time.time()
        view._run_vsc()
        while view.vsc_worker is not None and time.time() - t0 < 1800:
            app.processEvents()
            time.sleep(0.01)
        for _ in range(20):
            app.processEvents()
            time.sleep(0.01)
        print(f"[h1] round {r + 1}: idle again after {time.time() - t0:.1f} s; boxes "
              f"{[m['title'] for m in Immediate.shown]}", flush=True)
    for m in Immediate.shown:
        print(f"[h1] box ({m['kind']}) '{m['title']}': {m['text'][:300]}", flush=True)
    print("[h1] done", flush=True)


def parent():
    log = vc.Log("h1_vsc_thread")
    cases = [("today", "fake", 1), ("change", "fake", 5), ("change", "real", 1)]
    out = {"cases": []}
    for code, mode, n in cases:
        t0 = time.time()
        p = subprocess.run([sys.executable, __file__, "child", code, mode, str(n)], cwd=vc.HERE, capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           env=dict(os.environ, OMP_NUM_THREADS=str(vc.THREADS_MAX), PYTHONDONTWRITEBYTECODE="1"))
        text = p.stdout + p.stderr
        with open(os.path.join(vc.HERE, f"h1_vsc_thread_{code}_{mode}_console.txt"), "w", encoding="utf-8") as fh:
            fh.write(text)
        done = "[h1] done" in text
        rounds = sum(1 for ln in text.splitlines() if ln.startswith("[h1] round"))
        qt = [ln.strip() for ln in text.splitlines() if "QThread" in ln]
        rec = {"code": code, "mode": mode, "rounds_asked": n, "rounds_done": rounds, "exit": p.returncode,
               "finished": done, "qt_messages": qt, "seconds": round(time.time() - t0, 1),
               "boxes": [ln for ln in text.splitlines() if ln.startswith("[h1] box")]}
        out["cases"].append(rec)
        log(f"{code}'s code, {mode} service, {n} round(s): exit {p.returncode} ({p.returncode & 0xFFFFFFFF:#x}), "
            f"{rounds} round(s) done, {'finished' if done else 'did NOT finish'}"
            + (f"; Qt: {qt[0]}" if qt else "") + f" ({rec['seconds']} s)")
        for b in rec["boxes"]:
            log(f"  {b}")
    today = out["cases"][0]
    ok = (today["exit"] != 0 and not today["finished"]
          and all(c["exit"] == 0 and c["finished"] and c["rounds_done"] == c["rounds_asked"]
                  for c in out["cases"][1:]))
    out["pass"] = bool(ok)
    log(f"H1: {'PASS' if ok else 'FAIL'} (today's code must crash, the change must pass every case)")
    log.save(out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        parent()
