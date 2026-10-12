"""T2/T3's check of an installed environment: run with that environment's python.exe by its path, in isolated mode
and without `conda activate`, as the GUI's desktop shortcut starts it.

    <env>\\python.exe -I t2_env_check.py OUT.json PAGE.html

  - the interpreter (3.11), and the BLAS that conda installed (libblas's build: OpenBLAS, not MKL);
  - numpy's linear algebra (solve, inv, lstsq, eigvals: with MKL these killed the interpreter in refract);
  - every dependency imports: scipy, cv2, sklearn, matplotlib, pandas, numba (one jitted function), PIL, requests,
    qtawesome, pycine, win32api, PySide6; the C++ extension pyopenlpt and the package openlpt, from site-packages;
  - QtWebEngine as the GUI sets it up (gui/app.py prepare_qtwebengine, then the application): the camera
    calibration's Report tab loads PAGE (a results page with its charts) and draws it (as G4b), on the native
    platform with the window off the screen;
  - the main window builds (as G6, the update check off) and closes.
Writes OUT.json: {"checks": [[name, ok, detail], ...], "pass": bool}; exit 0 when every check passes.
"""
import glob
import json
import os
import sys
import time
import traceback

OUT, PAGE = sys.argv[1], sys.argv[2]
checks = []


def check(name, fn):
    try:
        ok, detail = fn()
    except Exception:                                           # noqa: BLE001
        ok, detail = False, traceback.format_exc(limit=3)
    checks.append([name, bool(ok), str(detail)])
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    return ok


def interpreter():
    return sys.version_info[:2] == (3, 11), f"{sys.version.split()[0]} at {sys.executable}"


def blas():
    metas = [os.path.basename(p) for p in glob.glob(os.path.join(sys.prefix, "conda-meta", "libblas-*.json"))]
    return len(metas) == 1 and "openblas" in metas[0], metas


def linalg():
    import numpy as np
    a = np.array([[3.0, 1.0], [1.0, 2.0]])
    x = np.linalg.solve(a, [9.0, 8.0])
    rng = np.random.default_rng(0)
    m = rng.standard_normal((200, 200)) + 200 * np.eye(200)
    r = np.linalg.solve(m, np.ones(200))
    ok = (np.allclose(x, [2.0, 3.0]) and np.allclose(np.linalg.inv(a) @ a, np.eye(2))
          and np.allclose(m @ r, 1.0) and np.linalg.lstsq(m[:, :50], np.ones(200), rcond=None)[0].shape == (50,)
          and np.linalg.eigvals(a).shape == (2,))
    return ok, f"numpy {np.__version__}: solve, inv, lstsq, eigvals"


def imports():
    import importlib
    import importlib.metadata as md
    got = {}
    for mod, dist in (("scipy", "scipy"), ("cv2", None), ("sklearn", "scikit-learn"), ("matplotlib", "matplotlib"),
                      ("pandas", "pandas"), ("numba", "numba"), ("PIL", "pillow"), ("requests", "requests"),
                      ("qtawesome", "qtawesome"), ("pycine", "pycine"), ("win32api", None), ("PySide6", None)):
        m = importlib.import_module(mod)
        got[mod] = getattr(m, "__version__", None) or (md.version(dist) if dist else "ok")
    import numba

    @numba.njit
    def add(a, b):
        return a + b
    return add(2, 3) == 5, got


def extension():
    import openlpt
    import pyopenlpt
    site = os.path.join(sys.prefix, "Lib", "site-packages")
    ok = (openlpt.pyopenlpt is not None and os.path.dirname(pyopenlpt.__file__) == site
          and os.path.dirname(openlpt.__file__) == site and hasattr(pyopenlpt, "Camera"))
    return ok, f"pyopenlpt {pyopenlpt.__file__}; openlpt {openlpt.__file__} ({openlpt.__version__})"


def wait_for(app, cond, timeout_s, step_s=0.05):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        app.processEvents()
        if cond():
            return True
        time.sleep(step_s)
    return cond()


def gui():
    import openlpt
    sys.path.insert(0, os.path.join(os.path.dirname(openlpt.__file__), "gui"))     # as openlpt.launch_gui does
    import app as guiapp
    state = {}
    check("QtWebEngine prepared (gui/app.py prepare_qtwebengine)", lambda: (guiapp.prepare_qtwebengine(), "ok"))
    from PySide6.QtWidgets import QApplication
    qapp = QApplication.instance() or QApplication([])
    from modules.camera_calibration import CameraCalibrationView

    def report():
        view = CameraCalibrationView()
        view.resize(1500, 1000)
        view.move(-6000, -6000)                                  # native window, off the screen
        view.show()
        qapp.processEvents()
        tab = view.report_tab
        if tab is None or tab.web is None:
            return False, "no web view in the Report tab"
        loaded = {"ok": None}
        tab.web.loadFinished.connect(lambda ok: loaded.__setitem__("ok", ok))
        for i in range(view.tabs.count()):             # the Wand Calibration tab, which holds the Report tab (as
            if view.tabs.widget(i).isAncestorOf(tab):   # G4b; hidden, the view grabs blank: first check, t2_*_first*)
                view.tabs.setCurrentIndex(i)
        view.vis_tabs.setCurrentWidget(tab)
        tab.show_page(PAGE)
        wait_for(qapp, lambda: loaded["ok"] is not None, 60)
        ready = {"v": None}
        for _ in range(100):
            tab.web.page().runJavaScript("window.__reportReady === true && document.querySelectorAll('svg').length",
                                         0, lambda v: ready.__setitem__("v", v))
            wait_for(qapp, lambda: ready["v"] is not None, 2)
            if ready["v"]:
                break
            ready["v"] = None
            time.sleep(0.1)
        wait_for(qapp, lambda: False, 1.5)                       # let it paint
        img = tab.web.grab().toImage()
        colours = {img.pixelColor(x, y).name() for x in range(0, img.width(), max(1, img.width() // 40))
                   for y in range(0, img.height(), max(1, img.height() // 40))}
        view.close()
        state["report"] = (loaded["ok"], ready["v"], len(colours))
        return (bool(loaded["ok"]) and bool(ready["v"]) and len(colours) > 5,
                f"loadFinished {loaded['ok']}, svg elements {ready['v']}, {len(colours)} colours sampled")
    check("the Report tab loads the results page and draws it", report)

    def main_window():
        guiapp.OpenLPTMainWindow._check_for_updates = lambda self: None
        w = guiapp.OpenLPTMainWindow()
        w.move(-6000, -6000)
        w.show()
        qapp.processEvents()
        n = len(w.findChildren(CameraCalibrationView))
        w.close()
        return n == 1, f"{n} Camera Calibration view"
    check("the main window builds", main_window)
    return True, state


def main():
    check("interpreter", interpreter)
    check("libblas is OpenBLAS (conda-meta)", blas)
    check("numpy's linear algebra", linalg)
    check("the dependencies import", imports)
    check("pyopenlpt and openlpt from site-packages", extension)
    check("the GUI parts ran", gui)
    ok = all(c[1] for c in checks)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"checks": checks, "pass": ok, "prefix": sys.prefix}, fh, indent=1)
    print("ALL PASS" if ok else "SOME FAILED", flush=True)
    return ok


if __name__ == "__main__":
    code = 0 if main() else 1
    sys.stdout.flush()
    os._exit(code)          # skip Qt's teardown (G6: the full window can exit with 0xC0000409 at shutdown)
