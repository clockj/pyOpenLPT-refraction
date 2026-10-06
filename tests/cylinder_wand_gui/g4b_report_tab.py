"""G4b: the Report tab shows the results page inside the GUI (QtWebEngine).

Qt's offscreen platform cannot run QtWebEngine here (its GPU process fails and the page never loads:
g4b_report_tab_offscreen_out.txt), so this test uses the native platform, as the GUI does, with the window placed
off the screen.

    python g4b_report_tab.py [page.html]

The page of G3's run (data/g3_run/results/calibration_report.html, or the one given) is loaded into the Wand tab's
Report tab (ResultsReportTab) of a CameraCalibrationView; the test waits for the page to finish loading and for its
script to report that it drew everything (window.__reportReady), reads the JavaScript console, and grabs the widget
as an image (g4b_report_tab.png).
Writes g4b_report_tab_out.txt and g4b_report_tab.json.
"""
import common as C  # noqa: I001

import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

log = C.Log("g4b_report_tab")


def main():
    app = C.qt_app(offscreen=False)
    page = sys.argv[1] if len(sys.argv) > 1 else str(C.DATA / "g3_run" / "results" / "calibration_report.html")
    from modules.camera_calibration import CameraCalibrationView
    view = CameraCalibrationView()
    view.resize(1500, 1000)
    view.move(-6000, -6000)                                      # native window, off the screen
    view.show()
    app.processEvents()
    tab = view.report_tab
    log.check("the Report tab exists and uses QtWebEngine", tab is not None and tab.web is not None,
              f"web view: {type(tab.web).__name__ if tab and tab.web is not None else None}")
    if tab is None or tab.web is None:
        return log.save()
    messages = []
    from PySide6.QtWebEngineCore import QWebEnginePage

    class Page(QWebEnginePage):
        def javaScriptConsoleMessage(self, level, message, line, source):
            messages.append((int(level.value) if hasattr(level, "value") else int(level), message, line))
    tab.web.setPage(Page(tab.web))
    loaded = {"ok": None}
    tab.web.loadFinished.connect(lambda ok: loaded.__setitem__("ok", ok))
    view.tabs.setCurrentIndex(1)                                 # the Wand Calibration tab, as a user would be on it
    view.vis_tabs.setCurrentWidget(tab)
    app.processEvents()
    t0 = time.time()
    tab.show_page(page)
    C.wait_for(lambda: loaded["ok"] is not None, timeout_s=60)
    ready = {"v": None}

    def poll_ready():
        tab.web.page().runJavaScript("window.__reportReady === true && document.querySelectorAll('svg').length",
                                     0, lambda v: ready.__setitem__("v", v))
    for _ in range(100):
        poll_ready()
        C.wait_for(lambda: ready["v"] is not None, timeout_s=2, step_ms=50)
        if ready["v"]:
            break
        ready["v"] = None
        time.sleep(0.1)
    log.check("the page loads and its script draws every figure", bool(loaded["ok"]) and bool(ready["v"]),
              f"loadFinished {loaded['ok']}, svg elements {ready['v']}, {time.time() - t0:.1f} s")
    errors = [m for m in messages if m[0] >= 2]
    log.check("no JavaScript errors", not errors, f"{len(messages)} console messages, errors {errors[:3]}")
    theme = {"v": None}
    tab.web.page().runJavaScript("document.documentElement.getAttribute('data-theme')", 0,
                                 lambda v: theme.__setitem__("v", v))
    C.wait_for(lambda: theme["v"] is not None, timeout_s=5)
    log.check("inside the GUI the page is dark", theme["v"] == "dark", f"data-theme {theme['v']!r}")
    C.wait_for(lambda: False, timeout_s=1.5)                     # let it paint
    img = tab.web.grab()
    out = str(C.HERE / "g4b_report_tab.png")
    img.save(out)
    qi = img.toImage()
    colours = {qi.pixelColor(x, y).name() for x in range(0, qi.width(), max(1, qi.width() // 40))
               for y in range(0, qi.height(), max(1, qi.height() // 40))}
    log.check("the widget's image shows the page (more than a blank fill)", img.width() > 600 and len(colours) > 5,
              f"{img.width()} x {img.height()}, {len(colours)} distinct colours sampled -> {out}")
    log(f"  buttons enabled: browser {tab.btn_browser.isEnabled()}, folder {tab.btn_folder.isEnabled()}; label "
        f"{tab.path_label.text()}")
    view.close()
    return log.save({"console": messages[:20]})


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
