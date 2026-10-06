"""G4: the three results pages (cylinder G3, flat windows G8, pinhole G9; plan section 5 and O4).

    python g4_pages.py

For each page (run G3, G8 and G5/G9 first):
  - its numbers are the result's: the summary's RMS, detections, frames and wand error equal calibration_result.json
    (cylinder, flat windows), and every camera's f;
  - the lines of sight it draws are physical: each polyline's corners lie on the interfaces (tube: the outer and inner
    radii; flat windows: the two surfaces) within 1e-9 mm, and Snell's law holds at each corner within 1e-9;
  - in QtWebEngine (the GUI's Report tab engine, native platform, window off the screen): the page loads, its script
    draws every figure, no console error, no request other than the page itself; at 390 px wide nothing overflows
    horizontally; dark and light themes both render.
Writes g4_pages_out.txt and g4_pages.json, and g4_<page>_<theme>.png.
"""
import common as C  # noqa: I001

import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

log = C.Log("g4_pages")
PAGES = {"cylinder": C.DATA / "g3_run" / "results", "plate": C.DATA / "g8_run" / "results",
         "pinhole": C.DATA / "g5_run" / "pinhole"}


def embedded(path):
    txt = open(path, encoding="utf-8").read()
    m = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', txt, re.S)
    return json.loads(m.group(1).replace("<\\/", "</"))


def snell_ok(d_in, d_out, normal, n1, n2):
    """Vector Snell: n1 (d_in x N) = n2 (d_out x N), both unit, same side."""
    d_in, d_out = d_in / np.linalg.norm(d_in), d_out / np.linalg.norm(d_out)
    return float(np.linalg.norm(n1 * np.cross(d_in, normal) - n2 * np.cross(d_out, normal)))


def check_lines(rep):
    port = rep.get("port")
    worst_surface, worst_snell, n_lines = 0.0, 0.0, 0
    for _cam, line in rep.get("sight_lines", []):
        P = [np.array(p, float) for p in line]
        n_lines += 1
        if port is None:
            continue
        if port["type"] == "tube":
            R, nn = port["radii"], port["n"]
            for k, p in enumerate(P[1:-1]):
                worst_surface = max(worst_surface, abs(math.hypot(p[0], p[2]) - R[k]))
                N = np.array([p[0], 0.0, p[2]]) / math.hypot(p[0], p[2])
                worst_snell = max(worst_snell, snell_ok(P[k + 1] - P[k], P[k + 2] - P[k + 1], N, nn[k], nn[k + 1]))
        else:
            wins = port["windows"]
            best = None
            for w in wins:                                  # the window this line crosses: its first corner's plane
                n = np.array(w["normal"], float)
                d = float(n @ np.array(w["point"], float))
                if abs(float(n @ P[1]) - d) < 1e-6:
                    best = (n, d, w)
            if best is None:
                worst_surface = max(worst_surface, 1.0)
                continue
            n, d, w = best
            worst_surface = max(worst_surface, abs(float(n @ P[2]) - (d + w["thickness"])))
            nn = w["n"]
            worst_snell = max(worst_snell, snell_ok(P[1] - P[0], P[2] - P[1], n, nn[0], nn[1]),
                              snell_ok(P[2] - P[1], P[3] - P[2], n, nn[1], nn[2]))
    return n_lines, worst_surface, worst_snell


def main():
    app = C.qt_app(offscreen=False)
    from PySide6.QtCore import QUrl
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineUrlRequestInterceptor
    from PySide6.QtWebEngineWidgets import QWebEngineView
    out = {}
    for name, folder in PAGES.items():
        page = folder / "calibration_report.html"
        if not page.exists():
            log.check(f"{name}: page exists", False, f"{page} (run its test first)")
            continue
        rep = embedded(page)
        S = rep["summary"]
        log(f"\n{name}: {page}")
        res_path = folder / "calibration_result.json"
        if res_path.exists():
            res = json.load(open(res_path, encoding="utf-8"))
            w = res["wand_length_free_triangulation"]
            same = (S["rms_px"] == res["rms_px"] and S["detections"] == res["observations"]
                    and S["detections_used"] == res["active"] and abs(S["wand_rms_um"] - 1e3 * w["rms_mm"]) < 1e-9
                    and all(c["f_px"] == r["f_px"] for c, r in zip(rep["cameras"], res["cameras"])))
            log.check(f"{name}: the page's numbers are the result's", same,
                      f"RMS {S['rms_px']:.6f} px, {S['detections_used']}/{S['detections']}, wand {S['wand_rms_um']:.4f} um")
        n_lines, ws, wn = check_lines(rep)
        if rep.get("port"):
            log.check(f"{name}: its lines of sight are physical (on the interfaces, Snell) within 1e-9",
                      ws < 1e-9 and wn < 1e-9, f"{n_lines} lines; surfaces {ws:.1e} mm, Snell {wn:.1e}")
        # QtWebEngine
        requests = []

        class Spy(QWebEngineUrlRequestInterceptor):
            def interceptRequest(self, info):
                requests.append(info.requestUrl().toString())
        msgs = []

        class Page(QWebEnginePage):
            def javaScriptConsoleMessage(self, level, message, line, source):
                msgs.append((int(level.value) if hasattr(level, "value") else int(level), message))
        for theme, width in (("dark", 1200), ("light", 1200), ("dark", 390)):
            v = QWebEngineView()
            pg = Page(v)
            spy = Spy()
            pg.profile().setUrlRequestInterceptor(spy)
            v.setPage(pg)
            v.resize(width, 900)
            v.move(-6000, -6000)
            v.show()
            done = {}
            v.loadFinished.connect(lambda ok, d=done: d.__setitem__("ok", ok))
            url = QUrl.fromLocalFile(str(page))
            url.setQuery(f"theme={theme}")
            v.load(url)
            C.wait_for(lambda: "ok" in done, timeout_s=30)
            got = {}
            pg.runJavaScript("JSON.stringify([window.__reportReady === true, document.querySelectorAll('svg').length, "
                             "document.documentElement.scrollWidth, window.innerWidth, "
                             "document.documentElement.getAttribute('data-theme')])", 0,
                             lambda r, g=got: g.__setitem__("r", r))
            C.wait_for(lambda: "r" in got, timeout_s=10)
            ready, n_svg, sw, iw, th = json.loads(got.get("r") or "[false,0,0,0,null]")
            C.wait_for(lambda: False, timeout_s=0.8)
            img = v.grab()
            img.save(str(C.HERE / f"g4_{name}_{theme}_{width}.png"))
            label = f"{name} {theme} {width} px"
            if width == 390:
                log.check(f"{label}: nothing overflows horizontally", sw <= iw + 1, f"scrollWidth {sw}, window {iw}")
            else:
                log.check(f"{label}: loads, every figure drawn, theme applied", bool(done.get("ok")) and ready and th == theme,
                          f"svg {n_svg}, theme {th}")
            v.close()
            pg.profile().setUrlRequestInterceptor(None)
        errors = [m for m in msgs if m[0] >= 2]
        others = [r for r in requests if not r.startswith("file:///")]
        log.check(f"{name}: no console error, no request but the local page", not errors and not others,
                  f"{len(msgs)} console messages, errors {errors[:2]}; non-file requests {others[:3]}")
        out[name] = {"lines": n_lines, "surface_mm": ws, "snell": wn, "requests": requests[:5]}
    return log.save({"pages": out})


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
