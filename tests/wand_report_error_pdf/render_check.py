"""The error charts on the results page (2026-10-07-plan.md, reduced on 7 Oct to this check): render the pages of
saved runs with the new page.py and template.html, load each in QtWebEngine (the Report tab's engine; the native
platform, with the window off the screen) and check that it draws; grab the new section for a look.

    python render_check.py

Pages (written to data/, not tracked):
  - g3: tests/cylinder_wand_gui's G3 run (synthetic tube, 3 cameras);
  - g8: its G8 run (synthetic flat windows, 4 cameras);
  - pinhole: a synthetic 3-camera pinhole rig (tests/wand_gui_fixes W9's), with its true cameras;
  - t0, t2: the real tube runs 20261006/T0 and T2 (read from X:; nothing is written there);
  - old: G3's report without the new data, as a GUI started before the change writes it.
Per page: the numbers printed (ranges, bins, each camera's n and RMS beside the cameras table's); in QtWebEngine it
loads and its script finishes (window.__reportReady), no console error, no request but the page itself, the error
section has 3 charts plus one du-against-dv chart per camera (no section on the old page), the Log switch redraws,
a pointer over a chart shows the tooltip, nothing overflows at 390 px; dark and light.
Writes render_check_out.txt, render_check.json and render_<page>_<theme>[_log].png (the section).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE / "data"
X_RUNS = {"t0": Path(r"X:\Shijie Zhong\Refraction\Processed\20261006\T0\Results"),
          "t2": Path(r"X:\Shijie Zhong\Refraction\Processed\20261006\T2\Results")}
sys.path.insert(0, str(REPO / "tests" / "wand_gui_fixes"))      # its common (qt_app, wait_for) and W9's rig
import common as C  # noqa: E402,I001

LINES, CHECKS = [], []


def say(msg=""):
    print(msg, flush=True)
    LINES.append(str(msg))


def check(label, ok, detail=""):
    CHECKS.append({"check": label, "ok": bool(ok), "detail": detail})
    say(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))


def reports():
    from modules.camera_calibration.results_page import page as rp
    from w9_w10_pinhole import gui_points, rig
    out = {}
    for name, folder, fn in (("g3", REPO / "tests/cylinder_wand_gui/data/g3_run/results", rp.report_from_cylinder),
                             ("g8", REPO / "tests/cylinder_wand_gui/data/g8_run/results", rp.report_from_plate),
                             ("t0", X_RUNS["t0"], rp.report_from_cylinder),
                             ("t2", X_RUNS["t2"], rp.report_from_cylinder)):
        res = json.load(open(folder / "calibration_result.json", encoding="utf-8"))
        data = json.load(open(folder / "calibration_data.json", encoding="utf-8"))
        out[name] = fn(res, data, str(folder))
    final, uv, _a, _b, _on = rig(np.random.default_rng(9), [(1280, 800)] * 3)
    out["pinhole"] = rp.report_from_pinhole(final, gui_points(uv), 10.0, folder=str(DATA / "pinhole"),
                                            detect_mode="synthetic (W9's rig, true cameras)")
    old = json.loads(json.dumps(rp._clean(out["g3"])))
    old.pop("error_pdf")
    out["old"] = old
    for name, rep in out.items():
        rp.write_page(rep, str(DATA / f"{name}.html"))
    return out


def numbers(name, rep):
    E = rep.get("error_pdf")
    if E is None:
        say(f"\n{name}: no error_pdf")
        return
    say(f"\n{name}: du, dv over +-{E['uv']['hi']:g} px in {E['uv']['bins']} bins of {E['uv']['width']:g} px; |e| to "
        f"{E['mag']['hi']:g} px in {E['mag']['bins']} bins of {E['mag']['width']:g}; joint {E['joint']['bins']} cells "
        f"of {E['joint']['width']:g} px, max {E['joint']['max']} per cell")
    for cam, c in zip(rep["cameras"], E["cams"]):
        if not c["n"]:
            say(f"  {cam['name']}: no detections")
            continue
        say(f"  {cam['name']}: {c['n']} of {c['n_total']} used; du mean {c['mean_du']:+.5f} rms {c['rms_du']:.5f}; "
            f"dv mean {c['mean_dv']:+.5f} rms {c['rms_dv']:.5f}; |e| median {c['median']:.5f} rms {c['rms']:.5f} "
            f"(cameras table {cam['rms_px']:.5f}) p95 {c['p95']:.5f} max {c['max']:.5f}; beyond du/dv/|e|/joint "
            f"{c['beyond_du']}/{c['beyond_dv']}/{c['beyond_mag']}/{c['beyond_joint']}")


def main():
    t_start = time.time()
    DATA.mkdir(exist_ok=True)
    app = C.qt_app(offscreen=False)
    reps = reports()
    for name, rep in reps.items():
        numbers(name, rep)
    from PySide6.QtCore import QUrl
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineUrlRequestInterceptor
    from PySide6.QtWebEngineWidgets import QWebEngineView
    out = {}
    for name, rep in reps.items():
        page = DATA / f"{name}.html"
        E = rep.get("error_pdf")
        n_cams = sum(1 for c in (E or {}).get("cams", []) if c["n"])
        want = 3 + n_cams if E else 0
        say(f"\n{name}: {page.name}")
        requests, msgs = [], []

        class Spy(QWebEngineUrlRequestInterceptor):
            def interceptRequest(self, info):
                requests.append(info.requestUrl().toString())

        class Page(QWebEnginePage):
            def javaScriptConsoleMessage(self, level, message, line, source):
                msgs.append((int(level.value) if hasattr(level, "value") else int(level), message, line))
        res = {}
        for theme, width in (("dark", 1200), ("light", 1200), ("dark", 390)):
            v = QWebEngineView()
            pg = Page(v)
            spy = Spy()
            pg.profile().setUrlRequestInterceptor(spy)
            v.setPage(pg)
            v.resize(width, 1700 if width > 400 else 900)
            v.move(-6000, -6000)
            v.show()
            done = {}
            v.loadFinished.connect(lambda ok, d=done: d.__setitem__("ok", ok))
            url = QUrl.fromLocalFile(str(page))
            url.setQuery(f"theme={theme}")
            v.load(url)
            C.wait_for(lambda: "ok" in done, timeout_s=30)

            def js(code, timeout_s=10):
                got = {}
                pg.runJavaScript(code, 0, lambda r, g=got: g.__setitem__("r", r))
                C.wait_for(lambda: "r" in got, timeout_s=timeout_s)
                return got.get("r")
            for _ in range(100):                              # the page's script may still be drawing
                if js("window.__reportReady === true") is True:
                    break
                time.sleep(0.1)
            st = json.loads(js("JSON.stringify([window.__reportReady === true, "
                               "document.querySelectorAll('#errors svg').length, "
                               "document.documentElement.scrollWidth, window.innerWidth, "
                               "document.documentElement.getAttribute('data-theme'), "
                               "document.querySelectorAll('#errors polyline').length])") or "[false,0,0,0,null,0]")
            ready, n_svg, sw, iw, th, n_lines = st
            label = f"{name} {theme} {width} px"
            if width < 400:
                check(f"{label}: nothing overflows horizontally", sw <= iw + 1, f"scrollWidth {sw}, window {iw}")
            else:
                check(f"{label}: loads, the error section drawn ({want} charts), theme applied",
                      bool(done.get("ok")) and ready and n_svg == want and th == theme and (n_lines > 0 or not E),
                      f"charts {n_svg}, curves {n_lines}, theme {th}")
            if E and theme == "dark" and width > 400:
                tip = js("""(function () {
                  const r = document.querySelector('#errors svg rect[pointer-events="all"]');
                  const b = r.getBoundingClientRect();
                  r.dispatchEvent(new PointerEvent('pointermove', {clientX: b.left + 0.5 * b.width,
                    clientY: b.top + 0.5 * b.height, bubbles: true}));
                  const t = document.getElementById('tip');
                  const seen = t.style.display + '|' + t.innerText.replace(/\\n/g, ' / ');
                  r.dispatchEvent(new PointerEvent('pointerleave'));       // the crosshair and the tooltip go
                  return seen;
                })()""")
                check(f"{label}: a pointer over the du chart shows the tooltip", str(tip).startswith("block|du "),
                      str(tip)[:160])
                js("document.getElementById('tip').style.display = 'none'; "
                   "document.getElementById('errors').scrollIntoView(); true")
                C.wait_for(lambda: False, timeout_s=0.8)
                v.grab().save(str(HERE / f"render_{name}_{theme}.png"))
                n_log = js("""(function () {
                  [...document.querySelectorAll('#errors button')].find(b => b.textContent === 'Log').click();
                  return document.querySelectorAll('#errors polyline, #errors circle[r="2.5"]').length;
                })()""")
                C.wait_for(lambda: False, timeout_s=0.6)
                v.grab().save(str(HERE / f"render_{name}_{theme}_log.png"))
                check(f"{label}: the Log switch redraws the curves", isinstance(n_log, (int, float)) and n_log > 0,
                      f"{n_log} curves and points on the log scale")
            if not E and theme == "dark" and width > 400:
                js("document.getElementById('residuals').scrollIntoView(); true")
                C.wait_for(lambda: False, timeout_s=0.8)
                v.grab().save(str(HERE / f"render_{name}_{theme}.png"))
            if E and theme == "light" and width > 400:
                js("document.getElementById('errors').scrollIntoView(); true")
                C.wait_for(lambda: False, timeout_s=0.8)
                v.grab().save(str(HERE / f"render_{name}_{theme}.png"))
            res[label] = st
            v.close()
            pg.profile().setUrlRequestInterceptor(None)
        errors = [m for m in msgs if m[0] >= 2]
        others = [r for r in requests if not r.startswith("file:///")]
        check(f"{name}: no console error, no request but the local page", not errors and not others,
              f"{len(msgs)} console messages, errors {errors[:2]}; other requests {others[:3]}")
        out[name] = res
    n_fail = sum(not c["ok"] for c in CHECKS)
    say(f"\n{len(CHECKS) - n_fail} of {len(CHECKS)} checks passed; {time.time() - t_start:.0f} s")
    with open(HERE / "render_check_out.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    with open(HERE / "render_check.json", "w", encoding="utf-8") as fh:
        json.dump({"checks": CHECKS, "states": out}, fh, indent=1, default=str)
    return n_fail == 0


if __name__ == "__main__":
    ok = main()
    os._exit(0 if ok else 1)
