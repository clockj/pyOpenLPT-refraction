"""W8: the wand tab's control column fits its panels: no horizontal scroll bar on the Point Detection panel, nor on
the Calibration panel with each model (Pinhole, Pinhole+Refraction, Pinhole+Cylinder).

    python w8_panel_widths.py [--tag TAG]

The GUI's main window (gui/app.py) runs on the native platform, placed off the screen (as make_user_guide.py's
shots), at its opening size 1500 x 900 and at its minimum size 1200 x 700 (the control column is 370 px wide at
both; at the smaller height the vertical scroll bar takes its share).  For each panel it prints the scroll area's
viewport width, the content's minimum width, whether the horizontal scroll bar shows, and the panel's widgets whose
right edge passes the viewport (the widest first), to name what makes a panel too wide.
Writes w8_panel_widths[_TAG]_out.txt and .json beside this script.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    name = "w8_panel_widths" + (f"_{args.tag}" if args.tag else "")
    lines, checks, data = [], [], {}

    def log(msg=""):
        print(msg, flush=True)
        lines.append(str(msg))

    def check(label, ok, detail=""):
        checks.append({"check": label, "ok": bool(ok), "detail": detail})
        log(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))

    sys.path[:0] = [str(REPO / "gui"), str(REPO)]
    import openlpt  # noqa: F401  (the GUI's entry: the local pyopenlpt build first on the path)
    import app as guiapp
    guiapp.prepare_qtwebengine()
    from PySide6.QtCore import QEventLoop, QPoint, QTimer
    from PySide6.QtWidgets import (QAbstractScrollArea, QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QLabel,
                                   QPushButton, QScrollArea, QTabWidget, QWidget)
    qapp = QApplication.instance() or QApplication(sys.argv)
    from modules.camera_calibration import CameraCalibrationView

    def settle(s=0.4):
        loop = QEventLoop()
        QTimer.singleShot(int(1000 * s), loop.quit)
        loop.exec()

    guiapp.OpenLPTMainWindow._check_for_updates = lambda self: None
    win = guiapp.OpenLPTMainWindow()
    win.move(-9000, -9000)
    win.show()
    view = win.findChildren(CameraCalibrationView)[0]
    view.tabs.setCurrentIndex(1)                         # Wand Calibration
    p = view.det_scroll.parentWidget()
    while p is not None and not isinstance(p, QTabWidget):
        p = p.parentWidget()
    ctrl = p                                             # Point Detection / Calibration / Tutorial
    cal_scroll = next(s for s in view.findChildren(QScrollArea) if s.widget() and s.widget().isAncestorOf(
        view.cylinder_group))
    leaf_types = (QCheckBox, QComboBox, QLabel, QPushButton, QAbstractSpinBox, QAbstractScrollArea)

    def measure(scroll, what):
        settle()
        content = scroll.widget()
        vw = scroll.viewport().width()
        mw = content.minimumSizeHint().width()
        hbar = scroll.horizontalScrollBar()
        over = []
        for w in content.findChildren(QWidget):
            if not isinstance(w, leaf_types) or not w.isVisibleTo(content):
                continue
            if isinstance(w.parentWidget(), (QComboBox, QAbstractSpinBox, QAbstractScrollArea)):
                continue                                 # parts of a combo, spin box or table
            right = w.mapTo(content, QPoint(w.width(), 0)).x()
            if right > vw:
                txt = w.text() if hasattr(w, "text") and callable(w.text) else ""
                if isinstance(w, QComboBox):
                    txt = w.currentText()
                over.append({"widget": type(w).__name__, "name": w.objectName(), "text": str(txt)[:60],
                             "right": right, "min_w": w.minimumSizeHint().width(), "width": w.width()})
        over.sort(key=lambda o: -o["right"])

        # the rows that set the content's minimum width: every layout item at least `floor` px wide, as a tree
        floor = vw - 60
        rows = []

        def label_of(wid):
            for attr in ("title", "text", "currentText"):
                f = getattr(wid, attr, None)
                if callable(f):
                    try:
                        t = f()
                    except TypeError:
                        continue
                    if t:
                        return f"{type(wid).__name__} '{str(t)[:50]}'"
            return type(wid).__name__

        def walk(layout, depth, every=False):
            """Items at least `floor` wide, and every item of a wide row (a layout that is not a group's)."""
            if layout is None:
                return
            for i in range(layout.count()):
                it = layout.itemAt(i)
                wid, sub = it.widget(), it.layout()
                if wid is not None and not wid.isVisibleTo(content):
                    continue
                mwi = it.minimumSize().width()
                if mwi < floor and not every:
                    continue
                desc = label_of(wid) if wid is not None else type(sub).__name__ if sub is not None else "spacer"
                rows.append({"depth": depth, "item": desc, "min_w": mwi})
                if mwi < floor:
                    continue
                if wid is not None and not isinstance(wid, leaf_types):
                    walk(wid.layout(), depth + 1)
                elif sub is not None:
                    walk(sub, depth + 1, every=True)

        walk(content.layout(), 0)
        rec = {"viewport": vw, "content_min": mw, "content_w": content.width(), "hbar_visible": hbar.isVisible(),
               "hbar_max": hbar.maximum(), "vbar_visible": scroll.verticalScrollBar().isVisible(), "over": over,
               "wide_rows": rows}
        log(f"  {what}: viewport {vw} px, content minimum {mw} px (width {content.width()}), horizontal bar "
            f"{'SHOWN' if hbar.isVisible() else 'hidden'} (range {hbar.maximum()}), vertical bar "
            f"{'shown' if rec['vbar_visible'] else 'hidden'}")
        if rows:
            log(f"    layout items at least {floor} px wide (minimum width):")
            for r in rows:
                log(f"      {'  ' * r['depth']}{r['min_w']:4d}  {r['item']}")
        return rec

    for size in ((1500, 900), (1200, 700)):
        win.resize(*size)
        settle(0.8)
        log(f"Window {size[0]} x {size[1]} (control column {ctrl.width()} px)")
        ctrl.setCurrentIndex(0)
        key = f"{size[0]}x{size[1]}"
        data[key] = {"Point Detection": measure(view.det_scroll, "Point Detection")}
        ctrl.setCurrentIndex(1)
        for model in ("Pinhole", "Pinhole+Refraction", "Pinhole+Cylinder"):
            view.wand_model_combo.setCurrentText(model)
            data[key][model] = measure(cal_scroll, f"Calibration, {model}")
        for panel, rec in data[key].items():
            check(f"{key}: {panel} has no horizontal scroll bar", not rec["hbar_visible"],
                  f"content minimum {rec['content_min']} px, viewport {rec['viewport']} px")
        log()

    # pictures of the panels, to see that the narrowed fields show their text (cut from the window's image, so they
    # look as on the screen: a widget grabbed alone loses the window's style)
    from PySide6.QtCore import QRect
    win.resize(1500, 1300)
    ctrl.setCurrentIndex(1)
    view.wand_num_cams.setValue(4)
    view.window_count_spin.setValue(2)
    for c in (2, 3):
        view.cam_window_table.cellWidget(c, 1).setCurrentIndex(1)
    for model, group in (("Pinhole", None), ("Pinhole+Refraction", view.refraction_group),
                         ("Pinhole+Cylinder", view.cylinder_group)):
        view.wand_model_combo.setCurrentText(model)
        settle(0.8)
        if group is not None:
            cal_scroll.ensureWidgetVisible(group)
        else:
            cal_scroll.verticalScrollBar().setValue(cal_scroll.verticalScrollBar().maximum())
        settle(0.8)
        pm = win.grab()
        part = cal_scroll.viewport()
        r = QRect(part.mapTo(win, QPoint(0, 0)), part.size())
        dpr = pm.devicePixelRatio()
        pm = pm.copy(QRect(int(r.x() * dpr), int(r.y() * dpr), int(r.width() * dpr), int(r.height() * dpr)))
        out = HERE / f"{name}_{model.replace('+', '_').lower()}.png"
        pm.save(str(out))
        log(f"  picture: {out.name} ({pm.width()} x {pm.height()})")

    n_fail = sum(not c["ok"] for c in checks)
    log(f"{len(checks) - n_fail} of {len(checks)} checks passed")
    (HERE / f"{name}_out.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / f"{name}.json").write_text(json.dumps({"checks": checks, "panels": data}, indent=1), encoding="utf-8")
    sys.stdout.flush()
    os._exit(0 if n_fail == 0 else 1)                    # skip Qt's teardown (the full window can fault at shutdown)


if __name__ == "__main__":
    main()
