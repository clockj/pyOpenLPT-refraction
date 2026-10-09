"""G4c: does the camera table show its Pixel (um) column without scrolling? (plan: 2026-10-08-plan.md, test G4)

    python g4c_table_width.py [--tag NAME]

G4b's picture showed the wand tab's camera table cut after Width: Height and the new Pixel (um) column were behind a
horizontal scroll bar.  In the GUI's own window, off the screen (as G4b), at window sizes 1700 x 1080, 1500 x 900
and 1200 x 700, with three cameras: each column's width and its widgets' size hints, the table's viewport width, and
whether its horizontal scroll bar shows; a picture of the table at 1500 x 900 (g4c_table[_NAME].png).
Judged by: no horizontal scroll bar at any of the sizes.
Writes g4c_table_width[_NAME]_out.txt and .json beside itself.
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    name = "g4c_table_width" + (f"_{args.tag}" if args.tag else "")
    lines, checks, data = [], [], {}

    def log(msg=""):
        print(msg, flush=True)
        lines.append(str(msg))

    def check(label, ok, detail=""):
        checks.append({"check": label, "ok": bool(ok), "detail": detail})
        log(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))

    sys.path[:0] = [str(REPO / "gui"), str(REPO)]
    import openlpt  # noqa: F401
    import app as guiapp
    guiapp.prepare_qtwebengine()
    from PySide6.QtCore import QEventLoop, QPoint, QRect, QTimer
    from PySide6.QtWidgets import QApplication
    qapp = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    from modules.camera_calibration import CameraCalibrationView

    def settle(s=0.6):
        loop = QEventLoop()
        QTimer.singleShot(int(1000 * s), loop.quit)
        loop.exec()

    guiapp.OpenLPTMainWindow._check_for_updates = lambda self: None
    win = guiapp.OpenLPTMainWindow()
    win.move(-9000, -9000)
    win.show()
    view = win.findChildren(CameraCalibrationView)[0]
    view.tabs.setCurrentIndex(1)
    view.wand_num_cams.setValue(3)
    t = view.wand_cam_table
    for k, px in enumerate((20.0, 20.0, 13.5)):
        t.cellWidget(k, 5).setValue(px)
    for size in ((1700, 1080), (1500, 900), (1200, 700)):
        win.resize(*size)
        settle()
        hdr = t.horizontalHeader()
        widths = [hdr.sectionSize(i) for i in range(t.columnCount())]
        hints = [t.cellWidget(0, i).sizeHint().width() if t.cellWidget(0, i) is not None else None
                 for i in range(t.columnCount())]
        vp = t.viewport().width()
        hbar = t.horizontalScrollBar().isVisible()
        key = f"{size[0]}x{size[1]}"
        data[key] = {"widths": widths, "hints": hints, "viewport": vp, "sum": sum(widths), "hbar": hbar}
        log(f"window {key}: column widths {widths} (sum {sum(widths)}), widget hints {hints}, viewport {vp} px, "
            f"horizontal scroll bar {'shown' if hbar else 'hidden'}")
        check(f"{key}: every column shows without a horizontal scroll bar", not hbar and sum(widths) <= vp + 1,
              f"columns {sum(widths)} px in {vp} px")
        if key == "1500x900":
            pm = win.grab()
            r = QRect(t.mapTo(win, QPoint(0, 0)), t.size())
            dpr = pm.devicePixelRatio()
            pm = pm.copy(QRect(int(r.x() * dpr), int(r.y() * dpr), int(r.width() * dpr), int(r.height() * dpr)))
            pic = f"g4c_table{'_' + args.tag if args.tag else ''}.png"
            pm.save(str(HERE / pic))
            log(f"  picture: {pic}")
    n_fail = sum(not c["ok"] for c in checks)
    log(f"\n{len(checks) - n_fail} of {len(checks)} checks passed")
    (HERE / f"{name}_out.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / f"{name}.json").write_text(json.dumps({"checks": checks, "sizes": data}, indent=1), encoding="utf-8")
    sys.stdout.flush()
    os._exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
