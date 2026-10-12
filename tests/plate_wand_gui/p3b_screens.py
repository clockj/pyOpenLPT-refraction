"""P3 (b): pictures of the wand tab's flat-window panel with the new controls, cut from the main window's image as
tests/wand_gui_fixes/w8_panel_widths.py cuts them (a widget grabbed alone loses the window's style and fonts).

    python p3b_screens.py

The GUI's main window on the real platform, moved off the screen (as W8); Wand Calibration > Calibration; Pinhole+Refraction, 4 cameras, 2 windows (cameras 2
and 3 behind window 1), window 1 10 mm thick, channel widths 100 / 150 mm, Start: Saved fit (a folder), the
Constraints group on with the channel and one aim. The calibration panel's scroll area is shown at the top of the
Refraction Settings group, its middle and its end: p3b_panel_1.png, _2.png, _3.png. A look, not a test.
Writes p3b_screens_out.txt here.
"""
import pgcommon as P  # noqa: I001

log = P.Log("p3b_screens")


def main():
    P.qt_app(offscreen=False)      # the real platform, the window moved off the screen (W8: offscreen Qt has no fonts)
    import app as guiapp
    from PySide6.QtCore import QEventLoop, QPoint, QRect, QTimer
    from PySide6.QtWidgets import QScrollArea, QTabWidget, QTableWidgetItem
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
    view.tabs.setCurrentIndex(1)                         # Wand Calibration
    p = view.det_scroll.parentWidget()
    while p is not None and not isinstance(p, QTabWidget):
        p = p.parentWidget()
    p.setCurrentIndex(1)                                 # its Calibration panel
    cal_scroll = next(s for s in view.findChildren(QScrollArea) if s.widget() and s.widget().isAncestorOf(
        view.refraction_group))
    win.resize(1500, 1300)
    view.wand_num_cams.setValue(4)
    view.wand_model_combo.setCurrentText("Pinhole+Refraction")
    view.window_count_spin.setValue(2)
    for c in (2, 3):
        view.cam_window_table.cellWidget(c, 1).setCurrentIndex(1)
    view.media2_thick.setValue(12.7)
    view.refr_window_table.item(1, 1).setText("10.0")
    view.refr_window_table.item(0, 3).setText("100.0")
    view.refr_window_table.item(1, 3).setText("150.0")
    view.refr_start_combo.setCurrentIndex(2)
    view.refr_start_folder.setText(r"X:\...\T0\calib_plate_wand")
    view.refr_cons_group.setChecked(True)
    view.refr_channel_check.setChecked(True)
    view.refr_channel_tol.setValue(3.0)
    view.refr_ball_a.setValue(1.5)
    view.refr_ball_b.setValue(2.0)
    view.refr_aim_table.insertRow(0)
    for j, t in enumerate(["0,1", "1", "75", ""]):
        view.refr_aim_table.setItem(0, j, QTableWidgetItem(t))
    settle(1.0)
    bar = cal_scroll.verticalScrollBar()
    top = view.refraction_group.mapTo(cal_scroll.widget(), QPoint(0, 0)).y()
    bottom = top + view.refraction_group.height()
    vh = cal_scroll.viewport().height()
    for k, y in enumerate((top, (top + bottom - vh) // 2, bottom - vh), start=1):
        bar.setValue(max(0, min(bar.maximum(), int(y))))
        settle(0.8)
        pm = win.grab()
        part = cal_scroll.viewport()
        r = QRect(part.mapTo(win, QPoint(0, 0)), part.size())
        dpr = pm.devicePixelRatio()
        pm = pm.copy(QRect(int(r.x() * dpr), int(r.y() * dpr), int(r.width() * dpr), int(r.height() * dpr)))
        out = P.HERE / f"p3b_panel_{k}.png"
        pm.save(str(out))
        log(f"picture {out.name}: {pm.width()} x {pm.height()} px (scroll {bar.value()} of {bar.maximum()})")
    log(f"the Refraction Settings group: {view.refraction_group.width()} x {view.refraction_group.height()} px")
    # what reaches past the scroll area's viewport (the panel must fit its column, as the 7 Oct fixes made it)
    from PySide6.QtWidgets import QWidget
    vw = cal_scroll.viewport().width()
    over = []
    for w in view.refraction_group.findChildren(QWidget):
        if not w.isVisibleTo(view.refraction_group):
            continue
        right = w.mapTo(cal_scroll.widget(), QPoint(0, 0)).x() + w.width()
        if right > vw:
            txt = w.text() if hasattr(w, "text") and callable(w.text) else ""
            over.append((right, type(w).__name__, str(txt)[:30], w.minimumSizeHint().width()))
    over.sort(reverse=True)
    log(f"viewport {vw} px; widgets reaching past it (right edge, type, text, minimum width): {over[:14]}")
    log.save({"viewport_px": vw, "group_px": view.refraction_group.width(), "over": over})


if __name__ == "__main__":
    main()
