"""G3: the GUI's VSC on a tube project, the light check (it runs, renders, says the right things, no errors).

    python g3_gui.py [NAME] [BUILD]  # NAME: a project made by g1_project.py (default t1_g3); BUILD: vcf (default),
                                     # or gui: the GUI's installed module (outputs g3_gui_gui_*, g3_gui_*.png)

The tracking view (gui/views/tracking_view.py, from source) on Qt's offscreen platform, with build vcf's pyopenlpt
(your installed build is not touched): the project's path set as the settings would set it, Run VSC pressed, the
worker waited for; then screenshots of the view (g3_*.png here): the VSC group with its note, the Correction tab, the
2-D overlay of the first VSC frame (red: detections, yellow: projections before, green: after), the 3-D tab.
The view's message boxes are replaced by a recorder (offscreen, a modal box would wait for a click); their texts,
the execution log and anything printed to stderr are kept. Writes g3_gui_out.txt and .json here.
"""
from __future__ import annotations

import io
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import vcf_common as vc  # noqa: E402


class Recorder:
    """QMessageBox's static calls, recorded instead of shown. Like the modal box, each runs an event loop until it is
    "clicked" (after a second): the view counts on that loop, e.g. _on_vsc_finished lets go of the VSC's thread after
    its box, and the thread ends while the box is up."""
    shown = []

    @staticmethod
    def _rec(kind, parent, title, text, *a, **k):
        Recorder.shown.append({"kind": kind, "title": title, "text": text})
        from PySide6.QtCore import QEventLoop, QTimer
        loop = QEventLoop()
        QTimer.singleShot(1000, loop.quit)
        loop.exec()
        return 0

    @staticmethod
    def information(parent, title, text, *a, **k):
        return Recorder._rec("information", parent, title, text)

    @staticmethod
    def warning(parent, title, text, *a, **k):
        return Recorder._rec("warning", parent, title, text)

    @staticmethod
    def critical(parent, title, text, *a, **k):
        return Recorder._rec("critical", parent, title, text)

    @staticmethod
    def question(parent, title, text, *a, **k):
        Recorder._rec("question", parent, title, text)
        return 0


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "t1_g3"
    proj = os.path.join(vc.HERE, "projects", name)
    build = sys.argv[2] if len(sys.argv) > 2 else "vcf"
    tag0 = "" if build == "vcf" else f"{build}_"
    lpt = vc.lib(build)
    log = vc.Log(f"g3_gui{'' if build == 'vcf' else '_' + build}")
    log(f"G3 on {proj}; pyopenlpt {lpt.__file__}")
    err = io.StringIO()
    old_err = sys.stderr

    class Tee:
        def write(self, s):
            err.write(s)
            old_err.write(s)

        def flush(self):
            old_err.flush()
    sys.stderr = Tee()

    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    import gui.views.tracking_view as tv
    tv.QMessageBox = Recorder
    view = tv.TrackingView(settings_view=None)
    view.resize(1700, 1050)
    view.show()
    app.processEvents()
    view.proj_path_edit.setText(proj)
    app.processEvents()
    out = {"project": proj, "note_visible": bool(view.vsc_field_note.isVisible())}
    log(f"  the field VSC note under the VSC group: {'shown' if out['note_visible'] else 'NOT shown'}")
    shots = {}

    def shot(tag):
        app.processEvents()
        p = os.path.join(vc.HERE, f"g3_{tag0}{tag}.png")
        view.grab().save(p)
        shots[tag] = os.path.basename(p)

    shot("before")
    t0 = time.time()
    view._run_vsc()
    while view.vsc_worker is not None and time.time() - t0 < 1800:
        app.processEvents()
        time.sleep(0.05)
    app.processEvents()
    out["seconds"] = round(time.time() - t0, 1)
    out["messages"] = Recorder.shown
    for m in Recorder.shown:
        log(f"  message box ({m['kind']}) '{m['title']}': {m['text']}")
    out["correction_tab"] = view.vsc_corr_widget is not None and view.vis_tabs.indexOf(view.vsc_corr_widget) >= 0
    log(f"  the Correction tab: {'present' if out['correction_tab'] else 'MISSING'}; frames in the VSC table: "
        f"{view.vsc_frame_table.rowCount()}")
    shot("correction")
    # the 2-D overlay of the first frame, every camera shown
    for cb in getattr(view, "cam_checkboxes", []):
        cb.setChecked(True)
    app.processEvents()
    if view.vsc_frame_table.rowCount():
        view.vsc_frame_table.sortItems(0)
        item = view.vsc_frame_table.item(0, 0)
        view._on_vsc_frame_selected(item)
        app.processEvents()
        time.sleep(0.5)
        app.processEvents()
        shot("overlay")
    view.vis_tabs.setCurrentIndex(0)
    shot("3d")
    with open(os.path.join(vc.HERE, f"g3_{tag0}gui_log.txt"), "w", encoding="utf-8") as fh:
        fh.write(view.log_text.toPlainText())
    cfg = vc.load_json(os.path.join(proj, "config.json"))
    out["corrections"] = [c.get("correction") for c in cfg["cameras"]]
    sys.stderr = old_err
    errors = [ln for ln in err.getvalue().splitlines() if "Error" in ln or "Traceback" in ln]
    out["stderr_errors"] = errors
    ok = (out["note_visible"] and out["correction_tab"] and all(out["corrections"])
          and any(m["title"] == "VSC Complete" and "Field VSC pass" in m["text"] for m in Recorder.shown)
          and not errors)
    out["screenshots"] = shots
    out["pass"] = bool(ok)
    log(f"  config.json's corrections: {out['corrections']}; errors on stderr: {len(errors)}"
        + (f" (first: {errors[0]})" if errors else ""))
    log(f"  screenshots: {', '.join(shots.values())}")
    log(f"G3: {'PASS' if ok else 'FAIL'} (the screenshots still need a look)")
    log.save(out)
    view.close()


if __name__ == "__main__":
    main()
