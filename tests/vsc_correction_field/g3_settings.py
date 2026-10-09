"""G3 (b): what the tracking settings page does to a tube project's config.json after VSC (a light check).

    python g3_settings.py

The VSC's message suggests setting the 2-D tolerance in the tracking settings. Here the settings page
(gui/views/tracking_settings_view.py, from source, Qt offscreen) loads a copy of a project after a field VSC pass
(projects/t1_set: G1's config.json with the cameras' corrections), sets the 2-D tolerance to 1.0 px and saves; then
config.json before and after: the cameras' "correction" keys, the tracer's settings, the tolerance. Message boxes are
recorded, not shown. Writes g3_settings_out.txt and .json here.
"""
from __future__ import annotations

import copy
import os
import shutil

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import vcf_common as vc  # noqa: E402
from g3_gui import Recorder  # noqa: E402


def main():
    src = os.path.join(vc.HERE, "projects", "t1_selfcalproto")
    proj = os.path.join(vc.HERE, "projects", "t1_set")
    if os.path.exists(proj):
        shutil.rmtree(proj)
    os.makedirs(proj)
    shutil.copy2(os.path.join(src, "config.json"), proj)
    shutil.copytree(os.path.join(src, "camFile_VSC"), os.path.join(proj, "camFile_VSC"))
    os.makedirs(os.path.join(proj, "Results"))
    before = vc.load_json(os.path.join(proj, "config.json"))
    vc.lib("vcf")
    log = vc.Log("g3_settings")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    import gui.views.tracking_settings_view as ts
    ts.QMessageBox = Recorder
    view = ts.TrackingSettingsView()
    view.project_path.setText(proj)
    app.processEvents()
    view._load_config()
    app.processEvents()
    view.ipr_2d_tol.setValue(1.0)
    view._save_configuration()
    app.processEvents()
    after = vc.load_json(os.path.join(proj, "config.json"))
    corr_kept = [c.get("correction") for c in after["cameras"]] == [c.get("correction") for c in before["cameras"]]
    t0, t1 = before["objects"][0]["tracer"], after["objects"][0]["tracer"]
    dropped = sorted(set(t0) - set(t1))
    changed = sorted(k for k in set(t0) & set(t1) if t0[k] != t1[k])
    other0 = {k: v for k, v in before["objects"][0].items() if k not in ("tracer", "ipr")}
    other1 = {k: v for k, v in after["objects"][0].items() if k not in ("tracer", "ipr")}
    log(f"settings page on {proj}: loaded, 2-D tolerance set to 1.0 px, saved; message boxes "
        f"{[m['title'] for m in Recorder.shown]}")
    log(f"  the cameras' corrections kept: {corr_kept} ({[c.get('correction') for c in after['cameras']]})")
    log(f"  ipr.tol_2d_px: {before['objects'][0]['ipr']['tol_2d_px']} -> {after['objects'][0]['ipr']['tol_2d_px']}")
    log(f"  the tracer's settings dropped by the save: {dropped}; changed: {changed}")
    log(f"  the object's other settings unchanged: {other0 == other1}")
    log(f"  the top-level keys: before {sorted(before)}, after {sorted(after)}")
    log.save({"project": proj, "corrections_kept": corr_kept, "tol_before": before["objects"][0]["ipr"]["tol_2d_px"],
              "tol_after": after["objects"][0]["ipr"]["tol_2d_px"], "tracer_dropped": dropped,
              "tracer_changed": changed, "tracer_before": t0, "tracer_after": t1,
              "messages": copy.deepcopy(Recorder.shown)})


if __name__ == "__main__":
    main()
