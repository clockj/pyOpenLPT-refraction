"""G6 (part): the full GUI starts offscreen with the new code and its model switch shows the right panels.

    python g6_gui_launch.py

The main window (gui/app.py, OpenLPTMainWindow, with the update check off) is built offscreen; in its Camera
Calibration view the Wand tab's camera models and detection modes are listed, and switching between the models
shows and hides the window and tube groups, the k1+k2 choice and the Focal column as intended.
Writes g6_gui_launch_out.txt and g6_gui_launch.json.
"""
import common as C  # noqa: I001

import sys  # noqa: E402

log = C.Log("g6_gui_launch")


def main():
    app = C.qt_app()
    import app as guiapp
    guiapp.OpenLPTMainWindow._check_for_updates = lambda self: None
    w = guiapp.OpenLPTMainWindow()
    w.show()
    app.processEvents()
    from modules.camera_calibration import CameraCalibrationView
    views = w.findChildren(CameraCalibrationView)
    log.check("the main window builds, with one Camera Calibration view", len(views) == 1, f"{len(views)} views")
    v = views[0]
    models = [v.wand_model_combo.itemText(i) for i in range(v.wand_model_combo.count())]
    modes = [b.text() for b in v.detect_mode_group.buttons()]
    log.check("camera models", models == ["Pinhole", "Pinhole+Refraction", "Pinhole+Cylinder"], str(models))
    log.check("detection modes", modes == ["fast", "reliable", "ellipse"], str(modes))
    v.detect_mode_ellipse.setChecked(True)
    app.processEvents()
    shown = not v.ellipse_opts_container.isHidden()
    v.detect_mode_fast.setChecked(True)
    app.processEvents()
    log.check("the rod settings show for 'ellipse' only", shown and v.ellipse_opts_container.isHidden())
    states = {}
    for m in models:
        v.wand_model_combo.setCurrentText(m)
        app.processEvents()
        states[m] = {"window_group": not v.refraction_group.isHidden(), "tube_group": not v.cylinder_group.isHidden(),
                     "k1k2": v.dist_model_combo.model().item(2).isEnabled(),
                     "focal": v.wand_cam_table.cellWidget(0, 2).isEnabled()}
        log(f"  {m}: {states[m]}")
    ok = (states["Pinhole"] == {"window_group": False, "tube_group": False, "k1k2": True, "focal": True}
          and states["Pinhole+Refraction"] == {"window_group": True, "tube_group": False, "k1k2": True, "focal": True}
          and states["Pinhole+Cylinder"] == {"window_group": False, "tube_group": True, "k1k2": False, "focal": False})
    log.check("each model shows its own panel; the cylinder model greys k1+k2 and the Focal column", ok)
    v.wand_num_cams.setValue(2)
    v.wand_model_combo.setCurrentText("Pinhole+Cylinder")
    v.wand_num_cams.setValue(4)
    app.processEvents()
    log.check("the Focal column stays greyed when the camera count changes under the cylinder model",
              all(not v.wand_cam_table.cellWidget(r, 2).isEnabled() for r in range(4)))
    w.close()
    return log.save({"states": states})


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
