"""W6 (B6): changing Num Cameras keeps the cameras' setups; W7 (B7): the Refraction panel's indices take four
decimals.  Offscreen.

    python w6_w7_cameras_decimals.py

W6: two cameras' image folders loaded with their Load buttons (W1's six frames of T2, data/w1/dark/cam1 and cam2),
Focal, Width and Height set (camera 1 given another size), 2 windows with camera 0 behind window 1 and camera 1
behind window 0; then Num Cameras 2 -> 3 -> 2:
  - cameras 0 and 1 keep their images (the Load buttons' counts, the frame table), Focal, Width, Height and window;
  - camera 2 starts empty, with the defaults;
  - Number of Windows 2 -> 1 -> 2: a camera keeps its window while it exists (camera 1, window 0), and the camera
    whose window went (camera 0) is on window 0.
W7: the three indices show four decimals; 1.3330 and 1.4905 are kept as typed; the Water preset is 1.333, as the
Tube panel's; Acrylic 1.49, Glass 1.52, Air 1.0.
Writes w6_w7_cameras_decimals_out.txt and .json.
"""
import common as C  # noqa: I001

import sys  # noqa: E402

log = C.Log("w6_w7_cameras_decimals")


def main():
    app = C.qt_app()  # noqa: F841
    dlg = C.Dialogs()
    from modules.camera_calibration import CameraCalibrationView
    view = CameraCalibrationView()
    t = view.wand_cam_table

    def row(i):
        btn = t.cellWidget(i, 0)
        return {"images": len(view.wand_images.get(i, [])), "button": btn.text() if btn else None,
                "focal": t.cellWidget(i, 2).value(), "width": t.cellWidget(i, 3).value(),
                "height": t.cellWidget(i, 4).value()}

    def windows():
        return [view.cam_window_table.cellWidget(i, 1).currentText() for i in range(view.cam_window_table.rowCount())]

    # ---------------- W6
    view.wand_num_cams.setValue(2)
    for c in (0, 1):
        dlg.folder = str(C.DATA / "w1" / "dark" / f"cam{c + 1}")
        view._load_wand_folder_for_cam(c)                     # the camera's Load button
    t.cellWidget(0, 2).setValue(41500)
    t.cellWidget(1, 2).setValue(42000)
    t.cellWidget(1, 3).setValue(1024)
    t.cellWidget(1, 4).setValue(768)
    view.window_count_spin.setValue(2)
    view.cam_window_table.cellWidget(0, 1).setCurrentIndex(1)
    view.cam_window_table.cellWidget(1, 1).setCurrentIndex(0)
    before = [row(0), row(1)], windows(), view.frame_table.rowCount()
    log(f"W6: before: {before[0]}; windows {before[1]}; frame table {before[2]} rows")
    view.wand_num_cams.setValue(3)
    at3 = [row(0), row(1), row(2)], windows(), view.frame_table.rowCount()
    view.wand_num_cams.setValue(2)
    after = [row(0), row(1)], windows(), view.frame_table.rowCount()
    log(f"  at 3 cameras: {at3[0]}; windows {at3[1]}")
    log(f"  back at 2: {after[0]}; windows {after[1]}; frame table {after[2]} rows")
    log.check("cameras 0 and 1 keep their images, Focal, Width, Height and window through 2 -> 3 -> 2",
              at3[0][:2] == before[0] and after == before and before[0][0]["images"] == 6
              and before[0][0]["button"] == "6" and before[2] == 6,
              f"{after[0]}, windows {after[1]}")
    log.check("camera 2 starts empty, with the defaults",
              at3[0][2] == {"images": 0, "button": "Load", "focal": 9000, "width": 1280, "height": 800}
              and at3[1] == ["Window 1", "Window 0", "Window 0"], f"{at3[0][2]}, windows {at3[1]}")
    view.window_count_spin.setValue(1)
    one = windows()
    view.window_count_spin.setValue(2)
    two = windows()
    log.check("Number of Windows 2 -> 1 -> 2: a camera keeps its window while it exists",
              one == ["Window 0", "Window 0"] and two == ["Window 0", "Window 0"],
              f"at 1 window {one}; at 2 again {two}")
    view.window_count_spin.setValue(2)
    view.cam_window_table.cellWidget(0, 1).setCurrentIndex(1)
    view.window_count_spin.setValue(3)
    log.check("Number of Windows 2 -> 3: the cameras keep their windows", windows() == ["Window 1", "Window 0"],
              f"{windows()}")

    # ---------------- W7
    log("\nW7: the Refraction panel's indices")
    spins = (view.media1_index, view.media2_index, view.media3_index)
    log.check("the three indices take four decimals", [s.decimals() for s in spins] == [4, 4, 4],
              f"{[s.decimals() for s in spins]}")
    view.media3_combo.setCurrentText("Other")
    view.media3_index.setValue(1.3330)
    v1, txt1 = view.media3_index.value(), view.media3_index.text()
    view.media2_combo.setCurrentText("Other")
    view.media2_index.setValue(1.4905)
    v2, txt2 = view.media2_index.value(), view.media2_index.text()
    log.check("1.3330 and 1.4905 are kept as typed", abs(v1 - 1.333) < 1e-12 and abs(v2 - 1.4905) < 1e-12
              and txt1 == "1.3330" and txt2 == "1.4905", f"{v1!r} '{txt1}', {v2!r} '{txt2}'")
    presets = {}
    for combo, spin, names in ((view.media1_combo, view.media1_index, ("Air",)),
                               (view.media2_combo, view.media2_index, ("Acrylic", "Glass")),
                               (view.media3_combo, view.media3_index, ("Water",))):
        for n in names:
            combo.setCurrentText(n)
            presets[n] = (spin.value(), spin.isReadOnly())
    view.cyl_in_combo.setCurrentText("Water")
    log.check("the presets: Air 1.0, Acrylic 1.49, Glass 1.52, Water 1.333 (= the Tube panel's water), read-only",
              presets == {"Air": (1.0, True), "Acrylic": (1.49, True), "Glass": (1.52, True), "Water": (1.333, True)}
              and view.cyl_in_index.value() == presets["Water"][0], f"{presets}; tube water {view.cyl_in_index.value()}")
    view.close()
    return log.save()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
