"""G4: the wand tab with three cameras of two kinds, offscreen (plan: 2026-10-08-plan.md, test G4).

    python g4_gui.py

The GUI's own code paths, with run 20261008/T0's three folders loaded through the Load buttons (the folder dialog
answered by the test):
  1. the camera table has the Pixel (um) column; loading the folders fills Width/Height (cam3 2048 x 1952) and
     aligns the lists by the Phantom headers (10,605 frames each, cams 1-2 from frame002396.tif; the status line
     says so);
  2. pixel sizes 20 / 20 / 13.5: the scales {0: 1, 1: 1, 2: 1.4815}, the Radius Range's limit (in camera 1's
     pixels), the camera settings' pixel_um, the jobs' pixel sizes in mm; with every pixel size cleared the scales
     are None (the detection as before);
  3. Test Detect on image 5000 (row 2604) in the ellipse mode with cams 1-2's settings: the result handed to the
     display equals WandCalibrator.detect_single_frame with the same scales, and cam3 shows two balls;
  4. Num Cameras 3 -> 2 -> 3 keeps cams 1-2's pixel sizes and folders; without cam3 they are no longer cut
     (13,001 images each), and cam 3 comes back empty;
  5. no warning or error dialog.
Pictures: offscreen Qt has no fonts, so they come from G4b (g4b_screens.py, the GUI's own window).
Writes g4_gui_out.txt and .json beside itself.
"""
import common as C  # noqa: I001

import os  # noqa: E402

log = C.Log("g4_gui")
IMAGE = 5000


def main():
    app = C.qt_app()  # noqa: F841
    dlg = C.Dialogs()
    from modules.camera_calibration import CameraCalibrationView
    from modules.camera_calibration.wand_calibration.wand_calibrator import WandCalibrator
    view = CameraCalibrationView()
    view.resize(1700, 1050)
    view.show()
    t = view.wand_cam_table
    out = {}

    # 1 ----------------------------------------------------------------- table, loading, alignment
    view.wand_num_cams.setValue(3)
    labels = [t.horizontalHeaderItem(i).text() for i in range(t.columnCount())]
    log(f"1. columns: {labels}")
    log.check("the table has the Pixel (µm) column", labels == ["", "Cam ID", "Focal (px)", "Width", "Height", "Pixel (µm)"])
    for k in range(3):
        dlg.folder = str(C.T0_RAW / f"cam{k + 1}")
        view._load_wand_folder_for_cam(k)
        log(f"  loaded cam{k + 1}: {len(view.wand_images[k])} images, button {t.cellWidget(k, 0).text()!r}, "
            f"size {t.cellWidget(k, 3).value()} x {t.cellWidget(k, 4).value()}")
    n = [len(view.wand_images[k]) for k in range(3)]
    firsts = [os.path.basename(view.wand_images[k][0]) for k in range(3)]
    status = view.status_label.text() if hasattr(view, "status_label") else ""
    log(f"  first files {firsts}; frame table {view.frame_table.rowCount()} rows; status: {status!r}")
    log.check("aligned: 10,605 images each, cams 1-2 from frame002396.tif, cam3 from frame000000.tif",
              n == [10605] * 3 and firsts == ["frame002396.tif", "frame002396.tif", "frame000000.tif"], f"{n}")
    log.check("the Load buttons and the frame table show 10,605", [t.cellWidget(k, 0).text() for k in range(3)] == ["10605"] * 3
              and view.frame_table.rowCount() == 10605)
    log.check("the status line says the cameras were aligned", "aligned by their Phantom headers" in status
              and "2396" in status, status)
    log.check("Width/Height from the images (cam3 2048 x 1952)",
              [(t.cellWidget(k, 3).value(), t.cellWidget(k, 4).value()) for k in range(3)] == [(1280, 800), (1280, 800), (2048, 1952)])
    out["loaded"] = {"n": n, "first": firsts, "status": status}

    # 2 ----------------------------------------------------------------- pixel sizes
    log("\n2. pixel sizes 20 / 20 / 13.5")
    for k, p in enumerate((20.0, 20.0, 13.5)):
        t.cellWidget(k, 5).setValue(p)
    sc = view._wand_cam_scales()
    lim = view.radius_range._max_range
    st = view._collect_camera_settings_from_table()
    pm = view._wand_pixel_mm_for([0, 1, 2], st)
    log(f"  scales {sc}; Radius Range limit {lim}; pixel_um {[st[k]['pixel_um'] for k in range(3)]}; job pixel_mm {pm}")
    log.check("scales {0: 1, 1: 1, 2: 20/13.5}", sc is not None and sc[0] == 1.0 and sc[1] == 1.0
              and abs(sc[2] - 20 / 13.5) < 1e-12)
    log.check("the Radius Range's limit stays 200 (camera 1's pixels: cam3's 1952 / 4 / 1.48 = 329 is larger)", lim == 200)
    log.check("the jobs get [0.02, 0.02, 0.0135] mm", pm == [0.02, 0.02, 0.0135])
    for k in range(3):
        t.cellWidget(k, 5).setValue(0.0)
    log.check("every pixel size cleared: scales None, jobs' pixel_mm None",
              view._wand_cam_scales() is None and view._wand_pixel_mm_for([0, 1, 2]) is None)
    t.cellWidget(0, 5).setValue(20.0)
    t.cellWidget(1, 5).setValue(20.0)
    log.check("equal pixel sizes only: scales None (nothing to scale)", view._wand_cam_scales() is None)
    t.cellWidget(2, 5).setValue(13.5)
    out["scales"] = sc

    # 3 ----------------------------------------------------------------- Test Detect
    log(f"\n3. Test Detect on image {IMAGE} (row {IMAGE - 2396}), ellipse mode, 20-70 px, opening 51, mask 28")
    view.detect_mode_ellipse.setChecked(True)
    view.radius_range.setMaxValue(70)
    view.radius_range.setMinValue(20)
    view.ellipse_open_spin.setValue(51)
    view.ellipse_mask_spin.setValue(28.0)
    view.wand_type_combo.setCurrentIndex(0)                       # Dark on Bright
    row = IMAGE - 2396
    view.frame_table.setCurrentCell(row, 0)
    got = {}
    orig = view._on_single_detection_finished

    def keep(res):
        got["res"] = res
        orig(res)
    view._on_single_detection_finished = keep
    view._detect_single_frame()
    C.wait_for(lambda: "res" in got, 300)
    view._busy_end('wand_detect_single') if "res" not in got else None
    res = got.get("res", {})
    paths = {k: view.wand_images[k][row] for k in range(3)}
    ref = WandCalibrator().detect_single_frame(paths, "dark", 20, 70, view.sensitivity_slider.value(), "ellipse",
                                               detect_opts={"open_px": 51, "rod_mask_px": 28.0, "guard_px": 3.0,
                                                            "full": True}, cam_scales=sc)
    log(f"  balls: " + ", ".join(f"cam{k + 1} {0 if res.get(k) is None else len(res.get(k))}" for k in range(3))
        + f"; files {[os.path.basename(p) for p in paths.values()]}")
    log.check("the displayed result equals detect_single_frame with the same scales",
              all(C.same_rows(res.get(k), ref.get(k)) for k in range(3)))
    log.check("cam3 shows two balls (with cams 1-2's settings scaled)", res.get(2) is not None and len(res.get(2)) == 2)
    out["test_detect"] = {k: (None if res.get(k) is None else [list(map(float, r[:4])) for r in res.get(k)])
                          for k in range(3)}

    # 4 ----------------------------------------------------------------- Num Cameras
    log("\n4. Num Cameras 3 -> 2 -> 3")
    view.wand_num_cams.setValue(2)
    keep2 = [(t.cellWidget(k, 5).value(), len(view.wand_images[k])) for k in range(2)]
    view.wand_num_cams.setValue(3)
    after = [(t.cellWidget(k, 5).value(), len(view.wand_images[k])) for k in range(3)]
    log(f"  at 2: {keep2}; back at 3: {after}")
    log.check("cams 1-2 keep pixel size and folder, whole again without cam3; cam 3 starts empty",
              keep2 == [(20.0, 13001), (20.0, 13001)] and after[:2] == keep2 and after[2] == (0.0, 0))

    # 5 ----------------------------------------------------------------- dialogs
    bad = [m for m in dlg.messages if m[0] in ("warning", "critical")]
    log(f"\n5. dialogs: {dlg.messages}")
    log.check("no warning or error dialog", not bad)
    log.save(out)


if __name__ == "__main__":
    main()
