"""W4 (B4): the error table's Del marks and the deleted frames.  Offscreen, with G9's synthetic pinhole rig
(tests/cylinder_wand_gui/g5_g9_stop_input_pinhole.py: three OpenCV cameras, 1280 x 800 px, f 2,000 px, k1 0.05,
around a 100 mm volume; 300 wand frames of 10 mm, 0.1 px noise), whose true cameras stand in for a calibration's
result; 8 px is added to camera 0's ball A in frames 10-14, so that the filter "Delete when proj error > 5 px"
marks them.  The error table is filled as after a calibration (calculate_per_frame_errors, _populate_error_table).

    python w4_del_marks.py

Checks:
  - a Del box clicked by hand keeps its state when the filters re-mark the rows: frame 3 ticked by hand stays ticked,
    frame 11 (a bad frame) unticked by hand stays unticked, while the filter marks 10, 12, 13, 14 at 5 px and
    unmarks them at 9 px;
  - the next run (_apply_current_ui_filter, which Run Calibration and Precalibrate to Check call first) deletes the
    marked frames: 3, 10, 12, 13, 14; the table after it lists the other frames;
  - "Restore Deleted Frames" brings every frame back: no deleted frame, no mark, every box unticked; the next run
    uses every frame;
  - new points start with no deleted frame: after loading a CSV of wand points ("Load Wand Points (from CSV)"),
    the frames deleted before are not deleted (before the fix the earlier deletions applied to the new points).
(Detection, the other way to get new points, is checked in W5.)
Writes w4_del_marks_out.txt and .json.
"""
import common as C  # noqa: I001

import sys  # noqa: E402

import numpy as np  # noqa: E402

log = C.Log("w4_del_marks")
BAD = (10, 11, 12, 13, 14)


def gui_points(uv):
    wp = {}
    for i in range(uv.shape[0]):
        fr = {}
        for c in range(uv.shape[1]):
            if np.all(np.isfinite(uv[i, c])):
                a, b = uv[i, c]
                fr[c] = [[a[0], a[1], 20.0, 0.9, "Filtered_Small", 0], [b[0], b[1], 30.0, 0.9, "Filtered_Large", 1]]
        if len(fr) >= 2:
            wp[i] = fr
    return wp


def pinhole_rig(rng):
    """G9's rig (three cameras at -30, 0, 30 deg, 600 mm away) and 300 wand frames: final_params, uv, A, B."""
    import cv2
    W, H, f, k1 = 1280, 800, 2000.0, 0.05
    K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]])
    dist = np.array([k1, 0, 0, 0, 0])
    final = {}
    for k, az in enumerate((-30.0, 0.0, 30.0)):
        a = np.radians(az)
        Cc = np.array([600 * np.sin(a), 80.0, -600 * np.cos(a)])
        z = -Cc / np.linalg.norm(Cc)
        x = np.cross([0, 1, 0], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.vstack([x, y, z])
        final[k] = {"R": R, "T": (-R @ Cc).reshape(3, 1), "K": K, "dist": dist, "img_size": (H, W)}
    mid = rng.uniform(-40, 40, (300, 3))
    d = rng.normal(size=(300, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    A, B = mid - 5 * d, mid + 5 * d
    uv = np.full((300, 3, 2, 2), np.nan)
    for k, p in final.items():
        for e, X in enumerate((A, B)):
            q, _j = cv2.projectPoints(X, cv2.Rodrigues(p["R"])[0], p["T"], K, dist)
            uv[:, k, e] = q.reshape(-1, 2) + rng.normal(scale=0.1, size=(300, 2))
    return final, uv, A, B


def main():
    app = C.qt_app()  # noqa: F841
    csv = C.DATA / "w4_new_points.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    dlg = C.Dialogs(open_file=str(csv))
    from PySide6.QtWidgets import QCheckBox
    from modules.camera_calibration import CameraCalibrationView
    final, uv, A, B = pinhole_rig(np.random.default_rng(1314))
    uv[list(BAD), 0, 0, 0] += 8.0                          # camera 0, ball A: 8 px off in frames 10-14
    wp = gui_points(uv)
    view = CameraCalibrationView()
    view.wand_num_cams.setValue(3)
    wc = view.wand_calibrator

    def result_table(frames):
        """The table as after a calibration on `frames` (the true cameras and wand ends as its result)."""
        wc.final_params = final
        wc.wand_length = 10.0
        fl = sorted(frames)
        wc.points_3d = np.vstack([np.vstack([A[i], B[i]]) for i in fl])
        wc.per_frame_errors = {}
        wc.calculate_per_frame_errors()
        view._populate_error_table()

    def boxes():
        out = {}
        for row in range(view.frozen_table.rowCount()):
            w = view.frozen_table.cellWidget(row, 0)
            chk = w.findChild(QCheckBox) if w else None
            if chk is not None:
                out[int(chk.property("frame_id"))] = chk
        return out

    def ticked():
        return sorted(f for f, c in boxes().items() if c.isChecked())

    wc.wand_points = wp
    wc.wand_points_filtered = None
    result_table(wp)
    log(f"W4: {len(wp)} frames in the error table; frames {BAD} have 8 px on camera 0")
    worst = {f: max(wc.per_frame_errors[f]["cam_errors"].values()) for f in BAD}
    log("  their largest errors: " + ", ".join(f"{f}: {v:.2f} px" for f, v in worst.items()))

    b = boxes()
    b[3].click()                                           # ticked by hand
    b[11].click()
    b[11].click()                                          # ticked and unticked by hand: kept
    view.filter_proj_spin.setValue(5.0)
    view.filter_proj_check.setChecked(True)
    at5 = ticked()
    view.filter_proj_spin.setValue(9.0)
    at9 = ticked()
    view.filter_proj_spin.setValue(5.0)
    again5 = ticked()
    log.check("Del boxes clicked by hand keep their state when the filter re-marks the rows",
              at5 == [3, 10, 12, 13, 14] and at9 == [3] and again5 == at5,
              f"at 5 px {at5}; at 9 px {at9}; at 5 px again {again5}")

    view._apply_current_ui_filter()
    removed = sorted(view._removed_frames)
    kept = sorted(wc.wand_points_filtered or {})
    result_table(kept)
    log.check("the next run deletes the marked frames; the table after it lists the others",
              removed == [3, 10, 12, 13, 14] and len(kept) == len(wp) - 5 and not set(kept) & set(removed)
              and view.error_table.rowCount() == len(wp) - 5,
              f"deleted {removed}; {len(kept)} frames used; {view.error_table.rowCount()} rows")

    view.btn_restore_frames.click()
    status = view.status_label.text()
    state = (sorted(view._removed_frames), dict(view._manual_del), wc.wand_points_filtered, ticked())
    view._apply_current_ui_filter()
    used = len(wc.wand_points_filtered) if wc.wand_points_filtered is not None else len(wc.wand_points)
    log.check("Restore Deleted Frames brings every frame back (no deletion, no mark, no box ticked); the next run "
              "uses every frame", state == ([], {}, None, []) and used == len(wp) and "Restored 5" in status,
              f"deleted {state[0]}, marks {state[1]}, filtered {state[2]}, ticked {state[3]}; next run {used} "
              f"frames; status '{status}'")

    # new points: a CSV of the same points, loaded after frames were deleted
    result_table(wp)
    for f in (3, 20):
        boxes()[f].click()
    view._apply_current_ui_filter()
    before = sorted(view._removed_frames)
    import csv as csv_mod
    with open(csv, "w", newline="") as fh:                # the points as the detection's CSV holds them
        wr = csv_mod.writer(fh)
        wr.writerow(["Frame", "Camera", "Status", "PointIdx", "X", "Y", "Radius", "Metric"])
        for f, fr in wp.items():
            for c, pts in fr.items():
                for k, p in enumerate(pts[:2]):
                    wr.writerow([f, c, p[4], k, f"{p[0]:.3f}", f"{p[1]:.3f}", f"{p[2]:.3f}", f"{p[3]:.4f}"])
    view._load_wand_points_for_calibration()
    after = (sorted(getattr(view, "_removed_frames", set())), dict(getattr(view, "_manual_del", {})),
             wc.wand_points_filtered)
    view._apply_current_ui_filter()
    used = len(wc.wand_points_filtered) if wc.wand_points_filtered is not None else len(wc.wand_points)
    log.check("new points (Load Wand Points from CSV) start with no deleted frame",
              before == [3, 20] and after == ([], {}, None) and used == len(wc.wand_points) == len(wp),
              f"deleted before {before}; after loading: deleted {after[0]}, marks {after[1]}, filtered "
              f"{after[2]}; next run {used} of {len(wc.wand_points)} frames; messages {[m[:2] for m in dlg.messages]}")
    view.close()
    return log.save()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
