"""W1 (B1): "Bright on Dark" detects bright balls.  The GUI's Wand Type is now read by one helper, _wand_type();
before the fix the test was `"dark" if "Dark" in text`, which both texts pass, so "Bright on Dark" ran as "Dark on
Bright".

    python w1_bright_balls.py

Six frames of run 20261006/T2 (two cameras, dark balls on a bright background; X:) are written as the 8-bit images
the detector reads (data/w1/dark/), and inverted (data/w1/bright/: bright balls on a dark background).  Checks:
  - the mapping: "Dark on Bright" -> "dark", "Bright on Dark" -> "bright";
  - the detector (run_detection_task, in this process, numpy's random seed set before each call): the inverted
    images as "bright" give exactly the detections of the originals as "dark", in the fast, reliable and ellipse
    modes;
  - the GUI's Test Detect (offscreen): the inverted images with "Bright on Dark" against the originals with "Dark on
    Bright".  The reliable and ellipse modes give the same detections to the bit.  The fast mode refines its circles
    by RANSAC with numpy's unseeded random draws (refine_circle_with_edges) in the GUI's worker processes, so two
    runs on the same image differ: it must find as many points per camera, and differ by no more than twice what a
    second "Dark on Bright" run on the originals differs from the first;
  - the bug's effect, for the record: the inverted images detected as "dark" (what "Bright on Dark" ran before)
    do not give those detections.
(The first version of this test asked the fast mode for identical detections through the GUI and failed on its
random draws: w1_bright_balls_first_try_out.txt.)
Settings: T2's (radius range 40-120 px, ellipse opening 95 px, rod mask 50 px).
Writes w1_bright_balls_out.txt and .json.
"""
import common as C  # noqa: I001

import sys  # noqa: E402

import numpy as np  # noqa: E402

FRAMES = (1000, 2000, 3000, 4069, 5000, 8000)
log = C.Log("w1_bright_balls")


def same(a, b):
    """Two detection results ({camera: points or None}) identical to the last bit."""
    a = {k: v for k, v in a.items() if v is not None}
    b = {k: v for k, v in b.items() if v is not None}
    if set(a) != set(b):
        return False
    return all(np.array_equal(np.asarray(a[k], float), np.asarray(b[k], float)) for k in a)


def counts(res):
    return {int(k): (0 if v is None else len(v)) for k, v in res.items() if v is not None}


def largest_shift(a, b):
    """The largest difference of centre and radius between two results with the same points per camera (each
    point paired with the nearest of the other), or None when the counts differ."""
    if counts(a) != counts(b):
        return None
    worst = 0.0
    for k, pa in a.items():
        if pa is None:
            continue
        pa, pb = np.asarray(pa, float)[:, :3], np.asarray(b[k], float)[:, :3]
        for p in pa:
            q = pb[np.argmin(np.linalg.norm(pb[:, :2] - p[:2], axis=1))]
            worst = max(worst, float(np.linalg.norm(q[:2] - p[:2])), float(abs(q[2] - p[2])))
    return worst


def main():
    import cv2
    log("W1: Wand Type 'Bright on Dark' on bright balls (B1)")
    src = C.t2_frames(FRAMES)
    sets = {"dark": {}, "bright": {}}
    for kind in sets:
        for c in src:
            (C.DATA / "w1" / kind / f"cam{c + 1}").mkdir(parents=True, exist_ok=True)
    for c, paths in src.items():
        sets["dark"][c], sets["bright"][c] = [], []
        for f, p in zip(FRAMES, paths):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)                  # as run_detection_task reads it
            if img.dtype != np.uint8:
                img = (img / img.max() * 255).astype(np.uint8)
            for kind, im in (("dark", img), ("bright", cv2.bitwise_not(img))):
                out = C.DATA / "w1" / kind / f"cam{c + 1}" / f"frame{f:06d}.png"
                cv2.imwrite(str(out), im)
                sets[kind][c].append(str(out))
    log(f"  frames {FRAMES} of T2, cameras 1 and 2: originals (dark balls) and inverted copies (bright balls)")

    app = C.qt_app()  # noqa: F841
    C.Dialogs()
    from modules.camera_calibration import CameraCalibrationView
    from modules.camera_calibration.wand_calibration.wand_calibrator import run_detection_task
    view = CameraCalibrationView()
    view.wand_num_cams.setValue(2)
    view.radius_range.setMinValue(40)
    view.radius_range.setMaxValue(120)
    view.ellipse_open_spin.setValue(95)
    view.ellipse_mask_spin.setValue(50.0)
    sens = view.sensitivity_slider.value()

    maps = {}
    for text in ("Dark on Bright", "Bright on Dark"):
        view.wand_type_combo.setCurrentText(text)
        maps[text] = view._wand_type()
    items = [view.wand_type_combo.itemText(i) for i in range(view.wand_type_combo.count())]
    log.check("the mapping: 'Dark on Bright' -> dark, 'Bright on Dark' -> bright (the combo's only items)",
              maps == {"Dark on Bright": "dark", "Bright on Dark": "bright"} and sorted(items) == sorted(maps),
              f"{maps}; items {items}")

    got = []
    orig = view._on_single_detection_finished

    def capture(res):
        got.append(res)
        orig(res)
    view._on_single_detection_finished = capture       # the worker's signal is connected to this attribute

    def test_detect(kind, text, row):
        view.wand_images = {c: list(v) for c, v in sets[kind].items()}
        view.populate_wand_table()
        view.wand_type_combo.setCurrentText(text)
        view.frame_table.setCurrentCell(row, 0)
        n0 = len(got)
        view._detect_single_frame()
        ok = C.wait_for(lambda: len(got) > n0, 300)    # (the GUI deletes the worker once it has finished)
        C.wait_for(lambda: False, 0.2)                  # the handler's drawing and the dialog's close
        return got[-1] if ok else None

    results = {}
    radios = {"fast": view.detect_mode_fast, "reliable": view.detect_mode_reliable,
              "ellipse": view.detect_mode_ellipse}
    for mode, radio in radios.items():
        radio.setChecked(True)
        assert view._get_wand_detect_mode() == mode
        opts = view._get_wand_detect_opts(full=True)

        # the detector, seeded: the same images reach it, so the same detections
        n_seeded, bug = 0, 0
        for row in range(len(FRAMES)):
            r = {}
            for kind in ("dark", "bright"):
                r[kind] = {}
                for c in sets[kind]:
                    np.random.seed(20261007)
                    r[kind][c] = run_detection_task((0, c, sets[kind][c][row], kind, 40, 120, sens, mode, opts))[2]
            n_seeded += same(r["dark"], r["bright"])
            old = {}
            for c in sets["bright"]:
                np.random.seed(20261007)
                old[c] = run_detection_task((0, c, sets["bright"][c][row], "dark", 40, 120, sens, mode, opts))[2]
            bug += not same(r["dark"], old)
        log.check(f"{mode}, the detector (seeded): bright balls as 'bright' = dark balls as 'dark', to the bit",
                  n_seeded == len(FRAMES), f"{n_seeded} of {len(FRAMES)} frames")
        log(f"      the old mapping ('dark' on the bright-ball images) differs on {bug} of {len(FRAMES)} frames")

        # the GUI's Test Detect
        n_same, found, shifts, repeat = 0, 0, [], []
        for row, f in enumerate(FRAMES):
            r_dark = test_detect("dark", "Dark on Bright", row) or {}
            r_bright = test_detect("bright", "Bright on Dark", row) or {}
            n_same += same(r_dark, r_bright)
            found += sum(1 for v in counts(r_dark).values() if v >= 2)
            if mode == "fast":
                r_dark2 = test_detect("dark", "Dark on Bright", row) or {}
                shifts.append(largest_shift(r_dark, r_bright))
                repeat.append(largest_shift(r_dark, r_dark2))
        if mode == "fast":
            ok_counts = all(s is not None for s in shifts) and all(s is not None for s in repeat)
            m_b = max(s for s in shifts if s is not None) if any(s is not None for s in shifts) else float("nan")
            m_r = max(s for s in repeat if s is not None) if any(s is not None for s in repeat) else float("nan")
            log.check("fast, Test Detect: 'Bright on Dark' on the inverted frames finds the points of 'Dark on Bright' "
                      "on the originals, as close as a second run on the originals",
                      ok_counts and m_b <= 2 * m_r,
                      f"same counts {ok_counts}; largest difference {m_b:.2f} px against {m_r:.2f} px between two "
                      f"runs on the originals; {n_same} of {len(FRAMES)} frames identical; {found} of "
                      f"{2 * len(FRAMES)} images with two or more points")
            results[mode] = {"frames_identical": n_same, "largest_shift_px": shifts, "repeat_shift_px": repeat,
                             "old_mapping_differs": bug, "seeded_identical": n_seeded}
        else:
            log.check(f"{mode}, Test Detect: 'Bright on Dark' on the inverted frames = 'Dark on Bright' on the "
                      f"originals, to the bit", n_same == len(FRAMES),
                      f"{n_same} of {len(FRAMES)} frames; {found} of {2 * len(FRAMES)} images with two or more points")
            results[mode] = {"frames_identical": n_same, "old_mapping_differs": bug, "seeded_identical": n_seeded}
    view.close()
    return log.save({"mapping": maps, "results": results})


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
