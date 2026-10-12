"""G1: the ball detection with a pixel size per camera (plan: 2026-10-08-plan.md, test G1).  No Qt.

    python g1_scaled_detection.py

Run 20261008/T0, 24 image numbers (every 400th from 2600; all three cameras saved them), the ellipse mode with
cams 1-2's settings (Radius Range 20-70 px, Rod opening 51 px, Rod mask 28 px, Dark on Bright) and pixel sizes
20 / 20 / 13.5 um, i.e. the GUI's scales {0: 1, 1: 1, 2: 20/13.5}:
  1. scale_detect_settings: cam3's numbers, and the identity (the same objects) at scale 1 and None;
  2. WandCalibrator.detect_single_frame with cam_scales (the GUI's Test Detect): per camera, bitwise the same rows
     as run_detection_task called by hand with that camera's scaled settings;
  3. detect_wand_points_generator with cam_scales (the GUI's Process Frames), on the first 6 of those frames: the
     same raw balls as 2;
  4. against cases/cyl_20261008's own settings for cam3 (30-105 px, opening 77, mask 42; the GUI's scaled values
     are 29.63-103.70 px, 76 -> 77, 41.48): how far the centres move on the frames both keep;
  5. the benefit: cam3 detected with cams 1-2's settings unscaled (the GUI before this change): frames with two
     balls, against the scaled settings.
Writes g1_scaled_detection_out.txt and .json beside itself.
"""
import common as C  # noqa: I001

import math  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402

log = C.Log("g1_scaled_detection")
IMAGES = list(range(2600, 2600 + 24 * 400, 400))
BASE = dict(min_r=20.0, max_r=70.0, opts={"open_px": 51, "rod_mask_px": 28.0, "guard_px": 3.0, "full": True})
SCALES = {0: 1.0, 1: 1.0, 2: C.PIXEL_UM[1] / C.PIXEL_UM[3]}
CASE_CAM3 = dict(min_r=30.0, max_r=105.0, opts={"open_px": 77, "rod_mask_px": 42.0, "guard_px": 3.0, "full": True})


def two_balls(rows):
    """The two balls of a kept camera-frame (as the case's first rules: exactly two, the guard not fired), else None."""
    if rows is None or len(rows) != 2 or any(r[7] for r in rows):
        return None
    return sorted(rows, key=lambda r: r[2])


def main():
    from modules.camera_calibration.wand_calibration import wand_calibrator as wc
    out = {"images": IMAGES, "scales": SCALES}

    # 1 ----------------------------------------------------------------- the numbers
    log("1. scale_detect_settings")
    mn, mx, op = wc.scale_detect_settings(BASE["min_r"], BASE["max_r"], BASE["opts"], SCALES[2])
    log(f"  cam3: radius {mn:.4f}-{mx:.4f} px, opening {op['open_px']} px, mask {op['rod_mask_px']:.4f} px, "
        f"guard {op['guard_px']} px")
    s = SCALES[2]
    log.check("cam3's radius range, opening and mask are cams 1-2's x 20/13.5; the guard and 'full' unchanged",
              mn == 20.0 * s and mx == 70.0 * s and op["open_px"] == int(round(51 * s)) and op["rod_mask_px"] == 28.0 * s
              and op["guard_px"] == 3.0 and op["full"] is True, f"scale {s:.6f}")
    log.check("the caller's settings are not modified", BASE["opts"]["open_px"] == 51 and BASE["opts"]["rod_mask_px"] == 28.0)
    ident = [wc.scale_detect_settings(BASE["min_r"], BASE["max_r"], BASE["opts"], v) for v in (None, 1.0, 0)]
    log.check("scale None, 1 or 0: the same objects back", all(t[0] is BASE["min_r"] and t[1] is BASE["max_r"]
                                                              and t[2] is BASE["opts"] for t in ident))

    # 2 ----------------------------------------------------------------- Test Detect's path
    log("\n2. detect_single_frame with cam_scales against run_detection_task by hand")
    cal = wc.WandCalibrator()
    gui, hand, mism = {}, {}, []
    for f in IMAGES:
        paths = {k: C.frame_path(k + 1, f) for k in range(3)}
        res = cal.detect_single_frame(paths, "dark", BASE["min_r"], BASE["max_r"], 0.85, "ellipse",
                                      detect_opts=BASE["opts"], cam_scales=SCALES)
        for k in range(3):
            a, b, o = wc.scale_detect_settings(BASE["min_r"], BASE["max_r"], BASE["opts"], SCALES[k])
            _, _, ref = wc.run_detection_task((0, k, paths[k], "dark", a, b, 0.85, "ellipse", o))
            gui[(f, k)], hand[(f, k)] = res.get(k), ref
            if not C.same_rows(res.get(k), ref):
                mism.append((f, k))
    log.check("every camera-frame: bitwise the same rows", not mism, f"{len(IMAGES) * 3} camera-frames; "
              f"different: {mism[:6]}")
    kept = {k: sum(two_balls(gui[(f, k)]) is not None for f in IMAGES) for k in range(3)}
    log(f"  camera-frames with two balls (scaled): cam1 {kept[0]}, cam2 {kept[1]}, cam3 {kept[2]} of {len(IMAGES)}")
    out["kept_scaled"] = kept

    # 3 ----------------------------------------------------------------- Process Frames' path
    log("\n3. detect_wand_points_generator with cam_scales, first 6 frames")
    paths = {k: [C.frame_path(k + 1, f) for f in IMAGES[:6]] for k in range(3)}
    tmp = os.path.join(tempfile.mkdtemp(prefix="g1_"), "wand_points.csv")
    cal2 = wc.WandCalibrator()
    for _ in cal2.detect_wand_points_generator(paths, "dark", BASE["min_r"], BASE["max_r"], 0.85, autosave_path=tmp,
                                               detect_mode="ellipse", detect_opts=BASE["opts"], cam_scales=SCALES):
        pass
    raw = getattr(cal2, "wand_data_raw", {})
    bad = []
    for i, f in enumerate(IMAGES[:6]):
        for k in range(3):
            g = raw.get(i, {}).get(k)
            ref = gui[(f, k)]
            ref_xy = None if ref is None else np.asarray(ref, float)[:, :4]
            g_xy = None if g is None or len(g) == 0 else np.asarray(g, float)[:, :4]
            if (ref_xy is None) != (g_xy is None) or (ref_xy is not None and not (
                    ref_xy.shape == g_xy.shape and np.allclose(np.sort(ref_xy, axis=0), np.sort(g_xy, axis=0),
                                                               rtol=0, atol=1e-9))):
                bad.append((f, k))
    log.check("the generator's raw balls equal Test Detect's (u, v, r, metric)", not bad, f"18 camera-frames; "
              f"different: {bad}")

    # 4 ----------------------------------------------------------------- the case's settings for cam3
    log("\n4. cam3: the GUI's scaled settings against the case's (30-105 px, 77, 42)")
    d, both = [], 0
    case_rows = {}
    for f in IMAGES:
        _, _, ref = wc.run_detection_task((0, 2, C.frame_path(3, f), "dark", CASE_CAM3["min_r"], CASE_CAM3["max_r"], 0.85,
                                           "ellipse", CASE_CAM3["opts"]))
        case_rows[f] = ref
        a, b = two_balls(gui[(f, 2)]), two_balls(ref)
        if a is not None and b is not None:
            both += 1
            d += [math.hypot(a[j][0] - b[j][0], a[j][1] - b[j][1]) for j in range(2)]
    kept_case = sum(two_balls(case_rows[f]) is not None for f in IMAGES)
    log(f"  frames with two balls: GUI scaled {kept[2]}, case {kept_case}; both {both}; centre differences: "
        f"max {max(d) if d else float('nan'):.4f} px, median {np.median(d) if d else float('nan'):.4f} px")
    log.check("the same frames kept, centres within 0.05 px", kept_case == kept[2] == both and (max(d) if d else 0) < 0.05,
              f"{both} frames")
    out["vs_case"] = {"kept_case": kept_case, "both": both, "max_px": max(d) if d else None,
                      "median_px": float(np.median(d)) if d else None}

    # 5 ----------------------------------------------------------------- without the change
    log("\n5. cam3 with cams 1-2's settings, unscaled (the GUI before this change)")
    un = {}
    for f in IMAGES:
        _, _, rows = wc.run_detection_task((0, 2, C.frame_path(3, f), "dark", BASE["min_r"], BASE["max_r"], 0.85,
                                            "ellipse", BASE["opts"]))
        un[f] = rows
    kept_un = sum(two_balls(un[f]) is not None for f in IMAGES)
    nballs = [0 if un[f] is None else len(un[f]) for f in IMAGES]
    log(f"  cam3 frames with two balls: unscaled {kept_un}, scaled {kept[2]} of {len(IMAGES)}; balls found unscaled "
        f"per frame: {nballs}")
    log.check("scaling keeps more of cam3's frames than one setting for all cameras", kept[2] > kept_un,
              f"{kept[2]} against {kept_un}")
    out["cam3_unscaled_kept"] = kept_un
    log.save(out)


if __name__ == "__main__":
    main()
