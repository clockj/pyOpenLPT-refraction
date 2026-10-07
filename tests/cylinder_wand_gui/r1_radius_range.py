"""R1 (your question after the fixes, 6 Oct night): does the Radius Range matter in the "ellipse" detection mode?

    python tests/cylinder_wand_gui/r1_radius_range.py [FIRST LAST]      (default: T2 frames 4000-5199, as P3/P4)

In the ellipse mode the range reaches the detector (ball_ellipses.detect_wand_balls) in two ways:
  - the background window: the backlight is estimated with a closing 2.5 x the largest radius wide (at least T13's
    101 px), which must be wider than a ball;
  - the blob filter: blobs with an area outside pi (min/2)^2 .. pi (1.5 max)^2 are skipped before the two largest
    are taken.
The GUI's detection (WandCalibrator.detect_wand_points_generator, its own pool) on the same frames with the case's
rod settings (opening 95 px, mask 50 px) and three radius ranges:
  case     40-120 px  (the case's; background window 301 px)
  default  20-200 px  (the GUI's default; window 501 px)
  small    20-60  px  (a maximum below this wand's balls, r 58-88 px; window 151 px, under the large ball's
                       diameter of about 175 px)
Per range: frames paired, and against "case" on the frames both paired: each ball's centre difference (RMS and
largest, px) and radius difference.  Writes r1_radius_range_out.txt and r1_radius_range.json beside this script; the
CSVs to data/r1_<name>.csv.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RAW = r"X:\Shijie Zhong\Refraction\Raw\20261006\T2"
OPTS = {"open_px": 95, "rod_mask_px": 50.0, "guard_px": 3.0, "full": False}
RANGES = {"case": (40, 120), "default": (20, 200), "small": (20, 60)}


def main():
    sys.path[:0] = [os.path.join(REPO, "build-py", "Release"), REPO]
    from modules.camera_calibration.wand_calibration.wand_calibrator import WandCalibrator
    from modules.camera_calibration.curved_calibration import ball_ellipses as be
    first, last = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (4000, 5199)
    images = {k: [os.path.join(RAW, cam, f"frame{f:06d}.tif") for f in range(first, last + 1)]
              for k, cam in enumerate(("cam1", "cam2"))}
    lines, out, pts = [], {}, {}

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    say(f"R1: the Radius Range in the ellipse mode; T2 frames {first}-{last}, rod opening 95 px, rod mask 50 px")
    for name, (lo, hi) in RANGES.items():
        csv_path = os.path.join(HERE, "data", f"r1_{name}.csv")
        if os.path.exists(csv_path):
            os.remove(csv_path)
        wc = WandCalibrator()
        t0 = time.time()
        for _ in wc.detect_wand_points_generator(images, "dark", lo, hi, 0.85, autosave_path=csv_path, resume=False,
                                                 detect_mode="ellipse", detect_opts=OPTS):
            pass
        win = be.settings_for(hi)["background_close_px"]
        n_two = {c: sum(1 for v in wc.wand_data_raw.values() if len(v.get(c, [])) >= 2) for c in (0, 1)}
        pts[name] = {f: {c: (np.array(p[0][:3], float), np.array(p[1][:3], float)) for c, p in cams.items()}
                     for f, cams in wc.wand_points.items()}
        say(f"\n[{name}] radius {lo}-{hi} px; background window {win} px; {time.time() - t0:.0f} s")
        say(f"  frames with two or more blobs: cam1 {n_two[0]}, cam2 {n_two[1]}; paired: {len(wc.wand_points)}")
        out[name] = {"range": [lo, hi], "background_px": win, "two_blobs": n_two, "paired": len(wc.wand_points)}
    ref = pts["case"]
    for name in ("default", "small"):
        common = sorted(set(ref) & set(pts[name]))
        dA, dB, dr = [], [], []
        for f in common:
            for c in (0, 1):
                a, b = ref[f][c], pts[name][f][c]
                dA.append(np.hypot(*(b[0][:2] - a[0][:2])))
                dB.append(np.hypot(*(b[1][:2] - a[1][:2])))
                dr += [b[0][2] - a[0][2], b[1][2] - a[1][2]]
        dA, dB, dr = np.array(dA), np.array(dB), np.array(dr)
        only_ref = len(set(ref) - set(pts[name]))
        only_new = len(set(pts[name]) - set(ref))
        if common:
            say(f"\n{name} against case: {len(common)} frames paired by both; {only_ref} only by case, {only_new} only "
                f"by {name}")
            say(f"  small ball's centre: RMS {np.sqrt(np.mean(dA ** 2)):.3f} px, largest {dA.max():.3f} px; large "
                f"ball's: RMS {np.sqrt(np.mean(dB ** 2)):.3f} px, largest {dB.max():.3f} px; radius difference mean "
                f"{dr.mean():+.3f} px")
            out[name]["vs_case"] = {"common": len(common), "only_case": only_ref, "only_this": only_new,
                                    "small_rms_px": float(np.sqrt(np.mean(dA ** 2))), "small_max_px": float(dA.max()),
                                    "large_rms_px": float(np.sqrt(np.mean(dB ** 2))), "large_max_px": float(dB.max()),
                                    "radius_mean_diff_px": float(dr.mean())}
        else:
            say(f"\n{name} against case: no frame paired by both ({only_ref} only by case, {only_new} only by {name})")
            out[name]["vs_case"] = {"common": 0, "only_case": only_ref, "only_this": only_new}
    with open(os.path.join(HERE, "r1_radius_range_out.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "r1_radius_range.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
